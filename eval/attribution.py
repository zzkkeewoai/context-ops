"""失败归因器 —— 本项目的灵魂

它回答的问题不是"答得对不对"，而是【错在哪一环、该改哪里】：

    ① NOT_RETRIEVED           目标证据不在检索候选里        → 改检索/召回
    ② RETRIEVED_NOT_SELECTED  在候选里但被上下文组装砍掉了  → 改预算/去重/排序
    ③ SELECTED_NOT_USED       进了上下文但模型没用它        → 改位置/信噪比
    ④ GENERATION_ERROR        证据没问题但答案还是错        → 改 prompt/模型

为什么 ② 要再细分"被砍的原因"：
    同样是"没进上下文"，原因不同修法完全不同：
      · budget_exceeded → 该调 token 预算分配
      · duplicate       → 该改去重逻辑
      · low_score       → 该改检索/重排
    这就是 Schema 里 dropped.reason 字段存在的唯一理由 —— 没有它，归因到此为止。
"""
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import ContextSnapshot, DropReason  # noqa: E402


class FailureType(str, Enum):
    """失败类型（+ 三种成功/失败边界状态）

    拒答相关的两类单独列出，因为**它们的代价不对称**，混进 generation_error 就看不出方向：
        OVER_REFUSAL  该答却拒答 → 用户拿不到答案（可用性损失）
        MISSED_REFUSAL 该拒却答了 → 模型基于无关证据硬答（幻觉，代价更大）
    只报一个"准确率"会把这两种反向错误抵消掉，等于什么都没说。
    """

    CORRECT = "correct"                              # 正确回答
    CORRECT_REFUSAL = "correct_refusal"              # 正确拒答（库外问题 / 证据冲突）
    NOT_RETRIEVED = "not_retrieved"                  # ① 检索没召回
    RETRIEVED_NOT_SELECTED = "retrieved_not_selected"  # ② 召回但未入选
    SELECTED_NOT_USED = "selected_not_used"          # ③ 入选但未被使用
    GENERATION_ERROR = "generation_error"            # ④ 生成错误（证据没问题，答案还是错）
    OVER_REFUSAL = "over_refusal"                    # 过拒：本可回答却拒答（可用性损失）
    MISSED_REFUSAL = "missed_refusal"                # 漏拒：本应拒答却作答（幻觉风险）
    MISSED_CONFLICT = "missed_conflict"              # 漏检：证据互相矛盾却没被发现


@dataclass
class Attribution:
    """一条 case 的归因结果"""

    case_id: str
    failure_type: FailureType
    detail: str
    # ② 时细分"为什么被砍"，直接取自 Schema 的 dropped.reason
    drop_reasons: Optional[List[str]] = None
    # 便于人工核对
    gold_in_candidates: bool = False
    gold_in_selected: bool = False
    gold_cited: bool = False


def conflict_detected(snapshot: ContextSnapshot) -> bool:
    """本次请求有没有检出证据冲突 —— 直接读校验器留在快照里的结论。

    为什么单独抽成函数：归因器和指标都要用同一个判据，
    两处各写一遍迟早会漂移（一处改了另一处忘了）。
    """
    validator_result = snapshot.context.validator
    if validator_result is None:
        return False
    return validator_result.checks.get("conflict_pairs", 0.0) > 0


def attribute(snapshot: ContextSnapshot, case: Dict) -> Attribution:
    """对一条 case 做归因判定。

    判定是**确定性规则**（不调 LLM）—— 因为归因本身必须可靠、可复现。
    """
    case_id = case.get("case_id", snapshot.request_id)
    gold: List[str] = case.get("gold_evidence", []) or []

    candidate_ids = {c.id for c in snapshot.candidates}
    selected_ids = {e.id for e in snapshot.context.selected}
    cited_ids = set(snapshot.answer.cited_evidence)

    gold_set = set(gold)
    gold_in_candidates = bool(gold_set & candidate_ids)
    gold_in_selected = bool(gold_set & selected_ids)
    gold_cited = bool(gold_set & cited_ids)

    # ---------- conflict 类用例：考的是"矛盾有没有被发现" ----------
    # 【v3】这类用例的期望**不是**"拒答"。
    #   数据源里有互相矛盾的文档，是**数据质量**问题；
    #   系统的职责是发现它、按来源可信度裁决，而不是因为"我的知识库自相矛盾"就摆烂不答。
    #   （实测过：用"一检出冲突就拒答"的做法，在 204 条数据集上过拒率高达 59%，
    #     一半以上的问题答不了，产品上不可用。）
    #   所以这类用例只考一件事：矛盾有没有被检出。
    if case.get("category") == "conflict":
        if conflict_detected(snapshot):
            return Attribution(
                case_id,
                FailureType.CORRECT,
                "证据冲突已检出并按来源权威度裁决"
                + ("（已剔除低权威证据后正常作答）" if not snapshot.answer.refused else "（无法裁决，正确拒答）"),
                gold_in_candidates=gold_in_candidates,
                gold_in_selected=gold_in_selected,
            )
        return Attribution(
            case_id,
            FailureType.MISSED_CONFLICT,
            "证据之间存在矛盾但**没有被检出** —— 两条互相矛盾的说法会被一起交给模型，"
            "答案对不对全看运气。该检查冲突检测规则（否定词表 / 主题重叠阈值 / 是否误杀）",
        )

    # ---------- 先判定"拒答"这个动作对不对 ----------
    if not gold:
        # 知识库外的问题：正确行为是拒答
        if snapshot.answer.refused:
            return Attribution(
                case_id, FailureType.CORRECT_REFUSAL, "该拒答的问题正确拒答"
            )
        return Attribution(
            case_id,
            FailureType.MISSED_REFUSAL,
            "本应拒答却给出了答案（漏拒 → 幻觉风险）：生成侧没守住边界，"
            "说明校验器的相关性阈值偏低或缺失",
            gold_in_candidates=False,
            gold_in_selected=False,
            gold_cited=False,
        )

    if snapshot.answer.refused:
        # 该回答却拒答了 —— 过拒，用户什么也拿不到
        hint = (
            "证据其实都已就位，问题在校验器【阈值过严】"
            if gold_in_selected
            else (
                "证据在候选里但没进上下文，属组装阶段被裁"
                if gold_in_candidates
                else "证据压根没被召回，属检索阶段的问题"
            )
        )
        return Attribution(
            case_id,
            FailureType.OVER_REFUSAL,
            f"本可回答却拒答（过拒 → 可用性损失）：{hint}",
            gold_in_candidates=gold_in_candidates,
            gold_in_selected=gold_in_selected,
            gold_cited=False,
        )

    if gold_cited:
        return Attribution(
            case_id, FailureType.CORRECT, "答案引用了目标证据",
            gold_in_candidates=gold_in_candidates,
            gold_in_selected=gold_in_selected,
            gold_cited=True,
        )

    # ---------- 失败情形 ----------
    if not gold_in_candidates:
        # ① 检索这一层就没拿到
        return Attribution(
            case_id, FailureType.NOT_RETRIEVED,
            f"目标证据 {sorted(gold_set)} 未出现在检索候选中（可能知识库缺失或召回策略不佳）",
        )

    if not gold_in_selected:
        # ② 拿到了但被组装砍掉 —— 细分原因
        reasons = [
            (d.reason.value if isinstance(d.reason, DropReason) else str(d.reason))
            for d in snapshot.context.dropped
            if d.id in gold_set
        ]
        hint = {
            "budget_exceeded": "该调整 token 预算分配或做压缩",
            "duplicate": "该改进去重逻辑（误判为重复）",
            "low_score": "该改检索/重排（目标证据打分偏低被淘汰）",
            "validator_reject": "该检查上下文校验规则是否过严",
            "irrelevant": "该检查相关性判定是否误杀了相关证据",
        }
        advice = "；".join({hint.get(r, "") for r in reasons if hint.get(r)}) or "原因未记录"
        return Attribution(
            case_id, FailureType.RETRIEVED_NOT_SELECTED,
            f"目标证据在候选里但未进入上下文；被砍原因={reasons or ['未记录']} → {advice}",
            drop_reasons=reasons,
            gold_in_candidates=True,
            gold_in_selected=False,
        )

    # ③ 进了上下文但模型没引用它
    return Attribution(
        case_id, FailureType.SELECTED_NOT_USED,
        "目标证据已进入上下文，但答案没有引用它（可能是位置太靠中间被忽略，或信噪比不足）",
        gold_in_candidates=True,
        gold_in_selected=True,
        gold_cited=False,
    )


def summarize_attributions(items: List[Attribution]) -> Dict[str, int]:
    """统计归因分布 —— 这张表就是"该往哪优化"的结论。"""
    dist: Dict[str, int] = {}
    for it in items:
        key = it.failure_type.value
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: -kv[1]))


def drop_reason_distribution(items: List[Attribution]) -> Dict[str, int]:
    """② 类失败的细分原因分布 —— 直接告诉你"该改预算还是该改去重"。"""
    dist: Dict[str, int] = {}
    for it in items:
        for r in it.drop_reasons or []:
            dist[r] = dist.get(r, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: -kv[1]))
