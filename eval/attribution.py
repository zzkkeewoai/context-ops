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
    """失败类型（+ 两种成功状态）"""

    CORRECT = "correct"                              # 正确回答
    CORRECT_REFUSAL = "correct_refusal"              # 正确拒答（知识库外的问题）
    NOT_RETRIEVED = "not_retrieved"                  # ① 检索没召回
    RETRIEVED_NOT_SELECTED = "retrieved_not_selected"  # ② 召回但未入选
    SELECTED_NOT_USED = "selected_not_used"          # ③ 入选但未被使用
    GENERATION_ERROR = "generation_error"            # ④ 生成错误（含"该拒答却乱答"）


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

    # ---------- 成功情形 ----------
    if not gold:
        # 知识库外的问题：正确行为是拒答
        if snapshot.answer.refused:
            return Attribution(case_id, FailureType.CORRECT_REFUSAL, "库外问题正确拒答")
        return Attribution(
            case_id,
            FailureType.GENERATION_ERROR,
            "库外问题本应拒答，却给出了答案（生成侧未守住边界）",
            gold_in_candidates=False,
            gold_in_selected=False,
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
