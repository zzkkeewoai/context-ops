"""评测指标 —— 四层指标

为什么分四层而不是只算准确率：
    只报"准确率 78%"无法指导优化。分层之后才能回答：
      检索层差 → 改召回
      上下文层差 → 改组装（预算/去重/排序）
      生成层差 → 改 prompt/模型
      系统层差 → 改架构（缓存/并发/模型选择）

本项目最独特的指标是【上下文利用率】——它衡量"放进去的证据有没有被用上"，
是 90% 的 RAG 项目都没有的指标。
"""
import os
import statistics
import sys
from typing import Dict, List

# eval/ 目录下的模块要 import 项目根的 schemas：
# 从项目根运行 `python eval/run_eval.py` 时，sys.path[0] 是 eval/ 而不是项目根，
# 所以必须显式把项目根加进来。这个坑在拆分目录结构时非常常见。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import ContextSnapshot  # noqa: E402
from attribution import conflict_detected  # noqa: E402


# ============================================================
# 检索层
# ============================================================
def retrieval_hit(snapshot: ContextSnapshot, gold_evidence: List[str]) -> bool:
    """目标证据是否被检索到（在候选集里）"""
    if not gold_evidence:
        return True  # 库外问题不需要检索到任何东西
    candidate_ids = {c.id for c in snapshot.candidates}
    return any(g in candidate_ids for g in gold_evidence)


# ============================================================
# 上下文层
# ============================================================
def context_utilization(snapshot: ContextSnapshot) -> float:
    """上下文利用率 = 入选证据中被答案引用的比例（0~1）。

    为什么这个指标重要：
      · 利用率低 → 上下文里塞了模型不用的东西 → 浪费 token、拉低信噪比
      · 利用率高但不准 → 问题在生成侧
    它是"上下文质量"的直接度量，也是本项目区别于普通 RAG 项目的指标。
    """
    return snapshot.context_utilization


def key_evidence_selected(snapshot: ContextSnapshot, gold_evidence: List[str]) -> bool:
    """关键证据是否被选中进入上下文"""
    if not gold_evidence:
        return True
    selected_ids = {e.id for e in snapshot.context.selected}
    return any(g in selected_ids for g in gold_evidence)


# ============================================================
# 生成层
# ============================================================
def answer_correct(snapshot: ContextSnapshot, gold_evidence: List[str], category: str = "") -> bool:
    """答案是否正确。

    判定方式（简化版）：答案引用的证据命中目标证据即算正确。
    真实场景应换成：答案文本与标准答案的语义比对，或 LLM-as-judge。

    gold 为空的两种情况**不能用同一个标准判**：
        · conflict   → 期望"检出冲突"（数据源有矛盾，系统该发现并按可信度裁决，
                        而不是因为自己知识库自相矛盾就摆烂不答）
        · out_of_kb  → 期望"拒答"
    """
    if gold_evidence:
        cited = set(snapshot.answer.cited_evidence)
        return bool(cited & set(gold_evidence))

    if category == "conflict":
        validator_result = snapshot.context.validator
        if validator_result is None:
            return False
        return validator_result.checks.get("conflict_pairs", 0.0) > 0

    return snapshot.answer.refused


# ============================================================
# 系统层
# ============================================================
def percentile(values: List[float], p: float) -> float:
    """计算分位数（p=0.95 即 P95）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def summarize(snapshots: List[ContextSnapshot], cases: List[Dict]) -> Dict:
    """汇总一份完整的指标报告。"""
    if not snapshots:
        return {}

    n = len(snapshots)
    hits, selected_hits, corrects = 0, 0, 0
    utilizations: List[float] = []
    latencies: List[float] = []
    tokens_in_total = 0
    refusals = 0

    # 拒答相关的错误率。
    # 【注意】三类用例各用各的分母 —— 用一个总数做分母是错的：
    #   · 有 gold         → 该回答，怕的是"过拒"（该答却拒 = 可用性损失）
    #   · out_of_kb       → 该拒答，怕的是"漏拒"（该拒却答 = 幻觉风险）
    #   · conflict        → 该【检出冲突】并按可信度裁决，既不是"该答"也不是"该拒"
    #     （数据源里有矛盾是数据质量问题，不是"知识库没有"，不该用拒答来兜底）
    should_answer = 0    # 该回答的条数
    should_refuse = 0    # 该拒答的条数
    should_detect = 0    # 该检出冲突的条数
    over_refusal = 0     # 该答却拒了
    missed_refusal = 0   # 该拒却答了
    missed_conflict = 0  # 有矛盾却没检出

    for snap, case in zip(snapshots, cases):
        gold = case.get("gold_evidence", [])
        category = case.get("category", "")

        hits += 1 if retrieval_hit(snap, gold) else 0
        selected_hits += 1 if key_evidence_selected(snap, gold) else 0
        corrects += 1 if answer_correct(snap, gold, category) else 0
        utilizations.append(context_utilization(snap))
        latencies.append(snap.total_latency_ms)
        tokens_in_total += snap.generation.tokens_in
        refusals += 1 if snap.answer.refused else 0

        if category == "conflict":
            should_detect += 1
            missed_conflict += 1 if not conflict_detected(snap) else 0
        elif gold:
            should_answer += 1
            over_refusal += 1 if snap.answer.refused else 0
        else:
            should_refuse += 1
            missed_refusal += 1 if not snap.answer.refused else 0

    return {
        # 检索层
        "retrieval_recall": round(hits / n, 3),
        # 上下文层
        "key_evidence_selected": round(selected_hits / n, 3),
        "avg_context_utilization": round(statistics.mean(utilizations), 3),
        # 生成层
        "accuracy": round(corrects / n, 3),
        "refusal_rate": round(refusals / n, 3),
        # 生成层 · 错误方向相反的几类必须分开看
        "over_refusal_rate": round(over_refusal / should_answer, 3) if should_answer else 0.0,
        "missed_refusal_rate": round(missed_refusal / should_refuse, 3) if should_refuse else 0.0,
        "missed_conflict_rate": round(missed_conflict / should_detect, 3) if should_detect else 0.0,
        # 系统层
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_total_latency_ms": round(statistics.mean(latencies), 2),
        "avg_tokens_in": round(tokens_in_total / n, 1),
        "total_tokens_in": tokens_in_total,
        "case_count": n,
    }
