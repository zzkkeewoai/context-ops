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
def answer_correct(snapshot: ContextSnapshot, gold_evidence: List[str]) -> bool:
    """答案是否正确。

    判定方式（简化版）：答案引用的证据命中目标证据即算正确。
    真实场景应换成：答案文本与标准答案的语义比对，或 LLM-as-judge。
    """
    if not gold_evidence:
        # 库外问题：正确答案是"拒答"
        return snapshot.answer.refused
    cited = set(snapshot.answer.cited_evidence)
    return bool(cited & set(gold_evidence))


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

    for snap, case in zip(snapshots, cases):
        gold = case.get("gold_evidence", [])
        hits += 1 if retrieval_hit(snap, gold) else 0
        selected_hits += 1 if key_evidence_selected(snap, gold) else 0
        corrects += 1 if answer_correct(snap, gold) else 0
        utilizations.append(context_utilization(snap))
        latencies.append(snap.total_latency_ms)
        tokens_in_total += snap.generation.tokens_in
        refusals += 1 if snap.answer.refused else 0

    return {
        # 检索层
        "retrieval_recall": round(hits / n, 3),
        # 上下文层
        "key_evidence_selected": round(selected_hits / n, 3),
        "avg_context_utilization": round(statistics.mean(utilizations), 3),
        # 生成层
        "accuracy": round(corrects / n, 3),
        "refusal_rate": round(refusals / n, 3),
        # 系统层
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_total_latency_ms": round(statistics.mean(latencies), 2),
        "avg_tokens_in": round(tokens_in_total / n, 1),
        "total_tokens_in": tokens_in_total,
        "case_count": n,
    }
