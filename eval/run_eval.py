"""评测运行器 —— 对同一套用例跑不同上下文策略，输出指标 + 归因对比

运行：
    cd context_ops
    python eval/run_eval.py

它产出两样东西：
    1. 指标对比表（before/after）—— 证明优化有效
    2. 归因分布（错在哪一环）—— 告诉下一步该改哪里
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
# 显式加两个路径：项目根（import schemas/pipeline）+ eval 目录（import metrics/attribution）
sys.path.insert(0, ROOT)
sys.path.insert(0, EVAL_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import attribution as attr_mod  # noqa: E402
import metrics as metric_mod  # noqa: E402
from context_builder import get_strategy  # noqa: E402
from context_validator import ContextValidator  # noqa: E402
from pipeline import InMemoryRetriever, Pipeline, SnapshotStore  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
SNAPSHOT_DIR = os.path.join(ROOT, "snapshots")


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_strategy(strategy_name: str, kb, cases, with_validator: bool = False) -> tuple:
    """用指定策略跑完整套用例，返回 (快照列表, 归因列表)"""
    tag = strategy_name + ("+validator" if with_validator else "")
    store = SnapshotStore(os.path.join(SNAPSHOT_DIR, f"{tag.replace('+', '_')}.jsonl"))
    store.clear()
    pipeline = Pipeline(
        retriever=InMemoryRetriever(kb),
        strategy=get_strategy(strategy_name),
        store=store,
        validator=ContextValidator() if with_validator else None,
    )
    snapshots = [pipeline.run(c["query"]) for c in cases]
    attributions = [attr_mod.attribute(s, c) for s, c in zip(snapshots, cases)]
    return snapshots, attributions


def print_metrics_table(results: dict):
    """打印策略指标对比表"""
    names = list(results.keys())
    base = names[0]
    keys = [
        ("retrieval_recall", "检索召回率"),
        ("key_evidence_selected", "关键证据入选率"),
        ("avg_context_utilization", "上下文利用率 ★"),
        ("accuracy", "准确率"),
        ("over_refusal_rate", "过拒率 ↓"),
        ("missed_refusal_rate", "漏拒率 ↓"),
        ("avg_tokens_in", "平均输入 token"),
        ("p95_latency_ms", "P95 延迟(ms)"),
    ]
    print("\n" + "=" * 78)
    print("指标对比")
    print("=" * 78)
    header = f"{'指标':<22}" + "".join(f"{n:>14}" for n in names)
    if len(names) > 1:
        header += f"{'变化':>14}"
    print(header)
    print("-" * 78)
    for key, label in keys:
        row = f"{label:<22}"
        for n in names:
            row += f"{results[n]['metrics'].get(key, 0):>14}"
        if len(names) > 1:
            delta = results[names[-1]]["metrics"].get(key, 0) - results[base]["metrics"].get(key, 0)
            row += f"{delta:>+14.3f}"
        print(row)


def print_attribution(results: dict):
    """打印归因分布"""
    for name, r in results.items():
        print("\n" + "-" * 78)
        print(f"归因分布 · 策略 [{name}]")
        print("-" * 78)
        for ftype, cnt in r["attribution_dist"].items():
            print(f"  {ftype:<28} {cnt}")
        drops = r["drop_reason_dist"]
        if drops:
            print("  ② 类失败中被砍的原因细分：")
            for reason, cnt in drops.items():
                print(f"      · {reason:<24} {cnt}")


def print_failure_details(results: dict):
    """打印失败明细，便于人工核对"""
    print("\n" + "-" * 78)
    print("失败明细（最后一种策略）")
    print("-" * 78)
    last = list(results.values())[-1]
    for a in last["attributions"]:
        if a.failure_type.value in ("correct", "correct_refusal"):
            continue
        print(f"  [{a.case_id}] {a.failure_type.value}")
        print(f"        {a.detail}")


def main():
    kb = load_jsonl(os.path.join(DATA_DIR, "knowledge_base.jsonl"))
    cases = load_jsonl(os.path.join(DATA_DIR, "eval_cases.jsonl"))

    print("=" * 78)
    print(f"Context Ops 评测 · 知识库 {len(kb)} 条 · 用例 {len(cases)} 条")
    print("=" * 78)

    results = {}
    # 三组对比：baseline / 上下文优化 / 上下文优化 + 校验器
    # 分开跑的目的是把"组装策略的收益"和"校验器的收益"拆开归因
    configs = [("naive", False), ("headtail", False), ("headtail", True)]
    for name, with_validator in configs:
        tag = name + ("+validator" if with_validator else "")
        snapshots, attrs = run_strategy(name, kb, cases, with_validator)
        results[tag] = {
            "metrics": metric_mod.summarize(snapshots, cases),
            "attributions": attrs,
            "attribution_dist": attr_mod.summarize_attributions(attrs),
            "drop_reason_dist": attr_mod.drop_reason_distribution(attrs),
        }

    print_metrics_table(results)
    print_attribution(results)
    print_failure_details(results)

    print("\n" + "=" * 78)
    print(f"快照已落盘：{SNAPSHOT_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
