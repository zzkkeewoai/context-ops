"""阈值标定：把两个关键阈值扫一遍，看指标怎么随它们变

【为什么必须扫，不能拍脑袋】
    这两个阈值直接决定"作答还是拒答"：
        MIN_RELEVANCE      —— 低于它认为"知识库没有相关内容" → 拒答
        MIN_TOPIC_OVERLAP  —— 两条证据的主题重叠达到它才可能被判"互相矛盾"

    而拒答相关的错误是**方向相反、代价不对称**的：
        阈值调高 → 过拒上升（该答的答不了，可用性损失）
        阈值调低 → 漏拒上升（该拒的硬答，幻觉风险）
    调一边另一边就恶化，只看一个数永远定不下来。
    只有把整条曲线画出来（尤其看那两个错误率怎么交叉），才知道该停在哪。

    而且这两个阈值还会互相影响：主题重叠阈值调松 → 误报的"矛盾"变多 →
    更多问题被判"无法裁决" → 过拒率跟着涨。所以必须**联合扫**，不能分开调。

运行：
    cd context_ops
    python calibrate_thresholds.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import metrics as metric_mod  # noqa: E402
from context_builder import get_strategy  # noqa: E402
from context_validator import ContextValidator  # noqa: E402
from pipeline import InMemoryRetriever, Pipeline  # noqa: E402

RELEVANCE_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
OVERLAP_CANDIDATES = [6, 8, 10, 12, 14]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    kb = load(os.path.join(ROOT, "data", "knowledge_base.jsonl"))
    cases = load(os.path.join(ROOT, "data", "eval_cases.jsonl"))
    retriever = InMemoryRetriever(kb)

    print("=" * 100)
    print(f"阈值标定 · 知识库 {len(kb)} 条 · 用例 {len(cases)} 条 · 策略 headtail")
    print("=" * 100)
    print(
        f"{'MIN_RELEVANCE':>14}{'MIN_OVERLAP':>13}{'准确率':>10}"
        f"{'过拒率↓':>10}{'漏拒率↓':>10}{'漏检冲突率↓':>13}"
    )
    print("-" * 100)

    rows = []
    for rel in RELEVANCE_CANDIDATES:
        for ovl in OVERLAP_CANDIDATES:
            pipe = Pipeline(
                retriever=retriever,
                strategy=get_strategy("headtail"),
                validator=ContextValidator(min_relevance=rel, min_topic_overlap=ovl),
            )
            snaps = [pipe.run(c["query"]) for c in cases]
            m = metric_mod.summarize(snaps, cases)
            rows.append((rel, ovl, m))
            print(
                f"{rel:>14.2f}{ovl:>13}{m['accuracy']:>10.3f}"
                f"{m['over_refusal_rate']:>10.3f}{m['missed_refusal_rate']:>10.3f}"
                f"{m['missed_conflict_rate']:>13.3f}"
            )
        print("-" * 100)

    # 选点原则：先把两类"危险错误"压到可接受范围，再在其中挑准确率最高的。
    # 为什么不是直接挑准确率最高：过拒率高的方案准确率可能靠"少答"堆上来 ——
    # 把问题全拒了，"答对的比例"看着还行，但产品已经不可用了。
    print("\n候选方案（过拒率 <= 0.10 且 漏拒率 <= 0.05 且 漏检冲突率 <= 0.15）：")
    ok = [
        r
        for r in rows
        if r[2]["over_refusal_rate"] <= 0.10
        and r[2]["missed_refusal_rate"] <= 0.05
        and r[2]["missed_conflict_rate"] <= 0.15
    ]
    if not ok:
        print("  没有方案同时满足 —— 说明这两个阈值解决不了问题，得回头改判据本身")
    else:
        for rel, ovl, m in sorted(ok, key=lambda r: -r[2]["accuracy"])[:5]:
            print(
                f"  rel={rel:<5} overlap={ovl:<4} 准确率={m['accuracy']:.3f} "
                f"过拒={m['over_refusal_rate']:.3f} 漏拒={m['missed_refusal_rate']:.3f} "
                f"漏检冲突={m['missed_conflict_rate']:.3f}"
            )
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
