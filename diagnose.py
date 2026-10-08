"""诊断脚本：为什么两个策略的准确率完全一样？

打印每条证据在两种策略下的「位置 / 相关性 / 位置惩罚 / 最终权重」，
看清楚"位置惩罚到底有没有影响模型的选择"。

这是排查实验问题的正确姿势：不看数字猜，而是把中间变量打出来。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json  # noqa: E402

from context_builder import get_strategy  # noqa: E402
from pipeline import InMemoryRetriever, MockGenerator, Pipeline  # noqa: E402

QUERIES = ["Milvus 是什么？", "RRF 的公式是什么？", "SSE 是什么？"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    kb = load_jsonl(os.path.join(ROOT, "data", "knowledge_base.jsonl"))
    mock = MockGenerator()

    for q in QUERIES:
        print("=" * 76)
        print(f"query: {q}")
        print("=" * 76)

        retriever = InMemoryRetriever(kb)
        cands = retriever.retrieve(q, top_k=20)
        print("候选顺序（RRF 融合后）与相关性分数：")
        for i, c in enumerate(cands[:6]):
            rel = mock._relevance(q, c.text)
            print(f"  #{i} {c.id}  score={c.score:.3f}  relevance={rel:.3f}  {c.text[:28]}")

        # 相关性最高的那条（理论上的"正确证据"）
        best = max(cands, key=lambda c: mock._relevance(q, c.text))
        print(f"\n  相关性最高的候选: {best.id} (relevance={mock._relevance(q, best.text):.3f})")

        for name in ("naive", "headtail"):
            pipeline = Pipeline(retriever, get_strategy(name), generator=mock)
            snap = pipeline.run(q)
            sel = snap.context.selected
            n = len(sel)
            print(f"\n  [{name}] strategy={snap.context.strategy_version}")
            print(f"    selected {n} 条：")
            for e in sel:
                ratio = e.position / max(1, n - 1)
                penalty = MockGenerator.MIDDLE_PENALTY if 0.3 <= ratio <= 0.7 else 1.0
                rel = mock._relevance(q, e.text)
                print(
                    f"      pos={e.position} (ratio={ratio:.2f}, penalty={penalty}) "
                    f"{e.id} score={e.score:.3f} rel={rel:.3f} weight={penalty * rel:.3f}"
                )
            print(f"    cited={snap.answer.cited_evidence}  refused={snap.answer.refused}")
        print()


if __name__ == "__main__":
    main()
