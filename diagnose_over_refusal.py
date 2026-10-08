"""诊断：35 条「该答却拒答」到底是被什么规则拦下的

不能靠猜。校验器有几条不同的拒答路径（相关性不足 / 冲突无法裁决），
它们指向的修复方向完全不同：
    · 相关性不足   → 该调阈值，或该改检索
    · 冲突无法裁决 → 该补可信度信号，或该收紧冲突判定
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from context_builder import get_strategy  # noqa: E402
from context_validator import ContextValidator  # noqa: E402
from pipeline import InMemoryRetriever, Pipeline  # noqa: E402


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


kb = load(os.path.join(ROOT, "data", "knowledge_base.jsonl"))
cases = load(os.path.join(ROOT, "data", "eval_cases.jsonl"))

pipe = Pipeline(
    retriever=InMemoryRetriever(kb),
    strategy=get_strategy("headtail"),
    validator=ContextValidator(),
)

buckets = collections.Counter()
examples = collections.defaultdict(list)

for c in cases:
    snap = pipe.run(c["query"])
    v = snap.context.validator
    # 只关心「有 gold 却被拒」的过拒情况
    if not (c["gold_evidence"] and snap.answer.refused):
        continue

    if v is None:
        key = "无校验器"
    elif v.action == "refuse" and v.checks.get("conflict_pairs", 0) > 0:
        key = "冲突无法裁决"
    elif v.action == "refuse":
        key = "相关性不足"
    elif v.action == "rebuild":
        key = "重建后仍不足"
    else:
        key = f"其他({v.action})"

    buckets[key] += 1
    if len(examples[key]) < 3:
        examples[key].append(
            (
                c["case_id"],
                c["query"],
                f"relevance={v.checks.get('relevance')} "
                f"cand={v.checks.get('max_candidate_relevance')} "
                f"conflict={v.checks.get('conflict_pairs')}",
                (v.detail or "")[:150],
            )
        )

print("=" * 90)
print(f"过拒原因分布（共 {sum(buckets.values())} 条）")
print("=" * 90)
for k, n in buckets.most_common():
    print(f"  {k:<16} {n}")
    for cid, q, chk, detail in examples[k]:
        print(f"      [{cid}] {q}")
        print(f"          {chk}")
        print(f"          {detail}")
    print()
