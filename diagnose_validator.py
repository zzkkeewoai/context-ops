"""诊断脚本：检查 Validator 的实际判定，以及每条用例的 gold 命中链路"""
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


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


kb = load(os.path.join(ROOT, "data", "knowledge_base.jsonl"))
cases = load(os.path.join(ROOT, "data", "eval_cases.jsonl"))

print("=" * 90)
print("逐条用例诊断 · headtail + validator")
print("=" * 90)

pipe = Pipeline(
    retriever=InMemoryRetriever(kb),
    strategy=get_strategy("headtail"),
    validator=ContextValidator(),
)

for c in cases:
    snap = pipe.run(c["query"])
    v = snap.context.validator
    gold = set(c.get("gold_evidence", []))
    cand_ids = {x.id for x in snap.candidates}
    sel_ids = {e.id for e in snap.context.selected}
    cited = set(snap.answer.cited_evidence)

    print(f"\n[{c['case_id']}] ({c.get('category')}) {c['query']}")
    print(f"  gold={sorted(gold) or '（期望拒答）'}")
    print(f"  候选命中={bool(gold & cand_ids)}  入选命中={bool(gold & sel_ids)}  引用命中={bool(gold & cited)}")
    print(f"  策略版本={snap.context.strategy_version}  拒答={snap.answer.refused}")
    if v:
        print(f"  校验: status={v.status} action={v.action}")
        print(f"        checks={v.checks}")
        print(f"        detail={v.detail}")
    print(f"  上下文证据顺序: {[e.id for e in snap.context.selected]}")
    print(f"  答案引用: {sorted(cited) or '[]'}")
