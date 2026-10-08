"""从主题定义生成知识库与评测集

【为什么必须有这个脚本，而不是直接手写 JSONL】
    1. **gold 标签必须是"构造时确定"的，不能事后人工补**——
       200 条用例手工标 gold，一定会出现"标错/漏标/标了自己都忘了为什么"的情况，
       而这种错误**在评测结果里看不出来**，只会让指标莫名其妙地偏低或偏高。
       脚本把"哪条文档是标准答案"变成构造步骤的副产品，从根上消除这类错误。
    2. **数据集要可复现、可审计**：指标异常时能追回到"是哪个主题的哪条文档"。
    3. **扩充时不会手抖写错 id / 引用到不存在的文档**（脚本会校验并报错）。

【数据来源】
    data/_gen_a.json  data/_gen_b.json  data/_gen_c.json
    每个文件是"主题"数组，每个主题含 4 条文档（gold / distractor×2 / conflict）
    与 6 条查询（4 条 answer + 1 条库外 refuse + 1 条冲突 refuse）。

【产出】
    data/knowledge_base.jsonl   文档（带 topic / role，便于归因分析）
    data/eval_cases.jsonl       用例（带 category / topic，gold 已解析成真实 doc_id）

运行：
    cd context_ops
    python data/build_dataset.py
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

GEN_FILES = ("_gen_a.json", "_gen_b.json", "_gen_c.json", "_gen_d.json")

# ------------------------------------------------------------
# 文档权威度：模拟真实知识库的「来源分级」
# ------------------------------------------------------------
# 【为什么必须有这个字段】
#   没有它，系统面对"同一个主题的两份矛盾文档"时**没有任何裁决依据**，
#   只能一律拒答。实测在大数据集上这会直接把过拒率推到 59% ——
#   等于一半以上的问题答不了，产品不可用。
#   拒答对"数据源有矛盾"这种数据质量问题来说是过度反应：
#   能判断谁更可信就该采信谁，实在判断不了才拒答。
#
# 【真实系统里这个值从哪来】
#   不是手写的，而是从来源推断：官方文档 / 内部wiki / 社区帖子 /
#   更新时间新旧 / 被引用次数 / 人工标注。这里按文档角色直接赋固定值，
#   是为了让数据集自带一个**可裁决信号**，从而能测出裁决逻辑是否生效。
AUTHORITY_BY_ROLE = {
    "gold": 0.9,        # 官方文档
    "distractor": 0.6,  # 参考资料
    "conflict": 0.3,    # 未经验证的社区内容
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chars(text: str):
    """取实义字符（去问法词与虚词），用于相关性启发式判断"""
    t = text
    for w in ("是什么", "有什么", "有哪些", "为什么", "怎么样", "怎么", "如何", "哪些", "什么", "多少", "是否", "能否", "哪里", "哪个"):
        t = t.replace(w, "")
    return {c.lower() for c in t if c.isalnum() or "\u4e00" <= c <= "\u9fff"}


def _overlap(a: str, b: str) -> float:
    ca, cb = _chars(a), _chars(b)
    if not ca:
        return 0.0
    return len(ca & cb) / len(ca)


def classify_refusal(query: str, topic_docs):
    """区分两种"该拒答"，返回 (类别, 与 conflict 文档的重叠度)。

        · conflict   —— 问题问的正是证据互相矛盾的那个点（知识库里**有**相关内容，但说法打架）
        · out_of_kb  —— 知识库里根本没有这个话题

    为什么要分开：两者的修复方向完全不同 ——
        冲突类多 → 该去治理数据源的矛盾；
        库外类多 → 该去扩知识库。
        合成一个"拒答"就分不清了。

    判据（只用 conflict 文档的重叠度，不再和别的文档比）：
        初版写成"与 conflict 的重叠 >= 与其它文档的重叠"，结果误判了一条 ——
        有条冲突查询和 gold 的字面重叠反而更高（都提到"召回一方"），
        于是被丢进了 out_of_kb。库外问题（问天气、问报销）与任何文档重叠都极低，
        本来就不可能越界，多加那个比较纯属自找麻烦。
    """
    conflict_doc = next((d for d in topic_docs if d.get("role") == "conflict"), None)
    score = _overlap(query, conflict_doc["text"]) if conflict_doc else 0.0
    return ("conflict" if score >= 0.5 else "out_of_kb"), round(score, 2)


def main() -> int:
    topics = []
    for name in GEN_FILES:
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            print(f"[跳过] 未找到 {name}")
            continue
        data = load_json(path)
        # 兼容"直接是数组"和"包了一层 {topics: [...]}"两种写法
        if isinstance(data, dict):
            data = data.get("topics") or data.get("data") or []
        print(f"[读入] {name}: {len(data)} 个主题")
        topics.extend(data)

    if not topics:
        print("没有可用数据，退出")
        return 1

    docs = []
    cases = []
    errors = []
    warnings = []

    for t in topics:
        topic_name = t.get("topic", "（未命名主题）")
        role_to_ids = {}
        topic_docs = t.get("docs", [])

        # ---------- 分配文档 id ----------
        for d in topic_docs:
            role = d.get("role", "unknown")
            doc_id = f"doc_{len(docs) + 1:03d}"
            role_to_ids.setdefault(role, []).append(doc_id)
            docs.append(
                {
                    "id": doc_id,
                    "text": d["text"].strip(),
                    "topic": topic_name,
                    "role": role,
                    "authority": AUTHORITY_BY_ROLE.get(role, 0.5),
                }
            )

        # ---------- 主题结构校验 ----------
        if "gold" not in role_to_ids:
            errors.append(f"[{topic_name}] 缺少 gold 文档")
        for need in ("distractor",):
            if need not in role_to_ids:
                warnings.append(f"[{topic_name}] 缺少 {need} 文档")

        # ---------- 解析查询的 gold ----------
        for q in t.get("queries", []):
            query = q["q"].strip()
            expect = q.get("expect", "answer")
            gold_roles = q.get("gold", []) or []

            gold_ids = []
            for role in gold_roles:
                ids = role_to_ids.get(role)
                if not ids:
                    errors.append(
                        f"[{topic_name}] 查询 {query!r} 引用了不存在的 role: {role!r}"
                    )
                    continue
                gold_ids.extend(ids)

            conflict_overlap = None
            if expect == "answer":
                if not gold_ids:
                    errors.append(f"[{topic_name}] answer 类查询 {query!r} 的 gold 为空")
                category = "single" if len(gold_ids) <= 1 else "multi"
            else:
                if gold_ids:
                    warnings.append(
                        f"[{topic_name}] refuse 类查询 {query!r} 竟然带 gold，已清空"
                    )
                    gold_ids = []
                category, conflict_overlap = classify_refusal(query, topic_docs)

            case = {
                "case_id": f"c{len(cases) + 1:03d}",
                "category": category,
                "query": query,
                "gold_evidence": gold_ids,
                "topic": topic_name,
            }
            # 把分类依据写进用例：分类是脚本猜的，留个可核对的数字
            if conflict_overlap is not None:
                case["conflict_overlap"] = conflict_overlap
            cases.append(case)

    # ---------- 全局重复校验 ----------
    text_counter = Counter(d["text"] for d in docs)
    for text, n in text_counter.items():
        if n > 1:
            errors.append(f"文档文本重复 {n} 次：{text[:40]}...")
        # 长度检查
    for d in docs:
        if not (12 <= len(d["text"]) <= 90):
            warnings.append(f"文档长度异常（{len(d['text'])} 字）：{d['id']} {d['text'][:30]}...")

    query_counter = Counter(c["query"] for c in cases)
    for query, n in query_counter.items():
        if n > 1:
            warnings.append(f"查询重复 {n} 次：{query}")

    # ---------- 报告 ----------
    print("\n" + "=" * 74)
    print("数据集构建报告")
    print("=" * 74)
    print(f"  主题数      {len(topics)}")
    print(f"  文档数      {len(docs)}")
    print(f"  用例数      {len(cases)}")
    print(f"  文档角色分布 {dict(Counter(d['role'] for d in docs))}")
    print(f"  用例类型分布 {dict(Counter(c['category'] for c in cases))}")
    print(f"  每题平均候选 {round(sum(len(c['gold_evidence']) for c in cases) / max(1, len(cases)), 2)}")

    if warnings:
        print(f"\n  警告 {len(warnings)} 条（不阻断构建）：")
        for w in warnings[:12]:
            print(f"    · {w}")
        if len(warnings) > 12:
            print(f"    ... 其余 {len(warnings) - 12} 条省略")

    # 拒答分类明细：分类是脚本按字面重叠猜的，打出来供人工核对有没有分错
    refusals = [c for c in cases if c["category"] in ("conflict", "out_of_kb")]
    if refusals:
        picked = refusals[:6] + refusals[-4:]
        print(f"\n  拒答分类明细（共 {len(refusals)} 条，重叠度 < 0.5 判为库外）：")
        for c in picked:
            print(
                f"    [{c['category']:<10}] 重叠={str(c.get('conflict_overlap')):<5} {c['query']}"
            )

    if errors:
        print(f"\n  错误 {len(errors)} 条（**必须修复**）：")
        for e in errors[:20]:
            print(f"    ✗ {e}")
        if len(errors) > 20:
            print(f"    ... 其余 {len(errors) - 20} 条省略")
        print("\n构建中止：先修掉上面的错误")
        return 1

    # ---------- 落盘 ----------
    kb_path = os.path.join(DATA, "knowledge_base.jsonl")
    case_path = os.path.join(DATA, "eval_cases.jsonl")
    with open(kb_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(case_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n  已写入 {kb_path}")
    print(f"  已写入 {case_path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
