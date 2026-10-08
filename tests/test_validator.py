"""Context Validator 单元测试

运行：
    cd context_ops
    python tests/test_validator.py

【为什么校验器必须有自己的单元测试，而不能只靠评测集】
    评测集只能覆盖"真实发生过的场景"。这次就撞上了：
    校验器的 rebuild 分支在现有 15 条用例上**一次都没被触发过** ——
    因为这些问题里，相关性最高的那条证据恰好都进了入选集。
    换句话说，那行代码等于没验证过。**没验证过的代码路径 = 不存在的功能。**

    所以这里手工构造两类场景把它逼出来：
      ① 候选集里有相关证据，但入选集里没有（模拟"组装把好证据裁掉了"）
      ② 组装策略本身写错了（只选最不相关的那条），看校验器 + 重建能否兜住
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from context_builder import ContextStrategy, estimate_tokens, render_evidence  # noqa: E402
from context_validator import (  # noqa: E402
    ContextValidator,
    detect_conflicts,
    extract_features,
    relevance,
)
from pipeline import InMemoryRetriever, Pipeline  # noqa: E402
from schemas import (  # noqa: E402
    Budget,
    BuildResult,
    Candidate,
    DropReason,
    Dropped,
    Evidence,
)

_pass = 0
_fail = 0


def section(name: str) -> None:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")


def check(name: str, cond: bool, detail: str = "") -> None:
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}" + (f"   → {detail}" if detail else ""))


def ev(c: Candidate, position: int = 0) -> Evidence:
    return Evidence(
        id=c.id,
        text=c.text,
        score=c.score,
        position=position,
        tokens=estimate_tokens(c.text),
        source=c.source,
    )


# ============================================================
# 测试夹具
# ============================================================
MILVUS_REL = Candidate(
    id="rel_milvus", text="Milvus 是开源向量数据库，支持 HNSW 索引，常用于语义检索。", score=0.9
)
REDIS_IRR = Candidate(
    id="irr_redis", text="Redis 是内存键值数据库，支持 TTL 过期和多种数据结构。", score=0.1
)
# 一对互相矛盾的证据（同一主题：Milvus 部署是否依赖 MinIO）
DEP_POS = Candidate(
    id="dep_pos", text="Milvus 的部署依赖 etcd 与 MinIO 组件。", score=0.8
)
DEP_NEG = Candidate(
    id="dep_neg", text="Milvus 的部署只需要 etcd，不依赖 MinIO 等对象存储组件。", score=0.8
)

validator = ContextValidator()


# ============================================================
# 一、特征提取
# ============================================================
section("一、特征提取：英文按词、中文按字、问法词与虚词剔除")

en, zh = extract_features("SSE 是什么？")
check("英文缩写提取为完整词（不是散字符）", en == {"sse"}, f"得到 {en}")
check("问法词「是什么」被剔除后中文为空", zh == set(), f"得到 {zh}")

en, zh = extract_features("上下文工程包含哪些内容？")
check("中文实义字被保留", {"上", "下", "文", "工", "程", "内", "容"} <= zh, f"得到 {zh}")
check("问法词「哪些」被剔除", "哪" not in zh and "些" not in zh, f"得到 {zh}")

en, zh = extract_features("RRF 的公式是什么？")
check("中英混合：英文词与中文实义字各自提取", en == {"rrf"} and "公" in zh, f"en={en} zh={zh}")

# ============================================================
# 二、相关性：实体权重高于维度词
# ============================================================
section("二、相关性打分：实体命中的权重必须高于维度词")

q_en, q_zh = extract_features("FastAPI 有什么特点？")
r = relevance(q_en, q_zh, "FastAPI 是一个现代、高性能的 Python Web 框架，基于 Starlette。")
check(
    "维度词「特点」不在文档里时，仍判为相关（约 0.7，过阈值 0.6）",
    abs(r - 0.7) < 0.01,
    f"得到 {r} —— 若用等权平均会得到 0.5，直接把能答的问题判成「知识库没有」→ 过拒",
)

q_en, q_zh = extract_features("Milvus 是什么？")
check(
    "对象命中、中文路为空 → 满分",
    relevance(q_en, q_zh, "Milvus 是开源向量数据库。") == 1.0,
)

q_en, q_zh = extract_features("今天天气怎么样？")
check(
    "完全无关 → 0 分",
    relevance(q_en, q_zh, "Milvus 是开源向量数据库。") == 0.0,
)

# ============================================================
# 三、校验器三种结局
# ============================================================
section("三、校验器：pass / rebuild / refuse 三态")

# --- pass ---
v = validator.validate("Milvus 是什么？", [ev(MILVUS_REL)], [MILVUS_REL, REDIS_IRR])
check("入选证据相关 → pass（放行生成）", v.status == "pass" and v.action is None, f"得到 {v.action}")

# --- rebuild：候选里有，入选里没有 ---
v = validator.validate("Milvus 是什么？", [ev(REDIS_IRR)], [MILVUS_REL, REDIS_IRR])
check(
    "入选不相关但候选里有相关 → rebuild（是组装砍错了，不是知识库没有）",
    v.action == "rebuild",
    f"得到 {v.action} / detail={v.detail}",
)
check(
    "rebuild 的判定依据被写进 checks（相关性对比）",
    v.checks.get("max_candidate_relevance", 0) > v.checks.get("relevance", 1),
    f"checks={v.checks}",
)

# --- refuse：候选里也没有 ---
v = validator.validate("今天天气怎么样？", [ev(REDIS_IRR)], [REDIS_IRR])
check(
    "候选里也没有相关内容 → refuse（知识库确实没有，重建是浪费）",
    v.action == "refuse",
    f"得到 {v.action}",
)

# ============================================================
# 四、冲突裁决：只拒"与问题相关"的冲突
# ============================================================
section("四、冲突裁决：冲突必须与问题相关才影响回答")

conflicts = detect_conflicts(
    "Milvus 部署依赖 MinIO 吗？", [ev(DEP_POS), ev(DEP_NEG)], validator.min_topic_overlap
)
check("识别出证据间的矛盾对", len(conflicts) == 1, f"得到 {conflicts}")

v = validator.validate("Milvus 部署依赖 MinIO 吗？", [ev(DEP_POS), ev(DEP_NEG)], [DEP_POS, DEP_NEG])
check(
    "问的正是冲突点 → refuse（不给结论，而不是选择性采信一条）",
    v.action == "refuse" and "冲突" in (v.detail or ""),
    f"得到 action={v.action} detail={v.detail}",
)
check("冲突对数被记录进 checks", v.checks.get("conflict_pairs") == 1.0, f"checks={v.checks}")

# 同一对矛盾证据，但问题问的是别的方面 → 不该因此拒答
conflicts2 = detect_conflicts(
    "Milvus 是什么？", [ev(DEP_POS), ev(DEP_NEG)], validator.min_topic_overlap
)
check(
    "冲突与问题无关时不判定冲突（问「Milvus 是什么」，而它们在「部署依赖」上矛盾）",
    len(conflicts2) == 0,
    f"得到 {conflicts2} —— 若在此拒答，会把「Milvus 是什么」这种常规问题误杀",
)

v = validator.validate("Milvus 是什么？", [ev(DEP_POS), ev(DEP_NEG)], [DEP_POS, DEP_NEG])
check(
    "无关冲突不阻断回答 → pass",
    v.status == "pass",
    f"得到 action={v.action}",
)

# ============================================================
# 五、端到端：组装策略写错时，校验器 + 重建能否兜住
# ============================================================
section("五、端到端：把 rebuild 那条路强制走通")

DOCS = [
    {"id": "doc_a", "text": "Milvus 是开源向量数据库，支持 HNSW 索引，常用于语义检索。"},
    {"id": "doc_b", "text": "Neo4j 是图数据库，使用 Cypher 查询语言遍历实体关系。"},
    {"id": "doc_c", "text": "Redis 是内存键值数据库，支持 TTL 过期。"},
]


class BrokenStrategy(ContextStrategy):
    """故意选【最不相关】的那条证据 —— 模拟组装逻辑写错（比如排序方向反了）。

    用途：验证"就算组装策略有 bug，校验器 + 重建也能兜住"。
    这在真实系统里对应一类事故：改了排序逻辑但没测出来，线上答案开始变差。
    """

    version = "broken_for_test"

    def build(self, query, candidates, system_prompt, history, limit) -> BuildResult:
        raw = sorted(candidates, key=lambda c: c.score)[:1]  # 取分数最低的那条
        selected = [
            Evidence(
                id=c.id,
                text=c.text,
                score=c.score,
                position=0,
                tokens=estimate_tokens(c.text),
                source=c.source,
            )
            for c in raw
        ]
        dropped = [
            Dropped(id=c.id, score=c.score, reason=DropReason.LOW_SCORE)
            for c in candidates
            if c.id not in {s.id for s in selected}
        ]
        tokens = sum(e.tokens for e in selected)
        return BuildResult(
            strategy_version=self.version,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_evidence(selected)},
            ],
            selected=selected,
            dropped=dropped,
            budget=Budget(system=0, history=0, evidence=tokens, used=tokens, limit=limit),
        )


# --- 不带校验器：组装错了就只能错下去 ---
pipe_broken = Pipeline(
    retriever=InMemoryRetriever(DOCS), strategy=BrokenStrategy(), validator=None
)
snap_broken = pipe_broken.run("Milvus 是什么？")
check(
    "不带校验器时，组装选错就直接答错（对照组）",
    not snap_broken.answer.refused and "doc_a" not in snap_broken.answer.cited_evidence,
    f"引用={snap_broken.answer.cited_evidence}",
)

# --- 带校验器：应该走 rebuild 并救回正确答案 ---
pipe_fixed = Pipeline(
    retriever=InMemoryRetriever(DOCS),
    strategy=BrokenStrategy(),
    validator=ContextValidator(),
    max_rebuild=1,
)
snap_fixed = pipe_fixed.run("Milvus 是什么？")
check(
    "带校验器时触发了重建（策略版本带 +rebuild 标记）",
    "+rebuild" in snap_fixed.context.strategy_version,
    f"得到 {snap_fixed.context.strategy_version}",
)
check(
    "重建后答案引用了正确证据 doc_a",
    "doc_a" in snap_fixed.answer.cited_evidence,
    f"引用={snap_fixed.answer.cited_evidence}",
)
check(
    "重建次数被写进快照（degraded 标记）",
    snap_fixed.generation.degraded is True,
)

# --- 重建上限：不能变成另一种死循环 ---
pipe_nolimit = Pipeline(
    retriever=InMemoryRetriever(DOCS),
    strategy=BrokenStrategy(),
    validator=ContextValidator(),
    max_rebuild=0,  # 禁止重建
)
snap_nolimit = pipe_nolimit.run("Milvus 是什么？")
check(
    "max_rebuild=0 时不重建（重建有次数上限，否则是另一种死循环）",
    "+rebuild" not in snap_nolimit.context.strategy_version,
    f"得到 {snap_nolimit.context.strategy_version}",
)

# ============================================================
# 汇总
# ============================================================
print(f"\n{'=' * 72}")
print(f"结果：{_pass} 项通过 / {_fail} 项失败 / 共 {_pass + _fail} 项")
print("=" * 72)
sys.exit(1 if _fail else 0)
