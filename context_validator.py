"""上下文校验器（Context Validator）—— 判断「证据是否足以作答」

它在流水线里的位置（关键）：

    检索 → 组装 → 【校验】 → 生成 → 引用检测 → 快照
                     ↑
              在这里决定：放行 / 重建 / 拒答

为什么必须独立成模块，而不是继续塞在 Generator 里：
    1. 职责单一：Generator 只管"怎么答"，Validator 只管"能不能答"
    2. 可独立评测：校验器自己的准确率（该拒的拒了没 / 不该拒的放行了没）
       可以单独度量 —— 混在生成器里就永远测不出来
    3. 可替换：换模型、换 prompt 不影响校验规则；反之亦然

三种结局（action）：
    pass     证据充分            → 放行给 Generator
    rebuild  相关证据被组装砍掉了 → 放宽组装重来（有次数上限）
    refuse   知识库确实没有 / 证据互相矛盾 → 拒答

三个校验维度：
    ① 相关性 relevance   —— 入选证据里最相关的那条有多相关
    ② 充分性 sufficiency —— 候选集里是否存在比入选证据更相关、却被裁掉的证据
    ③ 一致性 consistency —— 入选证据之间是否互相矛盾

【最关键的判断】rebuild 与 refuse 的分野：
    证据不够时，先问一句：是"压根没召回到"，还是"召回到了但被组装砍了"？
        候选集里也没有 → refuse （知识库确实没有，重建也是浪费）
        候选集里有     → rebuild（放宽组装，而**不是**重新检索）
    这个判断直接复用组装阶段留下的 dropped[].reason —— 校验器和归因器共用同一份留痕。

【阈值为什么必须标定，不能拍脑袋】
    MIN_RELEVANCE 定高 → 本来能答的问题被拒 → 用户拿不到答案（可用性损失）
    MIN_RELEVANCE 定低 → 该拒的没拒，模型基于无关证据硬答（幻觉，代价更大）
    这两类错误代价不对称，所以必须**分别度量**「漏拒率」和「过拒率」，
    不能用"准确率"一个数字糊过去。标定方法：在评测集上扫阈值，画出两条错误率曲线再取交点。
"""
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import Candidate, Evidence, ValidatorResult


# ============================================================
# ① 文本特征提取
# ============================================================

# 问法用词：决定"问什么"，但不承载"要覆盖的内容"
_QUESTION_WORDS = (
    "是什么", "有什么", "有哪些", "为什么", "怎么样", "怎么", "如何",
    "哪些", "什么", "多少", "是否", "能否", "哪里", "哪个", "介绍", "说明",
)

# 虚词：不承载信息，参与覆盖率计算只会制造假阳性
_STOP_CHARS = set("的了呢吗吧啊呀哦嗯和与及或是在有对从把被给向为以等这那其个两")

# 英文/数字词（长度 >= 2），按【词】匹配而非按字符
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def extract_features(text: str) -> Tuple[Set[str], Set[str]]:
    """把文本拆成（英文词集合, 中文字集合）。

    为什么要分两路、而不是统一按字符算：
        字符级覆盖率对短英文缩写会彻底失灵 —— "SSE 是什么？" 的字符集只有 {s, e}，
        几乎所有含英文的文档都能"命中"，相关性就失去区分度（doc_01 和 doc_09 一样高）。
        所以：英文按【词】匹配（sse 必须作为完整词出现），中文按【字】匹配
        （没装分词器时的妥协，真实场景应换成 jieba/embedding）。
    """
    en = {m.group(0).lower() for m in _EN_WORD_RE.finditer(text)}

    zh_source = _EN_WORD_RE.sub(" ", text)
    for w in _QUESTION_WORDS:
        zh_source = zh_source.replace(w, "")
    zh = {
        ch
        for ch in zh_source
        if ("\u4e00" <= ch <= "\u9fff") and ch not in _STOP_CHARS
    }
    return en, zh


# 相关性权重：实体词（英文/数字）与内容字
# 为什么实体权重更高：
#   找到"问的对象"就找到了候选；而"问的维度"（特点 / 优势 / 作用 / 原理）
#   往往根本不出现在文档里 —— 文档写的是内容，不会写"这是特点"。
#   等权平均会让一条完全命中的证据被维度词拖到阈值以下，把能答的问题判成"知识库没有"。
W_ENTITY = 0.7
W_CONTENT = 0.3


def relevance(query_en: Set[str], query_zh: Set[str], text: str) -> float:
    """相关性 = 实体词覆盖率 与 内容字覆盖率 的加权平均（权重按"有内容的那一路"归一化）。

    只取加权平均而不是固定权重求和：不同问题天然只落在某一路
    （"SSE 是什么" 只有英文路，"上下文工程包含哪些内容" 只有中文路），
    固定权重会让纯中文/纯英文的问题被系统性压低分。
    """
    low = text.lower()
    parts: List[Tuple[float, float]] = []  # (权重, 覆盖率)
    if query_en:
        parts.append((W_ENTITY, sum(1 for w in query_en if w in low) / len(query_en)))
    if query_zh:
        parts.append((W_CONTENT, sum(1 for c in query_zh if c in low) / len(query_zh)))
    if not parts:
        return 0.0
    total_w = sum(w for w, _ in parts)
    return round(sum(w * c for w, c in parts) / total_w, 4)


# ============================================================
# ② 一致性检测
# ============================================================

# 否定 / 限定表述：出现这些词说明该证据在下"排除性"结论
_NEGATION_MARKERS = (
    "不依赖", "不需要", "不支持", "不适用", "不能", "不会", "不是", "不再",
    "不含", "不包括", "没有", "无需", "只依赖", "只支持", "只需", "仅需",
    "仅支持", "只需要",
)


def _has_negation(text: str) -> bool:
    return any(m in text for m in _NEGATION_MARKERS)


def _topic_overlap(a: str, b: str) -> int:
    """两条证据在讲同一件事的程度（中文字交集大小）"""
    _, za = extract_features(a)
    _, zb = extract_features(b)
    return len(za & zb)


def detect_conflicts(
    query: str,
    evidence: Sequence[Evidence],
    min_overlap: int,
    min_query_overlap: int = 3,
) -> List[Dict[str, str]]:
    """在入选证据里找出**与问题相关**的冲突对。

    简化规则（确定性、可复现）：
        两条证据讨论同一主题（实义字交集 >= min_overlap），
        但一条含否定/限定表述、另一条不含 → 判定为潜在冲突。

    【重要前提】只检测"与问题本身相关"的冲突 —— 不是所有矛盾都要拒绝回答：
        两条证据在某个点上互相矛盾，但如果那个点**不是问题所问**，
        就不该因此把整个问题拒掉。
        例：问「Milvus 是什么？」，而上下文里两条证据在"部署依赖"上冲突
            → 这个冲突跟问题无关，照常作答；
            但问「Milvus 部署依赖 MinIO 吗？」，同样的冲突就必须拒答。
        实现上用"问题与证据的中文实义字交集"来做这个判断。

    边界（诚实说明）：
        当问题里没有中文实义字时（如只问「Milvus 是什么？」），
        本实现会**跳过**冲突检测。理由是：这类问题问的是"整体介绍"，
        局部矛盾不足以否定整个回答。纯英文问答场景下这个判据会失效，
        真实系统应换成"问题意图 → 需要哪个字段 → 只校验该字段的冲突"。

    为什么不用 LLM 判断：
        校验结论必须可复现、可解释、零成本。LLM 适合做"候选召回"，
        不适合做最终裁决 —— 它每次给的答案可能不一样，评测就没法回归。
    """
    q_en, q_zh = extract_features(query)
    # 问题里没有中文实义字 → 无法判断冲突是否切题 → 不因此拒答
    if not q_zh:
        return []

    def conflict_is_on_topic(text: str) -> bool:
        _, z = extract_features(text)
        return len(q_zh & z) >= min_query_overlap

    out: List[Dict[str, str]] = []
    items = list(evidence)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            # 两条都带否定（或都不带）→ 不构成矛盾
            if _has_negation(a.text) == _has_negation(b.text):
                continue
            if _topic_overlap(a.text, b.text) < min_overlap:
                continue
            # 冲突主题必须与问题相关，否则不该影响回答
            if not (conflict_is_on_topic(a.text) and conflict_is_on_topic(b.text)):
                continue
            neg, pos = (a, b) if _has_negation(a.text) else (b, a)
            out.append(
                {
                    "negative": neg.id,
                    "positive": pos.id,
                    "detail": f"{neg.id} 含否定/限定表述，与 {pos.id} 的结论可能相悖",
                }
            )
    return out


# ============================================================
# ③ 校验器
# ============================================================
class ContextValidator:
    """上下文校验器：判断证据是否足以作答，并决定下一步动作。"""

    # 相关性下限：低于它视为"与问题无关"（经验起点，实际应扫阈值标定）
    MIN_RELEVANCE = 0.6
    # 两条证据的中文实义字交集达到此数，视为"在讨论同一主题"
    MIN_TOPIC_OVERLAP = 6
    # 比较相关性时的容差：避免浮点误差引发无意义的 rebuild
    EPS = 0.05

    def __init__(
        self,
        min_relevance: Optional[float] = None,
        min_topic_overlap: Optional[int] = None,
    ):
        self.min_relevance = self.MIN_RELEVANCE if min_relevance is None else min_relevance
        self.min_topic_overlap = (
            self.MIN_TOPIC_OVERLAP if min_topic_overlap is None else min_topic_overlap
        )

    # ---------------- 对外主入口 ----------------
    def validate(
        self,
        query: str,
        selected: Sequence[Evidence],
        candidates: Sequence[Candidate],
    ) -> ValidatorResult:
        q_en, q_zh = extract_features(query)

        best_selected = max(
            (relevance(q_en, q_zh, e.text) for e in selected), default=0.0
        )
        best_candidate = max(
            (relevance(q_en, q_zh, c.text) for c in candidates), default=0.0
        )

        checks: Dict[str, float] = {
            "relevance": best_selected,
            "max_candidate_relevance": best_candidate,
            "evidence_count": float(len(selected)),
        }

        # ---------- ① 入选证据全都不相关 ----------
        if best_selected < self.min_relevance:
            if best_candidate >= self.min_relevance:
                # 候选里有相关的 → 是组装把它砍掉了 → 放宽组装重来
                return ValidatorResult(
                    status="fail",
                    checks=checks,
                    action="rebuild",
                    detail=(
                        f"入选证据最高相关性 {best_selected} < 阈值 {self.min_relevance}，"
                        f"但候选集里存在相关性 {best_candidate} 的证据 —— "
                        f"属于【组装阶段被裁】，应放宽组装重试而非拒答"
                    ),
                )
            # 候选里也没有 → 知识库确实没有这个话题
            return ValidatorResult(
                status="fail",
                checks=checks,
                action="refuse",
                detail=(
                    f"候选集最高相关性仅 {best_candidate} < 阈值 {self.min_relevance} —— "
                    f"知识库中不存在与问题相关的内容，应拒答"
                ),
            )

        # ---------- ② 有更相关的候选证据被裁掉了 ----------
        if best_candidate > best_selected + self.EPS:
            checks["relevance_gap"] = round(best_candidate - best_selected, 4)
            return ValidatorResult(
                status="fail",
                checks=checks,
                action="rebuild",
                detail=(
                    f"候选集存在更相关的证据（{best_candidate} > 入选最高 {best_selected}）—— "
                    f"【组装阶段把更好的证据裁掉了】，应放宽组装重试"
                ),
            )

        # ---------- ③ 证据之间互相矛盾（仅当冲突与问题相关） ----------
        conflicts = detect_conflicts(query, selected, self.min_topic_overlap)
        checks["conflict_pairs"] = float(len(conflicts))
        if conflicts:
            return ValidatorResult(
                status="fail",
                checks=checks,
                action="refuse",
                detail=(
                    "证据之间存在冲突："
                    + "；".join(c["detail"] for c in conflicts)
                    + "。无法可靠裁决时应拒答，而不是选择性采信其中一条"
                ),
            )

        # ---------- 全部通过 ----------
        checks["consistency"] = 1.0
        return ValidatorResult(
            status="pass",
            checks=checks,
            action=None,
            detail="证据充分且无冲突，放行生成",
        )
