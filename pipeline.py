"""最简 Pipeline —— 一个问题从进入到快照落库的完整链路

这是「纵向打穿」的最小版本：每个环节都只做最简实现，
但整条链路（含快照落库）是完整的，能立刻跑出基线数字。
后续所有优化都是**替换其中某一环**，而不是重写整个系统。

链路：
    query → ①检索 → ②上下文组装 → ③生成 → ④引用检测 → ⑤快照落库
"""
import hashlib
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context_builder import ContextStrategy, estimate_tokens  # noqa: E402
from schemas import (  # noqa: E402
    AnswerInfo,
    Candidate,
    ContextSnapshot,
    GenerationInfo,
)


# ============================================================
# ① 检索器（最简：字符重叠打分，零外部依赖）
# ============================================================
class InMemoryRetriever:
    """内存检索器：用关键词/字符重叠打分。

    为什么不用 embedding：
      本项目要验证的是「上下文工程」的效果，不是检索算法本身。
      用最简打分器能让项目零依赖跑起来（新人 clone 就能复现），
      真实场景把 retrieve() 换成向量检索即可，接口不变。
    """

    def __init__(self, docs: List[Dict]):
        self.docs = docs  # [{"id": ..., "text": ...}, ...]

    @staticmethod
    def _score(query: str, text: str) -> float:
        """打分 = query 中出现的字符覆盖比例（简化版相关性）。"""
        q_chars = {c for c in query if "\u4e00" <= c <= "\u9fff" or c.isalnum()}
        if not q_chars:
            return 0.0
        hit = sum(1 for c in q_chars if c in text)
        return round(hit / len(q_chars), 4)

    RRF_K = 60  # RRF 融合常数，业界常用 60

    def retrieve(self, query: str, top_k: int = 20) -> List[Candidate]:
        """检索并返回候选（顺序 = 多路召回 RRF 融合后的顺序）。

        【实验设计关键点 —— 这里我踩了两次坑】
        坑 1：第一版直接 `sort by score` 返回 → naive 取前 5 条时，相关性最高的
              那条总在第一位，天然躲开"中间位置注意力衰减"，两个策略结果完全一样。
        坑 2：第二版改成"两路交错"，但交错时仍把相关性路的第一名放在最前，
              等于没打乱 → 结果还是没差异。

        正确做法：**真实实现 RRF 融合**。
            两路召回（相关性路 / 长度路）各自排名，用 1/(k+rank) 累加融合。
            一条证据只有在【两路都排得靠前】时融合排名才高，因此
            「相关性最高的那条」融合后未必仍排第一 —— 这正是真实系统的行为。
        """
        scored = [
            Candidate(
                id=d["id"],
                text=d["text"],
                score=self._score(query, d["text"]),
                source="keyword",
            )
            for d in self.docs
        ]
        by_relevance = sorted(scored, key=lambda c: c.score, reverse=True)
        # 第二路召回：偏好短文本（模拟另一种召回器的排序偏好）
        by_length = sorted(scored, key=lambda c: (len(c.text), -c.score))

        fused: Dict[str, float] = {}
        for ranked in (by_relevance, by_length):
            for rank, c in enumerate(ranked, 1):
                fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (self.RRF_K + rank)

        # 顺序由融合分数决定，但每条候选仍保留自身的相关性 score
        ordered = sorted(scored, key=lambda c: fused.get(c.id, 0.0), reverse=True)
        return ordered[:top_k]


# ============================================================
# ③ 生成器（Mock —— 故意模拟"中间位置注意力衰减"）
# ============================================================
class MockGenerator:
    """模拟一个「有 Lost in the Middle 倾向」的模型。

    为什么用 Mock 而不是真实 LLM：
      1. 本项目验证的是**上下文质量**，Mock 能把"模型能力"这个变量隔离掉；
      2. 零依赖、可复现、零成本；
      3. **可控**：可以故意模拟"忽略中间位置的证据"，从而让
         HeadTailPriority（首尾优先）策略体现出真实收益 ——
         这就是 A/B 对比能出效果差异的原因。

    真实场景替换为 LLM 调用，接口（generate）不变。
    """

    # 中间位置（30%~70%）的注意力衰减系数
    MIDDLE_PENALTY = 0.3
    # 资料相关性下限：低于此值认为"知识库里根本没有相关内容" → 拒答
    RELEVANCE_FLOOR = 0.3

    @staticmethod
    def _relevance(query: str, text: str) -> float:
        """简化的相关性打分（字符覆盖率）。真实场景由模型自身判断。"""
        q_chars = {c for c in query if "\u4e00" <= c <= "\u9fff" or c.isalnum()}
        if not q_chars:
            return 0.0
        hit = sum(1 for c in q_chars if c in text)
        return hit / len(q_chars)

    def generate(self, query: str, evidence) -> AnswerInfo:
        """生成答案。

        关键：**位置惩罚只影响"选哪条证据"，不影响"是否拒答"**。
          · 所有证据都不相关（max_relevance < FLOOR）→ 拒答（知识库没有）
          · 有相关证据但选错（最相关的那条被位置惩罚压下去）→ 答错
        这样才能区分"该拒答"和"答错"两种失败，归因器才有意义。
        """
        if not evidence:
            return AnswerInfo(text="当前资料不足，无法回答。", cited_evidence=[], refused=True)

        n = len(evidence)
        chosen = None
        best_weight = -1.0
        max_relevance = 0.0

        for e in evidence:
            pos_ratio = e.position / max(1, n - 1)
            penalty = self.MIDDLE_PENALTY if 0.3 <= pos_ratio <= 0.7 else 1.0
            relevance = self._relevance(query, e.text)
            max_relevance = max(max_relevance, relevance)
            weight = penalty * relevance
            if weight > best_weight:
                best_weight, chosen = weight, e

        # 知识库里确实没有相关内容 → 拒答
        if max_relevance < self.RELEVANCE_FLOOR or chosen is None:
            return AnswerInfo(
                text="提供的资料与问题不相关，无法给出可靠回答。",
                cited_evidence=[],
                refused=True,
            )

        return AnswerInfo(
            text=f"根据资料：{chosen.text}（引用 [{chosen.position + 1}]）",
            cited_evidence=[chosen.id],
            refused=False,
        )


# ============================================================
# 快照存储（最简：JSONL 追加，可换成 Postgres）
# ============================================================
class SnapshotStore:
    """快照存储：先落 JSONL（零依赖），后续可平滑换成 Postgres。

    为什么存储可以后换、Schema 不能后改：
      存哪儿只影响查询方式，字段错了要重跑全部数据。
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def save(self, snapshot: ContextSnapshot) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(snapshot.model_dump_json() + "\n")

    def load_all(self) -> List[ContextSnapshot]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(ContextSnapshot.model_validate_json(line))
        return out

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


# ============================================================
# Pipeline：把各环节串起来
# ============================================================
class Pipeline:
    """一次请求的完整编排。

    注意：Pipeline 只负责"按顺序调用"，不含业务逻辑——
    业务逻辑都在各模块内部。这样替换任一模块都不影响其它。
    """

    SYSTEM_PROMPT = "你是技术文档问答助手，只根据提供的资料回答，并标注引用编号。"

    def __init__(
        self,
        retriever: InMemoryRetriever,
        strategy: ContextStrategy,
        generator: Optional[MockGenerator] = None,
        store: Optional[SnapshotStore] = None,
        limit: int = 4096,
    ):
        self.retriever = retriever
        self.strategy = strategy
        self.generator = generator or MockGenerator()
        self.store = store
        self.limit = limit

    def run(self, query: str, session_id: str = "default", history=None) -> ContextSnapshot:
        history = history or []
        t0 = time.time()

        # ① 检索
        t_ret = time.time()
        candidates = self.retriever.retrieve(query, top_k=20)
        retrieval_latency = (time.time() - t_ret) * 1000

        # ② 上下文组装（策略可插拔）
        build = self.strategy.build(
            query=query,
            candidates=candidates,
            system_prompt=self.SYSTEM_PROMPT,
            history=history,
            limit=self.limit,
        )

        # ③ 生成
        t_gen = time.time()
        answer = self.generator.generate(query, build.selected)
        generation_latency = (time.time() - t_gen) * 1000

        # ④ 引用检测：答案里标注了 [n]，映射回证据 id
        cited = self._detect_citations(answer.text, build.selected, answer.cited_evidence)

        # ⑤ 组装快照
        prompt_text = json.dumps(build.messages, ensure_ascii=False)
        tokens_in = estimate_tokens(prompt_text)
        tokens_out = estimate_tokens(answer.text)
        snapshot = ContextSnapshot(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            query_original=query,
            candidates=candidates,
            retrieval_latency_ms=round(retrieval_latency, 2),
            context=build,
            generation=GenerationInfo(
                model="mock-generator",
                prompt_hash=hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:12],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=round(generation_latency, 2),
                provider="mock",
            ),
            answer=AnswerInfo(
                text=answer.text, cited_evidence=cited, refused=answer.refused
            ),
            total_latency_ms=round((time.time() - t0) * 1000, 2),
            cost_usd=round(tokens_in * 1e-6 + tokens_out * 2e-6, 8),
        )

        if self.store:
            self.store.save(snapshot)
        return snapshot

    @staticmethod
    def _detect_citations(text: str, evidence, hinted: List[str]) -> List[str]:
        """引用检测：把答案里的 [1] [2] 映射回证据 id。

        这是算「上下文利用率」的唯一数据来源——没有它，就无法判断
        "放进去的证据到底有没有被模型用上"。
        真实场景可用更鲁棒的方式（正则 + 语义匹配），接口不变。
        """
        import re

        cited = set(hinted or [])
        for m in re.finditer(r"\[(\d+)\]", text):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(evidence):
                cited.add(evidence[idx].id)
        return sorted(cited)
