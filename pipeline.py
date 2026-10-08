"""最简 Pipeline —— 一个问题从进入到快照落库的完整链路

这是「纵向打穿」的最小版本：每个环节都只做最简实现，
但整条链路（含快照落库）是完整的，能立刻跑出基线数字。
后续所有优化都是**替换其中某一环**，而不是重写整个系统。

链路：
    query → ①检索 → ②上下文组装 → ②'校验 → ③生成 → ④引用检测 → ⑤快照落库

关于 ②' 校验的位置（v2 新增，重要）：
    校验必须在【生成之前】，而不是让生成器自己判断"该不该答"。
    早先的实现把拒答判断写在 MockGenerator 里（max_relevance < FLOOR 就拒答），
    结果是：校验逻辑和生成逻辑缠在一起，校验器自己的准确率永远测不出来。
    独立成模块后，"该拒的拒了没 / 不该拒的放行了没" 才能被单独评测。
"""
import hashlib
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context_builder import (  # noqa: E402
    ContextStrategy,
    estimate_tokens,
    render_evidence,
)
from context_validator import ContextValidator, extract_features, relevance  # noqa: E402
from schemas import (  # noqa: E402
    AnswerInfo,
    Budget,
    BuildResult,
    Candidate,
    ContextSnapshot,
    DropReason,
    Dropped,
    Evidence,
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
                authority=d.get("authority", 0.5),
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

    【职责边界 —— v2 修正】
        本类**不再判断"该不该拒答"**。早先版本里有 `RELEVANCE_FLOOR`，
        低于它就返回拒答 —— 那是把"能不能答"的判断塞进了"怎么答"的模块里，
        导致校验逻辑无法单独评测。
        现在的分工：
            ContextValidator 决定 pass / rebuild / refuse
            MockGenerator     只在 pass 之后负责"从已有证据里选一条来组织答案"
        唯一保留的拒答分支是"证据为空"——那是"没得答"，不是"判断能不能答"。
    """

    # 中间位置（30%~70%）的注意力衰减系数
    MIDDLE_PENALTY = 0.3

    @staticmethod
    def _relevance(query: str, text: str) -> float:
        """简化的相关性打分（字符覆盖率）。真实场景由模型自身判断。"""
        q_chars = {c for c in query if "\u4e00" <= c <= "\u9fff" or c.isalnum()}
        if not q_chars:
            return 0.0
        hit = sum(1 for c in q_chars if c in text)
        return hit / len(q_chars)

    def generate(self, query: str, evidence) -> AnswerInfo:
        """生成答案（前置条件：Validator 已判定证据充分）。

        核心行为：**位置惩罚只影响"选哪条证据"**。
          · 最相关的证据若落在上下文中间，会被位置惩罚压下去 → 答错
          · 首尾优先排列把最相关的放到头尾 → 答对
        这正是 A/B 对比能出效果差异的来源。
        """
        if not evidence:
            return AnswerInfo(text="未检索到任何相关资料。", cited_evidence=[], refused=True)

        n = len(evidence)
        chosen = None
        best_weight = -1.0

        for e in evidence:
            pos_ratio = e.position / max(1, n - 1)
            penalty = self.MIDDLE_PENALTY if 0.3 <= pos_ratio <= 0.7 else 1.0
            relevance_score = self._relevance(query, e.text)
            weight = penalty * relevance_score
            if weight > best_weight:
                best_weight, chosen = weight, e

        if chosen is None:
            return AnswerInfo(text="未检索到任何相关资料。", cited_evidence=[], refused=True)

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
        validator: Optional[ContextValidator] = None,
        limit: int = 4096,
        top_k: int = 20,
        max_rebuild: int = 1,
    ):
        self.retriever = retriever
        self.strategy = strategy
        self.generator = generator or MockGenerator()
        self.store = store
        self.validator = validator
        self.limit = limit
        self.top_k = top_k
        # 重建次数上限：没有上限的重建就是另一种死循环
        # （和 Agent 运行时里的 max_steps 是同一个思想）
        self.max_rebuild = max_rebuild

    # ---------------- 主流程 ----------------
    def run(self, query: str, session_id: str = "default", history=None) -> ContextSnapshot:
        history = history or []
        t0 = time.time()

        # ① 检索
        t_ret = time.time()
        candidates = self.retriever.retrieve(query, top_k=self.top_k)
        retrieval_latency = (time.time() - t_ret) * 1000

        # ② 上下文组装（策略可插拔）
        build = self.strategy.build(
            query=query,
            candidates=candidates,
            system_prompt=self.SYSTEM_PROMPT,
            history=history,
            limit=self.limit,
        )

        # ②' 校验：放行 / 重建 / 拒答
        rebuild_count = 0
        verdict = None
        if self.validator is not None:
            verdict = self.validator.validate(query, build.selected, candidates)
            while verdict.action == "rebuild" and rebuild_count < self.max_rebuild:
                rebuild_count += 1
                build = self._rebuild_context(query, candidates, history)
                verdict = self.validator.validate(query, build.selected, candidates)
            # 裁决要求剔除的证据：真的把它们移出上下文，再交给生成
            if verdict.dropped_evidence:
                build = self._drop_evidence(build, verdict.dropped_evidence)
            build.validator = verdict

        # ③ 生成（拒答时根本不进入生成器 —— 这是职责分离的直接体现）
        t_gen = time.time()
        if verdict is not None and verdict.action == "refuse":
            answer = AnswerInfo(
                text=f"当前知识库无法可靠回答该问题：{verdict.detail}",
                cited_evidence=[],
                refused=True,
            )
        else:
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
                # 重建次数写进快照：它是"组装策略是否够用"的直接证据
                degraded=rebuild_count > 0,
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

    # ---------------- 重建上下文 ----------------
    def _rebuild_context(
        self, query: str, candidates: List[Candidate], history: List[Dict[str, str]]
    ) -> BuildResult:
        """放宽组装：把候选集里**所有相关性达标**的证据都纳入上下文。

        为什么是"放宽组装"而不是"重新检索"（关键设计）：
            能被重建救回来的场景，前提是证据**已经在候选集里**（只是被组装砍了）。
            重新检索一遍纯属浪费；真正该做的是松开组装的约束。
            而"候选集里也没有"的情形，Validator 已经直接判 refuse 了 —— 不浪费这一次重建。
            也就是说：**rebuild 与 refuse 的分野，让重建永远不会做无用功。**

        重建后仍然不达标怎么办：
            循环会再校验一次，此时 best_candidate 与 best_selected 已经相等，
            不会再触发 rebuild，于是落到 pass 或 refuse —— 不会无限循环。
        """
        q_en, q_zh = extract_features(query)
        assert self.validator is not None  # 由调用方保证

        selected: List[Evidence] = []
        for c in candidates:
            if relevance(q_en, q_zh, c.text) >= self.validator.min_relevance:
                selected.append(
                    Evidence(
                        id=c.id,
                        text=c.text,
                        score=c.score,
                        position=-1,
                        tokens=estimate_tokens(c.text),
                        source=c.source,
                        authority=c.authority,
                    )
                )

        # 借用策略自身的排序能力（首尾优先等），避免重建后丢掉排序优化
        if hasattr(self.strategy, "_head_tail_order"):
            selected = self.strategy._head_tail_order(selected)  # type: ignore[attr-defined]
        else:
            selected = [
                it.model_copy(update={"position": i}) for i, it in enumerate(selected)
            ]

        selected_ids = {e.id for e in selected}
        dropped = [
            Dropped(id=c.id, score=c.score, reason=DropReason.LOW_SCORE)
            for c in candidates
            if c.id not in selected_ids
        ]

        system_prompt = self.SYSTEM_PROMPT
        system_tokens = estimate_tokens(system_prompt)
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in history)
        evidence_tokens = sum(e.tokens for e in selected)
        budget = Budget(
            system=system_tokens,
            history=history_tokens,
            evidence=evidence_tokens,
            used=system_tokens + history_tokens + evidence_tokens,
            limit=self.limit,
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    "请仅根据下面提供的资料回答问题，并在答案中标注引用的编号（如 [1]）。"
                    "如果资料不足以回答，请明确说明不足。\n\n"
                    f"资料：\n{render_evidence(selected)}"
                ),
            }
        )

        return BuildResult(
            # 版本号带 +rebuild 后缀：评测时能一眼看出这条快照走过重建路径
            strategy_version=f"{self.strategy.version}+rebuild",
            messages=messages,
            selected=selected,
            dropped=dropped,
            budget=budget,
        )

    # ---------------- 按裁决结果剔除证据 ----------------
    def _drop_evidence(self, build: BuildResult, drop_ids: List[str]) -> BuildResult:
        """把校验器裁决出的低权威证据**真正移出上下文**。

        为什么必须真的剔除，而不是只在快照里记一笔：
            只标记等于把"该信谁"这个判断推给生成模型 ——
            而模型同时看到两条互相矛盾的说法时，恰恰是最容易被带偏的时候。
            校验器既然已经用权威度分出了高下，就该把噪声清掉。

        被剔除的证据以 VALIDATOR_REJECT 记进 dropped ——
        Schema 里早就留了这个原因枚举，归因器能直接统计"有多少条是被裁决掉的"。
        """
        drop_set = set(drop_ids)
        by_id = {e.id: e for e in build.selected}
        keep = [e for e in build.selected if e.id not in drop_set]

        # 重新编号：position 是分析 Lost in the Middle 的依据，剔除后必须重排，
        # 否则位置字段和实际顺序脱节，后续的位置分析全部失真
        keep = [e.model_copy(update={"position": i}) for i, e in enumerate(keep)]

        newly_dropped = [
            Dropped(id=i, score=by_id[i].score, reason=DropReason.VALIDATOR_REJECT)
            for i in drop_ids
            if i in by_id
        ]

        # 重建 messages：证据变了，渲染出来的编号必须跟着变，
        # 否则答案里的 [3] 会指向一条已经不在上下文里的证据
        messages = list(build.messages)
        if messages and messages[-1].get("role") == "user":
            messages[-1] = {
                "role": "user",
                "content": (
                    "请仅根据下面提供的资料回答问题，并在答案中标注引用的编号（如 [1]）。"
                    "如果资料不足以回答，请明确说明不足。\n\n"
                    f"资料：\n{render_evidence(keep)}"
                ),
            }

        tokens = sum(e.tokens for e in keep)
        budget = Budget(
            system=build.budget.system,
            history=build.budget.history,
            evidence=tokens,
            used=build.budget.system + build.budget.history + tokens,
            limit=build.budget.limit,
        )

        return BuildResult(
            strategy_version=f"{self.strategy.version}+resolved",
            messages=messages,
            selected=keep,
            dropped=list(build.dropped) + newly_dropped,
            budget=budget,
            validator=build.validator,
        )

    # ---------------- 引用检测 ----------------
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
