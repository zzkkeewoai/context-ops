"""上下文组装器 —— 本项目的核心模块

设计要点：
    1. 策略可插拔（同一接口，多个实现）→ 支持 A/B 对比，这是"可优化"的前提
    2. 每次裁剪都记录 reason → 这是归因器能工作的唯一依据
    3. 组装与优化分离：Builder 负责"收集 + 取舍"，不做模型调用

两个策略的差异（面试就讲这个对比）：
    NaiveTopK          baseline：按分数取前 k 条，不管顺序、不管预算、不去重
    HeadTailPriority   优化版  ：去重 + 预算控制 + 首尾优先排列（规避 Lost in the Middle）
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from schemas import (
    Budget,
    BuildResult,
    Candidate,
    DropReason,
    Dropped,
    Evidence,
    ValidatorResult,
)

# ============================================================
# Token 估算
# ============================================================
def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    中文约 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token。
    这里用保守混合估算；生产环境应换对应模型的 tokenizer（如 tiktoken）。
    估算不准会导致预算超限，所以宁可略高估。
    """
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    others = len(text) - chinese
    return chinese + max(1, others // 3)


def render_evidence(evidence: List[Evidence]) -> str:
    """把证据渲染成带编号的上下文块。

    编号 [1] [2] 很重要——它是「引用检测」的基础：
    只有带编号，才能从答案里识别出"模型引用了哪几条证据"，
    进而算出「上下文利用率」这个核心指标。
    """
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] {e.text}")
    return "\n".join(lines)


# ============================================================
# 策略接口
# ============================================================
class ContextStrategy(ABC):
    """上下文组装策略接口"""

    version: str = "base"

    @abstractmethod
    def build(
        self,
        query: str,
        candidates: List[Candidate],
        system_prompt: str,
        history: List[Dict[str, str]],
        limit: int,
    ) -> BuildResult:
        ...

    # ---------- 公共工具 ----------
    def _pack_messages(
        self, system_prompt: str, history: List[Dict[str, str]], evidence: List[Evidence]
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    "请仅根据下面提供的资料回答问题，并在答案中标注引用的编号（如 [1]）。"
                    "如果资料不足以回答，请明确说明不足。\n\n"
                    f"资料：\n{render_evidence(evidence)}"
                ),
            }
        )
        return messages

    @staticmethod
    def _dedup(candidates: List[Candidate]) -> tuple[List[Candidate], List[Dropped]]:
        """按文本前缀去重（简化实现）。

        生产环境应做语义去重（embedding 相似度），但前缀去重已能覆盖
        多路召回带来的大量字面重复。
        """
        kept: List[Candidate] = []
        dropped: List[Dropped] = []
        seen: set[str] = set()
        for c in candidates:
            key = c.text[:30].strip()
            if key in seen:
                dropped.append(Dropped(id=c.id, score=c.score, reason=DropReason.DUPLICATE))
                continue
            seen.add(key)
            kept.append(c)
        return kept, dropped


# ============================================================
# 策略 1：NaiveTopK（baseline）
# ============================================================
class NaiveTopK(ContextStrategy):
    """最朴素的实现：按分数取前 k 条。

    故意保留它的"缺点"（不排序优化、不管预算、不去重），
    这样它才能作为一个公平的 baseline 被对比。
    """

    version = "ctx_naive_topk_v1"

    def __init__(self, k: int = 5):
        self.k = k

    def build(self, query, candidates, system_prompt, history, limit) -> BuildResult:
        """【真实 naive 实现】直接取检索返回的前 k 条，**不做任何重排**。

        这里踩过一个关键的实验设计坑：第一版写了 `sorted(candidates, key=score)`
        再取前 k —— 结果 naive 白捡了一个按相关性排好的顺序，
        和优化策略跑出来的指标一模一样，A/B 对比完全失效。

        真实的朴素实现就是 `candidates[:k]`，而它的缺陷恰恰在于：
        检索返回顺序受多路召回融合影响（不等于相关性顺序），
        最相关的证据可能落在上下文中间而被模型忽略。
        """
        selected_raw = candidates[: self.k]
        rest = candidates[self.k:]

        selected = [
            Evidence(
                id=c.id,
                text=c.text,
                score=c.score,
                position=i,
                tokens=estimate_tokens(c.text),
                source=c.source,
                authority=c.authority,
            )
            for i, c in enumerate(selected_raw)
        ]
        dropped = [
            Dropped(id=c.id, score=c.score, reason=DropReason.LOW_SCORE) for c in rest
        ]

        evidence_tokens = sum(e.tokens for e in selected)
        system_tokens = estimate_tokens(system_prompt)
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in history)
        budget = Budget(
            system=system_tokens,
            history=history_tokens,
            evidence=evidence_tokens,
            used=system_tokens + history_tokens + evidence_tokens,
            limit=limit,
        )

        return BuildResult(
            strategy_version=self.version,
            messages=self._pack_messages(system_prompt, history, selected),
            selected=selected,
            dropped=dropped,
            budget=budget,
        )


# ============================================================
# 策略 2：HeadTailPriority（优化版）
# ============================================================
class HeadTailPriority(ContextStrategy):
    """优化版：去重 + 预算控制 + 首尾优先排列。

    三项优化各自解决一个具体问题：
      · 去重           → 多路召回必然重复，不去重等于浪费上下文窗口
      · 预算控制       → 证据不能无限塞，超了要按分数裁剪并记录原因
      · 首尾优先排序   → 模型对上下文中间位置的注意力最弱（Lost in the Middle），
                        最相关的证据要放开头和结尾
    """

    version = "ctx_headtail_v1"

    def __init__(self, k: int = 6, evidence_ratio: float = 0.6):
        self.k = k
        # 证据能占用总预算的比例（其余留给 system + history）
        self.evidence_ratio = evidence_ratio

    def build(self, query, candidates, system_prompt, history, limit) -> BuildResult:
        system_tokens = estimate_tokens(system_prompt)
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in history)

        # 证据可用预算 = 总预算 - 系统 - 历史，再乘以证据占比上限
        available = max(0, limit - system_tokens - history_tokens)
        evidence_budget = int(available * self.evidence_ratio)

        # ① 去重
        deduped, dropped = self._dedup(candidates)

        # ② 按分数排序后，按预算取用
        ranked = sorted(deduped, key=lambda c: c.score, reverse=True)
        selected: List[Evidence] = []
        used_tokens = 0
        for c in ranked:
            if len(selected) >= self.k:
                dropped.append(Dropped(id=c.id, score=c.score, reason=DropReason.LOW_SCORE))
                continue
            t = estimate_tokens(c.text)
            if used_tokens + t > evidence_budget:
                # 超预算 → 明确记录原因（归因器会统计这个比例）
                dropped.append(
                    Dropped(id=c.id, score=c.score, reason=DropReason.BUDGET_EXCEEDED)
                )
                continue
            selected.append(
                Evidence(
                    id=c.id,
                    text=c.text,
                    score=c.score,
                    position=-1,
                    tokens=t,
                    source=c.source,
                    authority=c.authority,
                )
            )
            used_tokens += t

        # ③ 首尾优先排列：最高分放开头，次高放结尾，其余填中间
        selected = self._head_tail_order(selected)

        budget = Budget(
            system=system_tokens,
            history=history_tokens,
            evidence=used_tokens,
            used=system_tokens + history_tokens + used_tokens,
            limit=limit,
        )
        return BuildResult(
            strategy_version=self.version,
            messages=self._pack_messages(system_prompt, history, selected),
            selected=selected,
            dropped=dropped,
            budget=budget,
        )

    @staticmethod
    def _head_tail_order(items: List[Evidence]) -> List[Evidence]:
        """首尾优先排列：把最相关的放在开头和结尾。

        假设 items 已按分数降序，则填充顺序为：
            开头 ← items[0]，结尾 ← items[1]，开头+1 ← items[2]，结尾-1 ← items[3] ...
        """
        n = len(items)
        if n <= 2:
            return [it.model_copy(update={"position": i}) for i, it in enumerate(items)]

        slots: List[Optional[Evidence]] = [None] * n
        lo, hi = 0, n - 1
        for i, item in enumerate(items):
            if i % 2 == 0:
                slots[lo] = item
                lo += 1
            else:
                slots[hi] = item
                hi -= 1
        ordered = [s for s in slots if s is not None]
        return [it.model_copy(update={"position": i}) for i, it in enumerate(ordered)]


# ============================================================
# 策略注册表（便于 A/B 与配置化）
# ============================================================
STRATEGIES = {
    "naive": NaiveTopK,
    "headtail": HeadTailPriority,
}


def get_strategy(name: str, **kwargs) -> ContextStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知策略: {name}，可选 {list(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)
