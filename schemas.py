"""上下文快照 Schema —— 整个项目的地基

为什么这个文件必须先定下来：
    归因、评测、优化全都依赖快照里的字段。Schema 设计错了，后面必须返工重跑数据。

三个「护城河字段」（★ 标记）——没有它们，整个闭环就断了：
    ★ dropped[].reason          没有它，无法回答"好证据为什么没进上下文"
    ★ answer.cited_evidence     没有它，算不出「上下文利用率」
    ★ context.strategy_version  没有它，评测结果无法归因到具体策略（不同策略无法对比）
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# 枚举：裁剪原因（归因器的核心输入）
# ============================================================
class DropReason(str, Enum):
    """一条证据为什么没被放进上下文 —— 每种原因对应不同的修法"""

    LOW_SCORE = "low_score"                # 相关性不够 → 该改召回/重排
    BUDGET_EXCEEDED = "budget_exceeded"    # 超 token 预算 → 该改预算分配或压缩
    DUPLICATE = "duplicate"                # 与已入选重复 → 该改去重逻辑
    VALIDATOR_REJECT = "validator_reject"  # 上下文校验判定不合格 → 该改校验规则
    IRRELEVANT = "irrelevant"              # 明确判定与问题无关


# ============================================================
# 检索层
# ============================================================
class Candidate(BaseModel):
    """一条检索候选（尚未经过上下文组装过滤）"""

    id: str
    text: str
    score: float = Field(description="相关性分数（不同来源量纲可能不同）")
    source: str = Field(default="vector", description="vector / keyword / graph")


# ============================================================
# 上下文层
# ============================================================
class Evidence(BaseModel):
    """最终进入上下文的一条证据"""

    id: str
    text: str
    score: float
    position: int = Field(description="在上下文中的位置（0 起）—— 用于分析 Lost in the Middle")
    tokens: int
    source: str = "vector"


class Dropped(BaseModel):
    """被裁掉的证据 + 原因 ★"""

    id: str
    score: float
    reason: DropReason


class Budget(BaseModel):
    """Token 预算的分配与使用（零和博弈：给证据多分，历史就少分）"""

    system: int = 0
    history: int = 0
    evidence: int = 0
    used: int = 0
    limit: int = 4096

    @property
    def usage_ratio(self) -> float:
        return round(self.used / self.limit, 3) if self.limit else 0.0


class ValidatorResult(BaseModel):
    """上下文校验结果（PASS / FAIL 分支的依据）"""

    status: str = Field(default="pass", description="pass / fail")
    checks: Dict[str, float] = Field(default_factory=dict, description="各维度得分")
    action: Optional[str] = Field(default=None, description="fail 时：rebuild / refuse")
    detail: Optional[str] = None


class BuildResult(BaseModel):
    """上下文组装结果（Context Builder 的输出）"""

    strategy_version: str  # ★ 策略版本，用于 A/B 对比与归因
    messages: List[Dict[str, str]]
    selected: List[Evidence]
    dropped: List[Dropped]
    budget: Budget
    validator: Optional[ValidatorResult] = None

    @property
    def selected_ids(self) -> List[str]:
        return [e.id for e in self.selected]


# ============================================================
# 生成层
# ============================================================
class GenerationInfo(BaseModel):
    model: str
    prompt_hash: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    degraded: bool = False
    provider: Optional[str] = None


class AnswerInfo(BaseModel):
    text: str
    cited_evidence: List[str] = Field(default_factory=list)  # ★ 引用检测结果
    refused: bool = False


# ============================================================
# 顶层：一次请求的完整快照
# ============================================================
class ContextSnapshot(BaseModel):
    """一次请求的完整快照 —— 可回放、可评测、可归因"""

    request_id: str
    session_id: str = "default"
    tenant_id: str = "default"  # 多租户：先留位置，实现可后补

    # ① 查询理解
    query_original: str
    query_rewritten: Optional[str] = None
    rewrite_method: str = "none"

    # ② 检索
    candidates: List[Candidate] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0

    # ③ 上下文（组装 + 校验）
    context: BuildResult

    # ④ 生成
    generation: GenerationInfo

    # ⑤ 答案
    answer: AnswerInfo

    # ⑥ 系统层
    total_latency_ms: float = 0.0
    cost_usd: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # ---------------- 派生指标（评测与归因直接读这些） ----------------
    @property
    def context_utilization(self) -> float:
        """上下文利用率 = 入选证据中被答案引用的比例。

        这是本项目的「独有指标」：它直接反映上下文质量。
        低利用率说明：放进去了但模型没用它（可能是位置问题/信噪比问题/根本无关）。
        """
        if not self.context.selected:
            return 0.0
        cited = set(self.answer.cited_evidence)
        hit = sum(1 for e in self.context.selected if e.id in cited)
        return round(hit / len(self.context.selected), 3)

    @property
    def dropped_by_reason(self) -> Dict[str, int]:
        """按原因统计被裁掉的证据数 —— 归因分布的数据源"""
        out: Dict[str, int] = {}
        for d in self.context.dropped:
            key = d.reason.value if isinstance(d.reason, DropReason) else str(d.reason)
            out[key] = out.get(key, 0) + 1
        return out
