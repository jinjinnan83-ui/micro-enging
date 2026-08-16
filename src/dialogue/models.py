"""Runtime contracts for knowledge routing and skill-grounded letter replies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RiskFlag = Literal["none", "self_harm", "violence", "immediate_danger"]
School = Literal[
    "精神分析",
    "拉康派",
    "客体关系",
    "分析心理学",
    "自体心理学",
    "关系精神分析",
    "个体心理学",
    "心智化",
]
SCHOOL_ALIASES = {
    "freud": "精神分析",
    "psychoanalysis": "精神分析",
    "lacan": "拉康派",
    "lacanian": "拉康派",
    "object_relations": "客体关系",
    "object relations": "客体关系",
    "jung": "分析心理学",
    "analytical_psychology": "分析心理学",
    "self_psychology": "自体心理学",
    "relational_psychoanalysis": "关系精神分析",
    "individual_psychology": "个体心理学",
    "mentalization": "心智化",
}
SAFETY_FORBIDDEN_ANALYSIS = (
    "精神分析里",
    "无意识",
    "压抑",
    "阻抗",
    "移情",
    "妥协形成",
    "客体",
    "驱力",
    "旧场面",
)


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=2, description="Chinese knowledge-base query")
    school: School | None = Field(None, description="Optional psychoanalytic school filter")
    rationale: str = Field(..., description="Short reason this query serves the letter")

    @field_validator("school", mode="before")
    @classmethod
    def normalize_school(cls, value: Any) -> Any:
        if isinstance(value, str):
            return SCHOOL_ALIASES.get(value.strip().lower(), value)
        return value


class DialogueDecision(BaseModel):
    risk_flag: RiskFlag = "none"
    risk_reason: str = ""
    needs_knowledge_explanation: bool
    reason: str = Field(..., description="Why explanatory knowledge is or is not useful")
    grounding_queries: list[SearchQuery] = Field(default_factory=list, max_length=2)
    explanatory_queries: list[SearchQuery] = Field(default_factory=list, max_length=2)
    needs_regulation_interaction: bool = False
    regulation_reason: str = ""

    @model_validator(mode="after")
    def routes_match_decision(self) -> "DialogueDecision":
        if not self.needs_knowledge_explanation:
            self.explanatory_queries = []
        return self


class KnowledgePassage(BaseModel):
    query: str
    text: str
    score: float
    source: str = "bm25"
    author: str | None = None
    school: str | None = None
    core_concepts: list[str] = Field(default_factory=list)
    source_document: str | None = None


class ReplyEnvelope(BaseModel):
    risk_flag: RiskFlag = "none"
    grounding_used: bool = False
    explanatory_knowledge_used: bool = False
    reply: str = Field(..., min_length=1, description="Complete Chinese letter reply")

    @model_validator(mode="after")
    def safety_reply_stops_analysis(self) -> "ReplyEnvelope":
        if self.risk_flag != "none" and any(
            term in self.reply for term in SAFETY_FORBIDDEN_ANALYSIS
        ):
            raise ValueError("Safety replies must not contain psychodynamic interpretation")
        return self


class RegulationOffer(BaseModel):
    type: Literal["breathing_or_meditation_offer"] = "breathing_or_meditation_offer"
    message: str = "不说话也没关系，也许呼吸或者冥想更适合现在的状态？"


class DialogueTrace(BaseModel):
    decision: DialogueDecision
    grounding_knowledge: list[KnowledgePassage] = Field(default_factory=list)
    explanatory_knowledge: list[KnowledgePassage] = Field(default_factory=list)
    retrieval_skipped_reason: str | None = None
    model: str
    usage: list[dict[str, Any]] = Field(default_factory=list)


class DialogueResult(BaseModel):
    reply: str
    interaction: RegulationOffer | None = None
    trace: DialogueTrace | None = None
