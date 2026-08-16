"""Bandura self-efficacy coach: Pydantic models for reply planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VicariousModel(BaseModel):
    similar_other: str = Field(..., description="起点相似的他人，禁止遥远成功名人")
    first_steps: str = Field(..., description="前一两步怎么走")
    why_similar: str = Field(..., description="与用户在任务、难度或卡点上的相似处")


class MicroStep(BaseModel):
    action: str = Field(..., description="一个极其微小、可独立完成的行动")
    duration_minutes: int = Field(..., ge=2, le=20)
    done_when: str = Field(..., description="可观察的完成标准")
    mastery_signal: str = Field(..., description="完成后可对自己说的‘我做到了’证据")


class BanduraReplyPlan(BaseModel):
    user_utterance: str
    task: str
    low_efficacy_belief: str
    unhelpful_attribution: str
    controllable_attribution: str
    efficacy_reframe: str
    vicarious_model: VicariousModel
    micro_step: MicroStep
    arousal_note: str | None = None
    risk_flag: Literal["none", "self_harm", "violence", "immediate_danger"] = "none"
    reply: str


class BanduraReplyInput(BaseModel):
    user_utterance: str = Field(..., description="用户原话")
    locale: str = Field(default="zh-CN")


TOOL_DEFINITION = {
    "name": "bandura_self_efficacy_coach",
    "description": (
        "用班杜拉自我效能教练策略回复带有困惑、焦虑、拖延或未来不确定感的用户输入。"
        "先评估低效能信念与消极归因，再输出含微步骤与相似榜样的掌控感回复。"
    ),
    "input_schema": BanduraReplyInput.model_json_schema(),
    "output_schema": BanduraReplyPlan.model_json_schema(),
}
