"""Freud depth analyst: Pydantic models for reply planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PsychodynamicMove = Literal[
    "manifest_latent",
    "repression_resistance",
    "compromise_repetition",
    "transference_position",
]


class FreudReplyPlan(BaseModel):
    user_utterance: str
    manifest_detail: str = Field(..., description="显意里被切开的具体细节")
    latent_structure: str = Field(..., description="被挡住、仍在运作的隐意结构")
    defense_or_resistance: str
    psychodynamic_move: PsychodynamicMove
    concept: str = Field(..., description="本轮使用的知识库概念")
    school: str = "精神分析"
    source_author: str | None = None
    risk_flag: Literal["none", "self_harm", "violence", "immediate_danger"] = "none"
    reply: str


class FreudReplyInput(BaseModel):
    user_utterance: str = Field(..., description="用户原话")
    locale: str = Field(default="zh-CN")


TOOL_DEFINITION = {
    "name": "freud_depth_analyst",
    "description": (
        "用弗洛伊德精神动力学回复谈及梦、欲望、焦虑、内疚、移情、关系重复或症状的用户输入。"
        "人设为冷峻深度分析师：极简、无攻击、无说教；先切开显意，再揭示被隐瞒的潜意识结构。"
    ),
    "input_schema": FreudReplyInput.model_json_schema(),
    "output_schema": FreudReplyPlan.model_json_schema(),
}
