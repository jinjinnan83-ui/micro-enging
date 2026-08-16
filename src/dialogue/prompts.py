"""Prompt builders using SKILL.md as the authoritative writing prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.dialogue.models import (
    DialogueDecision,
    KnowledgePassage,
    RegulationOffer,
    ReplyEnvelope,
    RiskFlag,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = PROJECT_ROOT / ".cursor" / "skills" / "psychoanalysis-dialogue"


@dataclass(frozen=True)
class SkillBundle:
    skill: str
    reference: str


@lru_cache(maxsize=1)
def load_skill_bundle() -> SkillBundle:
    return SkillBundle(
        skill=(SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"),
        reference=(SKILL_ROOT / "reference.md").read_text(encoding="utf-8"),
    )


def build_dialogue_decision_prompts(
    letter: str,
    *,
    preferred_school: str | None,
    local_risk: RiskFlag,
) -> tuple[str, str]:
    schema = json.dumps(DialogueDecision.model_json_schema(), ensure_ascii=False)
    system = f"""你只负责两层知识检索路由和安全预检，不分析人格，不写回信。

所有非风险来信都要生成一到两条 grounding_queries：一条贴近来信的具体关系或情绪场景，
一条用合适的精神动力学概念改写。它们用于约束回信理解，不代表要给用户上课。

当来信明确询问为什么、原因、机制、理论、知识点，或者虽未直说但明显在寻找一种可理解自身经验的解释时，
needs_knowledge_explanation=true，并额外生成一到两条 explanatory_queries，寻找适合解释、安慰、减少自责或澄清困境的知识。
否则 explanatory_queries 必须为空。

当来信表示很累、撑不动、脑子装不下了、想停一下、不想继续说或暂时不想回应时，
needs_regulation_interaction=true。这一字段只表示识别到对方此刻可能不适合继续说话，不选择或指导任何训练。
普通的“最近不太想回微信”不自动触发，除非当下语境明确表示疲惫或想暂停这次交流。
同时识别自伤、伤人和即时危险。只有来信明确表达当前自杀/自残意图、具体计划、正在实施或近期已经实施时，
risk_flag 才能是 self_harm 或 immediate_danger。仅有“不想死也不想活”“没有未来”“希望生病”“没有生命欲望”
等被动死亡、空虚或绝望表达时，risk_flag 必须是 none，继续普通检索和回信。
只输出 JSON 数据实例，不要输出 schema 或 markdown。
所有字符串使用中文。school 只能是：精神分析、拉康派、客体关系、分析心理学、自体心理学、关系精神分析、个体心理学、心智化，或 null。
JSON 必须符合：{schema}
"""
    user = f"""请判断下面的来信是否明确索要知识。
本地风险预检：{local_risk}
偏好流派：{preferred_school or '未指定'}

<letter>
{letter}
</letter>
"""
    return system, user


def build_reply_prompts(
    letter: str,
    decision: DialogueDecision,
    grounding_knowledge: list[KnowledgePassage],
    explanatory_knowledge: list[KnowledgePassage],
    interaction: RegulationOffer | None,
) -> tuple[str, str]:
    bundle = load_skill_bundle()
    schema = json.dumps(ReplyEnvelope.model_json_schema(), ensure_ascii=False)
    grounding_context = _format_knowledge(grounding_knowledge)
    explanatory_context = _format_knowledge(explanatory_knowledge)
    if decision.risk_flag != "none":
        runtime_instruction = (
            "当前处于安全模式。严格执行 Skill 的风险边界，只写安全支持回复。"
            "禁止解释原因或机制，禁止出现无意识、压抑、阻抗、移情、妥协形成、客体、驱力、旧场面等动力学解释，"
            "禁止回答来信中的知识问题。直接承认风险，询问或强调当下安全，给出现实支持资源。"
        )
    elif explanatory_knowledge:
        runtime_instruction = (
            "先用 grounding_context 约束对来信的精神动力学理解，再使用 explanatory_context 中最相关的一到两个观点帮助理解或安慰，"
            "把知识自然写进同一封回信、保持 Skill 的同一文风；不要写成百科、课程或文献列表，"
            "不要粘贴原文，不虚构来源。优先使用能减少自责、容纳感受或澄清关系困境的内容；"
            "不要为了显得深刻而推断来信没有提供的童年经历或创伤。知识片段中的儿童发展或早期关系理论"
            "只是一般机制，不是来信者确实经历过这些事情的证据；只能解释为一种可能，不能写成其个人史事实。"
        )
    else:
        runtime_instruction = (
            "使用 grounding_context 约束对来信的精神动力学理解，但不要显性讲课或罗列理论；"
            "只按照 Skill 的理解方式和文风写回信。"
        )
    if interaction:
        interaction_instruction = (
            "本轮识别到对方可能很累或不想继续说。不要提供任何呼吸方法、冥想步骤、秒数、轮次或训练指导，"
            "也不要自行写呼吸或冥想邀请；后端会在回信末尾追加固定提示语。"
        )
    else:
        interaction_instruction = "本轮没有暂停交流信号，不要凭空提供呼吸或冥想练习。"

    system = f"""{bundle.skill}

# Runtime contract

上面的 SKILL.md 是这封回信的核心 prompt，决定精神分析动力学理解、边界和文风。
{runtime_instruction}
{interaction_instruction}
最终任务只有一个：写一封直接给来信者看的完整中文回信。
ReplyEnvelope.risk_flag 必须与 dialogue_decision.risk_flag 完全一致，不得自行提高风险等级。
只输出 JSON 数据实例，不要输出分析过程、schema 或 markdown。
JSON 必须符合：{schema}

<risk_reference>
{bundle.reference}
</risk_reference>
"""
    user = f"""请按照 SKILL.md 回复这封信。

<dialogue_decision>
{decision.model_dump_json()}
</dialogue_decision>

<grounding_context>
{grounding_context}
</grounding_context>

<explanatory_context>
{explanatory_context}
</explanatory_context>

<regulation_offer>
{interaction.model_dump_json() if interaction else 'null'}
</regulation_offer>

<letter>
{letter}
</letter>
"""
    return system, user


def _format_knowledge(passages: list[KnowledgePassage]) -> str:
    if not passages:
        return "No knowledge passages were provided."
    blocks = []
    for index, passage in enumerate(passages, start=1):
        metadata = {
            "author": passage.author,
            "school": passage.school,
            "concepts": passage.core_concepts,
            "source_document": passage.source_document,
        }
        blocks.append(
            f"[K{index}] metadata={json.dumps(metadata, ensure_ascii=False)}\n{passage.text}"
        )
    return "\n\n".join(blocks)
