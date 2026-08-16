"""Skill-grounded letter writing with optional knowledge retrieval."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import Settings, get_settings
from src.dialogue.knowledge import DialogueKnowledgeBase
from src.dialogue.interactions import build_regulation_offer
from src.dialogue.llm import DeepSeekChatClient, JSONCompletion, JSONLLM, LLMError
from src.dialogue.models import (
    DialogueResult,
    DialogueTrace,
    DialogueDecision,
    KnowledgePassage,
    ReplyEnvelope,
    RiskFlag,
)
from src.dialogue.prompts import (
    build_dialogue_decision_prompts,
    build_reply_prompts,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class DialogueService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: JSONLLM | None = None,
        knowledge_base: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or DeepSeekChatClient(self.settings)
        self.knowledge_base = knowledge_base or DialogueKnowledgeBase(self.settings)

    def respond(
        self,
        letter: str,
        *,
        preferred_school: str | None = None,
        include_trace: bool = False,
    ) -> DialogueResult:
        local_risk = detect_explicit_risk(letter)
        router_system, router_user = build_dialogue_decision_prompts(
            letter,
            preferred_school=preferred_school,
            local_risk=local_risk,
        )
        decision, router_completions = self._complete_validated(
            model_type=DialogueDecision,
            system_prompt=router_system,
            user_prompt=router_user,
            max_tokens=900,
        )
        decision.risk_flag = local_risk

        grounding_knowledge: list[KnowledgePassage] = []
        explanatory_knowledge: list[KnowledgePassage] = []
        skipped_reason: str | None = None
        if decision.risk_flag != "none":
            decision.grounding_queries = []
            decision.explanatory_queries = []
            skipped_reason = "Safety routing paused psychodynamic knowledge retrieval."
        else:
            grounding_knowledge = self._retrieve(
                decision.grounding_queries,
                preferred_school=preferred_school,
            )
            if decision.needs_knowledge_explanation:
                explanatory_knowledge = self._retrieve(
                    decision.explanatory_queries,
                    preferred_school=preferred_school,
                    excluded_texts={item.text for item in grounding_knowledge},
                )
            elif not grounding_knowledge:
                skipped_reason = "Grounding retrieval returned no relevant passages."

        interaction = (
            None
            if decision.risk_flag != "none"
            else build_regulation_offer(decision)
        )
        reply_system, reply_user = build_reply_prompts(
            letter,
            decision,
            grounding_knowledge,
            explanatory_knowledge,
            interaction,
        )
        envelope, reply_completions = self._complete_validated(
            model_type=ReplyEnvelope,
            system_prompt=reply_system,
            user_prompt=reply_user,
            max_tokens=2200,
        )
        envelope.risk_flag = decision.risk_flag
        envelope.grounding_used = bool(grounding_knowledge) and envelope.risk_flag == "none"
        envelope.explanatory_knowledge_used = (
            bool(explanatory_knowledge) and envelope.risk_flag == "none"
        )
        if interaction and interaction.message not in envelope.reply:
            envelope.reply = f"{envelope.reply.rstrip()}\n\n{interaction.message}"

        trace = None
        if include_trace:
            trace = DialogueTrace(
                decision=decision,
                grounding_knowledge=grounding_knowledge,
                explanatory_knowledge=explanatory_knowledge,
                retrieval_skipped_reason=skipped_reason,
                model=reply_completions[-1].model,
                usage=[
                    completion.usage
                    for completion in [*router_completions, *reply_completions]
                ],
            )
        return DialogueResult(reply=envelope.reply, interaction=interaction, trace=trace)

    def _complete_validated(
        self,
        *,
        model_type: type[ModelT],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> tuple[ModelT, list[JSONCompletion]]:
        completions = [
            self.llm.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        ]
        try:
            return model_type.model_validate(completions[0].content), completions
        except ValidationError as first_error:
            repair_system = (
                f"{system_prompt}\n\n"
                "输出修复：上一次返回没有通过 JSON 或安全边界校验。必须输出 JSON 数据实例，"
                "遵守 validation_error 中的安全限制，不要输出 JSON Schema，"
                "不要出现 properties、title 或 type 作为顶层结构。"
            )
            repair_user = (
                f"{user_prompt}\n\n"
                f"<invalid_output>{json.dumps(completions[0].content, ensure_ascii=False)}</invalid_output>\n"
                f"<validation_error>{first_error}</validation_error>\n"
                "请重新输出完整且合法的 JSON 数据实例。"
            )
            completions.append(
                self.llm.complete_json(
                    system_prompt=repair_system,
                    user_prompt=repair_user,
                    max_tokens=max_tokens,
                )
            )
            try:
                return model_type.model_validate(completions[-1].content), completions
            except ValidationError as second_error:
                raise LLMError(
                    f"Invalid {model_type.__name__} after one format repair: {second_error}"
                ) from second_error

    def _retrieve(
        self,
        queries: list[Any],
        *,
        preferred_school: str | None,
        excluded_texts: set[str] | None = None,
    ) -> list[KnowledgePassage]:
        collected: list[KnowledgePassage] = []
        seen: set[str] = set(excluded_texts or set())
        for item in queries[:2]:
            school = preferred_school or item.school
            for passage in self.knowledge_base.query(item.query, top_n=3, school=school):
                if passage.text in seen:
                    continue
                seen.add(passage.text)
                collected.append(passage)
        return collected[:6]


def detect_explicit_risk(letter: str) -> RiskFlag:
    immediate_patterns = (
        r"(?:已经|正在)(?:实施|进行)?自杀",
        r"(?:马上|现在就)(?:要|去)?自杀",
        r"(?:已经|正在)(?:自残|割腕|伤害自己)",
        r"(?:已经吞|正在吞)(?:药|毒)",
        r"(?:准备|打算|计划)(?:马上|现在|今天|今晚)?(?:跳楼|割腕|服毒)",
    )
    self_harm_patterns = (
        r"(?<!不)(?:我)?(?:想|要|准备|打算|计划)(?:去)?自杀",
        r"(?<!不)(?:我)?(?:想|要|准备|打算|计划)(?:结束|了结)自己的生命",
        r"(?<!不)(?:我)?(?:想|要|准备|打算|计划)(?:伤害自己|自残|割腕)",
        r"(?<!不)(?:我)?(?:要|准备|打算)去死",
    )
    violence_patterns = (
        r"(?<!不)(?:我)?(?:想|要|准备|打算|计划)(?:杀人|杀了他|杀了她|伤害别人)",
        r"(?:正在|马上)(?:杀人|伤害别人)",
    )
    if any(re.search(pattern, letter) for pattern in immediate_patterns):
        return "immediate_danger"
    if any(re.search(pattern, letter) for pattern in violence_patterns):
        return "violence"
    if any(re.search(pattern, letter) for pattern in self_harm_patterns):
        return "self_harm"
    return "none"
