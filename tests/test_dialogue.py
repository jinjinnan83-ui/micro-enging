from __future__ import annotations

import json

from src.config import Settings
from src.dialogue.knowledge import DialogueKnowledgeBase
from src.dialogue.llm import JSONCompletion
from src.dialogue.models import KnowledgePassage, SearchQuery
from src.dialogue.service import DialogueService, detect_explicit_risk


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str, max_tokens: int):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        return JSONCompletion(
            content=self.responses.pop(0),
            model="deepseek-v4-flash",
            usage={"total_tokens": 100},
        )


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int, str | None]] = []

    def query(self, query: str, *, top_n: int, school: str | None):
        self.queries.append((query, top_n, school))
        return [
            KnowledgePassage(
                query=query,
                text=f"{query}：关系中的重复可以同时容纳靠近的愿望与被抛弃的预期。",
                score=4.2,
                author="Sigmund Freud",
                school="精神分析",
                core_concepts=["重复", "移情"],
                source_document="freud_technique_transference.md",
            )
        ]


def _decision(
    *,
    needs_explanation: bool,
    risk_flag: str = "none",
    needs_regulation: bool = False,
) -> dict:
    return {
        "risk_flag": risk_flag,
        "risk_reason": "",
        "needs_knowledge_explanation": needs_explanation,
        "reason": "来信需要知识解释" if needs_explanation else "来信只希望收到回信",
        "grounding_queries": [
            {"query": "害怕朋友离开 不敢表达需要", "school": None, "rationale": "贴近来信"},
            {"query": "移情 重复 依赖", "school": "精神分析", "rationale": "动力学改写"},
        ],
        "explanatory_queries": (
            [
                {"query": "分离焦虑 如何理解", "school": None, "rationale": "解释困境"},
                {"query": "依赖需求 减少自责", "school": None, "rationale": "寻找安慰性知识"},
            ]
            if needs_explanation
            else []
        ),
        "needs_regulation_interaction": needs_regulation,
        "regulation_reason": "来信表示很累并想暂停" if needs_regulation else "",
    }


def _reply(
    text: str,
    *,
    risk_flag: str = "none",
    grounding_used: bool = True,
    explanatory_used: bool = False,
) -> dict:
    return {
        "risk_flag": risk_flag,
        "grounding_used": grounding_used,
        "explanatory_knowledge_used": explanatory_used,
        "reply": text,
    }


def test_ordinary_letter_uses_skill_without_knowledge_retrieval() -> None:
    llm = FakeLLM(
        [
            _decision(needs_explanation=False),
            _reply("你把需要挡在门外，又等对方发现它。"),
        ]
    )
    knowledge = FakeKnowledgeBase()
    service = DialogueService(Settings(), llm=llm, knowledge_base=knowledge)

    result = service.respond("我不敢告诉朋友我需要她。", include_trace=True)

    assert result.reply == "你把需要挡在门外，又等对方发现它。"
    assert [query[0] for query in knowledge.queries] == [
        "害怕朋友离开 不敢表达需要",
        "移情 重复 依赖",
    ]
    assert len(llm.calls) == 2
    assert "psychoanalysis-dialogue" in llm.calls[1]["system_prompt"]
    assert "不要显性讲课" in llm.calls[1]["system_prompt"]
    assert "害怕朋友离开 不敢表达需要" in llm.calls[1]["user_prompt"]
    assert result.trace is not None
    assert result.trace.decision.needs_knowledge_explanation is False
    assert result.trace.grounding_knowledge
    assert result.trace.explanatory_knowledge == []


def test_explicit_knowledge_request_retrieves_and_feeds_final_writer() -> None:
    llm = FakeLLM(
        [
            _decision(needs_explanation=True),
            _reply(
                "你反复确认她会不会离开。精神分析把这种来回称为重复：旧的预期在新关系里继续要求答案。",
                explanatory_used=True,
            ),
        ]
    )
    knowledge = FakeKnowledgeBase()
    service = DialogueService(Settings(), llm=llm, knowledge_base=knowledge)

    result = service.respond(
        "我为什么总害怕朋友离开？请告诉我相关知识点。",
        include_trace=True,
    )

    assert [query[0] for query in knowledge.queries] == [
        "害怕朋友离开 不敢表达需要",
        "移情 重复 依赖",
        "分离焦虑 如何理解",
        "依赖需求 减少自责",
    ]
    assert "分离焦虑 如何理解" in llm.calls[1]["user_prompt"]
    assert "把知识自然写进同一封回信" in llm.calls[1]["system_prompt"]
    assert result.trace is not None
    assert result.trace.decision.needs_knowledge_explanation is True
    assert result.trace.grounding_knowledge[0].author == "Sigmund Freud"
    assert result.trace.explanatory_knowledge[0].author == "Sigmund Freud"


def test_safety_routing_overrides_knowledge_request() -> None:
    llm = FakeLLM(
        [
            _decision(
                needs_explanation=True,
                risk_flag="self_harm",
                needs_regulation=True,
            ),
            _reply(
                "先确认你此刻是否安全。请联系可信任的人或 12356；若有即时危险，请拨打 120 或 110。",
                risk_flag="self_harm",
            ),
        ]
    )
    knowledge = FakeKnowledgeBase()
    service = DialogueService(Settings(), llm=llm, knowledge_base=knowledge)

    result = service.respond(
        "我想自杀。请告诉我相关知识点。",
        include_trace=True,
    )

    assert detect_explicit_risk("我想自杀。") == "self_harm"
    assert knowledge.queries == []
    assert "12356" in result.reply
    assert result.trace is not None
    assert result.trace.decision.risk_flag == "self_harm"
    assert result.trace.retrieval_skipped_reason == "Safety routing paused psychodynamic knowledge retrieval."
    assert result.interaction is None


def test_safety_reply_with_psychodynamic_analysis_is_rewritten() -> None:
    llm = FakeLLM(
        [
            _decision(needs_explanation=True, risk_flag="self_harm"),
            _reply("精神分析里，这是压抑的冲突。", risk_flag="self_harm"),
            _reply(
                "我先不解释这份痛苦。请确认此刻安全，并联系可信任的人或 12356。",
                risk_flag="self_harm",
            ),
        ]
    )
    service = DialogueService(Settings(), llm=llm, knowledge_base=FakeKnowledgeBase())

    result = service.respond("我想自残。请告诉我相关知识。", include_trace=True)

    assert result.reply.startswith("我先不解释")
    assert len(llm.calls) == 3
    assert "安全边界校验" in llm.calls[2]["system_prompt"]


def test_passive_death_language_does_not_trigger_hard_safety_routing() -> None:
    llm = FakeLLM(
        [
            _decision(needs_explanation=True, risk_flag="self_harm"),
            _reply("你停在不想死也不想活之间，这份停滞仍然可以被理解。", explanatory_used=True),
        ]
    )
    knowledge = FakeKnowledgeBase()
    service = DialogueService(Settings(), llm=llm, knowledge_base=knowledge)

    result = service.respond("我不想死，也不想活，我觉得自己没有未来。", include_trace=True)

    assert detect_explicit_risk("我不想死，也不想活。") == "none"
    assert result.trace is not None
    assert result.trace.decision.risk_flag == "none"
    assert result.trace.grounding_knowledge
    assert result.trace.explanatory_knowledge


def test_tired_letter_returns_detection_only_offer() -> None:
    llm = FakeLLM(
        [
            _decision(
                needs_explanation=False,
                needs_regulation=True,
            ),
            _reply("你已经说了很多，现在可以先停在这里。"),
        ]
    )
    service = DialogueService(Settings(), llm=llm, knowledge_base=FakeKnowledgeBase())

    result = service.respond("我真的很累，不想再说了。", include_trace=True)

    assert result.interaction is not None
    assert result.interaction.type == "breathing_or_meditation_offer"
    assert result.interaction.message == "不说话也没关系，也许呼吸或者冥想更适合现在的状态？"
    assert result.reply == (
        "你已经说了很多，现在可以先停在这里。\n\n"
        "不说话也没关系，也许呼吸或者冥想更适合现在的状态？"
    )
    assert result.trace is not None
    assert result.trace.decision.needs_regulation_interaction is True
    assert "regulation_offer" in llm.calls[1]["user_prompt"]
    assert "不要提供任何呼吸方法" in llm.calls[1]["system_prompt"]


def test_offer_is_not_duplicated_when_model_already_returns_fixed_message() -> None:
    llm = FakeLLM(
        [
            _decision(
                needs_explanation=False,
                needs_regulation=True,
            ),
            _reply("不说话也没关系，也许呼吸或者冥想更适合现在的状态？"),
        ]
    )
    service = DialogueService(Settings(), llm=llm, knowledge_base=FakeKnowledgeBase())

    result = service.respond("我累了，不想说了。")

    assert result.interaction is not None
    assert result.reply.count(result.interaction.message) == 1


def test_letter_without_current_fatigue_returns_no_offer() -> None:
    llm = FakeLLM(
        [
            _decision(needs_explanation=False),
            _reply("你有时不回复，也许是在给关系里的自己留一点空间。"),
        ]
    )
    service = DialogueService(Settings(), llm=llm, knowledge_base=FakeKnowledgeBase())

    result = service.respond("我最近有时不太想回微信。")

    assert result.interaction is None


def test_trace_is_hidden_by_default() -> None:
    service = DialogueService(
        Settings(),
        llm=FakeLLM(
            [
                _decision(needs_explanation=False),
                _reply("一封回信。"),
            ]
        ),
        knowledge_base=FakeKnowledgeBase(),
    )

    result = service.respond("请回复我。")

    assert result.trace is None


def test_invalid_json_shape_gets_one_format_repair() -> None:
    llm = FakeLLM(
        [
            _decision(needs_explanation=False),
            {"title": "ReplyEnvelope", "type": "object"},
            _reply("修复后的回信。"),
        ]
    )
    service = DialogueService(Settings(), llm=llm, knowledge_base=FakeKnowledgeBase())

    result = service.respond("请回复我。", include_trace=True)

    assert result.reply == "修复后的回信。"
    assert len(llm.calls) == 3
    assert "输出修复" in llm.calls[2]["system_prompt"]
    assert result.trace is not None
    assert len(result.trace.usage) == 3


def test_local_bm25_knowledge_base_reads_processed_corpus(tmp_path) -> None:
    corpus = [
        {
            "text": "压抑并非消失，被挡住的欲望会以症状形式返回。",
            "author": "Sigmund Freud",
            "school": "精神分析",
            "core_concepts": ["压抑", "症状"],
            "source_document": "freud_unconscious.md",
        },
        {
            "text": "抱持环境让尚未整合的自体获得连续感。",
            "author": "D. W. Winnicott",
            "school": "客体关系",
            "core_concepts": ["抱持"],
            "source_document": "winnicott.md",
        },
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in corpus), encoding="utf-8")
    kb = DialogueKnowledgeBase(Settings(data_processed_dir=tmp_path))

    hits = kb.query("压抑 症状返回", top_n=1, school="精神分析")

    assert len(hits) == 1
    assert hits[0].author == "Sigmund Freud"
    assert "压抑" in hits[0].text


def test_english_school_alias_is_normalized_for_chinese_metadata() -> None:
    query = SearchQuery(
        query="关系冲突",
        school="object_relations",
        rationale="测试流派归一化",
    )

    assert query.school == "客体关系"
