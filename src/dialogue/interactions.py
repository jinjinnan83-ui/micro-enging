"""Detection-only frontend signal for users who want to pause talking."""

from __future__ import annotations

from src.dialogue.models import DialogueDecision, RegulationOffer


def build_regulation_offer(decision: DialogueDecision) -> RegulationOffer | None:
    if not decision.needs_regulation_interaction:
        return None
    return RegulationOffer()
