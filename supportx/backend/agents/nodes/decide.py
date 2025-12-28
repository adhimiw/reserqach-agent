from __future__ import annotations

from ..deps import SupportDeps
from ..state import SupportState
from ...improvement.clustering import cluster_id_for_text


def decide_node(state: SupportState, deps: SupportDeps) -> SupportState:
    msg = state["last_user_message"].lower()

    top_score = float(state.get("top_score", 0.0))
    confusion = float(state.get("confusion_score", 0.0))
    sentiment = state.get("sentiment", "neutral")

    explicit_handoff = any(p in msg for p in ("human", "agent", "representative"))

    low_confidence = top_score < 0.20
    struggling = confusion >= 0.65 or (sentiment == "negative" and top_score < 0.35)

    needs_escalation = bool(explicit_handoff or low_confidence or struggling)

    improvement_cluster_id = cluster_id_for_text(state["last_user_message"]) if needs_escalation else ""

    return {
        "needs_escalation": needs_escalation,
        "improvement_cluster_id": improvement_cluster_id,
    }
