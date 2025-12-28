from __future__ import annotations

from ..deps import SupportDeps
from ..state import SupportState
from ...models.sentiment import analyze_text


def analyze_node(state: SupportState, deps: SupportDeps) -> SupportState:
    text = state["last_user_message"]
    signals = analyze_text(text)

    deps.db.add_message(
        conversation_id=state["conversation_id"],
        sender="user",
        text_enc=deps.encryptor.encrypt(text),
        sentiment=signals.sentiment,
        confusion=signals.confusion_score,
    )

    return {
        "sentiment": signals.sentiment,
        "confusion_score": signals.confusion_score,
    }
