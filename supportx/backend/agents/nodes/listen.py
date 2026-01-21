from __future__ import annotations

from ..deps import SupportDeps
from ..state import SupportState


def listen_node(state: SupportState, deps: SupportDeps) -> SupportState:
    text = state["last_user_message"].strip()

    messages = list(state.get("messages", []))
    messages.append({"sender": "user", "text": text})

    return {"messages": messages, "last_user_message": text}
