from __future__ import annotations

from ..deps import SupportDeps
from ..state import SupportState


def _format_answer(docs) -> str:
    if not docs:
        return (
            "I couldn't find a matching help article in my knowledge base. "
            "Could you share a bit more detail (product, error message, and what you tried)?"
        )

    top = docs[0]
    citations = []
    for d in docs[:3]:
        if d.source:
            citations.append(f"- {d.title} ({d.source})")
        else:
            citations.append(f"- {d.title}")

    body_preview = top.body.strip()
    if len(body_preview) > 900:
        body_preview = body_preview[:900].rstrip() + "…"

    return (
        f"Here’s what I found that should help:\n\n{body_preview}\n\n"
        "Sources:\n" + "\n".join(citations)
    )


def respond_node(state: SupportState, deps: SupportDeps) -> SupportState:
    if state.get("needs_escalation"):
        ticket_id = deps.db.create_ticket(state["conversation_id"], priority="high")
        msg = (
            "I’m going to escalate this to a human support agent so you get the best help. "
            f"Your ticket id is {ticket_id}."
        )

        deps.db.add_message(
            conversation_id=state["conversation_id"],
            sender="agent",
            text_enc=deps.encryptor.encrypt(msg),
            sentiment=None,
            confusion=None,
        )

        messages = list(state.get("messages", []))
        messages.append({"sender": "agent", "text": msg})

        return {"ticket_id": ticket_id, "answer": msg, "messages": messages}

    answer = _format_answer(state.get("retrieved_docs", []))

    deps.db.add_message(
        conversation_id=state["conversation_id"],
        sender="agent",
        text_enc=deps.encryptor.encrypt(answer),
        sentiment=None,
        confusion=None,
    )

    messages = list(state.get("messages", []))
    messages.append({"sender": "agent", "text": answer})

    return {"answer": answer, "messages": messages}
