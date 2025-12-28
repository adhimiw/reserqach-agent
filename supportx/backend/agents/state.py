from __future__ import annotations

from typing import TypedDict, Literal

from ..rag.retriever import RetrievedDoc


Sender = Literal["user", "agent", "human"]


class SupportMessage(TypedDict):
    sender: Sender
    text: str


class SupportState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    messages: list[SupportMessage]
    last_user_message: str

    sentiment: str
    confusion_score: float

    retrieved_docs: list[RetrievedDoc]
    top_score: float

    needs_escalation: bool
    answer: str
    ticket_id: str
    improvement_cluster_id: str
