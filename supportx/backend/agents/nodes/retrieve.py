from __future__ import annotations

from ..deps import SupportDeps
from ..state import SupportState


def retrieve_node(state: SupportState, deps: SupportDeps) -> SupportState:
    docs = deps.retriever.search(state["last_user_message"], k=5)
    top_score = docs[0].score if docs else 0.0
    return {"retrieved_docs": docs, "top_score": top_score}
