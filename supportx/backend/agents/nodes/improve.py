from __future__ import annotations

import json

from ..deps import SupportDeps
from ..state import SupportState
from ...improvement.clustering import draft_kb_article


def improve_node(state: SupportState, deps: SupportDeps) -> SupportState:
    if not state.get("needs_escalation") and float(state.get("top_score", 0.0)) >= 0.35:
        return {}

    cluster_id = state.get("improvement_cluster_id") or "unknown"
    question = state["last_user_message"].strip()

    existing = deps.db.get_pending_improvement_by_cluster(cluster_id)
    if existing:
        samples = json.loads(deps.encryptor.decrypt(existing["sample_questions_enc"]))
        if question not in samples:
            samples.append(question)
        proposed = existing.get("proposed_answer_enc")
        deps.db.update_improvement_payload(
            improvement_id=existing["id"],
            sample_questions_enc=deps.encryptor.encrypt(json.dumps(samples)),
            proposed_answer_enc=proposed,
        )
        return {}

    samples = [question]
    draft = draft_kb_article(samples)

    deps.db.add_improvement(
        cluster_id=cluster_id,
        sample_questions_enc=deps.encryptor.encrypt(json.dumps(samples)),
        proposed_answer_enc=deps.encryptor.encrypt(draft),
    )

    return {}
