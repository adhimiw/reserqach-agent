from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..rag.embeddings import HashEmbedding


@dataclass(frozen=True)
class ClusterResult:
    cluster_id: str


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def cluster_id_for_text(text: str, embedding: HashEmbedding | None = None) -> str:
    emb = embedding or HashEmbedding()
    vec = emb.embed(_normalize(text))

    # Bucket using the top-weighted dimensions as a cheap semantic hash.
    idxs = sorted(range(len(vec)), key=lambda i: vec[i], reverse=True)[:6]
    signature = ":".join(str(i) for i in idxs)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return f"cl_{digest}"


def draft_kb_article(sample_questions: list[str]) -> str:
    title = sample_questions[0][:60] if sample_questions else "New Support Article"
    questions_md = "\n".join([f"- {q}" for q in sample_questions[:5]])
    return (
        f"# {title}\n\n"
        "## Summary\n"
        "(Auto-generated draft; human review recommended.)\n\n"
        "## Common user questions\n"
        f"{questions_md}\n\n"
        "## Suggested answer\n"
        "Write the canonical resolution steps here.\n"
    )
