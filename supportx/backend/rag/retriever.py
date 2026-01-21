from __future__ import annotations

from dataclasses import dataclass

from ..db.sqlite import Database
from ..security.encryption import Encryptor
from .embeddings import HashEmbedding, cosine


@dataclass(frozen=True)
class RetrievedDoc:
    id: str
    title: str
    body: str
    tags: str | None
    source: str | None
    score: float


class Retriever:
    def __init__(self, db: Database, encryptor: Encryptor, embedding: HashEmbedding | None = None):
        self._db = db
        self._encryptor = encryptor
        self._embedding = embedding or HashEmbedding()
        self._index: list[tuple[dict, list[float]]] = []
        self.refresh_index()

    def refresh_index(self) -> None:
        self._index = []
        for row in self._db.list_kb_docs():
            body = self._encryptor.decrypt(row["body_enc"])
            text = f"{row['title']}\n\n{body}"
            vec = self._embedding.embed(text)
            self._index.append(({"row": row, "body": body}, vec))

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        qv = self._embedding.embed(query)
        scored: list[RetrievedDoc] = []
        for item, vec in self._index:
            score = cosine(qv, vec)
            row = item["row"]
            scored.append(
                RetrievedDoc(
                    id=row["id"],
                    title=row["title"],
                    body=item["body"],
                    tags=row.get("tags"),
                    source=row.get("source"),
                    score=score,
                )
            )
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]
