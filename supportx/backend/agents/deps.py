from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..db.sqlite import Database
from ..rag.retriever import Retriever
from ..security.encryption import Encryptor


@dataclass(frozen=True)
class SupportDeps:
    settings: Settings
    db: Database
    encryptor: Encryptor
    retriever: Retriever
