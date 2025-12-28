from __future__ import annotations

from pathlib import Path

from supportx.backend.config import load_settings
from supportx.backend.db.sqlite import Database
from supportx.backend.rag.ingest import ingest_kb_dir
from supportx.backend.security.encryption import Encryptor


def main() -> None:
    settings = load_settings()
    db = Database(settings.sqlite_path)
    enc = Encryptor(settings.encryption_key)
    kb_dir = Path(__file__).resolve().parents[1] / "kb"
    count = ingest_kb_dir(db, enc, kb_dir)
    print(f"Ingested {count} documents")


if __name__ == "__main__":
    main()
