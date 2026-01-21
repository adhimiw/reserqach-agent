from __future__ import annotations

from pathlib import Path

from ..db.sqlite import Database
from ..security.encryption import Encryptor


def _parse_markdown(md: str, fallback_title: str) -> tuple[str, str]:
    lines = md.splitlines()
    title = None
    if lines and lines[0].lstrip().startswith("#"):
        title = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()
    else:
        body = md.strip()
    return (title or fallback_title, body)


def ingest_kb_dir(db: Database, encryptor: Encryptor, kb_dir: Path) -> int:
    count = 0
    for path in sorted(kb_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title, body = _parse_markdown(raw, fallback_title=path.stem)
        tags = ",".join([p for p in path.parts if p not in (".", "..")][-2:])
        db.add_kb_doc(
            title=title,
            body_enc=encryptor.encrypt(body),
            tags=tags,
            source=str(path.relative_to(kb_dir)),
        )
        count += 1
    return count
