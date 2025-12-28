from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    sqlite_path: Path
    encryption_key: bytes
    dead_mans_timeout_seconds: int
    dead_mans_check_interval_seconds: int


def _default_key() -> bytes:
    # Stable dev-only key (32 bytes) so local encrypted storage works out-of-the-box.
    # Override in production with SUPPORTX_MASTER_KEY (base64-urlsafe, 32 bytes).
    return hashlib.sha256(b"supportx-insecure-dev-key").digest()


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(os.getenv("SUPPORTX_DATA_DIR", repo_root / "supportx" / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = Path(os.getenv("SUPPORTX_SQLITE_PATH", str(data_dir / "supportx.db")))

    raw_key = os.getenv("SUPPORTX_MASTER_KEY")
    if raw_key:
        try:
            encryption_key = base64.urlsafe_b64decode(raw_key)
        except Exception as e:  # pragma: no cover
            raise ValueError("SUPPORTX_MASTER_KEY must be base64-url encoded") from e
    else:
        encryption_key = _default_key()

    if len(encryption_key) != 32:
        raise ValueError("Encryption key must be 32 bytes (AES-256)")

    return Settings(
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        encryption_key=encryption_key,
        dead_mans_timeout_seconds=int(os.getenv("SUPPORTX_DMS_TIMEOUT_SECONDS", "600")),
        dead_mans_check_interval_seconds=int(
            os.getenv("SUPPORTX_DMS_CHECK_INTERVAL_SECONDS", "30")
        ),
    )
