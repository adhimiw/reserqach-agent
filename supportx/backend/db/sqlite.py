from __future__ import annotations

import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@dataclass
class Database:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = _row_to_dict
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def _migrate(self) -> None:
        cur = self._conn.cursor()

        cur.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                biometric_secret_enc TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                text_enc TEXT NOT NULL,
                sentiment TEXT,
                confusion REAL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kb_docs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body_enc TEXT NOT NULL,
                tags TEXT,
                source TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS improvements (
                id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                sample_questions_enc TEXT NOT NULL,
                proposed_answer_enc TEXT,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS biometric_challenges (
                email TEXT PRIMARY KEY,
                challenge TEXT NOT NULL,
                expires_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dead_mans (
                id TEXT PRIMARY KEY,
                last_heartbeat REAL,
                last_triggered REAL
            );
            """
        )
        cur.execute(
            "INSERT OR IGNORE INTO dead_mans (id, last_heartbeat, last_triggered) VALUES ('singleton', NULL, NULL)"
        )
        self._conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        cur = self._conn.execute(sql, params)
        return cur.fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, params)
        return list(cur.fetchall())

    # --- Users / sessions ---

    def create_user(self, email: str, password_hash: str, biometric_secret_enc: str) -> str:
        user_id = secrets.token_hex(16)
        self.execute(
            "INSERT INTO users (id, email, password_hash, biometric_secret_enc, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email.lower(), password_hash, biometric_secret_enc, time.time()),
        )
        return user_id

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM users WHERE email = ?", (email.lower(),))

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, time.time()),
        )
        return token

    def get_session(self, token: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM sessions WHERE token = ?", (token,))

    # --- Conversations / messages ---

    def create_conversation(self, user_id: str) -> str:
        cid = secrets.token_hex(16)
        self.execute(
            "INSERT INTO conversations (id, user_id, status, created_at) VALUES (?, ?, ?, ?)",
            (cid, user_id, "open", time.time()),
        )
        return cid

    def set_conversation_status(self, conversation_id: str, status: str) -> None:
        self.execute(
            "UPDATE conversations SET status = ? WHERE id = ?",
            (status, conversation_id),
        )

    def add_message(
        self,
        conversation_id: str,
        sender: str,
        text_enc: str,
        sentiment: str | None,
        confusion: float | None,
    ) -> str:
        mid = secrets.token_hex(16)
        self.execute(
            "INSERT INTO messages (id, conversation_id, sender, text_enc, sentiment, confusion, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, sender, text_enc, sentiment, confusion, time.time()),
        )
        return mid

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.fetchall(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )

    # --- KB ---

    def add_kb_doc(self, title: str, body_enc: str, tags: str | None, source: str | None) -> str:
        did = secrets.token_hex(16)
        self.execute(
            "INSERT INTO kb_docs (id, title, body_enc, tags, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (did, title, body_enc, tags, source, time.time()),
        )
        return did

    def list_kb_docs(self) -> list[dict[str, Any]]:
        return self.fetchall("SELECT * FROM kb_docs ORDER BY created_at DESC")

    # --- Escalations / tickets ---

    def create_ticket(self, conversation_id: str, priority: str = "normal") -> str:
        tid = secrets.token_hex(16)
        self.execute(
            "INSERT INTO tickets (id, conversation_id, priority, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (tid, conversation_id, priority, "open", time.time()),
        )
        self.set_conversation_status(conversation_id, "escalated")
        return tid

    # --- Improvement suggestions ---

    def add_improvement(self, cluster_id: str, sample_questions_enc: str, proposed_answer_enc: str | None) -> str:
        iid = secrets.token_hex(16)
        self.execute(
            "INSERT INTO improvements (id, cluster_id, sample_questions_enc, proposed_answer_enc, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (iid, cluster_id, sample_questions_enc, proposed_answer_enc, "pending_review", time.time()),
        )
        return iid

    def list_improvements(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self.fetchall(
                "SELECT * FROM improvements WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        return self.fetchall("SELECT * FROM improvements ORDER BY created_at DESC")

    def update_improvement_status(self, improvement_id: str, status: str) -> None:
        self.execute(
            "UPDATE improvements SET status = ? WHERE id = ?",
            (status, improvement_id),
        )

    def get_pending_improvement_by_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        return self.fetchone(
            "SELECT * FROM improvements WHERE cluster_id = ? AND status = 'pending_review' ORDER BY created_at DESC LIMIT 1",
            (cluster_id,),
        )

    def update_improvement_payload(
        self,
        improvement_id: str,
        sample_questions_enc: str,
        proposed_answer_enc: str | None,
    ) -> None:
        self.execute(
            "UPDATE improvements SET sample_questions_enc = ?, proposed_answer_enc = ? WHERE id = ?",
            (sample_questions_enc, proposed_answer_enc, improvement_id),
        )

    # --- Biometric challenges ---

    def upsert_biometric_challenge(self, email: str, challenge: str, expires_at: float) -> None:
        self.execute(
            "INSERT INTO biometric_challenges (email, challenge, expires_at) VALUES (?, ?, ?) "
            "ON CONFLICT(email) DO UPDATE SET challenge=excluded.challenge, expires_at=excluded.expires_at",
            (email.lower(), challenge, expires_at),
        )

    def get_biometric_challenge(self, email: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM biometric_challenges WHERE email = ?", (email.lower(),))

    def delete_biometric_challenge(self, email: str) -> None:
        self.execute("DELETE FROM biometric_challenges WHERE email = ?", (email.lower(),))

    # --- Dead man's switch ---

    def set_heartbeat(self, ts: float) -> None:
        self.execute("UPDATE dead_mans SET last_heartbeat = ? WHERE id = 'singleton'", (ts,))

    def get_heartbeat(self) -> float | None:
        row = self.fetchone("SELECT last_heartbeat FROM dead_mans WHERE id = 'singleton'")
        if not row:
            return None
        return row.get("last_heartbeat")

    def set_last_triggered(self, ts: float) -> None:
        self.execute("UPDATE dead_mans SET last_triggered = ? WHERE id = 'singleton'", (ts,))

    def get_dead_mans_status(self) -> dict[str, Any]:
        row = self.fetchone("SELECT * FROM dead_mans WHERE id = 'singleton'")
        return row or {"id": "singleton", "last_heartbeat": None, "last_triggered": None}
