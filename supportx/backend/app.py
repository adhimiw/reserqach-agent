from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .agents.deps import SupportDeps
from .agents.graph import build_support_chain
from .config import Settings, load_settings
from .db.passwords import hash_password, verify_password
from .db.sqlite import Database
from .rag.ingest import ingest_kb_dir
from .rag.retriever import Retriever
from .security.biometrics_sim import generate_biometric_secret, generate_challenge, verify_signature
from .security.dead_mans_switch import DeadMansSwitch, DeadMansSwitchConfig
from .security.encryption import Encryptor


@dataclass
class Resources:
    settings: Settings
    db: Database
    encryptor: Encryptor
    retriever: Retriever
    deps: SupportDeps
    chain: Any


app = FastAPI(title="SupportX", version="0.1.0")
_RESOURCES: Resources | None = None


def resources() -> Resources:
    if _RESOURCES is None:
        raise RuntimeError("SupportX resources not initialized")
    return _RESOURCES


class RegisterIn(BaseModel):
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class ChallengeIn(BaseModel):
    email: str


class VerifyBiometricIn(BaseModel):
    email: str
    challenge: str
    signature: str


class StartConversationOut(BaseModel):
    conversation_id: str


class ChatIn(BaseModel):
    message: str


class ChatOut(BaseModel):
    answer: str
    needs_escalation: bool
    ticket_id: str | None = None
    sentiment: str | None = None
    confusion_score: float | None = None
    top_score: float | None = None
    retrieved_titles: list[str] = []


def _bearer_token(authorization: str | None = Header(default=None)) -> str | None:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip()


def require_user_id(token: str | None = Depends(_bearer_token)) -> str:
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    session = resources().db.get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    return session["user_id"]


@app.on_event("startup")
async def _startup() -> None:
    global _RESOURCES

    settings = load_settings()
    db = Database(settings.sqlite_path)
    encryptor = Encryptor(settings.encryption_key)
    retriever = Retriever(db, encryptor)
    deps = SupportDeps(settings=settings, db=db, encryptor=encryptor, retriever=retriever)
    chain = build_support_chain(deps)

    if not db.list_kb_docs():
        kb_dir = Path(__file__).resolve().parents[1] / "kb"
        if kb_dir.exists():
            ingest_kb_dir(db, encryptor, kb_dir)
            retriever.refresh_index()

    db.set_heartbeat(time.time())

    webhook_url = os.getenv("SUPPORTX_DMS_WEBHOOK_URL")
    dms = DeadMansSwitch(
        get_last_heartbeat=db.get_heartbeat,
        set_last_triggered=db.set_last_triggered,
        config=DeadMansSwitchConfig(timeout_seconds=settings.dead_mans_timeout_seconds, webhook_url=webhook_url),
    )

    async def _watchdog() -> None:
        while True:
            await asyncio.sleep(settings.dead_mans_check_interval_seconds)
            if dms.should_trigger():
                dms.trigger(reason="No heartbeat received within timeout")

    asyncio.create_task(_watchdog())

    _RESOURCES = Resources(
        settings=settings,
        db=db,
        encryptor=encryptor,
        retriever=retriever,
        deps=deps,
        chain=chain,
    )


@app.post("/auth/register")
def register(payload: RegisterIn) -> dict[str, Any]:
    r = resources()
    if r.db.get_user_by_email(payload.email):
        raise HTTPException(status_code=400, detail="User already exists")

    biometric_secret = generate_biometric_secret()
    user_id = r.db.create_user(
        email=payload.email,
        password_hash=hash_password(payload.password),
        biometric_secret_enc=r.encryptor.encrypt(biometric_secret),
    )

    return {"user_id": user_id, "biometric_token": biometric_secret}


@app.post("/auth/login")
def login(payload: LoginIn) -> dict[str, Any]:
    r = resources()
    user = r.db.get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = r.db.create_session(user["id"])
    return {"token": token}


@app.post("/auth/biometric/challenge")
def biometric_challenge(payload: ChallengeIn) -> dict[str, Any]:
    r = resources()
    if not r.db.get_user_by_email(payload.email):
        raise HTTPException(status_code=404, detail="Unknown user")
    challenge = generate_challenge()
    r.db.upsert_biometric_challenge(payload.email, challenge=challenge, expires_at=time.time() + 60)
    return {"challenge": challenge, "expires_in": 60}


@app.post("/auth/biometric/verify")
def biometric_verify(payload: VerifyBiometricIn) -> dict[str, Any]:
    r = resources()
    user = r.db.get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=404, detail="Unknown user")

    row = r.db.get_biometric_challenge(payload.email)
    if not row:
        raise HTTPException(status_code=400, detail="No active challenge")

    if row["expires_at"] < time.time() or row["challenge"] != payload.challenge:
        r.db.delete_biometric_challenge(payload.email)
        raise HTTPException(status_code=400, detail="Challenge expired")

    biometric_secret = r.encryptor.decrypt(user["biometric_secret_enc"])
    ok = verify_signature(biometric_secret, payload.challenge, payload.signature)
    r.db.delete_biometric_challenge(payload.email)

    if not ok:
        raise HTTPException(status_code=401, detail="Biometric verification failed")

    token = r.db.create_session(user["id"])
    return {"token": token}


@app.post("/chat/start", response_model=StartConversationOut)
def start_conversation(user_id: str = Depends(require_user_id)) -> StartConversationOut:
    r = resources()
    cid = r.db.create_conversation(user_id)
    return StartConversationOut(conversation_id=cid)


@app.post("/chat/{conversation_id}", response_model=ChatOut)
async def chat(conversation_id: str, payload: ChatIn, user_id: str = Depends(require_user_id)) -> ChatOut:
    r = resources()
    convo = r.db.fetchone("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    if not convo or convo["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    state = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "messages": [],
        "last_user_message": payload.message,
    }

    result = await r.chain.ainvoke(state)

    r.db.set_heartbeat(time.time())

    docs = result.get("retrieved_docs", [])
    return ChatOut(
        answer=result.get("answer", ""),
        needs_escalation=bool(result.get("needs_escalation")),
        ticket_id=result.get("ticket_id") or None,
        sentiment=result.get("sentiment"),
        confusion_score=result.get("confusion_score"),
        top_score=result.get("top_score"),
        retrieved_titles=[d.title for d in docs],
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user_id: str = Depends(require_user_id)) -> dict[str, Any]:
    r = resources()
    convo = r.db.fetchone("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    if not convo or convo["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    for row in r.db.get_messages(conversation_id):
        messages.append(
            {
                "sender": row["sender"],
                "text": r.encryptor.decrypt(row["text_enc"]),
                "sentiment": row.get("sentiment"),
                "confusion": row.get("confusion"),
                "created_at": row["created_at"],
            }
        )

    return {"conversation": convo, "messages": messages}


@app.get("/admin/improvements")
def list_improvements(user_id: str = Depends(require_user_id)) -> dict[str, Any]:
    r = resources()
    out = []
    for row in r.db.list_improvements():
        out.append(
            {
                "id": row["id"],
                "cluster_id": row["cluster_id"],
                "status": row["status"],
                "sample_questions": json.loads(r.encryptor.decrypt(row["sample_questions_enc"])),
                "proposed_answer": r.encryptor.decrypt(row["proposed_answer_enc"]) if row.get("proposed_answer_enc") else None,
                "created_at": row["created_at"],
            }
        )
    return {"items": out}


class ApproveIn(BaseModel):
    title: str | None = None


@app.post("/admin/improvements/{improvement_id}/approve")
def approve_improvement(improvement_id: str, payload: ApproveIn, user_id: str = Depends(require_user_id)) -> dict[str, Any]:
    r = resources()
    row = r.db.fetchone("SELECT * FROM improvements WHERE id = ?", (improvement_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    proposed = row.get("proposed_answer_enc")
    if not proposed:
        raise HTTPException(status_code=400, detail="No proposed draft")

    doc_body = r.encryptor.decrypt(proposed)
    title = payload.title or f"Support Article {row['cluster_id']}"
    r.db.add_kb_doc(title=title, body_enc=r.encryptor.encrypt(doc_body), tags="auto", source="improvement")
    r.db.update_improvement_status(improvement_id, "approved")
    r.retriever.refresh_index()

    return {"ok": True}


@app.post("/deadmans/heartbeat")
def deadmans_heartbeat(user_id: str = Depends(require_user_id)) -> dict[str, Any]:
    r = resources()
    r.db.set_heartbeat(time.time())
    return {"ok": True, "status": r.db.get_dead_mans_status()}


@app.get("/deadmans/status")
def deadmans_status(user_id: str = Depends(require_user_id)) -> dict[str, Any]:
    return resources().db.get_dead_mans_status()
