# SupportX (WB-4) — The Self-Improving Support Agent

SupportX is a prototype customer support agent that follows the loop:

**Listen → Decide → Respond → Improve**

It uses a small local knowledge base, detects frustration/confusion, escalates when needed, and logs repeated unanswered questions as improvement suggestions.

## Features (deliverables)

- **Knowledge-base answering (RAG-style)**: retrieves the most relevant KB docs and replies with excerpts + citations.
- **Unhappy / confused detection**: light-weight sentiment/confusion analysis.
- **Escalation to human**: creates a ticket and returns a handoff message when confidence is low.
- **Self-improvement queue**: clusters escalated/low-confidence questions into an improvement suggestion queue.
- **Encrypted local storage**: sensitive fields are stored encrypted (AES-GCM).
- **Biometric login simulation**: challenge/response login using a client-side “biometric token” (no real biometrics stored).
- **Dead Man’s Switch**: if heartbeats stop, an alert triggers (stdout or webhook).

## Quickstart

### 1) Install

Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi uvicorn cryptography pydantic
```

### 2) Run the backend

```bash
uvicorn supportx.backend.app:app --reload
```

The server will create a local encrypted SQLite DB at `supportx/data/supportx.db`.

### 3) Use the demo UI

Open `supportx/frontend/index.html` in your browser.

## Environment variables

- `SUPPORTX_MASTER_KEY`: base64-url encoded 32-byte key (AES-256). If not set, a dev key is used.
- `SUPPORTX_DATA_DIR`: where to store `supportx.db`.
- `SUPPORTX_DMS_TIMEOUT_SECONDS`: dead man’s switch timeout (default `600`).
- `SUPPORTX_DMS_CHECK_INTERVAL_SECONDS`: watchdog polling interval (default `30`).
- `SUPPORTX_DMS_WEBHOOK_URL`: optional webhook to POST when the dead man’s switch triggers.

## Notes

- This is a buildable hackathon prototype (intentionally minimal). You can swap the answer generator for a local LLM (Ollama/vLLM) without changing the control-loop.
- The sentiment/confusion detector is heuristic to avoid heavy model downloads in offline environments.
