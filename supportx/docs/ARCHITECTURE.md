# SupportX Architecture

## Control loop

SupportX implements the required loop:

1. **Listen**: accept the user message and update in-memory state.
2. **Decide**: compute signals (sentiment/confusion + retrieval score) and decide whether to escalate.
3. **Respond**: either answer from the KB or create a human ticket.
4. **Improve**: if we escalated or had low confidence, store the question into an improvement queue for later review.

## Agent orchestration

The orchestration is implemented in `supportx/backend/agents/graph.py`.

- If `langgraph` is installed, a `StateGraph` is used.
- If not, SupportX falls back to a small sequential runner with the same steps.

Nodes:

- `listen` — `supportx/backend/agents/nodes/listen.py`
- `analyze` — `supportx/backend/agents/nodes/analyze.py`
- `retrieve` — `supportx/backend/agents/nodes/retrieve.py`
- `decide` — `supportx/backend/agents/nodes/decide.py`
- `respond` — `supportx/backend/agents/nodes/respond.py`
- `improve` — `supportx/backend/agents/nodes/improve.py`

## Encrypted local storage

Sensitive content is stored encrypted using AES-GCM:

- message text (`messages.text_enc`)
- biometric secret (`users.biometric_secret_enc`)
- knowledge base bodies (`kb_docs.body_enc`)
- improvement payloads (`improvements.sample_questions_enc`, `improvements.proposed_answer_enc`)

Encryption helpers: `supportx/backend/security/encryption.py`.

## Biometric login simulation

The biometric flow is a challenge/response system:

1. `/auth/biometric/challenge` issues a short-lived challenge.
2. The client signs the challenge using a locally stored biometric token (HMAC-SHA256).
3. `/auth/biometric/verify` validates the signature and returns a session token.

Implementation: `supportx/backend/security/biometrics_sim.py`.

## Dead Man’s Switch

The backend stores a last-heartbeat timestamp and runs a watchdog task:

- Heartbeat set on each chat call and via `/deadmans/heartbeat`.
- If the heartbeat is missing for longer than `SUPPORTX_DMS_TIMEOUT_SECONDS`, SupportX triggers an alert.

Implementation: `supportx/backend/security/dead_mans_switch.py`.

## Self-improvement

Any escalated or low-confidence question is assigned a cheap semantic cluster id and stored in `improvements`.

A human can approve a draft in `/admin/improvements/{id}/approve` to promote it into the KB and re-index.

Clustering helpers: `supportx/backend/improvement/clustering.py`.
