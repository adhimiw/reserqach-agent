#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"

uvicorn supportx.backend.app:app --reload --port 8000
