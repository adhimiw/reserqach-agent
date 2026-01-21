from __future__ import annotations

import base64
import hashlib
import os
import hmac


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return "pbkdf2_sha256$120000$" + base64.urlsafe_b64encode(salt + dk).decode("ascii")


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations_str, blob = stored.split("$", 2)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        salt, expected = raw[:16], raw[16:]
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False
