from __future__ import annotations

import base64
import hmac
import os
from dataclasses import dataclass
from hashlib import sha256


def generate_biometric_secret() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def generate_challenge() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def sign_challenge(biometric_secret: str, challenge: str) -> str:
    secret = base64.urlsafe_b64decode(biometric_secret.encode("ascii"))
    mac = hmac.new(secret, challenge.encode("utf-8"), sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("ascii")


def verify_signature(biometric_secret: str, challenge: str, signature: str) -> bool:
    expected = sign_challenge(biometric_secret, challenge)
    return hmac.compare_digest(expected, signature)


@dataclass(frozen=True)
class BiometricLoginPayload:
    email: str
    challenge: str
    signature: str
