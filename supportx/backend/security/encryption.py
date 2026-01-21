from __future__ import annotations

import base64
import os
from dataclasses import dataclass

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None


@dataclass(frozen=True)
class Encryptor:
    key: bytes

    def encrypt(self, plaintext: str) -> str:
        if AESGCM is None:  # pragma: no cover
            raise RuntimeError("cryptography is required for encrypted storage")
        nonce = os.urandom(12)
        aesgcm = AESGCM(self.key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt(self, token: str) -> str:
        if AESGCM is None:  # pragma: no cover
            raise RuntimeError("cryptography is required for encrypted storage")
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(self.key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
