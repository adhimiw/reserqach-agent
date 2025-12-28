from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> Iterable[str]:
    for m in _TOKEN_RE.finditer(text.lower()):
        yield m.group(0)


@dataclass(frozen=True)
class HashEmbedding:
    dims: int = 256

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in _tokens(text):
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
