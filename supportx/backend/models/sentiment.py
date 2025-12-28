from __future__ import annotations

import re
from dataclasses import dataclass


_NEGATIVE = {
    "angry",
    "annoyed",
    "bad",
    "broken",
    "hate",
    "horrible",
    "terrible",
    "useless",
    "worst",
    "refund",
    "cancel",
    "not working",
    "doesn't work",
}

_POSITIVE = {
    "great",
    "awesome",
    "good",
    "love",
    "thanks",
    "thank you",
    "perfect",
    "amazing",
}

_CONFUSION = {
    "confused",
    "don't understand",
    "dont understand",
    "unclear",
    "what do you mean",
    "doesn't help",
    "doesnt help",
    "not helpful",
    "you are not listening",
}


@dataclass(frozen=True)
class TextSignals:
    sentiment: str
    sentiment_score: float
    confusion_score: float


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def analyze_text(text: str) -> TextSignals:
    t = _normalize(text)

    neg_hits = sum(1 for w in _NEGATIVE if w in t)
    pos_hits = sum(1 for w in _POSITIVE if w in t)

    if neg_hits > pos_hits and neg_hits > 0:
        sentiment = "negative"
    elif pos_hits > neg_hits and pos_hits > 0:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    sentiment_score = max(-1.0, min(1.0, (pos_hits - neg_hits) / 3.0))

    confusion_hits = sum(1 for w in _CONFUSION if w in t)
    q_marks = t.count("?")
    confusion_score = max(0.0, min(1.0, (confusion_hits + (1 if q_marks else 0)) / 3.0))

    return TextSignals(
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        confusion_score=confusion_score,
    )
