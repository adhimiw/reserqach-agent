from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass
class DeadMansSwitchConfig:
    timeout_seconds: int
    webhook_url: str | None = None


class DeadMansSwitch:
    def __init__(
        self,
        get_last_heartbeat: Callable[[], float | None],
        set_last_triggered: Callable[[float], None],
        config: DeadMansSwitchConfig,
    ):
        self._get_last_heartbeat = get_last_heartbeat
        self._set_last_triggered = set_last_triggered
        self._config = config

    def should_trigger(self, now: float | None = None) -> bool:
        now = now or time.time()
        last = self._get_last_heartbeat()
        if last is None:
            return False
        return (now - last) > self._config.timeout_seconds

    def trigger(self, reason: str, now: float | None = None) -> None:
        now = now or time.time()
        self._set_last_triggered(now)

        payload = {
            "event": "supportx.dead_mans_switch.triggered",
            "reason": reason,
            "ts": now,
        }

        if not self._config.webhook_url:
            print(json.dumps(payload))
            return

        req = urllib.request.Request(
            self._config.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as _:
                return
        except Exception:
            # Best-effort alerting; don't crash the app.
            print(json.dumps({**payload, "webhook_error": True}))
