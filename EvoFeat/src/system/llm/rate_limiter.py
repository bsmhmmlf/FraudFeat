from __future__ import annotations

import threading
import time
from collections import deque

from ..utils import get_logger


logger = get_logger("ratelimit")


class RateLimiter:
    def __init__(self, qpm: int, window_sec: float = 60.0, name: str = "global"):
        self.qpm = max(0, int(qpm))
        self.window_sec = float(window_sec)
        self.name = name
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()
        self._acquired = 0
        self._wait_total = 0.0
        self._wait_max = 0.0

    @property
    def enabled(self) -> bool:
        return self.qpm > 0

    def acquire(self) -> float:
        if not self.enabled:
            with self._lock:
                self._acquired += 1
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()

                while self._calls and now - self._calls[0] >= self.window_sec:
                    self._calls.popleft()
                if len(self._calls) < self.qpm:
                    self._calls.append(now)
                    self._acquired += 1
                    self._wait_total += waited
                    self._wait_max = max(self._wait_max, waited)
                    return waited
                sleep_for = self.window_sec - (now - self._calls[0]) + 0.01
            time.sleep(sleep_for)
            waited += sleep_for

    def stats(self) -> dict:
        with self._lock:
            return {
                "qpm_limit": self.qpm if self.enabled else None,
                "acquired": self._acquired,
                "wait_total_sec": round(self._wait_total, 1),
                "wait_max_sec": round(self._wait_max, 1),
            }


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def configure_rate_limiter(qpm: int, window_sec: float = 60.0) -> RateLimiter:
    global _limiter
    with _limiter_lock:
        _limiter = RateLimiter(qpm, window_sec=window_sec)
        if _limiter.enabled:
            logger.info(f"[rate limit] Global QPM quota = {qpm}/minute (shared by all agents)")
        else:
            logger.info("[rate limit] No QPM quota configured (unlimited)")
        return _limiter


def get_rate_limiter(default_qpm: int = 0) -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter(default_qpm)
    return _limiter
