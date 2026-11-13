from __future__ import annotations

import os
import threading
import time

import requests

from ..utils import get_logger, load_config
from .rate_limiter import get_rate_limiter


logger = get_logger("llm")
SHOW_LLM = os.environ.get("SYSTEM_SHOW_LLM", "").lower() in ("1", "true", "yes")


_CLIENTS: list["LLMClient"] = []


def aggregate_usage() -> dict:
    return {"calls": sum(c.calls for c in _CLIENTS),
            "prompt_tokens": sum(getattr(c, "prompt_tokens", 0) for c in _CLIENTS),
            "completion_tokens": sum(getattr(c, "completion_tokens", 0)
                                     for c in _CLIENTS)}


class LLMClient:
    def __init__(self, config: dict | None = None, name: str = "llm"):
        cfg = config or load_config("llm.yaml")
        self.api_url = os.environ.get("SYSTEM_LLM_API_URL", cfg.get("api_url"))
        self.api_key = os.environ.get("SYSTEM_LLM_API_KEY", cfg.get("api_key", ""))
        self.model = os.environ.get("SYSTEM_LLM_MODEL", cfg.get("model", "hy4-preview"))
        if not self.api_key or self.api_key == "YOUR_KEY_HERE":
            logger.warning(
                "LLM api_key is not configured (config/llm.yaml or SYSTEM_LLM_API_KEY). "
                "Requests will fail; supply your own access token first."
            )
        self.connect_timeout = int(cfg.get("connect_timeout", 10))
        self.timeout = int(cfg.get("timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.retry_base_delay = int(cfg.get("retry_base_delay", 5))
        self.retry_max_delay = int(cfg.get("retry_max_delay", 80))
        self.default_temperature = cfg.get("default_temperature", 0.9)
        self.default_max_tokens = int(cfg.get("default_max_tokens", 81920))
        self.default_reasoning_effort = cfg.get("default_reasoning_effort", "high")
        self.embedding_url = os.environ.get(
            "SYSTEM_LLM_EMBEDDING_URL",
            cfg.get("embedding_url") or (self.api_url.rsplit("/chat/completions", 1)[0]
                                         + "/embeddings") if self.api_url else None)
        self.embedding_model = os.environ.get(
            "SYSTEM_LLM_EMBEDDING_MODEL", cfg.get("embedding_model", "text-embedding-3-small"))
        self.embedding_api_key = os.environ.get(
            "SYSTEM_LLM_EMBEDDING_API_KEY", cfg.get("embedding_api_key", "") or "")
        self.embedding_qpm = int(cfg.get("embedding_qpm", 0) or 0)
        self._embed_broken = False
        self.supported_sampling_params = set(
            cfg.get("supported_sampling_params",
                    ["temperature", "top_p", "reasoning_effort", "max_completion_tokens"])
        )
        self.no_llm = os.environ.get("SYSTEM_NO_LLM", "").lower() in ("1", "true", "yes")
        self.name = name
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        _CLIENTS.append(self)

        self._counter_lock = threading.Lock()
        self.limiter = get_rate_limiter(int(cfg.get("qpm", 0) or 0))

    def complete(self, system_prompt: str, user_prompt: str, *,
                 temperature: float | None = None,
                 max_tokens: int | None = None,
                 reasoning_effort: str | None = None) -> str | None:
        if self.no_llm:
            logger.info(f"[{self.name}] SYSTEM_NO_LLM=1; skipping model requests and using fallback behavior")
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        sampling_full = {
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_completion_tokens": max_tokens or self.default_max_tokens,
            "reasoning_effort": reasoning_effort or self.default_reasoning_effort,
        }
        sampling = {k: v for k, v in sampling_full.items()
                    if k in self.supported_sampling_params}
        tried_unsupported: set[str] = set()
        for attempt in range(self.max_retries):
            cur = {k: v for k, v in sampling.items() if k not in tried_unsupported}
            body = dict(payload)
            body.update(cur)
            try:
                waited = self.limiter.acquire()
                if waited >= 1.0:
                    logger.info(f"[{self.name}] QPM rate-limit wait {waited:.1f}s "
                                f"(quota {self.limiter.qpm}/minute, shared by all agents)")
                logger.info(f"[{self.name}] Requesting LLM: url={self.api_url} model={self.model} "
                             f"(attempt {attempt + 1}/{self.max_retries})")
                resp = requests.post(self.api_url, headers=headers,
                                     json=body, timeout=(self.connect_timeout, self.timeout))
                if resp.status_code in (401, 403):
                    logger.error(f"[{self.name}] Authentication failed (HTTP {resp.status_code}): "
                                 f"invalid or unauthorized api_key; check SYSTEM_LLM_API_KEY / config/llm.yaml.")
                    return None
                if resp.status_code in (429, 500, 502, 503, 504):
                    self._sleep(attempt)
                    continue
                if resp.status_code == 400:
                    detail = (resp.text or "")[:600]
                    logger.warning(f"[{self.name}] HTTP 400 response details: {detail}")
                    low = detail.lower()
                    for p in ("reasoning_effort", "max_completion_tokens", "temperature", "top_p"):
                        if p in low and p in sampling and p not in tried_unsupported:
                            tried_unsupported.add(p)
                            logger.info(f"[{self.name}] Unsupported parameter {p}; removing it before retry")
                            break
                    if tried_unsupported:
                        continue
                    logger.error(f"[{self.name}] HTTP 400 cannot be repaired automatically; stopping retries")
                    return None
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                content = (message.get("content") or "")
                if not content.strip() and message.get("reasoning_content"):
                    content = message.get("reasoning_content") or ""
                usage = data.get("usage") or {}
                with self._counter_lock:
                    self.calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                if SHOW_LLM:
                    logger.info(f"[{self.name}] <<< message fields: {list(message.keys())}")
                    logger.info(f"[{self.name}] <<< LLM response content:\n{content}")
                if not content.strip():
                    logger.warning(
                        f"[{self.name}] Empty model response; raw response excerpt: {str(data)[:800]}"
                    )
                return content
            except requests.exceptions.Timeout:
                logger.warning(f"[{self.name}] Timeout (attempt {attempt + 1})")
                self._sleep(attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{self.name}] Request failed: {e}")
                self._sleep(attempt)
            except (KeyError, IndexError, ValueError) as e:
                logger.warning(f"[{self.name}] Response parsing failed: {e}")
                self._sleep(attempt)
        logger.error(f"[{self.name}] Failed after {self.max_retries} attempts")
        return None

    def complete_json(self, system_prompt: str, user_prompt: str, *,
                      temperature: float | None = None,
                      max_tokens: int | None = None,
                      reasoning_effort: str | None = None) -> dict | list | None:
        from ..utils import extract_json
        text = self.complete(system_prompt, user_prompt,
                             temperature=temperature, max_tokens=max_tokens,
                             reasoning_effort=reasoning_effort)
        return extract_json(text)

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        if self.no_llm:
            logger.info(f"[{self.name}] SYSTEM_NO_LLM=1; skipping embedding requests")
            return None
        if self._embed_broken:
            return None
        if not self.embedding_url:
            logger.warning(f"[{self.name}] embedding_url is not configured; falling back from embedding retrieval")
            self._embed_broken = True
            return None
        key = self.embedding_api_key or self.api_key
        if not key or key == "YOUR_KEY_HERE":
            logger.warning(f"[{self.name}] api_key / embedding_api_key is not configured; "
                           f"falling back to token-overlap retrieval")
            self._embed_broken = True
            return None
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body = {"model": self.embedding_model, "input": texts}
        try:
            if self.embedding_qpm and self.embedding_qpm > 0:
                limiter = get_rate_limiter(self.embedding_qpm)
            else:
                limiter = self.limiter
            waited = limiter.acquire()
            if waited >= 1.0:
                logger.info(f"[{self.name}] Embedding QPM rate-limit wait {waited:.1f}s")
            logger.info(f"[{self.name}] Requesting embeddings: url={self.embedding_url} "
                        f"model={self.embedding_model} texts={len(texts)}")
            resp = requests.post(self.embedding_url, headers=headers, json=body,
                                 timeout=(self.connect_timeout, self.timeout))
            if resp.status_code in (401, 403):
                logger.error(f"[{self.name}] Embedding authentication failed (HTTP {resp.status_code}): "
                             f"invalid or unauthorized embedding_api_key; disabling embeddings and using token overlap")
                self._embed_broken = True
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"[{self.name}] Embedding HTTP {resp.status_code} (rate limit/server error); "
                               f"using token overlap for this request")
                return None
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            vectors = [i.get("embedding", []) for i in items]
            if not vectors:
                logger.warning(f"[{self.name}] Empty embedding vectors; disabling embeddings and using token overlap")
                self._embed_broken = True
                return None
            usage = data.get("usage") or {}
            with self._counter_lock:
                self.calls += 1
                self.prompt_tokens += int(usage.get("prompt_tokens")
                                          or usage.get("total_tokens") or 0)
            if SHOW_LLM:
                logger.info(f"[{self.name}] <<< embedding dimension: {len(vectors[0]) if vectors else 0}")
            return vectors
        except requests.exceptions.RequestException as e:
            logger.warning(f"[{self.name}] Embedding request failed: {e} (using token-overlap retrieval)")
            if "404" in str(e):
                self._embed_broken = True
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"[{self.name}] Embedding response parsing failed: {e} (using token-overlap retrieval)")
        return None

    def _sleep(self, attempt: int):
        delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
        time.sleep(delay)
