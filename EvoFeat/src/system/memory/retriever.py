from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..llm.client import LLMClient
from ..paths import MEMORY_DIR
from ..utils import ensure_dir, get_logger


logger = get_logger("retriever")
DEFAULT_EMBED_CACHE = MEMORY_DIR / "embeddings.json"


@dataclass
class Retriever:
    llm: LLMClient | None = None

    def retrieve(self, query: str, items: list, k: int = 4,
                 key: str = "feature_desc") -> list:
        if not items or k <= 0:
            return []
        scored = [(self._score(query, it, key), it) for it in items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def _score(self, query: str, item, key: str) -> float:
        raise NotImplementedError

    @staticmethod
    def _text_of(item, key: str) -> str:
        if isinstance(item, dict):
            return str(item.get(key, ""))
        return str(getattr(item, key, str(item)))


class TokenRetriever(Retriever):
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split("_")), set(b.lower().split("_"))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _score(self, query: str, item, key: str) -> float:
        return self._jaccard(query, self._text_of(item, key))

    def warm_items(self, items: list, key: str = "feature_desc") -> int:
        return 0


class EmbeddingRetriever(Retriever):
    BATCH = 50

    def __init__(self, llm: LLMClient | None = None,
                 cache_path: str | Path | None = None):
        super().__init__(llm)
        self.cache_path = Path(cache_path or DEFAULT_EMBED_CACHE)
        self._cache: dict[str, list[float]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            vectors = data.get("vectors", {})
            if (not isinstance(vectors, dict)
                    or any(not self._valid_vector(v) for v in vectors.values())
                    or len({len(v) for v in vectors.values()}) > 1):
                logger.warning("Embedding cache contains invalid vectors; recomputing")
                return
            self._cache = vectors
            logger.info(f"Embedding cache: loaded {len(self._cache)} entries -> {self.cache_path}")
        except Exception as e:
            logger.warning(f"Embedding cache load failed ({e}); starting with an empty cache")

    def _save_cache(self) -> None:
        try:
            ensure_dir(self.cache_path.parent)
            self.cache_path.write_text(
                json.dumps({"vectors": self._cache}, ensure_ascii=False),
                encoding="utf-8")
        except Exception as e:
            logger.warning(f"Embedding cache save failed: {e}")

    @staticmethod
    def _valid_vector(vector) -> bool:
        try:
            return (isinstance(vector, list) and bool(vector)
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            and math.isfinite(x) for x in vector)
                    and any(x != 0 for x in vector))
        except (TypeError, OverflowError):
            return False

    def _vectors(self, texts: list[str]) -> list[list[float]] | None:
        fresh = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if fresh:
            if self.llm is None:
                return None
            for i in range(0, len(fresh), self.BATCH):
                batch = fresh[i:i + self.BATCH]
                try:
                    vecs = self.llm.embed(batch)
                except Exception as exc:
                    logger.warning(f"Embedding request failed ({exc}); retrieval unavailable for this request")
                    return None
                if vecs is None:
                    return None
                if (not isinstance(vecs, list) or len(vecs) != len(batch)
                        or any(not self._valid_vector(v) for v in vecs)
                        or len({len(v) for v in vecs}) != 1):
                    logger.warning("Embedding response is incomplete or invalid; no vectors from this batch cached")
                    return None
                cached_dim = len(next(iter(self._cache.values()))) if self._cache else None
                if cached_dim is not None and len(vecs[0]) != cached_dim:
                    self._cache.clear()
                    logger.warning("Embedding dimension changed; clearing incompatible vectors and using fallback")
                    return None
                for t, v in zip(batch, vecs):
                    self._cache[t] = v
            self._save_cache()
        return [self._cache[t] for t in texts]

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(y * y for y in b)) or 1e-9
        return dot / (na * nb)

    def retrieve(self, query: str, items: list, k: int = 4,
                 key: str = "feature_desc") -> list:
        if not items or k <= 0:
            return []
        texts = [query] + [self._text_of(it, key) for it in items]
        vecs = self._vectors(texts)
        if vecs is None:
            raise RuntimeError("Embeddings unavailable")
        qv = vecs[0]
        scored = [(self._cos(qv, vecs[i + 1]), it) for i, it in enumerate(items)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def _score(self, query: str, item, key: str) -> float:
        t = self._text_of(item, key)
        vecs = self._vectors([query, t])
        if vecs is None:
            raise RuntimeError("Embeddings unavailable")
        return self._cos(vecs[0], vecs[1])

    def warm_items(self, items: list, key: str = "feature_desc") -> int:
        texts = [self._text_of(it, key) for it in items if self._text_of(it, key)]
        if not texts:
            return 0
        fresh = [t for t in texts if t not in self._cache]
        if not fresh:
            return 0
        vecs = self._vectors(fresh)
        if vecs is None:
            return 0
        logger.info(f"[retrieval] Embedding warm-up: cached {len(fresh)} new entries "
                    f"(total {len(self._cache)}) -> {self.cache_path.name}")
        return len(fresh)


class HybridRetriever(Retriever):
    def __init__(self, llm: LLMClient | None = None,
                 cache_path: str | Path | None = None):
        super().__init__(llm)
        self._emb = EmbeddingRetriever(llm, cache_path=cache_path)
        self._tok = TokenRetriever(llm)

    def _embed_broken(self) -> bool:
        if self.llm is None:
            return True
        return bool(getattr(self.llm, "_embed_broken", False))

    def retrieve(self, query: str, items: list, k: int = 4,
                 key: str = "feature_desc") -> list:
        if not items or k <= 0:
            return []
        if self._embed_broken():
            return self._tok.retrieve(query, items, k=k, key=key)
        try:
            return self._emb.retrieve(query, items, k=k, key=key)
        except RuntimeError:
            if not getattr(self, "_warned", False):
                logger.warning("Embedding retrieval unavailable; falling back to token overlap")
                self._warned = True
            return self._tok.retrieve(query, items, k=k, key=key)

    def _score(self, query: str, item, key: str) -> float:
        try:
            return self._emb._score(query, item, key)
        except RuntimeError:
            if not getattr(self, "_warned", False):
                logger.warning("Embedding retrieval unavailable; falling back to token overlap")
                self._warned = True
            return self._tok._score(query, item, key)

    def warm_items(self, items: list, key: str = "feature_desc") -> int:
        if self._embed_broken():
            return 0
        try:
            return self._emb.warm_items(items, key=key)
        except RuntimeError:
            return 0


def make_retriever(mode: str = "hybrid", llm: LLMClient | None = None) -> Retriever:
    mode = (mode or "hybrid").lower()
    if mode == "embedding":
        return EmbeddingRetriever(llm)
    if mode == "token":
        return TokenRetriever(llm)
    return HybridRetriever(llm)
