from __future__ import annotations

import json
from pathlib import Path

from ..core.graph import Subgraph
from ..utils import get_logger


logger = get_logger("loader")
DATA_DIR = Path(__file__).resolve().parent


WORKSPACE_ROOT = DATA_DIR.parent.parent
SAMPLE_PATH = WORKSPACE_ROOT / "example.json"
LEGACY_PATH = DATA_DIR / "sample_subgraphs.json"


def _default_path() -> Path:
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH
    return LEGACY_PATH


class SubgraphStream:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _default_path()
        self._raw = self._load()
        self._idx = 0

    def _load(self) -> list[dict]:
        if not self.path.exists():
            logger.warning(f"Subgraph file not found: {self.path}; returning an empty stream")
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("subgraphs", [])

    def __len__(self):
        return len(self._raw)

    def next(self) -> Subgraph | None:
        if self._idx >= len(self._raw):
            return None
        d = self._raw[self._idx]
        self._idx += 1
        g = Subgraph.from_dict(d)
        logger.info(f"Loaded subgraph {g.subgraph_id} (center={g.center}, "
                    f"nodes={len(g.nodes)}, edges={len(g.edges)})")
        return g
