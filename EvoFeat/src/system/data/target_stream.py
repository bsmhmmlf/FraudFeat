from __future__ import annotations

import hashlib
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.graph import Subgraph
from ..paths import PROJECT_ROOT
from ..utils import ensure_dir, get_logger


logger = get_logger("data.target_stream")
TABLES: Dict[str, Tuple[str, str]] = {
    "individual": (os.environ.get("TARGET_TABLE_INDIVIDUAL", "<YOUR_INDIVIDUAL_TARGET_TABLE>"),
                   "unique_id"),
}


class StarRocksClient:
    def __init__(self, cfg: dict):
        self._cfg = dict(cfg or {})
        self._conn = None

    def _connect(self):
        if self._conn is not None:
            return self._conn
        try:
            import pymysql
        except ImportError as e:
            raise RuntimeError("Missing dependency pymysql (pip install pymysql)") from e
        cfg = self._cfg
        user = os.environ.get("STARROCKS_USER") or cfg.get("user") or ""
        password = os.environ.get("STARROCKS_PASSWORD") or cfg.get("password") or ""
        self._conn = pymysql.connect(
            host=cfg.get("host", "127.0.0.1"),
            port=int(cfg.get("port", 9030)),
            user=user, password=password,
            database=cfg.get("database", ""),
            charset=cfg.get("charset", "utf8mb4"),
            connect_timeout=int(cfg.get("connect_timeout", 10)),
            read_timeout=int(cfg.get("read_timeout", 60)),
            cursorclass=pymysql.cursors.DictCursor,
        )
        logger.info(f"[target stream] StarRocks connected {cfg.get('host')}:{cfg.get('port')} "
                    f"db={cfg.get('database')} user={user}")
        return self._conn

    def _has_column(self, table: str, col: str) -> bool:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME=%s",
                (self._cfg.get("database", ""), table, col))
            row = cur.fetchone()
        return bool(row and int(row.get("c", 0)) > 0)

    def fetch_random_seed_ids(self, table: str, key_col: str, n: int, seed: int) -> List[str]:
        conn = self._connect()
        key = ("unique_id" if self._has_column(table, "unique_id")
               else "target_id" if self._has_column(table, "target_id") else key_col)
        label_filter = "label = 1 AND " if self._has_column(table, "label") else ""
        sql = (f"SELECT DISTINCT `{key}` AS seed_key, "
               f"md5(concat(`{key}`, ':{int(seed)}')) AS _h FROM `{table}` "
               f"WHERE {label_filter}`{key}` IS NOT NULL AND `{key}` <> '' "
               f"ORDER BY _h LIMIT %s")
        with conn.cursor() as cur:
            cur.execute(sql, (int(n),))
            rows = cur.fetchall()
        return [str(r["seed_key"]) for r in rows if r.get("seed_key")]

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None


class TargetStream:
    def __init__(self, source, *, sr_cfg: dict, account_type: str = "individual",
                 sample: int = 100, seed: int = 20260825,
                 oversample: float = 1.5, max_rounds: int = 10,
                 cache_cfg: dict | None = None):
        if account_type != "individual":
            raise ValueError("Public labels are available only for wechat_pay_account")
        self._tables = [TABLES[account_type]]
        self._source = source
        self._client = StarRocksClient(sr_cfg)
        self._sample = int(sample)
        self._seed = int(seed)
        self._oversample = float(oversample)
        self._max_rounds = int(max_rounds)
        self._queue: deque = deque()
        self._used: set = set()
        self._rounds = 0
        self._produced = 0
        self._skipped = 0
        self._rng = random.Random(seed)
        cache_cfg = dict(cache_cfg or {})
        self._cache_enabled = bool(cache_cfg.get("enabled", True))
        self._cache_oversample = float(cache_cfg.get("oversample", 5) or 5)
        cache_dir = Path(cache_cfg.get("dir") or "data/target_cache")

        tables_key = hashlib.md5(
            ",".join(sorted(t for t, _ in self._tables)).encode()).hexdigest()[:8]
        self._tables_key = tables_key
        self._cache_path = (cache_dir if cache_dir.is_absolute()
                            else PROJECT_ROOT / cache_dir) / \
            f"target_ids_{account_type}_{self._seed}_{tables_key}.json"
        self._probe_min_hops = int(getattr(source, "probe_min_hops", 0) or 0)
        self._probe_cache: Dict[str, bool] = {}
        self._probe_new = 0
        self._probe_cache_path: Optional[Path] = None
        if self._probe_min_hops > 0 and getattr(source, "live", False):
            self._probe_cache_path = (cache_dir if cache_dir.is_absolute()
                                      else PROJECT_ROOT / cache_dir) / \
                f"probe_cache_{account_type}_{tables_key}_h{self._probe_min_hops}.json"
            self._probe_cache = self._load_probe_cache()
            if self._probe_cache:
                logger.info(f"[target stream] Depth-probe cache hit: {len(self._probe_cache)} entries "
                            f"({self._probe_cache_path.name})")
        self._cache_mode = False
        if self._cache_enabled:
            cached = self._load_cache()
            if cached is not None:
                self._queue = deque(cached)
                self._cache_mode = True
                logger.info(f"[target stream] Risk-positive target cache hit: {len(cached)} targets "
                            f"({self._cache_path}); skipping StarRocks")
        logger.info(f"[target stream] Positive tables={[t for t, _ in self._tables]} targets={sample} seed={seed}"
                    f" cache={'hit' if self._cache_mode else ('enabled' if self._cache_enabled else 'disabled')}")

    def next(self) -> Optional[Subgraph]:
        while self._produced < self._sample:
            if not self._queue:
                if not self._refill():
                    return None
            vid, table = self._queue.popleft()
            if vid in self._used:
                continue
            self._used.add(vid)
            sg = self._source.init_target(vid, entity_hint=table)
            if sg is None:
                self._skipped += 1
                continue
            if self._probe_cache_path is not None:
                ok = self._probe_cache.get(vid)
                if ok is None:
                    ok = bool(self._source.prepare(sg, None))
                    self._probe_cache[vid] = ok
                    self._probe_new += 1
                    if self._probe_new % 50 == 0:
                        self._save_probe_cache()
                if not ok:
                    self._skipped += 1
                    logger.info(f"[target stream] Target depth below {self._probe_min_hops} hops; "
                                f"discarding without consuming quota ({self._skipped} skipped)")
                    continue
            self._produced += 1
            return sg
        return None

    def _refill(self) -> bool:
        if self._cache_mode:
            logger.info(f"[target stream] Local cache exhausted ({self._cache_path}); "
                        f"ending without reconnecting to StarRocks ({self._produced} produced)")
            return False
        if self._rounds >= self._max_rounds:
            logger.info(f"[target stream] Maximum refill rounds {self._max_rounds} reached; stream ended")
            return False
        self._rounds += 1

        if self._cache_enabled and self._rounds == 1:
            need = max(20, int(self._sample * self._cache_oversample))
        else:
            need = max(20, int((self._sample - self._produced) * self._oversample))
        batch: List[Tuple[str, str]] = []
        first_batch: List[Tuple[str, str]] = []
        for table, key_col in self._tables:
            try:
                ids = self._client.fetch_random_seed_ids(
                    table, key_col, need, self._seed + self._rounds * 1000)
            except Exception as e:
                logger.error(f"[target stream] StarRocks sampling failed ({table}): {e}")
                continue
            fresh = [v for v in ids if v not in self._used]
            batch.extend((v, table) for v in fresh)
            first_batch.extend((v, table) for v in fresh)
        if self._cache_enabled and self._rounds == 1 and first_batch:
            self._save_cache(first_batch)
        self._rng.shuffle(batch)
        self._queue.extend(batch)
        logger.info(f"[target stream] Refill round {self._rounds}: +{len(batch)} targets "
                    f"({self._produced} produced, {self._skipped} skipped)")
        return bool(self._queue)

    def _load_cache(self) -> Optional[List[Tuple[str, str]]]:
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[target stream] Cache read failed ({e}); resampling: {self._cache_path}")
            return None
        by_table = data.get("ids") or {}
        cur_tables = {t for t, _ in self._tables}
        if set(by_table) != cur_tables:
            logger.warning(f"[target stream] Cached tables {sorted(by_table)} do not match positive tables "
                           f"{sorted(cur_tables)}; discarding cache and resampling")
            return None
        pool: List[Tuple[str, str]] = []
        for table, ids in by_table.items():
            pool.extend((str(v), table) for v in ids if v not in self._used)
        self._rng.shuffle(pool)
        logger.info(f"[target stream] Cache details: seed={data.get('seed')} sample={data.get('sample')} "
                    f"tables={list(by_table)} total={len(pool)}")
        return pool

    def _save_cache(self, batch: List[Tuple[str, str]]) -> None:
        ensure_dir(self._cache_path.parent)
        by_table: Dict[str, List[str]] = {}
        for vid, table in batch:
            by_table.setdefault(table, []).append(vid)
        self._cache_path.write_text(
            json.dumps({"seed": self._seed, "sample": self._sample,
                        "oversample": self._cache_oversample,
                        "ids": by_table},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        logger.info(f"[target stream] Cached {len(batch)} risk-positive target ids from initial StarRocks sampling "
                    f"-> {self._cache_path}")

    def _load_probe_cache(self) -> Dict[str, bool]:
        try:
            data = json.loads(self._probe_cache_path.read_text(encoding="utf-8"))
            return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_probe_cache(self) -> None:
        if self._probe_cache_path is None:
            return
        try:
            ensure_dir(self._probe_cache_path.parent)
            tmp = self._probe_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._probe_cache, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self._probe_cache_path)
        except OSError as e:
            logger.warning(f"[target stream] Probe cache write failed ({e}); targets will be probed again next time")

    def close(self) -> None:
        self._save_probe_cache()
        self._client.close()
