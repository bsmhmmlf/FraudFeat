from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..utils import get_logger


logger = get_logger("execution.eg")
_QUERY_COUNT = 0
_QUERY_LOCK = threading.Lock()


def get_query_count() -> int:
    return _QUERY_COUNT


_TRANSIENT_ERR_MARKERS = (
    "TRANSPORT_EXCEPTION", "Connection", "connection reset",
    "UNAVAILABLE", "Broken pipe", "EOFException",
    "no active replica", "COMMON_EMPTY", "not ready (warming up",
)
SLOW_QUERY_MARKERS = (
    "SLOW_QUERY", "timed out", "Timeout", "timeout", "TimeoutException",
    "Read timed out", "request timeout",
)


_PROBE_TEMPLATE = "{g}.V().hasId(s).count()"
_PROBE_FALLBACK_ID = "__eg_probe_nonexistent__|entity"


def is_transient_exec_error(e: Exception) -> bool:
    s = f"{type(e).__name__}: {e}"
    return any(m in s for m in _TRANSIENT_ERR_MARKERS)


def is_slow_query_error(e: Exception) -> bool:
    s = f"{type(e).__name__}: {e}"
    return any(m in s for m in SLOW_QUERY_MARKERS)


def _to_python(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_to_python(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_python(x) for k, x in v.items()}
    tn = type(v).__name__
    if "Decimal" in tn:
        try:
            return int(v)
        except Exception:
            try:
                return float(v)
            except Exception:
                return str(v)
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            pass
    if hasattr(v, "value"):
        return _to_python(v.value)
    return str(v)


def _parse_servers(raw: str) -> List[Tuple[str, int]]:
    servers: List[Tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"Invalid server address: {item!r}; expected host:port")
        servers.append((host, int(port)))
    return servers


class GremlinExecutor:
    def __init__(self, space: str, *, username: Optional[str] = None,
                 password: Optional[str] = None,
                 polaris_ns: str = "<YOUR_POLARIS_NAMESPACE>",
                 polaris_svc: str = "<YOUR_POLARIS_SERVICE>",
                 fixed_servers: str = ""):
        self.space = space
        self.g_prefix = f"g_{space}"
        self.username = username if username is not None else os.environ.get("EG_USER", "<YOUR_GRAPH_USER>")
        self.password = password if password is not None else os.environ.get("EG_PASSWORD", "<YOUR_GRAPH_PASSWORD>")
        self.polaris_ns = polaris_ns
        self.polaris_svc = polaris_svc
        self.fixed_servers = fixed_servers
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from easygraph_client_commons import ClientConfig
            from easygraph_v3_client import EasyGraphClient
        except Exception as e:
            raise RuntimeError(
                "easygraph-v3-client-python is unavailable or the graph environment is not configured: "
                f"{type(e).__name__}: {e}"
            )

        def _apply_recommended(cfg):
            cfg.pool_max = 8
            cfg.max_retries = 6
            cfg.script_max_retries = 1
            if hasattr(cfg, "request_timeout_ms"):
                cfg.request_timeout_ms = 60000
            if hasattr(cfg, "script_timeout_ms"):
                cfg.script_timeout_ms = 60000
            cfg.base_retry_delay_ms = 500
            cfg.max_retry_delay_ms = 5000
            cfg.cluster_replica_count = 3
            cfg.health_check_enabled = True
            return cfg

        env_servers = os.environ.get("EASYGRAPH_SERVERS", "").strip()
        use_fixed = os.environ.get(
            "EG_USE_FIXED_SERVERS", "0").strip().lower() not in ("0", "false", "no", "off", "")

        if env_servers or (use_fixed and self.fixed_servers):
            servers = _parse_servers(env_servers or self.fixed_servers)
            src = "EASYGRAPH_SERVERS" if env_servers else "fixed servers"
            logger.info(f"[EasyGraph] Direct connection via {src}: {servers} (user={self.username})")
            cfg = ClientConfig(servers=servers,
                               username=self.username, password=self.password)
        else:
            try:
                from easygraph_v3_client.discovery.polaris import (
                    PolarisDiscoveryConfig, PolarisServiceDiscovery,
                )
            except Exception as e:
                raise RuntimeError(
                    "Polaris discovery is unavailable (set EASYGRAPH_SERVERS=host:port for a direct connection): "
                    f"{type(e).__name__}: {e}"
                )
            sd = PolarisServiceDiscovery(
                PolarisDiscoveryConfig(namespace=self.polaris_ns, service=self.polaris_svc))
            logger.info(f"[EasyGraph] Polaris discovery ns={self.polaris_ns} "
                        f"svc={self.polaris_svc} (user={self.username})")
            cfg = ClientConfig(service_discovery=sd,
                               username=self.username, password=self.password)
        self._client = EasyGraphClient(_apply_recommended(cfg))
        logger.info(f"[EasyGraph] Connected user={self.username} "
                    f"max_retries={cfg.max_retries} script_max_retries={cfg.script_max_retries}")

    def submit(self, script: str, bindings: Optional[Dict[str, Any]] = None) -> List[Any]:
        global _QUERY_COUNT
        self._ensure_client()
        assert self._client is not None
        t0 = time.time()
        res = self._client.submit_script(script, bindings or {})
        with _QUERY_LOCK:
            _QUERY_COUNT += 1
        dt = time.time() - t0
        if res is None:
            rows: List[Any] = []
        elif isinstance(res, list):
            rows = [_to_python(x) for x in res]
        else:
            rows = [_to_python(res)]
        logger.info(f"    [exec] {dt:.2f}s rows={len(rows)} | {script[:220]}")
        return rows

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass

    def reset(self) -> None:
        self.close()
        self._client = None


def preflight(executor: GremlinExecutor, seed_vid: str = "",
              retries: int = 3) -> bool:
    if getattr(executor, "backend", "") == "fraudfeat-parquet":
        return bool(executor.preflight())

    probe = _PROBE_TEMPLATE.format(g=executor.g_prefix)
    anchor = seed_vid if str(seed_vid or "").strip() else _PROBE_FALLBACK_ID
    from .eg_plan import vid_to_token
    anchor = vid_to_token(anchor)
    last_err: Optional[Exception] = None
    for try_i in range(retries + 1):
        try:
            executor.submit(probe, {"s": anchor})
            return True
        except Exception as e:
            if not is_transient_exec_error(e):
                return True
            last_err = e
            if try_i < retries:
                logger.info(f"  [preflight] Probe failed ({str(e)[:140]}); retry {try_i + 1}/{retries}")
                try:
                    executor.reset()
                except Exception:
                    pass
    logger.warning(f"  [preflight] Failed after {retries} retries; falling back to no-exec; "
                   f"probe: {probe} (anchor={anchor!r}, last error: {last_err})")
    return False


def build_executor_from_config(enabled: bool = False) -> Optional[GremlinExecutor]:
    import os as _os
    if _os.environ.get("SYSTEM_EG_ENABLED", "").lower() in ("1", "true", "yes"):
        enabled = True
    if not enabled:
        return None
    try:
        from ..utils import load_config
        cfg = load_config("execution.yaml")
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    space = _os.environ.get("SYSTEM_EG_SPACE", cfg.get("space", "<YOUR_GRAPH_SPACE>"))
    try:
        ex = GremlinExecutor(
            space=space,
            username=cfg.get("username") or None,
            password=cfg.get("password") or None,
            polaris_ns=cfg.get("polaris_ns", "<YOUR_POLARIS_NAMESPACE>"),
            polaris_svc=cfg.get("polaris_svc", "<YOUR_POLARIS_SERVICE>"),
            fixed_servers=cfg.get("fixed_servers", ""),
        )
    except Exception as e:
        logger.warning(f"[EasyGraph] Executor initialization failed ({type(e).__name__}: {e}); falling back to no-exec")
        return None
    return ex
