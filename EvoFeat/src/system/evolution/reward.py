from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..core.schema import Candidate
from .true_iv import compute_true_iv
from ..utils import get_logger, load_config


logger = get_logger("reward")
DEFAULT_MAX_FEATURE_CONCURRENCY = 12


def _feature_concurrency(n_cands: int) -> int:
    try:
        cap = int(load_config("iv.yaml").get(
            "max_feature_concurrency", DEFAULT_MAX_FEATURE_CONCURRENCY)
            or DEFAULT_MAX_FEATURE_CONCURRENCY)
    except Exception:
        cap = DEFAULT_MAX_FEATURE_CONCURRENCY
    return max(1, min(int(n_cands), max(1, cap)))


def _eval_one(c: Candidate, pool: list[Candidate], executor=None) -> None:
    try:
        compute_true_iv(c, pool, executor=executor)
    except Exception as e:
        logger.warning(f"ComputeTrueIV failed ({c.feature_name}): {e}; "
                       f"true IV remains unset; excluded from grounding/memory/summarization this round")


def evaluate_batch(candidates: list[Candidate], pool: list[Candidate], executor=None) -> None:
    cands = list(candidates or [])
    if not cands:
        return
    workers = _feature_concurrency(len(cands))
    if workers <= 1:
        for c in cands:
            _eval_one(c, pool, executor=executor)
        return
    logger.info(f"[IV computation] Feature-level concurrency: {workers} workers ({len(cands)} candidates; "
                f"continuing after the entire batch completes)")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iv-feat") as ex:
        futs = [ex.submit(_eval_one, c, pool, executor) for c in cands]
        for f in futs:
            try:
                f.result()
            except Exception as e:
                logger.warning(f"[IV computation] Concurrent task failed: {type(e).__name__}: {e}")
