from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence, TypeVar

from .utils import begin_log_capture, end_log_capture, flush_log_records, get_logger


logger = get_logger("parallel")
T = TypeVar("T")
R = TypeVar("R")


def run_parallel(items: Sequence[T], worker: Callable[[T, int], R], *,
                 max_workers: int = 1, stage: str = "",
                 ordered_log: bool = True) -> list[R | None]:
    n = len(items)
    if n == 0:
        return []
    workers = max(1, min(int(max_workers), n))
    if workers == 1:
        return [_safe_call(worker, item, i, stage) for i, item in enumerate(items)]

    logger.info(f"[parallel] {stage}: starting {n} tasks with {workers} workers "
                f"(logs buffered per task and printed after completion)")
    t0 = time.monotonic()

    results: list[Any] = [None] * n
    buffers: list[list] = [[] for _ in range(n)]
    errors: list[BaseException | None] = [None] * n

    def _run(index: int, item: T):
        token = begin_log_capture(buffers[index])
        try:
            return worker(item, index)
        finally:
            end_log_capture(token)

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="llm") as pool:
        futures = {pool.submit(_run, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except BaseException as e:
                errors[i] = e
    if ordered_log:
        for i, buf in enumerate(buffers):
            if buf:
                logger.info(f"[parallel] {stage}: task {i + 1}/{n} logs:")
                flush_log_records(buf)

    for i, err in enumerate(errors):
        if err is not None:
            logger.error(f"[parallel] {stage}: task {i + 1}/{n} failed: {err}")
    ok = sum(1 for r in results if r is not None)
    elapsed = time.monotonic() - t0
    logger.info(f"[parallel] {stage}: {ok}/{n} tasks succeeded, elapsed {elapsed:.1f}s")
    return results


def _safe_call(worker: Callable[[T, int], R], item: T, index: int,
               stage: str) -> R | None:
    try:
        return worker(item, index)
    except BaseException as e:
        logger.error(f"[serial] {stage}: task {index + 1} failed: {e}")
        return None
