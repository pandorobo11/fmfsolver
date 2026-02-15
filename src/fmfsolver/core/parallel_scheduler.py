from __future__ import annotations

"""Cache-aware parallel scheduling helpers for case execution.

This module exists to keep :mod:`fmfsolver.core.solver` focused on the
physics/IO pipeline while isolating multiprocessing + scheduling policy.
"""

import multiprocessing as mp
import os
import queue
import traceback
from collections import OrderedDict, deque
from typing import Callable, Iterator

import pandas as pd


def resolve_parallel_chunk_cases() -> int:
    """Return default per-task case chunk size for parallel scheduling."""
    raw = os.getenv("FMFSOLVER_PARALLEL_CHUNK_CASES", "").strip()
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("FMFSOLVER_PARALLEL_CHUNK_CASES must be an integer >= 1.") from exc
    if value < 1:
        raise ValueError("FMFSOLVER_PARALLEL_CHUNK_CASES must be >= 1.")
    return value


def _parallel_bucket_key(row: pd.Series, index: int) -> tuple:
    """Return bucket key for parallel shielding-cache reuse scheduling.

    Shielding-enabled cases are grouped by mesh identity and flow direction.
    Non-shielding cases are treated as unique buckets to keep them freely
    schedulable for load balancing.
    """
    try:
        shielding_on = bool(int(row.get("shielding_on", 0)))
    except Exception:
        shielding_on = False
    if not shielding_on:
        return ("single", index)

    stl_paths = tuple(p.strip() for p in str(row.get("stl_path", "")).split(";") if p.strip())
    scale = round(float(row.get("stl_scale_m_per_unit", 1.0)), 12)
    alpha = round(float(row.get("alpha_deg", 0.0)), 12)
    beta = round(float(row.get("beta_deg", 0.0)), 12)
    ray_backend = str(row.get("ray_backend", "auto")).strip().lower() or "auto"
    return ("shield", stl_paths, scale, alpha, beta, ray_backend)


def _build_bucket_chunks(
    df: pd.DataFrame, exec_order: list[int], chunk_cases: int
) -> tuple[dict[tuple, deque[list[int]]], dict[tuple, int]]:
    """Group case indices into reusable buckets, then split into chunks."""
    buckets: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for i in exec_order:
        key = _parallel_bucket_key(df.iloc[i], i)
        buckets.setdefault(key, []).append(i)

    bucket_chunks: dict[tuple, deque[list[int]]] = {}
    bucket_remaining: dict[tuple, int] = {}
    for key, indices in buckets.items():
        dq: deque[list[int]] = deque()
        for start in range(0, len(indices), chunk_cases):
            dq.append(indices[start : start + chunk_cases])
        bucket_chunks[key] = dq
        bucket_remaining[key] = len(indices)
    return bucket_chunks, bucket_remaining


def _pick_next_chunk(
    worker_id: int,
    worker_last_bucket: list[tuple | None],
    bucket_chunks: dict[tuple, deque[list[int]]],
    bucket_remaining: dict[tuple, int],
    bucket_owner: dict[tuple, int | None],
) -> tuple[tuple, list[int]] | None:
    """Choose next chunk for one worker, preferring cache reuse and bucket isolation."""
    last = worker_last_bucket[worker_id]
    if last is not None:
        dq = bucket_chunks.get(last)
        if dq:
            chunk = dq.popleft()
            bucket_remaining[last] -= len(chunk)
            if bucket_remaining[last] <= 0:
                bucket_chunks.pop(last, None)
                bucket_remaining.pop(last, None)
                bucket_owner.pop(last, None)
            return last, chunk

    unowned = [b for b, dq in bucket_chunks.items() if dq and bucket_owner.get(b) is None]
    if unowned:
        # Prefer buckets with the most remaining work to reduce skew.
        b = max(unowned, key=lambda k: bucket_remaining.get(k, 0))
        bucket_owner[b] = worker_id
        worker_last_bucket[worker_id] = b
        dq = bucket_chunks[b]
        chunk = dq.popleft()
        bucket_remaining[b] -= len(chunk)
        if bucket_remaining[b] <= 0:
            bucket_chunks.pop(b, None)
            bucket_remaining.pop(b, None)
            bucket_owner.pop(b, None)
        return b, chunk

    if bucket_chunks:
        # Only touched buckets remain; accept overlap to keep workers busy.
        b = max(bucket_chunks.keys(), key=lambda k: bucket_remaining.get(k, 0))
        worker_last_bucket[worker_id] = b
        dq = bucket_chunks[b]
        chunk = dq.popleft()
        bucket_remaining[b] -= len(chunk)
        if bucket_remaining[b] <= 0:
            bucket_chunks.pop(b, None)
            bucket_remaining.pop(b, None)
            bucket_owner.pop(b, None)
        return b, chunk

    return None


def _null_log(_msg: str):
    """No-op logger used in worker processes."""
    return None


def _worker_loop(worker_id: int, task_q, result_q, cancel_event, run_case_fn) -> None:
    """Worker process loop for cache-aware chunk execution."""
    while True:
        msg = task_q.get()
        mtype = msg.get("type")
        if mtype == "shutdown":
            return
        if mtype != "run_chunk":
            result_q.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "error": f"Unknown task type: {mtype}",
                    "traceback": "",
                }
            )
            return

        bucket_id = msg.get("bucket_id")
        indices = list(msg.get("indices") or [])
        rows = list(msg.get("rows") or [])
        if len(indices) != len(rows):
            result_q.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "bucket_id": bucket_id,
                    "error": "Task indices/rows size mismatch.",
                    "traceback": "",
                }
            )
            return

        results: list[tuple[int, dict]] = []
        try:
            for i, row in zip(indices, rows):
                if cancel_event.is_set():
                    break
                results.append((int(i), run_case_fn(row, _null_log)))
        except Exception as exc:
            result_q.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "bucket_id": bucket_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return

        result_q.put(
            {
                "type": "chunk_done",
                "worker_id": worker_id,
                "bucket_id": bucket_id,
                "canceled": bool(cancel_event.is_set()),
                "results": results,
            }
        )


def iter_case_results_parallel(
    df: pd.DataFrame,
    exec_order: list[int],
    workers: int,
    run_case_fn: Callable[[dict, Callable[[str], None]], dict],
    *,
    chunk_cases: int | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, dict]]:
    """Yield ``(case_index, case_result)`` as cases complete in parallel.

    This function preserves input ordering only via the case_index returned;
    the caller is responsible for assembling final rows in the desired order.
    """
    total = int(len(df))
    if total <= 0:
        return
    if workers <= 1 or total <= 1:
        raise ValueError("iter_case_results_parallel requires workers>=2 and total>=2.")

    if chunk_cases is None:
        chunk_cases = resolve_parallel_chunk_cases()

    bucket_chunks, bucket_remaining = _build_bucket_chunks(df, exec_order, int(chunk_cases))
    bucket_owner: dict[tuple, int | None] = {b: None for b in bucket_chunks.keys()}
    worker_last_bucket: list[tuple | None] = [None for _ in range(workers)]

    ctx = mp.get_context("spawn")
    cancel_event = ctx.Event()
    task_queues = [ctx.Queue(maxsize=1) for _ in range(workers)]
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker_loop,
            args=(wid, task_queues[wid], result_queue, cancel_event, run_case_fn),
            daemon=True,
        )
        for wid in range(workers)
    ]
    for p in procs:
        p.start()

    worker_busy = [False] * workers

    def _assign_next(wid: int) -> bool:
        nonlocal worker_busy
        picked = _pick_next_chunk(
            worker_id=wid,
            worker_last_bucket=worker_last_bucket,
            bucket_chunks=bucket_chunks,
            bucket_remaining=bucket_remaining,
            bucket_owner=bucket_owner,
        )
        if picked is None:
            return False
        bucket_id, indices = picked
        rows = [df.iloc[i].to_dict() for i in indices]
        task_queues[wid].put(
            {"type": "run_chunk", "bucket_id": bucket_id, "indices": indices, "rows": rows}
        )
        worker_busy[wid] = True
        return True

    done_count = 0
    try:
        # Prime workers.
        for wid in range(workers):
            _assign_next(wid)

        while done_count < total:
            if cancel_cb is not None and cancel_cb():
                cancel_event.set()
                raise RuntimeError("Canceled by user.")

            try:
                msg = result_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            mtype = msg.get("type")
            if mtype == "error":
                cancel_event.set()
                fatal_error = str(msg.get("error") or "Unknown worker error.")
                fatal_tb = str(msg.get("traceback") or "")
                if fatal_tb:
                    raise RuntimeError(f"[WorkerError] {fatal_error}\n{fatal_tb}")
                raise RuntimeError(f"[WorkerError] {fatal_error}")

            if mtype != "chunk_done":
                cancel_event.set()
                raise RuntimeError(f"Unknown worker message type: {mtype}")

            wid = int(msg.get("worker_id", -1))
            worker_busy[wid] = False
            bucket_id = msg.get("bucket_id")
            if bucket_id is not None:
                worker_last_bucket[wid] = bucket_id

            if bool(msg.get("canceled")):
                cancel_event.set()
                raise RuntimeError("Canceled by user.")

            results = list(msg.get("results") or [])
            for i, case_result in results:
                done_count += 1
                yield int(i), case_result

            # Assign another chunk to this worker if work remains.
            if done_count < total:
                _assign_next(wid)
    finally:
        # Best-effort shutdown/cleanup. Setting the event helps workers exit early if
        # the caller stops consuming (error/cancel).
        cancel_event.set()
        for q in task_queues:
            try:
                q.put({"type": "shutdown"}, block=False)
            except Exception:
                pass
        for p in procs:
            p.join(timeout=2.0)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

