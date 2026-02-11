#!/usr/bin/env python3
"""Benchmark runtime and memory for the current rtree-based solver setup."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

from fmfsolver.core.solver import run_cases
from fmfsolver.io.csv_out import write_results_csv
from fmfsolver.io.io_cases import read_cases


def _parse_case_ids(values: list[str] | None) -> set[str] | None:
    """Parse ``--cases`` values into a normalized set of case IDs."""
    if not values:
        return None
    case_ids: set[str] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                case_ids.add(token)
    return case_ids or None


def _ru_maxrss_mib(resource_key: int) -> float:
    """Return ru_maxrss in MiB for the selected usage scope."""
    import resource

    raw = float(resource.getrusage(resource_key).ru_maxrss)
    # macOS reports bytes, Linux reports KiB.
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for rtree benchmark runs."""
    parser = argparse.ArgumentParser(
        prog="benchmark_rtree.py",
        description="Measure elapsed time and peak memory while running fmfsolver cases.",
    )
    parser.add_argument(
        "--input",
        default="samples/input_benchmark_rtree.csv",
        help="Input cases file (.csv/.xlsx/.xlsm/.xls).",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers passed to solver (default: 1).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of benchmark repetitions in one process (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_rtree_metrics.csv",
        help="Path to benchmark metrics CSV.",
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Also write solver result CSV per run for traceability.",
    )
    parser.add_argument(
        "--result-prefix",
        default="outputs/benchmark_rtree_result",
        help="Output prefix when --write-results is enabled.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Run only selected case_id values (space/comma separated).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute benchmark runs and write metrics CSV."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    input_path = Path(args.input).expanduser()
    metrics_path = Path(args.output).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_cases(str(input_path))
    if len(df) == 0:
        raise ValueError("Input file has no cases.")

    case_ids = _parse_case_ids(args.cases)
    if case_ids is not None:
        selected = df[df["case_id"].astype(str).isin(case_ids)].reset_index(drop=True)
        missing = sorted(case_ids - set(df["case_id"].astype(str)))
        if missing:
            raise ValueError(f"Unknown case_id values: {missing}")
        if len(selected) == 0:
            raise ValueError("No cases selected.")
        df_run = selected
    else:
        df_run = df.reset_index(drop=True)

    rows: list[dict] = []

    for i in range(1, args.repeat + 1):
        import resource
        import tracemalloc

        t0 = time.perf_counter()
        tracemalloc.start()
        result_df = run_cases(df_run, logfn=lambda _msg: None, workers=args.workers)
        _current, py_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall_s = time.perf_counter() - t0

        if args.write_results:
            result_path = Path(f"{args.result_prefix}_{i:02d}.csv").expanduser()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            write_results_csv(str(result_path), df_run, result_df)

        total_scope = result_df[result_df["scope"] == "total"]
        solver_elapsed_sum_s = float(total_scope["run_elapsed_s"].sum())

        rss_self_mib = _ru_maxrss_mib(resource.RUSAGE_SELF)
        rss_children_mib = _ru_maxrss_mib(resource.RUSAGE_CHILDREN)

        row = {
            "run_index": i,
            "input_path": str(input_path.resolve()),
            "workers": int(args.workers),
            "case_count": int(len(df_run)),
            "wall_elapsed_s": float(wall_s),
            "solver_elapsed_sum_s": solver_elapsed_sum_s,
            "python_peak_alloc_mib": float(py_peak_bytes / (1024.0 * 1024.0)),
            "peak_rss_self_mib": float(rss_self_mib),
            "peak_rss_children_mib": float(rss_children_mib),
            "peak_rss_combined_mib": float(rss_self_mib + rss_children_mib),
        }
        rows.append(row)
        print(
            "[RUN %d/%d] wall=%.3fs, peak_rss_combined=%.2fMiB"
            % (i, args.repeat, row["wall_elapsed_s"], row["peak_rss_combined_mib"]),
            flush=True,
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_path, index=False)

    wall_values = metrics_df["wall_elapsed_s"].tolist()
    peak_values = metrics_df["peak_rss_combined_mib"].tolist()
    print(f"[OK] Wrote metrics: {metrics_path}", flush=True)
    print(
        "[SUMMARY] wall_s min/mean/max = %.3f / %.3f / %.3f"
        % (
            min(wall_values),
            statistics.mean(wall_values),
            max(wall_values),
        ),
        flush=True,
    )
    print(
        "[SUMMARY] peak_rss_combined_mib min/mean/max = %.2f / %.2f / %.2f"
        % (
            min(peak_values),
            statistics.mean(peak_values),
            max(peak_values),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
