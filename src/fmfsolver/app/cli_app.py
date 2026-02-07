from __future__ import annotations

import argparse
from pathlib import Path

from ..core.solver import run_cases
from ..io.csv_out import write_results_csv
from ..io.io_cases import read_cases


def _parse_case_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    case_ids: set[str] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                case_ids.add(token)
    return case_ids or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmfsolver-cli",
        description="Run FMF solver from CSV/Excel input without GUI.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input cases file (.csv/.xlsx/.xlsm/.xls)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output result CSV path (default: outputs/<input_stem>_result.csv)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Run only selected case_id values (space/comma separated).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    input_path = Path(args.input).expanduser()
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

    if args.output:
        out_path = Path(args.output).expanduser()
    else:
        out_path = Path("outputs") / f"{input_path.stem}_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def logfn(msg: str):
        print(msg, flush=True)

    print(
        f"[RUN] cases={len(df_run)} workers={args.workers} input={input_path}",
        flush=True,
    )
    result_df = run_cases(df_run, logfn, workers=args.workers)
    write_results_csv(str(out_path), df_run, result_df)
    print(f"[OK] Wrote results: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
