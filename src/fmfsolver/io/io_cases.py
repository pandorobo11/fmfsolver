"""Input readers for case definition tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..common import is_filled

REQUIRED = [
    "case_id",
    "stl_path",
    "stl_scale_m_per_unit",
    "alpha_deg",
    "beta_or_bank_deg",
    "Tw_K",
    "ref_x_m",
    "ref_y_m",
    "ref_z_m",
    "Aref_m2",
    "Lref_Cl_m",
    "Lref_Cm_m",
    "Lref_Cn_m",
]

# Canonical display/output order for input-side columns.
INPUT_COLUMN_ORDER = [
    # 1) case id
    "case_id",
    # 2) geometry
    "stl_path",
    "stl_scale_m_per_unit",
    # 3) atmosphere (Mode A / Mode B)
    "S",
    "Ti_K",
    "Mach",
    "Altitude_km",
    # 4) physical condition
    "Tw_K",
    # 5) attitude
    "alpha_deg",
    "beta_or_bank_deg",
    "attitude_input",
    # 6) reference values
    "ref_x_m",
    "ref_y_m",
    "ref_z_m",
    "Aref_m2",
    "Lref_Cl_m",
    "Lref_Cm_m",
    "Lref_Cn_m",
    # 7) shielding settings
    "shielding_on",
    "ray_backend",
    # 8) I/O
    "out_dir",
    "save_vtp_on",
    "save_npz_on",
]

NUMERIC_REQUIRED = [
    "stl_scale_m_per_unit",
    "alpha_deg",
    "beta_or_bank_deg",
    "Tw_K",
    "ref_x_m",
    "ref_y_m",
    "ref_z_m",
    "Aref_m2",
    "Lref_Cl_m",
    "Lref_Cm_m",
    "Lref_Cn_m",
]

NUMERIC_OPTIONAL = [
    "S",
    "Ti_K",
    "Mach",
    "Altitude_km",
]

POSITIVE_COLUMNS = {
    "stl_scale_m_per_unit",
    "Tw_K",
    "Aref_m2",
    "Lref_Cl_m",
    "Lref_Cm_m",
    "Lref_Cn_m",
}

FLAG_COLUMNS = ["shielding_on", "save_vtp_on", "save_npz_on"]
RAY_BACKEND_VALUES = {"auto", "rtree", "embree"}
ATTITUDE_INPUT_VALUES = {"beta_tan", "beta_sin", "bank"}

DEFAULTS = {
    "shielding_on": 0,
    "save_vtp_on": 1,
    "save_npz_on": 0,
    "ray_backend": "auto",
    "attitude_input": "beta_tan",
    "out_dir": "outputs",
}

def _normalize_attitude_input(value) -> str:
    """Normalize attitude input selector text to canonical keyword."""
    raw = str(value or "").strip().lower()
    if not raw:
        return "beta_tan"
    return raw


@dataclass(frozen=True)
class ValidationIssue:
    """One structured validation error for an input case table."""

    row_number: int | None
    case_id: str | None
    field: str | None
    message: str


class InputValidationError(ValueError):
    """Raised when one or more input table validation checks fail."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        lines = ["Invalid input table:"]
        for issue in self.issues:
            parts = []
            if issue.row_number is not None:
                parts.append(f"row {issue.row_number}")
            if issue.case_id:
                parts.append(f"case_id='{issue.case_id}'")
            if issue.field:
                parts.append(issue.field)
            prefix = ", ".join(parts)
            if prefix:
                lines.append(f"- {prefix}: {issue.message}")
            else:
                lines.append(f"- {issue.message}")
        return "\n".join(lines)


_AddIssueFn = Callable[[int | None, str | None, str], None]


def _validate_case_ids(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Normalize and validate case identifiers."""
    df["case_id"] = df["case_id"].where(df["case_id"].notna(), "").astype(str).str.strip()
    blank_ids = df["case_id"] == ""
    for idx in df.index[blank_ids]:
        add_issue(int(idx), "case_id", "must not be blank.")

    duplicate_ids = sorted(df.loc[df["case_id"].duplicated(keep=False), "case_id"].unique())
    if duplicate_ids:
        add_issue(None, "case_id", f"Duplicate case_id values are not allowed: {duplicate_ids}")


def _validate_and_resolve_stl_paths(
    df: pd.DataFrame, input_path: Path, add_issue: _AddIssueFn
) -> None:
    """Validate `stl_path` cells and resolve relative entries to absolute paths."""
    base_dir = input_path.parent
    for idx, raw in df["stl_path"].items():
        if not is_filled(raw):
            add_issue(int(idx), "stl_path", "is required.")
            continue
        paths = [p.strip() for p in str(raw).split(";") if p.strip()]
        if not paths:
            add_issue(int(idx), "stl_path", "has no valid entry.")
            continue
        resolved_paths: list[str] = []
        for p in paths:
            candidate = Path(p).expanduser()
            resolved: Path | None = None
            if candidate.exists():
                resolved = candidate.resolve()
            elif not candidate.is_absolute() and (base_dir / candidate).exists():
                resolved = (base_dir / candidate).resolve()
            else:
                add_issue(
                    int(idx),
                    "stl_path",
                    (
                        f"STL file not found: '{p}' "
                        f"(checked relative to '{base_dir}')."
                    ),
                )
            if resolved is not None:
                resolved_paths.append(str(resolved))
        if resolved_paths:
            df.at[idx, "stl_path"] = ";".join(resolved_paths)


def _validate_required_numeric(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Parse required numeric columns and flag non-numeric cells."""
    for col in NUMERIC_REQUIRED:
        parsed = pd.to_numeric(df[col], errors="coerce")
        invalid = parsed.isna()
        for idx in df.index[invalid]:
            add_issue(int(idx), col, "must be numeric.")
        df[col] = parsed


def _validate_optional_numeric(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Parse optional mode columns while preserving blank cells as NaN."""
    for col in NUMERIC_OPTIONAL:
        if col not in df.columns:
            df[col] = float("nan")
        filled = df[col].map(is_filled)
        parsed = pd.to_numeric(df[col].where(filled), errors="coerce")
        invalid = filled & parsed.isna()
        for idx in df.index[invalid]:
            add_issue(int(idx), col, "must be numeric when specified.")
        df[col] = parsed


def _validate_positive_columns(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Enforce strictly positive constraints for configured columns."""
    for col in POSITIVE_COLUMNS:
        invalid = df[col] <= 0.0
        for idx in df.index[invalid]:
            add_issue(int(idx), col, "must be > 0.")


def _validate_mode_inputs(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Validate that each row specifies exactly one complete mode."""
    mode_a_s = df["S"].notna()
    mode_a_ti = df["Ti_K"].notna()
    mode_b_mach = df["Mach"].notna()
    mode_b_alt = df["Altitude_km"].notna()

    mode_a_partial = mode_a_s ^ mode_a_ti
    mode_b_partial = mode_b_mach ^ mode_b_alt
    for idx in df.index[mode_a_partial]:
        add_issue(int(idx), "S,Ti_K", "Mode A requires both 'S' and 'Ti_K'.")
    for idx in df.index[mode_b_partial]:
        add_issue(int(idx), "Mach,Altitude_km", "Mode B requires both 'Mach' and 'Altitude_km'.")

    mode_a_ok = mode_a_s & mode_a_ti
    mode_b_ok = mode_b_mach & mode_b_alt
    both_modes = mode_a_ok & mode_b_ok
    neither_mode = (~mode_a_ok) & (~mode_b_ok)
    for idx in df.index[both_modes]:
        add_issue(int(idx), "mode", "Specify either Mode A or Mode B, not both.")
    for idx in df.index[neither_mode]:
        add_issue(
            int(idx),
            "mode",
            (
                "Specify one complete mode "
                "(Mode A: S+Ti_K, Mode B: Mach+Altitude_km)."
            ),
        )

    for col in ("S", "Ti_K", "Mach"):
        invalid = df[col].notna() & (df[col] <= 0.0)
        for idx in df.index[invalid]:
            add_issue(int(idx), col, "must be > 0 when specified.")


def _validate_flags(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Normalize 0/1 flag columns and validate allowed values."""
    for col in FLAG_COLUMNS:
        parsed = pd.to_numeric(df[col], errors="coerce")
        invalid_numeric = parsed.isna()
        for idx in df.index[invalid_numeric]:
            add_issue(int(idx), col, "must be 0 or 1.")
        invalid_value = (~invalid_numeric) & (~parsed.isin([0, 1]))
        for idx in df.index[invalid_value]:
            add_issue(int(idx), col, "must be 0 or 1.")
        df[col] = parsed.fillna(0).astype(int)


def _validate_ray_backend(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Normalize and validate ray backend selector."""
    df["ray_backend"] = (
        df["ray_backend"].where(df["ray_backend"].notna(), "auto").astype(str).str.strip().str.lower()
    )
    blank_backend = df["ray_backend"] == ""
    if blank_backend.any():
        df.loc[blank_backend, "ray_backend"] = "auto"
    invalid_backend = ~df["ray_backend"].isin(RAY_BACKEND_VALUES)
    for idx in df.index[invalid_backend]:
        add_issue(int(idx), "ray_backend", "must be one of: auto, rtree, embree.")


def _validate_attitude_input(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Normalize and validate attitude input mode selector."""
    df["attitude_input"] = df["attitude_input"].map(_normalize_attitude_input)
    invalid_attitude = ~df["attitude_input"].isin(ATTITUDE_INPUT_VALUES)
    for idx in df.index[invalid_attitude]:
        add_issue(
            int(idx),
            "attitude_input",
            "must be one of: beta_tan, beta_sin, bank.",
        )


def _validate_out_dir(df: pd.DataFrame, add_issue: _AddIssueFn) -> None:
    """Validate output directory strings."""
    df["out_dir"] = df["out_dir"].astype(str).str.strip()
    blank_out = df["out_dir"] == ""
    for idx in df.index[blank_out]:
        add_issue(int(idx), "out_dir", "must not be blank.")


def _validate_and_normalize(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    """Validate input rows and normalize dtypes used by downstream solver code."""
    issues: list[ValidationIssue] = []

    def add_issue(idx: int | None, field: str | None, message: str):
        row_number = None if idx is None else int(idx) + 2
        case_id = None
        if idx is not None and "case_id" in df.columns:
            cid = str(df.at[idx, "case_id"]).strip()
            case_id = cid if cid else None
        issues.append(
            ValidationIssue(
                row_number=row_number,
                case_id=case_id,
                field=field,
                message=message,
            )
        )

    _validate_case_ids(df, add_issue)
    _validate_and_resolve_stl_paths(df, input_path, add_issue)
    _validate_required_numeric(df, add_issue)
    _validate_optional_numeric(df, add_issue)
    _validate_positive_columns(df, add_issue)
    _validate_mode_inputs(df, add_issue)
    _validate_flags(df, add_issue)
    _validate_ray_backend(df, add_issue)
    _validate_attitude_input(df, add_issue)
    _validate_out_dir(df, add_issue)

    if issues:
        raise InputValidationError(issues)

    return df


def read_cases(path: str) -> pd.DataFrame:
    """Load solver cases from CSV/Excel and normalize optional columns.

    Args:
        path: Input table path (`.csv`, `.xlsx`, `.xlsm`, or `.xls`).

    Returns:
        DataFrame containing required columns and default-filled optional ones.
        ``stl_path`` is normalized to absolute path(s), resolving relative paths
        from the input file directory.

    Raises:
        ValueError: If the file format is unsupported.
        InputValidationError: If required columns are missing or row validation fails.
    """
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(p, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported input format: {p.suffix}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        issues = [
            ValidationIssue(
                row_number=1,
                case_id=None,
                field="header",
                message=f"Missing required columns: {missing}",
            )
        ]
        raise InputValidationError(issues)

    for k, v in DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
        else:
            df[k] = df[k].fillna(v)

    normalized = _validate_and_normalize(df, p)
    ordered = [c for c in INPUT_COLUMN_ORDER if c in normalized.columns]
    extras = [c for c in normalized.columns if c not in ordered]
    return normalized[ordered + extras]
