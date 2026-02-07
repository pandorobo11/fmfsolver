"""Input readers for case definition tables."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REQUIRED = [
    "case_id",
    "stl_path",
    "stl_scale_m_per_unit",
    "alpha_deg",
    "beta_deg",
    "Tw_K",
    "ref_x_m",
    "ref_y_m",
    "ref_z_m",
    "Aref_m2",
    "Lref_Cl_m",
    "Lref_Cm_m",
    "Lref_Cn_m",
]

NUMERIC_REQUIRED = [
    "stl_scale_m_per_unit",
    "alpha_deg",
    "beta_deg",
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

DEFAULTS = {
    "shielding_on": 0,
    "save_vtp_on": 1,
    "save_npz_on": 0,
    "out_dir": "outputs",
}


def _is_filled(value) -> bool:
    """Return True if a table cell should be treated as specified."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


def _row_label(df: pd.DataFrame, idx: int) -> str:
    """Build a human-readable location string for validation messages."""
    if "case_id" not in df.columns:
        return f"row {idx + 2}"
    case_id = str(df.at[idx, "case_id"]).strip()
    if case_id:
        return f"row {idx + 2} (case_id='{case_id}')"
    return f"row {idx + 2} (case_id='<blank>')"


def _validate_and_normalize(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    """Validate input rows and normalize dtypes used by downstream solver code."""
    errors: list[str] = []

    # Normalize and validate case_id first to improve all later error messages.
    df["case_id"] = df["case_id"].astype(str).str.strip()
    blank_ids = df["case_id"] == ""
    for idx in df.index[blank_ids]:
        errors.append(f"{_row_label(df, int(idx))}: case_id must not be blank.")

    duplicate_ids = sorted(df.loc[df["case_id"].duplicated(keep=False), "case_id"].unique())
    if duplicate_ids:
        errors.append(f"Duplicate case_id values are not allowed: {duplicate_ids}")

    # Validate stl_path and normalize entries to absolute paths resolved from input file dir.
    base_dir = input_path.parent
    for idx, raw in df["stl_path"].items():
        label = _row_label(df, int(idx))
        if not _is_filled(raw):
            errors.append(f"{label}: stl_path is required.")
            continue
        paths = [p.strip() for p in str(raw).split(";") if p.strip()]
        if not paths:
            errors.append(f"{label}: stl_path has no valid entry.")
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
                errors.append(
                    f"{label}: STL file not found: '{p}' "
                    f"(checked relative to '{base_dir}')."
                )
            if resolved is not None:
                resolved_paths.append(str(resolved))
        if resolved_paths:
            df.at[idx, "stl_path"] = ";".join(resolved_paths)

    # Required numeric columns: must parse and be finite.
    for col in NUMERIC_REQUIRED:
        parsed = pd.to_numeric(df[col], errors="coerce")
        invalid = parsed.isna()
        for idx in df.index[invalid]:
            errors.append(f"{_row_label(df, int(idx))}: '{col}' must be numeric.")
        df[col] = parsed

    # Optional mode columns: convert filled cells to numeric and keep empty as NaN.
    for col in NUMERIC_OPTIONAL:
        if col not in df.columns:
            df[col] = float("nan")
        filled = df[col].map(_is_filled)
        parsed = pd.to_numeric(df[col].where(filled), errors="coerce")
        invalid = filled & parsed.isna()
        for idx in df.index[invalid]:
            errors.append(f"{_row_label(df, int(idx))}: '{col}' must be numeric when specified.")
        df[col] = parsed

    # Positive-only constraints.
    for col in POSITIVE_COLUMNS:
        invalid = df[col] <= 0.0
        for idx in df.index[invalid]:
            errors.append(f"{_row_label(df, int(idx))}: '{col}' must be > 0.")

    # Mode constraints: exactly one of A(S+Ti) or B(Mach+Altitude) per row.
    mode_a_s = df["S"].notna()
    mode_a_ti = df["Ti_K"].notna()
    mode_b_mach = df["Mach"].notna()
    mode_b_alt = df["Altitude_km"].notna()

    mode_a_partial = mode_a_s ^ mode_a_ti
    mode_b_partial = mode_b_mach ^ mode_b_alt
    for idx in df.index[mode_a_partial]:
        errors.append(f"{_row_label(df, int(idx))}: Mode A requires both 'S' and 'Ti_K'.")
    for idx in df.index[mode_b_partial]:
        errors.append(
            f"{_row_label(df, int(idx))}: Mode B requires both 'Mach' and 'Altitude_km'."
        )

    mode_a_ok = mode_a_s & mode_a_ti
    mode_b_ok = mode_b_mach & mode_b_alt
    both_modes = mode_a_ok & mode_b_ok
    neither_mode = (~mode_a_ok) & (~mode_b_ok)
    for idx in df.index[both_modes]:
        errors.append(
            f"{_row_label(df, int(idx))}: specify either Mode A or Mode B, not both."
        )
    for idx in df.index[neither_mode]:
        errors.append(
            f"{_row_label(df, int(idx))}: specify one complete mode "
            f"(Mode A: S+Ti_K, Mode B: Mach+Altitude_km)."
        )

    for col in ("S", "Ti_K", "Mach"):
        invalid = df[col].notna() & (df[col] <= 0.0)
        for idx in df.index[invalid]:
            errors.append(f"{_row_label(df, int(idx))}: '{col}' must be > 0 when specified.")

    # Normalize flags and enforce 0/1 values.
    for col in FLAG_COLUMNS:
        parsed = pd.to_numeric(df[col], errors="coerce")
        invalid_numeric = parsed.isna()
        for idx in df.index[invalid_numeric]:
            errors.append(f"{_row_label(df, int(idx))}: '{col}' must be 0 or 1.")
        invalid_value = (~invalid_numeric) & (~parsed.isin([0, 1]))
        for idx in df.index[invalid_value]:
            errors.append(f"{_row_label(df, int(idx))}: '{col}' must be 0 or 1.")
        df[col] = parsed.fillna(0).astype(int)

    # out_dir must be a non-empty string value.
    df["out_dir"] = df["out_dir"].astype(str).str.strip()
    blank_out = df["out_dir"] == ""
    for idx in df.index[blank_out]:
        errors.append(f"{_row_label(df, int(idx))}: 'out_dir' must not be blank.")

    if errors:
        head = "Invalid input table:"
        raise ValueError(head + "\n- " + "\n- ".join(errors))

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
        ValueError: If the file format is unsupported or required columns are missing.
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
        raise ValueError(f"Missing required columns: {missing}")

    for k, v in DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
        else:
            df[k] = df[k].fillna(v)

    return _validate_and_normalize(df, p)
