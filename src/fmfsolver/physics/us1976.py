from __future__ import annotations

"""US1976 atmosphere lookup utilities backed by bundled CSV tables."""

import importlib.resources as resources
import math
from pathlib import Path

import numpy as np
import pandas as pd

_US1976_TABLE1: pd.DataFrame | None = None
_US1976_TABLE2: pd.DataFrame | None = None


def _data_path(filename: str):
    """Return package data path/traversable for a bundled US1976 CSV file."""
    try:
        return resources.files("fmfsolver").joinpath("data", filename)
    except Exception:
        return Path(__file__).resolve().parent / "data" / filename


def _load_table1(path: Path) -> pd.DataFrame:
    """Load Table 1 subset required by the solver (Z, T, c)."""
    df = pd.read_csv(path)
    for col in ["Z", "T", "c"]:
        if col not in df.columns:
            raise ValueError(f"us1976_table1.csv missing required column '{col}'.")
    out = df[["Z", "T", "c"]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Z", "T", "c"]).sort_values("Z").reset_index(drop=True)
    return out


def _load_table2(path: Path) -> pd.DataFrame:
    """Load Table 2 subset required by the solver (Z, V)."""
    df = pd.read_csv(path)
    for col in ["Z", "V"]:
        if col not in df.columns:
            raise ValueError(f"us1976_table2.csv missing required column '{col}'.")
    out = df[["Z", "V"]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Z", "V"]).sort_values("Z").reset_index(drop=True)
    return out


def load_us1976_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache US1976 tables from bundled CSVs.

    Returns:
        ``(table1, table2)`` where:
        - table1 contains numeric ``Z``, ``T``, ``c``
        - table2 contains numeric ``Z``, ``V``
    """
    global _US1976_TABLE1, _US1976_TABLE2
    if _US1976_TABLE1 is None:
        path1 = _data_path("us1976_table1.csv")
        if isinstance(path1, Path):
            if not path1.exists():
                raise FileNotFoundError(f"us1976_table1.csv not found: {path1}")
            _US1976_TABLE1 = _load_table1(path1)
        else:
            with resources.as_file(path1) as p:
                if not p.exists():
                    raise FileNotFoundError(f"us1976_table1.csv not found: {p}")
                _US1976_TABLE1 = _load_table1(p)
    if _US1976_TABLE2 is None:
        path2 = _data_path("us1976_table2.csv")
        if isinstance(path2, Path):
            if not path2.exists():
                raise FileNotFoundError(f"us1976_table2.csv not found: {path2}")
            _US1976_TABLE2 = _load_table2(path2)
        else:
            with resources.as_file(path2) as p:
                if not p.exists():
                    raise FileNotFoundError(f"us1976_table2.csv not found: {p}")
                _US1976_TABLE2 = _load_table2(p)
    return _US1976_TABLE1, _US1976_TABLE2


def sample_at_altitude_km(alt_km: float) -> dict:
    """Interpolate atmosphere properties at a geometric altitude.

    Args:
        alt_km: Geometric altitude [km].

    Returns:
        Dict containing:
        - ``T_K``: Temperature [K]
        - ``c_ms``: Speed of sound [m/s]
        - ``Vmean_ms``: Mean molecular speed [m/s]
    """
    t1, t2 = load_us1976_tables()
    Z1 = t1["Z"].to_numpy(dtype=float)
    Z2 = t2["Z"].to_numpy(dtype=float)

    zmin = max(float(Z1.min()), float(Z2.min()))
    zmax = min(float(Z1.max()), float(Z2.max()))
    if alt_km < zmin or alt_km > zmax:
        raise ValueError(f"Altitude_km={alt_km} out of range [{zmin}, {zmax}]")

    def interp_t1(col: str) -> float:
        y = t1[col].to_numpy(dtype=float)
        return float(np.interp(alt_km, Z1, y))

    def interp_t2(col: str) -> float:
        y = t2[col].to_numpy(dtype=float)
        return float(np.interp(alt_km, Z2, y))

    return {
        "T_K": interp_t1("T"),
        "c_ms": interp_t1("c"),
        "Vmean_ms": interp_t2("V"),
    }


def mean_to_most_probable_speed(v_mean: float) -> float:
    """Convert mean molecular speed to most probable speed."""
    return (math.sqrt(math.pi) / 2.0) * float(v_mean)
