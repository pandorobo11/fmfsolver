"""End-to-end flat-plate verification against Sentman one-sided formulas."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

from ..core.solver import run_case


def sentman_flat_plate_one_side_cn_ca(
    S: float,
    alpha_rad: float,
    Tr_over_Ti: float = 1.0,
    A_over_Aref: float = 1.0,
) -> tuple[float, float]:
    """Return Sentman one-sided flat-plate ``(CN, CA)``.

    This implements the analytical expressions used for Section III-B style
    verification of one exposed side.
    """
    if S <= 0:
        raise ValueError("S must be > 0.")
    if Tr_over_Ti <= 0:
        raise ValueError("Tr_over_Ti must be > 0.")
    if A_over_Aref <= 0:
        raise ValueError("A_over_Aref must be > 0.")

    sa = math.sin(alpha_rad)
    ca = math.cos(alpha_rad)

    x = S * ca
    erf_term = 1.0 + math.erf(x)
    exp_term = math.exp(-(x * x))

    invS = 1.0 / S
    invS2 = invS * invS
    invS_sqrtpi = invS / math.sqrt(math.pi)

    cn = A_over_Aref * (sa * ca * erf_term + sa * invS_sqrtpi * exp_term)

    sqrt_TrTi = math.sqrt(Tr_over_Ti)
    ca_inc = (ca * ca + 0.5 * invS2) * erf_term + ca * invS_sqrtpi * exp_term
    ca_ref = sqrt_TrTi * (
        (math.sqrt(math.pi) * 0.5 * invS) * ca * erf_term + 0.5 * invS2 * exp_term
    )
    ca_total = A_over_Aref * (ca_inc + ca_ref)

    return cn, ca_total


def _parse_csv_floats(text: str) -> list[float]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("Empty value list.")
    return vals


def _write_one_sided_plate_stl(path: Path) -> float:
    """Write a 1 m x 1 m one-sided plate whose outward normal is ``-X_stl``."""
    vertices = np.array(
        [
            [0.0, -0.5, -0.5],
            [0.0, +0.5, -0.5],
            [0.0, +0.5, +0.5],
            [0.0, -0.5, +0.5],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],  # winding chosen for outward normal -X
            [0, 3, 2],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
    return float(mesh.area)


def _iter_checks(
    stl_path: Path,
    area_m2: float,
    S_values: Iterable[float],
    alpha_deg_values: Iterable[float],
    Ti_K: float,
    Tr_over_Ti: float,
) -> list[dict]:
    out_dir = stl_path.parent
    checks = []
    for S in S_values:
        for alpha_deg in alpha_deg_values:
            alpha_rad = math.radians(alpha_deg)
            cn_ref, ca_ref = sentman_flat_plate_one_side_cn_ca(
                S=S,
                alpha_rad=alpha_rad,
                Tr_over_Ti=Tr_over_Ti,
                A_over_Aref=1.0,
            )
            row = {
                "case_id": f"verify_S{S:g}_a{alpha_deg:g}",
                "stl_path": str(stl_path),
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": alpha_deg,
                "beta_deg": 0.0,
                "Tw_K": Ti_K * Tr_over_Ti,
                "ref_x_m": 0.0,
                "ref_y_m": 0.0,
                "ref_z_m": 0.0,
                "Aref_m2": area_m2,
                "Lref_Cl_m": 1.0,
                "Lref_Cm_m": 1.0,
                "Lref_Cn_m": 1.0,
                "S": S,
                "Ti_K": Ti_K,
                "shielding_on": 0,
                "save_vtp_on": 0,
                "save_npz_on": 0,
                "out_dir": str(out_dir),
            }
            res = run_case(row, lambda _msg: None)
            checks.append(
                {
                    "S": S,
                    "alpha_deg": alpha_deg,
                    "CN_solver": float(res["CN"]),
                    "CA_solver": float(res["CA"]),
                    "CN_ref": float(cn_ref),
                    "CA_ref": float(ca_ref),
                }
            )
    return checks


def build_parser() -> argparse.ArgumentParser:
    """Create parser for flat-plate end-to-end verification."""
    parser = argparse.ArgumentParser(
        prog="fmfsolver-verify-flat-plate",
        description="Compare run_case outputs against Sentman one-sided flat-plate formulas.",
    )
    parser.add_argument(
        "--S",
        default="1,10,100",
        help="Comma-separated S values (default: 1,10,100).",
    )
    parser.add_argument(
        "--alpha-deg",
        default="0,10,30,60",
        help="Comma-separated alpha_deg values (default: 0,10,30,60).",
    )
    parser.add_argument(
        "--ti-k",
        type=float,
        default=1000.0,
        help="Free-stream translational temperature Ti [K] (default: 1000).",
    )
    parser.add_argument(
        "--tr-over-ti",
        type=float,
        default=1.0,
        help="Temperature ratio Tr/Ti (default: 1.0).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for both CN and CA (default: 1e-10).",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated STL and temporary outputs under outputs/verify_flat_plate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run end-to-end solver validation for one-sided flat-plate conditions."""
    args = build_parser().parse_args(argv)
    S_values = _parse_csv_floats(args.S)
    alpha_deg_values = _parse_csv_floats(args.alpha_deg)

    if args.keep_artifacts:
        base_dir = Path("outputs") / "verify_flat_plate"
        base_dir.mkdir(parents=True, exist_ok=True)
        stl_path = base_dir / "one_sided_plate.stl"
        area_m2 = _write_one_sided_plate_stl(stl_path)
        checks = _iter_checks(
            stl_path=stl_path,
            area_m2=area_m2,
            S_values=S_values,
            alpha_deg_values=alpha_deg_values,
            Ti_K=float(args.ti_k),
            Tr_over_Ti=float(args.tr_over_ti),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="fmfsolver_verify_") as td:
            base_dir = Path(td)
            stl_path = base_dir / "one_sided_plate.stl"
            area_m2 = _write_one_sided_plate_stl(stl_path)
            checks = _iter_checks(
                stl_path=stl_path,
                area_m2=area_m2,
                S_values=S_values,
                alpha_deg_values=alpha_deg_values,
                Ti_K=float(args.ti_k),
                Tr_over_Ti=float(args.tr_over_ti),
            )

    max_err_cn = 0.0
    max_err_ca = 0.0
    print("S,alpha_deg,CN_solver,CN_ref,abs_err_CN,CA_solver,CA_ref,abs_err_CA", flush=True)
    for c in checks:
        err_cn = abs(c["CN_solver"] - c["CN_ref"])
        err_ca = abs(c["CA_solver"] - c["CA_ref"])
        max_err_cn = max(max_err_cn, err_cn)
        max_err_ca = max(max_err_ca, err_ca)
        print(
            f"{c['S']:.12g},{c['alpha_deg']:.12g},"
            f"{c['CN_solver']:.12e},{c['CN_ref']:.12e},{err_cn:.3e},"
            f"{c['CA_solver']:.12e},{c['CA_ref']:.12e},{err_ca:.3e}",
            flush=True,
        )

    print(
        f"[SUMMARY] max_abs_err_CN={max_err_cn:.3e}, max_abs_err_CA={max_err_ca:.3e}, tol={args.tol:.3e}",
        flush=True,
    )
    if max_err_cn > args.tol or max_err_ca > args.tol:
        print("[NG] Verification failed.", flush=True)
        return 1
    print("[OK] Verification passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
