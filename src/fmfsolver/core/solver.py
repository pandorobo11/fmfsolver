from __future__ import annotations

"""Case execution pipeline for FMF coefficient computation."""

import hashlib
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..io.exporters import export_npz, export_vtp
from ..physics.us1976 import mean_to_most_probable_speed, sample_at_altitude_km
from .mesh_utils import load_meshes
from .sentman_core import (
    rot_y,
    sentman_dC_dA_vector,
    stl_to_body,
    vhat_from_alpha_beta_stl,
)
from .shielding import compute_shield_mask


def _is_filled(x) -> bool:
    """Return True when a cell value should be treated as specified."""
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return str(x).strip() != ""


def _mode_from_row(row: dict) -> str:
    """Determine input mode for a case row.

    Mode A requires ``S`` and ``Ti_K``.
    Mode B requires ``Mach`` and ``Altitude_km``.
    """
    A_ok = _is_filled(row.get("S")) and _is_filled(row.get("Ti_K"))
    B_ok = _is_filled(row.get("Mach")) and _is_filled(row.get("Altitude_km"))
    if A_ok and B_ok:
        raise ValueError(f"Case '{row.get('case_id')}' has BOTH Mode A and Mode B inputs.")
    if (not A_ok) and (not B_ok):
        raise ValueError(
            f"Case '{row.get('case_id')}' has NEITHER complete Mode A nor Mode B inputs."
        )
    return "A" if A_ok else "B"


def build_case_signature(row: dict) -> str:
    """Build a stable signature hash for case identity and cache validation.

    The signature is stored in VTP metadata and used to verify that a VTP file
    matches the currently selected case parameters.
    """
    keys = [
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
        "S",
        "Ti_K",
        "Mach",
        "Altitude_km",
        "shielding_on",
    ]
    numeric_keys = {
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
        "S",
        "Ti_K",
        "Mach",
        "Altitude_km",
        "shielding_on",
    }

    def norm_value(k, v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if k in numeric_keys:
            try:
                return float(v)
            except Exception:
                return str(v)
        return str(v)

    data = {k: norm_value(k, row.get(k)) for k in keys}
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_S_Ti_R(row: dict) -> tuple[float, float, str]:
    """Return ``(S, Ti, mode)`` derived from case inputs."""
    mode = _mode_from_row(row)

    if mode == "A":
        S = float(row["S"])
        Ti = float(row["Ti_K"])
        return S, Ti, "A"

    Mach = float(row["Mach"])
    alt = float(row["Altitude_km"])

    atm = sample_at_altitude_km(alt)
    Ti = float(atm["T_K"])
    c = float(atm["c_ms"])
    v_mean = float(atm["Vmean_ms"])
    v_mp = mean_to_most_probable_speed(v_mean)

    V_bulk = Mach * c
    S = V_bulk / v_mp
    return S, Ti, "B"


def run_case(row: dict, logfn) -> dict:
    """Run a single aerodynamic case and return summary outputs.

    Side effects:
        - Writes VTP when ``save_vtp_on`` is truthy.
        - Writes NPZ when ``save_npz_on`` is truthy.

    Args:
        row: One case row converted to ``dict``.
        logfn: Logging callback accepting one string argument.

    Returns:
        Dictionary with integrated coefficients, face counts, and output paths.
    """
    case_id = str(row["case_id"])
    stl_paths = [p.strip() for p in str(row["stl_path"]).split(";") if p.strip()]
    scale = float(row["stl_scale_m_per_unit"])

    Aref = float(row["Aref_m2"])
    Lref_Cl = float(row["Lref_Cl_m"])
    Lref_Cm = float(row["Lref_Cm_m"])
    Lref_Cn = float(row["Lref_Cn_m"])

    Tw = float(row["Tw_K"])
    ref_body = np.array(
        [float(row["ref_x_m"]), float(row["ref_y_m"]), float(row["ref_z_m"])], dtype=float
    )

    alpha_deg = float(row["alpha_deg"])
    beta_deg = float(row["beta_deg"])

    shielding_on = bool(int(row.get("shielding_on", 0)))
    save_vtp = bool(int(row.get("save_vtp_on", 1)))
    save_npz = bool(int(row.get("save_npz_on", 0)))
    out_dir = Path(str(row.get("out_dir", "outputs"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    S, Ti, mode = _compute_S_Ti_R(row)
    signature = build_case_signature(row)

    Vhat = vhat_from_alpha_beta_stl(alpha_deg, beta_deg)

    md = load_meshes(stl_paths, scale, logfn)
    mesh = md.mesh
    centers_stl = md.centers_m
    normals_out_stl = md.normals_out
    areas = md.areas_m2

    if shielding_on:
        shielded = compute_shield_mask(mesh, centers_stl, Vhat)
    else:
        shielded = np.zeros(len(areas), dtype=bool)

    C_face_stl = np.zeros((len(areas), 3), dtype=float)
    Cp_n = np.zeros(len(areas), dtype=float)
    theta_deg = np.zeros(len(areas), dtype=float)

    C_force_stl = np.zeros(3, dtype=float)

    for i in range(len(areas)):
        dot_nv = float(np.dot(normals_out_stl[i], Vhat))
        dot_nv = max(-1.0, min(1.0, dot_nv))
        theta_deg[i] = math.degrees(math.acos(dot_nv))

        dC_dA = sentman_dC_dA_vector(
            Vhat=Vhat,
            n_out=normals_out_stl[i],
            S=S,
            Ti=Ti,
            Tw=Tw,
            Aref=Aref,
            shielded=shielded[i],
        )

        Ci = dC_dA * areas[i]
        C_face_stl[i] = Ci
        C_force_stl += Ci

        Cp_n[i] = -(Aref / areas[i]) * float(np.dot(Ci, normals_out_stl[i]))

    C_force_body = stl_to_body(C_force_stl)

    CA = -float(C_force_body[0])
    CY = float(C_force_body[1])
    CN = -float(C_force_body[2])

    C_M_body = np.zeros(3, dtype=float)
    for i in range(len(areas)):
        center_body = stl_to_body(centers_stl[i])
        r = center_body - ref_body
        C_face_body = stl_to_body(C_face_stl[i])
        C_M_body += np.cross(r, C_face_body)

    Cl = float(C_M_body[0]) / Lref_Cl
    Cm = float(C_M_body[1]) / Lref_Cm
    Cn = float(C_M_body[2]) / Lref_Cn

    Ry = rot_y(math.radians(alpha_deg))
    C_stab = Ry @ C_force_body

    CD = -float(C_stab[0])
    CL = -float(C_stab[2])

    vtp_path = out_dir / f"{case_id}.vtp"
    npz_path = out_dir / f"{case_id}.npz"

    if save_vtp:
        cell_data = {
            "area_m2": areas,
            "shielded": shielded.astype(np.uint8),
            "Cp_n": Cp_n,
            "theta_deg": theta_deg,
            "C_face_stl": C_face_stl,
            # NEW: face center scalars (STL axes, meters)
            "center_x_stl_m": centers_stl[:, 0],
            "center_y_stl_m": centers_stl[:, 1],
            "center_z_stl_m": centers_stl[:, 2],
        }
        export_vtp(
            str(vtp_path),
            mesh.vertices,
            mesh.faces,
            cell_data,
            field_data={"case_id": case_id, "case_signature": signature},
        )

    if save_npz:
        export_npz(
            str(npz_path),
            vertices=mesh.vertices,
            faces=mesh.faces,
            centers_stl_m=centers_stl,
            normals_out_stl=normals_out_stl,
            areas_m2=areas,
            shielded=shielded,
            Vhat_stl=Vhat,
            S=S,
            Ti_K=Ti,
            Tw_K=Tw,
            Aref_m2=Aref,
            C_force_stl=C_force_stl,
            C_force_body=C_force_body,
            C_M_body=C_M_body,
            CA=CA,
            CY=CY,
            CN=CN,
            Cl=Cl,
            Cm=Cm,
            Cn=Cn,
            CD=CD,
            CL=CL,
            Cp_n=Cp_n,
        )

    return {
        "case_id": case_id,
        "mode": mode,
        "S": float(S),
        "Ti_K": float(Ti),
        "Tw_K": float(Tw),
        "CA": CA,
        "CY": CY,
        "CN": CN,
        "Cl": Cl,
        "Cm": Cm,
        "Cn": Cn,
        "CD": CD,
        "CL": CL,
        "faces": int(len(mesh.faces)),
        "shielded_faces": int(shielded.sum()),
        "vtp_path": str(vtp_path) if save_vtp else "",
        "npz_path": str(npz_path) if save_npz else "",
    }

def _null_log(_msg: str):
    """No-op logger used in worker processes."""
    return None


def _run_case_worker(row: dict) -> dict:
    """Worker wrapper for ``ProcessPoolExecutor``."""
    return run_case(row, _null_log)


def run_cases(df: pd.DataFrame, logfn, workers: int = 1) -> pd.DataFrame:
    """Run multiple cases sequentially or in parallel.

    Args:
        df: Input cases dataframe.
        logfn: Logging callback accepting one string argument.
        workers: Process count for case-level parallel execution.

    Returns:
        Dataframe of per-case summary results, preserving input row order.
    """
    df = df.reset_index(drop=True)
    total = len(df)
    if workers <= 1 or total <= 1:
        rows = []
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            logfn(f"[RUN] ({i}/{total}) case_id={r['case_id']}")
            rows.append(run_case(r.to_dict(), logfn))
        return pd.DataFrame(rows)

    logfn(f"[RUN] Parallel execution with {workers} worker(s)")
    rows = [None] * total
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for i, (_, r) in enumerate(df.iterrows()):
            futures[ex.submit(_run_case_worker, r.to_dict())] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                rows[i] = fut.result()
                logfn(f"[OK] ({i+1}/{total}) case_id={df.loc[i, 'case_id']}")
            except Exception as e:
                logfn(f"[ERROR] ({i+1}/{total}) case_id={df.loc[i, 'case_id']}: {e}")
                raise

    return pd.DataFrame(rows)
