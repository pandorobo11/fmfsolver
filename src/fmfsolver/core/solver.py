from __future__ import annotations

"""Case execution pipeline for FMF coefficient computation."""

import hashlib
import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from trimesh import ray as trimesh_ray

from ..io.exporters import export_npz, export_vtp
from ..physics.us1976 import mean_to_most_probable_speed, sample_at_altitude_km
from .mesh_utils import load_meshes
from .sentman_core import (
    rot_y,
    sentman_dC_dA_vectors,
    stl_to_body,
    vhat_from_alpha_beta_stl,
)
from .shielding import compute_shield_mask_with_backend

try:
    SOLVER_VERSION = version("fmfsolver")
except PackageNotFoundError:
    SOLVER_VERSION = "dev"

_RAY_ACCEL_HINT_SHOWN = False


def _maybe_log_ray_accel_hint(logfn) -> None:
    """Log one-time hint when Embree acceleration is not available."""
    global _RAY_ACCEL_HINT_SHOWN
    if _RAY_ACCEL_HINT_SHOWN:
        return

    if trimesh_ray.has_embree:
        logfn("[INFO] Ray backend: Embree (ray_pyembree).")
    else:
        logfn(
            "[INFO] Ray backend: rtree (ray_triangle). Optional acceleration is "
            "available: uv sync --extra rayaccel (or pip install "
            "\"fmfsolver[rayaccel]\")."
        )
    _RAY_ACCEL_HINT_SHOWN = True


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
        "ray_backend",
        "shield_rays_mode",
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


def _compute_force_coeffs(C_force_stl: np.ndarray, alpha_deg: float) -> dict[str, float]:
    """Convert force vector in STL axes into force coefficients."""
    C_force_body = stl_to_body(C_force_stl)
    CA = -float(C_force_body[0])
    CY = float(C_force_body[1])
    CN = -float(C_force_body[2])

    Ry = rot_y(math.radians(alpha_deg))
    C_stab = Ry @ C_force_body
    CD = -float(C_stab[0])
    CL = -float(C_stab[2])
    return {
        "C_force_body": C_force_body,
        "CA": CA,
        "CY": CY,
        "CN": CN,
        "CD": CD,
        "CL": CL,
    }


def _stl_to_body_array(v_stl: np.ndarray) -> np.ndarray:
    """Convert STL-axis vectors to body axes while preserving array shape."""
    v = np.asarray(v_stl, dtype=float)
    out = np.array(v, copy=True)
    out[..., 0] *= -1.0
    out[..., 2] *= -1.0
    return out


def _expand_case_rows(case_result: dict) -> list[dict]:
    """Expand one case result into CSV rows (total + component rows)."""
    total = dict(case_result)
    components = list(total.pop("component_rows", []))
    rows = [total]
    for comp in components:
        row = dict(total)
        row.update(comp)
        rows.append(row)
    return rows


def _shield_reuse_sort_key(row: pd.Series, index: int) -> tuple:
    """Return execution sort key to cluster reusable shield-mask cases.

    Reuse condition is based on mesh identity and flow direction:
    ``stl_path``, ``stl_scale_m_per_unit``, ``alpha_deg``, ``beta_deg``, ``ray_backend``.
    Non-shielding cases are kept after shielding cases in input order.
    """
    try:
        shielding_on = bool(int(row.get("shielding_on", 0)))
    except Exception:
        shielding_on = False
    if not shielding_on:
        return (1, index)

    stl_paths = tuple(p.strip() for p in str(row.get("stl_path", "")).split(";") if p.strip())
    scale = round(float(row.get("stl_scale_m_per_unit", 1.0)), 12)
    alpha = round(float(row.get("alpha_deg", 0.0)), 12)
    beta = round(float(row.get("beta_deg", 0.0)), 12)
    ray_backend = str(row.get("ray_backend", "auto")).strip().lower() or "auto"
    return (0, stl_paths, scale, alpha, beta, ray_backend, index)


def _build_execution_order(df: pd.DataFrame) -> list[int]:
    """Build case execution order optimized for shield-mask reuse."""
    n = int(len(df))
    if n <= 1:
        return list(range(n))
    indices = list(range(n))
    return sorted(indices, key=lambda i: _shield_reuse_sort_key(df.iloc[i], i))


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
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    t0 = time.perf_counter()

    case_id = str(row["case_id"])
    stl_paths = [p.strip() for p in str(row["stl_path"]).split(";") if p.strip()]
    scale = float(row["stl_scale_m_per_unit"])

    Aref = float(row["Aref_m2"])
    Lref_Cl = float(row["Lref_Cl_m"])
    Lref_Cm = float(row["Lref_Cm_m"])
    Lref_Cn = float(row["Lref_Cn_m"])

    Tw = float(row["Tw_K"])
    ref_stl = np.array(
        [float(row["ref_x_m"]), float(row["ref_y_m"]), float(row["ref_z_m"])], dtype=float
    )
    ref_body = stl_to_body(ref_stl)

    alpha_deg = float(row["alpha_deg"])
    beta_deg = float(row["beta_deg"])

    shielding_on = bool(int(row.get("shielding_on", 0)))
    ray_backend = str(row.get("ray_backend", "auto")).strip().lower() or "auto"
    shield_rays_mode = str(row.get("shield_rays_mode", "auto")).strip().lower() or "auto"
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
    face_stl_index = md.face_stl_index
    stl_paths_order = md.stl_paths_order

    if shielding_on:
        shielded, ray_backend_used = compute_shield_mask_with_backend(
            mesh=mesh,
            centers_m=centers_stl,
            Vhat=Vhat,
            ray_backend=ray_backend,
            shield_rays_mode=shield_rays_mode,
        )
    else:
        shielded = np.zeros(len(areas), dtype=bool)
        ray_backend_used = "not_used"

    n_faces = len(areas)
    num_components = len(stl_paths_order)
    dC_dA_arr = sentman_dC_dA_vectors(
        Vhat=Vhat,
        n_out=normals_out_stl,
        S=S,
        Ti=Ti,
        Tw=Tw,
        Aref=Aref,
        shielded=shielded,
    )

    C_face_stl = dC_dA_arr * areas[:, None]
    C_force_stl = C_face_stl.sum(axis=0)

    C_force_stl_by_component = np.zeros((num_components, 3), dtype=float)
    np.add.at(C_force_stl_by_component, face_stl_index, C_face_stl)

    dot_nv = np.einsum("ij,j->i", normals_out_stl, Vhat)
    theta_deg = np.degrees(np.arccos(np.clip(dot_nv, -1.0, 1.0)))
    Cp_n = -(Aref / areas) * np.einsum("ij,ij->i", C_face_stl, normals_out_stl)

    centers_body = _stl_to_body_array(centers_stl)
    C_face_body = _stl_to_body_array(C_face_stl)
    dM_body = np.cross(centers_body - ref_body[None, :], C_face_body)
    C_M_body = dM_body.sum(axis=0)

    C_M_body_by_component = np.zeros((num_components, 3), dtype=float)
    np.add.at(C_M_body_by_component, face_stl_index, dM_body)

    total_force_coeffs = _compute_force_coeffs(C_force_stl, alpha_deg)
    C_force_body = total_force_coeffs["C_force_body"]
    CA = total_force_coeffs["CA"]
    CY = total_force_coeffs["CY"]
    CN = total_force_coeffs["CN"]
    CD = total_force_coeffs["CD"]
    CL = total_force_coeffs["CL"]

    Cl = float(C_M_body[0]) / Lref_Cl
    Cm = float(C_M_body[1]) / Lref_Cm
    Cn = float(C_M_body[2]) / Lref_Cn

    component_faces = np.bincount(face_stl_index, minlength=num_components).astype(int)
    component_shielded_faces = np.bincount(
        face_stl_index,
        weights=shielded.astype(np.int64),
        minlength=num_components,
    ).astype(int)
    component_rows = []
    if num_components > 1:
        for comp_i in range(num_components):
            coeffs = _compute_force_coeffs(C_force_stl_by_component[comp_i], alpha_deg)
            C_M_comp = C_M_body_by_component[comp_i]
            component_rows.append(
                {
                    "scope": "component",
                    "component_id": int(comp_i),
                    "component_stl_path": stl_paths_order[comp_i],
                    "CA": coeffs["CA"],
                    "CY": coeffs["CY"],
                    "CN": coeffs["CN"],
                    "Cl": float(C_M_comp[0]) / Lref_Cl,
                    "Cm": float(C_M_comp[1]) / Lref_Cm,
                    "Cn": float(C_M_comp[2]) / Lref_Cn,
                    "CD": coeffs["CD"],
                    "CL": coeffs["CL"],
                    "faces": int(component_faces[comp_i]),
                    "shielded_faces": int(component_shielded_faces[comp_i]),
                    # Keep paths only in total row for easier downstream handling.
                    "vtp_path": "",
                    "npz_path": "",
                }
            )

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
            "stl_index": face_stl_index.astype(np.int32),
        }
        export_vtp(
            str(vtp_path),
            mesh.vertices,
            mesh.faces,
            cell_data,
            field_data={
                "case_id": case_id,
                "case_signature": signature,
                "solver_version": SOLVER_VERSION,
                "stl_count": int(num_components),
                "ray_backend_used": ray_backend_used,
                "stl_paths_json": json.dumps(list(stl_paths_order), ensure_ascii=True),
            },
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
            face_stl_index=face_stl_index,
            stl_paths=np.array(stl_paths_order, dtype=object),
            ray_backend_used=ray_backend_used,
        )

    finished_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elapsed_s = time.perf_counter() - t0

    return {
        "case_id": case_id,
        "solver_version": SOLVER_VERSION,
        "case_signature": signature,
        "run_started_at_utc": started_at_utc,
        "run_finished_at_utc": finished_at_utc,
        "run_elapsed_s": float(elapsed_s),
        "mode": mode,
        "S": float(S),
        "Ti_K": float(Ti),
        "Tw_K": float(Tw),
        "scope": "total",
        "ray_backend_used": ray_backend_used,
        "component_id": "",
        "component_stl_path": "",
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
        "component_rows": component_rows,
    }

def _null_log(_msg: str):
    """No-op logger used in worker processes."""
    return None


def _run_case_worker(row: dict) -> dict:
    """Worker wrapper for ``ProcessPoolExecutor``."""
    return run_case(row, _null_log)


def run_cases(
    df: pd.DataFrame,
    logfn,
    workers: int = 1,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    flush_every_cases: int | None = None,
    chunk_cb: Callable[[pd.DataFrame, int, int, bool], None] | None = None,
) -> pd.DataFrame:
    """Run multiple cases sequentially or in parallel.

    Args:
        df: Input cases dataframe.
        logfn: Logging callback accepting one string argument.
        workers: Process count for case-level parallel execution.
        progress_cb: Optional callback receiving ``(done, total)``.
        cancel_cb: Optional callback returning ``True`` to request cancellation.
        flush_every_cases: Optional case count for chunk callbacks.
        chunk_cb: Optional callback receiving chunk dataframe and
            ``(done, total, is_final_chunk)``.

    Returns:
        Dataframe of per-case summary results, preserving input row order.
    """
    df = df.reset_index(drop=True)
    total = len(df)
    flush_every = int(flush_every_cases or 0)
    if flush_every < 0:
        raise ValueError("flush_every_cases must be >= 0.")
    if cancel_cb is not None and cancel_cb():
        raise RuntimeError("Canceled by user.")
    _maybe_log_ray_accel_hint(logfn)
    exec_order = _build_execution_order(df)

    chunk_rows: list[dict] = []
    pending_cases = 0

    def _emit_chunk(force: bool = False) -> None:
        nonlocal chunk_rows, pending_cases
        if chunk_cb is None:
            return
        if not chunk_rows:
            return
        if not force and flush_every <= 0:
            return
        if not force and pending_cases < flush_every:
            return
        chunk_df = pd.DataFrame(chunk_rows)
        chunk_cb(chunk_df, done_count, total, bool(force))
        chunk_rows = []
        pending_cases = 0

    if workers <= 1 or total <= 1:
        case_rows = [None] * total
        done_count = 0
        for run_idx, i in enumerate(exec_order, start=1):
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("Canceled by user.")
            r = df.iloc[i]
            logfn(f"[RUN] ({run_idx}/{total}) case_id={r['case_id']}")
            case_result = run_case(r.to_dict(), logfn)
            expanded = _expand_case_rows(case_result)
            case_rows[i] = expanded
            done_count += 1
            chunk_rows.extend(expanded)
            pending_cases += 1
            _emit_chunk(force=False)
            if progress_cb is not None:
                progress_cb(done_count, total)
        rows = []
        for bucket in case_rows:
            if bucket:
                rows.extend(bucket)
        _emit_chunk(force=True)
        return pd.DataFrame(rows)

    logfn(f"[RUN] Parallel execution with {workers} worker(s)")
    case_rows = [None] * total
    done_count = 0
    canceled = False
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {}
        for i in exec_order:
            if cancel_cb is not None and cancel_cb():
                canceled = True
                break
            r = df.iloc[i]
            futures[ex.submit(_run_case_worker, r.to_dict())] = i

        while futures and not canceled:
            done, _pending = wait(
                list(futures.keys()),
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if cancel_cb is not None and cancel_cb():
                    canceled = True
                continue

            for fut in done:
                i = futures.pop(fut)
                try:
                    expanded = _expand_case_rows(fut.result())
                    case_rows[i] = expanded
                    done_count += 1
                    chunk_rows.extend(expanded)
                    pending_cases += 1
                    _emit_chunk(force=False)
                    logfn(f"[OK] ({done_count}/{total}) case_id={df.loc[i, 'case_id']}")
                    if progress_cb is not None:
                        progress_cb(done_count, total)
                except Exception as e:
                    logfn(f"[ERROR] ({i+1}/{total}) case_id={df.loc[i, 'case_id']}: {e}")
                    raise

                if cancel_cb is not None and cancel_cb():
                    canceled = True
                    break
    finally:
        ex.shutdown(wait=not canceled, cancel_futures=canceled)

    if canceled:
        raise RuntimeError("Canceled by user.")

    rows = []
    for bucket in case_rows:
        if bucket:
            rows.extend(bucket)
    _emit_chunk(force=True)
    return pd.DataFrame(rows)
