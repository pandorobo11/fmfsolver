from __future__ import annotations

"""Case signature hashing for output validation and cache identity."""

import hashlib
import json
import math


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
        "beta_or_bank_deg",
        "attitude_input",
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
    ]
    numeric_keys = {
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
