from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from fmfsolver.core.solver import run_case


def _sentman_flat_plate_cn_ca(
    S: float,
    alpha_rad: float,
    Tr_over_Ti: float = 1.0,
    A_over_Aref: float = 1.0,
) -> tuple[float, float]:
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


def _write_one_sided_plate(path: Path) -> float:
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
            [0, 2, 1],
            [0, 3, 2],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)
    return float(mesh.area)


class TestFlatPlateVerification(unittest.TestCase):
    def test_run_case_matches_sentman_flat_plate_formula(self):
        S_values = (1.0, 10.0, 100.0)
        alpha_deg_values = (0.0, 10.0, 30.0, 60.0)
        Ti_K = 1000.0
        Tr_over_Ti = 1.0
        tol = 1e-10

        with tempfile.TemporaryDirectory(prefix="fmfsolver_flat_plate_") as td:
            stl_path = Path(td) / "one_sided_plate.stl"
            area_m2 = _write_one_sided_plate(stl_path)

            for S in S_values:
                for alpha_deg in alpha_deg_values:
                    alpha_rad = math.radians(alpha_deg)
                    cn_ref, ca_ref = _sentman_flat_plate_cn_ca(
                        S=S,
                        alpha_rad=alpha_rad,
                        Tr_over_Ti=Tr_over_Ti,
                        A_over_Aref=1.0,
                    )
                    row = {
                        "case_id": f"flat_S{S:g}_a{alpha_deg:g}",
                        "stl_path": str(stl_path),
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": alpha_deg,
                        "beta_or_bank_deg": 0.0,
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
                        "out_dir": td,
                    }
                    result = run_case(row, lambda _msg: None)
                    cn_err = abs(float(result["CN"]) - cn_ref)
                    ca_err = abs(float(result["CA"]) - ca_ref)
                    with self.subTest(S=S, alpha_deg=alpha_deg):
                        self.assertLessEqual(cn_err, tol)
                        self.assertLessEqual(ca_err, tol)


if __name__ == "__main__":
    unittest.main()
