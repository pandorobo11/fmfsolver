from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from fmfsolver.core.solver import build_case_signature, run_case


class TestSolverPipeline(unittest.TestCase):
    def test_build_case_signature_numeric_normalization(self):
        row_int = {
            "case_id": "sig_case",
            "stl_path": "samples/stl/cube.stl",
            "stl_scale_m_per_unit": 1,
            "alpha_deg": 5,
            "beta_deg": 0,
            "Tw_K": 300,
            "ref_x_m": 0,
            "ref_y_m": 0,
            "ref_z_m": 0,
            "Aref_m2": 1,
            "Lref_Cl_m": 1,
            "Lref_Cm_m": 1,
            "Lref_Cn_m": 1,
            "S": 5,
            "Ti_K": 300,
            "shielding_on": 0,
        }
        row_float = {
            **row_int,
            "stl_scale_m_per_unit": 1.0,
            "alpha_deg": 5.0,
            "Tw_K": 300.0,
            "Aref_m2": 1.0,
            "S": 5.0,
            "Ti_K": 300.0,
        }
        self.assertEqual(build_case_signature(row_int), build_case_signature(row_float))

    def test_run_case_mode_a_smoke(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "test_mode_a",
                "stl_path": "samples/stl/cube.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_deg": 0.0,
                "Tw_K": 300.0,
                "ref_x_m": 0.0,
                "ref_y_m": 0.0,
                "ref_z_m": 0.0,
                "Aref_m2": 1.0,
                "Lref_Cl_m": 1.0,
                "Lref_Cm_m": 1.0,
                "Lref_Cn_m": 1.0,
                "S": 5.0,
                "Ti_K": 300.0,
                "shielding_on": 0,
                "save_vtp_on": 0,
                "save_npz_on": 0,
                "out_dir": td,
            }
            result = run_case(row, lambda _msg: None)

            self.assertEqual(result["mode"], "A")
            self.assertGreater(result["faces"], 0)
            self.assertEqual(result["vtp_path"], "")
            self.assertEqual(result["npz_path"], "")
            self.assertEqual(result["case_signature"], build_case_signature(row))
            self.assertTrue(str(result["solver_version"]).strip() != "")
            self.assertTrue(str(result["run_started_at_utc"]).endswith("Z"))
            self.assertTrue(str(result["run_finished_at_utc"]).endswith("Z"))
            self.assertGreaterEqual(float(result["run_elapsed_s"]), 0.0)
            for key in ("CA", "CY", "CN", "Cl", "Cm", "Cn", "CD", "CL"):
                self.assertTrue(math.isfinite(float(result[key])), key)
            self.assertTrue(Path(td).exists())

    def test_run_case_mode_b_smoke(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "test_mode_b",
                "stl_path": "samples/stl/plate.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_deg": 0.0,
                "Tw_K": 300.0,
                "ref_x_m": 0.0,
                "ref_y_m": 0.0,
                "ref_z_m": 0.0,
                "Aref_m2": 1.0,
                "Lref_Cl_m": 1.0,
                "Lref_Cm_m": 1.0,
                "Lref_Cn_m": 1.0,
                "Mach": 25.0,
                "Altitude_km": 100.0,
                "shielding_on": 0,
                "save_vtp_on": 0,
                "save_npz_on": 0,
                "out_dir": td,
            }
            result = run_case(row, lambda _msg: None)

            self.assertEqual(result["mode"], "B")
            self.assertGreater(result["faces"], 0)
            self.assertEqual(result["vtp_path"], "")
            self.assertEqual(result["npz_path"], "")
            self.assertEqual(result["case_signature"], build_case_signature(row))
            self.assertTrue(str(result["solver_version"]).strip() != "")
            self.assertTrue(str(result["run_started_at_utc"]).endswith("Z"))
            self.assertTrue(str(result["run_finished_at_utc"]).endswith("Z"))
            self.assertGreaterEqual(float(result["run_elapsed_s"]), 0.0)
            for key in ("S", "Ti_K", "CA", "CY", "CN", "CD", "CL"):
                self.assertTrue(math.isfinite(float(result[key])), key)


if __name__ == "__main__":
    unittest.main()
