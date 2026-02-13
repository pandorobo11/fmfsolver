from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fmfsolver.io.io_cases import InputValidationError, read_cases


class TestIoCasesValidation(unittest.TestCase):
    def _base_row(self, stl_path: str) -> dict:
        return {
            "case_id": "case_ok",
            "stl_path": stl_path,
            "stl_scale_m_per_unit": 1.0,
            "alpha_deg": 0.0,
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
            "Mach": "",
            "Altitude_km": "",
            "shielding_on": 0,
            "save_vtp_on": 1,
            "save_npz_on": 0,
            "out_dir": "outputs",
        }

    def test_read_cases_accepts_valid_csv(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_valid_") as td:
            td_path = Path(td)
            stl_path = td_path / "mesh.stl"
            stl_path.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            df = pd.DataFrame([self._base_row("mesh.stl")])
            df.to_csv(csv_path, index=False)

            loaded = read_cases(str(csv_path))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(str(loaded.loc[0, "case_id"]), "case_ok")
            self.assertEqual(int(loaded.loc[0, "shielding_on"]), 0)
            self.assertEqual(str(loaded.loc[0, "ray_backend"]), "auto")
            self.assertEqual(str(loaded.loc[0, "shield_rays_mode"]), "auto")
            self.assertEqual(str(loaded.loc[0, "stl_path"]), str(stl_path.resolve()))

    def test_read_cases_normalizes_multi_stl_paths_to_absolute(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_multi_") as td:
            td_path = Path(td)
            stl1 = td_path / "a.stl"
            stl2 = td_path / "b.stl"
            stl1.write_text("solid a\nendsolid a\n", encoding="utf-8")
            stl2.write_text("solid b\nendsolid b\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            row = self._base_row("a.stl;b.stl")
            pd.DataFrame([row]).to_csv(csv_path, index=False)

            loaded = read_cases(str(csv_path))
            actual = str(loaded.loc[0, "stl_path"]).split(";")
            self.assertEqual(actual, [str(stl1.resolve()), str(stl2.resolve())])

    def test_read_cases_rejects_duplicate_case_id(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_dup_") as td:
            td_path = Path(td)
            stl_path = td_path / "mesh.stl"
            stl_path.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            row = self._base_row("mesh.stl")
            df = pd.DataFrame([row, row])
            df.to_csv(csv_path, index=False)

            with self.assertRaisesRegex(InputValidationError, "Duplicate case_id values"):
                read_cases(str(csv_path))

    def test_read_cases_rejects_partial_mode_and_bad_flags(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_mode_") as td:
            td_path = Path(td)
            stl_path = td_path / "mesh.stl"
            stl_path.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            row = self._base_row("mesh.stl")
            row["Ti_K"] = ""
            row["save_vtp_on"] = 2
            df = pd.DataFrame([row])
            df.to_csv(csv_path, index=False)

            with self.assertRaises(InputValidationError) as cm:
                read_cases(str(csv_path))
            msg = str(cm.exception)
            self.assertIn("Mode A requires both 'S' and 'Ti_K'", msg)
            self.assertIn("save_vtp_on", msg)
            self.assertIn("must be 0 or 1", msg)

    def test_read_cases_accepts_and_validates_ray_backend(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_backend_") as td:
            td_path = Path(td)
            stl_path = td_path / "mesh.stl"
            stl_path.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            row = self._base_row("mesh.stl")
            row["ray_backend"] = "rtree"
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            loaded = read_cases(str(csv_path))
            self.assertEqual(str(loaded.loc[0, "ray_backend"]), "rtree")

            row_bad = self._base_row("mesh.stl")
            row_bad["ray_backend"] = "invalid_backend"
            pd.DataFrame([row_bad]).to_csv(csv_path, index=False)
            with self.assertRaisesRegex(InputValidationError, "ray_backend"):
                read_cases(str(csv_path))

    def test_read_cases_accepts_and_validates_shield_rays_mode(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_shield_rays_") as td:
            td_path = Path(td)
            stl_path = td_path / "mesh.stl"
            stl_path.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            row = self._base_row("mesh.stl")
            row["shield_rays_mode"] = "precise"
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            loaded = read_cases(str(csv_path))
            self.assertEqual(str(loaded.loc[0, "shield_rays_mode"]), "precise")

            row_bad = self._base_row("mesh.stl")
            row_bad["shield_rays_mode"] = "slow"
            pd.DataFrame([row_bad]).to_csv(csv_path, index=False)
            with self.assertRaisesRegex(InputValidationError, "shield_rays_mode"):
                read_cases(str(csv_path))

            row_bad_num = self._base_row("mesh.stl")
            row_bad_num["shield_rays_mode"] = "3"
            pd.DataFrame([row_bad_num]).to_csv(csv_path, index=False)
            with self.assertRaisesRegex(InputValidationError, "shield_rays_mode"):
                read_cases(str(csv_path))

    def test_read_cases_exposes_structured_issues(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_io_structured_") as td:
            td_path = Path(td)
            stl_path = td_path / "mesh.stl"
            stl_path.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
            csv_path = td_path / "input.csv"

            row = self._base_row("mesh.stl")
            row["case_id"] = ""
            pd.DataFrame([row]).to_csv(csv_path, index=False)

            with self.assertRaises(InputValidationError) as cm:
                read_cases(str(csv_path))
            issues = cm.exception.issues
            self.assertGreaterEqual(len(issues), 1)
            self.assertEqual(issues[0].row_number, 2)
            self.assertEqual(issues[0].field, "case_id")


if __name__ == "__main__":
    unittest.main()
