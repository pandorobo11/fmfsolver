from __future__ import annotations

import pandas as pd
import pyvista as pv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from PySide6 import QtWidgets

    from fmfsolver.app.viewer import (
        ViewerPanel,
        _poly_matches_case_row,
        _resolve_display_case_row,
    )
except ImportError as exc:  # pragma: no cover - depends on CI image GUI libs
    raise unittest.SkipTest(f"PySide6 GUI tests unavailable: {exc}")

from fmfsolver.core.solver import build_case_signature


class TestViewerVtpMatching(unittest.TestCase):
    def _base_row(self) -> dict:
        return {
            "case_id": "case_a",
            "stl_path": "samples/stl/cube.stl",
            "stl_scale_m_per_unit": 1.0,
            "S": 5.0,
            "Ti_K": 300.0,
            "Mach": float("nan"),
            "Altitude_km": float("nan"),
            "Tw_K": 300.0,
            "alpha_deg": 0.0,
            "beta_or_bank_deg": 0.0,
            "attitude_input": "beta_tan",
            "ref_x_m": 0.0,
            "ref_y_m": 0.0,
            "ref_z_m": 0.0,
            "Aref_m2": 1.0,
            "Lref_Cl_m": 1.0,
            "Lref_Cm_m": 1.0,
            "Lref_Cn_m": 1.0,
            "shielding_on": 0,
            "ray_backend": "auto",
        }

    def test_poly_matching_rejects_stale_signature_for_same_case_id(self):
        expected_row = self._base_row()
        stale_row = {**expected_row, "alpha_deg": 15.0}
        poly = pv.PolyData()
        poly.field_data["case_id"] = [expected_row["case_id"]]
        poly.field_data["case_signature"] = [build_case_signature(stale_row)]

        self.assertFalse(_poly_matches_case_row(poly, expected_row))

    def test_resolve_display_case_row_requires_matching_signature(self):
        expected_row = self._base_row()
        stale_row = {**expected_row, "alpha_deg": 15.0}
        df_cases = pd.DataFrame([expected_row])
        poly = pv.PolyData()
        poly.field_data["case_id"] = [expected_row["case_id"]]
        poly.field_data["case_signature"] = [build_case_signature(stale_row)]

        self.assertIsNone(_resolve_display_case_row(poly, df_cases))

    def test_batch_image_export_skips_stale_vtp(self):
        row = {**self._base_row(), "out_dir": ""}
        stale_row = {**row, "alpha_deg": 15.0}
        poly = pv.PolyData()
        poly.field_data["case_id"] = [row["case_id"]]
        poly.field_data["case_signature"] = [build_case_signature(stale_row)]

        logs: list[str] = []
        fake_viewer = SimpleNamespace(
            logln=logs.append,
            load_vtp=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("load_vtp should not be called for stale VTP")
            ),
            plotter=SimpleNamespace(
                screenshot=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("screenshot should not be called for stale VTP")
                )
            ),
        )

        with tempfile.TemporaryDirectory(prefix="fmfsolver_viewer_") as td:
            with (
                patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=td),
                patch("fmfsolver.app.viewer.pv.read", return_value=poly),
                patch.object(QtWidgets.QApplication, "processEvents", return_value=None),
            ):
                row["out_dir"] = td
                Path(td, f"{row['case_id']}.vtp").write_text("dummy", encoding="utf-8")
                ViewerPanel.save_images_for_case_rows(fake_viewer, [row])

        self.assertTrue(any("signature mismatch" in msg for msg in logs))


if __name__ == "__main__":
    unittest.main()
