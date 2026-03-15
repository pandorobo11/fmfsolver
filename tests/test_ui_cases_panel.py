from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets

    from fmfsolver.app.ui_cases import CasesPanel
except ImportError as exc:  # pragma: no cover - depends on CI image GUI libs
    raise unittest.SkipTest(f"PySide6 GUI tests unavailable: {exc}")

from fmfsolver.io.io_cases import InputValidationError, ValidationIssue


class TestCasesPanelState(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _base_loaded_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "case_id": "case_ok",
                    "stl_path": str(Path("samples/stl/cube.stl").resolve()),
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
                    "out_dir": "outputs",
                    "save_vtp_on": 1,
                    "save_npz_on": 0,
                }
            ]
        )

    def test_pick_input_file_clears_old_state_after_validation_error(self):
        panel = CasesPanel()
        panel.df_cases = self._base_loaded_df()
        panel.input_path = "/tmp/previous.csv"
        panel.input_value.setText(panel.input_path)
        panel._populate_case_table()
        panel.btn_run.setEnabled(True)

        issues = [ValidationIssue(2, "bad_case", "stl_path", "is invalid.")]
        with tempfile.TemporaryDirectory(prefix="fmfsolver_ui_cases_") as td:
            with (
                patch.object(
                    QtWidgets.QFileDialog,
                    "getOpenFileName",
                    return_value=(str(Path(td) / "bad.csv"), "CSV"),
                ),
                patch(
                    "fmfsolver.app.ui_cases.read_cases",
                    side_effect=InputValidationError(issues),
                ),
                patch("fmfsolver.app.ui_cases._ValidationIssuesDialog.exec", return_value=0),
            ):
                panel.pick_input_file()

        self.assertIsNone(panel.df_cases)
        self.assertIsNone(panel.input_path)
        self.assertEqual(panel.input_value.text(), "")
        self.assertEqual(panel.case_table.rowCount(), 0)
        self.assertEqual(panel.case_table.columnCount(), 0)
        self.assertFalse(panel.btn_run.isEnabled())
        self.assertEqual(panel.selected_case_rows(), [])

    def test_run_selected_default_output_path_uses_input_file_directory(self):
        panel = CasesPanel()
        panel.df_cases = self._base_loaded_df()
        panel.input_path = "/tmp/fmfsolver_inputs/input.csv"
        panel.input_value.setText(panel.input_path)
        panel._populate_case_table()

        captured: dict[str, str] = {}

        def fake_get_save_file_name(_parent, _title, default_path, _filter):
            captured["default_path"] = default_path
            return ("", "")

        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", side_effect=fake_get_save_file_name):
            panel.run_selected()

        self.assertEqual(
            captured["default_path"],
            "/tmp/fmfsolver_inputs/outputs/input_result.csv",
        )


if __name__ == "__main__":
    unittest.main()
