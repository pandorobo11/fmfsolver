from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyvista as pv

from fmfsolver.core.solver import _build_execution_order, build_case_signature, run_case, run_cases
from fmfsolver.io.csv_out import append_results_csv, write_results_csv


class TestSolverPipeline(unittest.TestCase):
    def test_build_execution_order_clusters_reusable_shield_cases(self):
        df = pd.DataFrame(
            [
                {
                    "case_id": "off_1",
                    "stl_path": "samples/stl/cube.stl",
                    "stl_scale_m_per_unit": 1.0,
                    "alpha_deg": 0.0,
                    "beta_or_bank_deg": 0.0,
                    "shielding_on": 0,
                },
                {
                    "case_id": "on_a1",
                    "stl_path": "samples/stl/cube.stl",
                    "stl_scale_m_per_unit": 1.0,
                    "alpha_deg": 5.0,
                    "beta_or_bank_deg": 0.0,
                    "shielding_on": 1,
                },
                {
                    "case_id": "on_b",
                    "stl_path": "samples/stl/cube.stl",
                    "stl_scale_m_per_unit": 1.0,
                    "alpha_deg": 10.0,
                    "beta_or_bank_deg": 0.0,
                    "shielding_on": 1,
                },
                {
                    "case_id": "on_a2",
                    "stl_path": "samples/stl/cube.stl",
                    "stl_scale_m_per_unit": 1.0,
                    "alpha_deg": 5.0,
                    "beta_or_bank_deg": 0.0,
                    "shielding_on": 1,
                },
            ]
        )
        order = _build_execution_order(df)
        ordered_ids = [str(df.iloc[i]["case_id"]) for i in order]
        self.assertEqual(ordered_ids, ["on_a1", "on_a2", "on_b", "off_1"])

    def test_run_cases_returns_rows_in_input_order_even_if_execution_reordered(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "first",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 10.0,
                        "beta_or_bank_deg": 0.0,
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
                        "shielding_on": 1,
                        "save_vtp_on": 0,
                        "save_npz_on": 0,
                        "out_dir": td,
                    },
                    {
                        "case_id": "second",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 5.0,
                        "beta_or_bank_deg": 0.0,
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
                        "shielding_on": 1,
                        "save_vtp_on": 0,
                        "save_npz_on": 0,
                        "out_dir": td,
                    },
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=1)
            got = res[res["scope"] == "total"]["case_id"].tolist()
            self.assertEqual(got, ["first", "second"])

    def test_run_cases_parallel_smoke(self):
        """Parallel path should execute and preserve input ordering."""
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "p1",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 0.0,
                        "beta_or_bank_deg": 0.0,
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
                    },
                    {
                        "case_id": "p2",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 5.0,
                        "beta_or_bank_deg": 0.0,
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
                    },
                    {
                        "case_id": "p3",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 10.0,
                        "beta_or_bank_deg": 0.0,
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
                    },
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=2)
            got = res[res["scope"] == "total"]["case_id"].tolist()
            self.assertEqual(got, ["p1", "p2", "p3"])

    def test_build_case_signature_numeric_normalization(self):
        row_int = {
            "case_id": "sig_case",
            "stl_path": "samples/stl/cube.stl",
            "stl_scale_m_per_unit": 1,
            "alpha_deg": 5,
            "beta_or_bank_deg": 0,
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
                "beta_or_bank_deg": 0.0,
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
            self.assertEqual(result["ray_backend_used"], "not_used")
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
                "beta_or_bank_deg": 0.0,
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
            self.assertEqual(result["ray_backend_used"], "not_used")
            self.assertTrue(str(result["run_started_at_utc"]).endswith("Z"))
            self.assertTrue(str(result["run_finished_at_utc"]).endswith("Z"))
            self.assertGreaterEqual(float(result["run_elapsed_s"]), 0.0)
            for key in ("S", "Ti_K", "CA", "CY", "CN", "CD", "CL"):
                self.assertTrue(math.isfinite(float(result[key])), key)

    def test_run_case_forwards_ray_backend_to_shielding(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "test_ray_backend",
                "stl_path": "samples/stl/cube.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_or_bank_deg": 0.0,
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
                "shielding_on": 1,
                "ray_backend": "rtree",
                "save_vtp_on": 0,
                "save_npz_on": 0,
                "out_dir": td,
            }

            def _zeros_mask(mesh, centers_m, Vhat, batch_size=None, ray_backend="auto"):
                _ = mesh
                _ = batch_size
                _ = Vhat
                _ = ray_backend
                return np.zeros(len(centers_m), dtype=bool), "rtree"

            with patch(
                "fmfsolver.core.solver.compute_shield_mask_with_backend",
                side_effect=_zeros_mask,
            ) as mocked:
                result = run_case(row, lambda _msg: None)

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(mocked.call_args.kwargs.get("ray_backend"), "rtree")
            self.assertEqual(result["ray_backend_used"], "rtree")

    def test_run_cases_cancel_before_start(self):
        df = pd.DataFrame(
            [
                {
                    "case_id": "cancel_case",
                    "stl_path": "samples/stl/cube.stl",
                    "stl_scale_m_per_unit": 1.0,
                    "alpha_deg": 0.0,
                    "beta_or_bank_deg": 0.0,
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
                    "out_dir": ".",
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Canceled by user."):
            run_cases(df, lambda _msg: None, workers=1, cancel_cb=lambda: True)

    def test_run_cases_progress_callback(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "progress_case",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 0.0,
                        "beta_or_bank_deg": 0.0,
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
                ]
            )
            ticks: list[tuple[int, int]] = []
            _ = run_cases(
                df,
                lambda _msg: None,
                workers=1,
                progress_cb=lambda done, total: ticks.append((done, total)),
            )
            self.assertEqual(ticks, [(1, 1)])

    def test_run_cases_chunk_callback(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": f"chunk_case_{i}",
                        "stl_path": "samples/stl/cube.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": float(i),
                        "beta_or_bank_deg": 0.0,
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
                    for i in range(3)
                ]
            )
            chunks: list[tuple[int, int, bool, int]] = []
            res = run_cases(
                df,
                lambda _msg: None,
                workers=1,
                flush_every_cases=2,
                chunk_cb=lambda chunk_df, done, total, is_final: chunks.append(
                    (done, total, is_final, len(chunk_df))
                ),
            )
            self.assertEqual(len(res), 3)
            self.assertEqual(chunks, [(2, 3, False, 2), (3, 3, True, 1)])

    def test_run_cases_multi_stl_emits_total_and_component_rows(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "multi_case",
                        "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 5.0,
                        "beta_or_bank_deg": 0.0,
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
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=1)
            total_rows = res[res["scope"] == "total"].reset_index(drop=True)
            component_rows = res[res["scope"] == "component"].reset_index(drop=True)

            self.assertEqual(len(total_rows), 1)
            self.assertEqual(len(component_rows), 2)
            self.assertEqual(set(component_rows["component_id"].astype(int).tolist()), {0, 1})
            self.assertEqual(int(total_rows.loc[0, "faces"]), int(component_rows["faces"].sum()))
            self.assertEqual(
                int(total_rows.loc[0, "shielded_faces"]),
                int(component_rows["shielded_faces"].sum()),
            )

            for coef in ("CA", "CY", "CN", "Cl", "Cm", "Cn", "CD", "CL"):
                self.assertAlmostEqual(
                    float(total_rows.loc[0, coef]),
                    float(component_rows[coef].sum()),
                    places=12,
                    msg=coef,
                )

    def test_write_results_csv_keeps_component_rows(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "multi_case",
                        "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 0.0,
                        "beta_or_bank_deg": 0.0,
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
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=1)
            out_csv = Path(td) / "result.csv"
            write_results_csv(str(out_csv), df, res)
            out_df = pd.read_csv(out_csv)

            self.assertEqual(len(out_df), len(res))
            self.assertIn("scope", out_df.columns)
            self.assertEqual(int((out_df["scope"] == "total").sum()), 1)
            self.assertEqual(int((out_df["scope"] == "component").sum()), 2)

    def test_append_results_csv_keeps_component_rows(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df = pd.DataFrame(
                [
                    {
                        "case_id": "multi_case",
                        "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                        "stl_scale_m_per_unit": 1.0,
                        "alpha_deg": 0.0,
                        "beta_or_bank_deg": 0.0,
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
                ]
            )
            res = run_cases(df, lambda _msg: None, workers=1)
            out_csv = Path(td) / "result_append.csv"
            append_results_csv(str(out_csv), df, res.iloc[:2].copy())
            append_results_csv(str(out_csv), df, res.iloc[2:].copy())
            out_df = pd.read_csv(out_csv)

            self.assertEqual(len(out_df), len(res))
            self.assertIn("scope", out_df.columns)
            self.assertEqual(int((out_df["scope"] == "total").sum()), 1)
            self.assertEqual(int((out_df["scope"] == "component").sum()), 2)

    def test_write_results_csv_preserves_input_case_order(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            df_in = pd.DataFrame(
                [
                    {"case_id": "case_a", "stl_path": "samples/stl/cube.stl"},
                    {"case_id": "case_b", "stl_path": "samples/stl/cube.stl"},
                ]
            )
            # Intentionally shuffled output rows (case_b first).
            df_out = pd.DataFrame(
                [
                    {"case_id": "case_b", "scope": "component", "component_id": 1, "CA": 21.0},
                    {"case_id": "case_b", "scope": "total", "component_id": "", "CA": 20.0},
                    {"case_id": "case_a", "scope": "component", "component_id": 1, "CA": 11.0},
                    {"case_id": "case_a", "scope": "total", "component_id": "", "CA": 10.0},
                ]
            )
            out_csv = Path(td) / "ordered.csv"
            write_results_csv(str(out_csv), df_in, df_out)
            out_df = pd.read_csv(out_csv)

            self.assertEqual(out_df["case_id"].tolist(), ["case_a", "case_a", "case_b", "case_b"])
            self.assertEqual(out_df["scope"].tolist(), ["total", "component", "total", "component"])

    def test_run_case_multi_stl_vtp_has_component_metadata(self):
        with tempfile.TemporaryDirectory(prefix="fmfsolver_test_") as td:
            row = {
                "case_id": "multi_case_vtp",
                "stl_path": "samples/stl/cube.stl;samples/stl/plate_offset_x2.stl",
                "stl_scale_m_per_unit": 1.0,
                "alpha_deg": 5.0,
                "beta_or_bank_deg": 0.0,
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
                "save_vtp_on": 1,
                "save_npz_on": 0,
                "out_dir": td,
            }
            result = run_case(row, lambda _msg: None)
            poly = pv.read(result["vtp_path"])

            self.assertIn("stl_index", poly.cell_data)
            self.assertEqual(len(poly.cell_data["stl_index"]), int(result["faces"]))
            self.assertIn("stl_count", poly.field_data)
            self.assertEqual(int(poly.field_data["stl_count"][0]), 2)
            self.assertIn("stl_paths_json", poly.field_data)


if __name__ == "__main__":
    unittest.main()
