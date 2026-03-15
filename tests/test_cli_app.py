from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fmfsolver.app.cli_app import main


class TestCliCheckpointOutput(unittest.TestCase):
    def test_cli_checkpoint_snapshots_do_not_duplicate_rows(self):
        df_in = pd.DataFrame(
            [
                {"case_id": "case_a", "stl_path": "samples/stl/cube.stl"},
                {"case_id": "case_b", "stl_path": "samples/stl/cube.stl"},
                {"case_id": "case_c", "stl_path": "samples/stl/cube.stl"},
            ]
        )
        snapshot_1 = pd.DataFrame(
            [
                {"case_id": "case_b", "scope": "total", "component_id": "", "CA": 2.0},
                {"case_id": "case_c", "scope": "total", "component_id": "", "CA": 3.0},
            ]
        )
        snapshot_2 = pd.DataFrame(
            [
                {"case_id": "case_a", "scope": "total", "component_id": "", "CA": 1.0},
                {"case_id": "case_b", "scope": "total", "component_id": "", "CA": 2.0},
                {"case_id": "case_c", "scope": "total", "component_id": "", "CA": 3.0},
            ]
        )

        def fake_run_cases(
            _df_run,
            _logfn,
            *,
            workers,
            flush_every_cases,
            chunk_cb,
        ):
            self.assertEqual(workers, 2)
            self.assertEqual(flush_every_cases, 2)
            chunk_cb(snapshot_1, 2, 3, False)
            chunk_cb(snapshot_2, 3, 3, True)
            raise RuntimeError("stop after checkpoints")

        with tempfile.TemporaryDirectory(prefix="fmfsolver_cli_") as td:
            input_path = Path(td) / "input.csv"
            output_path = Path(td) / "out.csv"
            input_path.write_text("placeholder\n", encoding="utf-8")

            with (
                patch("fmfsolver.app.cli_app.read_cases", return_value=df_in),
                patch("fmfsolver.app.cli_app.run_cases", side_effect=fake_run_cases),
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after checkpoints"):
                    main(
                        [
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                            "--workers",
                            "2",
                            "--flush-every-cases",
                            "2",
                        ]
                    )

            out_df = pd.read_csv(output_path)

        self.assertEqual(out_df["case_id"].astype(str).tolist(), ["case_a", "case_b", "case_c"])
        self.assertEqual(out_df["case_id"].astype(str).nunique(), 3)
        self.assertEqual(len(out_df), 3)

    def test_cli_default_output_path_uses_input_file_directory(self):
        df_in = pd.DataFrame([{"case_id": "case_a", "stl_path": "samples/stl/cube.stl"}])
        captured: dict[str, object] = {}

        def fake_run_cases(
            _df_run,
            _logfn,
            *,
            workers,
            flush_every_cases,
            chunk_cb,
        ):
            self.assertEqual(workers, 1)
            self.assertEqual(flush_every_cases, 0)
            self.assertIsNone(chunk_cb)
            return pd.DataFrame(
                [{"case_id": "case_a", "scope": "total", "component_id": "", "CA": 1.0}]
            )

        def fake_write_results_csv(out_path, _df_run, result_df):
            captured["out_path"] = Path(out_path)
            captured["result_df"] = result_df.copy()

        with tempfile.TemporaryDirectory(prefix="fmfsolver_cli_default_out_") as td:
            input_dir = Path(td) / "inputs"
            input_dir.mkdir()
            input_path = input_dir / "input.csv"
            input_path.write_text("placeholder\n", encoding="utf-8")

            with (
                patch("fmfsolver.app.cli_app.read_cases", return_value=df_in),
                patch("fmfsolver.app.cli_app.run_cases", side_effect=fake_run_cases),
                patch("fmfsolver.app.cli_app.write_results_csv", side_effect=fake_write_results_csv),
                patch("builtins.print"),
            ):
                rc = main(["--input", str(input_path), "--flush-every-cases", "0"])

        self.assertEqual(rc, 0)
        self.assertEqual(
            Path(captured["out_path"]).resolve(),
            (input_dir / "outputs" / "input_result.csv").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
