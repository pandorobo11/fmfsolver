"""Cases panel for input loading, execution, and logging."""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import pyvista as pv
from PySide6 import QtCore, QtWidgets

from ..core.solver import build_case_signature, run_cases
from ..io.csv_out import write_results_csv
from ..io.io_cases import read_cases


class CasesPanel(QtWidgets.QWidget):
    """Left-side GUI panel that manages case selection and execution."""

    vtp_loaded = QtCore.Signal(str, object)
    cases_updated = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.input_label = QtWidgets.QLabel("Input: (not selected)")
        self.btn_pick_input = QtWidgets.QPushButton("Select Input File")
        self.btn_run = QtWidgets.QPushButton("Run Selected Cases")
        self.btn_run.setEnabled(False)

        self.lbl_workers = QtWidgets.QLabel("Workers:")
        self.spin_workers = QtWidgets.QSpinBox()
        max_workers = os.cpu_count() or 1
        self.spin_workers.setRange(1, max_workers)
        self.spin_workers.setValue(1)

        self.case_table = QtWidgets.QTableWidget()
        self.case_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.case_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.case_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.case_table.setAlternatingRowColors(True)
        self.case_table.horizontalHeader().setStretchLastSection(True)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)

        layout.addWidget(self.input_label)
        layout.addWidget(self.btn_pick_input)
        layout.addWidget(QtWidgets.QLabel("Cases:"))
        layout.addWidget(self.case_table, 3)
        workers_layout = QtWidgets.QHBoxLayout()
        workers_layout.addWidget(self.lbl_workers)
        workers_layout.addWidget(self.spin_workers)
        workers_layout.addStretch(1)
        layout.addLayout(workers_layout)
        layout.addWidget(self.btn_run)
        layout.addWidget(QtWidgets.QLabel("Log:"))
        layout.addWidget(self.log, 2)

        self.df_cases: pd.DataFrame | None = None
        self.input_path: str | None = None

        self.btn_pick_input.clicked.connect(self.pick_input_file)
        self.btn_run.clicked.connect(self.run_selected)
        self.case_table.itemSelectionChanged.connect(self.on_case_selection_changed)

    def logln(self, s: str):
        """Append one log line to the panel log view."""
        self.log.appendPlainText(s)

    def pick_input_file(self):
        """Open a file picker, read case definitions, and refresh the table."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            str(Path.cwd()),
            "CSV/Excel (*.csv *.xlsx *.xlsm *.xls)",
        )
        if not path:
            return
        self.input_path = path
        self.input_label.setText(f"Input: {path}")

        try:
            self.df_cases = read_cases(path)
        except Exception as e:
            self.logln(f"[ERROR] Failed to read input file: {e}")
            self.df_cases = None
            self.btn_run.setEnabled(False)
            return

        self._populate_case_table()
        self.btn_run.setEnabled(True)
        self.logln(f"[OK] Loaded {len(self.df_cases)} case(s). Select and run.")
        self.cases_updated.emit(self.df_cases)

    def _populate_case_table(self):
        """Render loaded cases into the table widget."""
        if self.df_cases is None:
            self.case_table.clear()
            self.case_table.setRowCount(0)
            return

        def _mode_from_row(r: dict) -> str:
            if pd.notna(r.get("S")) and pd.notna(r.get("Ti_K")):
                return "A"
            if pd.notna(r.get("Mach")) and pd.notna(r.get("Altitude_km")):
                return "B"
            return "?"

        cols = ["case_id", "mode"] + [c for c in self.df_cases.columns if c != "case_id"]
        self.case_table.clear()
        self.case_table.setColumnCount(len(cols))
        self.case_table.setRowCount(len(self.df_cases))
        self.case_table.setHorizontalHeaderLabels(cols)

        for row_idx, (_, row) in enumerate(self.df_cases.iterrows()):
            row_dict = row.to_dict()
            row_dict["mode"] = _mode_from_row(row_dict)
            for col_idx, col in enumerate(cols):
                val = row_dict.get(col, "")
                text = "" if pd.isna(val) else str(val)
                item = QtWidgets.QTableWidgetItem(text)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, int(row_idx))
                self.case_table.setItem(row_idx, col_idx, item)

        self.case_table.resizeColumnsToContents()

    def on_case_selection_changed(self):
        """Auto-load a matching VTP for the first selected case, if available."""
        if self.df_cases is None:
            return
        sel = self.case_table.selectionModel().selectedRows()
        if not sel:
            return
        row_idx = sel[0].row()
        item = self.case_table.item(row_idx, 0)
        if item is None:
            return
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        row = self.df_cases.loc[int(idx)].to_dict()
        case_id = str(row.get("case_id", "")).strip()
        out_dir = Path(str(row.get("out_dir", "outputs"))).expanduser()
        if not case_id:
            return
        vtp_path = out_dir / f"{case_id}.vtp"
        if not vtp_path.exists():
            return
        try:
            poly = pv.read(str(vtp_path))
        except Exception as e:
            self.logln(f"[ERROR] Failed to read VTP: {e}")
            return

        expected = build_case_signature(row)
        actual = None
        try:
            if "case_signature" in poly.field_data:
                actual = str(poly.field_data["case_signature"][0])
        except Exception:
            actual = None

        if actual == expected:
            self.vtp_loaded.emit(str(vtp_path), poly)

    def run_selected(self):
        """Run selected rows (or all rows) and write result CSV to chosen path."""
        if self.df_cases is None or self.input_path is None:
            return

        sel = self.case_table.selectionModel().selectedRows()
        if sel:
            idxs = []
            for s in sel:
                item = self.case_table.item(s.row(), 0)
                if item is not None:
                    idxs.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
            df_sel = self.df_cases.loc[idxs].reset_index(drop=True)
        else:
            df_sel = self.df_cases.copy().reset_index(drop=True)

        out_summary = Path(self.input_path)
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = out_dir / f"{out_summary.stem}_result.csv"

        out_path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Results",
            str(default_path),
            "CSV (*.csv)",
        )
        if not out_path_str:
            self.logln("[SKIP] Result output canceled.")
            return

        out_path = Path(out_path_str)

        workers = int(self.spin_workers.value())
        self.logln(f"[RUN] Running {len(df_sel)} case(s)...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            res = run_cases(df_sel, self.logln, workers=workers)

            write_results_csv(str(out_path), df_sel, res)
            self.logln(f"[OK] Wrote results: {out_path}")

            if len(res) and str(res.loc[0, "vtp_path"]).strip():
                vtp_path = str(res.loc[0, "vtp_path"])
                try:
                    poly = pv.read(vtp_path)
                except Exception as e:
                    self.logln(f"[ERROR] Failed to read VTP: {e}")
                    return
                self.vtp_loaded.emit(vtp_path, poly)
        except Exception as e:
            self.logln(f"[ERROR] {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
