"""Cases panel for input loading, execution, and logging."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pandas as pd
import pyvista as pv
from PySide6 import QtCore, QtWidgets

from ..core.solver import build_case_signature, run_cases
from ..io.csv_out import append_results_csv, write_results_csv
from ..io.io_cases import InputValidationError, read_cases


class _CaseRunWorker(QtCore.QObject):
    """Background worker running selected cases without blocking the GUI."""

    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(
        self,
        df_selected: pd.DataFrame,
        workers: int,
        out_path: Path,
        flush_every_cases: int = 100,
    ):
        super().__init__()
        self._df_selected = df_selected
        self._workers = workers
        self._out_path = out_path
        self._flush_every_cases = int(flush_every_cases)
        self._cancel_event = threading.Event()

    @QtCore.Slot()
    def run(self):
        """Run the solver pipeline and emit completion/progress signals."""
        try:
            if self._out_path.exists():
                self._out_path.unlink()

            def on_chunk(chunk_df, done: int, total: int, is_final: bool):
                append_results_csv(str(self._out_path), self._df_selected, chunk_df)
                phase = "final" if is_final else "checkpoint"
                self.log.emit(f"[SAVE] {phase} {done}/{total} -> {self._out_path}")

            result = run_cases(
                self._df_selected,
                self.log.emit,
                workers=self._workers,
                progress_cb=self._emit_progress,
                cancel_cb=self._cancel_event.is_set,
                flush_every_cases=self._flush_every_cases,
                chunk_cb=on_chunk if self._flush_every_cases > 0 else None,
            )
        except Exception as e:
            if str(e) == "Canceled by user.":
                self.canceled.emit()
            else:
                self.failed.emit(str(e))
            return

        if self._cancel_event.is_set():
            self.canceled.emit()
            return

        self.completed.emit(result)

    def cancel(self):
        """Request cooperative cancellation."""
        self._cancel_event.set()

    def _emit_progress(self, done: int, total: int):
        self.progress.emit(done, total)


class _ValidationIssuesDialog(QtWidgets.QDialog):
    """Tabular dialog for input validation issues."""

    def __init__(self, file_path: str, issues, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Validation Errors")
        self.resize(980, 420)
        layout = QtWidgets.QVBoxLayout(self)

        summary = QtWidgets.QLabel(
            f"Failed to load input file:\n{file_path}\n\nValidation issues: {len(issues)}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.table = QtWidgets.QTableWidget(len(issues), 4)
        self.table.setHorizontalHeaderLabels(["Row", "Case ID", "Field", "Message"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        self._issues = list(issues)
        for row, issue in enumerate(self._issues):
            row_text = "" if issue.row_number is None else str(issue.row_number)
            case_id_text = issue.case_id or ""
            field_text = issue.field or ""
            message_text = issue.message
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(row_text))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(case_id_text))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(field_text))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(message_text))

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_copy = QtWidgets.QPushButton("Copy")
        self.btn_close = QtWidgets.QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.btn_copy.clicked.connect(self._copy_issues_to_clipboard)
        self.btn_close.clicked.connect(self.accept)

    def _copy_issues_to_clipboard(self):
        lines = ["row\tcase_id\tfield\tmessage"]
        for issue in self._issues:
            row_text = "" if issue.row_number is None else str(issue.row_number)
            lines.append(
                f"{row_text}\t{issue.case_id or ''}\t{issue.field or ''}\t{issue.message}"
            )
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))


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
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
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
        run_layout = QtWidgets.QHBoxLayout()
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_cancel)
        layout.addLayout(run_layout)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        layout.addWidget(self.progress)
        layout.addWidget(QtWidgets.QLabel("Log:"))
        layout.addWidget(self.log, 2)

        self.df_cases: pd.DataFrame | None = None
        self.input_path: str | None = None
        self._run_thread: QtCore.QThread | None = None
        self._run_worker: _CaseRunWorker | None = None
        self._run_df_selected: pd.DataFrame | None = None
        self._run_out_path: Path | None = None

        self.btn_pick_input.clicked.connect(self.pick_input_file)
        self.btn_run.clicked.connect(self.run_selected)
        self.btn_cancel.clicked.connect(self.cancel_run)
        self.case_table.itemSelectionChanged.connect(self.on_case_selection_changed)

    def logln(self, s: str):
        """Append one log line to the panel log view."""
        self.log.appendPlainText(s)

    def selected_case_rows(self) -> list[dict]:
        """Return currently selected case rows as dicts in table order."""
        if self.df_cases is None:
            return []
        sel = self.case_table.selectionModel().selectedRows()
        if not sel:
            return []
        idxs: list[int] = []
        for s in sel:
            item = self.case_table.item(s.row(), 0)
            if item is None:
                continue
            idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if idx is None:
                continue
            idxs.append(int(idx))
        if not idxs:
            return []
        return self.df_cases.loc[idxs].to_dict(orient="records")

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

        try:
            loaded = read_cases(path)
        except InputValidationError as e:
            self.logln(f"[ERROR] Invalid input file: {len(e.issues)} issue(s).")
            dialog = _ValidationIssuesDialog(path, e.issues, self)
            dialog.exec()
            self.btn_run.setEnabled(self.df_cases is not None)
            return
        except Exception as e:
            self.logln(f"[ERROR] Failed to read input file: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Input Read Error",
                f"Failed to read input file:\n{path}\n\n{e}",
            )
            self.btn_run.setEnabled(self.df_cases is not None)
            return

        self.input_path = path
        self.input_label.setText(f"Input: {path}")
        self.df_cases = loaded
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
        headers = ["stl_name" if c == "stl_path" else c for c in cols]
        self.case_table.setHorizontalHeaderLabels(headers)
        stl_col_idx = cols.index("stl_path") if "stl_path" in cols else -1

        for row_idx, (_, row) in enumerate(self.df_cases.iterrows()):
            row_dict = row.to_dict()
            row_dict["mode"] = _mode_from_row(row_dict)
            for col_idx, col in enumerate(cols):
                val = row_dict.get(col, "")
                text = "" if pd.isna(val) else str(val)
                display_text = text
                if col == "stl_path" and text:
                    display_text = self._format_stl_name(text)
                item = QtWidgets.QTableWidgetItem(display_text)
                if col == "stl_path" and text:
                    item.setToolTip(text)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, int(row_idx))
                self.case_table.setItem(row_idx, col_idx, item)

        self.case_table.resizeColumnsToContents()
        if stl_col_idx >= 0:
            self.case_table.setColumnWidth(stl_col_idx, 220)

    @staticmethod
    def _format_stl_name(stl_path_value: str) -> str:
        """Format one or more STL paths as a compact filename list."""
        parts = [p.strip() for p in stl_path_value.split(";") if p.strip()]
        if not parts:
            return ""
        names = [Path(p).name for p in parts]
        return ", ".join(names)

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
        if self._run_thread is not None:
            return

        sel = self.case_table.selectionModel().selectedRows()
        if sel:
            rows = self.selected_case_rows()
            df_sel = pd.DataFrame(rows).reset_index(drop=True)
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
        self._run_df_selected = df_sel.copy().reset_index(drop=True)
        self._run_out_path = out_path
        self.progress.setRange(0, len(self._run_df_selected))
        self.progress.setValue(0)
        self.progress.setFormat(f"0/{len(self._run_df_selected)}")
        self._set_running_state(True)
        self.logln(f"[RUN] Running {len(self._run_df_selected)} case(s)...")

        self._run_thread = QtCore.QThread(self)
        self._run_worker = _CaseRunWorker(
            self._run_df_selected,
            workers,
            out_path=self._run_out_path,
            flush_every_cases=100,
        )
        self._run_worker.moveToThread(self._run_thread)

        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self.logln)
        self._run_worker.progress.connect(self._on_run_progress)
        self._run_worker.completed.connect(self._on_run_completed)
        self._run_worker.failed.connect(self._on_run_failed)
        self._run_worker.canceled.connect(self._on_run_canceled)

        self._run_worker.completed.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_worker.canceled.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_worker)

        self._run_thread.start()

    def cancel_run(self):
        """Request cancellation of an ongoing run."""
        if self._run_worker is None:
            return
        self._run_worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.logln("[CANCEL] Cancellation requested...")

    def _on_run_progress(self, done: int, total: int):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}")

    def _on_run_completed(self, res):
        if self._run_df_selected is None or self._run_out_path is None:
            self.logln("[ERROR] Missing run context when finishing.")
            self._set_running_state(False)
            return
        try:
            write_results_csv(str(self._run_out_path), self._run_df_selected, res)
            self.logln(f"[OK] Wrote results: {self._run_out_path}")
            total = len(self._run_df_selected)
            self.progress.setRange(0, max(total, 1))
            self.progress.setValue(total)
            self.progress.setFormat(f"{total}/{total}")

            if len(res) and str(res.loc[0, "vtp_path"]).strip():
                vtp_path = str(res.loc[0, "vtp_path"])
                try:
                    poly = pv.read(vtp_path)
                except Exception as e:
                    self.logln(f"[ERROR] Failed to read VTP: {e}")
                    self._set_running_state(False)
                    return
                self.vtp_loaded.emit(vtp_path, poly)
        except Exception as e:
            self.logln(f"[ERROR] {e}")
        finally:
            self._set_running_state(False)

    def _on_run_failed(self, message: str):
        self.logln(f"[ERROR] {message}")
        self.progress.setFormat("Failed")
        self._set_running_state(False)

    def _on_run_canceled(self):
        self.logln("[CANCEL] Run canceled.")
        self.progress.setFormat("Canceled")
        self._set_running_state(False)

    def _set_running_state(self, running: bool):
        self.btn_pick_input.setEnabled(not running)
        self.spin_workers.setEnabled(not running)
        self.case_table.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_run.setEnabled((not running) and (self.df_cases is not None))

    def _cleanup_run_worker(self):
        if self._run_worker is not None:
            self._run_worker.deleteLater()
            self._run_worker = None
        if self._run_thread is not None:
            self._run_thread.deleteLater()
            self._run_thread = None
        self._run_df_selected = None
        self._run_out_path = None
