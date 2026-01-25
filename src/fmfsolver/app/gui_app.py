from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore
import pyvista as pv
from pyvistaqt import QtInteractor

from ..io.io_excel import read_cases
from ..core.solver import run_cases
from ..io.excel_out import write_results_excel


def _format_case_text(row: dict) -> str:
    parts: list[str] = []

    def add(k, v):
        if v is None:
            return
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return
        parts.append(f"{k}={s}")

    add("case_id", row.get("case_id"))
    add("stl", row.get("stl_path"))
    add("scale", row.get("stl_scale_m_per_unit"))
    add("alpha", row.get("alpha_deg"))
    add("beta", row.get("beta_deg"))
    add("Tw", row.get("Tw_K"))
    add("Aref", row.get("Aref_m2"))
    add("Lcl", row.get("Lref_Cl_m"))
    add("Lcm", row.get("Lref_Cm_m"))
    add("Lcn", row.get("Lref_Cn_m"))
    add("ref", f"({row.get('ref_x_m')},{row.get('ref_y_m')},{row.get('ref_z_m')})")

    # Mode A/B inputs (as entered)
    if str(row.get("S")).strip() not in ("", "nan", "None") and str(row.get("Ti_K")).strip() not in ("", "nan", "None"):
        add("mode", "A")
        add("S", row.get("S"))
        add("Ti", row.get("Ti_K"))
    elif str(row.get("Mach")).strip() not in ("", "nan", "None") and str(row.get("Altitude_km")).strip() not in ("", "nan", "None"):
        add("mode", "B")
        add("Mach", row.get("Mach"))
        add("Alt_km", row.get("Altitude_km"))

    add("shield", row.get("shielding_on"))
    return " | ".join(parts)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentman FMF Solver (GUI)")
        self.resize(1480, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # -------------------------
        # Left panel
        # -------------------------
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        self.xlsx_label = QtWidgets.QLabel("Excel: (not selected)")
        self.btn_pick_xlsx = QtWidgets.QPushButton("Select Excel Input")
        self.btn_run = QtWidgets.QPushButton("Run Selected Cases")
        self.btn_run.setEnabled(False)

        self.case_list = QtWidgets.QListWidget()
        self.case_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.selected_case_label = QtWidgets.QLabel("Selected case: (none)")
        self.selected_case_label.setWordWrap(True)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)

        left_layout.addWidget(self.xlsx_label)
        left_layout.addWidget(self.btn_pick_xlsx)
        left_layout.addWidget(QtWidgets.QLabel("Cases:"))
        left_layout.addWidget(self.case_list, 3)
        left_layout.addWidget(QtWidgets.QLabel("Selected case info (list selection):"))
        left_layout.addWidget(self.selected_case_label, 0)
        left_layout.addWidget(self.btn_run)
        left_layout.addWidget(QtWidgets.QLabel("Log:"))
        left_layout.addWidget(self.log, 2)

        splitter.addWidget(left)

        # -------------------------
        # Right panel
        # -------------------------
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        self.plotter = QtInteractor(right)
        right_layout.addWidget(self.plotter.interactor, 6)

        ctrl = QtWidgets.QGridLayout()

        self.cmb_scalar = QtWidgets.QComboBox()
        self.cmb_scalar.addItems([
            "Cp_n", "shadowed", "theta_deg", "area_m2",
            "center_x_stl_m", "center_y_stl_m", "center_z_stl_m",
        ])

        self.chk_edges = QtWidgets.QCheckBox("Show edges")
        self.chk_edges.setChecked(True)

        self.chk_shadow_transparent = QtWidgets.QCheckBox("Shadowed transparent")
        self.chk_shadow_transparent.setChecked(True)

        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["jet", "viridis", "bwr"])
        self.cmb_cmap.setCurrentText("jet")

        self.edit_vmin = QtWidgets.QLineEdit()
        self.edit_vmax = QtWidgets.QLineEdit()
        self.edit_vmin.setPlaceholderText("vmin (blank=auto)")
        self.edit_vmax.setPlaceholderText("vmax (blank=auto)")
        self.btn_auto_range = QtWidgets.QPushButton("Auto range")

        self.btn_open_vtp = QtWidgets.QPushButton("Open VTP...")

        r = 0
        ctrl.addWidget(QtWidgets.QLabel("Scalar:"), r, 0)
        ctrl.addWidget(self.cmb_scalar, r, 1)
        ctrl.addWidget(self.chk_edges, r, 2)
        ctrl.addWidget(self.chk_shadow_transparent, r, 3)
        ctrl.addWidget(QtWidgets.QLabel("Colormap:"), r, 4)
        ctrl.addWidget(self.cmb_cmap, r, 5)
        ctrl.addWidget(self.btn_open_vtp, r, 6)

        r = 1
        ctrl.addWidget(QtWidgets.QLabel("Colorbar range:"), r, 0)
        ctrl.addWidget(self.edit_vmin, r, 1)
        ctrl.addWidget(self.edit_vmax, r, 2)
        ctrl.addWidget(self.btn_auto_range, r, 3)

        right_layout.addLayout(ctrl)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 4)

        # -------------------------
        # State
        # -------------------------
        self.df_cases: pd.DataFrame | None = None
        self.xlsx_path: str | None = None
        self._poly: pv.PolyData | None = None
        self._display_case_row: dict | None = None  # case info corresponding to currently displayed VTP
        self._overlay_actor = None

        # -------------------------
        # Signals
        # -------------------------
        self.btn_pick_xlsx.clicked.connect(self.pick_xlsx)
        self.btn_run.clicked.connect(self.run_selected)
        self.btn_open_vtp.clicked.connect(self.open_vtp)
        self.cmb_scalar.currentTextChanged.connect(self.update_view)
        self.chk_edges.toggled.connect(self.update_view)
        self.chk_shadow_transparent.toggled.connect(self.update_view)
        self.cmb_cmap.currentTextChanged.connect(self.update_view)
        self.edit_vmin.editingFinished.connect(self.update_view)
        self.edit_vmax.editingFinished.connect(self.update_view)
        self.btn_auto_range.clicked.connect(self.clear_range)
        self.case_list.currentItemChanged.connect(self.on_case_changed)

    # -------------------------
    # Utility
    # -------------------------
    def logln(self, s: str):
        self.log.appendPlainText(s)

    def clear_range(self):
        self.edit_vmin.setText("")
        self.edit_vmax.setText("")
        self.update_view()

    # -------------------------
    # Excel / Case list
    # -------------------------
    def pick_xlsx(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Excel Input", str(Path.cwd()), "Excel (*.xlsx *.xlsm)"
        )
        if not path:
            return
        self.xlsx_path = path
        self.xlsx_label.setText(f"Excel: {path}")

        try:
            self.df_cases = read_cases(path)
        except Exception as e:
            self.logln(f"[ERROR] Failed to read Excel: {e}")
            self.df_cases = None
            self.btn_run.setEnabled(False)
            return

        self.case_list.clear()
        for i, row in self.df_cases.iterrows():
            cid = str(row["case_id"])
            a = row.get("alpha_deg")
            b = row.get("beta_deg")
            mode = "A" if (pd.notna(row.get("S")) and pd.notna(row.get("Ti_K"))) else (
                "B" if (pd.notna(row.get("Mach")) and pd.notna(row.get("Altitude_km"))) else "?"
            )
            it = QtWidgets.QListWidgetItem(
                f"{cid} | mode={mode} | alpha={a} beta={b} | {row['stl_path']}"
            )
            it.setData(QtCore.Qt.ItemDataRole.UserRole, int(i))
            self.case_list.addItem(it)

        self.btn_run.setEnabled(True)
        self.logln(f"[OK] Loaded {len(self.df_cases)} case(s). Select and run.")
        self.selected_case_label.setText("Selected case: (none)")

    def on_case_changed(self, current, _prev):
        if current is None or self.df_cases is None:
            self.selected_case_label.setText("Selected case: (none)")
            return
        idx = current.data(QtCore.Qt.ItemDataRole.UserRole)
        row = self.df_cases.loc[int(idx)].to_dict()
        self.selected_case_label.setText(_format_case_text(row))

    # -------------------------
    # Run
    # -------------------------
    def _confirm_overwrite(self, out_path: Path) -> bool:
        if not out_path.exists():
            return True
        resp = QtWidgets.QMessageBox.question(
            self,
            "Overwrite?",
            f"Result file already exists:\n{out_path}\n\nOverwrite it?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return resp == QtWidgets.QMessageBox.StandardButton.Yes

    def run_selected(self):
        if self.df_cases is None or self.xlsx_path is None:
            return

        sel = self.case_list.selectedItems()
        if sel:
            idxs = [it.data(QtCore.Qt.ItemDataRole.UserRole) for it in sel]
            df_sel = self.df_cases.loc[idxs].reset_index(drop=True)
        else:
            df_sel = self.df_cases.copy().reset_index(drop=True)

        self.logln(f"[RUN] Running {len(df_sel)} case(s)...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            res = run_cases(df_sel, self.logln)

            out_summary = Path(self.xlsx_path)
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{out_summary.stem}_result.xlsx"

            if self._confirm_overwrite(out_path):
                write_results_excel(str(out_path), df_sel, res)
                self.logln(f"[OK] Wrote results: {out_path}")
            else:
                self.logln(f"[SKIP] Not overwriting existing file: {out_path}")

            # Auto load first vtp if exists
            if len(res) and str(res.loc[0, "vtp_path"]).strip():
                self.load_vtp(str(res.loc[0, "vtp_path"]))
        except Exception as e:
            self.logln(f"[ERROR] {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # -------------------------
    # VTP view
    # -------------------------
    def open_vtp(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open VTP", str(Path.cwd()), "VTK PolyData (*.vtp)"
        )
        if not path:
            return
        self.load_vtp(path)

    def load_vtp(self, path: str):
        try:
            self._poly = pv.read(path)
        except Exception as e:
            self.logln(f"[ERROR] Failed to read VTP: {e}")
            self._poly = None
            return

        # Determine which case this VTP corresponds to (for overlay)
        self._display_case_row = None
        case_id = None
        try:
            if "case_id" in self._poly.field_data:
                case_id = str(self._poly.field_data["case_id"][0])
        except Exception:
            case_id = None

        if case_id and self.df_cases is not None:
            m = self.df_cases[self.df_cases["case_id"].astype(str) == str(case_id)]
            if len(m):
                self._display_case_row = m.iloc[0].to_dict()

        self.logln(f"[VIEW] Loaded VTP: {path}")
        self.update_view()

    def _get_clim(self, poly: pv.PolyData, scalar: str):
        if scalar == "shadowed":
            return (0.0, 1.0)
        if scalar not in poly.cell_data:
            return None

        arr = np.asarray(poly.cell_data[scalar])

        vmin_txt = self.edit_vmin.text().strip()
        vmax_txt = self.edit_vmax.text().strip()
        if vmin_txt == "" and vmax_txt == "":
            return (float(np.nanmin(arr)), float(np.nanmax(arr)))

        try:
            vmin = float(vmin_txt) if vmin_txt != "" else float(np.nanmin(arr))
            vmax = float(vmax_txt) if vmax_txt != "" else float(np.nanmax(arr))
            if vmin == vmax:
                vmax = vmin + 1e-12
            return (vmin, vmax)
        except Exception:
            return (float(np.nanmin(arr)), float(np.nanmax(arr)))

    def _update_overlay(self):
        if self._overlay_actor is not None:
            try:
                self.plotter.remove_actor(self._overlay_actor)
            except Exception:
                pass
            self._overlay_actor = None

        if self._display_case_row is not None:
            txt = _format_case_text(self._display_case_row)
        else:
            # At least show case_id if embedded
            txt = ""
            if self._poly is not None:
                try:
                    if "case_id" in self._poly.field_data:
                        txt = f"case_id={self._poly.field_data['case_id'][0]}"
                except Exception:
                    txt = ""
            if txt == "":
                txt = "(no case info for displayed VTP)"

        self._overlay_actor = self.plotter.add_text(txt, position="upper_left", font_size=10)

    def update_view(self):
        if self._poly is None:
            return

        scalar = self.cmb_scalar.currentText()
        cmap = self.cmb_cmap.currentText()
        show_edges = self.chk_edges.isChecked()
        shadow_transparent = self.chk_shadow_transparent.isChecked()

        self.plotter.clear()
        poly = self._poly

        sh = np.asarray(poly.cell_data["shadowed"]).astype(int) if "shadowed" in poly.cell_data else None
        clim = self._get_clim(poly, scalar)

        if sh is not None:
            exposed_ids = np.where(sh == 0)[0]
            shadowed_ids = np.where(sh == 1)[0]

            exposed = poly.extract_cells(exposed_ids) if len(exposed_ids) else None
            shadowed = poly.extract_cells(shadowed_ids) if len(shadowed_ids) else None

            if exposed is not None and exposed.n_cells > 0:
                self.plotter.add_mesh(
                    exposed,
                    scalars=scalar if scalar in exposed.cell_data else None,
                    cmap=cmap,
                    clim=clim,
                    show_edges=show_edges,
                    opacity=1.0,
                )
            if shadowed is not None and shadowed.n_cells > 0:
                opacity = 0.30 if shadow_transparent else 1.0
                self.plotter.add_mesh(
                    shadowed,
                    scalars=scalar if scalar in shadowed.cell_data else None,
                    cmap=cmap,
                    clim=clim,
                    show_edges=show_edges,
                    opacity=opacity,
                )
        else:
            self.plotter.add_mesh(
                poly,
                scalars=scalar if scalar in poly.cell_data else None,
                cmap=cmap,
                clim=clim,
                show_edges=show_edges,
                opacity=1.0,
            )

        self._update_overlay()
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.render()


def launch():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
