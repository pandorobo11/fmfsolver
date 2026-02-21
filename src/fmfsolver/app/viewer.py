"""VTP viewer panel and visualization controls for the GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from ..core.sentman_core import resolve_attitude_to_vhat
from .ui_utils import format_case_text


class ViewerPanel(QtWidgets.QWidget):
    """Right-side panel that renders VTP results and camera controls."""

    log_message = QtCore.Signal(str)
    save_selected_images_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_layout = QtWidgets.QVBoxLayout(self)
        self._root_layout.setSpacing(6)
        self._root_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        try:
            self.plotter.enable_parallel_projection()
        except Exception:
            try:
                self.plotter.camera.parallel_projection = True
            except Exception:
                pass
        self._root_layout.addWidget(self.plotter.interactor, 6)

        self._init_controls()
        self._style_controls()
        self._build_controls_layout()
        self._connect_controls()

        self.df_cases = None
        self._poly: pv.PolyData | None = None
        self._display_case_row: dict | None = None
        self._overlay_actor = None
        self._default_view_vec = (-1, -1, 1)
        self._camera_initialized = False

    def _init_controls(self):
        """Create viewer control widgets."""
        self.cmb_scalar = QtWidgets.QComboBox()
        self.cmb_scalar.addItems(
            [
                "Cp_n",
                "shielded",
                "theta_deg",
                "area_m2",
                "center_x_stl_m",
                "center_y_stl_m",
                "center_z_stl_m",
                "stl_index",
            ]
        )

        self.chk_edges = QtWidgets.QCheckBox("Show edges")
        self.chk_edges.setChecked(True)
        self.chk_shield_transparent = QtWidgets.QCheckBox("Shielded transparent")
        self.chk_shield_transparent.setChecked(True)
        self.chk_overlay_text = QtWidgets.QCheckBox("Show info text")
        self.chk_overlay_text.setChecked(True)

        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(["jet", "viridis", "bwr"])
        self.cmb_cmap.setCurrentText("jet")

        self.edit_vmin = QtWidgets.QLineEdit()
        self.edit_vmax = QtWidgets.QLineEdit()
        self.edit_vmin.setPlaceholderText("vmin (blank=auto)")
        self.edit_vmax.setPlaceholderText("vmax (blank=auto)")
        self.btn_auto_range = QtWidgets.QPushButton("Auto range")
        self.btn_open_vtp = QtWidgets.QPushButton("Open VTP...")

        self.btn_view_xp = QtWidgets.QPushButton("+X")
        self.btn_view_xn = QtWidgets.QPushButton("-X")
        self.btn_view_yp = QtWidgets.QPushButton("+Y")
        self.btn_view_yn = QtWidgets.QPushButton("-Y")
        self.btn_view_zp = QtWidgets.QPushButton("+Z")
        self.btn_view_zn = QtWidgets.QPushButton("-Z")
        self.btn_view_iso_1 = QtWidgets.QPushButton("-X -Y +Z")
        self.btn_view_iso_2 = QtWidgets.QPushButton("+X -Y -Z")
        self.btn_view_wind = QtWidgets.QPushButton("Wind +")
        self.btn_view_wind_rev = QtWidgets.QPushButton("Wind -")

        self.btn_save_image = QtWidgets.QPushButton("Save Image...")
        self.btn_save_selected_images = QtWidgets.QPushButton("Save Selected...")

        self.lbl_scalar = QtWidgets.QLabel("SCALAR")
        self.lbl_colormap = QtWidgets.QLabel("COLORMAP")
        self.lbl_options = QtWidgets.QLabel("DISPLAY")
        self.lbl_colorbar = QtWidgets.QLabel("COLORBAR")
        self.lbl_camera = QtWidgets.QLabel("CAMERA")
        self.lbl_export = QtWidgets.QLabel("EXPORT")

    def _style_controls(self):
        """Apply consistent sizing and style for control widgets."""
        row_labels = (
            self.lbl_scalar,
            self.lbl_options,
            self.lbl_colorbar,
            self.lbl_camera,
            self.lbl_export,
        )
        for lbl in row_labels:
            lbl.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            lbl.setStyleSheet("QLabel { color: #b8bcc2; font-weight: 600; }")
        self.lbl_colormap.setStyleSheet("QLabel { color: #b8bcc2; font-weight: 600; }")

        max_label_width = max(lbl.sizeHint().width() for lbl in row_labels)
        for lbl in row_labels:
            lbl.setFixedWidth(max_label_width)

        ref_height = self.btn_auto_range.sizeHint().height()
        axis_buttons = [
            self.btn_view_xp,
            self.btn_view_xn,
            self.btn_view_yp,
            self.btn_view_yn,
            self.btn_view_zp,
            self.btn_view_zn,
        ]
        preset_buttons = [
            self.btn_view_iso_1,
            self.btn_view_iso_2,
            self.btn_view_wind,
            self.btn_view_wind_rev,
        ]
        save_buttons = [self.btn_save_image, self.btn_save_selected_images]
        axis_width = max(b.sizeHint().width() for b in axis_buttons)
        preset_width = max(b.sizeHint().width() for b in preset_buttons)
        save_width = max(b.sizeHint().width() for b in save_buttons)

        for buttons, width in (
            (axis_buttons, axis_width),
            (preset_buttons, preset_width),
            (save_buttons, save_width),
        ):
            for b in buttons:
                b.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Fixed,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                b.setFixedWidth(width)
                b.setFixedHeight(ref_height)

        self.cmb_scalar.setMinimumWidth(145)
        self.cmb_cmap.setMinimumWidth(105)
        self.edit_vmin.setMinimumWidth(120)
        self.edit_vmax.setMinimumWidth(120)

    def _build_controls_layout(self):
        """Build and attach control rows under the VTP interactor."""
        ctrl = QtWidgets.QVBoxLayout()
        ctrl.setSpacing(3)
        ctrl.setContentsMargins(0, 0, 0, 0)

        display_row = QtWidgets.QHBoxLayout()
        display_row.setSpacing(8)
        display_row.setContentsMargins(0, 0, 0, 0)
        display_row.addWidget(self.lbl_scalar)
        display_row.addWidget(self.cmb_scalar)
        display_row.addSpacing(10)
        display_row.addWidget(self.lbl_colormap)
        display_row.addWidget(self.cmb_cmap)
        display_row.addStretch(1)
        display_row.addWidget(self.btn_open_vtp)

        display_options_row = QtWidgets.QHBoxLayout()
        display_options_row.setSpacing(8)
        display_options_row.setContentsMargins(0, 0, 0, 0)
        display_options_row.addWidget(self.lbl_options)
        display_options_controls = QtWidgets.QHBoxLayout()
        display_options_controls.setSpacing(18)
        display_options_controls.setContentsMargins(0, 0, 0, 0)
        display_options_controls.addWidget(self.chk_edges)
        display_options_controls.addWidget(self.chk_shield_transparent)
        display_options_controls.addWidget(self.chk_overlay_text)
        display_options_row.addLayout(display_options_controls)
        display_options_row.addStretch(1)

        colorbar_row = QtWidgets.QHBoxLayout()
        colorbar_row.setSpacing(8)
        colorbar_row.setContentsMargins(0, 0, 0, 0)
        colorbar_row.addWidget(self.lbl_colorbar)
        colorbar_row.addWidget(self.edit_vmin)
        colorbar_row.addWidget(self.edit_vmax)
        colorbar_row.addWidget(self.btn_auto_range)
        colorbar_row.addStretch(1)

        camera_row = QtWidgets.QHBoxLayout()
        camera_row.setSpacing(6)
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.addWidget(self.lbl_camera)
        camera_row.addWidget(self.btn_view_xp)
        camera_row.addWidget(self.btn_view_xn)
        camera_row.addWidget(self.btn_view_yp)
        camera_row.addWidget(self.btn_view_yn)
        camera_row.addWidget(self.btn_view_zp)
        camera_row.addWidget(self.btn_view_zn)
        camera_row.addSpacing(12)
        camera_row.addWidget(self.btn_view_iso_1)
        camera_row.addWidget(self.btn_view_iso_2)
        camera_row.addSpacing(12)
        camera_row.addWidget(self.btn_view_wind)
        camera_row.addWidget(self.btn_view_wind_rev)
        camera_row.addStretch(1)

        export_row = QtWidgets.QHBoxLayout()
        export_row.setSpacing(6)
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.addWidget(self.lbl_export)
        export_row.addWidget(self.btn_save_image)
        export_row.addWidget(self.btn_save_selected_images)
        export_row.addStretch(1)

        ctrl.addLayout(display_row)
        ctrl.addLayout(display_options_row)
        ctrl.addLayout(colorbar_row)
        ctrl.addLayout(camera_row)
        ctrl.addLayout(export_row)
        self._root_layout.addLayout(ctrl)

    def _connect_controls(self):
        """Connect widget signals to viewer actions."""
        self.btn_open_vtp.clicked.connect(self.open_vtp)
        self.cmb_scalar.currentTextChanged.connect(self.update_view)
        self.chk_edges.toggled.connect(self.update_view)
        self.chk_shield_transparent.toggled.connect(self.update_view)
        self.chk_overlay_text.toggled.connect(self.update_view)
        self.cmb_cmap.currentTextChanged.connect(self.update_view)
        self.edit_vmin.editingFinished.connect(self.update_view)
        self.edit_vmax.editingFinished.connect(self.update_view)
        self.btn_auto_range.clicked.connect(self.clear_range)
        self.btn_view_xp.clicked.connect(lambda: self.set_view_vector((1, 0, 0)))
        self.btn_view_xn.clicked.connect(lambda: self.set_view_vector((-1, 0, 0)))
        self.btn_view_yp.clicked.connect(lambda: self.set_view_vector((0, 1, 0)))
        self.btn_view_yn.clicked.connect(lambda: self.set_view_vector((0, -1, 0)))
        self.btn_view_zp.clicked.connect(lambda: self.set_view_vector((0, 0, 1)))
        self.btn_view_zn.clicked.connect(lambda: self.set_view_vector((0, 0, -1)))
        self.btn_view_iso_1.clicked.connect(self.set_view_iso_1)
        self.btn_view_iso_2.clicked.connect(self.set_view_iso_2)
        self.btn_view_wind.clicked.connect(self.set_view_wind)
        self.btn_view_wind_rev.clicked.connect(self.set_view_wind_reverse)
        self.btn_save_image.clicked.connect(self.save_view_image)
        self.btn_save_selected_images.clicked.connect(self.save_selected_images_requested.emit)

    def logln(self, s: str):
        """Emit a message to the shared GUI log."""
        self.log_message.emit(s)

    def set_cases_df(self, df):
        """Store loaded case table for overlay lookup by case_id."""
        self.df_cases = df

    def clear_range(self):
        """Reset scalar colorbar limits to automatic range."""
        self.edit_vmin.setText("")
        self.edit_vmax.setText("")
        self.update_view()

    def set_view_vector(self, vec):
        """Set camera direction to a Cartesian view vector."""
        self.plotter.view_vector(vec)
        self.plotter.render()

    def set_view_iso_1(self):
        """Set camera to ISO view ``(-X, -Y, +Z)``."""
        self.plotter.view_vector((-1, -1, 1))
        self.plotter.render()

    def set_view_iso_2(self):
        """Set camera to ISO view ``(+X, -Y, -Z)``."""
        self.plotter.view_vector((1, -1, -1))
        self.plotter.render()

    def _resolve_current_vhat(self) -> np.ndarray | None:
        """Resolve current freestream vector from loaded case metadata."""
        if self._display_case_row is None:
            return None
        try:
            alpha_deg = float(self._display_case_row.get("alpha_deg"))
            beta_deg = float(self._display_case_row.get("beta_or_bank_deg"))
            attitude_input = self._display_case_row.get("attitude_input", "beta_tan")
        except Exception:
            return None
        try:
            vhat, _, _, _ = resolve_attitude_to_vhat(alpha_deg, beta_deg, attitude_input)
            return vhat
        except Exception:
            return None

    def set_view_wind(self):
        """Set camera to freestream direction ``+Vhat`` for current case."""
        vhat = self._resolve_current_vhat()
        if vhat is None:
            self.logln("[WARN] Wind view is unavailable (case alpha/beta not found).")
            return
        # view_vector(v) places camera at center+v and looks to center, so use
        # -Vhat to look along +Vhat.
        self.plotter.view_vector(tuple((-vhat).tolist()))
        self.plotter.render()

    def set_view_wind_reverse(self):
        """Set camera to reverse freestream direction ``-Vhat`` for current case."""
        vhat = self._resolve_current_vhat()
        if vhat is None:
            self.logln("[WARN] Reverse-wind view is unavailable (case alpha/beta not found).")
            return
        self.plotter.view_vector(tuple(vhat.tolist()))
        self.plotter.render()

    def save_view_image(self):
        """Save the current viewport image to a user-selected file."""
        if self.plotter is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save View Image",
            str(Path.cwd()),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            self.plotter.screenshot(path)
            self.logln(f"[OK] Saved image: {path}")
        except Exception as e:
            self.logln(f"[ERROR] Failed to save image: {e}")

    def save_images_for_case_rows(self, rows: list[dict]):
        """Batch-save images for selected case rows that have existing VTP files."""
        if not rows:
            self.logln("[WARN] No selected cases.")
            return

        default_dir = Path.cwd() / "outputs" / "images"
        out_dir_str = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Save Selected Images",
            str(default_dir),
        )
        if not out_dir_str:
            return
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        skipped = 0
        total = len(rows)
        self.logln(f"[SAVE] Batch image export start: {total} case(s)")

        for row in rows:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                skipped += 1
                self.logln("[SKIP] Missing case_id in selected row.")
                continue

            case_out_dir = Path(str(row.get("out_dir", "outputs"))).expanduser()
            vtp_path = case_out_dir / f"{case_id}.vtp"
            if not vtp_path.exists():
                skipped += 1
                self.logln(f"[SKIP] VTP not found: {vtp_path}")
                continue

            try:
                poly = pv.read(str(vtp_path))
                self.load_vtp(str(vtp_path), poly=poly)
                image_path = out_dir / f"{case_id}.png"
                self.plotter.screenshot(str(image_path))
                saved += 1
                self.logln(f"[OK] Saved image: {image_path}")
            except Exception as e:
                skipped += 1
                self.logln(f"[ERROR] Failed to save image for '{case_id}': {e}")

            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 10
            )

        self.logln(f"[SAVE] Batch image export done: saved={saved}, skipped={skipped}")

    # -------------------------
    # VTP view
    # -------------------------
    def open_vtp(self):
        """Open a VTP file from disk and display it."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open VTP", str(Path.cwd()), "VTK PolyData (*.vtp)"
        )
        if not path:
            return
        self.load_vtp(path)

    def load_vtp(self, path: str, poly: pv.PolyData | None = None):
        """Load VTP data, resolve case context, and refresh rendering."""
        if poly is None:
            try:
                self._poly = pv.read(path)
            except Exception as e:
                self.logln(f"[ERROR] Failed to read VTP: {e}")
                self._poly = None
                return
        else:
            self._poly = poly

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
        """Determine colorbar limits from UI inputs and scalar data."""
        if scalar == "shielded":
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
        """Update the corner text overlay with the active case summary."""
        if self._overlay_actor is not None:
            try:
                self.plotter.remove_actor(self._overlay_actor)
            except Exception:
                pass
            self._overlay_actor = None

        if not self.chk_overlay_text.isChecked():
            return

        if self._display_case_row is not None:
            txt = format_case_text(self._display_case_row)
        else:
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

    def _capture_camera_state(self):
        """Capture the current camera state for redraw-preserving updates."""
        try:
            cam = self.plotter.camera
            if cam is None:
                return None
            return {
                "position": tuple(cam.position),
                "focal_point": tuple(cam.focal_point),
                "view_up": tuple(cam.up),
                "clipping_range": tuple(cam.clipping_range),
                "parallel_projection": bool(getattr(cam, "parallel_projection", True)),
                "parallel_scale": float(getattr(cam, "parallel_scale", 1.0)),
            }
        except Exception:
            return None

    def _restore_camera_state(self, state) -> bool:
        """Restore camera state captured by `_capture_camera_state`."""
        if not state:
            return False
        try:
            cam = self.plotter.camera
            cam.position = state["position"]
            cam.focal_point = state["focal_point"]
            cam.up = state["view_up"]
            cam.clipping_range = state["clipping_range"]
            cam.parallel_projection = state["parallel_projection"]
            cam.parallel_scale = state["parallel_scale"]
            return True
        except Exception:
            return False

    def update_view(self):
        """Redraw the current mesh with selected scalar, style, and camera."""
        if self._poly is None:
            return

        scalar = self.cmb_scalar.currentText()
        cmap = self.cmb_cmap.currentText()
        show_edges = self.chk_edges.isChecked()
        shield_transparent = self.chk_shield_transparent.isChecked()

        prev_camera = self._capture_camera_state() if self._camera_initialized else None
        self.plotter.clear()
        poly = self._poly

        sh = (
            np.asarray(poly.cell_data["shielded"]).astype(int)
            if "shielded" in poly.cell_data
            else None
        )
        clim = self._get_clim(poly, scalar)

        if sh is not None:
            exposed_ids = np.where(sh == 0)[0]
            shielded_ids = np.where(sh == 1)[0]

            exposed = poly.extract_cells(exposed_ids) if len(exposed_ids) else None
            shielded = poly.extract_cells(shielded_ids) if len(shielded_ids) else None

            if exposed is not None and exposed.n_cells > 0:
                self.plotter.add_mesh(
                    exposed,
                    scalars=scalar if scalar in exposed.cell_data else None,
                    cmap=cmap,
                    clim=clim,
                    show_edges=show_edges,
                    opacity=1.0,
                )
            if shielded is not None and shielded.n_cells > 0:
                opacity = 0.30 if shield_transparent else 1.0
                self.plotter.add_mesh(
                    shielded,
                    scalars=scalar if scalar in shielded.cell_data else None,
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
        restored = self._restore_camera_state(prev_camera)
        if not restored:
            self.plotter.reset_camera()
            if self._default_view_vec is not None:
                self.plotter.view_vector(self._default_view_vec)
        self._camera_initialized = True
        self.plotter.render()
