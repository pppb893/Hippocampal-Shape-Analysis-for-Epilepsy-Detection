import os
import glob
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QRadioButton, QButtonGroup, QProgressBar, QCheckBox, QSlider, QFrame,
    QScrollArea, QApplication, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from models.predictor import HippocampalPredictor, parse_coef_file


class ResultPanel(QWidget):
    """
    Result & Diagnostic Panel
    Inference-only evaluation using trained ResNet1D models and interactive
    3D Grad-CAM attention and deformation heatmap visualization.
    """
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str)
    signal_gradcam_mesh_requested = pyqtSignal(str, str, str, str, str, float)
    signal_patient_overlay_requested = pyqtSignal(str, bool, float, str)
    signal_clear_gradcam_requested = pyqtSignal()

    def __init__(self, parent=None, get_input_folder=None, get_output_folder=None):
        super().__init__(parent)
        self.get_input_folder = get_input_folder
        self.get_output_folder = get_output_folder

        self.predictor = HippocampalPredictor()
        self.current_patient_data = None
        self.last_prediction_results = None

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # Scroll area to handle smaller screens gracefully
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)

        # ---------------------------------------------------------------------
        # 1. Header & Overview
        # ---------------------------------------------------------------------
        header_group = QGroupBox("1. Diagnostic Model & Input Data")
        header_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 1px solid #ced6e0;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        h_layout = QVBoxLayout(header_group)
        h_layout.setSpacing(8)

        # Path input
        path_row = QHBoxLayout()
        path_lbl = QLabel("SPHARM Dir:")
        path_lbl.setFixedWidth(75)
        path_lbl.setStyleSheet("font-size: 11px; color: #2c3e50;")
        self.spharm_dir_input = QLineEdit()
        self.spharm_dir_input.setPlaceholderText("Select directory with SPHARM .coef / .vtk results...")
        self.spharm_dir_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ced6e0;
                border-radius: 3px;
                padding: 4px 6px;
                font-size: 11px;
                background: #ffffff;
            }
            QLineEdit:focus { border: 1px solid #3498db; }
        """)
        self.spharm_dir_input.textChanged.connect(self.populate_patients_table)

        browse_btn = QPushButton("📁 Browse")
        browse_btn.setFixedHeight(26)
        browse_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                padding: 2px 10px;
            }
            QPushButton:hover { background: #e4e7eb; }
        """)
        browse_btn.clicked.connect(self.browse_spharm_dir)

        reload_btn = QPushButton("🔄 Refresh")
        reload_btn.setFixedHeight(26)
        reload_btn.setStyleSheet(browse_btn.styleSheet())
        reload_btn.clicked.connect(self.populate_patients_table)

        path_row.addWidget(path_lbl)
        path_row.addWidget(self.spharm_dir_input)
        path_row.addWidget(browse_btn)
        path_row.addWidget(reload_btn)
        h_layout.addLayout(path_row)

        # Model architecture badge
        model_badge = QLabel("Architecture: <b>ResNet1D (1D-CNN) + PLS-DA</b> | Weights: <b>Trained & Frozen</b>")
        model_badge.setStyleSheet("font-size: 10px; color: #576574; background: #f8f9fa; padding: 4px 6px; border-radius: 3px; border: 1px solid #e9ecef;")
        h_layout.addWidget(model_badge)

        container_layout.addWidget(header_group)

        # ---------------------------------------------------------------------
        # 2. Patients / SPHARM Meshes Selection Table
        # ---------------------------------------------------------------------
        patient_group = QGroupBox("2. Select Subject for Evaluation")
        patient_group.setStyleSheet(header_group.styleSheet())
        p_layout = QVBoxLayout(patient_group)
        p_layout.setSpacing(6)

        self.patient_table = QTableWidget(0, 4)
        self.patient_table.setHorizontalHeaderLabels(["Subject Name", "Left (.coef)", "Right (.coef)", "Mesh (.vtk)"])
        self.patient_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.patient_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.patient_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.patient_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.patient_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dcdde1;
                gridline-color: #ecf0f1;
                font-size: 11px;
                background-color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #dcdde1;
                font-size: 11px;
            }
        """)
        self.patient_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.patient_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.patient_table.itemSelectionChanged.connect(self.on_patient_selected)
        self.patient_table.setFixedHeight(120)
        p_layout.addWidget(self.patient_table)

        # Prediction evaluation target options
        eval_opts_layout = QHBoxLayout()
        eval_opts_layout.addWidget(QLabel("Target Hemisphere:"))
        self.side_eval_group = QButtonGroup(self)
        self.rb_both = QRadioButton("Bilateral (Both)")
        self.rb_left = QRadioButton("Left (LH)")
        self.rb_right = QRadioButton("Right (RH)")
        self.rb_both.setChecked(True)
        self.side_eval_group.addButton(self.rb_both, 0)
        self.side_eval_group.addButton(self.rb_left, 1)
        self.side_eval_group.addButton(self.rb_right, 2)
        eval_opts_layout.addWidget(self.rb_both)
        eval_opts_layout.addWidget(self.rb_left)
        eval_opts_layout.addWidget(self.rb_right)
        eval_opts_layout.addStretch()
        p_layout.addLayout(eval_opts_layout)

        # Run Prediction Action Button
        self.predict_btn = QPushButton("⚡ Run ResNet Prediction & 3D Grad-CAM")
        self.predict_btn.setFixedHeight(34)
        self.predict_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a99d8, stop:1 #2471a3);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #ecf0f1;
            }
        """)
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        p_layout.addWidget(self.predict_btn)

        container_layout.addWidget(patient_group)

        # ---------------------------------------------------------------------
        # 3. Diagnostic Report Card
        # ---------------------------------------------------------------------
        diag_group = QGroupBox("3. Diagnostic Scorecard & Lateralization")
        diag_group.setStyleSheet(header_group.styleSheet())
        d_layout = QVBoxLayout(diag_group)
        d_layout.setSpacing(8)

        # Prominent Result Status Banner
        self.status_banner = QFrame()
        self.status_banner.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px dashed #b2bec3;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        banner_layout = QVBoxLayout(self.status_banner)
        banner_layout.setContentsMargins(8, 6, 8, 6)
        banner_layout.setSpacing(4)

        self.diagnosis_badge = QLabel("Awaiting Evaluation")
        self.diagnosis_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diagnosis_badge.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.diagnosis_badge.setStyleSheet("color: #7f8c8d;")
        banner_layout.addWidget(self.diagnosis_badge)

        self.risk_badge = QLabel("Select a subject above and click 'Run ResNet Prediction'")
        self.risk_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.risk_badge.setStyleSheet("font-size: 11px; color: #576574;")
        banner_layout.addWidget(self.risk_badge)
        d_layout.addWidget(self.status_banner)

        # Confidence Bar
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Epilepsy Probability:"))
        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setValue(0)
        self.prob_bar.setTextVisible(True)
        self.prob_bar.setFormat("%v%")
        self.prob_bar.setFixedHeight(18)
        self.prob_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ced6e0;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
                font-size: 10px;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 2px;
            }
        """)
        conf_layout.addWidget(self.prob_bar)
        d_layout.addLayout(conf_layout)

        # Lateralization Meter (Left vs Right)
        lat_box = QFrame()
        lat_box.setStyleSheet("background: #f1f2f6; border: 1px solid #dcdde1; border-radius: 4px; padding: 6px;")
        lat_layout = QVBoxLayout(lat_box)
        lat_layout.setContentsMargins(4, 4, 4, 4)
        lat_layout.setSpacing(3)

        self.lat_title_lbl = QLabel("Hemispheric Lateralization Breakdown:")
        self.lat_title_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        self.lat_detail_lbl = QLabel("• Left: N/A  |  • Right: N/A  |  • Primary Focus: N/A")
        self.lat_detail_lbl.setStyleSheet("font-size: 11px; color: #34495e;")
        lat_layout.addWidget(self.lat_title_lbl)
        lat_layout.addWidget(self.lat_detail_lbl)
        d_layout.addWidget(lat_box)

        container_layout.addWidget(diag_group)

        # ---------------------------------------------------------------------
        # 4. 3D Grad-CAM & Deformation Heatmap Controls
        # ---------------------------------------------------------------------
        cam_group = QGroupBox("4. 3D Grad-CAM & Atrophy Visualization")
        cam_group.setStyleSheet(header_group.styleSheet())
        c_layout = QVBoxLayout(cam_group)
        c_layout.setSpacing(6)

        # Active view hemisphere
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Render Side:"))
        self.rb_cam_left = QRadioButton("Left Hippocampus (LH)")
        self.rb_cam_right = QRadioButton("Right Hippocampus (RH)")
        self.rb_cam_left.setChecked(True)
        self.cam_side_group = QButtonGroup(self)
        self.cam_side_group.addButton(self.rb_cam_left, 0)
        self.cam_side_group.addButton(self.rb_cam_right, 1)
        self.cam_side_group.buttonClicked.connect(self.update_3d_view)
        side_row.addWidget(self.rb_cam_left)
        side_row.addWidget(self.rb_cam_right)
        side_row.addStretch()
        c_layout.addLayout(side_row)

        # Visualization mode options
        c_layout.addWidget(QLabel("Colormap Heatmap Mode:"))
        self.rb_gradcam = QRadioButton("🔴 ResNet Grad-CAM Attention (Subfield Importance)")
        self.rb_signed = QRadioButton("🔵 Signed Atrophy Map (Blue: Inward | Red: Expansion)")
        self.rb_dist = QRadioButton("🟡 Deformation Magnitude (Displacement in mm)")
        self.rb_gradcam.setChecked(True)

        self.cam_mode_group = QButtonGroup(self)
        self.cam_mode_group.addButton(self.rb_gradcam, 0)
        self.cam_mode_group.addButton(self.rb_signed, 1)
        self.cam_mode_group.addButton(self.rb_dist, 2)
        self.cam_mode_group.buttonClicked.connect(self.update_3d_view)

        c_layout.addWidget(self.rb_gradcam)
        c_layout.addWidget(self.rb_signed)
        c_layout.addWidget(self.rb_dist)

        # Mesh Template variant: Mean vs Atrophy Milestones
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Atrophy Stage:"))
        self.rb_mean = QRadioButton("Mean (0 SD)")
        self.rb_minus2sd = QRadioButton("Atrophy (-2 SD)")
        self.rb_minus3sd = QRadioButton("Severe (-3 SD)")
        self.rb_mean.setChecked(True)
        self.stage_group = QButtonGroup(self)
        self.stage_group.addButton(self.rb_mean, 0)
        self.stage_group.addButton(self.rb_minus2sd, 1)
        self.stage_group.addButton(self.rb_minus3sd, 2)
        self.stage_group.buttonClicked.connect(self.update_3d_view)
        stage_row.addWidget(self.rb_mean)
        stage_row.addWidget(self.rb_minus2sd)
        stage_row.addWidget(self.rb_minus3sd)
        stage_row.addStretch()
        c_layout.addLayout(stage_row)

        # Patient Mesh Overlay Toggle
        overlay_row = QHBoxLayout()
        self.patient_overlay_cb = QCheckBox("Overlay Patient Mesh (Translucent Cyan)")
        self.patient_overlay_cb.setChecked(True)
        self.patient_overlay_cb.setStyleSheet("font-weight: bold; color: #16a085; font-size: 11px;")
        self.patient_overlay_cb.toggled.connect(self.on_patient_overlay_toggled)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setFixedWidth(80)
        self.opacity_slider.valueChanged.connect(self.on_patient_overlay_toggled)

        overlay_row.addWidget(self.patient_overlay_cb)
        overlay_row.addWidget(QLabel("Opacity:"))
        overlay_row.addWidget(self.opacity_slider)
        c_layout.addLayout(overlay_row)

        # Update 3D View Button
        render_btn = QPushButton("👁️ Refresh 3D Heatmap View")
        render_btn.setFixedHeight(28)
        render_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #e4e7eb; }
        """)
        render_btn.clicked.connect(self.update_3d_view)
        c_layout.addWidget(render_btn)

        container_layout.addWidget(cam_group)

        # ---------------------------------------------------------------------
        # 5. Summary & Clipboard
        # ---------------------------------------------------------------------
        bottom_row = QHBoxLayout()
        self.copy_summary_btn = QPushButton("📋 Copy Summary")
        self.copy_summary_btn.setFixedHeight(28)
        self.copy_summary_btn.setStyleSheet(render_btn.styleSheet())
        self.copy_summary_btn.clicked.connect(self.copy_summary)
        self.copy_summary_btn.setEnabled(False)

        self.clear_btn = QPushButton("🧹 Clear View")
        self.clear_btn.setFixedHeight(28)
        self.clear_btn.setStyleSheet(render_btn.styleSheet())
        self.clear_btn.clicked.connect(self.clear_view)

        bottom_row.addWidget(self.copy_summary_btn)
        bottom_row.addWidget(self.clear_btn)
        container_layout.addLayout(bottom_row)

        container_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # =========================================================================
    # SPHARM Directory & Patient Table Scanning
    # =========================================================================
    def browse_spharm_dir(self):
        start_dir = self.get_output_folder() if self.get_output_folder else os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select SPHARM Results Directory", start_dir)
        if folder:
            self.spharm_dir_input.setText(folder)

    def on_panel_activated(self):
        """Called whenever user switches to Result Panel tab."""
        if not self.spharm_dir_input.text().strip():
            if self.get_output_folder:
                out_dir = self.get_output_folder().strip()
                if out_dir and os.path.isdir(out_dir):
                    spharm_cand = os.path.join(out_dir, "output_SPHARM")
                    if os.path.isdir(spharm_cand):
                        self.spharm_dir_input.setText(spharm_cand)
                    else:
                        self.spharm_dir_input.setText(out_dir)
        self.populate_patients_table()

    def populate_patients_table(self):
        spharm_dir = self.spharm_dir_input.text().strip()
        search_dirs = []
        if spharm_dir and os.path.isdir(spharm_dir):
            search_dirs.append(spharm_dir)
            search_dirs.append(os.path.join(spharm_dir, "left", "spharm_results"))
            search_dirs.append(os.path.join(spharm_dir, "right", "spharm_results"))
            search_dirs.append(os.path.join(spharm_dir, "spharm_results"))

        # Fallback to output directory if provided
        if self.get_output_folder:
            out_d = self.get_output_folder().strip()
            if out_d and os.path.isdir(out_d) and out_d not in search_dirs:
                search_dirs.extend([
                    out_d,
                    os.path.join(out_d, "output_SPHARM"),
                    os.path.join(out_d, "output_left_hippocampus", "spharm_results_left"),
                    os.path.join(out_d, "output_right_hippocampus", "spharm_results_right")
                ])

        # Discover all .coef and .vtk files
        subjects = {}
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for coef_file in glob.glob(os.path.join(d, "**", "*.coef"), recursive=True):
                fname = os.path.basename(coef_file)
                side = "left" if ("lh_" in fname or "left" in fname.lower()) else ("right" if ("rh_" in fname or "right" in fname.lower()) else "unknown")
                subj = self._extract_subject_id(fname)
                if subj not in subjects:
                    subjects[subj] = {"left_coef": None, "right_coef": None, "left_vtk": None, "right_vtk": None}
                if side == "left":
                    subjects[subj]["left_coef"] = coef_file
                elif side == "right":
                    subjects[subj]["right_coef"] = coef_file

            for vtk_file in glob.glob(os.path.join(d, "**", "*SPHARM*.vtk"), recursive=True):
                fname = os.path.basename(vtk_file)
                side = "left" if ("lh_" in fname or "left" in fname.lower()) else ("right" if ("rh_" in fname or "right" in fname.lower()) else "unknown")
                subj = self._extract_subject_id(fname)
                if subj not in subjects:
                    subjects[subj] = {"left_coef": None, "right_coef": None, "left_vtk": None, "right_vtk": None}
                if side == "left":
                    subjects[subj]["left_vtk"] = vtk_file
                elif side == "right":
                    subjects[subj]["right_vtk"] = vtk_file

        # Fallback to demo template files if no SPHARM files detected
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tmpl_left_coef = os.path.join(repo_root, "Templates", "SPHARM", "template_spharm_left.coef")
        tmpl_right_coef = os.path.join(repo_root, "Templates", "SPHARM", "template_spharm_right.coef")
        tmpl_left_vtk = os.path.join(repo_root, "Templates", "SPHARM", "template_spharm_left.vtk")
        tmpl_right_vtk = os.path.join(repo_root, "Templates", "SPHARM", "template_spharm_right.vtk")

        if not subjects and os.path.isfile(tmpl_left_coef):
            subjects["Standard_Template_Subject"] = {
                "left_coef": tmpl_left_coef,
                "right_coef": tmpl_right_coef,
                "left_vtk": tmpl_left_vtk,
                "right_vtk": tmpl_right_vtk
            }

        # Populate table
        self.patient_table.setRowCount(0)
        self.patient_table.setRowCount(len(subjects))

        for row, (subj_name, data) in enumerate(sorted(subjects.items())):
            item_name = QTableWidgetItem(subj_name)
            item_name.setData(Qt.ItemDataRole.UserRole, data)

            lh_ok = "✅ Ready" if data["left_coef"] else "❌ Missing"
            rh_ok = "✅ Ready" if data["right_coef"] else "❌ Missing"
            vtk_ok = "✅" if (data["left_vtk"] or data["right_vtk"]) else "—"

            item_lh = QTableWidgetItem(lh_ok)
            item_rh = QTableWidgetItem(rh_ok)
            item_vtk = QTableWidgetItem(vtk_ok)

            item_lh.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_rh.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_vtk.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.patient_table.setItem(row, 0, item_name)
            self.patient_table.setItem(row, 1, item_lh)
            self.patient_table.setItem(row, 2, item_rh)
            self.patient_table.setItem(row, 3, item_vtk)

        if self.patient_table.rowCount() > 0:
            self.patient_table.selectRow(0)

    def _extract_subject_id(self, filename: str) -> str:
        clean = filename
        for sfx in ["_aligned_SPHARM.vtk", "_SPHARM.vtk", "_aligned.coef", "_SPHARM.coef", ".coef", ".vtk"]:
            if clean.endswith(sfx):
                clean = clean[:-len(sfx)]
                break
        if clean.startswith("lh_") or clean.startswith("rh_"):
            clean = clean[3:]
        if clean.startswith("left_") or clean.startswith("right_"):
            clean = clean[clean.find("_") + 1:]
        if clean.endswith("_lh") or clean.endswith("_rh"):
            clean = clean[:-3]
        return clean or filename

    def on_patient_selected(self):
        rows = self.patient_table.selectedItems()
        if not rows:
            self.current_patient_data = None
            self.predict_btn.setEnabled(False)
            return

        row = rows[0].row()
        item = self.patient_table.item(row, 0)
        subj_name = item.text()
        self.current_patient_data = item.data(Qt.ItemDataRole.UserRole)
        self.current_patient_data["subject_name"] = subj_name

        has_data = bool(self.current_patient_data.get("left_coef") or self.current_patient_data.get("right_coef"))
        self.predict_btn.setEnabled(has_data)
        self.signal_log_message.emit(f"Selected subject: {subj_name}")

    # =========================================================================
    # ResNet Prediction Execution
    # =========================================================================
    def run_prediction(self):
        if not self.current_patient_data:
            return

        subj_name = self.current_patient_data.get("subject_name", "Subject")
        lh_coef = self.current_patient_data.get("left_coef")
        rh_coef = self.current_patient_data.get("right_coef")

        eval_target = self.side_eval_group.checkedId() # 0: both, 1: left, 2: right
        self.signal_log_message.emit(f">>> Running ResNet diagnostic prediction for {subj_name}...")

        try:
            results = {'left': None, 'right': None, 'summary': None}

            # Predict Left
            if eval_target in (0, 1):
                if lh_coef and os.path.isfile(lh_coef):
                    results['left'] = self.predictor.predict(lh_coef, side='left')
                else:
                    self.signal_log_message.emit("[WARNING] Left .coef file not available.")

            # Predict Right
            if eval_target in (0, 2):
                if rh_coef and os.path.isfile(rh_coef):
                    results['right'] = self.predictor.predict(rh_coef, side='right')
                else:
                    self.signal_log_message.emit("[WARNING] Right .coef file not available.")

            # Synthesize combined prediction
            if results['left'] and results['right']:
                avg_prob = (results['left']['probability'] + results['right']['probability']) / 2.0
                is_epilepsy = avg_prob > 0.5
                higher_side = 'Left' if results['left']['probability'] >= results['right']['probability'] else 'Right'
                results['summary'] = {
                    'probability': avg_prob,
                    'is_epilepsy': is_epilepsy,
                    'label': "Temporal Lobe Epilepsy (TLE)" if is_epilepsy else "Healthy Control (HC)",
                    'primary_side': higher_side,
                    'left_prob': results['left']['probability'],
                    'right_prob': results['right']['probability']
                }
            elif results['left']:
                prob = results['left']['probability']
                results['summary'] = {
                    'probability': prob,
                    'is_epilepsy': prob > 0.5,
                    'label': "Temporal Lobe Epilepsy (TLE)" if prob > 0.5 else "Healthy Control (HC)",
                    'primary_side': "Left",
                    'left_prob': prob,
                    'right_prob': None
                }
            elif results['right']:
                prob = results['right']['probability']
                results['summary'] = {
                    'probability': prob,
                    'is_epilepsy': prob > 0.5,
                    'label': "Temporal Lobe Epilepsy (TLE)" if prob > 0.5 else "Healthy Control (HC)",
                    'primary_side': "Right",
                    'left_prob': None,
                    'right_prob': prob
                }
            else:
                QMessageBox.warning(self, "No Input Data", "No valid .coef files were found for this patient.")
                return

            self.last_prediction_results = results
            self.display_prediction_results(results)
            self.update_3d_view()
            self.copy_summary_btn.setEnabled(True)

        except Exception as e:
            self.signal_log_message.emit(f"[ERROR] Prediction failed: {str(e)}")
            QMessageBox.critical(self, "Prediction Error", f"Failed to run ResNet prediction:\n{str(e)}")

    def display_prediction_results(self, results):
        summary = results.get('summary')
        if not summary:
            return

        prob = summary['probability']
        is_tle = summary['is_epilepsy']
        label = summary['label']
        prob_pct = int(round(prob * 100))

        # Update Diagnosis Badge
        if is_tle:
            self.diagnosis_badge.setText(f"🚨 {label}")
            self.diagnosis_badge.setStyleSheet("color: #e74c3c; font-size: 13px; font-weight: bold;")
            self.status_banner.setStyleSheet("""
                QFrame {
                    background-color: #fdf2f2;
                    border: 2px solid #e74c3c;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            if prob > 0.75:
                risk_str = f"HIGH RISK — High morphological atrophy alignment ({prob*100:.1f}%)"
            else:
                risk_str = f"MODERATE / BORDERLINE TLE RISK ({prob*100:.1f}%)"
            self.risk_badge.setText(risk_str)
            self.risk_badge.setStyleSheet("color: #c0392b; font-weight: bold; font-size: 11px;")
            self.prob_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ced6e0;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #ecf0f1;
                    font-size: 10px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                QProgressBar::chunk {
                    background-color: #e74c3c;
                    border-radius: 2px;
                }
            """)
        else:
            self.diagnosis_badge.setText(f"✅ {label}")
            self.diagnosis_badge.setStyleSheet("color: #27ae60; font-size: 13px; font-weight: bold;")
            self.status_banner.setStyleSheet("""
                QFrame {
                    background-color: #f2fbf6;
                    border: 2px solid #27ae60;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            self.risk_badge.setText(f"NORMAL — Standard anatomical shape distribution ({(1.0-prob)*100:.1f}% Confidence)")
            self.risk_badge.setStyleSheet("color: #229954; font-weight: bold; font-size: 11px;")
            self.prob_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ced6e0;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #ecf0f1;
                    font-size: 10px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                QProgressBar::chunk {
                    background-color: #27ae60;
                    border-radius: 2px;
                }
            """)

        self.prob_bar.setValue(prob_pct)

        # Update Lateralization Breakdown
        lh_str = f"{summary['left_prob']*100:.1f}%" if summary.get('left_prob') is not None else "N/A"
        rh_str = f"{summary['right_prob']*100:.1f}%" if summary.get('right_prob') is not None else "N/A"
        focus = summary.get('primary_side', 'N/A')

        if is_tle:
            focus_str = f"<b style='color: #c0392b;'>{focus} Hippocampus</b> (Greater Atrophy Profile)"
        else:
            focus_str = "Bilateral Symmetrical Normative Shape"

        self.lat_detail_lbl.setText(
            f"• Left (LH) Risk: <b>{lh_str}</b> | • Right (RH) Risk: <b>{rh_str}</b><br>"
            f"• Suspected Seizure Focus: {focus_str}"
        )

        # Auto-switch 3D viewing side to the side with higher probability
        if summary.get('left_prob') is not None and summary.get('right_prob') is not None:
            if summary['right_prob'] > summary['left_prob']:
                self.rb_cam_right.setChecked(True)
            else:
                self.rb_cam_left.setChecked(True)

    # =========================================================================
    # 3D Grad-CAM & Heatmap Visualization
    # =========================================================================
    def update_3d_view(self):
        side = "left" if self.rb_cam_left.isChecked() else "right"

        # Determine Colormap and Scalar mode
        if self.rb_signed.isChecked():
            scalar_mode = "SignedDistance"
            lut_type = "signed_distance"
            title = f"{side.upper()} Signed Atrophy (Inward/Outward)"
        elif self.rb_dist.isChecked():
            scalar_mode = "DistanceMapping"
            lut_type = "distance_mapping"
            title = f"{side.upper()} Deformation Magnitude (mm)"
        else:
            scalar_mode = "GradCAM_Importance"
            lut_type = "gradcam"
            title = f"ResNet Grad-CAM Attention ({side.upper()})"

        # Determine milestone stage
        if self.rb_minus2sd.isChecked():
            milestone = "minus2SD"
        elif self.rb_minus3sd.isChecked():
            milestone = "minus3SD"
        else:
            milestone = "Mean"

        mesh_path = self.predictor.get_gradcam_mesh_path(
            side=side,
            component="PLS1",
            milestone=milestone,
            cohort="All_Augment_tain"
        )

        if not mesh_path or not os.path.isfile(mesh_path):
            self.signal_log_message.emit(f"[WARNING] 3D Grad-CAM mesh not found for {side}. Checking template...")
            mesh_path = self.predictor.get_template_mesh_path(side)

        if mesh_path and os.path.isfile(mesh_path):
            self.signal_gradcam_mesh_requested.emit(
                mesh_path, scalar_mode, lut_type, title, side, 1.0
            )

        # Handle Patient Mesh Overlay
        self.on_patient_overlay_toggled()

    def on_patient_overlay_toggled(self):
        is_overlay = self.patient_overlay_cb.isChecked()
        opacity = self.opacity_slider.value() / 100.0
        side = "left" if self.rb_cam_left.isChecked() else "right"

        patient_mesh = None
        if self.current_patient_data:
            patient_mesh = self.current_patient_data.get(f"{side}_vtk")

        if is_overlay and patient_mesh and os.path.isfile(patient_mesh):
            self.signal_patient_overlay_requested.emit(patient_mesh, True, opacity, side)
        else:
            self.signal_patient_overlay_requested.emit("", False, 0.0, side)

    def clear_view(self):
        self.signal_clear_gradcam_requested.emit()
        self.diagnosis_badge.setText("Awaiting Evaluation")
        self.diagnosis_badge.setStyleSheet("color: #7f8c8d;")
        self.risk_badge.setText("Select a subject above and click 'Run ResNet Prediction'")
        self.risk_badge.setStyleSheet("color: #576574;")
        self.status_banner.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px dashed #b2bec3;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.prob_bar.setValue(0)
        self.lat_detail_lbl.setText("• Left: N/A  |  • Right: N/A  |  • Primary Focus: N/A")
        self.copy_summary_btn.setEnabled(False)
        self.signal_log_message.emit("Diagnostic panel reset.")

    def copy_summary(self):
        if not self.last_prediction_results or not self.current_patient_data:
            return

        summary = self.last_prediction_results.get('summary', {})
        subj = self.current_patient_data.get('subject_name', 'Patient')
        diag = summary.get('label', 'N/A')
        prob = summary.get('probability', 0.0) * 100.0
        lh = summary.get('left_prob')
        rh = summary.get('right_prob')
        focus = summary.get('primary_side', 'N/A')

        text = (
            f"=== HIPPOCAMPAL SHAPE ANALYSIS - DIAGNOSTIC REPORT ===\n"
            f"Subject ID: {subj}\n"
            f"Model: ResNet1D Residual Neural Network + PLS-DA\n"
            f"Diagnosis: {diag}\n"
            f"Overall Epilepsy Probability: {prob:.2f}%\n"
            f"Left Hippocampus Risk: {lh*100:.2f}%\n" if lh is not None else ""
            f"Right Hippocampus Risk: {rh*100:.2f}%\n" if rh is not None else ""
            f"Suspected Seizure Focus: {focus} Hemisphere\n"
            f"Methodology: SPHARM-PDM Point Distribution Parameterization\n"
            f"======================================================\n"
        )
        cb = QApplication.clipboard()
        if cb:
            cb.setText(text)
            self.signal_log_message.emit("Diagnostic summary copied to clipboard.")
            QMessageBox.information(self, "Summary Copied", "Diagnostic summary has been copied to your clipboard.")
