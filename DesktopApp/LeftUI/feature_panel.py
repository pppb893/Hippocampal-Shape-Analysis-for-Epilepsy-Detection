import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QCheckBox, QGroupBox, QFormLayout, QComboBox, QSpinBox, 
                             QLineEdit, QFileDialog)
from PyQt6.QtCore import pyqtSignal, QProcess, QLocale

class FeaturePanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, get_folder_func, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.setup_ui()

    def setup_ui(self):
        feature_layout = QVBoxLayout(self)
        help_label = QLabel("This module extracts tabular shape features (Volume, Surface Area, SPHARM mesh 3D coordinates, and harmonic coefficients) into CSV datasets for Machine Learning.")
        help_label.setWordWrap(True)
        feature_layout.addWidget(help_label)
        
        feature_group = QGroupBox("Feature Extraction Parameters")
        feature_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 20px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 13px;
            }
            QSpinBox, QComboBox, QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                min-height: 22px;
                color: #34495e;
            }
            QSpinBox:disabled {
                background-color: #eef2f5;
                color: #7f8c8d;
                border: 1px solid #dcdde1;
            }
            QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                border: 1px solid #16a085;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
        """)
        
        feature_form = QFormLayout(feature_group)
        feature_form.setContentsMargins(15, 25, 15, 15)
        feature_form.setVerticalSpacing(12)
        
        self.feature_mode_combo = QComboBox()
        self.feature_mode_combo.addItems(["Standard Mode (Default)", "Custom Mode"])
        self.feature_mode_combo.setCurrentIndex(0)
        self.feature_mode_combo.setToolTip("Preset modes. Standard Mode extracts 1002 mesh landmark points.")
        self.feature_mode_combo.currentIndexChanged.connect(self.on_feature_mode_changed)

        self.feature_type_combo = QComboBox()
        self.feature_type_combo.addItems([
            "All Features (Coordinates & Coefficients)",
            "Mesh 3D Coordinates & Volume (.vtk)",
            "SPHARM Harmonic Coefficients (.coef)"
        ])
        self.feature_type_combo.setToolTip("Type of shape features to extract for Machine Learning.")

        self.feature_points_spin = QSpinBox()
        self.feature_points_spin.setLocale(QLocale.c())
        self.feature_points_spin.setRange(0, 10000)
        self.feature_points_spin.setValue(1002)
        self.feature_points_spin.setSingleStep(100)
        self.feature_points_spin.setEnabled(False)
        self.feature_points_spin.setToolTip("Number of mesh landmark points to extract per subject. (Default: 1002)")

        csv_layout = QHBoxLayout()
        self.feature_csv_input = QLineEdit()
        self.feature_csv_input.setPlaceholderText("Auto (Filename-based grouping)")
        self.feature_csv_input.setToolTip("Optional metadata CSV file for subject group labels.")
        csv_btn = QPushButton("📁 Browse")
        csv_btn.clicked.connect(self.select_feature_csv)
        csv_layout.addWidget(self.feature_csv_input)
        csv_layout.addWidget(csv_btn)

        self.feature_export_mesh_cb = QCheckBox("Export Mesh Points & Geometry (.csv)")
        self.feature_export_mesh_cb.setChecked(True)
        self.feature_export_mesh_cb.setToolTip("Export 3D mesh surface coordinates, volume, and area to features_dataset.csv.")

        self.feature_export_coef_cb = QCheckBox("Export SPHARM Coefficients (.csv)")
        self.feature_export_coef_cb.setChecked(True)
        self.feature_export_coef_cb.setToolTip("Export spherical harmonic expansion coefficients to spharm_coef_dataset.csv.")

        feature_form.addRow("Preset Mode:", self.feature_mode_combo)
        feature_form.addRow("Feature Category:", self.feature_type_combo)
        feature_form.addRow("Mesh Landmark Points:", self.feature_points_spin)
        feature_form.addRow("Group Metadata CSV:", csv_layout)
        feature_form.addRow("Export Datasets:", self.feature_export_mesh_cb)
        feature_form.addRow("", self.feature_export_coef_cb)
        
        feature_layout.addWidget(feature_group)
        
        self.run_feature_btn = QPushButton("📁 Export Feature Datasets")
        self.run_feature_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a085; 
                color: white; 
                padding: 10px 15px; 
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1abc9c;
            }
            QPushButton:pressed {
                background-color: #117864;
            }
        """)
        self.run_feature_btn.clicked.connect(self.run_feature_process)
        feature_layout.addWidget(self.run_feature_btn)
        
        feature_layout.addStretch()

    def on_feature_mode_changed(self, index):
        if index == 0:  # Standard Mode
            self.feature_points_spin.setValue(1002)
            self.feature_points_spin.setEnabled(False)
        elif index == 1:  # Custom Mode
            if self.feature_points_spin.value() == 0:
                self.feature_points_spin.setValue(1002)
            self.feature_points_spin.setEnabled(True)

    def select_feature_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Group Metadata CSV", "", "CSV Files (*.csv)")
        if filename:
            self.feature_csv_input.setText(filename)

    def run_feature_process(self):
        selected_dir = self.get_folder()
        if not selected_dir or not os.path.isdir(selected_dir):
            self.signal_log_message.emit("[ERROR] Please select a valid input directory first.")
            return

        folder_name = os.path.basename(os.path.normpath(selected_dir))
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        output_candidate = os.path.join(script_dir, f"output_{folder_name}", "spharm_results")
        if os.path.exists(output_candidate):
            spharm_dir = output_candidate
        else:
            spharm_dir = selected_dir

        export_mesh = self.feature_export_mesh_cb.isChecked()
        export_coef = self.feature_export_coef_cb.isChecked()

        if not export_mesh and not export_coef:
            self.signal_log_message.emit("[WARNING] Please check at least one dataset type to export.")
            return

        self.feature_queue = []
        ext_script1 = os.path.join(script_dir, 'Data_Processing', 'extract_ml_features.py')
        ext_script2 = os.path.join(script_dir, 'Data_Processing', 'extract_ml_features_coef.py')

        if export_mesh and os.path.exists(ext_script1):
            self.feature_queue.append(ext_script1)
        if export_coef and os.path.exists(ext_script2):
            self.feature_queue.append(ext_script2)

        self.signal_log_message.emit(">>> Starting Feature Dataset Export...")
        self.signal_log_message.emit(f"SPHARM Target Directory: {spharm_dir}")
        self.signal_log_message.emit(f"Export Tasks: {len(self.feature_queue)} script(s)")

        self.run_feature_btn.setEnabled(False)
        self.run_next_feature_process(spharm_dir)

    def run_next_feature_process(self, spharm_dir):
        if not self.feature_queue:
            self.run_feature_btn.setEnabled(True)
            self.signal_log_message.emit(">>> ALL FEATURE DATASETS EXPORTED SUCCESSFULLY!")
            return

        next_script = self.feature_queue.pop(0)
        script_name = os.path.basename(next_script)
        self.signal_log_message.emit(f">>> Running: {script_name}...")

        self.feature_process = QProcess(self)
        self.feature_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.feature_process.readyReadStandardOutput.connect(self.handle_feature_stdout)
        self.feature_process.finished.connect(lambda code, status: self.on_single_feature_process_finished(code, status, spharm_dir))

        args = [next_script, "--spharm_dir", spharm_dir]
        self.feature_process.start("python", args)

    def on_single_feature_process_finished(self, exit_code, exit_status, spharm_dir):
        self.signal_log_message.emit(f">>> Step finished with code {exit_code}")
        self.run_next_feature_process(spharm_dir)

    def handle_feature_stdout(self):
        data = self.feature_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.signal_log_message.emit(line)
