import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
                             QHBoxLayout, QLineEdit, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QProcess, QLocale

class IcpPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, get_folder_func, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.setup_ui()

    def setup_ui(self):
        icp_layout = QVBoxLayout(self)
        
        help_label = QLabel("This module aligns meshes rigidly. Please adjust parameters if needed.")
        help_label.setWordWrap(True)
        icp_layout.addWidget(help_label)
        
        icp_group = QGroupBox("ICP Parameters")
        
        icp_group.setStyleSheet("""
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
            QSpinBox, QDoubleSpinBox, QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                min-height: 22px;
                color: #34495e;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
                background-color: #eef2f5;
                color: #7f8c8d;
                border: 1px solid #dcdde1;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #3498db;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
        """)
        
        icp_form = QFormLayout(icp_group)
        icp_form.setContentsMargins(15, 25, 15, 15)
        icp_form.setVerticalSpacing(12)
        
        self.icp_mode_combo = QComboBox()
        self.icp_mode_combo.addItems(["Production Mode (Default)", "Custom Mode"])
        self.icp_mode_combo.setCurrentIndex(0)
        self.icp_mode_combo.setToolTip("Preset modes. Production Mode uses standard project settings.")
        self.icp_mode_combo.currentIndexChanged.connect(self.on_icp_mode_changed)

        self.icp_spacing_spin = QDoubleSpinBox()
        self.icp_spacing_spin.setLocale(QLocale.c())
        self.icp_spacing_spin.setRange(0.0, 10.0)
        self.icp_spacing_spin.setDecimals(3)
        self.icp_spacing_spin.setValue(0.02)
        self.icp_spacing_spin.setSingleStep(0.01)
        self.icp_spacing_spin.setEnabled(False)
        self.icp_spacing_spin.setToolTip("Image resolution. A smaller number makes the 3D model look smoother, but takes longer to process. (Recommended: 0.02)")
        
        self.icp_voxels_spin = QSpinBox()
        self.icp_voxels_spin.setLocale(QLocale.c())
        self.icp_voxels_spin.setRange(0, 1024)
        self.icp_voxels_spin.setValue(128)
        self.icp_voxels_spin.setSingleStep(16)
        self.icp_voxels_spin.setEnabled(False)
        self.icp_voxels_spin.setToolTip("The size of the 3D box that contains the model. A larger number makes the box bigger. (Recommended: 128)")
        
        self.icp_max_iter_spin = QSpinBox()
        self.icp_max_iter_spin.setLocale(QLocale.c())
        self.icp_max_iter_spin.setRange(0, 1000)
        self.icp_max_iter_spin.setValue(20)
        self.icp_max_iter_spin.setEnabled(False)
        self.icp_max_iter_spin.setToolTip("The maximum number of times the program will try to find the 'average shape' of all patients. More rounds = more accurate but slower. (Recommended: 20)")
        
        self.icp_tol_spin = QDoubleSpinBox()
        self.icp_tol_spin.setLocale(QLocale.c())
        self.icp_tol_spin.setRange(0.0, 1.0)
        self.icp_tol_spin.setDecimals(5)
        self.icp_tol_spin.setValue(0.00005)
        self.icp_tol_spin.setSingleStep(0.00001)
        self.icp_tol_spin.setEnabled(False)
        self.icp_tol_spin.setToolTip("The resting point. If the average shape barely changes after a round, the program will stop early to save time. (Recommended: 0.00005)")

        self.icp_pw_iter_spin = QSpinBox()
        self.icp_pw_iter_spin.setLocale(QLocale.c())
        self.icp_pw_iter_spin.setRange(0, 10000)
        self.icp_pw_iter_spin.setValue(100)
        self.icp_pw_iter_spin.setEnabled(False)
        self.icp_pw_iter_spin.setToolTip("The maximum number of attempts the program makes to match and overlap two shapes perfectly. (Recommended: 100)")

        self.icp_pw_tol_spin = QDoubleSpinBox()
        self.icp_pw_tol_spin.setLocale(QLocale.c())
        self.icp_pw_tol_spin.setRange(0.0, 1.0)
        self.icp_pw_tol_spin.setDecimals(6)
        self.icp_pw_tol_spin.setValue(0.0001)
        self.icp_pw_tol_spin.setSingleStep(0.00001)
        self.icp_pw_tol_spin.setEnabled(False)
        self.icp_pw_tol_spin.setToolTip("The acceptable matching error. If two shapes overlap closer than this number, the program considers it a success. (Recommended: 0.0001)")
        
        self.icp_pw_landmarks_spin = QSpinBox()
        self.icp_pw_landmarks_spin.setLocale(QLocale.c())
        self.icp_pw_landmarks_spin.setRange(0, 10000)
        self.icp_pw_landmarks_spin.setValue(200)
        self.icp_pw_landmarks_spin.setEnabled(False)
        self.icp_pw_landmarks_spin.setToolTip("The number of points sampled from the shape to calculate the match. More points = better accuracy, but much slower. (Recommended: 200)")

        self.icp_interp_combo = QComboBox()
        self.icp_interp_combo.addItems(["NearestNeighbor", "Linear", "BSpline", "Lanczos", "WindowedSinc"])
        self.icp_interp_combo.setCurrentText("NearestNeighbor")
        self.icp_interp_combo.setEnabled(False)
        self.icp_interp_combo.setToolTip("The drawing technique used to recreate the 3D image. 'NearestNeighbor' is highly recommended to keep the shape's original sharp edges intact.")
        
        out_layout = QHBoxLayout()
        self.icp_out_input = QLineEdit()
        self.icp_out_input.setPlaceholderText("Auto (output_<input_folder_name>)")
        self.icp_out_input.setToolTip("Optional: Select a custom output directory. If empty, a default output folder will be created.")
        out_btn = QPushButton("📁 Browse")
        out_btn.clicked.connect(self.select_output_directory)
        out_layout.addWidget(self.icp_out_input)
        out_layout.addWidget(out_btn)
        
        icp_form.addRow("Preset Mode:", self.icp_mode_combo)
        icp_form.addRow("Output Spacing:", self.icp_spacing_spin)
        icp_form.addRow("Output Voxels:", self.icp_voxels_spin)
        icp_form.addRow("Groupwise Max Iterations:", self.icp_max_iter_spin)
        icp_form.addRow("Groupwise Tolerance:", self.icp_tol_spin)
        icp_form.addRow("Pairwise Max Iterations:", self.icp_pw_iter_spin)
        icp_form.addRow("Pairwise Tolerance:", self.icp_pw_tol_spin)
        icp_form.addRow("Pairwise Landmarks:", self.icp_pw_landmarks_spin)
        icp_form.addRow("Interpolation Mode:", self.icp_interp_combo)
        icp_form.addRow("Output Directory:", out_layout)
        icp_layout.addWidget(icp_group)
        
        self.run_icp_btn = QPushButton("Run ICP Registration")
        self.run_icp_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; 
                color: white; 
                padding: 10px 15px; 
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        self.run_icp_btn.clicked.connect(self.run_icp_process)
        icp_layout.addWidget(self.run_icp_btn)
        
        icp_layout.addStretch()

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.icp_out_input.setText(directory)

    def on_icp_mode_changed(self, index):
        if index == 0:  # Production Mode (Locked)
            self.icp_spacing_spin.setValue(0.02)
            self.icp_voxels_spin.setValue(128)
            self.icp_max_iter_spin.setValue(20)
            self.icp_tol_spin.setValue(0.00005)
            self.icp_pw_iter_spin.setValue(100)
            self.icp_pw_tol_spin.setValue(0.0001)
            self.icp_pw_landmarks_spin.setValue(200)
            self.icp_interp_combo.setCurrentText("NearestNeighbor")
            
            self.icp_spacing_spin.setEnabled(False)
            self.icp_voxels_spin.setEnabled(False)
            self.icp_max_iter_spin.setEnabled(False)
            self.icp_tol_spin.setEnabled(False)
            self.icp_pw_iter_spin.setEnabled(False)
            self.icp_pw_tol_spin.setEnabled(False)
            self.icp_pw_landmarks_spin.setEnabled(False)
            self.icp_interp_combo.setEnabled(False)
        elif index == 1:  # Custom Mode (Reset all numeric values to 0 for user entry)
            self.icp_spacing_spin.setValue(0.0)
            self.icp_voxels_spin.setValue(0)
            self.icp_max_iter_spin.setValue(0)
            self.icp_tol_spin.setValue(0.0)
            self.icp_pw_iter_spin.setValue(0)
            self.icp_pw_tol_spin.setValue(0.0)
            self.icp_pw_landmarks_spin.setValue(0)
            
            self.icp_spacing_spin.setEnabled(True)
            self.icp_voxels_spin.setEnabled(True)
            self.icp_max_iter_spin.setEnabled(True)
            self.icp_tol_spin.setEnabled(True)
            self.icp_pw_iter_spin.setEnabled(True)
            self.icp_pw_tol_spin.setEnabled(True)
            self.icp_pw_landmarks_spin.setEnabled(True)
            self.icp_interp_combo.setEnabled(True)

    def run_icp_process(self):
        input_dir = self.get_folder()
        if not input_dir or not os.path.isdir(input_dir):
            self.signal_log_message.emit("[ERROR] Please select a valid input directory first.")
            return
            
        spacing = self.icp_spacing_spin.value()
        voxels = self.icp_voxels_spin.value()
        max_iter = self.icp_max_iter_spin.value()
        tolerance = self.icp_tol_spin.value()
        pw_iter = self.icp_pw_iter_spin.value()
        pw_tol = self.icp_pw_tol_spin.value()
        pw_landmarks = self.icp_pw_landmarks_spin.value()
        interp = self.icp_interp_combo.currentText()
        
        self.signal_log_message.emit(">>> Starting ICP Registration...")
        self.signal_log_message.emit(f"Parameters: Spacing={spacing}, Voxels={voxels}, GW_MaxIter={max_iter}, GW_Tol={tolerance}, PW_MaxIter={pw_iter}, PW_Tol={pw_tol}, PW_Landmarks={pw_landmarks}, Interp={interp}")
        
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        icp_script_path = os.path.join(script_dir, 'ICP', 'ICP.py')
        
        if not os.path.exists(icp_script_path):
            self.signal_log_message.emit(f"[ERROR] ICP script not found at {icp_script_path}")
            return
            
        self.icp_process = QProcess(self)
        self.icp_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.icp_process.readyReadStandardOutput.connect(self.handle_icp_stdout)
        self.icp_process.finished.connect(self.icp_process_finished)
        
        args = [
            icp_script_path,
            "--input_dir", input_dir,
            "--output_spacing", str(spacing),
            "--output_voxels", str(voxels),
            "--max_iterations", str(max_iter),
            "--tolerance", str(tolerance),
            "--pairwise_iterations", str(pw_iter),
            "--pairwise_tolerance", str(pw_tol),
            "--pairwise_landmarks", str(pw_landmarks),
            "--interpolation", interp
        ]
        
        output_dir = self.icp_out_input.text().strip()
        if output_dir:
            args.extend(["--output_dir", output_dir])
        
        self.run_icp_btn.setEnabled(False)
        self.icp_process.start("python", args)

    def handle_icp_stdout(self):
        data = self.icp_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.signal_log_message.emit(line)

    def icp_process_finished(self, exit_code, exit_status):
        self.run_icp_btn.setEnabled(True)
        self.signal_log_message.emit(f">>> ICP Registration finished with exit code {exit_code}")
