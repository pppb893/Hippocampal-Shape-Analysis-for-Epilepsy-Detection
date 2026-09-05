import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QCheckBox, QGroupBox, QFormLayout, QComboBox, QSpinBox, 
                             QLineEdit, QFileDialog)
from PyQt6.QtCore import pyqtSignal, QProcess, QLocale

class SpharmPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, get_folder_func, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.setup_ui()

    def setup_ui(self):
        spharm_layout = QVBoxLayout(self)
        help_label = QLabel("This module processes SPHARM spherical parameterization and shape description. Please adjust parameters if needed.")
        help_label.setWordWrap(True)
        spharm_layout.addWidget(help_label)
        
        spharm_group = QGroupBox("SPHARM Parameters")
        spharm_group.setStyleSheet("""
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
                border: 1px solid #8e44ad;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
        """)
        
        spharm_form = QFormLayout(spharm_group)
        spharm_form.setContentsMargins(15, 25, 15, 15)
        spharm_form.setVerticalSpacing(12)
        
        self.spharm_mode_combo = QComboBox()
        self.spharm_mode_combo.addItems(["Production Mode (Default)", "Fast Test Mode", "Custom Mode"])
        self.spharm_mode_combo.setCurrentIndex(0)
        self.spharm_mode_combo.setToolTip("Preset modes. Production Mode uses 1000 iterations and level 10 subdivision (~1002 surface points).")
        self.spharm_mode_combo.currentIndexChanged.connect(self.on_spharm_mode_changed)

        self.spharm_iter_spin = QSpinBox()
        self.spharm_iter_spin.setLocale(QLocale.c())
        self.spharm_iter_spin.setRange(0, 5000)
        self.spharm_iter_spin.setValue(1000)
        self.spharm_iter_spin.setSingleStep(100)
        self.spharm_iter_spin.setEnabled(False)
        self.spharm_iter_spin.setToolTip("GenParaMesh spherical parameterization iterations. (Default: 1000)")

        self.spharm_subdiv_spin = QSpinBox()
        self.spharm_subdiv_spin.setLocale(QLocale.c())
        self.spharm_subdiv_spin.setRange(0, 30)
        self.spharm_subdiv_spin.setValue(10)
        self.spharm_subdiv_spin.setSingleStep(1)
        self.spharm_subdiv_spin.setEnabled(False)
        self.spharm_subdiv_spin.setToolTip("Subdivision level for ParaToSPHARMMesh. Level 10 creates ~1002 points. (Default: 10)")

        self.spharm_degree_spin = QSpinBox()
        self.spharm_degree_spin.setLocale(QLocale.c())
        self.spharm_degree_spin.setRange(0, 30)
        self.spharm_degree_spin.setValue(12)
        self.spharm_degree_spin.setSingleStep(1)
        self.spharm_degree_spin.setEnabled(False)
        self.spharm_degree_spin.setToolTip("Degree of Spherical Harmonics expansion coefficients. (Default: 12)")

        self.spharm_regen_cb = QCheckBox("Regenerate SPHARM Only")
        self.spharm_regen_cb.setChecked(False)
        self.spharm_regen_cb.setToolTip("Skip SegPostProcess & GenParaMesh. Reuse existing _para.vtk and _surf.vtk files for ~5x faster processing.")

        tmpl_layout = QHBoxLayout()
        self.spharm_tmpl_input = QLineEdit()
        self.spharm_tmpl_input.setPlaceholderText("Auto (First subject ellalign)")
        self.spharm_tmpl_input.setToolTip("Path to reference _SPHARM.vtk used as regTemplate / flipTemplate for batch alignment.")
        tmpl_btn = QPushButton("📁 Browse")
        tmpl_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 12px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dee2e6, stop:1 #ced4da);
                border: 1px solid #95a5a6;
            }
        """)
        tmpl_btn.clicked.connect(self.select_spharm_template)
        tmpl_layout.addWidget(self.spharm_tmpl_input)
        tmpl_layout.addWidget(tmpl_btn)

        spharm_form.addRow("Preset Mode:", self.spharm_mode_combo)
        spharm_form.addRow("GenParaMesh Iterations:", self.spharm_iter_spin)
        spharm_form.addRow("Subdivision Level:", self.spharm_subdiv_spin)
        spharm_form.addRow("SPHARM Degree:", self.spharm_degree_spin)
        spharm_form.addRow("Reference Template:", tmpl_layout)
        spharm_form.addRow("", self.spharm_regen_cb)
        
        spharm_layout.addWidget(spharm_group)
        
        self.run_spharm_btn = QPushButton("Run SPHARM Processing")
        self.run_spharm_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 13px;
                padding: 10px 15px;
                border: 1px solid #ced6e0;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dee2e6, stop:1 #ced4da);
                border: 1px solid #95a5a6;
            }
            QPushButton:disabled {
                background: #f1f2f6;
                color: #a4b0be;
                border: 1px solid #dfe4ea;
            }
        """)
        self.run_spharm_btn.clicked.connect(self.run_spharm_process)
        spharm_layout.addWidget(self.run_spharm_btn)
        
        spharm_layout.addStretch()

    def on_spharm_mode_changed(self, index):
        if index == 0:  # Production Mode
            self.spharm_iter_spin.setValue(1000)
            self.spharm_subdiv_spin.setValue(10)
            self.spharm_degree_spin.setValue(12)
            self.spharm_iter_spin.setEnabled(False)
            self.spharm_subdiv_spin.setEnabled(False)
            self.spharm_degree_spin.setEnabled(False)
        elif index == 1:  # Fast Mode
            self.spharm_iter_spin.setValue(200)
            self.spharm_subdiv_spin.setValue(5)
            self.spharm_degree_spin.setValue(6)
            self.spharm_iter_spin.setEnabled(False)
            self.spharm_subdiv_spin.setEnabled(False)
            self.spharm_degree_spin.setEnabled(False)
        elif index == 2:  # Custom Mode
            if self.spharm_iter_spin.value() == 0:
                self.spharm_iter_spin.setValue(1000)
                self.spharm_subdiv_spin.setValue(10)
                self.spharm_degree_spin.setValue(12)
            self.spharm_iter_spin.setEnabled(True)
            self.spharm_subdiv_spin.setEnabled(True)
            self.spharm_degree_spin.setEnabled(True)

    def select_spharm_template(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Reference Template VTK", "", "VTK Files (*_SPHARM*.vtk *.vtk)")
        if filename:
            self.spharm_tmpl_input.setText(filename)

    def run_spharm_process(self):
        selected_dir = self.get_folder()
        if not selected_dir or not os.path.isdir(selected_dir):
            self.signal_log_message.emit("[ERROR] Please select a valid input directory first.")
            return

        # Smart Auto-Detect: Find the aligned_nifti folder generated by ICP step
        folder_name = os.path.basename(os.path.normpath(selected_dir))
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        candidate1 = os.path.join(script_dir, f"output_{folder_name}", "aligned_nifti")
        candidate2 = os.path.join(selected_dir, "aligned_nifti")
        
        if os.path.basename(selected_dir.rstrip("\\/")) == "aligned_nifti":
            input_dir = selected_dir
        elif os.path.exists(candidate1) and len(os.listdir(candidate1)) > 0:
            input_dir = candidate1
        elif os.path.exists(candidate2) and len(os.listdir(candidate2)) > 0:
            input_dir = candidate2
        else:
            input_dir = selected_dir

        num_iter = self.spharm_iter_spin.value()
        subdiv = self.spharm_subdiv_spin.value()
        degree = self.spharm_degree_spin.value()
        regen_only = self.spharm_regen_cb.isChecked()
        template_path = self.spharm_tmpl_input.text().strip()

        self.signal_log_message.emit(">>> Starting SPHARM Processing...")
        self.signal_log_message.emit(f"Input Directory: {input_dir}")
        self.signal_log_message.emit(f"Parameters: Iterations={num_iter}, SubdivLevel={subdiv}, SPHARMDegree={degree}, RegenOnly={regen_only}, Template={template_path or 'Auto'}")

        spharm_script_path = os.path.join(script_dir, 'SPHARM', 'run_spharm_batch.py')
        if not os.path.exists(spharm_script_path):
            self.signal_log_message.emit(f"[ERROR] SPHARM script not found at {spharm_script_path}")
            return

        slicer_exe = r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
        
        self.spharm_process = QProcess(self)
        self.spharm_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.spharm_process.readyReadStandardOutput.connect(self.handle_spharm_stdout)
        self.spharm_process.finished.connect(self.spharm_process_finished)

        args = [
            "--no-main-window", "--no-splash",
            "--python-script", spharm_script_path,
            "--input_dir", input_dir,
            "--num_iterations", str(num_iter),
            "--subdiv_level", str(subdiv),
            "--spharm_degree", str(degree)
        ]

        if regen_only:
            args.append("--regenerate_spharm_only")

        if template_path and os.path.exists(template_path):
            args.extend(["--reference_template", template_path])

        self.run_spharm_btn.setEnabled(False)
        
        if os.path.exists(slicer_exe):
            self.spharm_process.start(slicer_exe, args)
        else:
            self.signal_log_message.emit(f"[WARNING] SlicerSALT.exe not found at {slicer_exe}, trying system python...")
            args_py = [spharm_script_path, "--input_dir", input_dir, "--num_iterations", str(num_iter), "--subdiv_level", str(subdiv), "--spharm_degree", str(degree)]
            if regen_only: args_py.append("--regenerate_spharm_only")
            if template_path: args_py.extend(["--reference_template", template_path])
            self.spharm_process.start("python", args_py)

    def handle_spharm_stdout(self):
        data = self.spharm_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.signal_log_message.emit(line)

    def spharm_process_finished(self, exit_code, exit_status):
        self.run_spharm_btn.setEnabled(True)
        self.signal_log_message.emit(f">>> SPHARM Processing finished with exit code {exit_code}")
