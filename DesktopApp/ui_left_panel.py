import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QFileDialog, QStackedWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QDoubleSpinBox, QSpinBox, 
                             QGroupBox, QFormLayout, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal, QProcess, QLocale

class LeftPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        left_layout = QVBoxLayout(self)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)

        # --- Logo / Header ---
        logo_label = QLabel("Shape Analysis Toolbox\nHippocampal Pipeline")
        font = logo_label.font()
        font.setPointSize(14)
        font.setBold(True)
        logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        left_layout.addWidget(logo_label)

        # --- QStackedWidget (Module Switching) ---
        self.stacked_widget = QStackedWidget()
        left_layout.addWidget(self.stacked_widget)

        self.setup_import_panel()
        self.setup_icp_panel()
        self.setup_spharm_panel()
        self.setup_plsda_panel()
        self.setup_feature_panel()

        # Console output moved to main window

    def switch_module(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def setup_import_panel(self):
        import_widget = QWidget()
        import_layout = QVBoxLayout(import_widget)
        
        # 1. Import Data Properties
        import_group = QGroupBox("Import Data Properties")
        import_group.setStyleSheet("QGroupBox { margin-top: 15px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        ig_layout = QVBoxLayout(import_group)
        ig_layout.setContentsMargins(10, 20, 10, 10)
        ig_layout.setSpacing(10)
        btn_layout = QHBoxLayout()
        btn_import_dir = QPushButton("Import from directory")
        btn_import_csv = QPushButton("Import from CSV")
        btn_layout.addWidget(btn_import_dir)
        btn_layout.addWidget(btn_import_csv)
        ig_layout.addLayout(btn_layout)

        dir_select_btn = QPushButton("📁 Choose Data Directory")
        dir_select_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px;")
        dir_select_btn.clicked.connect(self.select_directory)
        ig_layout.addWidget(dir_select_btn)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        folder_layout.addWidget(self.folder_input)
        ig_layout.addLayout(folder_layout)
        
        import_action_btn = QPushButton("Import")
        import_action_btn.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
        import_action_btn.clicked.connect(self.on_import_clicked)
        ig_layout.addWidget(import_action_btn)
        import_layout.addWidget(import_group)

        # 2. Imported Subjects
        subj_group = QGroupBox("Imported Subjects")
        subj_group.setStyleSheet("QGroupBox { margin-top: 15px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        subj_layout = QVBoxLayout(subj_group)
        subj_layout.setContentsMargins(10, 20, 10, 10)
        subj_layout.setSpacing(10)
        self.subjects_table = QTableWidget(0, 2)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name", "Consistency"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.subjects_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.subjects_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.subjects_table.itemSelectionChanged.connect(self.on_subject_selection_changed)
        subj_layout.addWidget(self.subjects_table)
        
        display_layout = QHBoxLayout()
        self.display_selected_btn = QPushButton("Display Selected")
        self.display_selected_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.display_selected_btn.clicked.connect(self.display_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.display_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subj_layout.addLayout(display_layout)
        
        import_layout.addWidget(subj_group)
        self.stacked_widget.addWidget(import_widget)

    def setup_icp_panel(self):
        icp_widget = QWidget()
        icp_layout = QVBoxLayout(icp_widget)
        
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
        
        icp_form.addRow("Preset Mode:", self.icp_mode_combo)
        icp_form.addRow("Output Spacing:", self.icp_spacing_spin)
        icp_form.addRow("Output Voxels:", self.icp_voxels_spin)
        icp_form.addRow("Groupwise Max Iterations:", self.icp_max_iter_spin)
        icp_form.addRow("Groupwise Tolerance:", self.icp_tol_spin)
        icp_form.addRow("Pairwise Max Iterations:", self.icp_pw_iter_spin)
        icp_form.addRow("Pairwise Tolerance:", self.icp_pw_tol_spin)
        icp_form.addRow("Pairwise Landmarks:", self.icp_pw_landmarks_spin)
        icp_form.addRow("Interpolation Mode:", self.icp_interp_combo)
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
        self.stacked_widget.addWidget(icp_widget)

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

    def setup_spharm_panel(self):
        spharm_widget = QWidget()
        spharm_layout = QVBoxLayout(spharm_widget)
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
                background-color: #8e44ad; 
                color: white; 
                padding: 10px 15px; 
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #9b59b6;
            }
            QPushButton:pressed {
                background-color: #71368a;
            }
        """)
        self.run_spharm_btn.clicked.connect(self.run_spharm_process)
        spharm_layout.addWidget(self.run_spharm_btn)
        
        spharm_layout.addStretch()
        self.stacked_widget.addWidget(spharm_widget)

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
        selected_dir = self.folder_input.text()
        if not selected_dir or not os.path.isdir(selected_dir):
            self.log("[ERROR] Please select a valid input directory first.")
            return

        # Smart Auto-Detect: Find the aligned_nifti folder generated by ICP step
        folder_name = os.path.basename(os.path.normpath(selected_dir))
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
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

        self.log(">>> Starting SPHARM Processing...")
        self.log(f"Input Directory: {input_dir}")
        self.log(f"Parameters: Iterations={num_iter}, SubdivLevel={subdiv}, SPHARMDegree={degree}, RegenOnly={regen_only}, Template={template_path or 'Auto'}")

        spharm_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SPHARM', 'run_spharm_batch.py'))
        if not os.path.exists(spharm_script_path):
            self.log(f"[ERROR] SPHARM script not found at {spharm_script_path}")
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
            self.log(f"[WARNING] SlicerSALT.exe not found at {slicer_exe}, trying system python...")
            args_py = [spharm_script_path, "--input_dir", input_dir, "--num_iterations", str(num_iter), "--subdiv_level", str(subdiv), "--spharm_degree", str(degree)]
            if regen_only: args_py.append("--regenerate_spharm_only")
            if template_path: args_py.extend(["--reference_template", template_path])
            self.spharm_process.start("python", args_py)

    def handle_spharm_stdout(self):
        data = self.spharm_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.log(line)

    def spharm_process_finished(self, exit_code, exit_status):
        self.run_spharm_btn.setEnabled(True)
        self.log(f">>> SPHARM Processing finished with exit code {exit_code}")
        
    def setup_plsda_panel(self):
        plsda_widget = QWidget()
        plsda_layout = QVBoxLayout(plsda_widget)
        help_label = QLabel("This module runs Partial Least Squares Discriminant Analysis (PLS-DA) on SPHARM shape coefficients to differentiate Healthy Controls vs Epilepsy groups.")
        help_label.setWordWrap(True)
        plsda_layout.addWidget(help_label)
        
        plsda_group = QGroupBox("PLS-DA Parameters")
        plsda_group.setStyleSheet("""
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
                border: 1px solid #e67e22;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
        """)
        
        plsda_form = QFormLayout(plsda_group)
        plsda_form.setContentsMargins(15, 25, 15, 15)
        plsda_form.setVerticalSpacing(12)
        
        self.plsda_mode_combo = QComboBox()
        self.plsda_mode_combo.addItems(["Standard Mode (Default)", "Custom Mode"])
        self.plsda_mode_combo.setCurrentIndex(0)
        self.plsda_mode_combo.setToolTip("Preset modes. Standard Mode extracts 10 PLS-DA components.")
        self.plsda_mode_combo.currentIndexChanged.connect(self.on_plsda_mode_changed)

        self.plsda_comp_spin = QSpinBox()
        self.plsda_comp_spin.setLocale(QLocale.c())
        self.plsda_comp_spin.setRange(0, 50)
        self.plsda_comp_spin.setValue(10)
        self.plsda_comp_spin.setSingleStep(1)
        self.plsda_comp_spin.setEnabled(False)
        self.plsda_comp_spin.setToolTip("Number of PLS-DA components to extract. (Default: 10)")

        csv_layout = QHBoxLayout()
        self.plsda_csv_input = QLineEdit()
        self.plsda_csv_input.setPlaceholderText("Auto (Filenames: HEALTHY vs TLE)")
        self.plsda_csv_input.setToolTip("Optional CSV file containing Subject group labels.")
        csv_btn = QPushButton("📁 Browse")
        csv_btn.clicked.connect(self.select_plsda_csv)
        csv_layout.addWidget(self.plsda_csv_input)
        csv_layout.addWidget(csv_btn)

        self.plsda_export_scores_cb = QCheckBox("Export Score Table (.csv)")
        self.plsda_export_scores_cb.setChecked(True)
        self.plsda_export_scores_cb.setToolTip("Export computed PLS-DA subject component scores to plsda_scores.csv.")

        self.plsda_save_plot_cb = QCheckBox("Save Visualization Plot (.png)")
        self.plsda_save_plot_cb.setChecked(True)
        self.plsda_save_plot_cb.setToolTip("Automatically save high-resolution PLS-DA scatter plot image to plsda_visualization.png.")

        plsda_form.addRow("Preset Mode:", self.plsda_mode_combo)
        plsda_form.addRow("PLS Components:", self.plsda_comp_spin)
        plsda_form.addRow("Group Metadata CSV:", csv_layout)
        plsda_form.addRow("Auto Export:", self.plsda_export_scores_cb)
        plsda_form.addRow("", self.plsda_save_plot_cb)
        
        plsda_layout.addWidget(plsda_group)
        
        self.run_plsda_btn = QPushButton("Run PLS-DA Analysis")
        self.run_plsda_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22; 
                color: white; 
                padding: 10px 15px; 
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        self.run_plsda_btn.clicked.connect(self.run_plsda_process)
        plsda_layout.addWidget(self.run_plsda_btn)
        
        plsda_layout.addStretch()
        self.stacked_widget.addWidget(plsda_widget)

    def on_plsda_mode_changed(self, index):
        if index == 0:  # Standard Mode
            self.plsda_comp_spin.setValue(10)
            self.plsda_comp_spin.setEnabled(False)
        elif index == 1:  # Custom Mode
            if self.plsda_comp_spin.value() == 0:
                self.plsda_comp_spin.setValue(10)
            self.plsda_comp_spin.setEnabled(True)

    def select_plsda_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Group Metadata CSV", "", "CSV Files (*.csv)")
        if filename:
            self.plsda_csv_input.setText(filename)

    def run_plsda_process(self):
        selected_dir = self.folder_input.text()
        if not selected_dir or not os.path.isdir(selected_dir):
            self.log("[ERROR] Please select a valid input directory first.")
            return

        folder_name = os.path.basename(os.path.normpath(selected_dir))
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        output_candidate = os.path.join(script_dir, f"output_{folder_name}")
        if os.path.exists(output_candidate):
            target_dir = output_candidate
        else:
            target_dir = selected_dir

        n_comp = self.plsda_comp_spin.value()
        
        self.log(">>> Starting PLS-DA Analysis...")
        self.log(f"Target Directory: {target_dir}")
        self.log(f"Parameters: PLS Components={n_comp}")

        plsda_script_path = os.path.abspath(os.path.join(script_dir, 'Visualize', 'Data_Plots', 'visualize_plsda.py'))
        if not os.path.exists(plsda_script_path):
            self.log(f"[ERROR] PLS-DA script not found at {plsda_script_path}")
            return

        self.plsda_process = QProcess(self)
        self.plsda_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.plsda_process.readyReadStandardOutput.connect(self.handle_plsda_stdout)
        self.plsda_process.finished.connect(self.plsda_process_finished)

        args = [
            plsda_script_path,
            "--output_dir", target_dir,
            "--n_components", str(n_comp)
        ]

        self.run_plsda_btn.setEnabled(False)
        self.plsda_process.start("python", args)

    def handle_plsda_stdout(self):
        data = self.plsda_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.log(line)

    def spharm_process_finished(self, exit_code, exit_status):
        self.run_spharm_btn.setEnabled(True)
        self.log(f">>> SPHARM Processing finished with exit code {exit_code}")

    def plsda_process_finished(self, exit_code, exit_status):
        self.run_plsda_btn.setEnabled(True)
        self.log(f">>> PLS-DA Analysis finished with exit code {exit_code}")
        
    def setup_feature_panel(self):
        feature_widget = QWidget()
        feature_layout = QVBoxLayout(feature_widget)
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
        self.stacked_widget.addWidget(feature_widget)

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
        selected_dir = self.folder_input.text()
        if not selected_dir or not os.path.isdir(selected_dir):
            self.log("[ERROR] Please select a valid input directory first.")
            return

        folder_name = os.path.basename(os.path.normpath(selected_dir))
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        output_candidate = os.path.join(script_dir, f"output_{folder_name}", "spharm_results")
        if os.path.exists(output_candidate):
            spharm_dir = output_candidate
        else:
            spharm_dir = selected_dir

        export_mesh = self.feature_export_mesh_cb.isChecked()
        export_coef = self.feature_export_coef_cb.isChecked()

        if not export_mesh and not export_coef:
            self.log("[WARNING] Please check at least one dataset type to export.")
            return

        self.feature_queue = []
        ext_script1 = os.path.abspath(os.path.join(script_dir, 'Data_Processing', 'extract_ml_features.py'))
        ext_script2 = os.path.abspath(os.path.join(script_dir, 'Data_Processing', 'extract_ml_features_coef.py'))

        if export_mesh and os.path.exists(ext_script1):
            self.feature_queue.append(ext_script1)
        if export_coef and os.path.exists(ext_script2):
            self.feature_queue.append(ext_script2)

        self.log(">>> Starting Feature Dataset Export...")
        self.log(f"SPHARM Target Directory: {spharm_dir}")
        self.log(f"Export Tasks: {len(self.feature_queue)} script(s)")

        self.run_feature_btn.setEnabled(False)
        self.run_next_feature_process(spharm_dir)

    def run_next_feature_process(self, spharm_dir):
        if not self.feature_queue:
            self.run_feature_btn.setEnabled(True)
            self.log(">>> ALL FEATURE DATASETS EXPORTED SUCCESSFULLY!")
            return

        next_script = self.feature_queue.pop(0)
        script_name = os.path.basename(next_script)
        self.log(f">>> Running: {script_name}...")

        self.feature_process = QProcess(self)
        self.feature_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.feature_process.readyReadStandardOutput.connect(self.handle_feature_stdout)
        self.feature_process.finished.connect(lambda code, status: self.on_single_feature_process_finished(code, status, spharm_dir))

        args = [next_script, "--spharm_dir", spharm_dir]
        self.feature_process.start("python", args)

    def on_single_feature_process_finished(self, exit_code, exit_status, spharm_dir):
        self.log(f">>> Step finished with code {exit_code}")
        self.run_next_feature_process(spharm_dir)

    def handle_feature_stdout(self):
        data = self.feature_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.log(line)

    def run_icp_process(self):
        input_dir = self.folder_input.text()
        if not input_dir or not os.path.isdir(input_dir):
            self.log("[ERROR] Please select a valid input directory first.")
            return
            
        spacing = self.icp_spacing_spin.value()
        voxels = self.icp_voxels_spin.value()
        max_iter = self.icp_max_iter_spin.value()
        tolerance = self.icp_tol_spin.value()
        pw_iter = self.icp_pw_iter_spin.value()
        pw_tol = self.icp_pw_tol_spin.value()
        pw_landmarks = self.icp_pw_landmarks_spin.value()
        interp = self.icp_interp_combo.currentText()
        
        self.log(">>> Starting ICP Registration...")
        self.log(f"Parameters: Spacing={spacing}, Voxels={voxels}, GW_MaxIter={max_iter}, GW_Tol={tolerance}, PW_MaxIter={pw_iter}, PW_Tol={pw_tol}, PW_Landmarks={pw_landmarks}, Interp={interp}")
        
        icp_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ICP', 'ICP.py'))
        if not os.path.exists(icp_script_path):
            self.log(f"[ERROR] ICP script not found at {icp_script_path}")
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
        
        self.run_icp_btn.setEnabled(False)
        self.icp_process.start("python", args)

    def handle_icp_stdout(self):
        data = self.icp_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.log(line)

    def icp_process_finished(self, exit_code, exit_status):
        self.run_icp_btn.setEnabled(True)
        self.log(f">>> ICP Registration finished with exit code {exit_code}")

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.folder_input.setText(directory)
            self.log(f"Selected directory: {directory}")
            self.load_subjects_from_directory(directory)

    def on_import_clicked(self):
        directory = self.folder_input.text()
        if directory and os.path.isdir(directory):
            self.load_subjects_from_directory(directory)
            self.log(">>> Data Imported Successfully.")
        else:
            self.log("[ERROR] Please select a valid directory first.")

    def load_subjects_from_directory(self, directory):
        import glob
        
        search_patterns = ["*.nrrd", "*.nii.gz", "*.nii", "*.vtk"]
        files = []
        for pattern in search_patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
            
        self.subjects_table.setRowCount(0)
        
        if not files:
            self.log(f"No valid image/mesh files (*.nrrd, *.nii.gz, *.vtk) found in {directory}")
            return
            
        self.log(f"Found {len(files)} files in directory.")
        self.subjects_table.setRowCount(len(files))
        
        for row, filepath in enumerate(files):
            filename = os.path.basename(filepath)
            self.subjects_table.setItem(row, 0, QTableWidgetItem(filename))
            self.subjects_table.setItem(row, 1, QTableWidgetItem("OK"))

    def on_subject_selection_changed(self):
        if self.display_on_click_cb.isChecked():
            self.display_selected_subject()

    def display_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        subject_name = self.subjects_table.item(row, 0).text()
        
        self.log(f"Displaying subject: {subject_name}")
        self.signal_subject_selected.emit(subject_name)



    def log(self, message):
        self.signal_log_message.emit(message)
