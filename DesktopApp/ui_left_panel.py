import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QFileDialog, QStackedWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QDoubleSpinBox, QSpinBox, 
                             QGroupBox, QFormLayout, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal, QProcess

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
        
        # Premium CSS styling for the GroupBox and inputs
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
        
        self.icp_spacing_spin = QDoubleSpinBox()
        self.icp_spacing_spin.setRange(0.001, 10.0)
        self.icp_spacing_spin.setDecimals(3)
        self.icp_spacing_spin.setValue(0.02)
        self.icp_spacing_spin.setSingleStep(0.01)
        self.icp_spacing_spin.setToolTip("Image resolution. A smaller number makes the 3D model look smoother, but takes longer to process. (Recommended: 0.02)")
        
        self.icp_voxels_spin = QSpinBox()
        self.icp_voxels_spin.setRange(16, 1024)
        self.icp_voxels_spin.setValue(128)
        self.icp_voxels_spin.setSingleStep(16)
        self.icp_voxels_spin.setToolTip("The size of the 3D box that contains the model. A larger number makes the box bigger. (Recommended: 128)")
        
        self.icp_max_iter_spin = QSpinBox()
        self.icp_max_iter_spin.setRange(1, 1000)
        self.icp_max_iter_spin.setValue(20)
        self.icp_max_iter_spin.setToolTip("The maximum number of times the program will try to find the 'average shape' of all patients. More rounds = more accurate but slower. (Recommended: 20)")
        
        self.icp_tol_spin = QDoubleSpinBox()
        self.icp_tol_spin.setRange(0.00001, 1.0)
        self.icp_tol_spin.setDecimals(5)
        self.icp_tol_spin.setValue(0.00005)
        self.icp_tol_spin.setSingleStep(0.00001)
        self.icp_tol_spin.setToolTip("The resting point. If the average shape barely changes after a round, the program will stop early to save time. (Recommended: 0.00005)")

        self.icp_pw_iter_spin = QSpinBox()
        self.icp_pw_iter_spin.setRange(1, 10000)
        self.icp_pw_iter_spin.setValue(100)
        self.icp_pw_iter_spin.setToolTip("The maximum number of attempts the program makes to match and overlap two shapes perfectly. (Recommended: 100)")

        self.icp_pw_tol_spin = QDoubleSpinBox()
        self.icp_pw_tol_spin.setRange(0.000001, 1.0)
        self.icp_pw_tol_spin.setDecimals(6)
        self.icp_pw_tol_spin.setValue(0.0001)
        self.icp_pw_tol_spin.setSingleStep(0.00001)
        self.icp_pw_tol_spin.setToolTip("The acceptable matching error. If two shapes overlap closer than this number, the program considers it a success. (Recommended: 0.0001)")
        
        self.icp_pw_landmarks_spin = QSpinBox()
        self.icp_pw_landmarks_spin.setRange(10, 10000)
        self.icp_pw_landmarks_spin.setValue(200)
        self.icp_pw_landmarks_spin.setToolTip("The number of points sampled from the shape to calculate the match. More points = better accuracy, but much slower. (Recommended: 200)")

        self.icp_interp_combo = QComboBox()
        self.icp_interp_combo.addItems(["NearestNeighbor", "Linear", "BSpline", "Lanczos", "WindowedSinc"])
        self.icp_interp_combo.setCurrentText("NearestNeighbor")
        self.icp_interp_combo.setToolTip("The drawing technique used to recreate the 3D image. 'NearestNeighbor' is highly recommended to keep the shape's original sharp edges intact.")
        
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

    def setup_spharm_panel(self):
        spharm_widget = QWidget()
        spharm_layout = QVBoxLayout(spharm_widget)
        help_label = QLabel("This module processes SPHARM parameterization.")
        help_label.setWordWrap(True)
        spharm_layout.addWidget(help_label)
        
        self.run_spharm_btn = QPushButton("Run SPHARM Processing")
        self.run_spharm_btn.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px; font-weight: bold;")
        self.run_spharm_btn.clicked.connect(lambda: self.log(">>> Running SPHARM Processing..."))
        spharm_layout.addWidget(self.run_spharm_btn)
        spharm_layout.addStretch()
        self.stacked_widget.addWidget(spharm_widget)
        
    def setup_plsda_panel(self):
        plsda_widget = QWidget()
        plsda_layout = QVBoxLayout(plsda_widget)
        help_label = QLabel("This module runs PLS-DA analysis on extracted features.")
        help_label.setWordWrap(True)
        plsda_layout.addWidget(help_label)
        
        self.run_plsda_btn = QPushButton("Run PLS-DA Analysis")
        self.run_plsda_btn.setStyleSheet("background-color: #e67e22; color: white; padding: 5px; font-weight: bold;")
        self.run_plsda_btn.clicked.connect(lambda: self.log(">>> Running PLS-DA Analysis..."))
        plsda_layout.addWidget(self.run_plsda_btn)
        plsda_layout.addStretch()
        self.stacked_widget.addWidget(plsda_widget)
        
    def setup_feature_panel(self):
        feature_widget = QWidget()
        feature_layout = QVBoxLayout(feature_widget)
        help_label = QLabel("Feature Extraction module (Coming soon).")
        help_label.setWordWrap(True)
        feature_layout.addWidget(help_label)
        feature_layout.addStretch()
        self.stacked_widget.addWidget(feature_widget)

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
