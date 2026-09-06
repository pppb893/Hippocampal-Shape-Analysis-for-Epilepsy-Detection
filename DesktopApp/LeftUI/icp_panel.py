import os
import sys
import glob
import subprocess
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
                             QHBoxLayout, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QTabBar, QRadioButton, QButtonGroup, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QLocale

def get_project_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def find_slicer_salt_exe():
    env_path = os.environ.get("SLICER_EXE")
    if env_path and os.path.isfile(env_path):
        return env_path
        
    root = get_project_root()
    local_candidates = [
        os.path.join(root, "SlicerSALT", "SlicerSALT.exe"),
        os.path.join(root, "Prerequisites", "SlicerSALT", "SlicerSALT.exe"),
        os.path.join(root, "Prerequisites", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
    ]
    for cand in local_candidates:
        if os.path.isfile(cand):
            return cand
            
    candidates = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
    if candidates:
        return candidates[0]
    return "C:\\Program Files\\SlicerSALT 6.0.0\\SlicerSALT.exe"

class IcpWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, tasks, adv_params, parent=None):
        super().__init__(parent)
        self.tasks = tasks  # list of tuples: (side_name, input_dir, output_dir)
        self.adv_params = adv_params

    def run(self):
        slicer_exe = find_slicer_salt_exe()
        if not os.path.isfile(slicer_exe):
            self.signal_log.emit(f"[ERROR] SlicerSALT not found at: {slicer_exe}")
            self.signal_finished.emit(False)
            return

        project_root = get_project_root()
        icp_script = os.path.join(project_root, "ICP", "ICP.py")
        if not os.path.isfile(icp_script):
            self.signal_log.emit(f"[ERROR] ICP.py script not found at: {icp_script}")
            self.signal_finished.emit(False)
            return

        overall_success = True
        for side_name, in_dir, out_dir in self.tasks:
            self.signal_log.emit(f"\n==================================================")
            self.signal_log.emit(f">>> Running ICP Registration for [{side_name.upper()} Hippocampus]")
            self.signal_log.emit(f"    Input:  {in_dir}")
            self.signal_log.emit(f"    Output: {out_dir}")
            self.signal_log.emit(f"==================================================")
            
            os.makedirs(out_dir, exist_ok=True)
            
            cmd = [
                slicer_exe,
                "--no-main-window",
                "--no-splash",
                "--python-script", icp_script,
                "--input_dir", in_dir,
                "--output_dir", out_dir,
                "--output_spacing", str(self.adv_params.get("spacing", 0.02)),
                "--output_voxels", str(self.adv_params.get("voxels", 128)),
                "--max_iterations", str(self.adv_params.get("max_iter", 20)),
                "--tolerance", str(self.adv_params.get("tolerance", 0.00005)),
                "--pairwise_iterations", str(self.adv_params.get("pw_iter", 100)),
                "--pairwise_tolerance", str(self.adv_params.get("pw_tol", 0.0001)),
                "--pairwise_landmarks", str(self.adv_params.get("pw_landmarks", 200)),
                "--interpolation", str(self.adv_params.get("interp", "NearestNeighbor"))
            ]
            
            try:
                kwargs = {}
                if os.name == 'nt':
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                    
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    **kwargs
                )
                
                for line in process.stdout:
                    clean = line.strip()
                    if clean:
                        self.signal_log.emit(clean)
                        
                process.wait()
                
                # Verify outputs
                aligned_nii_dir = os.path.join(out_dir, "aligned_nifti")
                aligned_mesh_dir = os.path.join(out_dir, "aligned_meshes")
                has_nii = os.path.isdir(aligned_nii_dir) and len(os.listdir(aligned_nii_dir)) > 0
                has_mesh = os.path.isdir(aligned_mesh_dir) and len(os.listdir(aligned_mesh_dir)) > 0
                
                if has_nii or has_mesh:
                    self.signal_log.emit(f"[OK] ICP {side_name} completed successfully.")
                else:
                    self.signal_log.emit(f"[WARNING] ICP {side_name} finished without expected output files.")
                    overall_success = False
            except Exception as e:
                self.signal_log.emit(f"[ERROR] Exception running ICP {side_name}: {str(e)}")
                overall_success = False

        self.signal_finished.emit(overall_success)


class IcpPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(str, str) # filepath, side_filter ("all", "lh", "rh")

    def __init__(self, get_folder_func, get_output_folder_func=None, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.get_output_folder = get_output_folder_func
        self.all_files = []
        self.current_side_filter = "all"
        self.setup_ui()

    def setup_ui(self):
        icp_layout = QVBoxLayout(self)
        icp_layout.setContentsMargins(10, 10, 10, 10)
        icp_layout.setSpacing(10)
        
        help_label = QLabel("Groupwise rigid ICP alignment for Left and Right Hippocampus masks from FastSurfer.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 11px;")
        icp_layout.addWidget(help_label)
        
        # 1. Directory Location Group
        dir_group = QGroupBox("FastSurfer Inputs && Output Location")
        dir_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setContentsMargins(10, 20, 10, 10)
        dir_layout.setSpacing(6)

        dir_row = QHBoxLayout()
        self.icp_dir_input = QLineEdit()
        self.icp_dir_input.setPlaceholderText("Auto (output_dir/icp)")
        dir_row.addWidget(self.icp_dir_input)

        browse_dir_btn = QPushButton("📁 Browse")
        browse_dir_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
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
        browse_dir_btn.clicked.connect(self.browse_output_directory)
        dir_row.addWidget(browse_dir_btn)

        reload_btn = QPushButton("🔄 Reload")
        reload_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
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
        reload_btn.clicked.connect(self.populate_results_table)
        dir_row.addWidget(reload_btn)

        dir_layout.addLayout(dir_row)
        icp_layout.addWidget(dir_group)

        # 2. Side Selection
        side_group = QGroupBox("Side Execution Option")
        side_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 8px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        side_layout = QHBoxLayout(side_group)
        side_layout.setContentsMargins(10, 15, 10, 10)
        
        self.side_both_rb = QRadioButton("Both Sides (LH && RH)")
        self.side_both_rb.setChecked(True)
        self.side_lh_rb = QRadioButton("Left (LH) Only")
        self.side_rh_rb = QRadioButton("Right (RH) Only")

        self.side_btn_group = QButtonGroup(self)
        self.side_btn_group.addButton(self.side_both_rb, 0)
        self.side_btn_group.addButton(self.side_lh_rb, 1)
        self.side_btn_group.addButton(self.side_rh_rb, 2)

        side_layout.addWidget(self.side_both_rb)
        side_layout.addWidget(self.side_lh_rb)
        side_layout.addWidget(self.side_rh_rb)
        icp_layout.addWidget(side_group)

        # 3. Main Action Button & Status Hint
        self.run_icp_btn = QPushButton("▶ Run Groupwise ICP Registration")
        self.run_icp_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 9px 15px;
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
        self.run_icp_btn.clicked.connect(self.run_icp_process)
        icp_layout.addWidget(self.run_icp_btn)

        self.icp_status_hint = QLabel("")
        self.icp_status_hint.setWordWrap(True)
        self.icp_status_hint.setStyleSheet("font-size: 11px; padding: 5px 8px; border-radius: 4px;")
        icp_layout.addWidget(self.icp_status_hint)

        # 4. Collapsible Advanced Parameters
        self.toggle_adv_btn = QPushButton("⚙️ Advanced Parameters ▾")
        self.toggle_adv_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #57606f;
                font-weight: bold;
                font-size: 11px;
                text-align: left;
                padding: 4px 6px;
                border: none;
            }
            QPushButton:hover {
                color: #2f3542;
            }
        """)
        self.toggle_adv_btn.clicked.connect(self.toggle_advanced_params)
        icp_layout.addWidget(self.toggle_adv_btn)

        self.adv_container = QFrame()
        self.adv_container.setStyleSheet("""
            QFrame {
                border: 1px solid #e1e2e6;
                border-radius: 5px;
                background-color: #fafbfc;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 3px;
                background-color: white;
                color: #34495e;
                font-size: 11px;
            }
            QLabel {
                font-size: 11px;
                color: #2c3e50;
            }
        """)
        adv_form = QFormLayout(self.adv_container)
        adv_form.setContentsMargins(10, 10, 10, 10)
        adv_form.setSpacing(6)

        self.icp_mode_combo = QComboBox()
        self.icp_mode_combo.addItems(["Production Mode (Default)", "Custom Mode"])
        self.icp_mode_combo.currentIndexChanged.connect(self.on_icp_mode_changed)

        self.icp_spacing_spin = QDoubleSpinBox()
        self.icp_spacing_spin.setLocale(QLocale.c())
        self.icp_spacing_spin.setRange(0.001, 10.0)
        self.icp_spacing_spin.setDecimals(3)
        self.icp_spacing_spin.setValue(0.02)
        self.icp_spacing_spin.setEnabled(False)

        self.icp_voxels_spin = QSpinBox()
        self.icp_voxels_spin.setLocale(QLocale.c())
        self.icp_voxels_spin.setRange(16, 1024)
        self.icp_voxels_spin.setValue(128)
        self.icp_voxels_spin.setEnabled(False)

        self.icp_max_iter_spin = QSpinBox()
        self.icp_max_iter_spin.setRange(1, 1000)
        self.icp_max_iter_spin.setValue(20)
        self.icp_max_iter_spin.setEnabled(False)

        self.icp_tol_spin = QDoubleSpinBox()
        self.icp_tol_spin.setLocale(QLocale.c())
        self.icp_tol_spin.setRange(0.000001, 1.0)
        self.icp_tol_spin.setDecimals(6)
        self.icp_tol_spin.setValue(0.00005)
        self.icp_tol_spin.setEnabled(False)

        self.icp_pw_iter_spin = QSpinBox()
        self.icp_pw_iter_spin.setRange(1, 1000)
        self.icp_pw_iter_spin.setValue(100)
        self.icp_pw_iter_spin.setEnabled(False)

        self.icp_pw_tol_spin = QDoubleSpinBox()
        self.icp_pw_tol_spin.setLocale(QLocale.c())
        self.icp_pw_tol_spin.setRange(0.000001, 1.0)
        self.icp_pw_tol_spin.setDecimals(6)
        self.icp_pw_tol_spin.setValue(0.0001)
        self.icp_pw_tol_spin.setEnabled(False)

        self.icp_pw_landmarks_spin = QSpinBox()
        self.icp_pw_landmarks_spin.setRange(10, 5000)
        self.icp_pw_landmarks_spin.setValue(200)
        self.icp_pw_landmarks_spin.setEnabled(False)

        self.icp_interp_combo = QComboBox()
        self.icp_interp_combo.addItems(["NearestNeighbor", "Linear", "BSpline"])
        self.icp_interp_combo.setEnabled(False)

        adv_form.addRow("Preset Mode:", self.icp_mode_combo)
        adv_form.addRow("Output Spacing:", self.icp_spacing_spin)
        adv_form.addRow("Output Voxels:", self.icp_voxels_spin)
        adv_form.addRow("Groupwise Max Iterations:", self.icp_max_iter_spin)
        adv_form.addRow("Groupwise Tolerance:", self.icp_tol_spin)
        adv_form.addRow("Pairwise Max Iterations:", self.icp_pw_iter_spin)
        adv_form.addRow("Pairwise Tolerance:", self.icp_pw_tol_spin)
        adv_form.addRow("Pairwise Landmarks:", self.icp_pw_landmarks_spin)
        adv_form.addRow("Interpolation Mode:", self.icp_interp_combo)

        self.adv_container.setVisible(False)
        icp_layout.addWidget(self.adv_container)

        # 5. Results Table with Category Tabs
        res_group = QGroupBox("ICP Aligned Meshes & Results")
        res_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        res_layout = QVBoxLayout(res_group)
        res_layout.setContentsMargins(10, 20, 10, 10)
        res_layout.setSpacing(6)

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("All")
        self.tab_bar.addTab("Left (LH)")
        self.tab_bar.addTab("Right (RH)")
        self.tab_bar.setExpanding(True)
        self.tab_bar.setStyleSheet("""
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 6px 14px;
                margin-right: 3px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #ced6e0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
                color: #2c3e50;
                border: 1px solid #b2bec3;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
            }
        """)
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        res_layout.addWidget(self.tab_bar)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Aligned Mesh Name", "Side", "File Path"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dcdde1;
                gridline-color: #ecf0f1;
                font-size: 11px;
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
        self.results_table.itemSelectionChanged.connect(self.on_mesh_selected)
        self.results_table.cellClicked.connect(self.on_cell_clicked)
        res_layout.addWidget(self.results_table)

        icp_layout.addWidget(res_group)

        self.update_run_button_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_run_button_state()

    def toggle_advanced_params(self):
        should_show = self.adv_container.isHidden()
        self.adv_container.setVisible(should_show)
        self.toggle_adv_btn.setText("⚙️ Advanced Parameters ▴" if should_show else "⚙️ Advanced Parameters ▾")

    def on_icp_mode_changed(self, index):
        is_custom = (index == 1)
        for widget in [self.icp_spacing_spin, self.icp_voxels_spin, self.icp_max_iter_spin,
                       self.icp_tol_spin, self.icp_pw_iter_spin, self.icp_pw_tol_spin,
                       self.icp_pw_landmarks_spin, self.icp_interp_combo]:
            widget.setEnabled(is_custom)
            
        if not is_custom:
            self.icp_spacing_spin.setValue(0.02)
            self.icp_voxels_spin.setValue(128)
            self.icp_max_iter_spin.setValue(20)
            self.icp_tol_spin.setValue(0.00005)
            self.icp_pw_iter_spin.setValue(100)
            self.icp_pw_tol_spin.setValue(0.0001)
            self.icp_pw_landmarks_spin.setValue(200)
            self.icp_interp_combo.setCurrentText("NearestNeighbor")

    def browse_output_directory(self):
        initial = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(self, "Select ICP Output Directory", initial)
        if folder:
            self.icp_dir_input.setText(folder)
            self.populate_results_table()
            self.update_run_button_state()

    def get_source_paths(self):
        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if not out_base or not os.path.isdir(out_base):
            return None, None, None
            
        lh_dir = os.path.join(out_base, "fastsurfer", "left_hippocampus")
        rh_dir = os.path.join(out_base, "fastsurfer", "right_hippocampus")
        icp_out = os.path.join(out_base, "icp")
        return lh_dir, rh_dir, icp_out

    def update_run_button_state(self):
        lh_dir, rh_dir, icp_out = self.get_source_paths()
        
        has_lh = bool(lh_dir and os.path.isdir(lh_dir) and glob.glob(os.path.join(lh_dir, "*.nii.gz")))
        has_rh = bool(rh_dir and os.path.isdir(rh_dir) and glob.glob(os.path.join(rh_dir, "*.nii.gz")))
        
        custom_dir = self.icp_dir_input.text().strip()
        if not custom_dir and icp_out:
            self.icp_dir_input.setText(icp_out)
            
        if not has_lh and not has_rh:
            self.run_icp_btn.setEnabled(False)
            msg = "🔒 Locked: FastSurfer outputs (left_hippocampus / right_hippocampus) not found. Please run FastSurfer first."
            self.run_icp_btn.setToolTip(msg)
            self.icp_status_hint.setText(msg)
            self.icp_status_hint.setStyleSheet("""
                color: #c0392b; 
                background-color: #fdedec; 
                border: 1px solid #f5b7b1; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)
        else:
            self.run_icp_btn.setEnabled(True)
            self.run_icp_btn.setToolTip("Click to run Groupwise ICP Registration")
            lh_count = len(glob.glob(os.path.join(lh_dir, "*.nii.gz"))) if has_lh else 0
            rh_count = len(glob.glob(os.path.join(rh_dir, "*.nii.gz"))) if has_rh else 0
            target_out = custom_dir if custom_dir else icp_out
            msg = f"✓ Ready: Detected {lh_count} Left & {rh_count} Right subjects. Output will be saved to: {target_out}"
            self.icp_status_hint.setText(msg)
            self.icp_status_hint.setStyleSheet("""
                color: #1e8449; 
                background-color: #eafaf1; 
                border: 1px solid #a9dfbf; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)

        self.populate_results_table()

    def run_icp_process(self):
        lh_dir, rh_dir, default_icp_out = self.get_source_paths()
        target_base = self.icp_dir_input.text().strip() or default_icp_out
        
        if not target_base:
            self.signal_log_message.emit("[ERROR] Output directory is missing. Please configure Output in Data Importer.")
            return

        tasks = []
        mode = self.side_btn_group.checkedId()
        
        # 0: Both, 1: Left only, 2: Right only
        if mode in (0, 1):
            if lh_dir and os.path.isdir(lh_dir):
                tasks.append(("left", lh_dir, os.path.join(target_base, "left")))
            else:
                self.signal_log_message.emit("[WARNING] Left hippocampus folder not found.")
                
        if mode in (0, 2):
            if rh_dir and os.path.isdir(rh_dir):
                tasks.append(("right", rh_dir, os.path.join(target_base, "right")))
            else:
                self.signal_log_message.emit("[WARNING] Right hippocampus folder not found.")

        if not tasks:
            self.signal_log_message.emit("[ERROR] No valid hippocampus folders found to run ICP.")
            return

        adv_params = {
            "spacing": self.icp_spacing_spin.value(),
            "voxels": self.icp_voxels_spin.value(),
            "max_iter": self.icp_max_iter_spin.value(),
            "tolerance": self.icp_tol_spin.value(),
            "pw_iter": self.icp_pw_iter_spin.value(),
            "pw_tol": self.icp_pw_tol_spin.value(),
            "pw_landmarks": self.icp_pw_landmarks_spin.value(),
            "interp": self.icp_interp_combo.currentText()
        }

        self.run_icp_btn.setEnabled(False)
        self.results_table.setRowCount(0)
        self.signal_log_message.emit(">>> Initiating Groupwise ICP Alignment Pipeline...")

        self.worker = IcpWorker(tasks, adv_params)
        self.worker.signal_log.connect(self.signal_log_message.emit)
        self.worker.signal_finished.connect(self.on_icp_finished)
        self.worker.start()

    def on_icp_finished(self, success):
        self.update_run_button_state()
        if success:
            self.signal_log_message.emit(">>> Groupwise ICP Registration Pipeline completed successfully.")
        else:
            self.signal_log_message.emit("[ERROR] ICP Registration completed with warnings or errors.")
        self.populate_results_table()

    def on_tab_changed(self, index):
        if index == 1:
            self.current_side_filter = "lh"
        elif index == 2:
            self.current_side_filter = "rh"
        else:
            self.current_side_filter = "all"
        self.update_table_display()

    def populate_results_table(self):
        target_base = self.icp_dir_input.text().strip()
        if not target_base:
            _, _, default_icp_out = self.get_source_paths()
            target_base = default_icp_out
            
        self.all_files = []
        if target_base and os.path.isdir(target_base):
            search_dirs = [
                (os.path.join(target_base, "left", "aligned_meshes"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right", "aligned_meshes"), "rh", "Right (RH)"),
                (os.path.join(target_base, "aligned_meshes"), "all", "Aligned"),
            ]
            
            seen = set()
            for directory, side_key, side_label in search_dirs:
                if os.path.isdir(directory):
                    for vtk_file in glob.glob(os.path.join(directory, "*.vtk")):
                        norm_p = os.path.normpath(vtk_file)
                        if norm_p not in seen:
                            seen.add(norm_p)
                            basename = os.path.basename(norm_p)
                            
                            cur_key = side_key
                            cur_label = side_label
                            if cur_key == "all":
                                if basename.startswith("lh_") or "left" in norm_p.lower():
                                    cur_key, cur_label = "lh", "Left (LH)"
                                elif basename.startswith("rh_") or "right" in norm_p.lower():
                                    cur_key, cur_label = "rh", "Right (RH)"

                            self.all_files.append({
                                "filename": basename,
                                "side": cur_label,
                                "side_key": cur_key,
                                "filepath": norm_p
                            })

        # Update tab counts
        all_count = len(self.all_files)
        lh_count = sum(1 for f in self.all_files if f["side_key"] == "lh")
        rh_count = sum(1 for f in self.all_files if f["side_key"] == "rh")
        self.tab_bar.setTabText(0, f"All ({all_count})")
        self.tab_bar.setTabText(1, f"Left ({lh_count})")
        self.tab_bar.setTabText(2, f"Right ({rh_count})")

        self.update_table_display()

    def update_table_display(self):
        if self.current_side_filter == "lh":
            display_files = [f for f in self.all_files if f["side_key"] == "lh"]
        elif self.current_side_filter == "rh":
            display_files = [f for f in self.all_files if f["side_key"] == "rh"]
        else:
            display_files = self.all_files

        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(display_files))
        for i, item in enumerate(display_files):
            name_item = QTableWidgetItem(item["filename"])
            name_item.setData(Qt.ItemDataRole.UserRole, item["filepath"])
            self.results_table.setItem(i, 0, name_item)

            side_item = QTableWidgetItem(item["side"])
            side_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if item["side_key"] == "lh":
                side_item.setForeground(Qt.GlobalColor.blue)
            elif item["side_key"] == "rh":
                side_item.setForeground(Qt.GlobalColor.darkYellow)
            self.results_table.setItem(i, 1, side_item)

            path_item = QTableWidgetItem(item["filepath"])
            path_item.setToolTip(item["filepath"])
            self.results_table.setItem(i, 2, path_item)

        self.results_table.blockSignals(False)
        self.results_table.clearSelection()

    def on_cell_clicked(self, row, col):
        self.results_table.selectRow(row)
        self.on_mesh_selected()

    def on_mesh_selected(self):
        selected_items = self.results_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            name_item = self.results_table.item(row, 0)
            if name_item:
                filepath = name_item.data(Qt.ItemDataRole.UserRole)
                if filepath:
                    self.signal_mesh_selected.emit(filepath, self.current_side_filter)
