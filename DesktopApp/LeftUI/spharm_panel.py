import os
import sys
import glob
import subprocess
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QGroupBox, QFormLayout, QComboBox, QSpinBox,
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

class SpharmWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, tasks, adv_params, parent=None):
        super().__init__(parent)
        self.tasks = tasks  # list of tuples: (side_name, input_aligned_nii_dir, output_spharm_dir)
        self.adv_params = adv_params

    def run(self):
        slicer_exe = find_slicer_salt_exe()
        if not os.path.isfile(slicer_exe):
            self.signal_log.emit(f"[ERROR] SlicerSALT not found at: {slicer_exe}")
            self.signal_finished.emit(False)
            return

        project_root = get_project_root()
        batch_script = os.path.join(project_root, "SPHARM", "run_spharm_batch.py")
        realign_script = os.path.join(project_root, "SPHARM", "realign_spharm.py")

        if not os.path.isfile(batch_script):
            self.signal_log.emit(f"[ERROR] run_spharm_batch.py not found at: {batch_script}")
            self.signal_finished.emit(False)
            return

        overall_success = True
        for side_name, in_dir, out_dir in self.tasks:
            self.signal_log.emit(f"\n==================================================")
            self.signal_log.emit(f">>> Running Batch SPHARM for [{side_name.upper()} Hippocampus]")
            self.signal_log.emit(f"    Input:  {in_dir}")
            self.signal_log.emit(f"    Output: {out_dir}")
            self.signal_log.emit(f"==================================================")
            
            os.makedirs(out_dir, exist_ok=True)
            
            # Step 1: Batch SPHARM via SlicerSALT
            cmd = [
                slicer_exe,
                "--no-main-window",
                "--no-splash",
                "--python-script", batch_script,
                "--input_dir", in_dir,
                "--output_dir", out_dir,
                "--num_iterations", str(self.adv_params.get("num_iter", 1000)),
                "--subdiv_level", str(self.adv_params.get("subdiv", 10)),
                "--spharm_degree", str(self.adv_params.get("degree", 12))
            ]
            if self.adv_params.get("regen_only", False):
                cmd.append("--regen_spharm_only")

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
                
                spharm_results_dir = os.path.join(out_dir, "spharm_results")
                has_vtk = os.path.isdir(spharm_results_dir) and len(glob.glob(os.path.join(spharm_results_dir, "*_SPHARM*.vtk"))) > 0
                
                if not has_vtk:
                    self.signal_log.emit(f"[WARNING] SPHARM step for {side_name} did not generate expected VTK meshes.")
                    overall_success = False
                    continue

                self.signal_log.emit(f"[OK] SPHARM surface meshes generated for {side_name}.")

                # Step 2: Anatomical Re-alignment (Head/Tail orientation)
                if os.path.isfile(realign_script):
                    self.signal_log.emit(f">>> Re-aligning anatomical landmarks for {side_name}...")
                    realign_cmd = [
                        sys.executable,
                        realign_script,
                        "--spharm_dir", spharm_results_dir
                    ]
                    realign_proc = subprocess.Popen(
                        realign_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        **kwargs
                    )
                    for rline in realign_proc.stdout:
                        rclean = rline.strip()
                        if rclean:
                            self.signal_log.emit(f"  [realign] {rclean}")
                    realign_proc.wait()
                    if realign_proc.returncode == 0:
                        self.signal_log.emit(f"[OK] SPHARM Re-alignment completed for {side_name}.")
                    else:
                        self.signal_log.emit(f"[WARNING] SPHARM Re-alignment finished with non-zero exit code: {realign_proc.returncode}")
                else:
                    self.signal_log.emit(f"[INFO] realign_spharm.py not found, skipping landmark realignment.")

            except Exception as e:
                self.signal_log.emit(f"[ERROR] Exception running SPHARM {side_name}: {str(e)}")
                overall_success = False

        self.signal_finished.emit(overall_success)


class SpharmPanel(QWidget):
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
        spharm_layout = QVBoxLayout(self)
        spharm_layout.setContentsMargins(10, 10, 10, 10)
        spharm_layout.setSpacing(10)
        
        help_label = QLabel("SPHARM-PDM spherical parameterization and shape model generation for ICP-aligned hippocampus meshes.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 11px;")
        spharm_layout.addWidget(help_label)
        
        # 1. Directory Location Group
        dir_group = QGroupBox("ICP Inputs && SPHARM Output Location")
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
        self.spharm_dir_input = QLineEdit()
        self.spharm_dir_input.setPlaceholderText("Auto (output_dir/spharm)")
        dir_row.addWidget(self.spharm_dir_input)

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
        spharm_layout.addWidget(dir_group)

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
        spharm_layout.addWidget(side_group)

        # 3. Main Action Button & Status Hint
        self.run_spharm_btn = QPushButton("▶ Run Batch SPHARM Processing")
        self.run_spharm_btn.setStyleSheet("""
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
        self.run_spharm_btn.clicked.connect(self.run_spharm_process)
        spharm_layout.addWidget(self.run_spharm_btn)

        self.spharm_status_hint = QLabel("")
        self.spharm_status_hint.setWordWrap(True)
        self.spharm_status_hint.setStyleSheet("font-size: 11px; padding: 5px 8px; border-radius: 4px;")
        spharm_layout.addWidget(self.spharm_status_hint)

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
        spharm_layout.addWidget(self.toggle_adv_btn)

        self.adv_container = QFrame()
        self.adv_container.setStyleSheet("""
            QFrame {
                border: 1px solid #e1e2e6;
                border-radius: 5px;
                background-color: #fafbfc;
            }
            QSpinBox, QComboBox {
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

        self.spharm_mode_combo = QComboBox()
        self.spharm_mode_combo.addItems(["Production Mode (Default)", "Fast Test Mode", "Custom Mode"])
        self.spharm_mode_combo.currentIndexChanged.connect(self.on_spharm_mode_changed)

        self.spharm_iter_spin = QSpinBox()
        self.spharm_iter_spin.setRange(50, 5000)
        self.spharm_iter_spin.setValue(1000)
        self.spharm_iter_spin.setSingleStep(100)
        self.spharm_iter_spin.setEnabled(False)

        self.spharm_subdiv_spin = QSpinBox()
        self.spharm_subdiv_spin.setRange(1, 30)
        self.spharm_subdiv_spin.setValue(10)
        self.spharm_subdiv_spin.setEnabled(False)

        self.spharm_degree_spin = QSpinBox()
        self.spharm_degree_spin.setRange(1, 30)
        self.spharm_degree_spin.setValue(12)
        self.spharm_degree_spin.setEnabled(False)

        self.spharm_regen_cb = QCheckBox("Regenerate SPHARM Only (Reuse existing _surf.vtk and _para.vtk)")
        self.spharm_regen_cb.setChecked(False)

        adv_form.addRow("Preset Mode:", self.spharm_mode_combo)
        adv_form.addRow("GenParaMesh Iterations:", self.spharm_iter_spin)
        adv_form.addRow("Subdivision Level:", self.spharm_subdiv_spin)
        adv_form.addRow("SPHARM Degree:", self.spharm_degree_spin)
        adv_form.addRow("", self.spharm_regen_cb)

        self.adv_container.setVisible(False)
        spharm_layout.addWidget(self.adv_container)

        # 5. Results Table with Category Tabs
        res_group = QGroupBox("SPHARM Surface Meshes && Models")
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
        self.results_table.setHorizontalHeaderLabels(["SPHARM Mesh Name", "Side", "File Path"])
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

        spharm_layout.addWidget(res_group)

        self.update_run_button_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_run_button_state()

    def toggle_advanced_params(self):
        should_show = self.adv_container.isHidden()
        self.adv_container.setVisible(should_show)
        self.toggle_adv_btn.setText("⚙️ Advanced Parameters ▴" if should_show else "⚙️ Advanced Parameters ▾")

    def on_spharm_mode_changed(self, index):
        if index == 0:  # Production
            self.spharm_iter_spin.setValue(1000)
            self.spharm_subdiv_spin.setValue(10)
            self.spharm_degree_spin.setValue(12)
            self.spharm_iter_spin.setEnabled(False)
            self.spharm_subdiv_spin.setEnabled(False)
            self.spharm_degree_spin.setEnabled(False)
        elif index == 1:  # Fast Test
            self.spharm_iter_spin.setValue(200)
            self.spharm_subdiv_spin.setValue(5)
            self.spharm_degree_spin.setValue(6)
            self.spharm_iter_spin.setEnabled(False)
            self.spharm_subdiv_spin.setEnabled(False)
            self.spharm_degree_spin.setEnabled(False)
        elif index == 2:  # Custom
            self.spharm_iter_spin.setEnabled(True)
            self.spharm_subdiv_spin.setEnabled(True)
            self.spharm_degree_spin.setEnabled(True)

    def browse_output_directory(self):
        initial = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(self, "Select SPHARM Output Directory", initial)
        if folder:
            self.spharm_dir_input.setText(folder)
            self.populate_results_table()
            self.update_run_button_state()

    def get_source_paths(self):
        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if not out_base or not os.path.isdir(out_base):
            return None, None, None
            
        lh_in = os.path.join(out_base, "icp", "left", "aligned_nifti")
        rh_in = os.path.join(out_base, "icp", "right", "aligned_nifti")
        spharm_out = os.path.join(out_base, "spharm")
        return lh_in, rh_in, spharm_out

    def update_run_button_state(self):
        lh_in, rh_in, spharm_out = self.get_source_paths()
        
        has_lh = bool(lh_in and os.path.isdir(lh_in) and glob.glob(os.path.join(lh_in, "*.nii.gz")))
        has_rh = bool(rh_in and os.path.isdir(rh_in) and glob.glob(os.path.join(rh_in, "*.nii.gz")))
        
        custom_dir = self.spharm_dir_input.text().strip()
        if not custom_dir and spharm_out:
            self.spharm_dir_input.setText(spharm_out)
            
        if not has_lh and not has_rh:
            self.run_spharm_btn.setEnabled(False)
            msg = "🔒 Locked: Aligned NIfTI masks from ICP (left/right aligned_nifti) not found. Please run ICP Registration first."
            self.run_spharm_btn.setToolTip(msg)
            self.spharm_status_hint.setText(msg)
            self.spharm_status_hint.setStyleSheet("""
                color: #c0392b; 
                background-color: #fdedec; 
                border: 1px solid #f5b7b1; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)
        else:
            self.run_spharm_btn.setEnabled(True)
            self.run_spharm_btn.setToolTip("Click to run Batch SPHARM Processing")
            lh_count = len(glob.glob(os.path.join(lh_in, "*.nii.gz"))) if has_lh else 0
            rh_count = len(glob.glob(os.path.join(rh_in, "*.nii.gz"))) if has_rh else 0
            target_out = custom_dir if custom_dir else spharm_out
            msg = f"✓ Ready: Detected {lh_count} Left & {rh_count} Right ICP aligned subjects. Output: {target_out}"
            self.spharm_status_hint.setText(msg)
            self.spharm_status_hint.setStyleSheet("""
                color: #1e8449; 
                background-color: #eafaf1; 
                border: 1px solid #a9dfbf; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)

        self.populate_results_table()

    def run_spharm_process(self):
        lh_in, rh_in, default_spharm_out = self.get_source_paths()
        target_base = self.spharm_dir_input.text().strip() or default_spharm_out
        
        if not target_base:
            self.signal_log_message.emit("[ERROR] Output directory is missing. Please configure Output in Data Importer.")
            return

        tasks = []
        mode = self.side_btn_group.checkedId()
        
        # 0: Both, 1: Left only, 2: Right only
        if mode in (0, 1):
            if lh_in and os.path.isdir(lh_in):
                tasks.append(("left", lh_in, os.path.join(target_base, "left")))
            else:
                self.signal_log_message.emit("[WARNING] Left ICP aligned_nifti folder not found.")
                
        if mode in (0, 2):
            if rh_in and os.path.isdir(rh_in):
                tasks.append(("right", rh_in, os.path.join(target_base, "right")))
            else:
                self.signal_log_message.emit("[WARNING] Right ICP aligned_nifti folder not found.")

        if not tasks:
            self.signal_log_message.emit("[ERROR] No valid ICP aligned folders found to run SPHARM.")
            return

        adv_params = {
            "num_iter": self.spharm_iter_spin.value(),
            "subdiv": self.spharm_subdiv_spin.value(),
            "degree": self.spharm_degree_spin.value(),
            "regen_only": self.spharm_regen_cb.isChecked()
        }

        self.run_spharm_btn.setEnabled(False)
        self.results_table.setRowCount(0)
        self.signal_log_message.emit(">>> Initiating Batch SPHARM-PDM Pipeline...")

        self.worker = SpharmWorker(tasks, adv_params)
        self.worker.signal_log.connect(self.signal_log_message.emit)
        self.worker.signal_finished.connect(self.on_spharm_finished)
        self.worker.start()

    def on_spharm_finished(self, success):
        self.update_run_button_state()
        if success:
            self.signal_log_message.emit(">>> Batch SPHARM-PDM Pipeline completed successfully.")
        else:
            self.signal_log_message.emit("[ERROR] SPHARM Pipeline completed with warnings or errors.")
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
        target_base = self.spharm_dir_input.text().strip()
        if not target_base:
            _, _, default_spharm_out = self.get_source_paths()
            target_base = default_spharm_out
            
        self.all_files = []
        if target_base and os.path.isdir(target_base):
            search_dirs = [
                (os.path.join(target_base, "left", "spharm_results"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right", "spharm_results"), "rh", "Right (RH)"),
                (os.path.join(target_base, "spharm_results"), "all", "SPHARM"),
            ]
            
            seen = set()
            for directory, side_key, side_label in search_dirs:
                if os.path.isdir(directory):
                    for vtk_file in glob.glob(os.path.join(directory, "*_SPHARM*.vtk")):
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
