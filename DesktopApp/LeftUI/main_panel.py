import os
import sys
import json
import glob
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
    QLineEdit, QGroupBox, QMenu
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint

class MainPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, import_panel=None, fastsurfer_panel=None, icp_panel=None, spharm_panel=None, parent=None):
        super().__init__(parent)
        self.import_panel = import_panel
        self.fastsurfer_panel = fastsurfer_panel
        self.icp_panel = icp_panel
        self.spharm_panel = spharm_panel
        self.is_running = False

        self.load_history()
        self.setup_ui()
        self.connect_pipeline_signals()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_stage_preview()
        self.populate_main_table()

    def load_history(self):
        self.history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app_history.json")
        self.recent_dirs = []
        self.last_input_dir = None
        self.last_output_dir = None
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.recent_dirs = data.get("recent_directories", [])
                    self.last_input_dir = data.get("last_input_directory", None)
                    self.last_output_dir = data.get("last_output_directory", None)
            except Exception:
                pass

    def save_history(self):
        data = {
            "recent_directories": self.recent_dirs,
            "last_input_directory": self.last_input_dir,
            "last_output_directory": self.last_output_dir
        }
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    def add_to_history(self, directory):
        if not directory or not os.path.isdir(directory):
            return
        if directory in self.recent_dirs:
            self.recent_dirs.remove(directory)
        self.recent_dirs.insert(0, directory)
        self.recent_dirs = self.recent_dirs[:10]
        self.last_input_dir = directory
        self.save_history()

    @staticmethod
    def find_valid_mri_files(directory):
        """
        Scans for valid MRI images only (.nii.gz, .nii, .mgz).
        Excludes non-MRI files, masks, segmentations, and 3D meshes.
        """
        if not directory or not os.path.isdir(directory):
            return []

        mri_extensions = (".nii.gz", ".nii", ".mgz")
        skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk"]

        files = []
        for ext in mri_extensions:
            files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))

        seen = set()
        valid_mris = []
        for f in files:
            norm = os.path.normpath(f)
            if norm in seen:
                continue
            seen.add(norm)

            fname = os.path.basename(norm).lower()
            if any(keyword in fname for keyword in skip_keywords):
                continue

            valid_mris.append(norm)

        return valid_mris

    @staticmethod
    def check_existing_stages(output_dir):
        """
        Robustly checks which pipeline stages already have generated results in output_dir.
        Returns: (has_fastsurfer: bool, has_icp: bool, has_spharm: bool)
        """
        if not output_dir or not os.path.isdir(output_dir):
            return False, False, False

        # 1. Check FastSurfer results
        fs_dir = os.path.join(output_dir, "fastsurfer")
        fs_files = []
        if os.path.isdir(fs_dir):
            for ext in ("*.nii*", "*.vtk"):
                fs_files.extend(glob.glob(os.path.join(fs_dir, ext)))
                fs_files.extend(glob.glob(os.path.join(fs_dir, "**", ext), recursive=True))
        has_fastsurfer = len(fs_files) > 0

        # 2. Check ICP results
        icp_dir = os.path.join(output_dir, "output_ICP")
        icp_files = []
        if os.path.isdir(icp_dir):
            for ext in ("*.vtk", "*.ply", "*.nii*"):
                icp_files.extend(glob.glob(os.path.join(icp_dir, ext)))
                icp_files.extend(glob.glob(os.path.join(icp_dir, "**", ext), recursive=True))
            # Filter out reference templates and mean shapes
            icp_files = [f for f in icp_files if "mean_shape" not in os.path.basename(f).lower() and not os.path.basename(f).lower().startswith("template_")]
        has_icp = len(icp_files) > 0

        # 3. Check SPHARM results
        spharm_dir = os.path.join(output_dir, "output_SPHARM")
        sph_files = []
        if os.path.isdir(spharm_dir):
            for ext in ("*SPHARM*.vtk", "*.vtk"):
                sph_files.extend(glob.glob(os.path.join(spharm_dir, ext)))
                sph_files.extend(glob.glob(os.path.join(spharm_dir, "**", ext), recursive=True))
            sph_files = [
                f for f in sph_files 
                if "mean_shape" not in os.path.basename(f).lower() 
                and not os.path.basename(f).lower().startswith("template_") 
                and not any(aux in os.path.basename(f).lower() for aux in ("_para.", "_surf.", "medialaxis", "_grid."))
            ]
        has_spharm = len(sph_files) > 0

        return has_fastsurfer, has_icp, has_spharm

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 1. Directory & File Selection Properties
        config_group = QGroupBox("Pipeline Input & Output Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                margin-top: 15px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
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
        cg_layout = QVBoxLayout(config_group)
        cg_layout.setContentsMargins(10, 20, 10, 10)
        cg_layout.setSpacing(8)

        # Row 1: Choose Data Directory (MRI only) & Open File History
        data_btn_row = QHBoxLayout()
        data_btn_row.setSpacing(6)

        dir_select_btn = QPushButton("📁 Choose MRI Data Directory")
        dir_select_btn.setToolTip("Select directory containing raw MRI scans (*.nii.gz, *.nii, *.mgz)")
        dir_select_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
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
        dir_select_btn.clicked.connect(self.select_directory)
        data_btn_row.addWidget(dir_select_btn, stretch=3)

        self.history_btn = QPushButton("🕒 Open File History")
        self.history_btn.setToolTip("View and select from previously chosen directories")
        self.history_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 8px;
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
        self.history_btn.clicked.connect(self.show_history_menu)
        data_btn_row.addWidget(self.history_btn, stretch=2)

        cg_layout.addLayout(data_btn_row)

        # Row 2: Choose Output Directory
        out_dir_btn = QPushButton("📂 Choose Output Directory")
        out_dir_btn.setToolTip("Select base output directory. Subfolders (fastsurfer, output_ICP, output_SPHARM) will be organized automatically.")
        out_dir_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
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
        out_dir_btn.clicked.connect(self.select_out_directory)
        cg_layout.addWidget(out_dir_btn)

        # Row 3: Display Input & Output Paths
        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        input_col = QVBoxLayout()
        input_col.setSpacing(2)
        input_lbl = QLabel("MRI Input:")
        input_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("Select folder with MRI scans (.nii.gz, .nii, .mgz)...")
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        if getattr(self, 'last_input_dir', None) and os.path.isdir(self.last_input_dir):
            self.folder_input.setText(self.last_input_dir)
        input_col.addWidget(input_lbl)
        input_col.addWidget(self.folder_input)
        path_row.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(2)
        output_lbl = QLabel("Output Folder:")
        output_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.out_folder_input = QLineEdit()
        self.out_folder_input.setReadOnly(True)
        self.out_folder_input.setPlaceholderText("Select base output directory...")
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir):
            self.out_folder_input.setText(self.last_output_dir)
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        cg_layout.addLayout(path_row)
        main_layout.addWidget(config_group)

        # 2. Main Panel Pipeline Results & Subjects Table
        table_group = QGroupBox("Pipeline Subjects & Output Status")
        table_group.setStyleSheet("""
            QGroupBox {
                margin-top: 15px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
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
        tg_layout = QVBoxLayout(table_group)
        tg_layout.setContentsMargins(10, 20, 10, 10)
        tg_layout.setSpacing(6)

        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["Subject name"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 4px;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 4px;
                border: 1px solid #dcdde1;
            }
        """)
        tg_layout.addWidget(self.table)
        main_layout.addWidget(table_group)

        # 3. Pipeline Stages & Execution Controls
        workflow_group = QGroupBox("Pipeline Workflow Execution")
        workflow_group.setStyleSheet("""
            QGroupBox {
                margin-top: 15px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
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
        wg_layout = QVBoxLayout(workflow_group)
        wg_layout.setContentsMargins(10, 18, 10, 10)
        wg_layout.setSpacing(8)

        # Progress Stages Display
        stages_layout = QVBoxLayout()
        stages_layout.setSpacing(4)

        self.stage1_lbl = QLabel("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏸ Pending")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage1_lbl)

        self.stage2_lbl = QLabel("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage2_lbl)

        self.stage3_lbl = QLabel("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage3_lbl)

        wg_layout.addLayout(stages_layout)

        # Status notification label
        self.status_lbl = QLabel("Ready to run full pipeline.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("""
            color: #2c3e50;
            background-color: #eaf2f8;
            border: 1px solid #d4e6f1;
            font-size: 11px;
            padding: 6px 8px;
            border-radius: 4px;
            font-weight: 500;
        """)
        wg_layout.addWidget(self.status_lbl)

        # Run Button
        self.run_btn = QPushButton("▶ Run Full Pipeline (FastSurfer → ICP → SPHARM)")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #219a52);
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 10px 15px;
                border: 1px solid #1e8449;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60);
                border: 1px solid #196f3d;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e8449, stop:1 #145a32);
                border: 1px solid #145a32;
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
                border: 1px solid #95a5a6;
            }
        """)
        self.run_btn.clicked.connect(self.run_full_pipeline)
        wg_layout.addWidget(self.run_btn)

        main_layout.addWidget(workflow_group)

        # Initial check for stage status preview & populate table
        self.update_stage_preview()
        self.populate_main_table()

    def connect_pipeline_signals(self):
        if self.fastsurfer_panel:
            self.fastsurfer_panel.signal_fastsurfer_finished.connect(self.on_fastsurfer_finished_step)
        if self.icp_panel:
            self.icp_panel.signal_icp_finished.connect(self.on_icp_finished_step)
        if self.spharm_panel:
            self.spharm_panel.signal_spharm_finished.connect(self.on_spharm_finished_step)

    def set_directories(self, in_dir, out_dir):
        if in_dir:
            self.folder_input.setText(in_dir)
            self.last_input_dir = in_dir
        if out_dir:
            self.out_folder_input.setText(out_dir)
            self.last_output_dir = out_dir
            self.sync_panel_output_folders(out_dir)
        self.update_stage_preview()
        self.populate_main_table(in_dir, out_dir)

    def sync_panel_output_folders(self, out_dir):
        """Pre-configures individual subfolders matching each panel's standard structure and refreshes tables."""
        if not out_dir:
            return
        fs_dir = os.path.join(out_dir, "fastsurfer")
        icp_dir = os.path.join(out_dir, "output_ICP")
        spharm_dir = os.path.join(out_dir, "output_SPHARM")

        if self.fastsurfer_panel and hasattr(self.fastsurfer_panel, 'fs_dir_input'):
            self.fastsurfer_panel.fs_dir_input.setText(fs_dir)
            if os.path.isdir(fs_dir):
                self.fastsurfer_panel.populate_results_table()
        if self.icp_panel and hasattr(self.icp_panel, 'icp_dir_input'):
            self.icp_panel.icp_dir_input.setText(icp_dir)
            if os.path.isdir(icp_dir):
                self.icp_panel.populate_results_table()
        if self.spharm_panel and hasattr(self.spharm_panel, 'spharm_dir_input'):
            self.spharm_panel.spharm_dir_input.setText(spharm_dir)
            if os.path.isdir(spharm_dir):
                self.spharm_panel.populate_results_table()

    def update_stage_preview(self):
        """Checks output directory and previews which stages exist and which will be run."""
        out_dir = self.out_folder_input.text().strip()
        if not out_dir or not os.path.isdir(out_dir):
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏸ Pending")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.status_lbl.setText("Ready: Select Input MRI Directory & Output Directory to begin.")
            self.status_lbl.setStyleSheet("color: #2c3e50; background-color: #eaf2f8; border: 1px solid #d4e6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            return

        has_fs, has_icp, has_spharm = self.check_existing_stages(out_dir)

        if has_fs and has_icp and has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Results exist (Will skip)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Results exist (Will skip)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Results exist (Will skip)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.status_lbl.setText("✓ Complete results exist in output folder. Running will refresh all tables without re-computing.")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and not has_icp and not has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Results exist (Will skip)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Missing (Will run next)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Missing (Will run)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.status_lbl.setText("FastSurfer results exist (Skipping FastSurfer). Running will execute Groupwise ICP → SPHARM in sequence.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and has_icp and not has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Results exist (Will skip)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Results exist (Will skip)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Missing (Will run next)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            self.status_lbl.setText("FastSurfer & ICP results exist (Skipping both). Running will execute SPHARM-PDM.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and not has_icp and has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Results exist (Will skip)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Missing (Will run next)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Results exist (Will skip)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.status_lbl.setText("FastSurfer & SPHARM results exist. Running will execute Groupwise ICP.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        else:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏸ Missing (Will run first)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.status_lbl.setText("Ready to run full pipeline (FastSurfer → ICP → SPHARM).")
            self.status_lbl.setStyleSheet("color: #2c3e50; background-color: #eaf2f8; border: 1px solid #d4e6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")

    def populate_main_table(self, in_dir=None, out_dir=None):
        """Keeps Main Panel table blank as requested (results are displayed in respective module tables)."""
        self.table.setRowCount(0)

    def select_directory(self):
        initial_dir = self.last_input_dir if getattr(self, 'last_input_dir', None) and os.path.isdir(self.last_input_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        parent_win = self.window() if self.window() else self
        folder = QFileDialog.getExistingDirectory(parent_win, "Select Directory with MRI Scans", initial_dir)
        if folder:
            mri_files = self.find_valid_mri_files(folder)
            if not mri_files:
                self.signal_log_message.emit(f"[WARNING] No MRI scan files (*.nii.gz, *.nii, *.mgz) found in: {folder}")
                self.status_lbl.setText("⚠️ Warning: No valid MRI scan files (*.nii.gz, *.nii, *.mgz) found in selected directory.")
                self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
            else:
                self.status_lbl.setText(f"✓ Detected {len(mri_files)} MRI image(s). Ready to process.")
                self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px;")

            self.folder_input.setText(folder)
            self.add_to_history(folder)
            self.signal_log_message.emit(f"Main Panel selected MRI directory: {folder} (Found {len(mri_files)} MRI files)")

            if self.import_panel:
                self.import_panel.folder_input.setText(folder)
                self.import_panel.add_to_history(folder)
                self.import_panel.load_subjects_from_directory(folder)
                self.import_panel.signal_directories_changed.emit(folder, self.out_folder_input.text().strip())

            self.populate_main_table(folder, self.out_folder_input.text().strip())

    def select_out_directory(self):
        initial_dir = self.last_output_dir if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        parent_win = self.window() if self.window() else self
        folder = QFileDialog.getExistingDirectory(parent_win, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.last_output_dir = folder
            self.save_history()
            self.sync_panel_output_folders(folder)
            self.update_stage_preview()
            self.populate_main_table(self.folder_input.text().strip(), folder)
            self.signal_log_message.emit(f"Main Panel selected output directory: {folder}")

            if self.import_panel:
                self.import_panel.out_folder_input.setText(folder)
                self.import_panel.last_output_dir = folder
                self.import_panel.save_history()
                self.import_panel.signal_directories_changed.emit(self.folder_input.text().strip(), folder)

    def show_history_menu(self):
        if not self.recent_dirs:
            menu = QMenu(self)
            empty_action = menu.addAction("No recent directories")
            empty_action.setEnabled(False)
            menu.exec(self.history_btn.mapToGlobal(QPoint(0, self.history_btn.height())))
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #ffffff; border: 1px solid #ced6e0; border-radius: 4px; padding: 4px; }
            QMenu::item { padding: 6px 20px; font-size: 11px; color: #2c3e50; border-radius: 2px; }
            QMenu::item:selected { background-color: #3498db; color: #ffffff; }
        """)

        for d in self.recent_dirs:
            action = menu.addAction(d)
            action.triggered.connect(lambda checked, path=d: self.on_history_item_selected(path))

        menu.exec(self.history_btn.mapToGlobal(QPoint(0, self.history_btn.height())))

    def on_history_item_selected(self, path):
        if os.path.isdir(path):
            mri_files = self.find_valid_mri_files(path)
            self.folder_input.setText(path)
            self.add_to_history(path)
            self.signal_log_message.emit(f"Main Panel selected directory from history: {path} (Found {len(mri_files)} MRI files)")
            if not mri_files:
                self.status_lbl.setText("⚠️ Warning: No valid MRI scan files (*.nii.gz, *.nii, *.mgz) found.")
                self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
            else:
                self.status_lbl.setText(f"✓ Detected {len(mri_files)} MRI image(s). Ready to process.")
                self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px;")

            if self.import_panel:
                self.import_panel.folder_input.setText(path)
                self.import_panel.add_to_history(path)
                self.import_panel.load_subjects_from_directory(path)
                self.import_panel.signal_directories_changed.emit(path, self.out_folder_input.text().strip())

            self.populate_main_table(path, self.out_folder_input.text().strip())

    def run_full_pipeline(self):
        in_dir = self.folder_input.text().strip()
        out_dir = self.out_folder_input.text().strip()

        if not out_dir:
            self.signal_log_message.emit("[ERROR] Main Panel: Please select an Output Directory first.")
            self.status_lbl.setText("❌ Error: Output directory not specified.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            return

        # Create structured output folders exactly as each panel expects
        fs_dir = os.path.join(out_dir, "fastsurfer")
        icp_dir = os.path.join(out_dir, "output_ICP")
        spharm_dir = os.path.join(out_dir, "output_SPHARM")

        os.makedirs(fs_dir, exist_ok=True)
        os.makedirs(icp_dir, exist_ok=True)
        os.makedirs(spharm_dir, exist_ok=True)

        self.sync_panel_output_folders(out_dir)

        # Sync directories with ImportPanel
        if self.import_panel:
            if in_dir and os.path.isdir(in_dir):
                self.import_panel.folder_input.setText(in_dir)
                self.import_panel.load_subjects_from_directory(in_dir)
            self.import_panel.out_folder_input.setText(out_dir)
            self.import_panel.signal_directories_changed.emit(in_dir, out_dir)

        # Check existing results in output dir
        has_fs, has_icp, has_spharm = self.check_existing_stages(out_dir)

        # -------------------------------------------------------------
        # Condition A: Results exist for ALL panels!
        # "ถ้ามีผลลัพของ panel ครบทุก panel แล้วให้ข้ามได้เลยไม่ต้องทำ และให้แสดงผลลัพที่ตารางด้วย"
        # -------------------------------------------------------------
        if has_fs and has_icp and has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

            self.status_lbl.setText("🎉 All results already exist in output folder! Skipped execution and refreshed all tables.")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] Complete results for all panels (FastSurfer, ICP, SPHARM) already exist in output folder. Skipping execution and populating all tables.")

            # Populate all result tables across all panels immediately
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            self.populate_main_table(in_dir, out_dir)
            return

        # If FastSurfer is missing, raw MRI scans are strictly required
        if not has_fs:
            if not in_dir or not os.path.isdir(in_dir):
                self.signal_log_message.emit("[ERROR] Main Panel: FastSurfer results missing. Please select a valid MRI Data Directory first.")
                self.status_lbl.setText("❌ Error: FastSurfer results missing. Please select MRI data folder.")
                self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
                return

            mri_files = self.find_valid_mri_files(in_dir)
            if not mri_files:
                self.signal_log_message.emit(f"[ERROR] Main Panel: Selected directory '{in_dir}' contains no valid MRI files (*.nii.gz, *.nii, *.mgz).")
                self.status_lbl.setText("❌ Error: Selected folder contains no MRI scans. Please choose MRI data.")
                self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
                return

        self.is_running = True
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Running Pipeline...")

        # -------------------------------------------------------------
        # Condition B: FastSurfer exists, but ICP is missing (or ICP & SPHARM missing)
        # "ถ้ามีทุกอันยกเว้น icp spharm ให้ให้ทำ 2 อันนี้ต่อกัน"
        # -------------------------------------------------------------
        if has_fs and not has_icp:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()

            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏳ In Progress...")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            if has_spharm:
                self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Will skip)")
                self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            else:
                self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
                self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

            self.status_lbl.setText("🚀 FastSurfer results exist (Skipped). Running Groupwise ICP Registration...")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] Existing FastSurfer results detected (Skipping FastSurfer). Starting Groupwise ICP Registration...")

            self.populate_main_table(in_dir, out_dir)

            if self.icp_panel:
                self.icp_panel.update_run_button_state()
                self.icp_panel.run_icp_process()
            else:
                self.reset_run_state()
            return

        # -------------------------------------------------------------
        # Condition C: FastSurfer & ICP exist, but SPHARM is missing
        # -------------------------------------------------------------
        if has_fs and has_icp and not has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()
            if self.icp_panel:
                self.icp_panel.populate_results_table()

            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏳ In Progress...")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")

            self.status_lbl.setText("🚀 FastSurfer & ICP results exist (Skipped). Running SPHARM-PDM Processing...")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] Existing FastSurfer and ICP results detected (Skipped). Starting SPHARM-PDM Processing...")

            self.populate_main_table(in_dir, out_dir)

            if self.spharm_panel:
                self.spharm_panel.update_run_button_state()
                self.spharm_panel.run_spharm_process()
            else:
                self.reset_run_state()
            return

        # -------------------------------------------------------------
        # Condition D: FastSurfer missing -> Run from Stage 1: FastSurfer
        # "ไม่มีอันไหนทำอันนั้น"
        # -------------------------------------------------------------
        self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏳ In Progress...")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        mri_count = len(self.find_valid_mri_files(in_dir)) if in_dir else 0
        self.status_lbl.setText(f"🚀 Step 1/3: Running FastSurfer Segmentation on {mri_count} MRI scans...")
        self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        self.signal_log_message.emit(f">>> [MAIN PIPELINE] Starting FastSurfer Segmentation on {mri_count} MRI scans...")

        self.populate_main_table(in_dir, out_dir)

        if self.fastsurfer_panel:
            self.fastsurfer_panel.run_fastsurfer_process()
        else:
            self.signal_log_message.emit("[ERROR] FastSurfer panel not configured.")
            self.reset_run_state()

    def on_fastsurfer_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ❌ Failed")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #c0392b; font-weight: bold;")
            self.status_lbl.setText("❌ Pipeline stopped: FastSurfer Segmentation encountered an error.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            self.reset_run_state()
            return

        self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Completed")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        # Populate FastSurfer results table
        if self.fastsurfer_panel:
            self.fastsurfer_panel.populate_results_table()
        self.populate_main_table()

        # Check if downstream stages already exist
        out_dir = self.out_folder_input.text().strip()
        _, has_icp, has_spharm = self.check_existing_stages(out_dir)

        if has_icp and has_spharm:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            self.on_spharm_finished_step(True)
            return
        elif has_icp and not has_spharm:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏳ In Progress...")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            self.status_lbl.setText("🚀 Step 3/3: Running SPHARM-PDM Processing (ICP skipped, results exist)...")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] ICP results already exist. Starting SPHARM-PDM Processing...")
            if self.spharm_panel:
                self.spharm_panel.update_run_button_state()
                self.spharm_panel.run_spharm_process()
            else:
                self.reset_run_state()
            return

        # Step 2: ICP Registration
        self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏳ In Progress...")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.status_lbl.setText("🚀 Step 2/3: Running Groupwise ICP Registration...")
        self.signal_log_message.emit(">>> [MAIN PIPELINE] FastSurfer step completed. Starting Groupwise ICP Registration...")

        if self.icp_panel:
            self.icp_panel.update_run_button_state()
            self.icp_panel.run_icp_process()
        else:
            self.signal_log_message.emit("[ERROR] ICP panel not configured.")
            self.reset_run_state()

    def on_icp_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ❌ Failed")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #c0392b; font-weight: bold;")
            self.status_lbl.setText("❌ Pipeline stopped: Groupwise ICP Registration encountered an error.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            self.reset_run_state()
            return

        self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Completed")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        # Populate ICP results table
        if self.icp_panel:
            self.icp_panel.populate_results_table()
        self.populate_main_table()

        # Check if SPHARM already exists
        out_dir = self.out_folder_input.text().strip()
        _, _, has_spharm = self.check_existing_stages(out_dir)

        if has_spharm:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            self.on_spharm_finished_step(True)
            return

        # Step 3: SPHARM Processing
        self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏳ In Progress...")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.status_lbl.setText("🚀 Step 3/3: Running SPHARM-PDM Processing...")
        self.signal_log_message.emit(">>> [MAIN PIPELINE] ICP step completed. Starting SPHARM-PDM Processing...")

        if self.spharm_panel:
            self.spharm_panel.update_run_button_state()
            self.spharm_panel.run_spharm_process()
        else:
            self.signal_log_message.emit("[ERROR] SPHARM panel not configured.")
            self.reset_run_state()

    def on_spharm_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⚠️ Finished with warnings")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #d35400; font-weight: bold;")
            self.status_lbl.setText("⚠️ SPHARM-PDM finished with warnings or errors.")
            self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
        else:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Completed")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.status_lbl.setText("🎉 Complete Pipeline Finished Successfully!")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] All pipeline stages (FastSurfer, ICP, SPHARM) completed successfully!")

        # Populate all results tables across all panels so clicking to view them immediately displays all outputs
        in_dir = self.folder_input.text().strip()
        if self.import_panel and in_dir:
            self.import_panel.load_subjects_from_directory(in_dir)
        if self.fastsurfer_panel:
            self.fastsurfer_panel.populate_results_table()
        if self.icp_panel:
            self.icp_panel.populate_results_table()
        if self.spharm_panel:
            self.spharm_panel.populate_results_table()

        self.populate_main_table()
        self.reset_run_state()

    def reset_run_state(self):
        self.is_running = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ Run Full Pipeline (FastSurfer → ICP → SPHARM)")
