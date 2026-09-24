import os
import glob
import json
import re
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QGroupBox, QMenu, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint

class ToggleTableWidget(QTableWidget):
    """QTableWidget supporting ExtendedSelection (Ctrl/Shift multi-select)
    and single-click toggle/deselect when clicking an already selected sole row."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                row = item.row()
                modifiers = event.modifiers()
                has_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

                selected_rows = list(set(it.row() for it in self.selectedItems()))

                if not has_ctrl and not has_shift:
                    if selected_rows == [row]:
                        self.clearSelection()
                        return
                    elif row in selected_rows and len(selected_rows) > 1:
                        self.clearSelection()
                        self.selectRow(row)
                        return

        super().mousePressEvent(event)

class ImportPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str)
    signal_directories_changed = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_input_dir = None
        self.last_output_dir = None
        self.setup_ui()
        
    def setup_ui(self):
        import_layout = QVBoxLayout(self)
        
        # 1. Import Data Properties
        import_group = QGroupBox("Import Data Properties")
        import_group.setStyleSheet("""
            QGroupBox {
                margin-top: 10px;
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
        ig_layout = QVBoxLayout(import_group)
        ig_layout.setContentsMargins(10, 16, 10, 10)
        ig_layout.setSpacing(8)

        # Row 1: Choose Data Directory
        dir_select_btn = QPushButton("Choose Data Directory")
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
        ig_layout.addWidget(dir_select_btn)

        # Row 2: Choose Output Directory
        out_dir_btn = QPushButton("Choose Output Directory")
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
        ig_layout.addWidget(out_dir_btn)

        # Row 2: Input Path and Output Path side by side
        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        input_col = QVBoxLayout()
        input_col.setSpacing(2)
        input_lbl = QLabel("Input:")
        input_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("No input directory selected...")
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ced6e0; border-radius: 4px; padding: 5px 8px; font-size: 11px;")
        input_col.addWidget(input_lbl)
        input_col.addWidget(self.folder_input)
        path_row.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(2)
        output_lbl = QLabel("Output:")
        output_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.out_folder_input = QLineEdit()
        self.out_folder_input.setReadOnly(True)
        self.out_folder_input.setPlaceholderText("No output directory selected...")
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ced6e0; border-radius: 4px; padding: 5px 8px; font-size: 11px;")
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        ig_layout.addLayout(path_row)

        # Row 3: Standalone Import Data Button
        import_action_btn = QPushButton("Import Data")
        import_action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 10px;
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
        import_action_btn.clicked.connect(self.on_import_clicked)
        ig_layout.addWidget(import_action_btn)
        
        import_layout.addWidget(import_group)

        # 2. Imported Subjects
        subj_group = QGroupBox("Imported Subjects")
        subj_group.setStyleSheet("""
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
        subj_layout = QVBoxLayout(subj_group)
        subj_layout.setContentsMargins(10, 16, 10, 10)
        subj_layout.setSpacing(10)
        self.subjects_table = ToggleTableWidget(0, 1)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.subjects_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subjects_table.setStyleSheet("""
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
        self.subjects_table.itemSelectionChanged.connect(self.on_subject_selection_changed)
        subj_layout.addWidget(self.subjects_table)
        
        display_layout = QHBoxLayout()
        
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 10px;
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
        self.remove_selected_btn.clicked.connect(self.remove_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.remove_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subj_layout.addLayout(display_layout)
        
        import_layout.addWidget(subj_group)

    def add_to_history(self, path):
        pass

    def get_folder(self):
        return self.folder_input.text()

    def get_output_folder(self):
        return self.out_folder_input.text()

    def select_directory(self):
        initial_dir = self.last_input_dir if getattr(self, 'last_input_dir', None) and os.path.isdir(self.last_input_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Directory with NIFTI files", initial_dir)
        if folder:
            self.folder_input.setText(folder)
            self.last_input_dir = folder
            self.signal_log_message.emit(f"Selected input directory: {folder}")
            self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())

    def select_out_directory(self):
        initial_dir = self.last_output_dir if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.last_output_dir = folder
            self.load_subjects_from_output(folder)
            self.signal_log_message.emit(f"Selected output directory: {folder}")
            self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())

    def load_subjects_from_output(self, out_dir):
        if not out_dir or not os.path.isdir(out_dir):
            return

        # If user explicitly loaded raw files in folder_input, don't overwrite unless table empty
        if self.subjects_table.rowCount() > 0 and self.folder_input.text().strip():
            return

        fs_mri_dir = os.path.join(out_dir, "fastsurfer", "mri")
        os.makedirs(fs_mri_dir, exist_ok=True)

        # 1. Discover all subject IDs from output folders
        subjects = set()
        
        # From fastsurfer_temp subdirectories
        fs_temp = os.path.join(out_dir, "fastsurfer", "fastsurfer_temp")
        if os.path.isdir(fs_temp):
            for d in os.listdir(fs_temp):
                if os.path.isdir(os.path.join(fs_temp, d)):
                    subjects.add(d)

        # From fastsurfer left/right hippocampus
        for side_dir in ["left_hippocampus", "right_hippocampus"]:
            p = os.path.join(out_dir, "fastsurfer", side_dir)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    m = re.search(r'(sub-[a-zA-Z0-9]+)', f)
                    if m:
                        subjects.add(m.group(1))

        # From fastsurfer/mri
        if os.path.isdir(fs_mri_dir):
            for f in os.listdir(fs_mri_dir):
                m = re.search(r'(sub-[a-zA-Z0-9]+)', f)
                if m:
                    subjects.add(m.group(1))

        # From ICP, SPHARM, or Result directories
        for vtk_file in glob.glob(os.path.join(out_dir, "**", "*.vtk"), recursive=True):
            m = re.search(r'(sub-[a-zA-Z0-9]+)', os.path.basename(vtk_file))
            if m:
                subjects.add(m.group(1))

        # 2. For each subject, find or convert the paired MRI volume
        paired_items = []
        seen_files = set()

        for subj in sorted(subjects):
            mri_filepath = None
            
            # Check fastsurfer/mri/{subj}_t1.nii.gz or variations
            mri_candidates = [
                os.path.join(fs_mri_dir, f"{subj}_t1.nii.gz"),
                os.path.join(fs_mri_dir, f"{subj}.nii.gz"),
                os.path.join(fs_mri_dir, f"{subj}_T1w.nii.gz"),
                os.path.join(out_dir, "mri", f"{subj}_t1.nii.gz"),
            ]
            for cand in mri_candidates:
                if os.path.isfile(cand):
                    mri_filepath = cand
                    break

            # If not yet converted, search for orig.mgz in fastsurfer_temp
            if not mri_filepath:
                mgz_candidates = [
                    os.path.join(fs_temp, subj, "mri", "orig.mgz"),
                    os.path.join(out_dir, "fastsurfer", subj, "mri", "orig.mgz"),
                    os.path.join(out_dir, "fastsurfer_temp", subj, "mri", "orig.mgz"),
                ]
                for mgz in mgz_candidates:
                    if os.path.isfile(mgz):
                        # Convert orig.mgz to NIfTI on the fly for VTK visualization
                        dst_nii = os.path.join(fs_mri_dir, f"{subj}_t1.nii.gz")
                        if not os.path.exists(dst_nii):
                            try:
                                import nibabel as nib
                                import numpy as np
                                img = nib.load(mgz)
                                nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
                                nib.save(nii_img, dst_nii)
                            except Exception as e:
                                print(f"[ERROR] Could not convert {mgz}: {e}")
                        if os.path.exists(dst_nii):
                            mri_filepath = dst_nii
                            break

            # If MRI found, add to paired items
            if mri_filepath and os.path.isfile(mri_filepath):
                fname = os.path.basename(mri_filepath)
                if fname not in seen_files:
                    seen_files.add(fname)
                    paired_items.append((fname, mri_filepath))
            else:
                # Fallback to hippocampus mask or aligned mesh if MRI is missing
                fallback_candidates = [
                    os.path.join(out_dir, "fastsurfer", "left_hippocampus", f"lh_{subj}_hippocampus.nii.gz"),
                    os.path.join(out_dir, "output_ICP", "left", "aligned_meshes", f"lh_{subj}_hippocampus_aligned.vtk"),
                ]
                for fb in fallback_candidates:
                    if os.path.isfile(fb):
                        fname = os.path.basename(fb)
                        if fname not in seen_files:
                            seen_files.add(fname)
                            paired_items.append((fname, fb))
                        break

        # Also add any other MRI volumes found in fastsurfer/mri that weren't matched above
        if os.path.isdir(fs_mri_dir):
            for pat in ["*.nii.gz", "*.nii", "*.mgz"]:
                for extra in glob.glob(os.path.join(fs_mri_dir, pat)):
                    fname = os.path.basename(extra)
                    if fname not in seen_files:
                        seen_files.add(fname)
                        paired_items.append((fname, extra))

        if paired_items:
            self.subjects_table.setRowCount(0)
            for fname, fpath in paired_items:
                row = self.subjects_table.rowCount()
                self.subjects_table.setRowCount(row + 1)
                item = QTableWidgetItem(fname)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setData(Qt.ItemDataRole.UserRole, fpath)
                self.subjects_table.setItem(row, 0, item)

            self.signal_log_message.emit(f"Import Panel loaded {len(paired_items)} paired subject MRI volume(s) from output directory.")
            if self.subjects_table.rowCount() > 0:
                self.subjects_table.blockSignals(True)
                if not self.subjects_table.selectedItems():
                    self.subjects_table.selectRow(0)
                self.subjects_table.blockSignals(False)
                if self.isVisible():
                    self.display_selected_subject()

    # Alias for backward compatibility
    select_output_directory = select_out_directory

    def on_import_clicked(self):
        directory = self.folder_input.text()
        if directory and os.path.isdir(directory):
            self.load_subjects_from_directory(directory)
            self.signal_log_message.emit(">>> Data Imported Successfully.")
        else:
            self.signal_log_message.emit("[ERROR] Please select a valid directory first.")

    def load_subjects_from_directory(self, directory, clear_existing=False):
        if not directory or not os.path.isdir(directory):
            return

        if clear_existing or getattr(self, 'loaded_directory', None) != directory:
            self.subjects_table.setRowCount(0)
        self.loaded_directory = directory

        search_patterns = ["*.nrrd", "*.nii.gz", "*.nii", "*.vtk", "*.mgz"]
        files = []
        for pattern in search_patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
            files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
            
        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk"]
        for f in files:
            norm = os.path.normpath(f)
            if norm not in seen:
                seen.add(norm)
                fname = os.path.basename(norm).lower()
                if not any(kw in fname for kw in skip_keywords):
                    unique_files.append(norm)
        files = unique_files

        if not files:
            self.signal_log_message.emit(f"No valid MRI image files found in {directory}")
            return
            
        # Prevent duplicates
        existing_paths = set()
        for i in range(self.subjects_table.rowCount()):
            item = self.subjects_table.item(i, 0)
            if item:
                existing_paths.add(item.data(Qt.ItemDataRole.UserRole))
                
        new_files = [f for f in files if f not in existing_paths]
        
        if not new_files:
            self.signal_log_message.emit(f"All files in {directory} are already imported.")
            if self.subjects_table.rowCount() > 0:
                if not self.subjects_table.selectedItems():
                    self.subjects_table.selectRow(0)
                self.display_selected_subject()
            return
            
        self.signal_log_message.emit(f"Importing {len(new_files)} new files...")
        
        current_row_count = self.subjects_table.rowCount()
        self.subjects_table.setRowCount(current_row_count + len(new_files))
        
        for i, filepath in enumerate(new_files):
            row = current_row_count + i
            filename = os.path.basename(filepath)
            item = QTableWidgetItem(filename)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            
            self.subjects_table.setItem(row, 0, item)

        # Auto-select and display first subject
        if self.subjects_table.rowCount() > 0:
            self.subjects_table.blockSignals(True)
            if not self.subjects_table.selectedItems():
                self.subjects_table.selectRow(0)
            self.subjects_table.blockSignals(False)
            if self.isVisible():
                self.display_selected_subject()

    def on_subject_selection_changed(self):
        if not self.isVisible():
            return
        if self.display_on_click_cb.isChecked():
            self.display_selected_subject()

    def display_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
        
        selected_rows = list(set([item.row() for item in selected_items]))
        if len(selected_rows) > 1:
            return  # Do not display if multiple subjects are selected
            
        row = selected_rows[0]
        item = self.subjects_table.item(row, 0)
        subject_name = item.text()
        filepath = item.data(Qt.ItemDataRole.UserRole)
        
        self.signal_log_message.emit(f"Displaying subject: {subject_name}")
        if filepath and (filepath.lower().endswith(".vtk") or filepath.lower().endswith(".stl") or filepath.lower().endswith(".ply")):
            self.signal_mesh_selected.emit(filepath, "all")
        else:
            self.signal_subject_selected.emit(filepath)

    def remove_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
            
        selected_rows = sorted(list(set([item.row() for item in selected_items])), reverse=True)
        count = len(selected_rows)
        if count == 0:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove {count} selected subject(s) from the list?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply != QMessageBox.StandardButton.Ok:
            return
        
        for row in selected_rows:
            item = self.subjects_table.item(row, 0)
            subject_name = item.text() if item else f"Row {row}"
            self.subjects_table.removeRow(row)
            self.signal_log_message.emit(f"Removed subject from list: {subject_name}")
