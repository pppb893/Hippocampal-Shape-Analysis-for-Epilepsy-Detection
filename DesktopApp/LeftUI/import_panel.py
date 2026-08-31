import os
import glob
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QGroupBox)
from PyQt6.QtCore import pyqtSignal, Qt

class ImportPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        import_layout = QVBoxLayout(self)
        
        # 1. Import Data Properties
        import_group = QGroupBox("Import Data Properties")
        import_group.setStyleSheet("QGroupBox { margin-top: 15px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        ig_layout = QVBoxLayout(import_group)
        ig_layout.setContentsMargins(10, 20, 10, 10)
        ig_layout.setSpacing(10)


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
        self.subjects_table = QTableWidget(0, 1)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.subjects_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.subjects_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.subjects_table.itemSelectionChanged.connect(self.on_subject_selection_changed)
        subj_layout.addWidget(self.subjects_table)
        
        display_layout = QHBoxLayout()
        
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.remove_selected_btn.clicked.connect(self.remove_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.remove_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subj_layout.addLayout(display_layout)
        
        import_layout.addWidget(subj_group)

    def get_folder(self):
        return self.folder_input.text()

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.folder_input.setText(directory)
            self.signal_log_message.emit(f"Selected directory: {directory}")

    def on_import_clicked(self):
        directory = self.folder_input.text()
        if directory and os.path.isdir(directory):
            self.load_subjects_from_directory(directory)
            self.signal_log_message.emit(">>> Data Imported Successfully.")
        else:
            self.signal_log_message.emit("[ERROR] Please select a valid directory first.")

    def load_subjects_from_directory(self, directory):
        search_patterns = ["*.nrrd", "*.nii.gz", "*.nii", "*.vtk"]
        files = []
        for pattern in search_patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
            
        if not files:
            self.signal_log_message.emit(f"No valid image/mesh files (*.nrrd, *.nii.gz, *.vtk) found in {directory}")
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

    def on_subject_selection_changed(self):
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
        self.signal_subject_selected.emit(filepath)

    def remove_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
            
        selected_rows = sorted(list(set([item.row() for item in selected_items])), reverse=True)
        
        for row in selected_rows:
            item = self.subjects_table.item(row, 0)
            subject_name = item.text()
            self.subjects_table.removeRow(row)
            self.signal_log_message.emit(f"Removed subject from list: {subject_name}")
