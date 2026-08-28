import os
import glob
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QGroupBox)
from PyQt6.QtCore import pyqtSignal

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

    def get_folder(self):
        return self.folder_input.text()

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.folder_input.setText(directory)
            self.signal_log_message.emit(f"Selected directory: {directory}")
            self.load_subjects_from_directory(directory)

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
            
        self.subjects_table.setRowCount(0)
        
        if not files:
            self.signal_log_message.emit(f"No valid image/mesh files (*.nrrd, *.nii.gz, *.vtk) found in {directory}")
            return
            
        self.signal_log_message.emit(f"Found {len(files)} files in directory.")
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
        
        self.signal_log_message.emit(f"Displaying subject: {subject_name}")
        self.signal_subject_selected.emit(subject_name)
