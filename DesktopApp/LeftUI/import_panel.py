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
        import_group.setStyleSheet("""
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
        ig_layout = QVBoxLayout(import_group)
        ig_layout.setContentsMargins(10, 20, 10, 10)
        ig_layout.setSpacing(8)

        # Row 1: Choose Data Directory and Choose Output Directory side by side
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        dir_select_btn = QPushButton("Choose Data Directory")
        dir_select_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4fa3e3, stop:1 #2980b9);
                color: white;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
                border: 1px solid #1f618d;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #64b5f6, stop:1 #3498db);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1f618d, stop:1 #1a5276);
            }
        """)
        dir_select_btn.clicked.connect(self.select_directory)
        btn_row.addWidget(dir_select_btn)

        out_dir_btn = QPushButton("Choose Output Directory")
        out_dir_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f39c12, stop:1 #d35400);
                color: white;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
                border: 1px solid #b94a00;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f5b041, stop:1 #e67e22);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #b94a00, stop:1 #933c00);
            }
        """)
        out_dir_btn.clicked.connect(self.select_out_directory)
        btn_row.addWidget(out_dir_btn)

        ig_layout.addLayout(btn_row)

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
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
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
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        ig_layout.addLayout(path_row)

        # Row 3: Standalone Import Data Button
        import_action_btn = QPushButton("Import Data")
        import_action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #219653);
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 10px;
                border: 1px solid #1e7e34;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4cd97b, stop:1 #27ae60);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e7e34, stop:1 #145a24);
            }
        """)
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
        self.remove_selected_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff7961, stop:1 #e74c3c);
                color: white;
                font-weight: bold;
                padding: 5px 10px;
                border: 1px solid #c0392b;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8a80, stop:1 #ff5252);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c0392b, stop:1 #96281b);
            }
        """)
        self.remove_selected_btn.clicked.connect(self.remove_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.remove_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subj_layout.addLayout(display_layout)
        
        import_layout.addWidget(subj_group)

    def get_folder(self):
        return self.folder_input.text()

    def get_output_folder(self):
        return self.out_folder_input.text()

    def select_directory(self):
        initial_dir = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(self, "Select Directory with NIFTI files", initial_dir)
        if folder:
            self.folder_input.setText(folder)
            self.signal_log_message.emit(f"Selected input directory: {folder}")

    def select_out_directory(self):
        initial_dir = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.signal_log_message.emit(f"Selected output directory: {folder}")

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
