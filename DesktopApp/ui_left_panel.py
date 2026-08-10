import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QFileDialog, QToolBox, 
                             QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal

class LeftPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        left_layout = QVBoxLayout(self)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # --- Logo / Header ---
        logo_label = QLabel("Shape Analysis Toolbox\nHippocampal Pipeline")
        font = logo_label.font()
        font.setPointSize(14)
        font.setBold(True)
        logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        left_layout.addWidget(logo_label)

        # --- QToolBox (Collapsible Panels) ---
        self.toolbox = QToolBox()
        left_layout.addWidget(self.toolbox)

        self.setup_help_panel()
        self.setup_import_panel()
        self.setup_subjects_panel()
        self.setup_analysis_panel()

    def setup_help_panel(self):
        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_label = QLabel("This project is for Hippocampal Shape Analysis.\nUse the tools below to run the ICP -> SPHARM -> PLS-DA pipeline.")
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        help_layout.addStretch()
        self.toolbox.addItem(help_widget, "Help & Acknowledgement")

    def setup_import_panel(self):
        import_widget = QWidget()
        import_layout = QVBoxLayout(import_widget)
        
        btn_layout = QHBoxLayout()
        btn_import_dir = QPushButton("Import from directory")
        btn_import_csv = QPushButton("Import from CSV")
        btn_layout.addWidget(btn_import_dir)
        btn_layout.addWidget(btn_import_csv)
        import_layout.addLayout(btn_layout)

        dir_select_btn = QPushButton("📁 Choose Data Directory")
        dir_select_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 5px;")
        dir_select_btn.clicked.connect(self.select_directory)
        import_layout.addWidget(dir_select_btn)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        folder_layout.addWidget(self.folder_input)
        import_layout.addLayout(folder_layout)
        
        import_action_btn = QPushButton("Import")
        import_action_btn.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")
        import_layout.addWidget(import_action_btn)
        
        import_layout.addStretch()
        self.toolbox.addItem(import_widget, "Import Data Properties")

    def setup_subjects_panel(self):
        subjects_widget = QWidget()
        subjects_layout = QVBoxLayout(subjects_widget)
        self.subjects_table = QTableWidget(0, 2)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name", "Consistency"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.subjects_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.subjects_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.subjects_table.itemSelectionChanged.connect(self.on_subject_selection_changed)
        subjects_layout.addWidget(self.subjects_table)
        
        display_layout = QHBoxLayout()
        self.display_selected_btn = QPushButton("Display Selected")
        self.display_selected_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.display_selected_btn.clicked.connect(self.display_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.display_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subjects_layout.addLayout(display_layout)
        
        self.add_mock_subjects()
        self.toolbox.addItem(subjects_widget, "Imported Subjects")

    def setup_analysis_panel(self):
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        self.run_icp_btn = QPushButton("Run ICP Registration")
        self.run_icp_btn.clicked.connect(lambda: self.log(">>> Running ICP Registration..."))
        self.run_spharm_btn = QPushButton("Run SPHARM Processing")
        self.run_spharm_btn.clicked.connect(lambda: self.log(">>> Running SPHARM Processing..."))
        self.run_plsda_btn = QPushButton("Run PLS-DA Analysis")
        self.run_plsda_btn.clicked.connect(lambda: self.log(">>> Running PLS-DA Analysis..."))
        
        analysis_layout.addWidget(self.run_icp_btn)
        analysis_layout.addWidget(self.run_spharm_btn)
        analysis_layout.addWidget(self.run_plsda_btn)
        
        analysis_layout.addWidget(QLabel("Console Output:"))
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        analysis_layout.addWidget(self.log_window)
        self.toolbox.addItem(analysis_widget, "Analysis & Logging")

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.folder_input.setText(directory)
            self.log(f"Selected directory: {directory}")

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

    def add_mock_subjects(self):
        mock_data = [
            ("001_tp3CranialReg.nrrd", "OK"),
            ("001_tp1CranialReg.nrrd", "# Inconsistencies: 1"),
            ("001_tp2CranialReg.nrrd", "# Inconsistencies: 1")
        ]
        self.subjects_table.setRowCount(len(mock_data))
        for row, (name, cons) in enumerate(mock_data):
            self.subjects_table.setItem(row, 0, QTableWidgetItem(name))
            self.subjects_table.setItem(row, 1, QTableWidgetItem(cons))

    def log(self, message):
        self.log_window.append(message)
        self.signal_log_message.emit(message)
