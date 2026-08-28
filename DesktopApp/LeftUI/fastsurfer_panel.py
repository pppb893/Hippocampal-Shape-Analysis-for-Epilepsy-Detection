import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, QGroupBox, QFormLayout)
from PyQt6.QtCore import pyqtSignal

class FastsurferPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    
    def __init__(self, get_folder_func, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.setup_ui()
        
    def setup_ui(self):
        fs_layout = QVBoxLayout(self)
        
        help_label = QLabel("This module runs FastSurfer to segment the brain (including the hippocampus) from T1w MRI scans.")
        help_label.setWordWrap(True)
        fs_layout.addWidget(help_label)
        
        fs_group = QGroupBox("FastSurfer Parameters")
        fs_group.setStyleSheet("""
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
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
        """)
        
        fs_form = QFormLayout(fs_group)
        fs_form.setContentsMargins(15, 25, 15, 15)
        fs_form.setVerticalSpacing(12)
        
        self.fs_gpu_cb = QCheckBox("Use GPU Acceleration (if available)")
        self.fs_gpu_cb.setChecked(True)
        
        self.fs_seg_only_cb = QCheckBox("Run Segmentation Only (Skip surface generation)")
        self.fs_seg_only_cb.setChecked(False)
        
        fs_form.addRow("Compute Device:", self.fs_gpu_cb)
        fs_form.addRow("Pipeline Mode:", self.fs_seg_only_cb)
        
        fs_layout.addWidget(fs_group)
        
        self.run_fs_btn = QPushButton("Run FastSurfer")
        self.run_fs_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; 
                color: white; 
                padding: 10px 15px; 
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
        """)
        self.run_fs_btn.clicked.connect(self.run_fastsurfer_process)
        fs_layout.addWidget(self.run_fs_btn)
        
        fs_layout.addStretch()

    def run_fastsurfer_process(self):
        input_dir = self.get_folder()
        if not input_dir or not os.path.isdir(input_dir):
            self.signal_log_message.emit("[ERROR] Please select a valid input directory first.")
            return
            
        self.signal_log_message.emit(">>> Starting FastSurfer Segmentation...")
        self.signal_log_message.emit(f"Input Directory: {input_dir}")
        self.signal_log_message.emit(f"Use GPU: {self.fs_gpu_cb.isChecked()}")
        self.signal_log_message.emit(f"Segmentation Only: {self.fs_seg_only_cb.isChecked()}")
        self.signal_log_message.emit("[INFO] FastSurfer pipeline execution is a placeholder.")
        self.signal_log_message.emit(">>> FastSurfer Segmentation finished.")
