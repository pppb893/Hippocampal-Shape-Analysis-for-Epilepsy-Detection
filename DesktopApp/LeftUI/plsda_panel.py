import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QCheckBox, QGroupBox, QFormLayout, QComboBox, QSpinBox, 
                             QLineEdit, QFileDialog)
from PyQt6.QtCore import pyqtSignal, QProcess, QLocale

class PlsdaPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, get_folder_func, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.setup_ui()

    def setup_ui(self):
        plsda_layout = QVBoxLayout(self)
        help_label = QLabel("This module runs Partial Least Squares Discriminant Analysis (PLS-DA) on SPHARM shape coefficients to differentiate Healthy Controls vs Epilepsy groups.")
        help_label.setWordWrap(True)
        plsda_layout.addWidget(help_label)
        
        plsda_group = QGroupBox("PLS-DA Parameters")
        plsda_group.setStyleSheet("""
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
            QSpinBox, QComboBox, QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                min-height: 22px;
                color: #34495e;
            }
            QSpinBox:disabled {
                background-color: #eef2f5;
                color: #7f8c8d;
                border: 1px solid #dcdde1;
            }
            QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                border: 1px solid #e67e22;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
        """)
        
        plsda_form = QFormLayout(plsda_group)
        plsda_form.setContentsMargins(15, 25, 15, 15)
        plsda_form.setVerticalSpacing(12)
        
        self.plsda_mode_combo = QComboBox()
        self.plsda_mode_combo.addItems(["Standard Mode (Default)", "Custom Mode"])
        self.plsda_mode_combo.setCurrentIndex(0)
        self.plsda_mode_combo.setToolTip("Preset modes. Standard Mode extracts 10 PLS-DA components.")
        self.plsda_mode_combo.currentIndexChanged.connect(self.on_plsda_mode_changed)

        self.plsda_comp_spin = QSpinBox()
        self.plsda_comp_spin.setLocale(QLocale.c())
        self.plsda_comp_spin.setRange(0, 50)
        self.plsda_comp_spin.setValue(10)
        self.plsda_comp_spin.setSingleStep(1)
        self.plsda_comp_spin.setEnabled(False)
        self.plsda_comp_spin.setToolTip("Number of PLS-DA components to extract. (Default: 10)")

        csv_layout = QHBoxLayout()
        self.plsda_csv_input = QLineEdit()
        self.plsda_csv_input.setPlaceholderText("Auto (Filenames: HEALTHY vs TLE)")
        self.plsda_csv_input.setToolTip("Optional CSV file containing Subject group labels.")
        csv_btn = QPushButton("📁 Browse")
        csv_btn.clicked.connect(self.select_plsda_csv)
        csv_layout.addWidget(self.plsda_csv_input)
        csv_layout.addWidget(csv_btn)

        self.plsda_export_scores_cb = QCheckBox("Export Score Table (.csv)")
        self.plsda_export_scores_cb.setChecked(True)
        self.plsda_export_scores_cb.setToolTip("Export computed PLS-DA subject component scores to plsda_scores.csv.")

        self.plsda_save_plot_cb = QCheckBox("Save Visualization Plot (.png)")
        self.plsda_save_plot_cb.setChecked(True)
        self.plsda_save_plot_cb.setToolTip("Automatically save high-resolution PLS-DA scatter plot image to plsda_visualization.png.")

        plsda_form.addRow("Preset Mode:", self.plsda_mode_combo)
        plsda_form.addRow("PLS Components:", self.plsda_comp_spin)
        plsda_form.addRow("Group Metadata CSV:", csv_layout)
        plsda_form.addRow("Auto Export:", self.plsda_export_scores_cb)
        plsda_form.addRow("", self.plsda_save_plot_cb)
        
        plsda_layout.addWidget(plsda_group)
        
        self.run_plsda_btn = QPushButton("Run PLS-DA Analysis")
        self.run_plsda_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22; 
                color: white; 
                padding: 10px 15px; 
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #f39c12;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        self.run_plsda_btn.clicked.connect(self.run_plsda_process)
        plsda_layout.addWidget(self.run_plsda_btn)
        
        plsda_layout.addStretch()

    def on_plsda_mode_changed(self, index):
        if index == 0:  # Standard Mode
            self.plsda_comp_spin.setValue(10)
            self.plsda_comp_spin.setEnabled(False)
        elif index == 1:  # Custom Mode
            if self.plsda_comp_spin.value() == 0:
                self.plsda_comp_spin.setValue(10)
            self.plsda_comp_spin.setEnabled(True)

    def select_plsda_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Group Metadata CSV", "", "CSV Files (*.csv)")
        if filename:
            self.plsda_csv_input.setText(filename)

    def run_plsda_process(self):
        selected_dir = self.get_folder()
        if not selected_dir or not os.path.isdir(selected_dir):
            self.signal_log_message.emit("[ERROR] Please select a valid input directory first.")
            return

        folder_name = os.path.basename(os.path.normpath(selected_dir))
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        output_candidate = os.path.join(script_dir, f"output_{folder_name}")
        if os.path.exists(output_candidate):
            target_dir = output_candidate
        else:
            target_dir = selected_dir

        n_comp = self.plsda_comp_spin.value()
        
        self.signal_log_message.emit(">>> Starting PLS-DA Analysis...")
        self.signal_log_message.emit(f"Target Directory: {target_dir}")
        self.signal_log_message.emit(f"Parameters: PLS Components={n_comp}")

        plsda_script_path = os.path.join(script_dir, 'Visualize', 'Data_Plots', 'visualize_plsda.py')
        if not os.path.exists(plsda_script_path):
            self.signal_log_message.emit(f"[ERROR] PLS-DA script not found at {plsda_script_path}")
            return

        self.plsda_process = QProcess(self)
        self.plsda_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.plsda_process.readyReadStandardOutput.connect(self.handle_plsda_stdout)
        self.plsda_process.finished.connect(self.plsda_process_finished)

        args = [
            plsda_script_path,
            "--output_dir", target_dir,
            "--n_components", str(n_comp)
        ]

        self.run_plsda_btn.setEnabled(False)
        self.plsda_process.start("python", args)

    def handle_plsda_stdout(self):
        data = self.plsda_process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        for line in stdout.splitlines():
            if line.strip():
                self.signal_log_message.emit(line)

    def plsda_process_finished(self, exit_code, exit_status):
        self.run_plsda_btn.setEnabled(True)
        self.signal_log_message.emit(f">>> PLS-DA Analysis finished with exit code {exit_code}")
