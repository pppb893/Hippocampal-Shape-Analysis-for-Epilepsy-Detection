from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt6.QtCore import Qt, pyqtSignal

from .import_panel import ImportPanel
from .fastsurfer_panel import FastsurferPanel
from .icp_panel import IcpPanel
from .spharm_panel import SpharmPanel
from .plsda_panel import PlsdaPanel
from .feature_panel import FeaturePanel

class AdaptiveStackedWidget(QStackedWidget):
    def sizeHint(self):
        cw = self.currentWidget()
        return cw.sizeHint() if cw else super().sizeHint()

    def minimumSizeHint(self):
        cw = self.currentWidget()
        return cw.minimumSizeHint() if cw else super().minimumSizeHint()

class LeftPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        left_layout = QVBoxLayout(self)
        left_layout.setContentsMargins(12, 10, 12, 10)
        left_layout.setSpacing(10)

        # --- Logo / Header ---
        logo_label = QLabel("Shape Analysis Toolbox\nHippocampal Pipeline")
        font = logo_label.font()
        font.setPointSize(13)
        font.setBold(True)
        logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #2c3e50; padding: 8px 10px; background-color: #ecf0f1; border-radius: 5px;")
        left_layout.addWidget(logo_label)

        # --- QStackedWidget (Module Switching) ---
        self.stacked_widget = AdaptiveStackedWidget()
        left_layout.addWidget(self.stacked_widget)

        # Create panels
        self.import_panel = ImportPanel()
        self.fastsurfer_panel = FastsurferPanel(self.import_panel.get_folder, self.import_panel.get_output_folder)
        self.icp_panel = IcpPanel(self.import_panel.get_folder)
        self.spharm_panel = SpharmPanel(self.import_panel.get_folder)
        self.plsda_panel = PlsdaPanel(self.import_panel.get_folder)
        self.feature_panel = FeaturePanel(self.import_panel.get_folder)

        # Add to stacked widget
        self.stacked_widget.addWidget(self.import_panel)
        self.stacked_widget.addWidget(self.fastsurfer_panel)
        self.stacked_widget.addWidget(self.icp_panel)
        self.stacked_widget.addWidget(self.spharm_panel)
        self.stacked_widget.addWidget(self.plsda_panel)
        self.stacked_widget.addWidget(self.feature_panel)

        # Connect signals
        for panel in [self.import_panel, self.fastsurfer_panel, self.icp_panel, self.spharm_panel, self.plsda_panel, self.feature_panel]:
            panel.signal_log_message.connect(self.signal_log_message)
            
        self.import_panel.signal_subject_selected.connect(self.signal_subject_selected)
        self.import_panel.signal_directories_changed.connect(lambda i, o: self.fastsurfer_panel.update_run_button_state())

    def switch_module(self, index):
        self.stacked_widget.setCurrentIndex(index)
        if index == 1:
            self.fastsurfer_panel.update_run_button_state()
        self.stacked_widget.updateGeometry()
        self.updateGeometry()
