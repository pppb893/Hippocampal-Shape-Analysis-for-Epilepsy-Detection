from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

from .vtk_viewer import VtkViewer

class RightPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        right_layout = QVBoxLayout(self)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        self.viewer = VtkViewer(self)
        right_layout.addWidget(self.viewer)
        
        self.viewer.signal_log_message.connect(self.signal_log_message)

    def display_subject(self, subject_name):
        self.viewer.display_subject(subject_name)
