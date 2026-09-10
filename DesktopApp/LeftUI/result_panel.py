from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

class ResultPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, parent=None, get_input_folder=None, get_output_folder=None):
        super().__init__(parent)
        self.get_input_folder = get_input_folder
        self.get_output_folder = get_output_folder
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        # Blank page (Left UI) ready for future components
