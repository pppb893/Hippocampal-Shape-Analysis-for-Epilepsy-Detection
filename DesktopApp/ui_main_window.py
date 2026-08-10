import sys
import webbrowser
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter, QLabel, QComboBox, QMessageBox
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from ui_left_panel import LeftPanel
from ui_right_panel import RightPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hippocampal Shape Analysis Pipeline (Slicer-style)")
        self.resize(1200, 800)

        self.create_menus_and_toolbar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.left_panel = LeftPanel(self)
        self.right_panel = RightPanel(self)

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([350, 850])

        # Connect signals between panels
        self.left_panel.signal_subject_selected.connect(self.right_panel.display_subject)
        self.right_panel.signal_log_message.connect(self.left_panel.log)

        self.left_panel.log("SlicerSALT-style UI initialized successfully.")
        
        self.check_slicer_salt()

    def create_menus_and_toolbar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        save_action = QAction("💾 Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Modules: "))
        self.module_combo = QComboBox()
        self.module_combo.addItems(["Data Importer", "ICP Registration", "SPHARM Processing", "PLS-DA Analysis", "Feature Extraction"])
        self.module_combo.setMinimumWidth(200)
        toolbar.addWidget(self.module_combo)

    def check_slicer_salt(self):
        import glob
        slicer_paths = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
        
        if not slicer_paths:
            QMessageBox.critical(self, "SlicerSALT Required", 
                                "SlicerSALT could not be found in the default installation directory (C:\\Program Files\\SlicerSALT*).\n\n"
                                "This program requires SlicerSALT to function. The application will now close and open the download page.")
            webbrowser.open("https://salt.slicer.org/")
            sys.exit(1)
        else:
            self.left_panel.log(f"SUCCESS: SlicerSALT detected at {slicer_paths[0]}")
