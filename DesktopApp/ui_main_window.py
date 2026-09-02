import sys
import webbrowser
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QLabel, QComboBox, QMessageBox, QTextEdit
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from LeftUI.left_panel import LeftPanel
from RightUI.right_panel import RightPanel

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

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(v_splitter)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = LeftPanel(self)
        self.right_panel = RightPanel(self)

        h_splitter.addWidget(self.left_panel)
        h_splitter.addWidget(self.right_panel)
        h_splitter.setSizes([350, 850])
        
        v_splitter.addWidget(h_splitter)
        
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(5, 0, 5, 5)
        console_layout.addWidget(QLabel("Console Output:"))
        
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        console_layout.addWidget(self.log_window)
        
        v_splitter.addWidget(console_widget)
        v_splitter.setSizes([600, 200])

        # Connect signals between panels
        self.left_panel.signal_subject_selected.connect(self.right_panel.display_subject)
        self.left_panel.fastsurfer_panel.signal_mesh_selected.connect(self.right_panel.display_mesh)
        self.left_panel.signal_log_message.connect(self.log)
        self.right_panel.signal_log_message.connect(self.log)
        self.module_combo.currentTextChanged.connect(self.on_module_changed)

        self.log("SlicerSALT-style UI initialized successfully.")
        
        self.check_slicer_salt()

    def create_menus_and_toolbar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Add basic Edit actions
        undo_action = QAction("Undo", self)
        redo_action = QAction("Redo", self)
        preferences_action = QAction("Preferences...", self)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(preferences_action)
        
        # Add basic View actions
        toggle_toolbar_action = QAction("Toggle Toolbar", self, checkable=True)
        toggle_toolbar_action.setChecked(True)
        reset_view_action = QAction("Reset View", self)
        view_menu.addAction(toggle_toolbar_action)
        view_menu.addSeparator()
        view_menu.addAction(reset_view_action)

        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        save_action = QAction("💾 Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Modules: "))
        self.module_combo = QComboBox()
        self.module_combo.addItems(["Data Importer", "FastSurfer Segmentation", "ICP Registration", "SPHARM Processing", "PLS-DA Analysis", "Feature Extraction"])
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
            self.log(f"SUCCESS: SlicerSALT detected at {slicer_paths[0]}")

    def log(self, message):
        self.log_window.append(message)

    def on_module_changed(self, module_name):
        index = self.module_combo.findText(module_name)
        self.left_panel.switch_module(index)
        
        if module_name == "FastSurfer Segmentation":
            self.right_panel.viewer.set_mesh_view_visible(True)
        else:
            self.right_panel.viewer.set_mesh_view_visible(False)
