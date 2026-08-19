import sys
import time
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor, QFont, QPainter, QPen
from PyQt6.QtCore import Qt, QLocale

from ui_main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set default locale to C/English to force standard Arabic numerals (1000, 10, 12)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    
    # ----------------------------------------------------
    # Splash Screen (Slicer/SlicerSALT style startup)
    # ----------------------------------------------------
    # Create a dynamic pixmap for the splash screen
    pixmap = QPixmap(600, 350)
    pixmap.fill(QColor("#2c3e50")) # Dark blue Slicer-style background
    
    # Draw a simple logo/title directly on the splash image
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor("white")))
    painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Shape Analysis Toolbox\nInitializing...")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Arial", 11, QFont.Weight.Bold))
    splash.show()
    
    # Simulate loading steps with visual feedback
    splash.showMessage("Starting Hippocampal Shape Analysis Pipeline...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Loading VTK Rendering Engine...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Checking System Requirements & SlicerSALT...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Initializing User Interface Modules...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.4)
    
    # Initialize the heavy main window
    window = MainWindow()
    
    # Close splash and show main window
    splash.finish(window)
    window.show()
    
    sys.exit(app.exec())
