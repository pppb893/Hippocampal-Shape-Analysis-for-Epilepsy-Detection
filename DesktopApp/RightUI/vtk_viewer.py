import vtk
import os
from PyQt6.QtWidgets import QWidget, QGridLayout, QFrame, QVBoxLayout, QSlider, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal, Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# A subclass that intercepts and drops Left Clicks to prevent VTK's default Window/Level adjustment
class CustomQVTKWidget(QVTKRenderWindowInteractor):
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            # Ignore left clicks so vtkImageViewer2 doesn't trigger Window/Level adjustment
            return
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            # Ignore left click drags
            return
        super().mouseMoveEvent(ev)

class VtkViewer(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.maximized_frame = None
        self.setup_ui()

    def setup_ui(self):
        # 2x2 Grid Layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2) # Small spacing between panels

        # Create 3 Frames with colored borders
        self.axial_frame = self.create_view_frame("#e74c3c")    # Red
        self.coronal_frame = self.create_view_frame("#2ecc71")  # Green
        self.sagittal_frame = self.create_view_frame("#f1c40f") # Yellow

        self.grid_layout.addWidget(self.axial_frame, 0, 0)
        self.grid_layout.addWidget(self.coronal_frame, 1, 0)
        self.grid_layout.addWidget(self.sagittal_frame, 1, 1)

        # Set up VTK Viewers with Sliders
        self.axial_slider, self.axial_vtkWidget, self.axial_slice_lbl, self.axial_flip_btn = self.setup_vtk_with_slider(self.axial_frame, "Axial View")
        self.axial_viewer = vtk.vtkImageViewer2()
        self.axial_viewer.SetRenderWindow(self.axial_vtkWidget.GetRenderWindow())
        self.axial_viewer.SetupInteractor(self.axial_vtkWidget.GetRenderWindow().GetInteractor())
        self.axial_viewer.SetSliceOrientationToXY()
        self.axial_slider.valueChanged.connect(lambda val: self.change_slice(self.axial_viewer, val, self.axial_slider, self.axial_slice_lbl))
        self.axial_flip_btn.clicked.connect(lambda: self.toggle_flip(self.axial_viewer))

        self.coronal_slider, self.coronal_vtkWidget, self.coronal_slice_lbl, self.coronal_flip_btn = self.setup_vtk_with_slider(self.coronal_frame, "Coronal View")
        self.coronal_viewer = vtk.vtkImageViewer2()
        self.coronal_viewer.SetRenderWindow(self.coronal_vtkWidget.GetRenderWindow())
        self.coronal_viewer.SetupInteractor(self.coronal_vtkWidget.GetRenderWindow().GetInteractor())
        self.coronal_viewer.SetSliceOrientationToXZ()
        self.coronal_slider.valueChanged.connect(lambda val: self.change_slice(self.coronal_viewer, val, self.coronal_slider, self.coronal_slice_lbl))
        self.coronal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.coronal_viewer))

        self.sagittal_slider, self.sagittal_vtkWidget, self.sagittal_slice_lbl, self.sagittal_flip_btn = self.setup_vtk_with_slider(self.sagittal_frame, "Sagittal View")
        self.sagittal_viewer = vtk.vtkImageViewer2()
        self.sagittal_viewer.SetRenderWindow(self.sagittal_vtkWidget.GetRenderWindow())
        self.sagittal_viewer.SetupInteractor(self.sagittal_vtkWidget.GetRenderWindow().GetInteractor())
        self.sagittal_viewer.SetSliceOrientationToYZ()
        self.sagittal_slider.valueChanged.connect(lambda val: self.change_slice(self.sagittal_viewer, val, self.sagittal_slider, self.sagittal_slice_lbl))
        self.sagittal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.sagittal_viewer))

        self.axial_vtkWidget.Initialize()
        self.coronal_vtkWidget.Initialize()
        self.sagittal_vtkWidget.Initialize()

    def create_view_frame(self, color):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ border: 2px solid {color}; background-color: #1a1a1a; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        return frame
        
    def setup_vtk_with_slider(self, frame, title_str):
        layout = frame.layout()
        
        # Top Title and Maximize Button Toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(5, 5, 5, 5)
        
        title_lbl = QLabel(title_str)
        title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent;")
        
        flip_btn = QPushButton("⇅")
        flip_btn.setFixedSize(24, 24)
        flip_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        
        max_btn = QPushButton("◻")
        max_btn.setFixedSize(24, 24)
        max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        max_btn.clicked.connect(lambda: self.toggle_maximize(frame))
        
        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(flip_btn)
        top_bar.addWidget(max_btn)
        
        layout.addLayout(top_bar)
        
        # Slider
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(5, 0, 5, 5)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setEnabled(False)
        slider.setStyleSheet("""
            QSlider { background: transparent; border: none; }
            QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: #333333; margin: 2px 0; }
            QSlider::handle:horizontal { background: #7c7c7c; border: 1px solid #999999; width: 18px; margin: -4px 0; border-radius: 3px; }
            QSlider::handle:horizontal:hover { background: #a0a0a0; }
        """)
        
        slice_lbl = QLabel("Slice: - / -")
        slice_lbl.setStyleSheet("color: white; font-size: 11px; border: none; background: transparent;")
        slice_lbl.setFixedWidth(80)
        slice_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(slice_lbl)
        layout.addLayout(slider_layout)
        
        # We use our custom widget that ignores left clicks
        vtkWidget = CustomQVTKWidget(frame)
        layout.addWidget(vtkWidget)
        
        return slider, vtkWidget, slice_lbl, flip_btn

    def toggle_maximize(self, frame):
        frames = [self.axial_frame, self.coronal_frame, self.sagittal_frame]
        
        if self.maximized_frame is None:
            # Maximize the selected frame
            self.maximized_frame = frame
            for f in frames:
                if f != frame:
                    f.hide()
        else:
            # Restore all frames
            self.maximized_frame = None
            for f in frames:
                f.show()

    def toggle_flip(self, viewer):
        camera = viewer.GetRenderer().GetActiveCamera()
        vu = camera.GetViewUp()
        camera.SetViewUp(-vu[0], -vu[1], -vu[2])
        viewer.Render()

    def change_slice(self, viewer, val, slider, lbl):
        if viewer.GetInput():
            viewer.SetSlice(val)
            viewer.Render()
            lbl.setText(f"Slice: {val} / {slider.maximum()}")

    def reset_camera(self):
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            if viewer.GetInput():
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()
        self.signal_log_message.emit("Camera reset to original position.")

    def display_subject(self, filepath):
        filename = os.path.basename(filepath)
        ext = filepath.lower()
        
        try:
            if ext.endswith('.nii') or ext.endswith('.nii.gz'):
                reader = vtk.vtkNIFTIImageReader()
            elif ext.endswith('.nrrd'):
                reader = vtk.vtkNrrdReader()
            elif ext.endswith('.mha'):
                reader = vtk.vtkMetaImageReader()
            else:
                self.signal_log_message.emit(f"Unsupported image format for slice viewer: {filepath}.")
                return
                
            reader.SetFileName(filepath)
            reader.Update()
            
            image_data = reader.GetOutput()
            if not image_data or image_data.GetDimensions()[0] == 0:
                self.signal_log_message.emit(f"Failed to load image data from {filename}")
                return
                
            # Connect image to viewers
            self.axial_viewer.SetInputData(image_data)
            self.coronal_viewer.SetInputData(image_data)
            self.sagittal_viewer.SetInputData(image_data)
            
            dims = image_data.GetDimensions()
            
            # Setup Sliders
            self.axial_slider.setEnabled(True)
            self.axial_slider.setRange(0, dims[2] - 1)
            self.axial_slider.setValue(dims[2] // 2)
            self.axial_slice_lbl.setText(f"Slice: {dims[2] // 2} / {dims[2] - 1}")
            
            self.coronal_slider.setEnabled(True)
            self.coronal_slider.setRange(0, dims[1] - 1)
            self.coronal_slider.setValue(dims[1] // 2)
            self.coronal_slice_lbl.setText(f"Slice: {dims[1] // 2} / {dims[1] - 1}")
            
            self.sagittal_slider.setEnabled(True)
            self.sagittal_slider.setRange(0, dims[0] - 1)
            self.sagittal_slider.setValue(dims[0] // 2)
            self.sagittal_slice_lbl.setText(f"Slice: {dims[0] // 2} / {dims[0] - 1}")
            
            # Auto window/level based on scalar range
            scalar_range = image_data.GetScalarRange()
            window = scalar_range[1] - scalar_range[0]
            level = (scalar_range[0] + scalar_range[1]) / 2.0
            
            # Explicitly set the flipped ViewUp by default (as it usually loads upside down)
            self.axial_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, -1, 0)
            self.coronal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            self.sagittal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            
            # Update viewers
            for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
                viewer.SetColorWindow(window)
                viewer.SetColorLevel(level)
                
                # Make sure the image acts correctly 
                viewer.GetImageActor().InterpolateOn()
                
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()
                
            self.signal_log_message.emit(f"Loaded MRI volume: {filename}")
            
        except Exception as e:
            self.signal_log_message.emit(f"Error loading volume: {str(e)}")
