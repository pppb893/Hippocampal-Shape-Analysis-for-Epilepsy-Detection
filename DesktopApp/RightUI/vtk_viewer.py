import vtk
import os
from PyQt6.QtWidgets import QWidget, QGridLayout, QFrame, QVBoxLayout, QSlider, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal, Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

def create_mask_actor(mask_path, r, g, b, alpha=0.5):
    if not os.path.exists(mask_path):
        return None
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()
    
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableRange(0, 1)
    lut.SetTableValue(0, 0, 0, 0, 0.0) # Transparent background
    lut.SetTableValue(1, r/255.0, g/255.0, b/255.0, alpha)
    
    map_colors = vtk.vtkImageMapToColors()
    map_colors.SetInputData(reader.GetOutput())
    map_colors.SetLookupTable(lut)
    map_colors.SetOutputFormatToRGBA()
    map_colors.Update()
    
    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputData(map_colors.GetOutput())
    return actor

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
        self.mesh_view_enabled = False
        self.setup_ui()

    def setup_ui(self):
        # 2x2 Grid Layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2) # Small spacing between panels

        # Create 4 Frames with colored borders
        self.axial_frame = self.create_view_frame("#e74c3c")    # Red
        self.coronal_frame = self.create_view_frame("#2ecc71")  # Green
        self.sagittal_frame = self.create_view_frame("#f1c40f") # Yellow
        self.mesh_frame = self.create_view_frame("#3498db")     # Blue

        self.grid_layout.addWidget(self.axial_frame, 0, 0)
        self.grid_layout.addWidget(self.mesh_frame, 0, 1)
        self.grid_layout.addWidget(self.coronal_frame, 1, 0)
        self.grid_layout.addWidget(self.sagittal_frame, 1, 1)

        # Set up VTK Viewers with Sliders, Reset Buttons and 3D Plane Toggles
        self.axial_slider, self.axial_vtkWidget, self.axial_slice_lbl, self.axial_flip_btn, self.axial_3d_btn, self.axial_reset_btn = self.setup_vtk_with_slider(self.axial_frame, "Axial View")
        self.axial_viewer = vtk.vtkImageViewer2()
        self.axial_viewer.SetRenderWindow(self.axial_vtkWidget.GetRenderWindow())
        self.axial_viewer.SetupInteractor(self.axial_vtkWidget.GetRenderWindow().GetInteractor())
        self.axial_viewer.SetSliceOrientationToXY()
        self.axial_slider.valueChanged.connect(lambda val: self.change_slice(self.axial_viewer, val, self.axial_slider, self.axial_slice_lbl))
        self.axial_flip_btn.clicked.connect(lambda: self.toggle_flip(self.axial_viewer))
        self.axial_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("axial", chk))
        self.axial_reset_btn.clicked.connect(lambda: self.reset_slice("axial"))

        self.coronal_slider, self.coronal_vtkWidget, self.coronal_slice_lbl, self.coronal_flip_btn, self.coronal_3d_btn, self.coronal_reset_btn = self.setup_vtk_with_slider(self.coronal_frame, "Coronal View")
        self.coronal_viewer = vtk.vtkImageViewer2()
        self.coronal_viewer.SetRenderWindow(self.coronal_vtkWidget.GetRenderWindow())
        self.coronal_viewer.SetupInteractor(self.coronal_vtkWidget.GetRenderWindow().GetInteractor())
        self.coronal_viewer.SetSliceOrientationToXZ()
        self.coronal_slider.valueChanged.connect(lambda val: self.change_slice(self.coronal_viewer, val, self.coronal_slider, self.coronal_slice_lbl))
        self.coronal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.coronal_viewer))
        self.coronal_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("coronal", chk))
        self.coronal_reset_btn.clicked.connect(lambda: self.reset_slice("coronal"))

        self.sagittal_slider, self.sagittal_vtkWidget, self.sagittal_slice_lbl, self.sagittal_flip_btn, self.sagittal_3d_btn, self.sagittal_reset_btn = self.setup_vtk_with_slider(self.sagittal_frame, "Sagittal View")
        self.sagittal_viewer = vtk.vtkImageViewer2()
        self.sagittal_viewer.SetRenderWindow(self.sagittal_vtkWidget.GetRenderWindow())
        self.sagittal_viewer.SetupInteractor(self.sagittal_vtkWidget.GetRenderWindow().GetInteractor())
        self.sagittal_viewer.SetSliceOrientationToYZ()
        self.sagittal_slider.valueChanged.connect(lambda val: self.change_slice(self.sagittal_viewer, val, self.sagittal_slider, self.sagittal_slice_lbl))
        self.sagittal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.sagittal_viewer))
        self.sagittal_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("sagittal", chk))
        self.sagittal_reset_btn.clicked.connect(lambda: self.reset_slice("sagittal"))

        # Configure solid black background for 2D slice viewers (Axial, Coronal, Sagittal)
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            ren = viewer.GetRenderer()
            ren.GradientBackgroundOff()
            ren.SetBackground(0.0, 0.0, 0.0)
            viewer.GetImageActor().SetVisibility(False)

        # Set up VTK Viewer for 3D Mesh
        layout_mesh = self.mesh_frame.layout()
        mesh_top_bar = QHBoxLayout()
        mesh_top_bar.setContentsMargins(5, 5, 5, 5)
        
        mesh_title_lbl = QLabel("3D Mesh View")
        mesh_title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent;")
        
        mesh_max_btn = QPushButton("◻")
        mesh_max_btn.setFixedSize(24, 24)
        mesh_max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        mesh_max_btn.clicked.connect(lambda: self.toggle_maximize(self.mesh_frame))
        
        mesh_top_bar.addWidget(mesh_title_lbl)
        mesh_top_bar.addStretch()
        mesh_top_bar.addWidget(mesh_max_btn)
        
        layout_mesh.addLayout(mesh_top_bar)
        
        self.mesh_vtkWidget = QVTKRenderWindowInteractor(self.mesh_frame)
        self.mesh_vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        layout_mesh.addWidget(self.mesh_vtkWidget)
        
        self.mesh_renderer = vtk.vtkRenderer()
        self.mesh_renderer.GradientBackgroundOn()
        self.mesh_renderer.SetBackground(0.741, 0.749, 0.902)
        self.mesh_renderer.SetBackground2(0.459, 0.475, 0.745)
        self.mesh_vtkWidget.GetRenderWindow().AddRenderer(self.mesh_renderer)
        
        # 3D Orthogonal Slice Plane Actors & Outline Borders
        self.axial_3d_actor = vtk.vtkImageActor()
        self.axial_3d_actor.SetVisibility(False)
        self.axial_voi = vtk.vtkExtractVOI()
        self.axial_outline = vtk.vtkOutlineFilter()
        self.axial_outline.SetInputConnection(self.axial_voi.GetOutputPort())
        axial_out_map = vtk.vtkPolyDataMapper()
        axial_out_map.SetInputConnection(self.axial_outline.GetOutputPort())
        self.axial_outline_actor = vtk.vtkActor()
        self.axial_outline_actor.SetMapper(axial_out_map)
        self.axial_outline_actor.GetProperty().SetColor(0.95, 0.25, 0.25) # Red outline
        self.axial_outline_actor.GetProperty().SetLineWidth(2.5)
        self.axial_outline_actor.SetVisibility(False)

        self.coronal_3d_actor = vtk.vtkImageActor()
        self.coronal_3d_actor.SetVisibility(False)
        self.coronal_voi = vtk.vtkExtractVOI()
        self.coronal_outline = vtk.vtkOutlineFilter()
        self.coronal_outline.SetInputConnection(self.coronal_voi.GetOutputPort())
        coronal_out_map = vtk.vtkPolyDataMapper()
        coronal_out_map.SetInputConnection(self.coronal_outline.GetOutputPort())
        self.coronal_outline_actor = vtk.vtkActor()
        self.coronal_outline_actor.SetMapper(coronal_out_map)
        self.coronal_outline_actor.GetProperty().SetColor(0.25, 0.85, 0.35) # Green outline
        self.coronal_outline_actor.GetProperty().SetLineWidth(2.5)
        self.coronal_outline_actor.SetVisibility(False)

        self.sagittal_3d_actor = vtk.vtkImageActor()
        self.sagittal_3d_actor.SetVisibility(False)
        self.sagittal_voi = vtk.vtkExtractVOI()
        self.sagittal_outline = vtk.vtkOutlineFilter()
        self.sagittal_outline.SetInputConnection(self.sagittal_voi.GetOutputPort())
        sagittal_out_map = vtk.vtkPolyDataMapper()
        sagittal_out_map.SetInputConnection(self.sagittal_outline.GetOutputPort())
        self.sagittal_outline_actor = vtk.vtkActor()
        self.sagittal_outline_actor.SetMapper(sagittal_out_map)
        self.sagittal_outline_actor.GetProperty().SetColor(0.95, 0.8, 0.1) # Yellow outline
        self.sagittal_outline_actor.GetProperty().SetLineWidth(2.5)
        self.sagittal_outline_actor.SetVisibility(False)

        self.mesh_renderer.AddActor(self.axial_3d_actor)
        self.mesh_renderer.AddActor(self.axial_outline_actor)
        self.mesh_renderer.AddActor(self.coronal_3d_actor)
        self.mesh_renderer.AddActor(self.coronal_outline_actor)
        self.mesh_renderer.AddActor(self.sagittal_3d_actor)
        self.mesh_renderer.AddActor(self.sagittal_outline_actor)
        
        self.mesh_actor = None
        self.mesh_frame.hide()

        self.axial_vtkWidget.Initialize()
        self.axial_vtkWidget.GetRenderWindow().Render()
        self.coronal_vtkWidget.Initialize()
        self.coronal_vtkWidget.GetRenderWindow().Render()
        self.sagittal_vtkWidget.Initialize()
        self.sagittal_vtkWidget.GetRenderWindow().Render()
        self.mesh_vtkWidget.Initialize()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def create_view_frame(self, color):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ border: 2px solid {color}; background-color: #1e2230; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        return frame
        
    def setup_vtk_with_slider(self, frame, title_str):
        layout = frame.layout()
        
        # Top Title and Maximize Button Toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(5, 5, 5, 5)
        top_bar.setSpacing(4)
        
        title_lbl = QLabel(title_str)
        title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent;")
        
        plane_3d_btn = QPushButton("🔲 3D")
        plane_3d_btn.setCheckable(True)
        plane_3d_btn.setChecked(False)
        plane_3d_btn.setVisible(False)  # Hidden initially until a mesh file is clicked in FastSurfer
        plane_3d_btn.setToolTip("Show / Hide slice plane in 3D Mesh View")
        plane_3d_btn.setFixedSize(45, 24)
        plane_3d_btn.setStyleSheet("""
            QPushButton {
                background: #2c3e50;
                color: #bdc3c7;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #34495e;
                color: white;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: 1px solid #1f618d;
            }
        """)
        
        flip_btn = QPushButton("⇅")
        flip_btn.setFixedSize(24, 24)
        flip_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        
        max_btn = QPushButton("◻")
        max_btn.setFixedSize(24, 24)
        max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        max_btn.clicked.connect(lambda: self.toggle_maximize(frame))
        
        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(plane_3d_btn)
        top_bar.addWidget(flip_btn)
        top_bar.addWidget(max_btn)
        
        layout.addLayout(top_bar)
        
        # Slider and Reset Button Row
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(5, 0, 5, 5)
        slider_layout.setSpacing(6)
        
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
        slice_lbl.setFixedWidth(75)
        slice_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        reset_slice_btn = QPushButton("↺ Reset")
        reset_slice_btn.setToolTip("Reset to initial slice")
        reset_slice_btn.setEnabled(False)
        reset_slice_btn.setFixedSize(58, 22)
        reset_slice_btn.setStyleSheet("""
            QPushButton {
                background: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #415b76;
                color: white;
            }
            QPushButton:pressed {
                background: #2c3e50;
            }
            QPushButton:disabled {
                background: #242735;
                color: #666;
                border: 1px solid #3d414d;
            }
        """)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(slice_lbl)
        slider_layout.addWidget(reset_slice_btn)
        layout.addLayout(slider_layout)
        
        # We use our custom widget that ignores left clicks
        vtkWidget = CustomQVTKWidget(frame)
        layout.addWidget(vtkWidget)
        
        return slider, vtkWidget, slice_lbl, flip_btn, plane_3d_btn, reset_slice_btn

    def set_3d_plane_buttons_visible(self, visible: bool):
        self.axial_3d_btn.setVisible(visible)
        self.coronal_3d_btn.setVisible(visible)
        self.sagittal_3d_btn.setVisible(visible)

    def reset_slice(self, orientation):
        if orientation == "axial" and hasattr(self, 'axial_initial_slice'):
            self.axial_slider.setValue(self.axial_initial_slice)
            self.signal_log_message.emit(f"Axial view reset to initial slice {self.axial_initial_slice}.")
        elif orientation == "coronal" and hasattr(self, 'coronal_initial_slice'):
            self.coronal_slider.setValue(self.coronal_initial_slice)
            self.signal_log_message.emit(f"Coronal view reset to initial slice {self.coronal_initial_slice}.")
        elif orientation == "sagittal" and hasattr(self, 'sagittal_initial_slice'):
            self.sagittal_slider.setValue(self.sagittal_initial_slice)
            self.signal_log_message.emit(f"Sagittal view reset to initial slice {self.sagittal_initial_slice}.")

    def toggle_3d_plane(self, orientation, visible):
        if visible:
            self.set_mesh_view_visible(True)
        
        if orientation == "axial":
            self.axial_3d_actor.SetVisibility(visible)
            self.axial_outline_actor.SetVisibility(visible)
            if visible and hasattr(self, 'current_dims'):
                val = self.axial_slider.value()
                dims = self.current_dims
                self.axial_3d_actor.SetDisplayExtent(0, dims[0]-1, 0, dims[1]-1, val, val)
                self.axial_voi.SetVOI(0, dims[0]-1, 0, dims[1]-1, val, val)
        elif orientation == "coronal":
            self.coronal_3d_actor.SetVisibility(visible)
            self.coronal_outline_actor.SetVisibility(visible)
            if visible and hasattr(self, 'current_dims'):
                val = self.coronal_slider.value()
                dims = self.current_dims
                self.coronal_3d_actor.SetDisplayExtent(0, dims[0]-1, val, val, 0, dims[2]-1)
                self.coronal_voi.SetVOI(0, dims[0]-1, val, val, 0, dims[2]-1)
        elif orientation == "sagittal":
            self.sagittal_3d_actor.SetVisibility(visible)
            self.sagittal_outline_actor.SetVisibility(visible)
            if visible and hasattr(self, 'current_dims'):
                val = self.sagittal_slider.value()
                dims = self.current_dims
                self.sagittal_3d_actor.SetDisplayExtent(val, val, 0, dims[1]-1, 0, dims[2]-1)
                self.sagittal_voi.SetVOI(val, val, 0, dims[1]-1, 0, dims[2]-1)
                
        self.mesh_renderer.ResetCameraClippingRange()
        self.mesh_vtkWidget.GetRenderWindow().Render()
        self.signal_log_message.emit(f"3D Slice Plane ({orientation.capitalize()}) {'shown' if visible else 'hidden'} in 3D View.")

    def set_mesh_view_visible(self, visible):
        self.mesh_view_enabled = visible
        if self.maximized_frame is None:
            self.mesh_frame.setVisible(visible)

    def toggle_maximize(self, frame):
        frames = [self.axial_frame, self.coronal_frame, self.sagittal_frame, self.mesh_frame]
        
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
                if f == self.mesh_frame:
                    f.setVisible(self.mesh_view_enabled)
                else:
                    f.show()

    def toggle_flip(self, viewer):
        camera = viewer.GetRenderer().GetActiveCamera()
        vu = camera.GetViewUp()
        camera.SetViewUp(-vu[0], -vu[1], -vu[2])
        viewer.Render()

    def change_slice(self, viewer, val, slider, slice_lbl):
        viewer.SetSlice(val)
        min_val = viewer.GetSliceMin()
        max_val = viewer.GetSliceMax()
        slice_lbl.setText(f"Slice: {val} / {max_val}")
        
        # Sync mask actors' display extent
        extent = viewer.GetImageActor().GetDisplayExtent()
        
        if hasattr(self, 'mask_actors'):
            for orientation_actors in self.mask_actors:
                if orientation_actors['viewer'] == viewer:
                    for actor in orientation_actors['actors']:
                        if actor:
                            actor.SetDisplayExtent(extent)

        # Sync 3D slice plane
        if hasattr(self, 'current_dims'):
            dims = self.current_dims
            if viewer == self.axial_viewer:
                self.axial_3d_actor.SetDisplayExtent(0, dims[0]-1, 0, dims[1]-1, val, val)
                self.axial_voi.SetVOI(0, dims[0]-1, 0, dims[1]-1, val, val)
                if self.axial_3d_actor.GetVisibility():
                    self.mesh_vtkWidget.GetRenderWindow().Render()
            elif viewer == self.coronal_viewer:
                self.coronal_3d_actor.SetDisplayExtent(0, dims[0]-1, val, val, 0, dims[2]-1)
                self.coronal_voi.SetVOI(0, dims[0]-1, val, val, 0, dims[2]-1)
                if self.coronal_3d_actor.GetVisibility():
                    self.mesh_vtkWidget.GetRenderWindow().Render()
            elif viewer == self.sagittal_viewer:
                self.sagittal_3d_actor.SetDisplayExtent(val, val, 0, dims[1]-1, 0, dims[2]-1)
                self.sagittal_voi.SetVOI(val, val, 0, dims[1]-1, 0, dims[2]-1)
                if self.sagittal_3d_actor.GetVisibility():
                    self.mesh_vtkWidget.GetRenderWindow().Render()

    def display_mesh(self, filepath):
        if not filepath.endswith(".nii.gz") and not filepath.endswith(".mgz"):
            self.signal_log_message.emit("[ERROR] Unsupported mesh format. Expected .nii.gz")
            return
            
        self.set_mesh_view_visible(True)
        self.set_3d_plane_buttons_visible(True)
        
        # Clear existing mesh actor only (preserve 3D slice plane actors)
        if self.mesh_actor is not None:
            self.mesh_renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        
        # Load NIFTI mask
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        # Extract surface
        mc2 = vtk.vtkMarchingCubes()
        mc2.SetInputConnection(reader.GetOutputPort())
        mc2.SetValue(0, 0.5)
        
        # Smooth surface
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(mc2.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(0.1)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(15)
        
        self.mesh_actor = actor
        self.mesh_renderer.AddActor(self.mesh_actor)
        self.mesh_renderer.ResetCamera()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def reset_camera(self):
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            if viewer.GetInput():
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()
        self.signal_log_message.emit("Camera reset to original position.")

    def display_segmentation_overlays(self, base_img_path, lh_mask_path, rh_mask_path, side_filter="all"):
        filter_text = "ALL (Left & Right)" if side_filter == "all" else ("LEFT (LH)" if side_filter == "lh" else "RIGHT (RH)")
        self.signal_log_message.emit(f"Loading 2D MRI with {filter_text} Hippocampus overlays...")
        
        # Load the base image first
        self.display_subject(base_img_path)
        
        # Remove old mask actors if they exist
        if hasattr(self, 'mask_actors'):
            for orientation in self.mask_actors:
                for actor in orientation['actors']:
                    if actor:
                        orientation['viewer'].GetRenderer().RemoveActor(actor)
        
        self.mask_actors = []
        
        # For each orientation, we need unique actors (because an actor can only be in one renderer)
        viewers = [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]
        
        for viewer in viewers:
            actors = []
            
            # Left Hippocampus: #A7BCE9 (167, 188, 233)
            if side_filter in ("all", "lh") and lh_mask_path and os.path.exists(lh_mask_path):
                lh_actor = create_mask_actor(lh_mask_path, 167, 188, 233, alpha=0.55)
                if lh_actor:
                    viewer.GetRenderer().AddActor(lh_actor)
                    actors.append(lh_actor)
                    
            # Right Hippocampus: #FFA87B (255, 168, 123)
            if side_filter in ("all", "rh") and rh_mask_path and os.path.exists(rh_mask_path):
                rh_actor = create_mask_actor(rh_mask_path, 255, 168, 123, alpha=0.55)
                if rh_actor:
                    viewer.GetRenderer().AddActor(rh_actor)
                    actors.append(rh_actor)
                
            self.mask_actors.append({
                'viewer': viewer,
                'actors': actors
            })
            
            # Initialize their display extent to match the current slice
            extent = viewer.GetImageActor().GetDisplayExtent()
            for actor in actors:
                actor.SetDisplayExtent(extent)
            
            viewer.Render()

    def display_subject(self, filepath):
        filename = os.path.basename(filepath)
        ext = filepath.lower()
        
        try:
            if ext.endswith('.nii') or ext.endswith('.nii.gz'):
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(filepath)
            elif ext.endswith('.mgz'):
                # Convert mgz to temp nii.gz for VTK reader
                import nibabel as nib
                import numpy as np
                temp_nii = filepath.replace('.mgz', '_conformed.nii.gz')
                if not os.path.exists(temp_nii):
                    img = nib.load(filepath)
                    nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
                    nib.save(nii_img, temp_nii)
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(temp_nii)
            elif ext.endswith('.nrrd'):
                reader = vtk.vtkNrrdReader()
                reader.SetFileName(filepath)
            elif ext.endswith('.mha'):
                reader = vtk.vtkMetaImageReader()
                reader.SetFileName(filepath)
            else:
                self.signal_log_message.emit(f"Unsupported image format for slice viewer: {filepath}.")
                return
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
            self.current_dims = dims
            
            # Setup Sliders and Reset Buttons
            self.axial_slider.setEnabled(True)
            self.axial_slider.setRange(0, dims[2] - 1)
            self.axial_initial_slice = dims[2] // 2
            self.axial_slider.setValue(self.axial_initial_slice)
            self.axial_slice_lbl.setText(f"Slice: {self.axial_initial_slice} / {dims[2] - 1}")
            self.axial_reset_btn.setEnabled(True)
            
            self.coronal_slider.setEnabled(True)
            self.coronal_slider.setRange(0, dims[1] - 1)
            self.coronal_initial_slice = dims[1] // 2
            self.coronal_slider.setValue(self.coronal_initial_slice)
            self.coronal_slice_lbl.setText(f"Slice: {self.coronal_initial_slice} / {dims[1] - 1}")
            self.coronal_reset_btn.setEnabled(True)
            
            self.sagittal_slider.setEnabled(True)
            self.sagittal_slider.setRange(0, dims[0] - 1)
            self.sagittal_initial_slice = dims[0] // 2
            self.sagittal_slider.setValue(self.sagittal_initial_slice)
            self.sagittal_slice_lbl.setText(f"Slice: {self.sagittal_initial_slice} / {dims[0] - 1}")
            self.sagittal_reset_btn.setEnabled(True)
            
            # Auto window/level based on scalar range
            scalar_range = image_data.GetScalarRange()
            window = scalar_range[1] - scalar_range[0]
            level = (scalar_range[0] + scalar_range[1]) / 2.0

            # Connect 3D slice plane actors & VOIs
            self.axial_3d_actor.GetMapper().SetInputData(image_data)
            self.axial_3d_actor.SetDisplayExtent(0, dims[0] - 1, 0, dims[1] - 1, dims[2] // 2, dims[2] // 2)
            self.axial_voi.SetInputData(image_data)
            self.axial_voi.SetVOI(0, dims[0] - 1, 0, dims[1] - 1, dims[2] // 2, dims[2] // 2)

            self.coronal_3d_actor.GetMapper().SetInputData(image_data)
            self.coronal_3d_actor.SetDisplayExtent(0, dims[0] - 1, dims[1] // 2, dims[1] // 2, 0, dims[2] - 1)
            self.coronal_voi.SetInputData(image_data)
            self.coronal_voi.SetVOI(0, dims[0] - 1, dims[1] // 2, dims[1] // 2, 0, dims[2] - 1)

            self.sagittal_3d_actor.GetMapper().SetInputData(image_data)
            self.sagittal_3d_actor.SetDisplayExtent(dims[0] // 2, dims[0] // 2, 0, dims[1] - 1, 0, dims[2] - 1)
            self.sagittal_voi.SetInputData(image_data)
            self.sagittal_voi.SetVOI(dims[0] // 2, dims[0] // 2, 0, dims[1] - 1, 0, dims[2] - 1)

            for act_3d in [self.axial_3d_actor, self.coronal_3d_actor, self.sagittal_3d_actor]:
                act_3d.GetProperty().SetColorWindow(window)
                act_3d.GetProperty().SetColorLevel(level)
                act_3d.InterpolateOn()
            
            # Explicitly set the flipped ViewUp by default (as it usually loads upside down)
            self.axial_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, -1, 0)
            self.coronal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            self.sagittal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            
            # Update viewers
            for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
                viewer.GetImageActor().SetVisibility(True)
                viewer.SetColorWindow(window)
                viewer.SetColorLevel(level)
                
                # Make sure the image acts correctly 
                viewer.GetImageActor().InterpolateOn()
                
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()

            if any([self.axial_3d_actor.GetVisibility(), self.coronal_3d_actor.GetVisibility(), self.sagittal_3d_actor.GetVisibility()]):
                self.mesh_vtkWidget.GetRenderWindow().Render()
                
            self.signal_log_message.emit(f"Loaded MRI volume: {filename}")
            
        except Exception as e:
            self.signal_log_message.emit(f"Error loading volume: {str(e)}")
