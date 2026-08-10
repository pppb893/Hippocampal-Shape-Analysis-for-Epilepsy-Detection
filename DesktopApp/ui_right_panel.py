import vtk
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class RightPanel(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        right_layout = QVBoxLayout(self)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # View Controller Toolbar (Slicer style blue bar)
        view_toolbar = QWidget()
        view_toolbar.setStyleSheet("background-color: #6A82D2; border-bottom: 1px solid #5A72C2;")
        view_toolbar.setFixedHeight(28)
        view_toolbar_layout = QHBoxLayout(view_toolbar)
        view_toolbar_layout.setContentsMargins(5, 0, 5, 0)
        view_toolbar_layout.setSpacing(2)
        
        pin_btn = QPushButton("📌")
        pin_btn.setFixedSize(24, 24)
        pin_btn.setStyleSheet("border: none; color: black; background: transparent;")
        
        num_label = QLabel("1")
        num_label.setStyleSheet("color: black; font-weight: bold; border: none; padding-left: 2px; padding-right: 2px;")
        
        center_btn = QPushButton("⌖")
        center_btn.setFixedSize(24, 24)
        center_btn.setStyleSheet("border: none; color: black; background: transparent; font-size: 18px;")
        center_btn.setToolTip("Reset Camera")
        center_btn.clicked.connect(self.reset_vtk_camera)
        
        box_btn = QPushButton("☐")
        box_btn.setFixedSize(24, 24)
        box_btn.setStyleSheet("border: none; color: black; background: transparent; font-size: 18px; font-weight: bold;")
        
        view_toolbar_layout.addWidget(pin_btn)
        view_toolbar_layout.addWidget(num_label)
        view_toolbar_layout.addWidget(center_btn)
        view_toolbar_layout.addWidget(box_btn)
        view_toolbar_layout.addStretch()
        
        right_layout.addWidget(view_toolbar)
        
        # VTK Widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        right_layout.addWidget(self.vtk_widget)

        # Setup VTK Rendering pipeline
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.add_vtk_placeholder()

    def reset_vtk_camera(self):
        if hasattr(self, 'renderer'):
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.signal_log_message.emit("Camera reset to center.")

    def display_subject(self, subject_name):
        if hasattr(self, 'title_actor'):
            self.title_actor.SetInput(f"Preview: {subject_name}")
            self.vtk_widget.GetRenderWindow().Render()

    def add_vtk_placeholder(self):
        """Adds a 3D placeholder to the VTK viewer with Slicer-like decorations."""
        # 1. Gradient Background (Classic Slicer Style)
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        self.renderer.SetBackground2(0.4, 0.4, 0.5)
        self.renderer.GradientBackgroundOn()
        
        # 2. Create a hippocampus-like shape (Ellipsoid)
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(10.0)
        sphere_source.SetPhiResolution(60)
        sphere_source.SetThetaResolution(60)
        
        transform = vtk.vtkTransform()
        transform.Scale(2.5, 1.0, 1.2)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(sphere_source.GetOutputPort())
        transform_filter.SetTransform(transform)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        # Main mesh actor
        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(mapper)
        self.mesh_actor.GetProperty().SetColor(0.8, 0.6, 0.5)
        self.mesh_actor.GetProperty().SetAmbient(0.2)
        self.mesh_actor.GetProperty().SetDiffuse(0.8)
        self.mesh_actor.GetProperty().SetSpecular(0.2)
        self.renderer.AddActor(self.mesh_actor)
        
        # 3. Add a Wireframe Bounding Box
        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(transform_filter.GetOutputPort())
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)
        self.renderer.AddActor(outline_actor)

        # 4. Add Orientation Marker
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        # 5. Add 3D Text Labels
        def add_3d_text(text, position):
            text_source = vtk.vtkVectorText()
            text_source.SetText(text)
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())
            text_actor = vtk.vtkActor()
            text_actor.SetMapper(text_mapper)
            text_actor.SetPosition(position)
            text_actor.SetScale(4, 4, 4)
            text_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            self.renderer.AddActor(text_actor)
            
        add_3d_text("R", (-35, -2, 0))
        add_3d_text("L", (32, -2, 0))
        add_3d_text("A", (-2, 18, 0))
        add_3d_text("P", (-2, -18, 0))
        
        # Setup Title Overlay
        self.title_actor = vtk.vtkTextActor()
        self.title_actor.SetInput("Preview: Hippocampal Mesh (Placeholder)")
        self.title_actor.GetTextProperty().SetColor(1, 1, 1)
        self.title_actor.GetTextProperty().SetFontSize(16)
        self.title_actor.SetDisplayPosition(10, 10)
        self.renderer.AddActor(self.title_actor)
        
        self.renderer.ResetCamera()
        
        # Enable Clicking
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.interactor_style)
        self.interactor_style.AddObserver("LeftButtonPressEvent", self.on_vtk_click)
        
        self.interactor.Initialize()

    def on_vtk_click(self, obj, event):
        obj.OnLeftButtonDown()
        
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = picker.GetActor()
        
        if picked_actor and hasattr(self, 'mesh_actor') and picked_actor == self.mesh_actor:
            self.signal_log_message.emit(f">>> 3D Mesh clicked at screen coordinates {click_pos}!")
            
            current_color = self.mesh_actor.GetProperty().GetColor()
            if round(current_color[0], 2) == 0.80:
                self.mesh_actor.GetProperty().SetColor(0.2, 0.8, 0.2)
            else:
                self.mesh_actor.GetProperty().SetColor(0.8, 0.6, 0.5)
                
            self.vtk_widget.GetRenderWindow().Render()
