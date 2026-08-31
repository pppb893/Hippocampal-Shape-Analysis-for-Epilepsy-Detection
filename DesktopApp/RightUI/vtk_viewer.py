import vtk
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class VtkViewer(QWidget):
    signal_log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.add_vtk_placeholder()

    def reset_camera(self):
        if hasattr(self, 'renderer'):
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(0, 0, 100)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.signal_log_message.emit("Camera reset to original position.")

    def display_subject(self, filepath):
        import os
        filename = os.path.basename(filepath)
        if hasattr(self, 'title_actor'):
            self.title_actor.SetInput(f"Preview: {filename}")
            
        self.load_real_mesh(filepath)
        self.vtk_widget.GetRenderWindow().Render()

    def load_real_mesh(self, filepath):
        import os
        filename = os.path.basename(filepath)
        ext = filepath.lower()
        
        try:
            if ext.endswith('.vtk'):
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(filepath)
                reader.Update()
                poly = reader.GetOutput()
                
            elif ext.endswith('.nii') or ext.endswith('.nii.gz'):
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(filepath)
                reader.Update()
                
                dmc = vtk.vtkDiscreteMarchingCubes()
                dmc.SetInputConnection(reader.GetOutputPort())
                dmc.GenerateValues(1, 1, 100)
                dmc.Update()
                poly = dmc.GetOutput()
                
            elif ext.endswith('.nrrd'):
                reader = vtk.vtkNrrdReader()
                reader.SetFileName(filepath)
                reader.Update()
                
                dmc = vtk.vtkDiscreteMarchingCubes()
                dmc.SetInputConnection(reader.GetOutputPort())
                dmc.GenerateValues(1, 1, 100)
                dmc.Update()
                poly = dmc.GetOutput()
            else:
                self.signal_log_message.emit(f"Unsupported format: {filepath}")
                return

            if poly.GetNumberOfPoints() == 0:
                self.signal_log_message.emit("Warning: Loaded mesh has 0 points.")

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.ScalarVisibilityOff()  # Prevent VTK from overriding our custom color with scalar data
            self.mesh_actor.SetMapper(mapper)
            
            # Set color based on hemisphere
            fname_lower = filename.lower()
            if '_lh' in fname_lower or 'lh_' in fname_lower:
                self.mesh_actor.GetProperty().SetColor(0.655, 0.737, 0.914)  # Light Blue (#A7BCE9)
            elif '_rh' in fname_lower or 'rh_' in fname_lower:
                self.mesh_actor.GetProperty().SetColor(1.0, 0.659, 0.482)    # Orange (#FFA87B)
            else:
                self.mesh_actor.GetProperty().SetColor(0.8, 0.6, 0.5)        # Default

            # Center the actor so it doesn't float away
            poly.ComputeBounds()
            bounds = poly.GetBounds()
            cx = (bounds[0] + bounds[1]) / 2.0
            cy = (bounds[2] + bounds[3]) / 2.0
            cz = (bounds[4] + bounds[5]) / 2.0
            self.mesh_actor.SetPosition(-cx, -cy, -cz)

            self.renderer.ResetCamera()
            self.signal_log_message.emit(f"Loaded mesh from: {filename}")
            
        except Exception as e:
            self.signal_log_message.emit(f"Error loading mesh: {str(e)}")

    def add_vtk_placeholder(self):
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        self.renderer.SetBackground2(0.4, 0.4, 0.5)
        self.renderer.GradientBackgroundOn()
        
        # Create empty actors (no placeholder mesh)
        mapper = vtk.vtkPolyDataMapper()
        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(mapper)
        self.mesh_actor.GetProperty().SetColor(0.8, 0.6, 0.5)
        self.mesh_actor.GetProperty().SetAmbient(0.2)
        self.mesh_actor.GetProperty().SetDiffuse(0.8)
        self.mesh_actor.GetProperty().SetSpecular(0.2)
        self.renderer.AddActor(self.mesh_actor)

        self.axes_actor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()
        
        self.title_actor = vtk.vtkTextActor()
        self.title_actor.SetInput("Preview: Hippocampal Mesh (Placeholder)")
        self.title_actor.GetTextProperty().SetColor(1, 1, 1)
        self.title_actor.GetTextProperty().SetFontSize(16)
        self.title_actor.SetDisplayPosition(10, 10)
        self.renderer.AddActor(self.title_actor)
        
        self.renderer.ResetCamera()
        
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.interactor_style)
        self.interactor_style.AddObserver("LeftButtonPressEvent", self.on_vtk_click)
        self.interactor_style.AddObserver("KeyPressEvent", self.on_key_press)
        
        self.interactor.Initialize()

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym()
        if key in ['r', 'R', 'พ']:
            self.reset_camera()
        else:
            # Let the default interactor style handle other keys (like 'w', 's')
            if hasattr(obj, 'OnKeyPress'):
                obj.OnKeyPress()

    def on_vtk_click(self, obj, event):
        obj.OnLeftButtonDown()
        
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = picker.GetActor()
        
        if picked_actor and hasattr(self, 'mesh_actor') and picked_actor == self.mesh_actor:
            self.signal_log_message.emit(f">>> 3D Mesh clicked at screen coordinates {click_pos}!")
                
            self.vtk_widget.GetRenderWindow().Render()
