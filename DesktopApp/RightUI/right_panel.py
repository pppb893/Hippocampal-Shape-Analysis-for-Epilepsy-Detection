import os
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

    def display_subject(self, filepath):
        self.signal_log_message.emit(f"Displaying subject: {os.path.basename(filepath)}")
        self.viewer.display_subject(filepath)

    def display_mesh(self, filepath, side_filter="all"):
        self.signal_log_message.emit(f"Displaying 3D Mesh: {os.path.basename(filepath)}")
        self.viewer.display_mesh(filepath)
        self.viewer.set_3d_plane_buttons_visible(True)
        
        # Infer output directory and subject id from the mesh filepath
        # Filepath looks like: .../output_dir/left_hippocampus/lh_sub-XXXX_hippocampus.nii.gz
        parent_dir = os.path.dirname(os.path.abspath(filepath))
        output_dir = os.path.dirname(parent_dir)
        filename = os.path.basename(filepath)
        
        subject_id = None
        if filename.startswith("lh_"):
            subject_id = filename[3:].replace("_hippocampus.nii.gz", "").replace(".nii.gz", "")
        elif filename.startswith("rh_"):
            subject_id = filename[3:].replace("_hippocampus.nii.gz", "").replace(".nii.gz", "")
        elif filename.endswith("_hippocampus_lh.nii.gz"):
            subject_id = filename.replace("_hippocampus_lh.nii.gz", "")
        elif filename.endswith("_hippocampus_rh.nii.gz"):
            subject_id = filename.replace("_hippocampus_rh.nii.gz", "")
            
        if subject_id:
            # Search for matching conformed MRI (.nii.gz or .mgz)
            mri_file = None
            search_bases = [output_dir, parent_dir, os.path.dirname(output_dir), os.path.join(output_dir, "fastsurfer")]
            
            for base_d in search_bases:
                if not os.path.isdir(base_d):
                    continue
                # 1. Check for conformed NIfTI MRI in mri folder
                nii_candidate = os.path.join(base_d, "mri", f"{subject_id}_t1.nii.gz")
                if os.path.isfile(nii_candidate):
                    mri_file = nii_candidate
                    break
                
                # 2. Check for orig.mgz in fastsurfer_temp
                mgz_candidates = [
                    os.path.join(base_d, "fastsurfer_temp", subject_id, "mri", "orig.mgz"),
                    os.path.join(base_d, subject_id, "mri", "orig.mgz"),
                ]
                for mgz_c in mgz_candidates:
                    if os.path.isfile(mgz_c):
                        # Convert mgz to NIfTI on the fly for VTK
                        try:
                            import nibabel as nib
                            import numpy as np
                            mri_dir = os.path.join(base_d, "mri")
                            os.makedirs(mri_dir, exist_ok=True)
                            nii_dst = os.path.join(mri_dir, f"{subject_id}_t1.nii.gz")
                            if not os.path.exists(nii_dst):
                                img = nib.load(mgz_c)
                                nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
                                nib.save(nii_img, nii_dst)
                            mri_file = nii_dst
                            break
                        except Exception as e:
                            print(f"[ERROR] Could not convert mgz: {e}")
                if mri_file:
                    break
                    
            # Search for LH and RH mask files
            lh_mask = None
            rh_mask = None
            for base_d in search_bases:
                if not os.path.isdir(base_d):
                    continue
                lh_candidates = [
                    os.path.join(base_d, "left_hippocampus", f"lh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, "left_hippocampus", f"{subject_id}_hippocampus_lh.nii.gz"),
                    os.path.join(base_d, f"lh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, f"{subject_id}_hippocampus_lh.nii.gz"),
                ]
                rh_candidates = [
                    os.path.join(base_d, "right_hippocampus", f"rh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, "right_hippocampus", f"{subject_id}_hippocampus_rh.nii.gz"),
                    os.path.join(base_d, f"rh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, f"{subject_id}_hippocampus_rh.nii.gz"),
                ]
                if not lh_mask:
                    for c in lh_candidates:
                        if os.path.exists(c):
                            lh_mask = c
                            break
                if not rh_mask:
                    for c in rh_candidates:
                        if os.path.exists(c):
                            rh_mask = c
                            break
            
            # If the currently selected file is directly one of the masks, prioritize it
            if filename.startswith("lh_") or "left_hippocampus" in filepath:
                lh_mask = filepath
            elif filename.startswith("rh_") or "right_hippocampus" in filepath:
                rh_mask = filepath
                
            if mri_file:
                self.viewer.display_segmentation_overlays(mri_file, lh_mask, rh_mask, side_filter=side_filter)
            else:
                self.signal_log_message.emit(f"[INFO] 3D mesh rendered for {subject_id}.")
