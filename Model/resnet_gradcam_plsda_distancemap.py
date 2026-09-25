"""
================================================================================
ResNet Grad-CAM & PLS-DA 8-Component Distance Mapping & 3D Mesh Reconstruction
================================================================================
Pipeline:
1. Load aligned 3D hippocampal surface coordinates (1002 vertices x 3 = 3006 XYZ features).
2. Train ResNet1D on hippocampal shape features to discriminate Healthy vs Epilepsy.
3. Compute Grad-CAM from the last convolutional layer (b3_conv2) to identify key anatomical regions.
4. Fit PLS-DA with 8 components (n_components=8) and rank all 8 components.
5. Select Top 3 components with the highest discriminative importance.
6. For each of the Top 3 components:
   - Sweep latent score from -3.0 to +3.0 with step 0.1 (61 steps total).
   - Invert back to 3D space to reconstruct the deformed shape.
   - Compute point-wise Distance Mapping (displacement magnitude, signed distance, deformation vectors).
   - Attach Grad-CAM attention weights onto the mesh vertices.
   - Export 3D VTK meshes (.vtk) with scalar arrays (DistanceMapping, GradCAM_Importance, DisplacementVectors).
   - Save standard milestone meshes (-3SD to +3SD) and continuous step meshes.
================================================================================
"""

import os
import sys
import glob
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import vtk
from vtk.util import numpy_support

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =============================================================================
# 1. ResNet1D Architecture & Grad-CAM Hook
# =============================================================================

class ResNet1D(nn.Module):
    """
    1D Residual Convolutional Neural Network for Hippocampal Shape Analysis.
    Input: (B, in_channels, seq_len) -> (B, 3, 1002) for 3D coordinates of 1002 vertices.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super(ResNet1D, self).__init__()
        # Initial convolution & max-pooling
        self.init_conv = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.BatchNorm1d(32)
        self.init_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Residual Block 1 (32 -> 32 channels)
        self.b1_conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.b1_bn1 = nn.BatchNorm1d(32)
        self.b1_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.b1_bn2 = nn.BatchNorm1d(32)
        
        # Residual Block 2 (32 -> 64 channels, stride 2)
        self.b2_conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.b2_bn1 = nn.BatchNorm1d(64)
        self.b2_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.b2_bn2 = nn.BatchNorm1d(64)
        self.b2_skip_conv = nn.Conv1d(32, 64, kernel_size=1, stride=2)
        self.b2_skip_bn = nn.BatchNorm1d(64)
        
        # Residual Block 3 (64 -> 128 channels, stride 2)
        # Note: b3_conv2 is the LAST convolutional layer before global pooling
        self.b3_conv1 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.b3_bn1 = nn.BatchNorm1d(128)
        self.b3_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.b3_bn2 = nn.BatchNorm1d(128)
        self.b3_skip_conv = nn.Conv1d(64, 128, kernel_size=1, stride=2)
        self.b3_skip_bn = nn.BatchNorm1d(128)
        
        # Classifier head
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Initial block
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.init_pool(x)
        
        # Block 1
        identity = x
        out = F.relu(self.b1_bn1(self.b1_conv1(x)))
        out = F.relu(self.b1_bn2(self.b1_conv2(out)) + identity)
        
        # Block 2
        identity = self.b2_skip_bn(self.b2_skip_conv(out))
        out = F.relu(self.b2_bn1(self.b2_conv1(out)))
        out = self.b2_bn2(self.b2_conv2(out))
        if out.shape[2] != identity.shape[2]:
            out = F.pad(out, (0, identity.shape[2] - out.shape[2]))
        out = F.relu(out + identity)
        
        # Block 3
        identity = self.b3_skip_bn(self.b3_skip_conv(out))
        out = F.relu(self.b3_bn1(self.b3_conv1(out)))
        out = self.b3_bn2(self.b3_conv2(out))
        if out.shape[2] != identity.shape[2]:
            out = F.pad(out, (0, identity.shape[2] - out.shape[2]))
        out = F.relu(out + identity)
        
        # Pooling & FC
        pooled = self.global_avg_pool(out).view(out.size(0), -1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits


class GradCAM1D:
    """
    Grad-CAM implementation for 1D CNNs targeting the last convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def compute_cam(self, input_tensor, target_class=1):
        """
        Compute Grad-CAM heatmap for the given input_tensor (B, C, L).
        target_class: 1 for Epilepsy / Diseased class.
        Returns: numpy array of shape (B, L) with values normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        
        input_var = input_tensor.clone().requires_grad_(True)
        output = self.model(input_var) # (B, 1)
        
        # For binary classification with single logit: target class 1 is positive logit
        if target_class == 1:
            loss = output.sum()
        else:
            loss = (-output).sum()
            
        loss.backward(retain_graph=True)
        
        # Channel weights: global average of gradients over sequence length
        weights = torch.mean(self.gradients, dim=2, keepdim=True) # (B, C, 1)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (B, 1, L_feat)
        cam = F.relu(cam) # Only features with positive influence
        
        # Interpolate CAM back to the original vertex length (L=1002)
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear', align_corners=False)
        cam = cam.squeeze(1).cpu().numpy() # (B, L)
        
        # Normalize each sample to [0, 1]
        c_min = cam.min(axis=1, keepdims=True)
        c_max = cam.max(axis=1, keepdims=True)
        norm_cam = np.where(c_max > c_min, (cam - c_min) / (c_max - c_min + 1e-8), 0.0)
        return norm_cam


# =============================================================================
# 2. VTK Mesh Helpers
# =============================================================================

def find_template_vtk(side="left", base_dir=None):
    """Locate a valid SPHARM template VTK mesh in the repository."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
    pattern = f"*{side}*_SPHARM.vtk"
    matches = glob.glob(os.path.join(base_dir, "ICP", f"output_{side}_hippocampus", "**", pattern), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(base_dir, "SPHARM", "**", pattern), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
        
    for m in matches:
        if not any(s in os.path.basename(m) for s in ("_ellalign", "_grid", "_realigned", "_procalign", "MedialAxis")):
            return m
    return matches[0] if matches else None


def load_template_polydata(vtk_path):
    """Read a VTK polydata file and compute smooth normals."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not load valid VTK mesh from: {vtk_path}")
        
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    return normals.GetOutput()


def compute_vertex_normals(polydata):
    """Extract (N, 3) vertex normal vectors from vtkPolyData without splitting points."""
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.SplittingOff()
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.Update()
    out_poly = normals_filter.GetOutput()
    normals_arr = out_poly.GetPointData().GetNormals()
    return numpy_support.vtk_to_numpy(normals_arr)


def create_mesh_with_arrays(template_poly, coords_3d, dist_mag, signed_dist, gradcam_weights, disp_vectors, out_path):
    """
    Construct a new vtkPolyData with point coordinates and scalar/vector fields attached,
    then save to out_path.
    """
    new_poly = vtk.vtkPolyData()
    new_poly.DeepCopy(template_poly)
    
    # 1. Update Points
    pts = vtk.vtkPoints()
    for pt in coords_3d:
        pts.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
    new_poly.SetPoints(pts)
    
    # Recompute normals on deformed surface
    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.SetInputData(new_poly)
    norm_filter.ConsistencyOn()
    norm_filter.AutoOrientNormalsOn()
    norm_filter.SplittingOff()
    norm_filter.ComputePointNormalsOn()
    norm_filter.ComputeCellNormalsOff()
    norm_filter.Update()
    new_poly = norm_filter.GetOutput()
    
    # 2. Distance Mapping (Magnitude in mm)
    sc_dist = numpy_support.numpy_to_vtk(dist_mag.astype(np.float64), deep=True)
    sc_dist.SetName("DistanceMapping")
    new_poly.GetPointData().AddArray(sc_dist)
    new_poly.GetPointData().SetScalars(sc_dist) # Default active scalar
    
    # 3. Signed Distance (Inward atrophy vs Outward expansion)
    sc_signed = numpy_support.numpy_to_vtk(signed_dist.astype(np.float64), deep=True)
    sc_signed.SetName("SignedDistance")
    new_poly.GetPointData().AddArray(sc_signed)
    
    # 4. Grad-CAM Saliency / Importance
    if gradcam_weights is not None:
        sc_grad = numpy_support.numpy_to_vtk(gradcam_weights.astype(np.float64), deep=True)
        sc_grad.SetName("GradCAM_Importance")
        new_poly.GetPointData().AddArray(sc_grad)
        
    # 5. Displacement 3D Vectors
    vec_arr = numpy_support.numpy_to_vtk(disp_vectors.astype(np.float64), deep=True)
    vec_arr.SetName("DisplacementVectors")
    new_poly.GetPointData().SetVectors(vec_arr)
    
    # 6. Write out VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(new_poly)
    writer.Write()


# =============================================================================
# 3. Pipeline Runner
# =============================================================================

def run_pipeline(
    side="left",
    dataset="Ds005602",
    n_components=8,
    step=0.1,
    epochs=60,
    batch_size=16,
    lr=0.001,
    output_base_dir=None
):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Locate data files based on dataset selection
    if str(dataset).lower() in ("dataset_1", "dataset1", "ds1", "1"):
        dataset_name = "Dataset_1"
        data_dir = os.path.join(repo_root, "Model", "Dataset_1", side)
        candidates_tr = [
            os.path.join(data_dir, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(data_dir, f"Dataset_1_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "All_Augment_tain", side, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
        ]
        candidates_te = [
            os.path.join(data_dir, f"ALL_{side.capitalize()}_test_xyz_coords.csv"),
            os.path.join(data_dir, f"Dataset_1_{side.capitalize()}_test_xyz_coords.csv"),
            os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side.capitalize()}_test_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "All_Augment_tain", side, f"ALL_{side.capitalize()}_test_xyz_coords.csv"),
        ]
        train_csv = next((c for c in candidates_tr if os.path.exists(c)), None)
        test_csv = next((c for c in candidates_te if os.path.exists(c)), None)
    elif str(dataset).lower() in ("ds005602", "5602", "dataset_2", "dataset2", "ds2", "2"):
        dataset_name = "Ds005602"
        data_dir = os.path.join(repo_root, "Model", "Ds005602", side)
        train_csv = os.path.join(data_dir, f"Ds005602_{side.capitalize()}_train_xyz_coords.csv")
        test_csv = os.path.join(data_dir, f"Ds005602_{side.capitalize()}_test_xyz_coords.csv")
    else:
        dataset_name = "All_Augment_tain"
        data_dir = os.path.join(repo_root, "Model", "All_Augment_tain", side)
        candidates_tr = [
            os.path.join(data_dir, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "Dataset_1", side, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
            os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
        ]
        candidates_te = [
            os.path.join(data_dir, f"ALL_{side.capitalize()}_test_xyz_coords.csv"),
            os.path.join(repo_root, "Model", "Dataset_1", side, f"ALL_{side.capitalize()}_test_xyz_coords.csv"),
            os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side.capitalize()}_test_xyz_coords.csv"),
        ]
        train_csv = next((c for c in candidates_tr if os.path.exists(c)), None)
        test_csv = next((c for c in candidates_te if os.path.exists(c)), None)

    print("=" * 70)
    print(f"ResNet Grad-CAM & PLS-DA 8-Component Distance Mapping: {side.upper()} [{dataset_name}]")
    print("=" * 70)
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training coordinates CSV not found at: {train_csv}")
        
    print(f"Loading data from:\n  Train: {train_csv}\n  Test:  {test_csv}")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv) if os.path.exists(test_csv) else None
    
    # 2. Extract vertex coordinate columns (x_0, y_0, z_0 ... x_1001, y_1001, z_1001)
    coord_cols = [c for c in train_df.columns if c.startswith(('x_', 'y_', 'z_'))]
    if not coord_cols:
        meta_cols = ["Subject", "Group", "Group_Name", "Group_Label", "BinaryClass", "Class", "DataType", "Unnamed: 0"]
        coord_cols = [c for c in train_df.columns if c not in meta_cols]
    
    num_coords = len(coord_cols)
    num_pts = num_coords // 3
    print(f"Loaded {len(train_df)} training subjects. Points per mesh: {num_pts} (Total coords: {num_coords})")
    
    X_train_flat = train_df[coord_cols].values.astype(np.float32)
    y_train = (train_df["Group_Label"].values if "Group_Label" in train_df.columns else train_df["BinaryClass"].values).astype(int)
    
    if test_df is not None:
        X_test_flat = test_df[coord_cols].values.astype(np.float32)
        y_test = (test_df["Group_Label"].values if "Group_Label" in test_df.columns else test_df["BinaryClass"].values).astype(int)
        print(f"Loaded {len(test_df)} test subjects.")
    else:
        X_test_flat, y_test = X_train_flat, y_train
        
    # Reshape coordinates into 3 channels x num_pts vertices: (N, 3, num_pts)
    # X_train_flat has pattern: x_0, y_0, z_0, x_1, y_1, z_1 ...
    def flat_to_tensor(X_flat):
        N = X_flat.shape[0]
        pts_3d = X_flat.reshape(N, num_pts, 3) # (N, 1002, 3)
        pts_ch = np.transpose(pts_3d, (0, 2, 1)) # (N, 3, 1002)
        return torch.tensor(pts_ch, dtype=torch.float32)
        
    X_tr_t = flat_to_tensor(X_train_flat)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = flat_to_tensor(X_test_flat)
    y_te_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # 3. Setup output directory
    if output_base_dir is None:
        output_base_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", dataset_name, side)
    os.makedirs(output_base_dir, exist_ok=True)
    plots_dir = os.path.join(output_base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 4. Train ResNet1D
    print("\n--- Training ResNet1D on 3D Hippocampal Coordinates ---")
    model = ResNet1D(in_channels=3, num_classes=1).to(device)
    
    c0 = np.sum(y_train == 0)
    c1 = np.sum(y_train == 1)
    pos_weight = torch.tensor([float(c0) / float(c1) if c1 > 0 else 1.0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
            
        epoch_loss /= len(train_dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs}] Training Loss: {epoch_loss:.4f}")
            
    model.load_state_dict(best_weights)
    model.eval()
    
    # Evaluate ResNet1D on Test Set
    with torch.no_grad():
        test_logits = model(X_te_t.to(device)).cpu()
        test_probs = torch.sigmoid(test_logits).numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)
        
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 1.0
    print(f"\nResNet Evaluation -> Test Accuracy: {test_acc:.4f} | ROC-AUC: {test_auc:.4f}")
    
    # 5. Grad-CAM on the Last Convolutional Layer (b3_conv2)
    print("\n--- Computing Grad-CAM from Last Conv Layer (b3_conv2) ---")
    target_layer = model.b3_conv2
    cam_extractor = GradCAM1D(model, target_layer)
    
    # Compute Grad-CAM for all test subjects for class 1 (Epilepsy / Diseased)
    all_cams = []
    tle_cams = []
    for i in range(len(X_te_t)):
        single_input = X_te_t[i:i+1].to(device)
        cam = cam_extractor.compute_cam(single_input, target_class=1)[0] # (1002,)
        all_cams.append(cam)
        if y_test[i] == 1:
            tle_cams.append(cam)
            
    cam_extractor.remove_hooks()
    
    # Average Grad-CAM map across epilepsy subjects to represent disease-specific shape importance
    if tle_cams:
        mean_gradcam = np.mean(tle_cams, axis=0)
    else:
        mean_gradcam = np.mean(all_cams, axis=0)
        
    # Min-max scale the aggregated Grad-CAM
    mean_gradcam = (mean_gradcam - mean_gradcam.min()) / (mean_gradcam.max() - mean_gradcam.min() + 1e-8)
    print(f"Grad-CAM generated for {num_pts} vertices. Saliency range: [{mean_gradcam.min():.4f}, {mean_gradcam.max():.4f}]")
    
    # 6. Fit PLS-DA with 8 Components
    print(f"\n--- Fitting PLS-DA Model (n_components = {n_components}) ---")
    Y_train_ohe = np.zeros((len(y_train), 2), dtype=np.float64)
    for idx, label in enumerate(y_train):
        Y_train_ohe[idx, label] = 1.0
        
    pls = PLSRegression(n_components=n_components, scale=True)
    X_scores, _ = pls.fit_transform(X_train_flat, Y_train_ohe)
    
    # PLS components analysis
    # X = T P^T, Y = U Q^T
    # Score matrix T: (N, n_components), Loadings P: (D, n_components), Q: (2, n_components)
    P = pls.x_loadings_ # (3006, 8)
    Q = pls.y_loadings_ # (2, 8)
    
    # Total variance of scaled X in PLS
    X_std = np.std(X_train_flat, axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X_train_flat - np.mean(X_train_flat, axis=0)) / X_std
    total_var_X = np.sum(X_scaled ** 2)
    
    var_explained = []
    q_weights = []
    
    for k in range(n_components):
        t_k = X_scores[:, k]
        p_k = P[:, k]
        # Reconstruct scaled component
        recon_k = np.outer(t_k, p_k)
        var_k = np.sum(recon_k ** 2)
        var_pct = min((var_k / total_var_X) * 100.0, 100.0)
        var_explained.append(var_pct)
        # Q[1, k] corresponds to the Epilepsy class loading
        q_weights.append(abs(Q[1, k]))
        
    var_explained = np.array(var_explained)
    q_weights = np.array(q_weights)
    
    # Composite discriminative score: class correlation * explained variance
    discrim_score = q_weights * np.sqrt(np.maximum(var_explained, 1e-4))
    sorted_comp_indices = np.argsort(-discrim_score)
    top3_indices = sorted_comp_indices[:3]
    
    print("\nPLS-DA 8 Components Summary:")
    print(f"{'Comp #':<8} {'Variance Expl (%)':<20} {'Class Load (|Q_tle|)':<22} {'Importance Score':<18} {'Top-3 Selected'}")
    print("-" * 80)
    
    comp_summary_rows = []
    for k in range(n_components):
        is_top3 = "YES (Rank {})".format(list(top3_indices).index(k) + 1) if k in top3_indices else "No"
        print(f"PLS {k+1:<4} {var_explained[k]:>10.2f}%         {q_weights[k]:>10.4f}             {discrim_score[k]:>10.4f}         {is_top3}")
        comp_summary_rows.append({
            "Component": f"PLS{k+1}",
            "Variance_Explained_Pct": var_explained[k],
            "Class_Loading_Q": q_weights[k],
            "Importance_Score": discrim_score[k],
            "Is_Top3": k in top3_indices,
            "Top3_Rank": list(top3_indices).index(k) + 1 if k in top3_indices else -1
        })
        
    pd.DataFrame(comp_summary_rows).to_csv(os.path.join(output_base_dir, "plsda_8_components_summary.csv"), index=False)
    
    print(f"\n-> Selected Top 3 Components to Compare: PLS{top3_indices[0]+1}, PLS{top3_indices[1]+1}, PLS{top3_indices[2]+1}")
    
    # 7. Locate Template VTK PolyData
    template_vtk_path = find_template_vtk(side, repo_root)
    if not template_vtk_path:
        raise FileNotFoundError(f"Could not locate template VTK file for {side} hippocampus.")
    print(f"\nUsing template mesh: {template_vtk_path}")
    template_poly = load_template_polydata(template_vtk_path)
    
    # Compute mean shape (s = 0)
    mean_coords_flat = pls.inverse_transform(np.zeros((1, n_components)))[0]
    mean_coords_3d = mean_coords_flat.reshape(num_pts, 3)
    
    # Extract mean vertex normals for signed distance computation
    mean_poly = vtk.vtkPolyData()
    mean_poly.DeepCopy(template_poly)
    m_pts = vtk.vtkPoints()
    for pt in mean_coords_3d:
        m_pts.InsertNextPoint(pt[0], pt[1], pt[2])
    mean_poly.SetPoints(m_pts)
    mean_normals = compute_vertex_normals(mean_poly)
    
    # 8. Distance Mapping: -3.0 to +3.0 with Step 0.1 for Top 3 Components
    print(f"\n--- Generating Distance Mapping Meshes (-3.0 to +3.0, step {step}) ---")
    steps = np.round(np.arange(-3.0, 3.0 + 0.05, step), 2)
    print(f"Total steps: {len(steps)} points per component")
    
    milestone_sds = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    sd_names = {
        -3.0: "minus3SD",
        -2.0: "minus2SD",
        -1.0: "minus1SD",
         0.0: "Mean",
         1.0: "plus1SD",
         2.0: "plus2SD",
         3.0: "plus3SD"
    }
    
    dist_summary_rows = []
    
    for rank_idx, comp_idx in enumerate(top3_indices, 1):
        comp_name = f"PLS{comp_idx + 1}"
        comp_dir = os.path.join(output_base_dir, comp_name)
        steps_dir = os.path.join(comp_dir, "steps")
        os.makedirs(steps_dir, exist_ok=True)
        
        # Standard deviation of training scores for this component
        comp_std = np.std(X_scores[:, comp_idx])
        
        print(f"\nProcessing Top {rank_idx}: {comp_name} (Score SD: {comp_std:.4f})...")
        
        for s_idx, s in enumerate(steps):
            # Create score vector with only component comp_idx activated
            latent_vec = np.zeros((1, n_components), dtype=np.float64)
            latent_vec[0, comp_idx] = s * comp_std
            
            # Inverse transform to 3D flattened coordinates
            coords_flat = pls.inverse_transform(latent_vec)[0]
            coords_3d = coords_flat.reshape(num_pts, 3)
            
            # Displacement vectors from Mean shape
            disp_vectors = coords_3d - mean_coords_3d # (1002, 3)
            
            # Euclidean distance mapping (Magnitude in mm)
            dist_mag = np.linalg.norm(disp_vectors, axis=1) # (1002,)
            
            # Signed distance: positive along normal, negative inward (atrophy)
            signed_dist = np.sum(disp_vectors * mean_normals, axis=1)
            
            # Save continuous step mesh
            step_filename = f"{comp_name}_step_{s_idx:02d}_val_{s:+.1f}.vtk"
            step_path = os.path.join(steps_dir, step_filename)
            create_mesh_with_arrays(
                template_poly, coords_3d, dist_mag, signed_dist, mean_gradcam, disp_vectors, step_path
            )
            
            # If this step matches a key SD milestone (-3SD to +3SD), save to comp_dir
            for m_sd, m_label in sd_names.items():
                if np.isclose(s, m_sd, atol=0.01):
                    milestone_filename = f"{comp_name}_{m_label}.vtk"
                    milestone_path = os.path.join(comp_dir, milestone_filename)
                    create_mesh_with_arrays(
                        template_poly, coords_3d, dist_mag, signed_dist, mean_gradcam, disp_vectors, milestone_path
                    )
                    
            dist_summary_rows.append({
                "Component": comp_name,
                "Top_Rank": rank_idx,
                "Step_Index": s_idx,
                "SD_Value": s,
                "Mean_Displacement_mm": np.mean(dist_mag),
                "Max_Displacement_mm": np.max(dist_mag),
                "Mean_Inward_Atrophy_mm": np.mean(np.clip(-signed_dist, 0, None)),
                "Mean_Outward_Expansion_mm": np.mean(np.clip(signed_dist, 0, None))
            })
            
        print(f"  -> Generated {len(steps)} meshes in {comp_dir}")
        
    # Save distance mapping summary CSV
    df_dist = pd.DataFrame(dist_summary_rows)
    df_dist.to_csv(os.path.join(output_base_dir, "top3_distance_mapping_summary.csv"), index=False)
    
    # 9. Generate Summary Figures
    print("\n--- Generating Scientific Visualizations ---")
    
    # Figure 1: Scree Plot of 8 PLS-DA Components
    fig, ax1 = plt.subplots(figsize=(9, 5))
    bar_colors = ["#e74c3c" if k in top3_indices else "#3498db" for k in range(n_components)]
    bars = ax1.bar(range(1, n_components + 1), var_explained, color=bar_colors, alpha=0.85, edgecolor="black", width=0.6)
    ax1.set_xlabel("PLS-DA Component", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Shape Variance Explained (%)", fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.set_title(f"PLS-DA 8 Components Breakdown ({side.capitalize()} Hippocampus)\nRed = Top 3 Selected Components for Distance Mapping", fontsize=13, fontweight="bold")
    ax1.set_xticks(range(1, n_components + 1))
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Secondary axis for Epilepsy Class Loading |Q|
    ax2 = ax1.twinx()
    ax2.plot(range(1, n_components + 1), q_weights, color="#f39c12", marker="o", linewidth=2.5, label="Class Correlation |Q|")
    ax2.set_ylabel("Class Correlation Loading |Q|", fontsize=12, fontweight="bold", color="#d35400")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "plsda_8_components_scree.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # Figure 2: Grad-CAM Saliency across Hippocampal Vertices
    plt.figure(figsize=(10, 4))
    plt.plot(range(num_pts), mean_gradcam, color="#c0392b", linewidth=1.5)
    plt.fill_between(range(num_pts), mean_gradcam, color="#e74c3c", alpha=0.3)
    plt.title(f"ResNet Grad-CAM Attention Across 1002 Hippocampal Vertices ({side.capitalize()})", fontsize=13, fontweight="bold")
    plt.xlabel("Surface Vertex Index (0 to 1001)", fontsize=11, fontweight="bold")
    plt.ylabel("Grad-CAM Saliency Score", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "resnet_gradcam_vertex_profile.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # Figure 3: Distance Mapping Profiles vs SD (-3 to +3)
    plt.figure(figsize=(10, 5))
    colors_top3 = ["#e74c3c", "#2980b9", "#27ae60"]
    for idx, comp_idx in enumerate(top3_indices):
        c_name = f"PLS{comp_idx+1}"
        df_sub = df_dist[df_dist["Component"] == c_name]
        plt.plot(df_sub["SD_Value"], df_sub["Mean_Displacement_mm"], label=f"{c_name} Mean Displacement", color=colors_top3[idx], linewidth=2)
        plt.plot(df_sub["SD_Value"], df_sub["Max_Displacement_mm"], linestyle="--", color=colors_top3[idx], alpha=0.6, label=f"{c_name} Max Displacement")
        
    plt.title(f"Distance Mapping vs SD Trajectory (-3.0 to +3.0) for Top 3 Components", fontsize=13, fontweight="bold")
    plt.xlabel("Component Latent Score (Standard Deviations)", fontsize=11, fontweight="bold")
    plt.ylabel("Displacement from Mean (mm)", fontsize=11, fontweight="bold")
    plt.legend(ncol=3, fontsize=9, loc="upper center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "distance_mapping_vs_sd.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")
    
    # Figure 4 & 5: Violin Plots for PLS-DA Components and Model Predictions
    try:
        sys.path.append(repo_root)
        from Visualize.Data_Plots.plot_plsda_model_violin import generate_violin_plots
        generate_violin_plots(side=side, dataset=dataset_name, output_dir=plots_dir)
    except Exception as e:
        print(f"[WARN] Could not generate violin plots: {e}")
    
    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print(f"Output files stored in: {output_base_dir}")
    print("=" * 70)
    return output_base_dir, top3_indices


# =============================================================================
# 4. CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet Grad-CAM & PLS-DA Top-3 Distance Mapping Pipeline")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="Hippocampus side (left or right, default: right)")
    parser.add_argument("--dataset", type=str, default="Dataset_1", choices=["Dataset_1", "All_Augment_tain", "Ds005602"], help="Dataset (default: Dataset_1)")
    parser.add_argument("--n_components", type=int, default=8, help="Number of PLS-DA components to extract (default: 8)")
    parser.add_argument("--step", type=float, default=0.1, help="Step size for distance mapping sweep (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for ResNet1D (default: 50)")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()
    
    run_pipeline(
        side=args.side,
        dataset=args.dataset,
        n_components=args.n_components,
        step=args.step,
        epochs=args.epochs,
        output_base_dir=args.output_dir
    )
