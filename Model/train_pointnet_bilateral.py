import os
import sys
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score, f1_score, roc_auc_score

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def extract_features_labels(df):
    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType', 'Group_Name', 'Group_Label', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    features_3d_raw = df[feature_cols].values
    
    num_points = features_3d_raw.shape[1] // 3
    features_3d = features_3d_raw.reshape(-1, num_points, 3)
    
    mean_val = np.mean(features_3d, axis=1, keepdims=True)
    std_val = np.std(features_3d, axis=1, keepdims=True) + 1e-8
    features_3d = (features_3d - mean_val) / std_val
    
    X = np.transpose(features_3d, (0, 2, 1))
    
    if 'BinaryClass' in df.columns:
        y = df['BinaryClass'].values
    elif 'Group_Label' in df.columns:
        y = df['Group_Label'].values
    elif 'Class' in df.columns:
        y = df['Class'].values
    else:
        raise ValueError("No binary target label column found")
        
    return X, y

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 512)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = (x + iden).view(-1, self.k, self.k)
        return x

class PointNetDual(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetDual, self).__init__()
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        trans3 = self.input_transform(x)
        x = torch.bmm(trans3, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x_max = torch.max(x, 2)[0]
        x_mean = torch.mean(x, 2)
        global_feat = torch.cat([x_max, x_mean], dim=1)
        
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

def run_pointnet_side(side_name, train_csv, test_csv, output_dir, epochs=100, patience=30):
    print("=" * 70, flush=True)
    print(f"STARTING POINTNET TRAINING: {side_name.upper()} HIPPOCAMPUS", flush=True)
    print(f"Train File: {train_csv}", flush=True)
    print(f"Test File:  {test_csv}", flush=True)
    print("=" * 70, flush=True)

    plots_dir = os.path.join(output_dir, "plots")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading point clouds...", flush=True)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train, y_train = extract_features_labels(train_df)
    X_test, y_test = extract_features_labels(test_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Training set shape: {X_train.shape} | Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)}", flush=True)
    print(f"Test set shape:     {X_test.shape} | Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)}", flush=True)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    c0, c1 = np.sum(y_train == 0), np.sum(y_train == 1)
    weights = torch.tensor([len(y_train)/(2.0*c0), len(y_train)/(2.0*c1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_splits = min(10, min(c0, c1))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_models = []
    cv_accuracies = []

    batch_size = min(32, max(8, len(X_train) // 4))
    print(f"Running PointNetDual {n_splits}-Fold CV with Checkpointing (Batch Size: {batch_size}, Max Epochs: {epochs})...", flush=True)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_t.cpu(), y_train_t.cpu())):
        X_tr, y_tr = X_train_t[train_idx], y_train_t[train_idx]
        X_val, y_val = X_train_t[val_idx], y_train_t[val_idx]
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        model = PointNetDual(num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        
        best_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        best_ep = 1
        no_improve = 0
        actual_ep = 1
        
        for epoch in range(1, epochs + 1):
            actual_ep = epoch
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                    total += batch_x.size(0)
            val_loss /= total
            
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_ep = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                
        best_model = PointNetDual(num_classes=2).to(device)
        best_model.load_state_dict(best_weights)
        best_model.eval()
        with torch.no_grad():
            preds = torch.argmax(best_model(X_val), dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), preds)
        
        cv_accuracies.append(val_acc)
        fold_models.append(best_model)
        print(f"  Fold {fold+1:2d} / {n_splits} Best Val Acc: {val_acc:.4f} (Best Checkpoint: Epoch {best_ep}/{actual_ep})", flush=True)

    elapsed_train = time.time() - t0
    mean_cv = np.mean(cv_accuracies)
    print(f"PointNet Mean CV Accuracy: {mean_cv:.4f} (+/- {np.std(cv_accuracies):.4f}) [Time: {elapsed_train:.1f}s]", flush=True)

    print("\nEvaluating on Test Set (10-Fold Checkpoint Ensemble)...", flush=True)
    probs_list = []
    for m in fold_models:
        m.eval()
        with torch.no_grad():
            probs = F.softmax(m(X_test_t), dim=1)[:, 1].cpu().numpy()
            probs_list.append(probs)

    y_prob = np.mean(probs_list, axis=0)
    y_pred = (y_prob > 0.5).astype(int)

    test_acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n*** Final Test Accuracy: {test_acc:.4f} ***", flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)
    print(f"Sensitivity (TLE Recall): {sens:.4f} | Specificity (Healthy Recall): {spec:.4f} | F1-Macro: {f1_m:.4f} | ROC-AUC: {auc_score:.4f}", flush=True)

    # Save predictions
    pred_path = os.path.join(results_dir, "test_predictions.npz")
    np.savez(pred_path, y_test=y_test, y_pred=y_pred, y_prob=y_prob)
    print(f"Saved test predictions to: {pred_path}", flush=True)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'TLE'], yticklabels=['Healthy', 'TLE'])
    plt.title(f'PointNet {side_name.capitalize()} - Test Acc: {test_acc:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_plot_path}", flush=True)

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'PointNet (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'PointNet {side_name.capitalize()} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_plot_path = os.path.join(plots_dir, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    print(f"Saved ROC Curve to: {roc_plot_path}", flush=True)

    # 1,000-Round True Bootstrapping
    print("\nComputing 1,000-Round True Bootstrapping for 95% Confidence Intervals...", flush=True)
    n_boot = 1000
    n = len(y_test)
    boot_records = []
    
    for i in range(1, n_boot + 1):
        idx = np.random.choice(n, size=n, replace=True)
        yt_b = y_test[idx]
        yp_b = y_pred[idx]
        ypr_b = y_prob[idx]
        
        cm_b = confusion_matrix(yt_b, yp_b, labels=[0, 1])
        tn, fp, fn, tp = cm_b.ravel()
        acc_b = (tp + tn) / n
        sens_b = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_b = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec_b = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_b = 2 * prec_b * sens_b / (prec_b + sens_b) if (prec_b + sens_b) > 0 else 0
        
        if len(np.unique(yt_b)) < 2:
            auc_b = np.nan
        else:
            try:
                fpr_b, tpr_b, _ = roc_curve(yt_b, ypr_b)
                auc_b = auc(fpr_b, tpr_b)
            except:
                auc_b = np.nan
                
        boot_records.append({
            'Dataset': 'Dataset_1',
            'Side': side_name.capitalize(),
            'Model': 'PointNet',
            'round': i,
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'Accuracy': round(acc_b, 4),
            'Sensitivity': round(sens_b, 4),
            'Specificity': round(spec_b, 4),
            'F1': round(f1_b, 4),
            'AUC': round(auc_b, 4) if not np.isnan(auc_b) else ''
        })
        
    boot_df = pd.DataFrame(boot_records)
    boot_csv_path = os.path.join(results_dir, "bootstrap_confusion_matrix.csv")
    boot_df.to_csv(boot_csv_path, index=False)
    print(f"Saved 1,000-round bootstrap to: {boot_csv_path}", flush=True)

    acc_ci = (boot_df['Accuracy'].quantile(0.025), boot_df['Accuracy'].quantile(0.975))
    sens_ci = (boot_df['Sensitivity'].quantile(0.025), boot_df['Sensitivity'].quantile(0.975))
    spec_ci = (boot_df['Specificity'].quantile(0.025), boot_df['Specificity'].quantile(0.975))
    f1_ci = (boot_df['F1'].quantile(0.025), boot_df['F1'].quantile(0.975))
    auc_numeric = pd.to_numeric(boot_df['AUC'], errors='coerce').dropna()
    auc_ci = (auc_numeric.quantile(0.025), auc_numeric.quantile(0.975)) if len(auc_numeric) > 0 else (0.5, 0.5)

    return {
        'Side': side_name.capitalize(),
        'CV_Acc': f"{mean_cv*100:.2f}%",
        'Test_Acc': f"{test_acc*100:.2f}%",
        'Sensitivity': f"{sens*100:.2f}%",
        'Specificity': f"{spec*100:.2f}%",
        'F1_Macro': f"{f1_m*100:.2f}%",
        'ROC_AUC': f"{auc_score:.4f}",
        'Boot_Acc_95CI': f"[{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]",
        'Boot_Sens_95CI': f"[{sens_ci[0]*100:.2f}%, {sens_ci[1]*100:.2f}%]",
        'Boot_Spec_95CI': f"[{spec_ci[0]*100:.2f}%, {spec_ci[1]*100:.2f}%]",
        'Boot_F1_95CI': f"[{f1_ci[0]*100:.2f}%, {f1_ci[1]*100:.2f}%]",
        'Boot_AUC_95CI': f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
        'Time': f"{elapsed_train:.1f}s"
    }, boot_df

def main():
    poinnet_dir = r"C:\Users\IHCK\Desktop\poinnet"
    out_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "PointNet_Results"))
    desktop_dst = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results"

    summary_list = []
    
    # 1. Left Hippocampus
    left_train = os.path.join(poinnet_dir, "ALL_Left_train_xyz_coords.csv")
    left_test = os.path.join(poinnet_dir, "ALL_Left_test_xyz_coords.csv")
    left_out = os.path.join(out_base, "left")
    res_left, boot_left = run_pointnet_side("left", left_train, left_test, left_out, epochs=100, patience=30)
    summary_list.append(res_left)

    # Copy to Desktop folder if exists
    if os.path.exists(desktop_dst):
        os.makedirs(os.path.join(desktop_dst, "Left"), exist_ok=True)
        left_boot_dst = os.path.join(desktop_dst, "Left", "PointNet_bootstrap_1000.csv")
        boot_left.to_csv(left_boot_dst, index=False)
        print(f"Copied Left PointNet bootstrap to Desktop: {left_boot_dst}", flush=True)

    print("\n" + "#" * 70 + "\n", flush=True)

    # 2. Right Hippocampus
    right_train = os.path.join(poinnet_dir, "ALL_Right_train_xyz_coords.csv")
    right_test = os.path.join(poinnet_dir, "ALL_Right_test_xyz_coords.csv")
    right_out = os.path.join(out_base, "right")
    res_right, boot_right = run_pointnet_side("right", right_train, right_test, right_out, epochs=100, patience=30)
    summary_list.append(res_right)

    # Copy to Desktop folder if exists
    if os.path.exists(desktop_dst):
        os.makedirs(os.path.join(desktop_dst, "Right"), exist_ok=True)
        right_boot_dst = os.path.join(desktop_dst, "Right", "PointNet_bootstrap_1000.csv")
        boot_right.to_csv(right_boot_dst, index=False)
        print(f"Copied Right PointNet bootstrap to Desktop: {right_boot_dst}", flush=True)

    # Update workspace consolidated bootstrap file
    all_raw_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset_1", "dataset1_all_models_raw_bootstrap_1000.csv"))
    if os.path.exists(all_raw_csv):
        raw_df = pd.read_csv(all_raw_csv)
        raw_df = raw_df[raw_df['Model'] != 'PointNet']
        updated_raw = pd.concat([raw_df, boot_left, boot_right], ignore_index=True)
        updated_raw.to_csv(all_raw_csv, index=False)
        print(f"Updated full benchmark bootstrap file with PointNet: {all_raw_csv}", flush=True)

    # Save overall summary
    sum_df = pd.DataFrame(summary_list)
    sum_csv = os.path.join(out_base, "pointnet_summary.csv")
    sum_df.to_csv(sum_csv, index=False)
    print("\n" + "=" * 80, flush=True)
    print("POINTNET BILATERAL TRAINING BENCHMARK SUMMARY (100 EPOCHS, 10-FOLD CV)", flush=True)
    print("=" * 80, flush=True)
    print(sum_df.to_string(), flush=True)
    print(f"\nSaved summary to: {sum_csv}", flush=True)

if __name__ == "__main__":
    main()
