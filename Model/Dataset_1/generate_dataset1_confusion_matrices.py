"""
================================================================================
Dataset 1: Publication-Quality Confusion Matrices for All 7 Models (Separate Plots)
================================================================================
Specifications:
1. Strict Publication Styling:
   - Font: Serif ('Times New Roman', 14pt)
   - Frame: Solid black 1.5pt spines (top, bottom, left, right)
   - Grid: NO internal grid (grid=False)
   - Ticks: Outward ticks on axes and colorbar
   - Header: NO word "Dataset" in title (e.g. [Right Hippocampus])
   - Cell Dividers: Clean white quadrant separation lines
   - Cell Text: High contrast bold Serif numbers
2. Outputs:
   - Separate plot for each model ('แยก plot')
   - Generated for both Right and Left Hippocampus
   - Saved in high-resolution (300 DPI) across all key project folders
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Publication Styling Configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['mathtext.fontset'] = 'stix'

MODELS_CONFIG = [
    ("MLP", "Dataset_1", "{side}", "MLP", "results", "test_predictions.npz"),
    ("MobileNet", "Dataset_1", "{side}", "MobileNet", "results", "test_predictions.npz"),
    ("PointNet", "PointNet_Results", "{side}", "", "results", "test_predictions.npz"),
    ("ResNet+AE", "Dataset_1", "{side}", "ResNet", "results", "test_predictions_ae.npz"),
    ("ResNet", "Dataset_1", "{side}", "ResNet", "results", "test_predictions.npz"),
    ("SqueezeNet", "Dataset_1", "{side}", "SqueezeNet", "results", "test_predictions.npz"),
    ("SVM", "Dataset_1", "{side}", "SVM", "results", "test_predictions.npz"),
]

def format_spines_and_ticks(ax):
    """Applies strict publication black spines and outward ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.grid(False)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Healthy', 'TLE'], fontsize=13, fontweight='bold')
    ax.set_yticklabels(['Healthy', 'TLE'], fontsize=13, fontweight='bold')
    ax.tick_params(which='major', direction='out', length=5, width=1.5, color='black')
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)

def plot_single_confusion_matrix(cm, model_name, side, acc, out_paths, mode='count_pct'):
    """
    Renders and saves a single publication-grade confusion matrix.
    mode: 'count_pct' (count + percentage) or 'count' (raw integer count only)
    """
    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=300)
    
    vmax = max(55, int(cm.max() * 1.1))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=vmax)
    
    format_spines_and_ticks(ax)
    
    # White quadrant divider lines
    ax.axhline(0.5, color='white', linewidth=2.0)
    ax.axvline(0.5, color='white', linewidth=2.0)
    
    # Text annotation inside cells
    thresh = cm.max() / 2.0
    total_samples = np.sum(cm)
    
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total_samples) * 100.0
            color = 'white' if val > thresh else 'black'
            
            if mode == 'count_pct':
                txt = f"{val}\n({pct:.1f}%)"
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=15, fontweight='bold')
            else:
                txt = f"{val}"
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=18, fontweight='bold')
                
    # Title without the word "Dataset"
    ax.set_title(
        f"Confusion Matrix - {model_name} (Acc: {acc*100:.2f}%)\n[{side.capitalize()} Hippocampus]",
        fontsize=15, fontweight='bold', pad=14
    )
    
    # Colorbar with solid black 1.5pt frame and outward ticks
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.5)
    cbar.ax.tick_params(direction='out', length=4, width=1.2, color='black', labelsize=11)
    cbar.set_label('Sample Count', fontsize=12, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    model_dir = os.path.join(repo_root, "Model")
    
    # Primary output directories
    cm_plots_repo = os.path.join(model_dir, "Dataset_1", "plots", "confusion_matrices")
    cm_plots_excel = os.path.join(repo_root, "Model_Results_Excel", "07_Detailed_Model_Plots", "Dataset_1", "confusion_matrices")
    cm_plots_desktop = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\confusion_matrices"
    
    for d in [cm_plots_repo, cm_plots_excel, cm_plots_desktop]:
        os.makedirs(d, exist_ok=True)
        
    sides = ["right", "left"]
    summary_records = []
    
    for side in sides:
        print(f"\n=======================================================")
        print(f"GENERATING CONFUSION MATRICES: {side.upper()} HIPPOCAMPUS")
        print(f"=======================================================")
        
        for entry in MODELS_CONFIG:
            m_name = entry[0]
            if m_name == "PointNet":
                npz_p = os.path.join(model_dir, "PointNet_Results", side, "results", "test_predictions.npz")
                model_local_plot_dir = os.path.join(model_dir, "PointNet_Results", side, "plots")
            else:
                sub = entry[3]
                npz_name = entry[5]
                npz_p = os.path.join(model_dir, "Dataset_1", side, sub, "results", npz_name)
                model_local_plot_dir = os.path.join(model_dir, "Dataset_1", side, sub, "plots")
                
            if not os.path.exists(npz_p):
                print(f"[ERROR] Prediction file missing: {npz_p}")
                continue
                
            data = np.load(npz_p)
            y_test = data['y_test'] if 'y_test' in data else data['y_true']
            
            if 'y_pred' in data:
                y_pred = data['y_pred']
            else:
                y_prob = data['y_prob']
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            acc = accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred) # Recall for class 1 (TLE)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred)
            spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) # Specificity for class 0 (Healthy)
            
            summary_records.append({
                "Side": side.capitalize(),
                "Model": m_name,
                "Accuracy": acc,
                "Sensitivity_Recall": sens,
                "Specificity": spec,
                "Precision": prec,
                "F1_Score": f1,
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1],
                "Total_N": len(y_test)
            })
            
            safe_name = m_name.replace("+", "_")
            fn_base = f"confusion_matrix_{safe_name}_{side}"
            
            # Paths to save
            save_targets_pct = [
                os.path.join(cm_plots_repo, f"{fn_base}.png"),
                os.path.join(cm_plots_excel, f"{fn_base}.png"),
                os.path.join(cm_plots_desktop, f"{fn_base}.png"),
            ]
            
            # Also update local model folder
            if m_name == "ResNet+AE":
                save_targets_pct.append(os.path.join(model_local_plot_dir, "confusion_matrix_ae.png"))
            else:
                save_targets_pct.append(os.path.join(model_local_plot_dir, "confusion_matrix.png"))
                
            # Render primary plot (Count + Percentage)
            plot_single_confusion_matrix(cm, m_name, side, acc, save_targets_pct, mode='count_pct')
            
            # Also save count-only version
            save_targets_count = [
                os.path.join(cm_plots_repo, f"{fn_base}_count_only.png"),
                os.path.join(cm_plots_excel, f"{fn_base}_count_only.png"),
                os.path.join(cm_plots_desktop, f"{fn_base}_count_only.png"),
            ]
            plot_single_confusion_matrix(cm, m_name, side, acc, save_targets_count, mode='count')
            
            print(f"[{side.upper()}] {m_name:<11} | Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} -> Saved plots")

    # Save summary dataframe
    df_summary = pd.DataFrame(summary_records)
    csv_repo = os.path.join(cm_plots_repo, "confusion_matrices_summary.csv")
    csv_excel = os.path.join(cm_plots_excel, "confusion_matrices_summary.csv")
    df_summary.to_csv(csv_repo, index=False)
    df_summary.to_csv(csv_excel, index=False)
    print(f"\n[OK] Summary table saved to: {csv_repo}")
    print("\n[SUCCESS] All 14 confusion matrices generated successfully!")

if __name__ == "__main__":
    main()
