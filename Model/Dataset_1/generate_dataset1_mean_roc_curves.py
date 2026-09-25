"""
================================================================================
Dataset 1: Publication-Quality Mean ROC Curves (All Models & Best Model Standalone)
================================================================================
Specifications:
1. Strict Publication Styling:
   - Font: Serif ('Times New Roman', 14pt)
   - Frame: Solid black 1.5pt spines (top, bottom, left, right)
   - Grid: NO internal grid (grid=False)
   - Ticks: Outward ticks (major 0.10, minor 0.02)
   - NO smoothing: Empirical piecewise-linear mean ROC curve (no splines or gaussian blur)
2. Two Target Plots per Side:
   - Plot 1: Comparison of all 7 models ranked by Mean AUC with Chance line
   - Plot 2: Best Performer Standalone plot with +/- 1 SD Bootstrap Confidence Band
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import roc_curve, auc

# Global Serif publication styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'

MODEL_STYLES = {
    "MobileNet": {"color": "#3498DB", "lw": 2.5, "ls": "-"},     # Steel Blue
    "MLP":       {"color": "#708090", "lw": 2.2, "ls": "--"},    # Slate Gray
    "ResNet+AE": {"color": "#E67E22", "lw": 2.2, "ls": "-."},    # Warm Orange
    "SqueezeNet":{"color": "#1ABC9C", "lw": 2.2, "ls": ":"},     # Emerald Teal
    "ResNet":    {"color": "#E74C3C", "lw": 2.2, "ls": "-"},     # Coral Red
    "SVM":       {"color": "#2ECC71", "lw": 2.0, "ls": "--"},    # Green
    "PointNet":  {"color": "#9B59B6", "lw": 2.4, "ls": "-"},     # Royal Purple
}

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
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
    ax.grid(False)
    
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ax.tick_params(which='major', direction='out', length=6, width=1.5, color='black', labelsize=12)
    ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
    ax.tick_params(axis='x', pad=8)
    ax.tick_params(axis='y', pad=8)
    
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=14, fontweight='bold', labelpad=10)

def compute_empirical_bootstrap_roc(models_dict, n_boot=1000, seed=42):
    """
    Computes empirical un-smoothed bootstrap mean ROC curves using shared resamplings.
    """
    # Verify sample size
    first_key = list(models_dict.keys())[0]
    yt_first = models_dict[first_key]["y_true"]
    N = len(yt_first)
    
    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, N, size=(n_boot, N))
    mean_fpr = np.linspace(0, 1, 101) # Standard grid, no smoothing
    
    results = []
    for m_name, d in models_dict.items():
        y_true = d["y_true"]
        y_prob = d["y_prob"]
        
        tprs = []
        aucs = []
        for b in range(n_boot):
            idx = boot_indices[b]
            yt_b = y_true[idx]
            yp_b = y_prob[idx]
            if len(np.unique(yt_b)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt_b, yp_b)
            # Piecewise linear interpolation on FPR grid (NO splines, NO smoothing)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
        
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        ci_low = float(np.percentile(aucs, 2.5))
        ci_high = float(np.percentile(aucs, 97.5))
        
        results.append({
            "model": m_name,
            "fpr": mean_fpr,
            "tpr_mean": mean_tpr,
            "tpr_lower": tpr_lower,
            "tpr_upper": tpr_upper,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
        
    results.sort(key=lambda x: x["mean_auc"], reverse=True)
    return results

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    model_dir = os.path.join(repo_root, "Model")
    
    # Destination directories
    desktop_plots = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\roc_curves"
    repo_plots = os.path.join(model_dir, "Dataset_1", "plots", "roc_curves")
    excel_plots = os.path.join(repo_root, "Model_Results_Excel", "06_ROC_Curves", "Dataset_1")
    
    out_dirs = [desktop_plots, repo_plots, excel_plots]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    sides = ["right", "left"]
    
    for side in sides:
        print(f"\n=======================================================")
        print(f"PROCESSING DATASET 1: {side.upper()} HIPPOCAMPUS")
        print(f"=======================================================")
        
        # 1. Load predictions
        models_data = {}
        for entry in MODELS_CONFIG:
            m_name = entry[0]
            if m_name == "PointNet":
                npz_p = os.path.join(model_dir, "PointNet_Results", side, "results", "test_predictions.npz")
            else:
                sub = entry[3]
                npz_name = entry[5]
                npz_p = os.path.join(model_dir, "Dataset_1", side, sub, "results", npz_name)
                
            if not os.path.exists(npz_p):
                print(f"[ERROR] Missing file: {npz_p}")
                sys.exit(1)
                
            data = np.load(npz_p)
            y_true = data["y_test"] if "y_test" in data else data["y_true"]
            y_prob = data["y_prob"]
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.flatten()
            models_data[m_name] = {"y_true": y_true, "y_prob": y_prob}
            
        # 2. Compute empirical un-smoothed bootstrap ROC
        roc_results = compute_empirical_bootstrap_roc(models_data, n_boot=1000, seed=42)
        
        best_model_item = roc_results[0]
        print(f"Best Performer: {best_model_item['model']} (AUC = {best_model_item['mean_auc']:.4f} +/- {best_model_item['std_auc']:.4f})")
        
        # -------------------------------------------------------------
        # PLOT 1: ALL 7 MODELS COMPARISON (Mean ROC Curves)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=300)
        
        # Chance line
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D", lw=1.5,
                label="Chance (AUC = 0.500)", zorder=1)
        
        # All 7 models (sorted by AUC descending)
        for item in roc_results:
            m_name = item["model"]
            st = MODEL_STYLES.get(m_name, {"color": "black", "lw": 2.2, "ls": "-"})
            lbl = f"{m_name:<11} : AUC = {item['mean_auc']:.3f} \u00b1 {item['std_auc']:.3f}"
            is_best = (m_name == best_model_item["model"])
            
            ax.plot(
                item["fpr"],
                item["tpr_mean"],
                color=st["color"],
                lw=st["lw"] if not is_best else st["lw"] + 0.6,
                linestyle=st["ls"],
                label=lbl,
                zorder=4 if is_best else 2
            )
            
        format_spines_and_ticks(ax)
        
        ax.set_title(
            f"Mean ROC Curves Comparison (1,000 Bootstraps)\n[{side.capitalize()} Hippocampus]",
            fontsize=15, fontweight="bold", pad=14
        )
        
        # Professional legend placed lower right
        leg = ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.96,
            edgecolor="black",
            fontsize=11.5,
            title="Model Architectures (Mean AUC \u00b1 SD)"
        )
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(12)
        leg.get_frame().set_linewidth(1.2)
        
        plt.tight_layout()
        fn_all = f"mean_roc_curves_all_7models_{side}.png"
        for od in out_dirs:
            p_out = os.path.join(od, fn_all)
            plt.savefig(p_out, dpi=300)
            print(f"[OK] Saved Plot 1: {p_out}")
        plt.close()
        
        # -------------------------------------------------------------
        # PLOT 2: BEST MODEL ONLY (Standalone Plot with +/- 1 SD Band)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=300)
        
        best_name = best_model_item["model"]
        st = MODEL_STYLES.get(best_name, {"color": "#3498DB", "lw": 2.8, "ls": "-"})
        
        # 1. Best model empirical mean ROC line (first in legend)
        ax.plot(
            best_model_item["fpr"],
            best_model_item["tpr_mean"],
            color=st["color"],
            lw=3.0,
            linestyle="-",
            label=f"{best_name} (Mean AUC = {best_model_item['mean_auc']:.3f} \u00b1 {best_model_item['std_auc']:.3f})",
            zorder=4
        )
        
        # 2. Shaded Confidence Band (+/- 1 SD)
        ax.fill_between(
            best_model_item["fpr"],
            best_model_item["tpr_lower"],
            best_model_item["tpr_upper"],
            color=st["color"],
            alpha=0.22,
            label=f"\u00b11 SD Bootstrap Band",
            zorder=2
        )
        
        # 3. Chance line
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D", lw=1.5,
                label="Chance (AUC = 0.500)", zorder=1)
        
        format_spines_and_ticks(ax)
        
        ax.set_title(
            f"Mean ROC Curve (1,000 Bootstraps) - Best Performer\n{best_name} [{side.capitalize()} Hippocampus]",
            fontsize=15, fontweight="bold", pad=14
        )
        
        leg = ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.96,
            edgecolor="black",
            fontsize=12,
            title="Evaluation Metric"
        )
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(12.5)
        leg.get_frame().set_linewidth(1.2)
        
        plt.tight_layout()
        fn_best = f"mean_roc_curve_best_model_{best_name}_{side}.png"
        for od in out_dirs:
            p_out = os.path.join(od, fn_best)
            plt.savefig(p_out, dpi=300)
            print(f"[OK] Saved Plot 2: {p_out}")
        plt.close()

    print("\n[SUCCESS] All requested mean ROC plots generated successfully!")

if __name__ == "__main__":
    main()
