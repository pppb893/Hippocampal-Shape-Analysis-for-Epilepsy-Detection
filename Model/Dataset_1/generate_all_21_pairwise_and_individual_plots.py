import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Set Serif font globally - publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'

# UNIFIED IDENTICAL SCALE ACROSS ALL PLOTS (AUC & F1):
# Lowest whisker across all 14 models is 0.308 (SVM Right F1).
# At 0.20, there is a >0.10 clearance from bottom spine.
# At 1.10, with max value 1.00, bracket at 1.035 and text at 1.050,
# there is a >0.05 clearance below top spine. Zero line collision!
UNIFIED_YLIM = (0.20, 1.10)

# Optional metric-specific scales (AUC zoomed to 0.50-1.10, F1 0.20-1.10)
METRIC_YLIMS = {
    "AUC": (0.50, 1.10),
    "F1": (0.20, 1.10)
}

# Distinctive publication model colors
MODEL_COLORS = {
    "MLP": "#708090",        # Slate Gray
    "MobileNet": "#3498DB",  # Steel Blue
    "PointNet": "#9B59B6",   # Royal Purple (Proposed)
    "ResNet+AE": "#E67E22",  # Warm Orange
    "ResNet": "#E74C3C",     # Coral Red
    "SqueezeNet": "#1ABC9C", # Emerald Teal
    "SVM": "#2ECC71"         # Green
}

def get_p_val_and_symbol(vals_a, vals_b):
    diff = vals_a - vals_b
    p_left = np.mean(diff <= 0)
    p_right = np.mean(diff >= 0)
    p = 2 * min(p_left, p_right)
    p = min(1.0, max(p, 1 / len(diff)))
    
    if p < 0.001:
        return p, "***", "*** (p < 0.001)"
    elif p < 0.01:
        return p, "**", f"** (p = {p:.3f})"
    elif p < 0.05:
        return p, "*", f"* (p = {p:.3f})"
    else:
        return p, "ns", f"ns (p = {p:.2f})"

def plot_2box_panel(ax, vals_a, vals_b, model_a, model_b, title, y_lim, y_label=None, color_a=None, color_b=None):
    """
    Renders exactly 2 boxes (a true pair / คู่) with generous headroom and footroom.
    Guaranteed: NO line touching top or bottom border!
    """
    p_val, p_sym, p_text = get_p_val_and_symbol(vals_a, vals_b)
    
    col_a = color_a or MODEL_COLORS.get(model_a, "#5D9CEC")
    col_b = color_b or MODEL_COLORS.get(model_b, "#48CFAD")
    
    box_data = [vals_a, vals_b]
    positions = [1.0, 2.0]
    
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=7),
        medianprops=dict(color='black', linewidth=2.0),
        boxprops=dict(linewidth=1.5, edgecolor='black'),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3.5, markerfacecolor='gray', markeredgecolor='none', alpha=0.3)
    )
    
    bp['boxes'][0].set_facecolor(col_a)
    bp['boxes'][1].set_facecolor(col_b)
    
    # Bracket placement: 1.035 ensures >0.035 clearance above max data (1.00)
    # and leaves >0.050 clearance below top frame (1.10)
    bracket_y = 1.035
    bar_h = 0.015
    
    # Draw bracket
    ax.plot([1.0, 1.0, 2.0, 2.0], [bracket_y - bar_h, bracket_y, bracket_y, bracket_y - bar_h],
            lw=1.5, c='black')
    # Text placed cleanly above bracket line
    ax.text(1.5, bracket_y + 0.015, p_text, ha='center', va='bottom',
            fontsize=13, fontweight='bold', fontfamily='serif')
    
    # Frame formatting: solid black 1.5pt, no internal grid
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
        
    ax.grid(False)
    
    # Unified ticks: major 0.05, minor 0.01, outward
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(which='major', direction='out', length=6, width=1.5, color='black', labelsize=12)
    ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([model_a, model_b], fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', direction='out', length=6, width=1.5, color='black', pad=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(y_lim)
    ax.set_xlim(0.4, 2.6)
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

def generate_7x7_matrices(df, models, out_dirs):
    """
    Generates and saves the complete 7x7 pairwise significance matrices for:
    1. AUC Left
    2. AUC Right
    3. F1 Left
    4. F1 Right
    """
    matrix_configs = [
        ("AUC", 0, "AUC_Left"),
        ("AUC", 7, "AUC_Right"),
        ("F1", 0, "F1_Left"),
        ("F1", 7, "F1_Right"),
    ]
    
    for metric, offset, tag in matrix_configs:
        mat = pd.DataFrame(index=models, columns=models)
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i == j:
                    mat.loc[m1, m2] = "—"
                else:
                    a = df[f"{metric}_{i + offset}"].values
                    b = df[f"{metric}_{j + offset}"].values
                    p, sym, p_text = get_p_val_and_symbol(a, b)
                    mat.loc[m1, m2] = p_text
                    
        for d in out_dirs:
            csv_fn = f"PValue_Matrix_7x7_{tag}.csv"
            mat.to_csv(os.path.join(d, csv_fn))
            print(f"[OK] Saved 7x7 Matrix: {os.path.join(d, csv_fn)}")

def run_pipeline(use_unified_scale=True):
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    csv_path = os.path.join(repo_root, "Model", "Dataset_1", "Combined_Bootstrap_Results.csv")
    df = pd.read_csv(csv_path)
    
    models = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    safe_names = ["MLP", "MobileNet", "PointNet", "ResNet_AE", "ResNet", "SqueezeNet", "SVM"]
    
    scale_label = "unified_020_to_110" if use_unified_scale else "metric_specific"
    
    # Destination directories
    desktop_root = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results"
    repo_plots = os.path.join(repo_root, "Model", "Dataset_1", "plots")
    excel_dir = os.path.join(repo_root, "Model_Results_Excel", "02_Bootstrap_and_Statistical_Tests", "Dataset_1_Combined")
    
    # Target folders for plots
    dirs_individual = [
        os.path.join(desktop_root, "plots", "individual_models"),
        os.path.join(repo_plots, "individual_models"),
        os.path.join(desktop_root, "plots", f"individual_models_{scale_label}"),
    ]
    dirs_pairwise_bilateral = [
        os.path.join(desktop_root, "plots", "pairwise_all_21_pairs", "bilateral"),
        os.path.join(repo_plots, "pairwise_all_21_pairs", "bilateral"),
        os.path.join(desktop_root, "plots", f"pairwise_all_21_pairs_{scale_label}", "bilateral"),
    ]
    dirs_pairwise_left = [
        os.path.join(desktop_root, "plots", "pairwise_all_21_pairs", "left_only"),
        os.path.join(repo_plots, "pairwise_all_21_pairs", "left_only"),
        os.path.join(desktop_root, "plots", f"pairwise_all_21_pairs_{scale_label}", "left_only"),
    ]
    dirs_pairwise_right = [
        os.path.join(desktop_root, "plots", "pairwise_all_21_pairs", "right_only"),
        os.path.join(repo_plots, "pairwise_all_21_pairs", "right_only"),
        os.path.join(desktop_root, "plots", f"pairwise_all_21_pairs_{scale_label}", "right_only"),
    ]
    
    for d in dirs_individual + dirs_pairwise_bilateral + dirs_pairwise_left + dirs_pairwise_right:
        os.makedirs(d, exist_ok=True)
        
    # Generate 7x7 matrices
    matrix_dirs = [desktop_root, os.path.join(repo_root, "Model", "Dataset_1"), excel_dir]
    for md in matrix_dirs:
        os.makedirs(md, exist_ok=True)
    generate_7x7_matrices(df, models, matrix_dirs)
    
    # -------------------------------------------------------------
    # 1. GENERATE ALL 14 INDIVIDUAL MODEL BOXPLOTS (Clean, NO p-value)
    # -------------------------------------------------------------
    print(f"\n--- 1. Generating 14 Individual Boxplots (Scale: {scale_label}) ---")
    for metric in ["AUC", "F1"]:
        y_lim = UNIFIED_YLIM if use_unified_scale else METRIC_YLIMS[metric]
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        
        for i, (m_name, s_name) in enumerate(zip(models, safe_names)):
            left_col = f"{metric}_{i}"
            right_col = f"{metric}_{i+7}"
            left_vals = df[left_col].values
            right_vals = df[right_col].values
            
            fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
            box_data = [left_vals, right_vals]
            bp = ax.boxplot(
                box_data,
                positions=[1, 2],
                widths=0.45,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=7),
                medianprops=dict(color='black', linewidth=2.0),
                boxprops=dict(linewidth=1.5, edgecolor='black'),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3.5, markerfacecolor='gray', markeredgecolor='none', alpha=0.3)
            )
            bp['boxes'][0].set_facecolor("#5D9CEC") # Left
            bp['boxes'][1].set_facecolor("#ED5565") # Right
            
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.5)
            ax.grid(False)
            
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(which='major', direction='out', length=6, width=1.5, color='black', labelsize=12)
            ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Left Hippocampus', 'Right Hippocampus'], fontsize=13, fontweight='bold')
            ax.tick_params(axis='x', direction='out', length=6, width=1.5, color='black', pad=10)
            
            ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
            ax.set_title(f"{m_name} - Bootstrap {metric} Distribution", fontsize=15, fontweight='bold', pad=15)
            ax.set_ylim(y_lim)
            ax.set_xlim(0.4, 2.6)
            plt.tight_layout(pad=2.0)
            
            fn = f"boxplot_{metric.lower()}_{s_name}.png"
            for out_d in dirs_individual:
                plt.savefig(os.path.join(out_d, fn), dpi=300)
            plt.close()
            
    print("[OK] Finished generating 14 Individual Boxplots.")

    # -------------------------------------------------------------
    # 2. GENERATE ALL 21 PAIRS (COMBINATIONS OF 7 MODELS)
    # -------------------------------------------------------------
    all_pairs = list(itertools.combinations(models, 2))
    print(f"\n--- 2. Generating ALL {len(all_pairs)} Model Pairwise Comparisons (Scale: {scale_label}) ---")
    
    for metric in ["AUC", "F1"]:
        y_lim = UNIFIED_YLIM if use_unified_scale else METRIC_YLIMS[metric]
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        
        for pair_idx, (m_a, m_b) in enumerate(all_pairs, start=1):
            idx_a_l = models.index(m_a)
            idx_b_l = models.index(m_b)
            idx_a_r = idx_a_l + 7
            idx_b_r = idx_b_l + 7
            
            vals_a_l = df[f"{metric}_{idx_a_l}"].values
            vals_b_l = df[f"{metric}_{idx_b_l}"].values
            vals_a_r = df[f"{metric}_{idx_a_r}"].values
            vals_b_r = df[f"{metric}_{idx_b_r}"].values
            
            m_a_c = m_a.replace("+", "")
            m_b_c = m_b.replace("+", "")
            
            # --- A. Bilateral Plot (Left Subplot: 2 boxes, Right Subplot: 2 boxes) ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True, dpi=300)
            fig.suptitle(f"Pairwise Comparison: {m_a} vs. {m_b} ({metric} Score)",
                         fontsize=16, fontweight='bold', y=0.98)
            
            plot_2box_panel(ax1, vals_a_l, vals_b_l, m_a, m_b,
                            "Left Hippocampus", y_lim, y_label=y_label)
            plot_2box_panel(ax2, vals_a_r, vals_b_r, m_a, m_b,
                            "Right Hippocampus", y_lim, y_label=None)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fn_bi = f"pairwise_{metric.lower()}_pair{pair_idx:02d}_{m_a_c}_vs_{m_b_c}_bilateral.png"
            for out_d in dirs_pairwise_bilateral:
                plt.savefig(os.path.join(out_d, fn_bi), dpi=300)
            plt.close()
            
            # --- B. Standalone Left Subplot (Only 1 panel, strictly 2 boxes) ---
            fig, ax = plt.subplots(figsize=(5.5, 7), dpi=300)
            plot_2box_panel(ax, vals_a_l, vals_b_l, m_a, m_b,
                            f"{m_a} vs. {m_b} (Left)", y_lim, y_label=y_label)
            plt.tight_layout(pad=2.0)
            fn_l = f"pairwise_{metric.lower()}_pair{pair_idx:02d}_{m_a_c}_vs_{m_b_c}_left.png"
            for out_d in dirs_pairwise_left:
                plt.savefig(os.path.join(out_d, fn_l), dpi=300)
            plt.close()
            
            # --- C. Standalone Right Subplot (Only 1 panel, strictly 2 boxes) ---
            fig, ax = plt.subplots(figsize=(5.5, 7), dpi=300)
            plot_2box_panel(ax, vals_a_r, vals_b_r, m_a, m_b,
                            f"{m_a} vs. {m_b} (Right)", y_lim, y_label=y_label)
            plt.tight_layout(pad=2.0)
            fn_r = f"pairwise_{metric.lower()}_pair{pair_idx:02d}_{m_a_c}_vs_{m_b_c}_right.png"
            for out_d in dirs_pairwise_right:
                plt.savefig(os.path.join(out_d, fn_r), dpi=300)
            plt.close()
            
            if pair_idx % 7 == 0 or pair_idx == 21:
                print(f"[{pair_idx}/21] {metric} pairs generated.")

def main():
    print("=" * 80)
    print("RUNNING MASTER BOOTSTRAP PLOT & 7X7 MATRIX GENERATION")
    print("=" * 80)
    
    # 1. Generate Primary Unified Set (All plots use exact same scale: 0.20 to 1.10)
    print("\n>>> Phase 1: Generating UNIFIED SCALE (0.20 - 1.10) for ALL plots")
    run_pipeline(use_unified_scale=True)
    
    # 2. Also Generate Metric-Specific Set (AUC: 0.50-1.10, F1: 0.20-1.10)
    print("\n>>> Phase 2: Generating METRIC-SPECIFIC SCALE (AUC 0.50-1.10, F1 0.20-1.10)")
    run_pipeline(use_unified_scale=False)
    
    print("\n[SUCCESS] Pipeline completed successfully!")

if __name__ == "__main__":
    main()
