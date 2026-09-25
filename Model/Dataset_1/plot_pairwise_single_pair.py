import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Set Serif font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['mathtext.fontset'] = 'stix'

def get_p_val_and_symbol(vals_a, vals_b):
    diff = vals_a - vals_b
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    p = max(p, 1 / len(diff))
    if p < 0.001:
        return p, "***", "*** (p < 0.001)"
    elif p < 0.01:
        return p, "**", f"** (p = {p:.3f})"
    elif p < 0.05:
        return p, "*", f"* (p = {p:.3f})"
    else:
        return p, "ns", f"ns (p = {p:.2f})"

def plot_single_pair_panel(ax, vals_a, vals_b, model_a, model_b, title, y_lim, y_label=None, color_a="#5D9CEC", color_b="#48CFAD"):
    """
    Plots strictly 2 boxes (Model 1 vs Model 2) with comfortable headroom and footroom.
    """
    p_val, p_sym, p_text = get_p_val_and_symbol(vals_a, vals_b)
    
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
    
    bp['boxes'][0].set_facecolor(color_a)
    bp['boxes'][1].set_facecolor(color_b)
    
    # Generous bracket spacing: well above highest whisker/data
    y_max = max(np.max(vals_a), np.max(vals_b))
    bracket_y = y_max + 0.035
    bar_h = 0.012
    
    # Draw bracket
    ax.plot([1.0, 1.0, 2.0, 2.0], [bracket_y - bar_h, bracket_y, bracket_y, bracket_y - bar_h],
            lw=1.5, c='black')
    # Text cleanly above bracket line
    ax.text(1.5, bracket_y + 0.012, p_text, ha='center', va='bottom',
            fontsize=13, fontweight='bold', fontfamily='serif')
    
    # Formatting
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
        
    ax.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(which='major', direction='out', length=6, width=1.5, color='black', labelsize=13)
    ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([model_a, model_b], fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', direction='out', length=6, width=1.5, color='black', pad=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(y_lim)
    ax.set_xlim(0.4, 2.6)
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    csv_path = os.path.join(repo_root, "Model", "Dataset_1", "Combined_Bootstrap_Results.csv")
    
    desktop_base = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\pairwise_single_pair"
    repo_base = os.path.join(repo_root, "Model", "Dataset_1", "plots", "pairwise_single_pair")
    excel_base = os.path.join(repo_root, "Model_Results_Excel", "05_Bootstrap_Violin_Plots", "Dataset_1", "pairwise_single_pair")
    
    for d in [desktop_base, repo_base, excel_base]:
        os.makedirs(d, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    models_list = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    
    # 8 Key Pairs of Models (Model 1 vs Model 2)
    pairs = [
        ("PointNet", "SqueezeNet"),
        ("PointNet", "ResNet"),
        ("PointNet", "ResNet+AE"),
        ("PointNet", "MLP"),
        ("PointNet", "MobileNet"),
        ("PointNet", "SVM"),
        ("MLP", "SVM"),
        ("ResNet", "ResNet+AE")
    ]
    
    metrics = ["AUC", "F1"]
    
    for metric in metrics:
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        # y_lim with generous margin at BOTH top (1.10) and bottom (0.55 for AUC, 0.35 for F1)
        y_lim = (0.55, 1.10) if metric == "AUC" else (0.35, 1.10)
        
        for m_a, m_b in pairs:
            idx_a_l = models_list.index(m_a)
            idx_b_l = models_list.index(m_b)
            idx_a_r = idx_a_l + 7
            idx_b_r = idx_b_l + 7
            
            vals_a_l = df[f"{metric}_{idx_a_l}"].values
            vals_b_l = df[f"{metric}_{idx_b_l}"].values
            vals_a_r = df[f"{metric}_{idx_a_r}"].values
            vals_b_r = df[f"{metric}_{idx_b_r}"].values
            
            m_a_c = m_a.replace("+", "")
            m_b_c = m_b.replace("+", "")
            
            # --- Format 1: Bilateral Plot (Left panel [Model 1 vs Model 2], Right panel [Model 1 vs Model 2]) ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6.8), sharey=True, dpi=300)
            fig.suptitle(f"Pairwise Comparison: {m_a} vs. {m_b} ({metric} Score)",
                         fontsize=16, fontweight='bold', y=0.98)
            
            plot_single_pair_panel(ax1, vals_a_l, vals_b_l, m_a, m_b,
                                   "Left Hippocampus", y_lim, y_label=y_label,
                                   color_a="#5D9CEC", color_b="#48CFAD")
            plot_single_pair_panel(ax2, vals_a_r, vals_b_r, m_a, m_b,
                                   "Right Hippocampus", y_lim, y_label=None,
                                   color_a="#5D9CEC", color_b="#48CFAD")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fn_bilateral = f"pairwise_{metric.lower()}_bilateral_{m_a_c}_vs_{m_b_c}.png"
            for out_dir in [desktop_base, repo_base, excel_base]:
                plt.savefig(os.path.join(out_dir, fn_bilateral), dpi=300)
            plt.close()
            
            # --- Format 2: Left Hippocampus Standalone (Only 1 panel, strictly 2 boxes: Model 1 vs Model 2) ---
            fig, ax = plt.subplots(figsize=(5.5, 6.8), dpi=300)
            plot_single_pair_panel(ax, vals_a_l, vals_b_l, m_a, m_b,
                                   f"{m_a} vs. {m_b} (Left Hippocampus)", y_lim, y_label=y_label,
                                   color_a="#5D9CEC", color_b="#48CFAD")
            plt.tight_layout(pad=2.0)
            fn_left = f"pairwise_{metric.lower()}_left_{m_a_c}_vs_{m_b_c}.png"
            for out_dir in [desktop_base, repo_base, excel_base]:
                plt.savefig(os.path.join(out_dir, fn_left), dpi=300)
            plt.close()
            
            # --- Format 3: Right Hippocampus Standalone (Only 1 panel, strictly 2 boxes: Model 1 vs Model 2) ---
            fig, ax = plt.subplots(figsize=(5.5, 6.8), dpi=300)
            plot_single_pair_panel(ax, vals_a_r, vals_b_r, m_a, m_b,
                                   f"{m_a} vs. {m_b} (Right Hippocampus)", y_lim, y_label=y_label,
                                   color_a="#5D9CEC", color_b="#48CFAD")
            plt.tight_layout(pad=2.0)
            fn_right = f"pairwise_{metric.lower()}_right_{m_a_c}_vs_{m_b_c}.png"
            for out_dir in [desktop_base, repo_base, excel_base]:
                plt.savefig(os.path.join(out_dir, fn_right), dpi=300)
            plt.close()
            
            print(f"[OK] Generated {m_a} vs {m_b} ({metric}): Bilateral, Left, and Right plots.")

if __name__ == "__main__":
    main()
