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
plt.rcParams['axes.titlesize'] = 14
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

def plot_2box_pair(ax, vals_a, vals_b, model_a, model_b, title, y_lim, y_label=None, color_a="#9B59B6", color_b="#1ABC9C"):
    """
    Plots EXACTLY 2 boxes (a pair / คู่) with generous headroom for the p-value bracket.
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
    
    # Draw bracket with ample space
    ax.plot([1.0, 1.0, 2.0, 2.0], [bracket_y - bar_h, bracket_y, bracket_y, bracket_y - bar_h],
            lw=1.5, c='black')
    # Text placed clearly above bracket line
    ax.text(1.5, bracket_y + 0.012, p_text, ha='center', va='bottom',
            fontsize=13, fontweight='bold', fontfamily='serif')
    
    # Frame formatting
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
    ax.tick_params(axis='x', direction='out', length=6, width=1.5, color='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(y_lim)
    ax.set_xlim(0.4, 2.6)
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    csv_path = os.path.join(repo_root, "Model", "Dataset_1", "Combined_Bootstrap_Results.csv")
    
    desktop_base = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\pairwise_2pairs"
    repo_base = os.path.join(repo_root, "Model", "Dataset_1", "plots", "pairwise_2pairs")
    excel_base = os.path.join(repo_root, "Model_Results_Excel", "05_Bootstrap_Violin_Plots", "Dataset_1", "pairwise_2pairs")
    
    for d in [desktop_base, repo_base, excel_base]:
        os.makedirs(d, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    models_list = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    
    # 4 Key Pair Sets to compare 2 pairs at a time
    pair_sets = [
        {
            "id": "set1",
            "pair1": ("PointNet", "SqueezeNet"),
            "pair2": ("PointNet", "ResNet")
        },
        {
            "id": "set2",
            "pair1": ("PointNet", "ResNet+AE"),
            "pair2": ("PointNet", "MLP")
        },
        {
            "id": "set3",
            "pair1": ("PointNet", "MobileNet"),
            "pair2": ("PointNet", "SVM")
        },
        {
            "id": "set4",
            "pair1": ("MLP", "SVM"),
            "pair2": ("ResNet", "ResNet+AE")
        }
    ]
    
    metrics = ["AUC", "F1"]
    
    # Generation A: 2 Pairs per figure (Side-by-side subplots, each subplot has ONLY 2 BOXES!)
    # Separated by Side: Left Hippocampus and Right Hippocampus
    for metric in metrics:
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        # y_lim up to 1.10 so bracket, text, and whiskers NEVER touch the top frame!
        y_lim = (0.60, 1.10) if metric == "AUC" else (0.45, 1.10)
        
        for side in ["Left", "Right"]:
            side_offset = 0 if side == "Left" else 7
            side_title = f"{side} Hippocampus"
            
            for p_set in pair_sets:
                set_id = p_set["id"]
                p1_a, p1_b = p_set["pair1"]
                p2_a, p2_b = p_set["pair2"]
                
                # Fetch data
                idx1_a = models_list.index(p1_a) + side_offset
                idx1_b = models_list.index(p1_b) + side_offset
                idx2_a = models_list.index(p2_a) + side_offset
                idx2_b = models_list.index(p2_b) + side_offset
                
                vals1_a = df[f"{metric}_{idx1_a}"].values
                vals1_b = df[f"{metric}_{idx1_b}"].values
                vals2_a = df[f"{metric}_{idx2_a}"].values
                vals2_b = df[f"{metric}_{idx2_b}"].values
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True, dpi=300)
                fig.suptitle(f"Pairwise Model Comparison ({side_title}): {metric} Score",
                             fontsize=16, fontweight='bold', y=0.98)
                
                # Subplot 1: Pair 1 (EXACTLY 2 BOXES)
                plot_2box_pair(ax1, vals1_a, vals1_b, p1_a, p1_b,
                               f"Pair 1: {p1_a} vs. {p1_b}", y_lim, y_label=y_label,
                               color_a="#9B59B6", color_b="#1ABC9C")
                
                # Subplot 2: Pair 2 (EXACTLY 2 BOXES)
                plot_2box_pair(ax2, vals2_a, vals2_b, p2_a, p2_b,
                               f"Pair 2: {p2_a} vs. {p2_b}", y_lim, y_label=None,
                               color_a="#3498DB", color_b="#E67E22")
                
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                
                p1_clean = f"{p1_a}_vs_{p1_b}".replace("+", "")
                p2_clean = f"{p2_a}_vs_{p2_b}".replace("+", "")
                filename = f"pairwise_{metric.lower()}_{side.lower()}_{set_id}_{p1_clean}__{p2_clean}.png"
                
                for out_dir in [desktop_base, repo_base, excel_base]:
                    save_p = os.path.join(out_dir, filename)
                    plt.savefig(save_p, dpi=300)
                plt.close()
                print(f"[OK] Generated {filename}")
                
    # Generation B: 1 Pair per figure comparing Left vs Right (Subplot 1: Left [2 boxes], Subplot 2: Right [2 boxes])
    single_pairs = [
        ("PointNet", "SqueezeNet"),
        ("PointNet", "ResNet"),
        ("PointNet", "ResNet+AE"),
        ("PointNet", "MLP"),
        ("PointNet", "MobileNet"),
        ("PointNet", "SVM"),
        ("MLP", "SVM"),
        ("ResNet", "ResNet+AE")
    ]
    
    for metric in metrics:
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        y_lim = (0.60, 1.10) if metric == "AUC" else (0.45, 1.10)
        
        for m_a, m_b in single_pairs:
            idx_a_l = models_list.index(m_a)
            idx_b_l = models_list.index(m_b)
            idx_a_r = idx_a_l + 7
            idx_b_r = idx_b_l + 7
            
            vals_a_l = df[f"{metric}_{idx_a_l}"].values
            vals_b_l = df[f"{metric}_{idx_b_l}"].values
            vals_a_r = df[f"{metric}_{idx_a_r}"].values
            vals_b_r = df[f"{metric}_{idx_b_r}"].values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True, dpi=300)
            fig.suptitle(f"Pairwise Comparison: {m_a} vs. {m_b} ({metric} Score)",
                         fontsize=16, fontweight='bold', y=0.98)
            
            # Left Subplot: EXACTLY 2 BOXES
            plot_2box_pair(ax1, vals_a_l, vals_b_l, m_a, m_b,
                           f"Left Hippocampus", y_lim, y_label=y_label,
                           color_a="#5D9CEC", color_b="#48CFAD")
            
            # Right Subplot: EXACTLY 2 BOXES
            plot_2box_pair(ax2, vals_a_r, vals_b_r, m_a, m_b,
                           f"Right Hippocampus", y_lim, y_label=None,
                           color_a="#5D9CEC", color_b="#48CFAD")
            
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            
            m_a_c = m_a.replace("+", "")
            m_b_c = m_b.replace("+", "")
            filename = f"pairwise_{metric.lower()}_bilateral_{m_a_c}_vs_{m_b_c}.png"
            
            for out_dir in [desktop_base, repo_base, excel_base]:
                save_p = os.path.join(out_dir, filename)
                plt.savefig(save_p, dpi=300)
            plt.close()
            print(f"[OK] Generated {filename}")

if __name__ == "__main__":
    main()
