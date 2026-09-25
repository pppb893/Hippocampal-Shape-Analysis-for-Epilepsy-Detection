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
        return p, "***", "*** (p<0.001)"
    elif p < 0.01:
        return p, "**", f"** (p={p:.3f})"
    elif p < 0.05:
        return p, "*", f"* (p={p:.3f})"
    else:
        return p, "ns", f"ns (p={p:.2f})"

def plot_single_pair(ax, df, metric, model_a, model_b, pair_title, y_lim, y_label=None):
    models_list = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    idx_a = models_list.index(model_a)
    idx_b = models_list.index(model_b)
    
    # Left side (idx 0-6)
    left_a = df[f"{metric}_{idx_a}"].values
    left_b = df[f"{metric}_{idx_b}"].values
    p_left, sym_left, text_left = get_p_val_and_symbol(left_a, left_b)
    
    # Right side (idx 7-13)
    right_a = df[f"{metric}_{idx_a+7}"].values
    right_b = df[f"{metric}_{idx_b+7}"].values
    p_right, sym_right, text_right = get_p_val_and_symbol(right_a, right_b)
    
    # Positions: 1, 2 for Left; 3.5, 4.5 for Right
    positions = [1.0, 1.9, 3.2, 4.1]
    box_data = [left_a, left_b, right_a, right_b]
    
    # Color palette
    colors = ["#9B59B6", "#1ABC9C", "#9B59B6", "#1ABC9C"]
    
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6),
        medianprops=dict(color='black', linewidth=1.8),
        boxprops=dict(linewidth=1.4, edgecolor='black'),
        whiskerprops=dict(color='black', linewidth=1.4),
        capprops=dict(color='black', linewidth=1.4),
        flierprops=dict(marker='o', markersize=3, markerfacecolor='gray', markeredgecolor='none', alpha=0.3)
    )
    
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        
    # Significance bracket Left
    y_max_l = max(np.max(left_a), np.max(left_b))
    bracket_y_l = min(y_max_l + 0.04, 1.01)
    bar_h = 0.015
    ax.plot([1.0, 1.0, 1.9, 1.9], [bracket_y_l - bar_h, bracket_y_l, bracket_y_l, bracket_y_l - bar_h],
            lw=1.4, c='black')
    ax.text(1.45, bracket_y_l + 0.007, text_left, ha='center', va='bottom',
            fontsize=12, fontweight='bold', fontfamily='serif')
    
    # Significance bracket Right
    y_max_r = max(np.max(right_a), np.max(right_b))
    bracket_y_r = min(y_max_r + 0.04, 1.01)
    ax.plot([3.2, 3.2, 4.1, 4.1], [bracket_y_r - bar_h, bracket_y_r, bracket_y_r, bracket_y_r - bar_h],
            lw=1.4, c='black')
    ax.text(3.65, bracket_y_r + 0.007, text_right, ha='center', va='bottom',
            fontsize=12, fontweight='bold', fontfamily='serif')
    
    # Side labels at bottom
    ax.text(1.45, y_lim[0] + 0.012, "[Left Hippocampus]", ha='center', va='bottom', fontsize=11, fontfamily='serif')
    ax.text(3.65, y_lim[0] + 0.012, "[Right Hippocampus]", ha='center', va='bottom', fontsize=11, fontfamily='serif')
    
    # Formatting
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
        
    ax.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(which='major', direction='out', length=6, width=1.4, color='black', labelsize=13)
    ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([model_a, model_b, model_a, model_b], fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', direction='out', length=6, width=1.4, color='black')
    
    ax.set_title(pair_title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(y_lim)
    ax.set_xlim(0.4, 4.7)
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    csv_path = os.path.join(repo_root, "Model", "Dataset_1", "Combined_Bootstrap_Results.csv")
    
    out_dirs = [
        os.path.join(repo_root, "Model", "Dataset_1", "plots", "pairwise_2pairs"),
        r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\pairwise_2pairs",
        os.path.join(repo_root, "Model_Results_Excel", "05_Bootstrap_Violin_Plots", "Dataset_1", "pairwise_2pairs")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    
    # Define pairs sets (2 pairs at a time)
    pair_sets = [
        {
            "name": "set1_PointNet_vs_SqueezeNet__PointNet_vs_ResNet",
            "pair1": ("PointNet", "SqueezeNet", "Pair 1: PointNet vs. SqueezeNet"),
            "pair2": ("PointNet", "ResNet", "Pair 2: PointNet vs. ResNet")
        },
        {
            "name": "set2_PointNet_vs_ResNetAE__PointNet_vs_MLP",
            "pair1": ("PointNet", "ResNet+AE", "Pair 1: PointNet vs. ResNet+AE"),
            "pair2": ("PointNet", "MLP", "Pair 2: PointNet vs. MLP")
        },
        {
            "name": "set3_PointNet_vs_MobileNet__PointNet_vs_SVM",
            "pair1": ("PointNet", "MobileNet", "Pair 1: PointNet vs. MobileNet"),
            "pair2": ("PointNet", "SVM", "Pair 2: PointNet vs. SVM")
        },
        {
            "name": "set4_MLP_vs_SVM__ResNet_vs_ResNetAE",
            "pair1": ("MLP", "SVM", "Pair 1: MLP vs. SVM"),
            "pair2": ("ResNet", "ResNet+AE", "Pair 2: ResNet vs. ResNet+AE")
        }
    ]
    
    metrics = ["AUC", "F1"]
    
    generated_files = []
    
    for metric in metrics:
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        y_lim = (0.65, 1.05) if metric == "AUC" else (0.45, 1.02)
        
        for p_set in pair_sets:
            set_name = p_set["name"]
            p1_a, p1_b, p1_title = p_set["pair1"]
            p2_a, p2_b, p2_title = p_set["pair2"]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True, dpi=300)
            fig.suptitle(f"Pairwise Model Comparison (2 Pairs at a Time): {metric} Score",
                         fontsize=16, fontweight='bold', y=0.98)
            
            plot_single_pair(ax1, df, metric, p1_a, p1_b, p1_title, y_lim, y_label=y_label)
            plot_single_pair(ax2, df, metric, p2_a, p2_b, p2_title, y_lim, y_label=None)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            filename = f"pairwise_{metric.lower()}_{set_name}.png"
            for out_dir in out_dirs:
                save_p = os.path.join(out_dir, filename)
                plt.savefig(save_p, dpi=300)
            plt.close()
            generated_files.append(filename)
            print(f"[OK] Generated {filename}")
            
    print("\nTotal pairwise (2 pairs at a time) plots created:", len(generated_files))

if __name__ == "__main__":
    main()
