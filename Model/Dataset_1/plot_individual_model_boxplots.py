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

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    csv_path = os.path.join(repo_root, "Model", "Dataset_1", "Combined_Bootstrap_Results.csv")
    
    out_dirs = [
        os.path.join(repo_root, "Model", "Dataset_1", "plots", "individual_models"),
        r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\individual_models",
        os.path.join(repo_root, "Model_Results_Excel", "05_Bootstrap_Violin_Plots", "Dataset_1", "individual_models")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    
    models = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    safe_names = ["MLP", "MobileNet", "PointNet", "ResNet_AE", "ResNet", "SqueezeNet", "SVM"]
    metrics = ["AUC", "F1"]
    
    # Colors for Left and Right Hippocampus
    left_color = "#5D9CEC"   # Professional Soft Blue
    right_color = "#ED5565"  # Professional Coral Red
    
    for metric in metrics:
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        # y_lim with generous margin at BOTH top and bottom!
        # AUC: 0.55 to 1.05 (whisker ~0.65 min, ~1.00 max)
        # F1: 0.35 to 1.05 (whisker ~0.46 min, ~0.95 max)
        y_lim = (0.55, 1.05) if metric == "AUC" else (0.35, 1.05)
        
        for i, (m_name, s_name) in enumerate(zip(models, safe_names)):
            left_col = f"{metric}_{i}"
            right_col = f"{metric}_{i+7}"
            
            left_vals = df[left_col].values
            right_vals = df[right_col].values
            
            # Figure with extra vertical height to give bottom labels comfortable breathing room
            fig, ax = plt.subplots(figsize=(6, 6.8), dpi=300)
            
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
            
            # Color boxes
            bp['boxes'][0].set_facecolor(left_color)
            bp['boxes'][1].set_facecolor(right_color)
            
            # Formatting: Black spines, outward ticks, no grid
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.5)
                
            ax.grid(False)
            
            # Ticks: major every 0.05, minor every 0.01
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(which='major', direction='out', length=6, width=1.5, color='black', labelsize=13)
            ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Left Hippocampus', 'Right Hippocampus'], fontsize=13, fontweight='bold')
            # Extra pad for x-axis tick labels so they don't crowd the bottom frame
            ax.tick_params(axis='x', direction='out', length=6, width=1.5, color='black', pad=8)
            
            ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
            ax.set_title(f"{m_name} - Bootstrap {metric} Distribution", fontsize=15, fontweight='bold', pad=15)
            ax.set_ylim(y_lim)
            ax.set_xlim(0.4, 2.6)
            
            # Leave generous margin around borders
            plt.tight_layout(pad=2.0)
            
            filename = f"boxplot_{metric.lower()}_{s_name}.png"
            for out_dir in out_dirs:
                save_p = os.path.join(out_dir, filename)
                plt.savefig(save_p, dpi=300)
            
            plt.close()
            print(f"[OK] Generated {filename} (y_lim={y_lim})")

if __name__ == "__main__":
    main()
