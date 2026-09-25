import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    csv_path = os.path.join(repo_root, "Model", "Dataset_1", "Combined_Bootstrap_Results.csv")
    
    out_dirs = [
        os.path.join(repo_root, "Model", "Dataset_1", "plots"),
        r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots",
        os.path.join(repo_root, "Model_Results_Excel", "05_Bootstrap_Violin_Plots", "Dataset_1")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    models = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]
    metrics = ["AUC", "F1"]
    
    left_color = "#5D9CEC"
    right_color = "#ED5565"
    
    for metric in metrics:
        y_label = "ROC-AUC Score" if metric == "AUC" else "F1 Score"
        y_lim = (0.60, 1.00) if metric == "AUC" else (0.45, 1.00)
        
        fig, axes = plt.subplots(1, 7, figsize=(24, 5.5), sharey=True, dpi=300)
        fig.suptitle(f"Bilateral Bootstrap Distribution per Model: {metric} Score (N=1,000 Iterations)",
                     fontsize=16, fontweight='bold', y=0.98)
        
        for i, (m_name, ax) in enumerate(zip(models, axes)):
            left_col = f"{metric}_{i}"
            right_col = f"{metric}_{i+7}"
            
            left_vals = df[left_col].values
            right_vals = df[right_col].values
            
            box_data = [left_vals, right_vals]
            bp = ax.boxplot(
                box_data,
                positions=[1, 2],
                widths=0.5,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6),
                medianprops=dict(color='black', linewidth=1.8),
                boxprops=dict(linewidth=1.4, edgecolor='black'),
                whiskerprops=dict(color='black', linewidth=1.4),
                capprops=dict(color='black', linewidth=1.4),
                flierprops=dict(marker='o', markersize=3, markerfacecolor='gray', markeredgecolor='none', alpha=0.3)
            )
            bp['boxes'][0].set_facecolor(left_color)
            bp['boxes'][1].set_facecolor(right_color)
            
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.4)
                
            ax.grid(False)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(which='major', direction='out', length=5, width=1.3, color='black')
            ax.tick_params(which='minor', direction='out', length=3, width=0.9, color='black')
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Left', 'Right'], fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', direction='out', length=5, width=1.3, color='black')
            ax.set_title(m_name, fontsize=14, fontweight='bold')
            ax.set_ylim(y_lim)
            ax.set_xlim(0.4, 2.6)
            
            if i == 0:
                ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"bootstrap_grid_7models_{metric.lower()}.png"
        for out_dir in out_dirs:
            save_p = os.path.join(out_dir, filename)
            plt.savefig(save_p, dpi=300)
        plt.close()
        print(f"[OK] Generated {filename} (Clean, NO p-value)")

if __name__ == "__main__":
    main()
