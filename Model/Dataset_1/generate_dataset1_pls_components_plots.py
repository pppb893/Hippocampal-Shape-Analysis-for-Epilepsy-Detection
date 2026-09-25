"""
================================================================================
Publication-Quality PLS-DA Components Comparison Plots (10 Points, Legend Lower Right)
================================================================================
Specifications:
1. Strict Publication Styling:
   - Font: Serif ('Times New Roman', 14pt base)
   - Frame: Solid black 1.5pt spines (top, bottom, left, right)
   - Grid: NO internal grid (grid=False)
   - Ticks: Outward ticks on both axes (major & minor)
   - Header: NO word "Dataset" in title (e.g. [Right Hippocampus], [Left Hippocampus])
   - Optimal Component Highlight: Crimson marker with black border
   - Legend Location: Lower Right (loc='lower right') so it never obstructs data points
   - 10 Data Points: [2, 5, 10, 15, 20, 25, 30, 40, 50, 100] (exactly 10 evaluated points)
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Publication Styling Configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'

def format_spines_and_ticks(ax, x_max=105, y_min=0.88, y_max=1.008):
    """Applies strict publication black spines and outward ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.grid(False)
    
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
        
    y_range = y_max - y_min
    if y_range <= 0.06:
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(MultipleLocator(0.002))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.tick_params(which='major', direction='out', length=6, width=1.5, color='black', labelsize=12)
    ax.tick_params(which='minor', direction='out', length=3.5, width=1.0, color='black')
    
    ax.set_xlabel('Number of Components', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Mean CV Accuracy', fontsize=14, fontweight='bold', labelpad=10)

def plot_pls_curve(components, accuracies, best_n, best_acc, side, out_paths, cv_type="10-Fold"):
    """
    Renders and saves a publication-grade PLS-DA component comparison curve with 10 points.
    Legend is positioned at lower right (loc='lower right').
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=300)
    
    # 1. Main line & points (10 points total)
    ax.plot(components, accuracies, color='#1F77B4', lw=2.4, marker='o',
            markersize=7.0, markerfacecolor='#1F77B4', markeredgecolor='white',
            markeredgewidth=1.2, zorder=3, label=f'{cv_type} CV Accuracy')
    
    # 2. Optimal point highlight
    ax.scatter([best_n], [best_acc], color='#E74C3C', s=120, zorder=5,
               edgecolor='black', linewidth=1.3,
               label=f'Optimal: {best_n} Components ({best_acc*100:.2f}%)')
    
    # Set generous limits so curve and points do not collide with spines
    ax.set_xlim(-2, 105)
    
    y_min = min(accuracies)
    y_max = max(accuracies)
    plot_y_max = min(1.008, y_max + 0.009)
    plot_y_min = max(0.88, y_min - 0.012)
    ax.set_ylim(plot_y_min, plot_y_max)
    
    format_spines_and_ticks(ax, x_max=105, y_min=plot_y_min, y_max=plot_y_max)
    
    # Title without the word "Dataset"
    ax.set_title(
        f"PLS-DA Cross-Validation Accuracy vs. Number of Components\n[{side.capitalize()} Hippocampus]",
        fontsize=15, fontweight='bold', pad=14
    )
    
    # Legend explicitly placed at LOWER RIGHT to avoid covering any points
    leg = ax.legend(loc='lower right', frameon=True, framealpha=0.96, edgecolor='black', fontsize=12)
    leg.get_frame().set_linewidth(1.2)
    
    plt.tight_layout()
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"[OK] Saved: {p}")
    plt.close()

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    
    # Output target directories
    ds1_plots_dir = os.path.join(repo_root, "Model", "Dataset_1", "plots", "pls_components")
    excel_plots_dir = os.path.join(repo_root, "Model_Results_Excel", "07_Detailed_Model_Plots", "Dataset_1", "pls_components")
    desktop_plots_dir = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\pls_components"
    
    # Exactly 10 points evaluated with 10-Fold CV on Dataset 1
    ds1_data_10pts = {
        "right": {
            "components": [2, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            "accuracies": [0.9220, 0.9646, 0.9793, 0.9720, 0.9659, 0.9634, 0.9634, 0.9659, 0.9634, 0.9659],
            "best_n": 10,
            "best_acc": 0.9793,
            "cv_type": "10-Fold"
        },
        "left": {
            "components": [2, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            "accuracies": [0.9000, 0.9558, 0.9788, 0.9769, 0.9731, 0.9731, 0.9712, 0.9692, 0.9673, 0.9663],
            "best_n": 10,
            "best_acc": 0.9788,
            "cv_type": "10-Fold"
        }
    }
    
    for side, d in ds1_data_10pts.items():
        targets = [
            os.path.join(ds1_plots_dir, f"pls_components_comparison_{side}.png"),
            os.path.join(excel_plots_dir, f"pls_components_comparison_{side}.png"),
            os.path.join(desktop_plots_dir, f"pls_components_comparison_{side}.png"),
            os.path.join(repo_root, "Model", "Dataset_1", side, "plots", "pls_components_comparison.png"),
        ]
        plot_pls_curve(d["components"], d["accuracies"], d["best_n"], d["best_acc"], side, targets, d["cv_type"])
        
    # Also update All_Augment_tain with legend at lower right and complete points
    all_augment_data = {
        "right": {
            "components": [2, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            "accuracies": [0.9500, 0.9983, 0.9983, 0.9954, 0.9908, 0.9908, 0.9908, 0.9908, 0.9901, 0.9314],
            "best_n": 5,
            "best_acc": 0.9983,
            "cv_type": "5-Fold"
        },
        "left": {
            "components": [2, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            "accuracies": [0.9204, 0.9844, 0.9899, 0.9844, 0.9866, 0.9869, 0.9872, 0.9866, 0.9866, 0.9331],
            "best_n": 10,
            "best_acc": 0.9899,
            "cv_type": "5-Fold"
        }
    }
    
    for side, d in all_augment_data.items():
        targets = [
            os.path.join(ds1_plots_dir, f"pls_components_comparison_all_augment_{side}.png"),
            os.path.join(excel_plots_dir, f"pls_components_comparison_all_augment_{side}.png"),
            os.path.join(desktop_plots_dir, f"pls_components_comparison_all_augment_{side}.png"),
            os.path.join(repo_root, "Model_Results_Excel", "07_Detailed_Model_Plots", "All_Augment_tain", side, "SVM", "pls_components_comparison.png"),
        ]
        for m in ["SVM", "MLP", "MobileNet", "ResNet", "SqueezeNet"]:
            p_local = os.path.join(repo_root, "Model", "All_Augment_tain", side, m, "plots", "pls_components_comparison.png")
            targets.append(p_local)
            
        plot_pls_curve(d["components"], d["accuracies"], d["best_n"], d["best_acc"], side, targets, d["cv_type"])

    print("\n[SUCCESS] All plots successfully updated with 10 points and Legend placed at Lower Right!")

if __name__ == "__main__":
    main()
