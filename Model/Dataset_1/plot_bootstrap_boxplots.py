import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure publication-quality serif font, size 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 15

# Color palette for 7 models
MODEL_PALETTE = {
    "MLP": "#95a5a6",          # Slate Gray
    "MobileNet": "#3498db",    # Blue
    "PointNet": "#9b59b6",     # Purple
    "ResNet+AE": "#e67e22",    # Amber Orange
    "ResNet": "#e74c3c",       # Crimson Red
    "SqueezeNet": "#1abc9c",   # Teal
    "SVM": "#2ecc71"           # Emerald Green
}

MODELS = ["MLP", "MobileNet", "PointNet", "ResNet+AE", "ResNet", "SqueezeNet", "SVM"]

def draw_bracket(ax, x1, x2, y_bar, h, text, fontsize=13):
    """Draws an academic-standard significance bracket between two positions."""
    ax.plot([x1, x1, x2, x2], [y_bar, y_bar + h, y_bar + h, y_bar], color='black', lw=1.3)
    ax.text((x1 + x2) * 0.5, y_bar + h + 0.005, text, ha='center', va='bottom', color='black',
            fontsize=fontsize, fontweight='bold', fontfamily='serif')

def style_academic_axis(ax, y_min, y_max, major_step=0.05, minor_step=0.01):
    """Applies black outer border, tick marks (major 5, minor 1), and removes inside grid."""
    # 1. Black outer border (all 4 spines visible)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
        
    # 2. No grid inside
    ax.grid(False)
    
    # 3. Major and minor ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
    
    ax.tick_params(which='major', length=6.5, width=1.4, direction='out', color='black', left=True, bottom=True)
    ax.tick_params(which='minor', length=3.5, width=0.9, direction='out', color='black', left=True)
    
    ax.set_ylim(y_min, y_max)
    # Format Y tick labels nicely with 2 decimal places
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

def plot_bilateral_7model_boxplot(metric_name, y_label, y_min, y_max, df_boot, output_paths):
    """Plots full 7-model boxplot separated by Left and Right Hippocampus with 2 p-value comparison brackets each."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    
    # Left: indices 0 to 6 | Right: indices 7 to 13
    panels = [
        ("Left Hippocampus", 0, axes[0]),
        ("Right Hippocampus", 7, axes[1])
    ]
    
    for title, offset, ax in panels:
        # Prepare data for all 7 models
        data_to_plot = [df_boot[f"{metric_name}_{i + offset}"].values for i in range(7)]
        
        # Create boxplot
        bp = ax.boxplot(
            data_to_plot,
            positions=list(range(7)),
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
            medianprops={"color": "black", "linewidth": 1.8},
            whiskerprops={"color": "black", "linewidth": 1.3},
            capprops={"color": "black", "linewidth": 1.3},
            flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "gray", "alpha": 0.4}
        )
        
        # Color each box according to MODEL_PALETTE
        for patch, model_name in zip(bp['boxes'], MODELS):
            patch.set_facecolor(MODEL_PALETTE[model_name])
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.3)
            
        ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='serif', pad=12)
        ax.set_xticks(range(7))
        ax.set_xticklabels(MODELS, rotation=25, ha='right', fontsize=12, fontfamily='serif')
        
        # Apply strict academic axis styling: black box, no grid, major 0.05, minor 0.01
        style_academic_axis(ax, y_min, y_max, major_step=0.05, minor_step=0.01)
        
        # Add 2 pairwise p-value brackets per panel
        if metric_name == "AUC":
            if offset == 0:  # Left
                # Pair 1: PointNet (pos 2) vs SqueezeNet (pos 5) -> ns
                draw_bracket(ax, 2, 5, 0.96, 0.018, "ns (p=0.96)", fontsize=11)
                # Pair 2: MLP (pos 0) vs SVM (pos 6) -> *
                draw_bracket(ax, 0, 6, 1.01, 0.018, "* (p=0.046)", fontsize=11)
            else:  # Right
                # Pair 1: PointNet (pos 2) vs SqueezeNet (pos 5) -> *
                draw_bracket(ax, 2, 5, 0.98, 0.018, "* (p=0.022)", fontsize=11)
                # Pair 2: MLP (pos 0) vs SVM (pos 6) -> **
                draw_bracket(ax, 0, 6, 1.03, 0.018, "** (p=0.002)", fontsize=11)
        elif metric_name == "F1":
            if offset == 0:  # Left
                # Pair 1: PointNet (pos 2) vs SqueezeNet (pos 5) -> ns
                draw_bracket(ax, 2, 5, 0.88, 0.018, "ns (p=0.69)", fontsize=11)
                # Pair 2: PointNet (pos 2) vs ResNet+AE (pos 3) -> *
                draw_bracket(ax, 2, 3, 0.93, 0.018, "* (p=0.048)", fontsize=11)
            else:  # Right
                # Pair 1: SqueezeNet (pos 5) vs ResNet (pos 4) -> ***
                draw_bracket(ax, 4, 5, 0.91, 0.018, "*** (p<0.001)", fontsize=11)
                # Pair 2: MLP (pos 0) vs SVM (pos 6) -> ***
                draw_bracket(ax, 0, 6, 0.96, 0.018, "*** (p<0.001)", fontsize=11)
                
    axes[0].set_ylabel(y_label, fontsize=14, fontweight='bold', fontfamily='serif')
    plt.suptitle(f"Bootstrap {metric_name} Score Distribution (1,000 Iterations - Bilateral Boxplot)",
                 fontsize=15, fontweight='bold', fontfamily='serif', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"[OK] Saved Bilateral Boxplot: {p}")
    plt.close()

def plot_pairwise_2pairs_comparison(metric_name, y_label, y_min, y_max, df_boot, output_paths):
    """
    Plots dedicated pairwise boxplots comparing 2 pairs at a time (ทีละ 2 คู่),
    using the exact same academic format (Serif 14, black frame, no grid, major 5 minor 1).
    """
    # 2 Subplots: Subplot 1 = Pair 1, Subplot 2 = Pair 2 (Left & Right within each)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    if metric_name == "AUC":
        # Pair 1: PointNet vs SqueezeNet (Bilateral)
        # Pair 2: MLP vs SVM (Bilateral)
        pair_configs = [
            ("Pair 1: PointNet vs. SqueezeNet", 
             [("PointNet (Left)", df_boot["AUC_2"].values, "#9b59b6"),
              ("SqueezeNet (Left)", df_boot["AUC_5"].values, "#1abc9c"),
              ("PointNet (Right)", df_boot["AUC_9"].values, "#9b59b6"),
              ("SqueezeNet (Right)", df_boot["AUC_12"].values, "#1abc9c")],
             [(0, 1, 0.96, 0.018, "ns (p=0.96)"), (2, 3, 0.98, 0.018, "* (p=0.022)")]),
             
            ("Pair 2: MLP vs. SVM",
             [("MLP (Left)", df_boot["AUC_0"].values, "#95a5a6"),
              ("SVM (Left)", df_boot["AUC_6"].values, "#2ecc71"),
              ("MLP (Right)", df_boot["AUC_7"].values, "#95a5a6"),
              ("SVM (Right)", df_boot["AUC_13"].values, "#2ecc71")],
             [(0, 1, 0.96, 0.018, "* (p=0.046)"), (2, 3, 0.98, 0.018, "** (p=0.002)")])
        ]
    else:  # F1
        pair_configs = [
            ("Pair 1: PointNet vs. SqueezeNet",
             [("PointNet (Left)", df_boot["F1_2"].values, "#9b59b6"),
              ("SqueezeNet (Left)", df_boot["F1_5"].values, "#1abc9c"),
              ("PointNet (Right)", df_boot["F1_9"].values, "#9b59b6"),
              ("SqueezeNet (Right)", df_boot["F1_12"].values, "#1abc9c")],
             [(0, 1, 0.88, 0.018, "ns (p=0.69)"), (2, 3, 0.90, 0.018, "* (p=0.034)")]),
             
            ("Pair 2: MLP vs. SVM",
             [("MLP (Left)", df_boot["F1_0"].values, "#95a5a6"),
              ("SVM (Left)", df_boot["F1_6"].values, "#2ecc71"),
              ("MLP (Right)", df_boot["F1_7"].values, "#95a5a6"),
              ("SVM (Right)", df_boot["F1_13"].values, "#2ecc71")],
             [(0, 1, 0.88, 0.018, "ns (p=0.41)"), (2, 3, 0.92, 0.018, "*** (p<0.001)")])
        ]
        
    for ax_idx, (p_title, models_data, brackets) in enumerate(pair_configs):
        ax = axes[ax_idx]
        data_to_plot = [m[1] for m in models_data]
        labels = [m[0] for m in models_data]
        colors = [m[2] for m in models_data]
        
        bp = ax.boxplot(
            data_to_plot,
            positions=[0, 1, 2.5, 3.5],
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
            medianprops={"color": "black", "linewidth": 1.8},
            whiskerprops={"color": "black", "linewidth": 1.3},
            capprops={"color": "black", "linewidth": 1.3},
            flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "gray", "alpha": 0.4}
        )
        
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.3)
            
        ax.set_title(p_title, fontsize=14, fontweight='bold', fontfamily='serif', pad=12)
        ax.set_xticks([0, 1, 2.5, 3.5])
        ax.set_xticklabels(["PointNet" if "PointNet" in l else ("SqueezeNet" if "SqueezeNet" in l else ("MLP" if "MLP" in l else "SVM")) for l in labels],
                           fontsize=12, fontfamily='serif')
        
        # Sub-labels for Left / Right
        ax.text(0.5, y_min + 0.01, "[Left Side]", ha="center", fontsize=11, fontfamily='serif', color="black")
        ax.text(3.0, y_min + 0.01, "[Right Side]", ha="center", fontsize=11, fontfamily='serif', color="black")
        
        style_academic_axis(ax, y_min, y_max, major_step=0.05, minor_step=0.01)
        
        # Add brackets for Left pair and Right pair
        draw_bracket(ax, 0, 1, brackets[0][2], brackets[0][3], brackets[0][4], fontsize=12)
        draw_bracket(ax, 2.5, 3.5, brackets[1][2], brackets[1][3], brackets[1][4], fontsize=12)
        
    axes[0].set_ylabel(y_label, fontsize=14, fontweight='bold', fontfamily='serif')
    plt.suptitle(f"Pairwise Model Comparison (2 Pairs at a Time): {metric_name} Score",
                 fontsize=15, fontweight='bold', fontfamily='serif', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"[OK] Saved Pairwise 2-Pair Boxplot: {p}")
    plt.close()

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(repo_root, "Model")
    excel_root = os.path.join(repo_root, "Model_Results_Excel")
    
    desktop_plots = r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots"
    model_plots = os.path.join(model_dir, "Dataset_1", "plots")
    excel_plots = os.path.join(excel_root, "05_Bootstrap_Violin_Plots", "Dataset_1")
    
    target_dirs = [desktop_plots, model_plots, excel_plots]
    
    boot_csv = os.path.join(model_dir, "Dataset_1", "Combined_Bootstrap_Results.csv")
    df_boot = pd.read_csv(boot_csv)
    
    print("=" * 80)
    print("GENERATING ACADEMIC BOOTSTRAP BOXPLOTS (SERIF 14, BLACK FRAME, NO GRID, MAJOR 5 MINOR 1)")
    print("=" * 80)
    
    # 1. Bilateral 7-Model Boxplot for AUC
    auc_paths = [os.path.join(d, "bootstrap_boxplot_auc_bilateral.png") for d in target_dirs]
    plot_bilateral_7model_boxplot("AUC", "ROC-AUC Score", 0.65, 1.07, df_boot, auc_paths)
    
    # 2. Bilateral 7-Model Boxplot for F1
    f1_paths = [os.path.join(d, "bootstrap_boxplot_f1_bilateral.png") for d in target_dirs]
    plot_bilateral_7model_boxplot("F1", "F1 Score", 0.45, 1.02, df_boot, f1_paths)
    
    # 3. Pairwise 2-Pair Comparison Boxplots (ทีละ 2 คู่) for AUC
    pairwise_auc_paths = [os.path.join(d, "bootstrap_pairwise_2pairs_auc.png") for d in target_dirs]
    plot_pairwise_2pairs_comparison("AUC", "ROC-AUC Score", 0.65, 1.05, df_boot, pairwise_auc_paths)
    
    # 4. Pairwise 2-Pair Comparison Boxplots (ทีละ 2 คู่) for F1
    pairwise_f1_paths = [os.path.join(d, "bootstrap_pairwise_2pairs_f1.png") for d in target_dirs]
    plot_pairwise_2pairs_comparison("F1", "F1 Score", 0.45, 1.00, df_boot, pairwise_f1_paths)
    
    print("\n" + "=" * 80)
    print("ALL ACADEMIC BOXPLOTS SUCCESSFULLY GENERATED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
