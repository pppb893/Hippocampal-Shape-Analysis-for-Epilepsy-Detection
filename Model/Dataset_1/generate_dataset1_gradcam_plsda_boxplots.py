"""
================================================================================
Publication-Quality Top 3 PLS-DA Components Box Plot for Dataset 1
================================================================================
Specifications:
1. Strict Publication Styling:
   - Font: Serif ('Times New Roman', 14pt base)
   - Frame: Solid black 1.5pt spines (top, bottom, left, right)
   - Grid: NO internal grid (grid=False)
   - Ticks: Outward ticks on both axes
   - Header: NO word "Dataset" in header
2. Plot Characteristics:
   - Box Plot (replacing violin plot)
   - No datapoints / jitter points (do not add datapoint in)
   - Cut effect size badge (cut eff size)
   - Keep p-value bracket & exact statistical significance (keep p value)
   - Color palette: Healthy Control (#3498db), Epilepsy (TLE) (#e74c3c)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

# Publication Serif Configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 15
})

def compute_significance_stars(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"

def generate_top3_boxplot_for_side(side="right", repo_root=None):
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_dir = os.path.join(repo_root, "Model", "Dataset_1", side)
    candidates_tr = [
        os.path.join(data_dir, f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
        os.path.join(data_dir, f"Dataset_1_{side.capitalize()}_train_xyz_coords.csv"),
        os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side.capitalize()}_train_xyz_coords.csv"),
        os.path.join(repo_root, "Model", "All_Augment_tain", side, f"ALL_{side.capitalize()}_train_xyz_coords.csv")
    ]
    train_csv = next((c for c in candidates_tr if os.path.exists(c)), None)
    if not train_csv:
        raise FileNotFoundError(f"Could not find training coordinates for {side} hippocampus in Dataset 1.")

    print(f"[{side.upper()}] Loading training coordinates from: {train_csv}")
    train_df = pd.read_csv(train_csv)

    coord_cols = [c for c in train_df.columns if c.startswith(('x_', 'y_', 'z_'))]
    if not coord_cols:
        meta_cols = ["Subject", "Group", "Group_Name", "Group_Label", "BinaryClass", "Class", "DataType", "Unnamed: 0"]
        coord_cols = [c for c in train_df.columns if c not in meta_cols]
    
    X_train_flat = train_df[coord_cols].values.astype(np.float32)
    y_train = (train_df["Group_Label"].values if "Group_Label" in train_df.columns else train_df["BinaryClass"].values).astype(int)

    class_names = {0: "Healthy Control", 1: "Epilepsy (TLE)"}
    class_palette = {"Healthy Control": "#3498db", "Epilepsy (TLE)": "#e74c3c"}

    # Fit PLS-DA with 8 Components
    n_components = 8
    Y_train_ohe = np.zeros((len(y_train), 2), dtype=np.float64)
    for idx, label in enumerate(y_train):
        Y_train_ohe[idx, label] = 1.0

    pls = PLSRegression(n_components=n_components, scale=True)
    X_scores, _ = pls.fit_transform(X_train_flat, Y_train_ohe)

    # Check summary CSV or compute discriminative ranking
    summary_csv = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Dataset_1", side, "plsda_8_components_summary.csv")
    if not os.path.exists(summary_csv):
        summary_csv = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "All_Augment_tain", side, "plsda_8_components_summary.csv")

    top3_indices = [0, 2, 1]  # Default: PLS1, PLS3, PLS2
    top3_ranks = {0: 1, 2: 2, 1: 3}
    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        if "Is_Top3" in df_sum.columns and "Top3_Rank" in df_sum.columns:
            top3_rows = df_sum[df_sum["Is_Top3"] == True].sort_values("Top3_Rank")
            top3_indices = [int(c.replace("PLS", "")) - 1 for c in top3_rows["Component"].tolist()]
            top3_ranks = {int(row["Component"].replace("PLS", "")) - 1: int(row["Top3_Rank"]) for _, row in top3_rows.iterrows()}
    else:
        # Compute discriminative ranking
        P = pls.x_loadings_
        Q = pls.y_loadings_
        X_std = np.std(X_train_flat, axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_train_flat - np.mean(X_train_flat, axis=0)) / X_std
        total_var_X = np.sum(X_scaled ** 2)
        discrim_scores = []
        for k in range(n_components):
            t_k = X_scores[:, k]
            p_k = P[:, k]
            recon_k = np.outer(t_k, p_k)
            var_k = np.sum(recon_k ** 2)
            var_pct = min((var_k / total_var_X) * 100.0, 100.0)
            q_w = abs(Q[1, k])
            discrim_scores.append(q_w * np.sqrt(max(var_pct, 1e-4)))
        sorted_indices = np.argsort(-np.array(discrim_scores))
        top3_indices = list(sorted_indices[:3])
        top3_ranks = {comp: rank + 1 for rank, comp in enumerate(top3_indices)}

    print(f"[{side.upper()}] Top 3 Selected Components: {[f'PLS{k+1} (Rank {top3_ranks[k]})' for k in top3_indices]}")

    # Build Tidy DataFrame
    plot_rows = []
    stats_dict = {}
    for k in range(n_components):
        comp_name = f"PLS{k+1}"
        s0 = X_scores[y_train == 0, k]
        s1 = X_scores[y_train == 1, k]

        t_stat, p_val = stats.ttest_ind(s0, s1, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(s0, s1)
        stars = compute_significance_stars(p_val)

        stats_dict[comp_name] = {
            "T_Statistic": t_stat,
            "P_Value": p_val,
            "MannWhitney_PVal": u_pval,
            "Significance": stars
        }

        for val in s0:
            plot_rows.append({
                "Component": comp_name,
                "Latent Score": val,
                "Group": "Healthy Control"
            })
        for val in s1:
            plot_rows.append({
                "Component": comp_name,
                "Latent Score": val,
                "Group": "Epilepsy (TLE)"
            })

    df_plot = pd.DataFrame(plot_rows)

    # -------------------------------------------------------------------------
    # Render Publication 3-Panel Box Plot
    # -------------------------------------------------------------------------
    top3_names = [f"PLS{k+1}" for k in top3_indices]
    df_top3 = df_plot[df_plot["Component"].isin(top3_names)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=False, dpi=300)

    for idx, (k, ax_sub) in enumerate(zip(top3_indices, axes)):
        c_name = f"PLS{k+1}"
        rank = top3_ranks[k]
        sub_df = df_top3[df_top3["Component"] == c_name]
        row_stat = stats_dict[c_name]

        # Publication-grade Box Plot
        # No violin plot, no datapoints/stripplot, no outlier fliers
        sns.boxplot(
            data=sub_df,
            x="Group",
            y="Latent Score",
            hue="Group",
            palette=class_palette,
            legend=False,
            width=0.45,
            boxprops=dict(alpha=0.85, edgecolor="black", linewidth=1.5),
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(color="black", linewidth=1.5, linestyle="-"),
            capprops=dict(color="black", linewidth=1.5),
            showcaps=True,
            showfliers=False,  # Strict user requirement: do not add datapoint in
            ax=ax_sub
        )

        # Locked y-axis scale: strictly -80 to 80 for all components
        ax_sub.set_ylim(-80, 80)
        ax_sub.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])

        # Statistical test bracket & exact p-value at aligned top position
        bar_y = 65.0
        bar_h = 3.5

        ax_sub.plot([0, 0, 1, 1], [bar_y, bar_y + bar_h, bar_y + bar_h, bar_y], lw=1.5, c="#2c3e50")
        
        # P-value formatting
        p_val = row_stat['P_Value']
        stars = row_stat['Significance']
        if p_val < 0.001:
            p_str = f"p = {p_val:.2e} ({stars})"
        else:
            p_str = f"p = {p_val:.3f} ({stars})"
            
        ax_sub.text(0.5, 70.0, p_str, ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold", color="#c0392b")

        # Subplot Title & Axis Labels
        ax_sub.set_title(f"Rank {rank}: {c_name}\n({side.capitalize()} Hippocampus)",
                         fontsize=13, fontweight="bold", color="#2c3e50", pad=12)
        ax_sub.set_xlabel("")
        ax_sub.set_ylabel("Component Latent Score ($t$)", fontsize=12, fontweight="bold", labelpad=8)

        # Strict publication frame: Solid black 1.5pt spines on all sides
        for spine in ax_sub.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)

        # No internal grid
        ax_sub.grid(False)

        # Outward ticks
        ax_sub.tick_params(which="major", direction="out", length=6, width=1.5, color="black", labelsize=11)
        ax_sub.tick_params(which="minor", direction="out", length=3.5, width=1.0, color="black")

    # Main title: Strictly omit the word "Dataset"
    plt.suptitle(
        f"Top 3 Selected PLS-DA Components Comparison Across Classes ({side.capitalize()} Hippocampus)\n"
        f"[Directly feeding into -3.0 SD to +3.0 SD Distance Mapping & Grad-CAM Mesh Generation]",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save to all target locations
    target_paths = [
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Dataset_1", side, "plots", "plsda_top3_parameters_boxplot.png"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Dataset_1", side, "plots", "plsda_top3_parameters_detailed_violin.png"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "All_Augment_tain", side, "plots", "plsda_top3_parameters_boxplot.png"),
        os.path.join(repo_root, "Model", "Dataset_1", side, "plots", "plsda_top3_parameters_boxplot.png"),
        os.path.join(repo_root, "Model", "Dataset_1", "plots", "gradcam_plsda", f"plsda_top3_parameters_boxplot_{side}.png"),
        os.path.join(repo_root, "Model_Results_Excel", "07_Detailed_Model_Plots", "Dataset_1", "gradcam_plsda", f"plsda_top3_parameters_boxplot_{side}.png"),
        os.path.join(repo_root, "Model_Results_Excel", "08_PLSDA_Class_Violin_Plots", "Dataset_1", side, "plsda_top3_parameters_boxplot.png"),
        os.path.join(r"C:\Users\IHCK\Desktop\Dataset_1_Bootstrap_Results\plots\gradcam_plsda", f"plsda_top3_parameters_boxplot_{side}.png")
    ]

    for p in target_paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            plt.savefig(p, dpi=300)
            print(f"[OK] Saved Box Plot: {p}")
        except Exception as e:
            print(f"[WARN] Could not save to {p}: {e}")

    plt.close()

def main():
    repo_root = r"c:\Users\IHCK\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
    for side in ["right", "left"]:
        print("=" * 75)
        print(f"Generating Publication Box Plot for Dataset 1 [{side.upper()} Hippocampus]")
        print("=" * 75)
        generate_top3_boxplot_for_side(side=side, repo_root=repo_root)

if __name__ == "__main__":
    main()
