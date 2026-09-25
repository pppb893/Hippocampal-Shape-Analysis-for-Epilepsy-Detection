"""
================================================================================
Publication-Quality PLS-DA Component 1 vs 10 Train + Test Overlay Plot
================================================================================
Specifications based on user instructions:
1. Components: PLS-DA Component 1 (X-axis) vs PLS-DA Component 10 (Y-axis) [Optimal component]
2. Color Scheme: Same class uses SAME color:
   - Healthy Control: Blue (#1f77b4) for BOTH Train and Test
   - Epilepsy / TLE: Red (#e74c3c) for BOTH Train and Test
3. Markers: Different markers for Train vs Test within the same class:
   - Healthy Control (Train): Circle 'o' (semi-transparent alpha 0.45)
   - Healthy Control (Test): Triangle '^' (solid alpha 1.0 with white border)
   - Epilepsy / TLE (Train): Cross 'X' (semi-transparent alpha 0.45)
   - Epilepsy / TLE (Test): Diamond 'D' (solid alpha 1.0 with white border)
   - Misclassified Test points: Highlighted with dashed red circle outline
4. Publication Formatting:
   - Font: Serif ('Times New Roman', 12-14pt base)
   - Frame: Solid black 1.5pt spines on all 4 sides
   - Ticks: Outward ticks on both axes (length 6, width 1.5, black)
   - Grid: NO internal grid lines (grid=False)
   - Header: NO word "Dataset" in header!
   - Decision Boundary: Crisp dashed line separating the classes with soft pastel class shading
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Publication Serif Configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

# Class Colors (Strictly identical color per class!)
COLOR_HEALTHY = "#1f77b4"   # Elegant Clinical Blue
COLOR_EPILEPSY = "#e74c3c"  # Elegant Clinical Crimson/Red

PALETTE = {
    "train_healthy": {"color": COLOR_HEALTHY, "marker": "o", "label": "Healthy Control (Train, o)"},
    "train_epilepsy": {"color": COLOR_EPILEPSY, "marker": "X", "label": "Epilepsy / TLE (Train, X)"},
    "test_healthy": {"color": COLOR_HEALTHY, "marker": "^", "label": "Healthy Control (Test, ^)"},
    "test_epilepsy": {"color": COLOR_EPILEPSY, "marker": "D", "label": "Epilepsy / TLE (Test, D)"},
    "boundary": "#2c3e50",
    "healthy_bg": "#ebf5fb",
    "epilepsy_bg": "#fdedec"
}

def load_cohort_data(dataset="All_Augment_tain", side="Right", repo_root=None):
    """
    Loads coordinate data for the specified cohort and side.
    """
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    side_cap = side.capitalize()
    side_lower = side.lower()
    
    prefix = "ALL" if dataset in ("All_Augment_tain", "ALL") else dataset
    data_dir = os.path.join(repo_root, "Model", "Output_Dataset")
    
    candidates_train = [
        os.path.join(data_dir, f"{prefix}_{side_cap}_train_xyz_coords.csv"),
        os.path.join(data_dir, f"{dataset}_{side_cap}_train_xyz_coords.csv"),
        os.path.join(repo_root, "Model", dataset, side_lower, f"{dataset}_{side_cap}_train_xyz_coords.csv"),
        os.path.join(repo_root, "Model", dataset, side_lower, f"{prefix}_{side_cap}_train_xyz_coords.csv"),
        os.path.join(repo_root, "Model", "Dataset_1", side_lower, f"ALL_{side_cap}_train_xyz_coords.csv"),
        os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side_cap}_train_xyz_coords.csv"),
    ]
    candidates_test = [
        os.path.join(data_dir, f"{prefix}_{side_cap}_test_xyz_coords.csv"),
        os.path.join(data_dir, f"{dataset}_{side_cap}_test_xyz_coords.csv"),
        os.path.join(repo_root, "Model", dataset, side_lower, f"{dataset}_{side_cap}_test_xyz_coords.csv"),
        os.path.join(repo_root, "Model", dataset, side_lower, f"{prefix}_{side_cap}_test_xyz_coords.csv"),
        os.path.join(repo_root, "Model", "Dataset_1", side_lower, f"ALL_{side_cap}_test_xyz_coords.csv"),
        os.path.join(r"C:\Users\IHCK\Desktop\poinnet", f"ALL_{side_cap}_test_xyz_coords.csv"),
    ]

    train_path = next((cp for cp in candidates_train if os.path.isfile(cp)), None)
    test_path = next((cp for cp in candidates_test if os.path.isfile(cp)), None)

    if train_path is None or test_path is None:
        raise FileNotFoundError(f"Could not locate train/test CSVs for {dataset} ({side})")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    meta_cols = ["Subject", "Group_Name", "Group_Label", "BinaryClass", "Class", "Group", "DataType", "Unnamed: 0"]
    feat_cols = [c for c in train_df.columns if c not in meta_cols]

    y_col = "Group_Label" if "Group_Label" in train_df.columns else "BinaryClass"
    y_train = train_df[y_col].values.astype(int)
    X_train = train_df[feat_cols].values.astype(np.float32)

    y_test = test_df[y_col].values.astype(int)
    X_test = test_df[feat_cols].values.astype(np.float32)

    return X_train, y_train, X_test, y_test, feat_cols


def run_plsda_comp1_vs_comp10(X_train, y_train, X_test, y_test):
    """
    Fits PLS-DA with 10 components on training data.
    Projects both train and test into the 10 components.
    Trains 2D linear decision boundary on PLS Component 1 (idx 0) and Component 10 (idx 9).
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # One-Hot Encoding for PLS-DA
    Y_tr_ohe = np.zeros((len(y_train), 2), dtype=np.float64)
    for i, y in enumerate(y_train):
        Y_tr_ohe[i, y] = 1.0

    pls = PLSRegression(n_components=10, scale=False)
    X_tr_scores, _ = pls.fit_transform(X_tr_sc, Y_tr_ohe)
    X_te_scores = pls.transform(X_te_sc)

    # Component 1 (idx 0) and Component 10 (idx 9)
    idx_x, idx_y = 0, 9
    X_tr_2d = X_tr_scores[:, [idx_x, idx_y]]
    X_te_2d = X_te_scores[:, [idx_x, idx_y]]

    clf = LogisticRegression(C=1e4)
    clf.fit(X_tr_2d, y_train)

    train_preds = clf.predict(X_tr_2d)
    train_acc = accuracy_score(y_train, train_preds)

    test_preds = clf.predict(X_te_2d)
    test_acc = accuracy_score(y_test, test_preds)
    test_err = (1.0 - test_acc) * 100.0
    misclassified_mask = (test_preds != y_test)
    misclassified_count = int(misclassified_mask.sum())

    return {
        "X_tr_2d": X_tr_2d,
        "X_te_2d": X_te_2d,
        "y_train": y_train,
        "y_test": y_test,
        "clf": clf,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_err": test_err,
        "misclassified_mask": misclassified_mask,
        "misclassified_count": misclassified_count,
    }


def plot_plsda_comp1_vs_10_overlay(
    res,
    cohort_label="All_Augment_tain",
    side_label="Right",
    output_path=None
):
    """
    Renders publication-grade PLS-DA Train + Test Overlay for Component 1 vs 10:
    - Same color for same class (Blue for Healthy, Red for Epilepsy)
    - Different markers (Train: o / X, Test: ^ / D)
    - Solid black 1.5pt spines on all sides
    - Outward ticks
    - NO internal grid
    - NO word 'Dataset' in header
    - Misclassified test points highlighted with dashed red halo
    """
    X_tr_2d = res["X_tr_2d"]
    X_te_2d = res["X_te_2d"]
    y_tr = res["y_train"]
    y_te = res["y_test"]
    clf = res["clf"]
    mis_mask = res["misclassified_mask"]
    mis_count = res["misclassified_count"]

    fig, ax = plt.subplots(figsize=(8.2, 7.0))

    # Compute bounding limits with 12% padding
    all_x = np.concatenate([X_tr_2d[:, 0], X_te_2d[:, 0]])
    all_y = np.concatenate([X_tr_2d[:, 1], X_te_2d[:, 1]])
    x_pad = (all_x.max() - all_x.min()) * 0.12
    y_pad = (all_y.max() - all_y.min()) * 0.12
    xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
    ylim = (all_y.min() - y_pad, all_y.max() + y_pad)

    # 1. Pure White Graph Background & Crisp Decision Boundary
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 400),
        np.linspace(ylim[0], ylim[1], 400)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid_points).reshape(xx.shape)

    # Crisp Decision boundary line (no background shading fill, pure white canvas)
    ax.contour(
        xx, yy, zz,
        levels=[0.5],
        colors=[PALETTE["boundary"]],
        linestyles=["--"],
        linewidths=[2.2],
        zorder=3
    )

    # 2. Train points (Semi-transparent background, distinct marker, SAME color per class)
    h_tr = (y_tr == 0)
    e_tr = (y_tr == 1)

    ax.scatter(
        X_tr_2d[h_tr, 0], X_tr_2d[h_tr, 1],
        c=COLOR_HEALTHY,
        marker="o",
        s=55,
        alpha=0.40,
        edgecolors="none",
        zorder=4,
        label=f"Healthy Control (Train, o, N={h_tr.sum()})"
    )
    ax.scatter(
        X_tr_2d[e_tr, 0], X_tr_2d[e_tr, 1],
        c=COLOR_EPILEPSY,
        marker="X",
        s=65,
        alpha=0.40,
        edgecolors="none",
        zorder=4,
        label=f"Epilepsy / TLE (Train, X, N={e_tr.sum()})"
    )

    # 3. Test points (Solid, prominent, distinct marker, SAME color per class)
    h_te = (y_te == 0)
    e_te = (y_te == 1)

    ax.scatter(
        X_te_2d[h_te, 0], X_te_2d[h_te, 1],
        c=COLOR_HEALTHY,
        marker="^",
        s=115,
        edgecolors="white",
        linewidths=1.4,
        alpha=1.0,
        zorder=5,
        label=f"Healthy Control (Test, ^, N={h_te.sum()})"
    )
    ax.scatter(
        X_te_2d[e_te, 0], X_te_2d[e_te, 1],
        c=COLOR_EPILEPSY,
        marker="D",
        s=100,
        edgecolors="white",
        linewidths=1.4,
        alpha=1.0,
        zorder=5,
        label=f"Epilepsy / TLE (Test, D, N={e_te.sum()})"
    )

    # 4. Highlight Misclassified Test Points (Dashed ring)
    if mis_count > 0:
        ax.scatter(
            X_te_2d[mis_mask, 0], X_te_2d[mis_mask, 1],
            facecolors="none",
            edgecolors="#e74c3c",
            s=220,
            linewidths=2.2,
            linestyle="--",
            zorder=6,
            label=f"Misclassified Test ({mis_count})"
        )

    # 5. Title & Subtitle: STRICTLY omit the word "Dataset"
    side_display = f"{side_label.capitalize()} Hippocampus"
    title_line1 = f"PLS-DA Train + Test Overlay: Component 1 vs 10 [{side_display}]"
    title_line2 = (
        f"Train Acc: {res['train_acc']*100:.1f}% | Test Acc: {res['test_acc']*100:.1f}% "
        f"(Test Error: {res['test_err']:.1f}% | Misclassified: {mis_count}/{len(y_te)})"
    )

    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=13, fontweight="bold", pad=12, color="#1a252f")
    ax.set_xlabel("PLS-DA Component 1", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("PLS-DA Component 10", fontsize=12, fontweight="bold", labelpad=8)

    # 6. Spines: Solid black 1.5pt frame on all 4 borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    # 7. No internal grid
    ax.grid(False)

    # 8. Outward ticks
    ax.tick_params(which="major", direction="out", length=6, width=1.5, color="black", labelsize=11)
    ax.tick_params(which="minor", direction="out", length=3.5, width=1.0, color="black")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 9. Clean Legend
    legend = ax.legend(
        loc="lower right",
        fontsize=9.5,
        framealpha=0.92,
        edgecolor="#2c3e50",
        fancybox=True,
        facecolor="white"
    )
    legend.get_frame().set_linewidth(1.2)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"  [SAVED] {output_path}")

    plt.close()
    return output_path


def plot_combined_side_by_side(
    res_left,
    res_right,
    cohort_label="All_Augment_tain",
    output_path=None
):
    """
    Renders publication-grade 1x2 panel comparing Left and Right Hippocampus side by side.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16.0, 7.2))

    for ax, res, side_name in [(ax_l, res_left, "Left"), (ax_r, res_right, "Right")]:
        X_tr_2d = res["X_tr_2d"]
        X_te_2d = res["X_te_2d"]
        y_tr = res["y_train"]
        y_te = res["y_test"]
        clf = res["clf"]
        mis_mask = res["misclassified_mask"]
        mis_count = res["misclassified_count"]

        all_x = np.concatenate([X_tr_2d[:, 0], X_te_2d[:, 0]])
        all_y = np.concatenate([X_tr_2d[:, 1], X_te_2d[:, 1]])
        x_pad = (all_x.max() - all_x.min()) * 0.12
        y_pad = (all_y.max() - all_y.min()) * 0.12
        xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
        ylim = (all_y.min() - y_pad, all_y.max() + y_pad)

        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 350),
            np.linspace(ylim[0], ylim[1], 350)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        zz = clf.predict(grid_points).reshape(xx.shape)

        # Pure White Background & Decision boundary line
        ax.set_facecolor("white")

        # Boundary line
        ax.contour(
            xx, yy, zz,
            levels=[0.5],
            colors=[PALETTE["boundary"]],
            linestyles=["--"],
            linewidths=[2.2],
            zorder=3
        )

        h_tr = (y_tr == 0)
        e_tr = (y_tr == 1)
        ax.scatter(
            X_tr_2d[h_tr, 0], X_tr_2d[h_tr, 1],
            c=COLOR_HEALTHY, marker="o", s=55, alpha=0.40,
            edgecolors="none", zorder=4,
            label=f"Healthy Control (Train, N={h_tr.sum()})"
        )
        ax.scatter(
            X_tr_2d[e_tr, 0], X_tr_2d[e_tr, 1],
            c=COLOR_EPILEPSY, marker="X", s=65, alpha=0.40,
            edgecolors="none", zorder=4,
            label=f"Epilepsy / TLE (Train, N={e_tr.sum()})"
        )

        h_te = (y_te == 0)
        e_te = (y_te == 1)
        ax.scatter(
            X_te_2d[h_te, 0], X_te_2d[h_te, 1],
            c=COLOR_HEALTHY, marker="^", s=115, edgecolors="white",
            linewidths=1.4, alpha=1.0, zorder=5,
            label=f"Healthy Control (Test, ^, N={h_te.sum()})"
        )
        ax.scatter(
            X_te_2d[e_te, 0], X_te_2d[e_te, 1],
            c=COLOR_EPILEPSY, marker="D", s=100, edgecolors="white",
            linewidths=1.4, alpha=1.0, zorder=5,
            label=f"Epilepsy / TLE (Test, D, N={e_te.sum()})"
        )

        if mis_count > 0:
            ax.scatter(
                X_te_2d[mis_mask, 0], X_te_2d[mis_mask, 1],
                facecolors="none", edgecolors="#e74c3c", s=220,
                linewidths=2.2, linestyle="--", zorder=6,
                label=f"Misclassified Test (N={mis_count})"
            )

        title_l1 = f"PLS-DA Train + Test Overlay: Component 1 vs 10 [{side_name} Hippocampus]"
        title_l2 = (
            f"Train Acc: {res['train_acc']*100:.1f}% | Test Acc: {res['test_acc']*100:.1f}% "
            f"(Test Error: {res['test_err']:.1f}% | Misclassified: {mis_count}/{len(y_te)})"
        )
        ax.set_title(f"{title_l1}\n{title_l2}", fontsize=12.5, fontweight="bold", pad=10, color="#1a252f")
        ax.set_xlabel("PLS-DA Component 1", fontsize=12, fontweight="bold", labelpad=8)
        ax.set_ylabel("PLS-DA Component 10", fontsize=12, fontweight="bold", labelpad=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)

        ax.grid(False)
        ax.tick_params(which="major", direction="out", length=6, width=1.5, color="black", labelsize=11)
        ax.tick_params(which="minor", direction="out", length=3.5, width=1.0, color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        leg = ax.legend(
            loc="lower right",
            fontsize=9.0,
            framealpha=0.92,
            edgecolor="#2c3e50",
            fancybox=True,
            facecolor="white"
        )
        leg.get_frame().set_linewidth(1.2)

    cohort_display = "All Augmented Cohort" if "augment" in cohort_label.lower() else "Primary Clinical Cohort"
    fig.suptitle(
        f"PLS-DA Latent Score Space (Component 1 vs 10 Optimal Subspace): {cohort_display}",
        fontsize=14.5, fontweight="bold", y=0.99, color="#1a252f"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"  [SAVED COMBINED] {output_path}")

    plt.close()
    return output_path


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    targets = [
        ("All_Augment_tain", "Right"),
        ("All_Augment_tain", "Left"),
        ("Ds005602", "Right"),
        ("Ds005602", "Left"),
    ]

    print("=" * 75)
    print("GENERATING PUBLICATION PLS-DA COMPONENT 1 vs 10 OVERLAY PLOTS")
    print("  - Components: Component 1 (X) vs Component 10 (Y)")
    print("  - Colors: Healthy = Blue (#1f77b4), Epilepsy = Red (#e74c3c)")
    print("  - Markers: Train Healthy='o', Test Healthy='^', Train Epilepsy='X', Test Epilepsy='D'")
    print("  - Frame: Solid black 1.5pt spines, Outward ticks, NO grid, NO 'Dataset' in title")
    print("=" * 75)

    saved_files = []
    cohort_results = {"All_Augment_tain": {}, "Ds005602": {}}

    for ds, side in targets:
        print(f"\nProcessing Cohort: {ds} | Side: {side}...")
        X_tr, y_tr, X_te, y_te, _ = load_cohort_data(ds, side, repo_root=repo_root)
        res = run_plsda_comp1_vs_comp10(X_tr, y_tr, X_te, y_te)
        cohort_results[ds][side] = res

        print(f"  Train Acc (Comp 1 vs 10): {res['train_acc']*100:.2f}%")
        print(f"  Test Acc  (Comp 1 vs 10): {res['test_acc']*100:.2f}%")
        print(f"  Misclassified: {res['misclassified_count']}/{len(y_te)} ({res['test_err']:.2f}%)")

        # Destination paths
        # 1. Inside Model_Results_Excel/09_PLSDA_Score_Plots/{ds}/{Side}/
        dest1 = os.path.join(
            repo_root, "Model_Results_Excel", "09_PLSDA_Score_Plots", ds, side,
            f"plsda_comp1_vs_10_overlay_{side.lower()}.png"
        )
        plot_plsda_comp1_vs_10_overlay(res, cohort_label=ds, side_label=side, output_path=dest1)
        saved_files.append(dest1)

        # 2. Inside Model/Dataset_1/plots/plsda_comp1_vs_10/ if ds == Ds005602
        if ds == "Ds005602":
            dest2 = os.path.join(
                repo_root, "Model", "Dataset_1", "plots", "plsda_comp1_vs_10",
                f"plsda_comp1_vs_10_overlay_{side.lower()}.png"
            )
            plot_plsda_comp1_vs_10_overlay(res, cohort_label=ds, side_label=side, output_path=dest2)
            saved_files.append(dest2)
        elif ds == "All_Augment_tain":
            dest3 = os.path.join(
                repo_root, "Model", "All_Augment_tain", "plots", "plsda_comp1_vs_10",
                f"plsda_comp1_vs_10_overlay_{side.lower()}.png"
            )
            plot_plsda_comp1_vs_10_overlay(res, cohort_label=ds, side_label=side, output_path=dest3)
            saved_files.append(dest3)

    # Generate 1x2 combined plots for each cohort
    for ds in ["All_Augment_tain", "Ds005602"]:
        if "Left" in cohort_results[ds] and "Right" in cohort_results[ds]:
            comb_path1 = os.path.join(
                repo_root, "Model_Results_Excel", "09_PLSDA_Score_Plots", ds,
                f"plsda_comp1_vs_10_bilateral_overlay_{ds.lower()}.png"
            )
            plot_combined_side_by_side(
                cohort_results[ds]["Left"], cohort_results[ds]["Right"],
                cohort_label=ds, output_path=comb_path1
            )
            saved_files.append(comb_path1)

            if ds == "Ds005602":
                comb_path2 = os.path.join(
                    repo_root, "Model", "Dataset_1", "plots", "plsda_comp1_vs_10",
                    f"plsda_comp1_vs_10_bilateral_overlay_dataset1.png"
                )
                plot_combined_side_by_side(
                    cohort_results[ds]["Left"], cohort_results[ds]["Right"],
                    cohort_label=ds, output_path=comb_path2
                )
                saved_files.append(comb_path2)

    print("\n" + "=" * 75)
    print("ALL PLOTS SUCCESSFULLY GENERATED!")
    for f in saved_files:
        print(f" - {f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
