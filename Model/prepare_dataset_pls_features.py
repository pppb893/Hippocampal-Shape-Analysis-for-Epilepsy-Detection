import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def find_csv_files(side_dir):
    """Find train and test coef feature CSV files in side_dir."""
    train_files = glob.glob(os.path.join(side_dir, "*train*coef*.csv"))
    test_files = glob.glob(os.path.join(side_dir, "*test*coef*.csv"))
    
    if not train_files or not test_files:
        all_csv = glob.glob(os.path.join(side_dir, "*.csv"))
        train_files = [f for f in all_csv if "train" in os.path.basename(f).lower() and "coef" in os.path.basename(f).lower()]
        test_files = [f for f in all_csv if "test" in os.path.basename(f).lower() and "coef" in os.path.basename(f).lower()]

    if not train_files:
        raise FileNotFoundError(f"Cannot find train coef CSV in {side_dir}")
    if not test_files:
        raise FileNotFoundError(f"Cannot find test coef CSV in {side_dir}")

    return train_files[0], test_files[0]

def extract_and_save_pls_features(side_dir, dataset_name, side_name, force_recompute=False):
    """
    Computes PLS-DA features once for the given dataset and side.
    Saves:
      - pls_transformed_features.npz
      - pls_model.joblib
      - plots/pls_components_comparison.png
    """
    out_npz = os.path.join(side_dir, "pls_transformed_features.npz")
    if os.path.exists(out_npz) and not force_recompute:
        print(f"[{dataset_name} - {side_name}] 'pls_transformed_features.npz' already exists. Loading...")
        data = np.load(out_npz)
        print(f"  -> Found {data['best_n_comp']} components. Train: {data['X_train_pls'].shape}, Test: {data['X_test_pls'].shape}")
        return data

    plots_dir = os.path.join(side_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_path, test_path = find_csv_files(side_dir)
    print(f"\n========================================================")
    print(f"  PROCESSING PLS-DA FOR: {dataset_name} ({side_name.upper()})")
    print(f"  Train: {os.path.basename(train_path)}")
    print(f"  Test:  {os.path.basename(test_path)}")
    print(f"========================================================")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType', 'Group_Name', 'Group_Label', 'Unnamed: 0']
    
    train_drop = [c for c in meta_cols if c in train_df.columns]
    X_train = train_df.drop(columns=train_drop)
    y_train = train_df['BinaryClass'].values if 'BinaryClass' in train_df.columns else train_df['Class'].values

    test_drop = [c for c in meta_cols if c in test_df.columns]
    X_test = test_df.drop(columns=test_drop)
    y_test = test_df['BinaryClass'].values if 'BinaryClass' in test_df.columns else test_df['Class'].values

    feature_names = list(X_train.columns)
    print(f"  Raw Training shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"  Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)} in Train")
    print(f"  Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)} in Test")

    c0, c1 = np.sum(y_train == 0), np.sum(y_train == 1)
    n_splits = min(10, min(c0, c1))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    components_to_try = [2, 5, 10, 15, 20, 30, 40, 50]
    max_comp = min(int(X_train.shape[0] * 0.8), X_train.shape[1])
    components_to_try = [c for c in components_to_try if c <= max_comp]

    print(f"  Evaluating PLS-DA across components: {components_to_try}...")
    pls_cv_scores = []
    for n_comp in components_to_try:
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train.iloc[train_idx])
            X_val_sc = scaler.transform(X_train.iloc[val_idx])
            
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr_sc, y_train[train_idx])
            y_pred_val = pls.predict(X_val_sc)
            scores.append(accuracy_score(y_train[val_idx], (y_pred_val > 0.5).astype(int).flatten()))
        mean_score = np.mean(scores)
        pls_cv_scores.append(mean_score)
        print(f"    Components: {n_comp:2d} -> CV Acc: {mean_score:.4f}")

    best_idx = np.argmax(pls_cv_scores)
    best_n_comp = components_to_try[best_idx]
    best_cv_acc = pls_cv_scores[best_idx]
    print(f"  -> Best Selected PLS Components: {best_n_comp} with CV Acc: {best_cv_acc:.4f}")

    # Plot & Save comparison
    plt.figure(figsize=(8, 4))
    plt.plot(components_to_try, pls_cv_scores, marker='o', color='royalblue', lw=2)
    plt.scatter([best_n_comp], [best_cv_acc], color='red', s=100, zorder=5, label=f'Best: {best_n_comp} ({best_cv_acc:.4f})')
    plt.title(f'PLS Components vs CV Accuracy - {dataset_name} ({side_name.upper()})')
    plt.xlabel('Number of Components')
    plt.ylabel('CV Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "pls_components_comparison.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_file}")

    # Final Fit & Transform
    print(f"  Fitting final PLS model with {best_n_comp} components on full Train set...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    pls_final = PLSRegression(n_components=best_n_comp)
    pls_final.fit(X_train_sc, y_train)

    X_train_pls = pls_final.transform(X_train_sc)
    X_test_pls = pls_final.transform(X_test_sc)

    # Save features NPZ
    np.savez(
        out_npz,
        X_train_pls=X_train_pls.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_test_pls=X_test_pls.astype(np.float32),
        y_test=y_test.astype(np.int64),
        best_n_comp=np.array(best_n_comp),
        best_cv_acc=np.array(best_cv_acc),
        feature_names=np.array(feature_names)
    )
    print(f"  Saved transformed features to: {out_npz}")

    # Save fitted model & scaler
    joblib_file = os.path.join(side_dir, "pls_model.joblib")
    joblib.dump({"scaler": scaler, "pls": pls_final, "best_n_comp": best_n_comp}, joblib_file)
    print(f"  Saved PLS model object to: {joblib_file}")

    return {
        "dataset": dataset_name,
        "side": side_name,
        "best_n_comp": best_n_comp,
        "best_cv_acc": best_cv_acc,
        "train_samples": len(y_train),
        "test_samples": len(y_test)
    }

def run_all_datasets(force_recompute=False):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    datasets = [
        ("Dataset_1", ["left", "right"]),
        ("Dataset_2", ["left", "right"]),
    ]

    summary = []
    print("======================================================================")
    print("STARTING DATASET-LEVEL PLS-DA FEATURE EXTRACTION ACROSS 4 DATASETS")
    print("======================================================================")

    for ds_name, sides in datasets:
        for side in sides:
            side_dir = os.path.join(base_dir, ds_name, side)
            if not os.path.isdir(side_dir):
                print(f"[SKIP] Directory not found: {side_dir}")
                continue
            res = extract_and_save_pls_features(side_dir, ds_name, side, force_recompute=force_recompute)
            if isinstance(res, dict):
                summary.append(res)
            else:
                summary.append({
                    "dataset": ds_name,
                    "side": side,
                    "best_n_comp": int(res['best_n_comp']),
                    "best_cv_acc": float(res['best_cv_acc']) if 'best_cv_acc' in res else 0.0,
                    "train_samples": len(res['y_train']),
                    "test_samples": len(res['y_test'])
                })

    print("\n======================================================================")
    print("PLS-DA DATASET-LEVEL PREPROCESSING SUMMARY")
    print("======================================================================")
    print(f"{'Dataset':<18} {'Side':<8} {'Best PLS Comp':<15} {'CV Accuracy':<12} {'Train N':<10} {'Test N':<8}")
    print("-" * 72)
    for s in summary:
        print(f"{s['dataset']:<18} {s['side'].capitalize():<8} {s['best_n_comp']:<15} {s['best_cv_acc']*100:<11.2f}% {s['train_samples']:<10} {s['test_samples']:<8}")
    print("======================================================================\n")

if __name__ == "__main__":
    force = "--force" in sys.argv
    run_all_datasets(force_recompute=force)
