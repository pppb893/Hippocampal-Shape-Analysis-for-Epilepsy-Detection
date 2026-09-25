import os
import sys
import subprocess
import time
import re
import csv
import pandas as pd
import numpy as np

base_model_dir = os.path.abspath(os.path.dirname(__file__))

datasets = [
    ("Dataset_1", ["left", "right"]),
    ("Dataset_2", ["left", "right"]),
]

models = [
    ("SVM", "train_svm_pls.py"),
    ("MLP", "train_mlp_pls.py"),
    ("ResNet (Standard)", "train_resnet_pls.py", "ResNet"),
    ("ResNet (AutoEncoder)", "train_resnet_ae_pls.py", "ResNet"),
    ("MobileNet", "train_mobilenet_pls.py"),
    ("SqueezeNet", "train_squeezenet_pls.py"),
]

results_summary = []
bootstrap_summary = []

total_start = time.time()
print("=" * 80, flush=True)
print("STARTING 10-FOLD CV BATCH TRAINING PIPELINE ACROSS 4 DATASETS (NO POINTNET)", flush=True)
print("=" * 80, flush=True)

for ds_name, sides in datasets:
    ds_dir = os.path.join(base_model_dir, ds_name)
    
    for side in sides:
        side_dir = os.path.join(ds_dir, side)
        logs_dir = os.path.join(side_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        if not os.path.isdir(side_dir):
            print(f"[SKIP] Directory not found: {side_dir}", flush=True)
            continue
            
        dataset_label = f"{ds_name} ({side.upper()})"
        print(f"\n========================================================", flush=True)
        print(f"  DATASET: {dataset_label}", flush=True)
        print(f"========================================================", flush=True)
        
        for item in models:
            model_name = item[0]
            script_file = item[1]
            subfolder = item[2] if len(item) > 2 else model_name
            
            script_path = os.path.join(side_dir, subfolder, script_file)
            if not os.path.isfile(script_path):
                print(f"  [MISSING SCRIPT]: {script_path}", flush=True)
                continue
                
            log_name = f"{side}_{subfolder}_{os.path.splitext(script_file)[0]}.log"
            log_file = os.path.join(logs_dir, log_name)
            
            print(f"  >> Training {model_name} (10-Fold CV Checkpoint Ensemble)...", flush=True)
            start_t = time.time()
            
            proc = subprocess.run(
                [sys.executable, script_file],
                cwd=os.path.dirname(script_path),
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_t
            
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(proc.stdout)
                if proc.stderr:
                    lf.write(f"\n=== STDERR ===\n{proc.stderr}\n")
                
            test_acc = "N/A"
            cv_acc = "N/A"
            sens = "N/A"
            spec = "N/A"
            f1_macro = "N/A"
            roc_auc = "N/A"
            
            m_test = re.search(r"\*\*\* Final Test Accuracy:\s*([0-9.]+)\s*\*\*\*", proc.stdout) or re.search(r"Final Test Accuracy:\s*([0-9.]+)", proc.stdout)
            if m_test:
                test_acc = f"{float(m_test.group(1))*100:.2f}%"
                
            m_cv = re.search(r"(?:Mean CV Accuracy|CV Accuracy \(Strict without leakage\)|with CV accuracy):\s*([0-9.]+)", proc.stdout)
            if m_cv:
                cv_acc = f"{float(m_cv.group(1))*100:.2f}%"

            m_metrics = re.search(r"Sensitivity \(TLE Recall\):\s*([0-9.]+)\s*\|\s*Specificity \(Healthy Recall\):\s*([0-9.]+)\s*\|\s*F1-Macro:\s*([0-9.]+)\s*\|\s*ROC-AUC:\s*([0-9.]+)", proc.stdout)
            if m_metrics:
                sens = f"{float(m_metrics.group(1))*100:.2f}%"
                spec = f"{float(m_metrics.group(2))*100:.2f}%"
                f1_macro = f"{float(m_metrics.group(3))*100:.2f}%"
                roc_auc = f"{float(m_metrics.group(4)):.4f}"
                
            status = "OK" if proc.returncode == 0 else "FAIL"
            print(f"     Status: {status} | CV: {cv_acc} | Test: {test_acc} | Sens: {sens} | Spec: {spec} | F1: {f1_macro} | AUC: {roc_auc} | Time: {elapsed:.1f}s", flush=True)
            
            results_summary.append({
                "Dataset": ds_name,
                "Side": side.capitalize(),
                "Model": model_name,
                "Status": status,
                "CV_Acc": cv_acc,
                "Test_Acc": test_acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "F1_Macro": f1_macro,
                "ROC_AUC": roc_auc,
                "Time": f"{elapsed:.1f}s"
            })

            # Check for bootstrap confusion matrix CSV
            boot_csv = os.path.join(os.path.dirname(script_path), "results", "bootstrap_confusion_matrix.csv")
            if os.path.exists(boot_csv):
                try:
                    b_df = pd.read_csv(boot_csv)
                    b_row = {
                        "Dataset": ds_name,
                        "Side": side.capitalize(),
                        "Model": model_name,
                        "Boot_Acc_Mean": f"{b_df['Accuracy'].mean()*100:.2f}%",
                        "Boot_Acc_95CI": f"[{b_df['Accuracy'].quantile(0.025)*100:.2f}%, {b_df['Accuracy'].quantile(0.975)*100:.2f}%]",
                        "Boot_Sens_Mean": f"{b_df['Sensitivity'].mean()*100:.2f}%",
                        "Boot_Sens_95CI": f"[{b_df['Sensitivity'].quantile(0.025)*100:.2f}%, {b_df['Sensitivity'].quantile(0.975)*100:.2f}%]",
                        "Boot_Spec_Mean": f"{b_df['Specificity'].mean()*100:.2f}%",
                        "Boot_Spec_95CI": f"[{b_df['Specificity'].quantile(0.025)*100:.2f}%, {b_df['Specificity'].quantile(0.975)*100:.2f}%]",
                        "Boot_F1_Mean": f"{b_df['F1'].mean()*100:.2f}%",
                        "Boot_F1_95CI": f"[{b_df['F1'].quantile(0.025)*100:.2f}%, {b_df['F1'].quantile(0.975)*100:.2f}%]",
                    }
                    if 'AUC' in b_df.columns:
                        auc_vals = pd.to_numeric(b_df['AUC'], errors='coerce').dropna()
                        if len(auc_vals) > 0:
                            b_row["Boot_AUC_Mean"] = f"{auc_vals.mean():.4f}"
                            b_row["Boot_AUC_95CI"] = f"[{auc_vals.quantile(0.025):.4f}, {auc_vals.quantile(0.975):.4f}]"
                    bootstrap_summary.append(b_row)
                except Exception as ex:
                    print(f"     [Bootstrap parse error]: {ex}", flush=True)

total_time = time.time() - total_start
print("\n" + "=" * 80, flush=True)
print(f"ALL TRAINING COMPLETED IN {total_time/60:.2f} MINUTES", flush=True)
print("=" * 80, flush=True)

# Print Summary Table
print("\n" + "=" * 105, flush=True)
print("BENCHMARK SUMMARY ACROSS 4 DATASETS (NO POINTNET)", flush=True)
print("=" * 105, flush=True)
print(f"{'Dataset':<12} {'Side':<7} {'Model':<22} {'CV Acc':<10} {'Test Acc':<10} {'Sens':<8} {'Spec':<8} {'F1-M':<8} {'AUC':<8} {'Time':<7}", flush=True)
print("-" * 105, flush=True)
for r in results_summary:
    print(f"{r['Dataset']:<12} {r['Side']:<7} {r['Model']:<22} {r['CV_Acc']:<10} {r['Test_Acc']:<10} {r['Sensitivity']:<8} {r['Specificity']:<8} {r['F1_Macro']:<8} {r['ROC_AUC']:<8} {r['Time']:<7}", flush=True)

summary_csv = os.path.join(base_model_dir, "dataset1_dataset2_models_summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["Dataset", "Side", "Model", "Status", "CV_Acc", "Test_Acc", "Sensitivity", "Specificity", "F1_Macro", "ROC_AUC", "Time"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_summary)

print(f"\nSaved training benchmark table to: {summary_csv}", flush=True)

# Run full 1000-bootstrap computation for all 24 models
print("\n" + "=" * 80, flush=True)
print("RUNNING 1000-ROUND TRUE BOOTSTRAPPING FOR ALL 24 MODELS...", flush=True)
print("=" * 80, flush=True)

try:
    boot_script = os.path.join(base_model_dir, "compute_all_bootstrap.py")
    if os.path.exists(boot_script):
        subprocess.run([sys.executable, boot_script], cwd=base_model_dir, check=True)
except Exception as e:
    print(f"Error computing bootstrap: {e}", flush=True)

# Determine the Best Models
df_res = pd.DataFrame(results_summary)
df_res['Test_Acc_Num'] = df_res['Test_Acc'].str.rstrip('%').astype(float)
df_res['AUC_Num'] = pd.to_numeric(df_res['ROC_AUC'], errors='coerce')
df_res['F1_Num'] = df_res['F1_Macro'].str.rstrip('%').astype(float)

best_models = []
print("\n" + "=" * 90, flush=True)
print("BEST MODEL SELECTION PER DATASET & SIDE", flush=True)
print("=" * 90, flush=True)

for (ds, side), group in df_res.groupby(['Dataset', 'Side']):
    best_row = group.sort_values(by=['Test_Acc_Num', 'AUC_Num', 'F1_Num'], ascending=[False, False, False]).iloc[0]
    best_models.append(best_row.to_dict())
    print(f"  * {ds} ({side}): Best Model is [{best_row['Model']}]")
    print(f"    -> Test Acc: {best_row['Test_Acc']} | Sens: {best_row['Sensitivity']} | Spec: {best_row['Specificity']} | F1: {best_row['F1_Macro']} | AUC: {best_row['ROC_AUC']}")

df_best = pd.DataFrame(best_models)
best_csv = os.path.join(base_model_dir, "best_models_selection.csv")
df_best.to_csv(best_csv, index=False)
print(f"\nSaved Best Models selection to: {best_csv}", flush=True)
print("=" * 90, flush=True)

