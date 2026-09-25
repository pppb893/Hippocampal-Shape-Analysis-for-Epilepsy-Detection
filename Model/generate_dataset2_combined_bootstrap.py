import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

np.random.seed(42)

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(repo_root, "Model")
    excel_root = os.path.join(repo_root, "Model_Results_Excel")
    
    dataset2_dir = os.path.join(model_dir, "Dataset_2")
    excel_out_dir = os.path.join(excel_root, "02_Bootstrap_and_Statistical_Tests", "Dataset_2_Combined")
    
    os.makedirs(dataset2_dir, exist_ok=True)
    os.makedirs(excel_out_dir, exist_ok=True)
    
    dataset_name = "Dataset_2"
    sides = ["left", "right"]
    
    models_config = [
        ("MLP", os.path.join(model_dir, "Dataset_2", "{side}", "MLP", "results", "test_predictions.npz")),
        ("MobileNet", os.path.join(model_dir, "Dataset_2", "{side}", "MobileNet", "results", "test_predictions.npz")),
        ("ResNet+AE", os.path.join(model_dir, "Dataset_2", "{side}", "ResNet", "results", "test_predictions_ae.npz")),
        ("ResNet", os.path.join(model_dir, "Dataset_2", "{side}", "ResNet", "results", "test_predictions.npz")),
        ("SqueezeNet", os.path.join(model_dir, "Dataset_2", "{side}", "SqueezeNet", "results", "test_predictions.npz")),
        ("SVM", os.path.join(model_dir, "Dataset_2", "{side}", "SVM", "results", "test_predictions.npz")),
    ]
    
    print("=" * 80)
    print("GENERATING COMBINED BOOTSTRAP RESULTS FOR DATASET 2 (6 MODELS)")
    print("=" * 80)
    
    model_entries = []
    idx = 0
    for side in sides:
        for m_name, path_pattern in models_config:
            npz_path = path_pattern.format(side=side)
            if not os.path.exists(npz_path):
                print(f"[ERROR] Prediction file not found: {npz_path}")
                sys.exit(1)
            model_tag = f"{dataset_name}_{side}_{m_name}"
            model_entries.append({
                "index": idx,
                "model_tag": model_tag,
                "dataset": dataset_name,
                "side": side,
                "model_name": m_name,
                "npz_path": npz_path
            })
            idx += 1
            
    print(f"Total evaluated models: {len(model_entries)} (6 Left, 6 Right)")
    
    legend_lines = ["Index | Model Name", "-" * 50]
    for me in model_entries:
        legend_lines.append(f"{me['index']} : {me['model_tag']}")
    legend_text = "\n".join(legend_lines) + "\n"
    
    N_BOOT = 1000
    bootstrap_columns = {}
    bootstrap_columns["round"] = list(range(1, N_BOOT + 1))
    
    final_summary_rows = []
    perf_summary_rows = []
    
    for side in sides:
        cohort_models = [m for m in model_entries if m["side"] == side]
        first_npz = np.load(cohort_models[0]["npz_path"])
        yt_first = first_npz["y_test"] if "y_test" in first_npz else first_npz["y_true"]
        N = len(yt_first)
        
        boot_indices = np.random.randint(0, N, size=(N_BOOT, N))
        print(f"\nProcessing Side: {side.upper()} (N_test={N}, Models={len(cohort_models)})...")
        
        for me in cohort_models:
            i = me["index"]
            data = np.load(me["npz_path"])
            y_true = data["y_test"] if "y_test" in data else data["y_true"]
            y_prob = data["y_prob"]
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            
            cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn_f, fp_f, fn_f, tp_f = cm_full.ravel()
            acc_full = accuracy_score(y_true, y_pred)
            sens_c0_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0
            spec_c0_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
            prec_c0_f = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else 0.0
            f1_c0_f = 2 * prec_c0_f * sens_c0_f / (prec_c0_f + sens_c0_f) if (prec_c0_f + sens_c0_f) > 0 else 0.0
            
            sens_c1_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
            spec_c1_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0
            prec_c1_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
            f1_c1_f = 2 * prec_c1_f * sens_c1_f / (prec_c1_f + sens_c1_f) if (prec_c1_f + sens_c1_f) > 0 else 0.0
            auc_full = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            
            tp_arr = np.zeros(N_BOOT, dtype=int)
            tn_arr = np.zeros(N_BOOT, dtype=int)
            fp_arr = np.zeros(N_BOOT, dtype=int)
            fn_arr = np.zeros(N_BOOT, dtype=int)
            acc_arr = np.zeros(N_BOOT, dtype=float)
            sens_arr = np.zeros(N_BOOT, dtype=float)
            spec_arr = np.zeros(N_BOOT, dtype=float)
            f1_arr = np.zeros(N_BOOT, dtype=float)
            auc_arr = np.zeros(N_BOOT, dtype=float)
            
            sens_c0_arr = np.zeros(N_BOOT, dtype=float)
            spec_c0_arr = np.zeros(N_BOOT, dtype=float)
            f1_c0_arr = np.zeros(N_BOOT, dtype=float)
            
            for b_idx in range(N_BOOT):
                sample_idx = boot_indices[b_idx]
                yt = y_true[sample_idx]
                yp = y_pred[sample_idx]
                ypr = y_prob[sample_idx]
                
                tp = int(np.sum((yp == 1) & (yt == 1)))
                tn = int(np.sum((yp == 0) & (yt == 0)))
                fp = int(np.sum((yp == 1) & (yt == 0)))
                fn = int(np.sum((yp == 0) & (yt == 1)))
                
                tp_arr[b_idx] = tp
                tn_arr[b_idx] = tn
                fp_arr[b_idx] = fp
                fn_arr[b_idx] = fn
                
                acc = (tp + tn) / N
                acc_arr[b_idx] = round(acc, 4)
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                sens_arr[b_idx] = round(sens, 4)
                
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                spec_arr[b_idx] = round(spec, 4)
                
                denom1 = 2 * tp + fp + fn
                f1 = (2 * tp) / denom1 if denom1 > 0 else 0.0
                f1_arr[b_idx] = round(f1, 4)
                
                sens_c0 = spec
                spec_c0 = sens
                denom0 = 2 * tn + fp + fn
                f1_c0 = (2 * tn) / denom0 if denom0 > 0 else 0.0
                sens_c0_arr[b_idx] = sens_c0
                spec_c0_arr[b_idx] = spec_c0
                f1_c0_arr[b_idx] = f1_c0
                
                if len(np.unique(yt)) > 1:
                    try:
                        auc_val = roc_auc_score(yt, ypr)
                    except ValueError:
                        auc_val = 0.5
                else:
                    auc_val = 0.5
                auc_arr[b_idx] = round(auc_val, 4)
                
            bootstrap_columns[f"TP_{i}"] = tp_arr
            bootstrap_columns[f"TN_{i}"] = tn_arr
            bootstrap_columns[f"FP_{i}"] = fp_arr
            bootstrap_columns[f"FN_{i}"] = fn_arr
            bootstrap_columns[f"Accuracy_{i}"] = acc_arr
            bootstrap_columns[f"Sensitivity_{i}"] = sens_arr
            bootstrap_columns[f"Specificity_{i}"] = spec_arr
            bootstrap_columns[f"F1_{i}"] = f1_arr
            bootstrap_columns[f"AUC_{i}"] = auc_arr
            
            final_summary_rows.append({
                "Dataset": me["dataset"],
                "Side": me["side"],
                "Model": me["model_name"],
                "Mean_Accuracy": round(float(np.mean(acc_arr)), 4),
                "SD_Accuracy": round(float(np.std(acc_arr)), 4),
                "Mean_Sensitivity_Class0": round(float(np.mean(sens_c0_arr)), 4),
                "SD_Sensitivity_Class0": round(float(np.std(sens_c0_arr)), 4),
                "Mean_Specificity_Class0": round(float(np.mean(spec_c0_arr)), 4),
                "SD_Specificity_Class0": round(float(np.std(spec_c0_arr)), 4),
                "Mean_F1_Class0": round(float(np.mean(f1_c0_arr)), 4),
                "SD_F1_Class0": round(float(np.std(f1_c0_arr)), 4),
                "Mean_Sensitivity_Class1": round(float(np.mean(sens_arr)), 4),
                "SD_Sensitivity_Class1": round(float(np.std(sens_arr)), 4),
                "Mean_Specificity_Class1": round(float(np.mean(spec_arr)), 4),
                "SD_Specificity_Class1": round(float(np.std(spec_arr)), 4),
                "Mean_F1_Class1": round(float(np.mean(f1_arr)), 4),
                "SD_F1_Class1": round(float(np.std(f1_arr)), 4),
                "Mean_AUC": round(float(np.mean(auc_arr)), 4),
                "SD_AUC": round(float(np.std(auc_arr)), 4),
            })
            
            perf_summary_rows.append({
                "Dataset": me["dataset"],
                "Side": me["side"].capitalize(),
                "Model": me["model_name"],
                "Overall Accuracy": round(acc_full, 4),
                "Overall Mean ± SD": f"{np.mean(acc_arr):.4f} ± {np.std(acc_arr):.4f}",
                "Class 0 Acc": round(sens_c0_f, 4),
                "Class 0 Precision": round(prec_c0_f, 4),
                "Class 0 Recall": round(sens_c0_f, 4),
                "Class 0 F1": round(f1_c0_f, 4),
                "Class 0 Mean ± SD": f"{np.mean(f1_c0_arr):.4f} ± {np.std(f1_c0_arr):.4f}",
                "Class 1 Acc": round(sens_c1_f, 4),
                "Class 1 Precision": round(prec_c1_f, 4),
                "Class 1 Recall": round(sens_c1_f, 4),
                "Class 1 F1": round(f1_c1_f, 4),
                "Class 1 Mean ± SD": f"{np.mean(f1_arr):.4f} ± {np.std(f1_arr):.4f}",
                "ROC AUC": round(auc_full, 4)
            })
            
    df_combined_boot = pd.DataFrame(bootstrap_columns)
    df_final_summary = pd.DataFrame(final_summary_rows)
    df_perf_summary = pd.DataFrame(perf_summary_rows)
    
    metrics_to_test = [
        ("Accuracy", "Accuracy"),
        ("AUC", "AUC"),
        ("F1_Class0", "F1_C0"),
        ("F1_Class1", "F1"),
        ("Sensitivity_Class0", "Specificity"),
        ("Sensitivity_Class1", "Sensitivity"),
        ("Specificity_Class0", "Sensitivity"),
        ("Specificity_Class1", "Specificity")
    ]
    
    pvalue_rows = []
    for metric_name, col_prefix in metrics_to_test:
        for side in sides:
            cohort_models = [m for m in model_entries if m["side"] == side]
            n_m = len(cohort_models)
            for a_idx in range(n_m):
                for b_idx in range(a_idx + 1, n_m):
                    mA = cohort_models[a_idx]
                    mB = cohort_models[b_idx]
                    idxA = mA["index"]
                    idxB = mB["index"]
                    
                    if col_prefix == "F1_C0":
                        tnA = df_combined_boot[f"TN_{idxA}"]
                        fpA = df_combined_boot[f"FP_{idxA}"]
                        fnA = df_combined_boot[f"FN_{idxA}"]
                        denomA = 2 * tnA + fpA + fnA
                        seriesA = np.where(denomA > 0, 2 * tnA / denomA, 0.0)
                        
                        tnB = df_combined_boot[f"TN_{idxB}"]
                        fpB = df_combined_boot[f"FP_{idxB}"]
                        fnB = df_combined_boot[f"FN_{idxB}"]
                        denomB = 2 * tnB + fpB + fnB
                        seriesB = np.where(denomB > 0, 2 * tnB / denomB, 0.0)
                    else:
                        seriesA = df_combined_boot[f"{col_prefix}_{idxA}"].values
                        seriesB = df_combined_boot[f"{col_prefix}_{idxB}"].values
                        
                    meanA = float(np.mean(seriesA))
                    meanB = float(np.mean(seriesB))
                    
                    if np.all(seriesA == seriesB):
                        p_val = 1.0
                    else:
                        t_stat, p_val = ttest_rel(seriesA, seriesB, nan_policy="omit")
                        if np.isnan(p_val):
                            p_val = 1.0
                            
                    if p_val < 0.01:
                        sig = "Yes (p<0.01)"
                    elif p_val < 0.05:
                        sig = "Yes (p<0.05)"
                    else:
                        sig = "No"
                        
                    pvalue_rows.append({
                        "Metric": metric_name,
                        "Dataset": dataset_name,
                        "Side": side.capitalize(),
                        "Model_A": mA["model_name"],
                        "Model_B": mB["model_name"],
                        "Mean_Value_A": round(meanA, 4),
                        "Mean_Value_B": round(meanB, 4),
                        "P-Value": p_val,
                        "Significant": sig
                    })
                    
    df_pvalue = pd.DataFrame(pvalue_rows)
    df_non_sig = df_pvalue[df_pvalue["Significant"] == "No"].copy()
    
    target_dirs = [dataset2_dir, excel_out_dir]
    for t_dir in target_dirs:
        df_combined_boot.to_csv(os.path.join(t_dir, "Combined_Bootstrap_Results.csv"), index=False)
        with open(os.path.join(t_dir, "Model_Legend.txt"), "w", encoding="utf-8") as f:
            f.write(legend_text)
        df_final_summary.to_csv(os.path.join(t_dir, "Final_Summary_Mean_SD.csv"), index=False)
        df_pvalue.to_csv(os.path.join(t_dir, "PValue_Significance_Summary.csv"), index=False)
        df_non_sig.to_csv(os.path.join(t_dir, "Non_Significant_Pairs_Summary.csv"), index=False)
        df_perf_summary.to_csv(os.path.join(t_dir, "Model_Performance_Summary.csv"), index=False)
        
    print("\n[OK] Dataset 2 Combined Bootstrap Files generated successfully!")

if __name__ == "__main__":
    main()
