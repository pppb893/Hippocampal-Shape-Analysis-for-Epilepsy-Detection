import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

np.random.seed(42)

def compute_bootstrap(y_true, y_prob, n_boot=1000):
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.flatten()
            
    y_pred = (y_prob > 0.5).astype(int)
    n = len(y_true)
    
    boot_indices = np.random.randint(0, n, size=(n_boot, n))
    yt_all = y_true[boot_indices]
    yp_all = y_pred[boot_indices]
    
    boot_acc = np.mean(yt_all == yp_all, axis=1)
    
    pos_mask = (yt_all == 1)
    pos_correct = (yp_all == 1) & pos_mask
    pos_total = np.sum(pos_mask, axis=1)
    boot_sens = np.divide(np.sum(pos_correct, axis=1), pos_total, out=np.zeros(n_boot), where=(pos_total > 0))
    
    neg_mask = (yt_all == 0)
    neg_correct = (yp_all == 0) & neg_mask
    neg_total = np.sum(neg_mask, axis=1)
    boot_spec = np.divide(np.sum(neg_correct, axis=1), neg_total, out=np.zeros(n_boot), where=(neg_total > 0))
    
    tp1 = np.sum((yp_all == 1) & (yt_all == 1), axis=1)
    fp1 = np.sum((yp_all == 1) & (yt_all == 0), axis=1)
    fn1 = np.sum((yp_all == 0) & (yt_all == 1), axis=1)
    denom1 = 2 * tp1 + fp1 + fn1
    f1_class1 = np.divide(2 * tp1, denom1, out=np.zeros(n_boot), where=(denom1 > 0))
    
    tp0 = np.sum((yp_all == 0) & (yt_all == 0), axis=1)
    fp0 = np.sum((yp_all == 0) & (yt_all == 1), axis=1)
    fn0 = np.sum((yp_all == 1) & (yt_all == 0), axis=1)
    denom0 = 2 * tp0 + fp0 + fn0
    f1_class0 = np.divide(2 * tp0, denom0, out=np.zeros(n_boot), where=(denom0 > 0))
    
    boot_f1 = 0.5 * (f1_class1 + f1_class0)
    
    boot_auc = []
    for i in range(n_boot):
        yt_s = yt_all[i]
        ypr_s = y_prob[boot_indices[i]]
        if len(np.unique(yt_s)) == 2:
            try:
                boot_auc.append(roc_auc_score(yt_s, ypr_s))
            except:
                pass
    boot_auc = np.array(boot_auc)
    
    return {
        'Acc_Mean': np.mean(boot_acc), 'Acc_CI': (np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5)),
        'Sens_Mean': np.mean(boot_sens), 'Sens_CI': (np.percentile(boot_sens, 2.5), np.percentile(boot_sens, 97.5)),
        'Spec_Mean': np.mean(boot_spec), 'Spec_CI': (np.percentile(boot_spec, 2.5), np.percentile(boot_spec, 97.5)),
        'F1_Mean': np.mean(boot_f1), 'F1_CI': (np.percentile(boot_f1, 2.5), np.percentile(boot_f1, 97.5)),
        'AUC_Mean': np.mean(boot_auc) if len(boot_auc) > 0 else np.nan,
        'AUC_CI': (np.percentile(boot_auc, 2.5), np.percentile(boot_auc, 97.5)) if len(boot_auc) > 0 else (np.nan, np.nan)
    }

base = os.path.abspath(os.path.dirname(__file__))
all_boot = []

for ds in ['Dataset_1', 'Dataset_2']:
    for side in ['left', 'right']:
        p_side = os.path.join(base, ds, side)
        models_map = [
            ('SVM', os.path.join(p_side, 'SVM', 'results', 'test_predictions.npz')),
            ('MLP', os.path.join(p_side, 'MLP', 'results', 'test_predictions.npz')),
            ('ResNet (Standard)', os.path.join(p_side, 'ResNet', 'results', 'test_predictions.npz')),
            ('ResNet (AutoEncoder)', os.path.join(p_side, 'ResNet', 'results', 'test_predictions_ae.npz')),
            ('MobileNet', os.path.join(p_side, 'MobileNet', 'results', 'test_predictions.npz')),
            ('SqueezeNet', os.path.join(p_side, 'SqueezeNet', 'results', 'test_predictions.npz')),
        ]
        for m_name, npz_path in models_map:
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                y_true = data['y_test']
                y_prob = data['y_prob']
                res = compute_bootstrap(y_true, y_prob)
                all_boot.append({
                    'Dataset': ds,
                    'Side': side.capitalize(),
                    'Model': m_name,
                    'Boot_Acc': f"{res['Acc_Mean']*100:.2f}% [{res['Acc_CI'][0]*100:.2f}%, {res['Acc_CI'][1]*100:.2f}%]",
                    'Boot_Sens': f"{res['Sens_Mean']*100:.2f}% [{res['Sens_CI'][0]*100:.2f}%, {res['Sens_CI'][1]*100:.2f}%]",
                    'Boot_Spec': f"{res['Spec_Mean']*100:.2f}% [{res['Spec_CI'][0]*100:.2f}%, {res['Spec_CI'][1]*100:.2f}%]",
                    'Boot_F1': f"{res['F1_Mean']*100:.2f}% [{res['F1_CI'][0]*100:.2f}%, {res['F1_CI'][1]*100:.2f}%]",
                    'Boot_AUC': f"{res['AUC_Mean']:.4f} [{res['AUC_CI'][0]:.4f}, {res['AUC_CI'][1]:.4f}]",
                })

out_csv = os.path.join(base, 'dataset1_dataset2_bootstrap_all_24_models.csv')
df = pd.DataFrame(all_boot)
df.to_csv(out_csv, index=False)
print(f"Successfully computed and saved 1000-bootstrap for all {len(df)} models to: {out_csv}")
