import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

np.random.seed(42)

def run_pipeline():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'logs'))
        os.makedirs(logs_dir, exist_ok=True)
        side = os.path.basename(os.path.abspath(os.path.join(script_dir, '..')))
        m_name = os.path.basename(script_dir)
        s_stem = os.path.splitext(os.path.basename(__file__))[0]
        log_file = os.path.join(logs_dir, f"{side}_{m_name}_{s_stem}.log")
        sys.stdout = Logger(log_file)
    except Exception as e:
        pass

    pls_npz = os.path.join('..', 'pls_transformed_features.npz')


    if os.path.exists(pls_npz):


        print(f"Loading precomputed shared PLS-DA features from '{pls_npz}'...")


        pls_data = np.load(pls_npz)


        X_train_pls = pls_data['X_train_pls']


        y_train = pls_data['y_train']


        X_test_pls = pls_data['X_test_pls']


        y_test = pls_data['y_test']


        best_n_comp = int(pls_data['best_n_comp'])


        print(f"Loaded PLS features with {best_n_comp} components: Training shape {X_train_pls.shape}, Test shape {X_test_pls.shape}")


    else:


        print("Precomputed PLS features not found, please run prepare_dataset_pls_features.py first.")


        sys.exit(1)


    c0, c1 = np.sum(y_train == 0), np.sum(y_train == 1)


    n_splits = min(10, min(c0, c1))


    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    c0, c1 = np.sum(y_train == 0), np.sum(y_train == 1)
    n_splits = min(10, min(c0, c1))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Training MLP Model with {n_splits}-Fold CV (Early Stopping active)...")
    mlp_cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_pls, y_train)):
        m = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=600, 
                          early_stopping=True, n_iter_no_change=20, validation_fraction=0.15, random_state=42)
        m.fit(X_train_pls[train_idx], y_train[train_idx])
        val_acc = accuracy_score(y_train[val_idx], m.predict(X_train_pls[val_idx]))
        mlp_cv_scores.append(val_acc)
        stopped_iter = m.n_iter_
        print(f"  Fold {fold+1:2d} / {n_splits} Validation Acc: {val_acc:.4f} (Early stopped at iteration: {stopped_iter})")
    
    cv_acc = np.mean(mlp_cv_scores)
    print(f"MLP {n_splits}-Fold CV Accuracy (Strict without leakage): {cv_acc:.4f} (+/- {np.std(mlp_cv_scores):.4f})")

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=600, 
                        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15, random_state=42)
    mlp.fit(X_train_pls, y_train)
    
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(7, 4))
        plt.plot(mlp.loss_curve_, color='teal')
        plt.title('MLP Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/mlp_loss_curve.png')
        plt.close()
        print("Saved MLP loss curve graph to 'plots/mlp_loss_curve.png'")

    y_pred = mlp.predict(X_test_pls)
    y_prob = mlp.predict_proba(X_test_pls)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    print("")
    print("Evaluating on Test Set...")
    print(f"*** Final Test Accuracy: {test_acc:.4f} ***")
    print("")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print(f"Sensitivity (TLE Recall): {sens:.4f} | Specificity (Healthy Recall): {spec:.4f} | F1-Macro: {f1_m:.4f} | ROC-AUC: {auc_score:.4f}")

    np.savez('results/test_predictions.npz', y_test=y_test, y_pred=y_pred, y_prob=y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'TLE'], yticklabels=['Healthy', 'TLE'])
    plt.title(f'MLP Confusion Matrix (Acc: {test_acc:.2%})')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("Saved Confusion Matrix to 'plots/confusion_matrix.png'")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png')
    plt.close()
    print("Saved ROC curve to 'plots/roc_curve.png'")
    print("")
    print("Pipeline finished successfully! All files are in the 'plots' directory.")

if __name__ == "__main__":
    run_pipeline()
