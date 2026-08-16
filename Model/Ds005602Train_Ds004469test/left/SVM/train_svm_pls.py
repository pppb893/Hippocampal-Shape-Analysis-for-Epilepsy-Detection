import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class PLSWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        
    def fit(self, X, y):
        self.pls.fit(X, y)
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        return self.pls.transform(X)
        
    def predict(self, X):
        return self.pls.predict(X)


def run_pipeline():
    # Ensure the output directory for plots exists
    os.makedirs('plots', exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv('../ds005602_left_full_augmented.csv')
    test_df = pd.read_csv('../ds004469_left_coef_features.csv')

    # Drop metadata columns to get features
    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType']
    
    train_drop = [c for c in meta_cols if c in train_df.columns]
    X_train = train_df.drop(columns=train_drop)
    y_train = train_df['BinaryClass']

    test_drop = [c for c in meta_cols if c in test_df.columns]
    X_test = test_df.drop(columns=test_drop)
    y_test = test_df['BinaryClass']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Define Scaler but we will apply it inside Pipeline to prevent CV leak
    scaler = StandardScaler()
    # We still need to scale the full sets for the final evaluation later
    X_train_scaled_full = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. PLS-DA Component Comparison
    print("Evaluating PLS-DA components...")
    # Try different numbers of components to compare
    components_to_try = [2, 5, 10, 15, 20, 30, 40, 50, 100]
    # Filter out components that are larger than the number of features or samples
    max_comp = min(int(X_train.shape[0] * 0.8), X_train.shape[1])
    components_to_try = [c for c in components_to_try if c <= max_comp]
    
    pls_cv_scores = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for n_comp in components_to_try:
        # Use Pipeline to avoid data leakage during CV
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pls', PLSWrapper(n_components=n_comp))
        ])
        
        # Custom scoring for PLS-DA (threshold at 0.5 for binary classification)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_pred_val = pipeline.predict(X_val)
            y_pred_class = (y_pred_val > 0.5).astype(int).flatten()
            scores.append(accuracy_score(y_val, y_pred_class))
        
        mean_score = np.mean(scores)
        pls_cv_scores.append(mean_score)
        print(f"PLS components: {n_comp}, CV Accuracy: {mean_score:.4f}")

    # Plot PLS Component Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(components_to_try, pls_cv_scores, marker='o', linestyle='-', color='b')
    plt.title('PLS-DA Cross-Validation Accuracy vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean CV Accuracy')
    plt.grid(True)
    plt.savefig('plots/pls_components_comparison.png')
    plt.close()
    print("Saved PLS components comparison graph to 'plots/pls_components_comparison.png'")

    # Select best number of components
    best_idx = np.argmax(pls_cv_scores)
    best_n_comp = components_to_try[best_idx]
    print(f"-> Selected Best number of PLS components: {best_n_comp} with CV accuracy: {pls_cv_scores[best_idx]:.4f}")

    # 3. Fit PLS with best components and transform data
    print(f"Fitting final PLS model with {best_n_comp} components...")
    pls_final = PLSRegression(n_components=best_n_comp)
    pls_final.fit(X_train_scaled_full, y_train)

    X_train_pls = pls_final.transform(X_train_scaled_full)
    X_test_pls = pls_final.transform(X_test_scaled)

    # 4. Train SVM with Cross Validation
    print("Training SVM Model...")
    # Basic SVM configuration
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)

    # Cross-validation for SVM using Pipeline to strictly prevent data leakage
    # We apply scaling and PLS-DA strictly on the train fold of each split
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pls', PLSWrapper(n_components=best_n_comp)),
        ('svm', svm)
    ])
    svm_cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"SVM 5-Fold CV Accuracy (Strict without leakage): {np.mean(svm_cv_scores):.4f} (+/- {np.std(svm_cv_scores):.4f})")

    # Fit on full PLS-transformed training set
    svm.fit(X_train_pls, y_train)

    # 5. Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    y_pred = svm.predict(X_test_pls)
    y_prob = svm.predict_proba(X_test_pls)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    print(f"*** Final Test Accuracy: {test_acc:.4f} ***\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix - Test Set (SVM)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("Saved Confusion Matrix to 'plots/confusion_matrix.png'")

        # Bootstrapping for 95% Confidence Interval
    n_bootstraps = 1000
    tprs_array = []
    base_fpr = np.linspace(0, 1, 101)
    
    y_test_arr = np.array(y_test)
    y_prob_arr = np.array(y_prob)

    np.random.seed(42)
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_test_arr), len(y_test_arr))
        if len(np.unique(y_test_arr[indices])) < 2:
            continue

        fpr_b, tpr_b, _ = roc_curve(y_test_arr[indices], y_prob_arr[indices])
        tpr_interp = np.interp(base_fpr, fpr_b, tpr_b)
        tpr_interp[0] = 0.0
        tprs_array.append(tpr_interp)

    tprs_array = np.array(tprs_array)
    mean_tprs = tprs_array.mean(axis=0)
    mean_tprs[-1] = 1.0
    
    tpr_lower = np.percentile(tprs_array, 2.5, axis=0)
    tpr_upper = np.percentile(tprs_array, 97.5, axis=0)

    original_fpr, original_tpr, _ = roc_curve(y_test_arr, y_prob_arr)
    roc_auc = auc(original_fpr, original_tpr)

    # === Plot 1: Bootstrap Lines Version ===
    plt.figure(figsize=(8, 6))
    sample_indices = np.random.choice(len(tprs_array), size=100, replace=False)
    for i, idx in enumerate(sample_indices):
        if i == 0:
            plt.plot(base_fpr, tprs_array[idx], color='steelblue', lw=1, alpha=0.3, label='Bootstrap Samples')
        else:
            plt.plot(base_fpr, tprs_array[idx], color='steelblue', lw=1, alpha=0.3)
            
    plt.plot(base_fpr, mean_tprs, color='darkorange', lw=3, label=f'Mean ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Bootstrap Lines) - Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plots/roc_curve_lines.png')
    plt.close()

    # === Plot 2: Shaded 95% CI Version ===
    plt.figure(figsize=(8, 6))
    plt.plot(base_fpr, mean_tprs, color='darkorange', lw=2, label=f'Mean ROC (area = {roc_auc:.2f})')
    plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='95% CI')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (95% CI Shaded) - Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('plots/roc_curve_ci.png')
    plt.close()
    
    print("Saved ROC curves to 'plots/roc_curve_lines.png' and 'plots/roc_curve_ci.png'")

    print("\nPipeline finished successfully! All files are in the 'plots' directory.")

if __name__ == "__main__":
    run_pipeline()
