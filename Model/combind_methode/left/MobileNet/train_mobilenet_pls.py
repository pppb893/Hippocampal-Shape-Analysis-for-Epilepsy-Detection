import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import copy

class InvertedResidual1D(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual1D, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend([
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            # dw
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_1D(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(MobileNetV2_1D, self).__init__()
        
        # Initial conv: 1 -> 16
        self.features = [
            nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU6(inplace=True)
        ]
        
        # Inverted residual blocks
        # t: expand_ratio, c: output_channels, n: num_blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 2, 1],
            [6, 64, 1, 2],
        ]
        
        input_channel = 16
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual1D(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                
        # Last layer
        self.features.append(nn.Conv1d(input_channel, 128, 1, 1, 0, bias=False))
        self.features.append(nn.BatchNorm1d(128))
        self.features.append(nn.ReLU6(inplace=True))
        
        self.features = nn.Sequential(*self.features)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


def train_mobilenet_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, device='cpu'):
    # Reshape for 1D CNN: (batch, channels, sequence_length) -> (batch, 1, n_components)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MobileNetV2_1D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        losses.append(epoch_loss / len(train_loader.dataset))
        
    # Evaluate on val if provided
    val_acc = 0.0
    if X_val is not None and y_val is not None:
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
            
            outputs = model(X_val_t)
            preds = (outputs > 0.5).float()
            val_acc = (preds == y_val_t).float().mean().item()
            
    return model, losses, val_acc


def run_pipeline():
    os.makedirs('plots', exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv('../left_train_augmented_coef_features.csv')
    test_df = pd.read_csv('../left_test_coef_features.csv')

    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType']
    
    train_drop = [c for c in meta_cols if c in train_df.columns]
    X_train = train_df.drop(columns=train_drop)
    y_train = train_df['BinaryClass'].values

    test_drop = [c for c in meta_cols if c in test_df.columns]
    X_test = test_df.drop(columns=test_drop)
    y_test = test_df['BinaryClass'].values
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. PLS-DA Component Comparison
    print("Evaluating PLS-DA components...")
    components_to_try = [2, 5, 10, 15, 20, 30, 40, 50, 100]
    max_comp = min(X_train.shape[0], X_train.shape[1])
    components_to_try = [c for c in components_to_try if c <= max_comp]
    
    pls_cv_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for n_comp in components_to_try:
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr_sc, y_tr)
            
            # Predict just using PLS to select components quickly (same as before)
            y_pred_val = pls.predict(X_val_sc)
            y_pred_class = (y_pred_val > 0.5).astype(int).flatten()
            scores.append(accuracy_score(y_val, y_pred_class))
        
        mean_score = np.mean(scores)
        pls_cv_scores.append(mean_score)
        print(f"PLS components: {n_comp}, CV Accuracy (PLS only): {mean_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(components_to_try, pls_cv_scores, marker='o', linestyle='-', color='b')
    plt.title('PLS-DA Cross-Validation Accuracy vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean CV Accuracy')
    plt.grid(True)
    plt.savefig('plots/pls_components_comparison.png')
    plt.close()

    best_idx = np.argmax(pls_cv_scores)
    best_n_comp = components_to_try[best_idx]
    print(f"-> Selected Best number of PLS components: {best_n_comp}")

    # 2. MobileNetV2 5-Fold Cross Validation
    print(f"Running MobileNetV2 CV with {best_n_comp} PLS components...")
    mobilenet_cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Strict scaling and PLS fitting inside CV
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        
        pls = PLSRegression(n_components=best_n_comp)
        pls.fit(X_tr_sc, y_tr)
        
        X_tr_pls = pls.transform(X_tr_sc)
        X_val_pls = pls.transform(X_val_sc)
        
        _, _, val_acc = train_mobilenet_model(
            X_tr_pls, y_tr, X_val_pls, y_val, 
            epochs=50, batch_size=32, device=device
        )
        mobilenet_cv_scores.append(val_acc)

    print(f"MobileNetV2 5-Fold CV Accuracy: {np.mean(mobilenet_cv_scores):.4f} (+/- {np.std(mobilenet_cv_scores):.4f})")

    # 3. Final Model Training on Full Data
    print("Training Final MobileNetV2 Model...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    pls_final = PLSRegression(n_components=best_n_comp)
    pls_final.fit(X_train_sc, y_train)
    
    X_train_pls = pls_final.transform(X_train_sc)
    X_test_pls = pls_final.transform(X_test_sc)
    
    final_model, losses, _ = train_mobilenet_model(
        X_train_pls, y_train, epochs=100, batch_size=32, device=device
    )

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('MobileNetV2 Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('plots/mobilenet_loss_curve.png')
    plt.close()

    # 4. Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_pls, dtype=torch.float32).unsqueeze(1).to(device)
        y_prob = final_model(X_test_t).cpu().numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)

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
    plt.title('Confusion Matrix - Test Set (MobileNetV2)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/mobilenet_confusion_matrix.png')
    plt.close()

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
