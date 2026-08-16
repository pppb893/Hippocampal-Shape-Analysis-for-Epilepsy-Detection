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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import copy

class ResNetAutoencoder1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, seq_length=10):
        super(ResNetAutoencoder1D, self).__init__()
        
        self.target_seq_length = seq_length
        
        # Initial block: 3x3 conv, 32 -> batch norm, relu -> 2x2 max pool, /2
        # Adapted to 1D
        self.init_conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm1d(32)
        self.init_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Block 1: 32 -> 32
        self.b1_conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.b1_bn1 = nn.BatchNorm1d(32)
        self.b1_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.b1_bn2 = nn.BatchNorm1d(32)
        
        # Block 2: 32 -> 64, stride 2
        self.b2_conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.b2_bn1 = nn.BatchNorm1d(64)
        self.b2_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.b2_bn2 = nn.BatchNorm1d(64)
        self.b2_skip_conv = nn.Conv1d(32, 64, kernel_size=1, stride=2)
        self.b2_skip_bn = nn.BatchNorm1d(64)
        
        # Block 3: 64 -> 128, stride 2
        self.b3_conv1 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.b3_bn1 = nn.BatchNorm1d(128)
        self.b3_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.b3_bn2 = nn.BatchNorm1d(128)
        self.b3_skip_conv = nn.Conv1d(64, 128, kernel_size=1, stride=2)
        self.b3_skip_bn = nn.BatchNorm1d(128)
        
        # Output
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # --- CLASSIFIER HEAD ---
        self.fc_class = nn.Linear(128, num_classes)
        
        # --- DECODER HEAD ---
        self.dec_fc = nn.Linear(128, 128 * 2) 
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) 
        self.dec_out = nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # x is (batch, in_channels, sequence_length)
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = F.relu(x)
        
        if x.shape[2] < 2:
            x = F.pad(x, (0, 2 - x.shape[2]))
        x = self.init_pool(x)
        
        # Block 1
        identity = x
        out = self.b1_conv1(x)
        out = self.b1_bn1(out)
        out = F.relu(out)
        out = self.b1_conv2(out)
        out = self.b1_bn2(out)
        out += identity
        out = F.relu(out)
        
        # Block 2
        identity = self.b2_skip_conv(out)
        identity = self.b2_skip_bn(identity)
        
        out2 = self.b2_conv1(out)
        out2 = self.b2_bn1(out2)
        out2 = F.relu(out2)
        out2 = self.b2_conv2(out2)
        out2 = self.b2_bn2(out2)
        
        if out2.shape[2] != identity.shape[2]:
            diff = identity.shape[2] - out2.shape[2]
            out2 = F.pad(out2, (0, diff))
            
        out2 += identity
        out2 = F.relu(out2)
        
        # Block 3
        identity = self.b3_skip_conv(out2)
        identity = self.b3_skip_bn(identity)
        
        out3 = self.b3_conv1(out2)
        out3 = self.b3_bn1(out3)
        out3 = F.relu(out3)
        out3 = self.b3_conv2(out3)
        out3 = self.b3_bn2(out3)
        
        if out3.shape[2] != identity.shape[2]:
            diff = identity.shape[2] - out3.shape[2]
            out3 = F.pad(out3, (0, diff))
            
        out3 += identity
        out3 = F.relu(out3)
        
        # Latent Space
        out_pool = self.global_avg_pool(out3)
        latent = out_pool.view(out_pool.size(0), -1)
        
        # CLASSIFIER HEAD
        class_out = torch.sigmoid(self.fc_class(latent))
        
        # DECODER HEAD
        dec = self.dec_fc(latent)
        dec = dec.view(dec.size(0), 128, 2)
        dec = F.relu(self.dec_conv1(dec))
        dec = F.relu(self.dec_conv2(dec))
        reconstruction = self.dec_out(dec)
        
        if reconstruction.shape[2] > self.target_seq_length:
            reconstruction = reconstruction[:, :, :self.target_seq_length]
        elif reconstruction.shape[2] < self.target_seq_length:
            reconstruction = F.pad(reconstruction, (0, self.target_seq_length - reconstruction.shape[2]))
            
        return class_out, reconstruction


def train_resnet_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, device='cpu'):
    # Reshape for 1D CNN: (batch, channels, sequence_length) -> (batch, 1, n_components)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = ResNetAutoencoder1D(seq_length=X_train_t.shape[2]).to(device)
    criterion_cls = nn.BCELoss()
    criterion_recon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            class_out, recon_out = model(batch_x)
            
            loss_cls = criterion_cls(class_out, batch_y)
            loss_recon = criterion_recon(recon_out, batch_x)
            
            # Combine losses (adjust weight as needed)
            loss = loss_cls + 0.5 * loss_recon
            
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
            
            outputs, _ = model(X_val_t)
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
    plt.savefig('plots/pls_components_comparison_full.png')
    plt.close()

    best_idx = np.argmax(pls_cv_scores)
    best_n_comp = components_to_try[best_idx]
    print(f"-> Selected Best number of PLS components: {best_n_comp}")

    # 2. ResNet 5-Fold Cross Validation
    print(f"Running ResNet CV with {best_n_comp} PLS components...")
    resnet_cv_scores = []
    
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
        
        _, _, val_acc = train_resnet_model(
            X_tr_pls, y_tr, X_val_pls, y_val, 
            epochs=50, batch_size=32, device=device
        )
        resnet_cv_scores.append(val_acc)

    print(f"ResNet 5-Fold CV Accuracy: {np.mean(resnet_cv_scores):.4f} (+/- {np.std(resnet_cv_scores):.4f})")

    # 3. Final Model Training on Full Data
    print("Training Final ResNet Model...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    pls_final = PLSRegression(n_components=best_n_comp)
    pls_final.fit(X_train_sc, y_train)
    
    X_train_pls = pls_final.transform(X_train_sc)
    X_test_pls = pls_final.transform(X_test_sc)
    
    final_model, losses, _ = train_resnet_model(
        X_train_pls, y_train, epochs=100, batch_size=32, device=device
    )

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('ResNet+AE Training Loss Curve (Class + Recon)')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.grid(True)
    plt.savefig('plots/resnet_loss_curve_ae_full.png')
    plt.close()

    # 4. Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_pls, dtype=torch.float32).unsqueeze(1).to(device)
        y_prob, _ = final_model(X_test_t)
        y_prob = y_prob.cpu().numpy().flatten()
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
    plt.title('Confusion Matrix - Test Set (ResNet+AE)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix_ae_full.png')
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
