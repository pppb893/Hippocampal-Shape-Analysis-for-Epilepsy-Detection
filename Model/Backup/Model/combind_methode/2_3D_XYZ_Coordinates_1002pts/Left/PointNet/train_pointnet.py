import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading data...")
train_df = pd.read_csv('../left_train_augmented_xyz_coords.csv')
test_df = pd.read_csv('../left_test_xyz_coords.csv')

def extract_features_labels(df):
    labels = df['BinaryClass'].values
    # Exclude metadata columns
    metadata_cols = ['Subject', 'Group_Name', 'Group_Label', 'BinaryClass', 'DataType']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Reshape from (N, 3006) -> (N, 1002, 3) -> (N, 3, 1002)
    features = df[feature_cols].values
    num_points = 1002
    # reshape to (N, 1002, 3)
    features_3d = features.reshape(-1, num_points, 3)
    # transpose to (N, 3, 1002) for Conv1D
    features_3d = np.transpose(features_3d, (0, 2, 1))
    return features_3d, labels

X_train, y_train = extract_features_labels(train_df)
X_test, y_test = extract_features_labels(test_df)

# Tensor conversion
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- PointNet Architecture ---

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
            
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNet, self).__init__()
        self.tnet3 = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.tnet64 = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        trans3 = self.tnet3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans3)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        trans64 = self.tnet64(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans64)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x, trans64

def pointnet_loss(outputs, labels, trans64, alpha=0.001):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # Orthogonal regularization loss for Feature T-Net
    d = trans64.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans64.is_cuda:
        I = I.cuda()
    loss_reg = torch.mean(torch.norm(torch.bmm(trans64, trans64.transpose(2, 1)) - I, dim=(1, 2)))
    
    return loss + alpha * loss_reg

# --- 5-Fold Cross Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

print("Running PointNet 5-Fold CV...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_t)):
    X_f_train, X_f_val = X_train_t[train_idx], X_train_t[val_idx]
    y_f_train, y_f_val = y_train_t[train_idx], y_train_t[val_idx]
    
    train_dataset = torch.utils.data.TensorDataset(X_f_train, y_f_train)
    val_dataset = torch.utils.data.TensorDataset(X_f_val, y_f_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = PointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_acc = 0.0
    for epoch in range(5):  # 40 epochs for CV
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs, trans64 = model(bx)
            loss = pointnet_loss(outputs, by, trans64)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs, _ = model(bx)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.cpu().numpy())
        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    cv_accuracies.append(best_val_acc)
    print(f"Fold {fold+1} Accuracy: {best_val_acc:.4f}")

print(f"Mean CV Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})")

# --- Train Final Model ---
print("Training Final PointNet Model...")
train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = PointNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

train_losses = []
for epoch in range(15):
    model.train()
    epoch_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        outputs, trans64 = model(bx)
        loss = pointnet_loss(outputs, by, trans64)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

# --- Evaluation ---
print("\nEvaluating on Test Set...")
model.eval()
X_test_t = X_test_t.to(device)
with torch.no_grad():
    test_outputs, _ = model(X_test_t)
    _, test_preds = torch.max(test_outputs, 1)
    test_probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
    test_preds = test_preds.cpu().numpy()

test_acc = accuracy_score(y_test, test_preds)
print(f"*** Final Test Accuracy: {test_acc:.4f} ***\n")
print("Classification Report:")
print(classification_report(y_test, test_preds))

# --- Plotting ---
os.makedirs('plots', exist_ok=True)

# 1. Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.title('PointNet Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/pointnet_loss_curve.png')
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Disease'], yticklabels=['Control', 'Disease'])
plt.title('PointNet Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('plots/pointnet_confusion_matrix.png')
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - PointNet')
plt.legend(loc='lower right')
plt.savefig('plots/pointnet_roc_curve.png')
plt.close()

print("\nPipeline finished successfully! All files are in the 'plots' directory.")
