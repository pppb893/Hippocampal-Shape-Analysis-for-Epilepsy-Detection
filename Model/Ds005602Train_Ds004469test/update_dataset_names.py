import os

def update_filenames(filepath, side):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if side == 'left':
        content = content.replace(
            "train_df = pd.read_csv('../ds005602_left_train_balanced.csv')",
            "train_df = pd.read_csv('../ds005602_left_balanced_full.csv')"
        )
        content = content.replace(
            "test_df = pd.read_csv('../ds005602_left_test.csv')",
            "test_df = pd.read_csv('../ds004469_left_full.csv')"
        )
    elif side == 'right':
        content = content.replace(
            "train_df = pd.read_csv('../ds005602_right_train_balanced.csv')",
            "train_df = pd.read_csv('../ds005602_right_balanced_full.csv')"
        )
        content = content.replace(
            "test_df = pd.read_csv('../ds005602_right_test.csv')",
            "test_df = pd.read_csv('../ds004469_right_full.csv')"
        )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

base_dir = r'c:\Users\IHCK\Desktop\train_model\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\Ds005602Train_Ds004469'
files_to_modify = {
    'left/MobileNet/train_mobilenet_pls.py': 'left',
    'left/ResNet/train_resnet_pls.py': 'left',
    'left/ResNet/train_resnet_ae_pls.py': 'left',
    'left/SVM/train_svm_pls.py': 'left',
    'left/MLP/train_mlp_pls.py': 'left',
    
    'right/MobileNet/train_mobilenet_pls.py': 'right',
    'right/ResNet/train_resnet_pls.py': 'right',
    'right/ResNet/train_resnet_ae_pls.py': 'right',
    'right/SVM/train_svm_pls.py': 'right',
    'right/MLP/train_mlp_pls.py': 'right'
}

for file, side in files_to_modify.items():
    path = os.path.join(base_dir, file.replace('/', '\\'))
    if os.path.exists(path):
        update_filenames(path, side)
        print(f'Updated {file}')
    else:
        print(f'Missing {file}')
