import os

def optimize_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Preload tensors
    content = content.replace(
        'X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)\n    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)',
        'X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)\n    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)'
    )
    
    # 2. CUDNN Benchmark
    content = content.replace(
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    print(f"Using device: {device}")',
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    if device.type == "cuda":\n        torch.backends.cudnn.benchmark = True\n    print(f"Using device: {device}")'
    )
    
    # 3. Increase batch_size to 256 for CV
    content = content.replace('epochs=50, batch_size=32, device=device', 'epochs=50, batch_size=256, device=device')
    
    # 4. Increase batch_size to 256 for Final Training
    content = content.replace('epochs=100, batch_size=32, device=device', 'epochs=100, batch_size=256, device=device')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

base_dir = r'c:\Users\IHCK\Desktop\train_model\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model\Ds005602'
files_to_modify = [
    'left/MobileNet/train_mobilenet_pls.py',
    'left/ResNet/train_resnet_pls.py',
    'left/ResNet/train_resnet_ae_pls.py',
    'right/MobileNet/train_mobilenet_pls.py',
    'right/ResNet/train_resnet_pls.py',
    'right/ResNet/train_resnet_ae_pls.py'
]

for file in files_to_modify:
    path = os.path.join(base_dir, file.replace('/', '\\'))
    if os.path.exists(path):
        optimize_file(path)
        print(f'Optimized {file}')
    else:
        print(f'Missing {file}')
