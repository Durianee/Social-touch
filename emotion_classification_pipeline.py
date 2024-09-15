import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 忽略警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='scipy')
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 定义Dataset类
class EmotionDataset(Dataset):
    def __init__(self, resnet_dir, openpose_dir, labels_df, max_length):
        self.resnet_dir = resnet_dir
        self.openpose_dir = openpose_dir
        self.labels_df = labels_df
        self.max_length = max_length

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        video_number = str(self.labels_df.iloc[idx]['video number']).zfill(3)
        label = self.labels_df.iloc[idx]['Emotion']
        
        # 加载 ResNet 特征
        resnet_path = os.path.join(self.resnet_dir, f'resnet3d50features_{video_number}.pt')
        resnet_features = torch.load(resnet_path).squeeze(0)
        
        # 加载 OpenPose 特征
        openpose_path = os.path.join(self.openpose_dir, f'{video_number}_keypoints_data.json')
        with open(openpose_path, 'r') as f:
            openpose_data = json.load(f)
        
        # 预处理 OpenPose 特征
        openpose_features = process_openpose_features(openpose_data, self.max_length)

        return resnet_features, openpose_features, label


# 预处理 OpenPose 特征
def process_openpose_features(openpose_data, max_length):
    all_frame_features = []
    
    for frame_data in openpose_data.values():
        frame_features = []
        for part in ['body', 'face', 'left_hand', 'right_hand']:
            for person in frame_data.get(part, []):
                for keypoint in person:
                    frame_features.extend(keypoint[:2])
                    
        if len(frame_features) < max_length:
            frame_features.extend([0] * (max_length - len(frame_features)))
        
        all_frame_features.append(torch.tensor(frame_features).float())
    
    avg_pooled_features = torch.stack(all_frame_features).mean(dim=0)
    return avg_pooled_features

# 定义Attention融合模型
class AttentionFusion(nn.Module):
    def __init__(self, resnet_dim, openpose_dim, num_heads):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=resnet_dim, num_heads=num_heads)
        if openpose_dim != resnet_dim:
            self.linear = nn.Linear(openpose_dim, resnet_dim)
        else:
            self.linear = None

    def forward(self, resnet_features, openpose_features):
        resnet_features = resnet_features.unsqueeze(0)
        openpose_features = openpose_features.unsqueeze(0)
        if self.linear is not None:
            openpose_features = self.linear(openpose_features)
        attn_output, _ = self.attention(resnet_features, openpose_features, openpose_features)
        return attn_output.squeeze(0)

# 定义分类器
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(EmotionClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 训练函数
def train_epoch(model, fusion_model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    fusion_model.train()
    total_loss = 0
    for batch_idx, (resnet_features, openpose_features, labels) in enumerate(dataloader):
        resnet_features, openpose_features, labels = resnet_features.to(device), openpose_features.to(device), labels.to(device)

        optimizer.zero_grad()
        fused_features = fusion_model(resnet_features, openpose_features)
        outputs = model(fused_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 模型评估函数
def evaluate_model(model, fusion_model, dataloader, criterion, device):
    model.eval()
    fusion_model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for resnet_features, openpose_features, labels in dataloader:
            resnet_features, openpose_features, labels = resnet_features.to(device), openpose_features.to(device), labels.to(device)
            fused_features = fusion_model(resnet_features, openpose_features)
            outputs = model(fused_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, f1, precision, recall, all_preds, all_labels

# 定义 EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# 主函数
def main():
    set_seed(42)

    # 加载标签数据
    df = pd.read_excel('/Users/lee/Desktop/emotion_datascience.xlsx')
    df['Emotion'] = df['Emotion'].map({-1: 0, 0: 1, 1: 2})

    # 查找最大OpenPose特征长度
    max_length = find_global_max_length('/Users/lee/Desktop/openpose_results/keypoints', df)

    # 5折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 最佳参数
    best_params = {'learning_rate': 0.0005, 'hidden_dims': [512, 256], 'num_heads': 4}
    lr = best_params['learning_rate']
    hidden_dims = best_params['hidden_dims']
    num_heads = best_params['num_heads']

    all_f1_scores = []
    all_accuracy_scores = []
    all_confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df['Emotion'])):
        print(f"Starting fold {fold + 1}")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = EmotionDataset('/Users/lee/Desktop/moments_models-2/Features_file/resnet3d50',
                                       '/Users/lee/Desktop/openpose_results/keypoints',
                                       train_df, max_length)
        
        val_dataset = EmotionDataset('/Users/lee/Desktop/moments_models-2/Features_file/resnet3d50',
                                     '/Users/lee/Desktop/openpose_results/keypoints',
                                     val_df, max_length)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 初始化模型和优化器
        fusion_model = AttentionFusion(resnet_dim=2048, openpose_dim=max_length, num_heads=num_heads).to(device)
        classifier_model = EmotionClassifier(input_dim=2048, hidden_dims=hidden_dims, output_dim=3).to(device)
        optimizer = optim.Adam(classifier_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

        early_stopping = EarlyStopping(patience=5, delta=0.001, verbose=True)
        best_val_loss = float('inf')

        # 训练和验证
        for epoch in range(20):
            train_loss = train_epoch(classifier_model, fusion_model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels = evaluate_model(classifier_model, fusion_model, val_loader, criterion, device)

            print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
                  f"Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} Val Precision: {val_precision:.4f} "
                  f"Val Recall: {val_recall:.4f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step(val_loss)

        all_f1_scores.append(val_f1)
        all_accuracy_scores.append(val_acc)
        cm = confusion_matrix(val_labels, val_preds)
        all_confusion_matrices.append(cm)

    # 打印模型的平均结果
    mean_f1_score = np.mean(all_f1_scores)
    mean_accuracy_score = np.mean(all_accuracy_scores)
    mean_confusion_matrix = np.mean(all_confusion_matrices, axis=0)

    print(f"Accuracy: {mean_accuracy_score:.4f}")
    print(f"F1 Score: {mean_f1_score:.4f}")
    print(f"Matrix confusion:\n{mean_confusion_matrix}")

# 查找最大OpenPose特征长度
def find_global_max_length(openpose_dir, labels_df):
    max_length = 0
    for idx in range(len(labels_df)):
        video_number = str(labels_df.iloc[idx]['video number']).zfill(3)
        openpose_path = os.path.join(openpose_dir, f'{video_number}_keypoints_data.json')
        with open(openpose_path, 'r') as f:
            openpose_data = json.load(f)
        
        for frame_data in openpose_data.values():
            frame_features = []
            for part in ['body', 'face', 'left_hand', 'right_hand']:
                for person in frame_data.get(part, []):
                    for keypoint in person:
                        frame_features.extend(keypoint[:2])
            max_length = max(max_length, len(frame_features))
    return max_length

if __name__ == "__main__":
    main()
