import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tqdm import tqdm
import logging
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import os
from datetime import datetime
import optuna

# 创建日志文件夹
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 根据当前日期时间生成日志文件名
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f"training_{current_time}.log")

# 设置日志
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class SkeletonDataset(Dataset):
    def __init__(self, bone_data, labels=None, augment=False):
        self.bone_data = torch.FloatTensor(bone_data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.bone_data)

    def __getitem__(self, idx):
        bone = self.bone_data[idx]
        label = self.labels[idx] if self.labels is not None else torch.tensor(-1)

        if self.augment:
            if random.random() > 0.5:
                bone = self.random_rotate(bone)
            if random.random() > 0.5:
                bone = self.random_scale(bone)
            if random.random() > 0.5:
                bone = self.random_time_jitter(bone)
            if random.random() > 0.5:
                bone = self.add_noise(bone)

        return bone, label

    @staticmethod
    def random_rotate(bone):
        angle = random.uniform(-10, 10)
        rot_mat = torch.tensor([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]], dtype=torch.float32)

        for data in [bone]:
            shape = data.shape
            data = data.reshape(shape[0], -1, 2)
            data = torch.matmul(data, rot_mat.t())
            data = data.reshape(shape)

        return bone

    @staticmethod
    def random_scale(bone):
        scale = random.uniform(0.9, 1.1)
        return bone * scale

    @staticmethod
    def random_time_jitter(bone):
        jitter = random.randint(-2, 2)
        return torch.roll(bone, shifts=jitter, dims=2)

    @staticmethod
    def add_noise(bone):
        noise = torch.normal(0, 0.02, size=bone.shape)
        return bone + noise


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), (stride, 1), (padding, 0),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SpatialGraphConv, self).__init__()
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.A))
        return F.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(STGCNBlock, self).__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class TE_GCN_Stream(nn.Module):
    def __init__(self, num_class, num_point, num_person, graph, in_channels=6):
        super(TE_GCN_Stream, self).__init__()

        self.in_channels = in_channels
        self.num_point = num_point
        self.num_person = num_person

        self.bn = nn.BatchNorm1d(in_channels * num_point * num_person)

        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A=graph, residual=False),
            STGCNBlock(64, 64, A=graph),
            STGCNBlock(64, 64, A=graph),
            STGCNBlock(64, 128, A=graph, stride=2),
            STGCNBlock(128, 128, A=graph),
            STGCNBlock(128, 128, A=graph),
            STGCNBlock(128, 256, A=graph, stride=2),
            STGCNBlock(256, 256, A=graph),
            STGCNBlock(256, 256, A=graph),
            STGCNBlock(256, 512, A=graph, stride=2),
            STGCNBlock(512, 512, A=graph),
            STGCNBlock(512, 512, A=graph),
        ])

        self.fc = nn.Linear(512, num_class)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, C * V * M, T)
        x = self.bn(x)

        x = x.view(N, C, V, M, T).permute(0, 3, 1, 4, 2).contiguous().view(N * M, C, T, V)

        for layer in self.layers:
            x = layer(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        x = self.drop_out(x)
        return x


class MultiStreamTE_GCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, graph, dropout_rate):
        super(MultiStreamTE_GCN, self).__init__()

        self.bone_stream = TE_GCN_Stream(num_class, num_point, num_person, graph, in_channels=6)
        self.fc_bone = nn.Linear(512, num_class)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, bone):
        out_bone = self.bone_stream(bone)
        out_bone = self.fc_bone(out_bone)
        out = self.drop_out(out_bone)
        return out


def create_graph_adjacency_matrix(num_nodes):
    A = np.zeros((num_nodes, num_nodes))

    connections = [
        (10, 8), (8, 6), (9, 7), (7, 5),  # arms
        (15, 13), (13, 11), (16, 14), (14, 12),  # legs
        (11, 5), (12, 6), (11, 12), (5, 6),  # torso
        (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)  # nose, eyes and ears
    ]

    for (i, j) in connections:
        A[i, j] = 1
        A[j, i] = 1

    return A


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', lr=0.01, weight_decay=1e-4,
                warmup_epochs=10, initial_lr=5e-4, max_lr=0.001, dropout_rate=0.3, optimizer_name='Adam',
                scheduler_name='CosineAnnealingLR', trial_number=None):
    criterion = LabelSmoothingCrossEntropy(eps=0.1)

    # 选择优化器
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 选择学习率调度器
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    scaler = GradScaler()
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0
    best_model_path = None

    log_file = f"logs/trial_{trial_number}.log" if trial_number is not None else "logs/training.log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if scheduler_name != 'ReduceLROnPlateau':
                scheduler.step()

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for bone, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            bone, labels = bone.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(bone)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if best_model_path:
                os.remove(best_model_path)  # 删除之前的最佳模型
            best_model_path = f'best_model_trial_{trial_number}.pth'
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved: {best_model_path}")

    # 在训练结束后，绘制学习曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.savefig(f'learning_curve_trial_{trial_number}.png')
    plt.close()

    return model, history, best_val_acc


def evaluate(model, data_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for bone, labels in data_loader:
            bone, labels = bone.to(device), labels.to(device)
            outputs = model(bone)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(data_loader), 100. * correct / total


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_bone = np.load('data/train_joint_bone.npy')
    train_label = np.load('data/train_label.npy')

    test_bone = np.load('data/test_A_joint_bone.npy')
    test_label = np.load('data/test_label_A.npy')

    num_class = 155
    num_point = 17
    num_person = 2
    graph = create_graph_adjacency_matrix(num_point)

    train_dataset = SkeletonDataset(train_bone, train_label, augment=True)
    val_dataset = SkeletonDataset(test_bone, test_label)

    batch_size = trial.suggest_int('batch_size', 32, 128, log=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    lr = trial.suggest_float('lr', 1e-6, 5e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 15)
    initial_lr = trial.suggest_float('initial_lr', 1e-5, 1e-3, log=True)
    max_lr = trial.suggest_float('max_lr', 5e-4, 5e-2, log=True)

    dropout_rate = trial.suggest_float('dropout', 0.3, 0.6)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])

    model = MultiStreamTE_GCN(num_class=155, num_point=17, num_person=num_person, graph=graph,
                              dropout_rate=dropout_rate).to(device)

    model, history, best_val_acc = train_model(
        model, train_loader, val_loader, num_epochs=160, device=device,
        lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs,
        initial_lr=initial_lr, max_lr=max_lr, dropout_rate=dropout_rate,
        optimizer_name=optimizer_name, scheduler_name=scheduler_name,
        trial_number=trial.number
    )

    # 对模型进行测试
    test_dataset = SkeletonDataset(test_bone, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for bone, _ in tqdm(test_loader, desc="Testing"):
            bone = bone.to(device)
            outputs = model(bone)
            predictions.extend(outputs.cpu().numpy())

    # 保存测试结果
    np.save(f'predictions_trial_{trial.number}.npy', np.array(predictions))
    print(f"Test predictions saved to predictions_trial_{trial.number}.npy")

    return best_val_acc


def main():
    # 创建一个新的研究，目标是最大化，并将记录保存到 SQLite 数据库
    study = optuna.create_study(direction='maximize', storage='sqlite:///test.db', load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()



