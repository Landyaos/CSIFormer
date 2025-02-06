#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import hdf5storage
import torch.optim as optim
import gc

# 数据预处理与数据集构建
class MIMODataset(Dataset):
    def __init__(self, tx_signal, rx_signal, csi):
        """
        输入数据说明：
        tx_signal: [data_size, n_subc, n_sym, n_tx, 2] (实部虚部分量)
        rx_signal: [data_size, n_subc, n_sym, n_rx, 2]
        csi:       [data_size, n_subc, n_sym, n_tx, n_rx, 2]
        """
        # 合并所有数据样本
        self.data_size = tx_signal.shape[0]
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.csi = csi

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.csi[idx], self.rx_signal[idx], self.tx_signal[idx]

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)

# 深度残差网络模型
class DNNResEQ(nn.Module):
    def __init__(self, input_dim=12, output_dim=4, hidden_dim=128, num_blocks=4):
        super().__init__()
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 残差块堆叠
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_dim*2)
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, csi, rx_signal):
        csi = csi.reshape(*csi.shape[:3], -1)  # [batch_size, n_subc, n_sym, n_tx*n_rx*2]
        rx_signal = rx_signal.reshape(*rx_signal.shape[:3], -1) #[batch_size, n_subc, n_sym, n_rx*2]
        x = torch.cat([csi, rx_signal], dim=-1) # [batch_size, n_subc, n_sym, (n_tx*n_rx + n_rx)*2]
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        x = x.reshape(*x.shape[:3],-1,2)
        return x


def dataset_preprocess(data):
    # 将数据转换为PyTorch张量
    tx_signal = torch.tensor(data['txSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    rx_signal = torch.tensor(data['rxSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    csi = torch.tensor(data['csiLabelData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    del data
    gc.collect()
    return MIMODataset(tx_signal, rx_signal, csi)

class ComplexMSELoss(nn.Module):
    def __init__(self):
        """
        :param alpha: 第一部分损失的权重
        :param beta:  第二部分损失的权重
        """
        super(ComplexMSELoss, self).__init__()


    def forward(self, csi_est, csi_label):
        """
        复数信道估计的均方误差 (MSE) 损失函数。
        x_py: (batch_size, csi_matrix, 2)，估计值
        y_py: (batch_size, csi_matrix, 2)，真实值
        """
        diff = csi_est - csi_label  # 差值，形状保持一致
        loss = torch.mean(diff[..., 0]**2 + diff[..., 1]**2)  # 实部和虚部平方和
        return loss


# 模型训练
def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, epochs, device, checkpoint_dir='./checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0
    model.to(device)
    # 查看是否有可用的最近 checkpoint
    latest_path = os.path.join(checkpoint_dir, model.__class__.__name__ + '_latest.pth')
    best_path = os.path.join(checkpoint_dir, model.__class__.__name__ + '_best.pth')

    if os.path.isfile(latest_path):
        print(f"[INFO] Resuming training from '{latest_path}'")
        checkpoint = torch.load(latest_path, map_location=device)

        # 加载模型、优化器、调度器状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', best_loss)
        print(f"[INFO] Resumed epoch {start_epoch}, best_loss={best_loss:.6f}")
    
    # 分epoch训练

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        # --------------------- Train ---------------------
        model.train()
        total_loss = 0
        for batch_idx, (csi, rx_signal, tx_signal) in enumerate(dataloader_train):
            csi = csi.to(device)
            rx_signal = rx_signal.to(device)
            tx_signal = tx_signal.to(device)
            optimizer.zero_grad()
            output = model(csi, rx_signal)
            loss = criterion(output, tx_signal)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader_train)}, Loss: {loss.item():.4f}")
        
        train_loss = total_loss / len(dataloader_train)
        # 学习率调度器步进（根据策略）
        if scheduler is not None:
            scheduler.step(train_loss)  # 对于 ReduceLROnPlateau 等需要传入指标的调度器

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader_train)}")

        # --------------------- Validate ---------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (csi, rx_signal, tx_signal) in enumerate(dataloader_val):
                csi = csi.to(device)
                rx_signal = rx_signal.to(device)
                tx_signal = tx_signal.to(device)
                output = model(csi, rx_signal)
                loss = criterion(output, tx_signal)
                val_loss += loss.item()
        
        val_loss /= len(dataloader_val)
        print(f"Val Loss: {val_loss:.4f}")

        # --------------------- Checkpoint 保存 ---------------------
        # 1) 保存最新checkpoint（确保断点续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_loss': best_loss,
        }, latest_path)

        # 2) 如果当前验证集 Loss 最佳，则保存为 best.pth
        if val_loss < best_loss:
            best_loss = val_loss 
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_loss': best_loss,
            }, best_path)
            print(f"[INFO] Best model saved at epoch {epoch + 1}, val_loss={val_loss:.4f}")
        # 3) 每隔5个epoch保存当前epoch的权重
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_loss': best_loss,
            }, os.path.join(checkpoint_dir, model.__class__.__name__ + '_epoch_'+str(epoch)+'.pth'))

print("load data")
data_train = hdf5storage.loadmat('./data/raw/eqTrainData.mat')
data_val = hdf5storage.loadmat('./data/raw/eqValData.mat')
print("load done")
# 主函数执行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-3
epochs = 20
batch_size = 128
shuffle_flag = True
model = DNNResEQ()
dataset_train = dataset_preprocess(data_train)
dataset_val = dataset_preprocess(data_val)
criterion = ComplexMSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle_flag)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle_flag)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model)}")
print('train model')

train_model(model, dataloader_train,dataloader_val, criterion, optimizer,scheduler, epochs, device, checkpoint_dir='./checkpoints')
