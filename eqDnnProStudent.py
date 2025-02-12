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
import math
import gc

# 数据预处理与数据集构建
class MIMODataset(torch.utils.data.Dataset):
    """
    MIMO-OFDM 数据集加载器
    维度说明：
    - tx_signal: [data_size, n_subc, n_sym, n_tx, 2] (实虚分量)
    - rx_signal: [data_size, n_subc, n_sym, n_rx, 2]
    - csi:       [data_size, n_subc, n_sym, n_tx, n_rx, 2]
    """
    def __init__(self, tx_signal, rx_signal, csi):
        # 合并所有数据样本
        self.data_size = tx_signal.shape[0]
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.csi = csi

        # 维度校验
        assert tx_signal.shape[:-2] == rx_signal.shape[:-2], "数据维度不匹配"
        assert csi.shape[:-3] == tx_signal.shape[:-2], "CSI维度不匹配"

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return (
            self.csi[idx],    # [n_subc, n_sym, n_tx, n_rx, 2] 
            self.rx_signal[idx], # [n_subc, n_sym, n_rx, 2]
            self.tx_signal[idx]  # [n_subc, n_sym, n_tx, 2]
        )

def dataset_preprocess(data):
    # 将数据转换为PyTorch张量
    tx_signal = torch.tensor(data['txSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, 2]
    rx_signal = torch.tensor(data['rxSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_rx, 2]
    csi = torch.tensor(data['csiLabelData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    del data
    gc.collect()
    return MIMODataset(tx_signal, rx_signal, csi)

class ResidualBlock(nn.Module):
    """带预激活的残差块"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x

class SubcarrierAttention(nn.Module):
    """子载波级自注意力模块"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [batch, subc, embed_dim]
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class DNNResEQWithAttention(nn.Module):
    def __init__(self, n_subc=224, n_sym=14, n_tx=2, n_rx=2, 
                 hidden_dim=256, num_blocks=6):
        """
        参数说明:
        - n_subc: 子载波数 (默认224)
        - n_sym:  OFDM符号数 (默认14)
        - n_tx:   发射天线数 (默认2)
        - n_rx:   接收天线数 (默认2)
        """
        super().__init__()
        self.n_subc = n_subc
        self.n_sym = n_sym
        
        # 输入特征维度计算 (CSI + RX)
        csi_feat_dim = n_tx * n_rx * 2  # 每个CSI矩阵展平维度
        rx_feat_dim = n_rx * 2
        input_dim = csi_feat_dim + rx_feat_dim
        
        # 输入预处理
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 子载波注意力编码器
        self.subc_attention = SubcarrierAttention(hidden_dim)
        
        # 时空残差块
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(hidden_dim, hidden_dim*2),
                SubcarrierAttention(hidden_dim) if i%2==0 else nn.Identity()
            ) for i in range(num_blocks)
        ])
        
        # 输出重建层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_tx*2),  # 每个发射天线的实虚部
            nn.Tanh()  # 约束输出范围
        )

    def forward(self, csi, rx_signal):
        """
        输入维度:
        - csi: [batch, n_subc, n_sym, n_tx, n_rx, 2]
        - rx_signal: [batch, n_subc, n_sym, n_rx, 2]
        
        输出维度: 
        [batch, n_subc, n_sym, n_tx, 2]
        """
        batch_size = csi.size(0)
        
        # 特征展平处理 ------------------------------------------
        # CSI特征: [batch, subc, sym, tx, rx, 2] => [batch, subc, sym, tx*rx*2]
        csi_flat = csi.view(*csi.shape[:3], -1)  
        
        # RX特征: [batch, subc, sym, rx, 2] => [batch, subc, sym, rx*2]
        rx_flat = rx_signal.view(*rx_signal.shape[:3], -1)
        
        # 合并特征: [batch, subc, sym, (tx*rx + rx)*2]
        x = torch.cat([csi_flat, rx_flat], dim=-1)
        
        # 维度重组为: [batch*sym, subc, features]
        x = x.permute(0, 2, 1, 3)  # [batch, sym, subc, features]
        x = x.reshape(batch_size*self.n_sym, self.n_subc, -1)
        
        # 特征投影 ----------------------------------------------
        x = self.input_proj(x)  # [batch*sym, subc, hidden_dim]
        
        # 子载波级注意力编码 -------------------------------------
        x = self.subc_attention(x)  # 保持维度 [batch*sym, subc, hidden]
        
        # 残差块处理 ---------------------------------------------
        for block in self.res_blocks:
            x = block(x)  # 每个块处理都保持维度
            
        # 输出重建 -----------------------------------------------
        # 每个子载波独立输出
        output = self.output_layer(x)  # [batch*sym, subc, n_tx*2]
        
        # 维度恢复
        output = output.view(batch_size, self.n_sym, self.n_subc, -1)
        output = output.permute(0, 2, 1, 3)  # [batch, subc, sym, n_tx*2]
        
        # 重塑为最终输出格式
        return output.view(batch_size, self.n_subc, self.n_sym, -1, 2)

class DNNResEQWithAttentionStudent(nn.Module):
    def __init__(self, n_subc=224, n_sym=14, n_tx=2, n_rx=2, 
                 hidden_dim=128, num_blocks=6):
        """
        参数说明:
        - n_subc: 子载波数 (默认224)
        - n_sym:  OFDM符号数 (默认14)
        - n_tx:   发射天线数 (默认2)
        - n_rx:   接收天线数 (默认2)
        """
        super().__init__()
        self.n_subc = n_subc
        self.n_sym = n_sym
        
        # 输入特征维度计算 (CSI + RX)
        csi_feat_dim = n_tx * n_rx * 2  # 每个CSI矩阵展平维度
        rx_feat_dim = n_rx * 2
        input_dim = csi_feat_dim + rx_feat_dim
        
        # 输入预处理
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 子载波注意力编码器
        self.subc_attention = SubcarrierAttention(hidden_dim)
        
        # 时空残差块
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(hidden_dim, hidden_dim*2),
            ) for i in range(num_blocks)
        ])
        
        # 输出重建层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_tx*2),  # 每个发射天线的实虚部
            nn.Tanh()  # 约束输出范围
        )

    def forward(self, csi, rx_signal):
        """
        输入维度:
        - csi: [batch, n_subc, n_sym, n_tx, n_rx, 2]
        - rx_signal: [batch, n_subc, n_sym, n_rx, 2]
        
        输出维度: 
        [batch, n_subc, n_sym, n_tx, 2]
        """
        batch_size = csi.size(0)
        
        # 特征展平处理 ------------------------------------------
        # CSI特征: [batch, subc, sym, tx, rx, 2] => [batch, subc, sym, tx*rx*2]
        csi_flat = csi.view(*csi.shape[:3], -1)  
        
        # RX特征: [batch, subc, sym, rx, 2] => [batch, subc, sym, rx*2]
        rx_flat = rx_signal.view(*rx_signal.shape[:3], -1)
        
        # 合并特征: [batch, subc, sym, (tx*rx + rx)*2]
        x = torch.cat([csi_flat, rx_flat], dim=-1)
        
        # 维度重组为: [batch*sym, subc, features]
        x = x.permute(0, 2, 1, 3)  # [batch, sym, subc, features]
        x = x.reshape(batch_size*self.n_sym, self.n_subc, -1)
        
        # 特征投影 ----------------------------------------------
        x = self.input_proj(x)  # [batch*sym, subc, hidden_dim]
        
        # 子载波级注意力编码 -------------------------------------
        x = self.subc_attention(x)  # 保持维度 [batch*sym, subc, hidden]
        
        # 残差块处理 ---------------------------------------------
        for block in self.res_blocks:
            x = block(x)  # 每个块处理都保持维度
            
        # 输出重建 -----------------------------------------------
        # 每个子载波独立输出
        output = self.output_layer(x)  # [batch*sym, subc, n_tx*2]
        
        # 维度恢复
        output = output.view(batch_size, self.n_sym, self.n_subc, -1)
        output = output.permute(0, 2, 1, 3)  # [batch, subc, sym, n_tx*2]
        
        # 重塑为最终输出格式
        return output.view(batch_size, self.n_subc, self.n_sym, -1, 2)


class ComplexMSELoss(nn.Module):
    def __init__(self):
        """
        :param alpha: 第一部分损失的权重
        :param beta:  第二部分损失的权重
        """
        super(ComplexMSELoss, self).__init__()


    def forward(self, output, target):
        """
        复数信道估计的均方误差 (MSE) 损失函数。
        x_py: (batch_size, csi_matrix, 2)，估计值
        y_py: (batch_size, csi_matrix, 2)，真实值
        """
        diff = output - target  # 差值，形状保持一致
        loss = torch.mean(diff[..., 0]**2 + diff[..., 1]**2)  # 实部和虚部平方和
        return loss

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.3, gamma=0.5):
        super().__init__()
        self.alpha = alpha  # 软目标蒸馏权重
        self.beta = beta    # 特征蒸馏权重
        self.gamma = gamma  # 真实标签损失权重
        self.mse = ComplexMSELoss()

    def forward(self, student_out, teacher_out, student_feat, teacher_feat, labels):
        # 软目标蒸馏损失（学生输出与教师输出的MSE）
        soft_loss = self.mse(student_out, teacher_out.detach())
        # 特征蒸馏损失（编码器输出的MSE）
        feat_loss = self.mse(student_feat, teacher_feat.detach())
        # 真实标签损失（学生输出与真实值的MSE）
        label_loss = self.mse(student_out, labels)
        # 加权总损失
        total_loss = (self.alpha * soft_loss + 
                     self.beta * feat_loss + 
                     self.gamma * label_loss)
        return total_loss


# # 计算参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# teacher = DNNResEQWithAttention()
# print(f"Teacher Model total trainable parameters: {count_parameters(teacher)}")

# student = DNNResEQWithAttentionStudent()
# print(f"Student Model total trainable parameters: {count_parameters(student)}")

# 模型训练
def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, epochs, device, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0
    model.to(device)
    # 查看是否有可用的最近 checkpoint
    latest_path = os.path.join(checkpoint_dir, model.__class__.__name__ + '_v1_latest.pth')
    best_path = os.path.join(checkpoint_dir, model.__class__.__name__ + '_v1_best.pth')

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


if __name__ == '__main__':


    print("load data")
    data_train = hdf5storage.loadmat('/root/autodl-tmp/data/raw/trainDataV4.mat')
    data_val = hdf5storage.loadmat('/root/autodl-tmp/data/raw/valDataV4.mat')
    checkpoint_dir = '/root/autodl-tmp/checkpoints'
    # checkpoint_dir = './checkpoints'
    # data_train = hdf5storage.loadmat('./data/raw/trainData.mat')
    # data_val = hdf5storage.loadmat('./data/raw/valData.mat')
    print("load done")

    dataset_train = dataset_preprocess(data_train)
    dataset_val = dataset_preprocess(data_val)

    model = DNNResEQWithAttentionStudent()
    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {count_parameters(model)}")
    print('train model')


    # 主函数执行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lr = 1e-3
    epochs = 20
    batch_size = 60
    shuffle_flag = True
    criterion = ComplexMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle_flag)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle_flag)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

    train_model(model, dataloader_train,dataloader_val, criterion, optimizer,scheduler, epochs, device, checkpoint_dir)