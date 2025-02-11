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

# ##### 数据集预处理

class CSIFormerDataset(Dataset):
    
    def __init__(self, csi_ls, csi_pre, csi_label):
        """
        初始化数据集
        :param csi_ls: 导频CSI矩阵  [data_size, n_subc, n_sym, n_tx, n_rx, 2]
        :param csi: CSI矩阵 [data_size, n_subc, n_sym, n_tx, n_rx, 2]
        :param csi_pre: 历史CSI矩阵 [data_size, n_frame, n_subc, n_sym, n_tx, n_rx, 2]
        """
        self.csi_ls = csi_ls
        self.csi_pre = csi_pre
        self.csi_label = csi_label

    def __len__(self):
        """返回数据集大小"""
        return self.csi_label.size(0)

    def __getitem__(self, idx):
        """
        返回单个样本
        :param idx: 样本索引
        :return: 发射导频、接收导频、CSI矩阵
        """
        return self.csi_ls[idx], self.csi_pre[idx], self.csi_label[idx]

def dataset_preprocess(data):
    # 将数据转换为PyTorch张量
    csi_ls = torch.tensor(data['csiLSData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    csi_pre = torch.tensor(data['csiPreData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    csi_label = torch.tensor(data['csiLabelData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    del data
    gc.collect()
    return CSIFormerDataset(csi_ls, csi_pre, csi_label)

###############################################################################
# 正弦/余弦位置编码
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=7000):
        """
        :param d_model: 嵌入特征的维度
        :param max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入张量 [B, seq_len, d_model]
        :return: 加入位置编码的张量 [B, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

###############################################################################
# 第一部分：CSIFormer (编码器)
###############################################################################
class CSIEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, n_layers=4, n_tx=2, n_rx=2, dim_feedforward=1024, max_len=7000):
        """
        编码器模块
        :param d_model: Transformer 嵌入维度
        :param nhead: 多头注意力头数
        :param n_layers: Transformer 层数
        :param n_tx: 发射天线数
        :param n_rx: 接收天线数
        :param max_len: 序列的最大长度
        """
        super(CSIEncoder, self).__init__()
        self.d_model = d_model
        self.num_tx = n_tx
        self.num_rx = n_rx

        # 线性层将输入映射到 d_model 维度
        self.input_proj = nn.Linear(n_tx * n_rx * 2, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=n_layers
        )

    def forward(self, csi_ls):
        """
        :param csi_ls: 当前帧的导频估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        :return: 编码后的特征 [B, seq_len, d_model]
        """
        B, n_subc, n_sym, n_tx, n_rx, _ = csi_ls.shape

        # 展平 CSI 矩阵并投影到 d_model
        csi_ls = csi_ls.view(B, n_subc, n_sym, -1)
        input_features = self.input_proj(csi_ls)  # [B, n_subc, n_sym, d_model]

        # 展平 (n_subc, n_sym) 维度为 seq_len
        input_features = input_features.view(B, n_subc * n_sym, self.d_model)

        # 添加位置编码
        input_features = self.pos_encoder(input_features)

        # Transformer 编码器
        output_features = self.transformer_encoder(input_features)
        return output_features

###############################################################################
# 第二部分：EnhancedCSIDecoder (解码器)
###############################################################################
class EnhancedCSIDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, n_layers=4, n_tx=2, n_rx=2, dim_feedforward=1024, max_len=7000):
        """
        :param d_model: Decoder 嵌入维度
        :param nhead: 注意力头数
        :param n_layers: 解码器层数
        :param n_tx: 发射天线数
        :param n_rx: 接收天线数
        :param max_len: 序列的最大长度
        """
        super(EnhancedCSIDecoder, self).__init__()
        self.d_model = d_model
        self.num_tx = n_tx
        self.num_rx = n_rx

        # Transformer 解码器 (batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # 位置编码
        self.pos_query = PositionalEncoding(d_model, max_len)
        self.pos_memory = PositionalEncoding(d_model, max_len)

        # 输出映射层，将 d_model 映射回原始 CSI 空间
        self.output_proj = nn.Linear(d_model, n_tx * n_rx * 2)

        # 投影历史 CSI 到 d_model 维度
        self.memory_proj = nn.Linear(n_tx * n_rx * 2, d_model)

    def forward(self, encoder_features, previous_csi):
        """
        :param encoder_features: 编码器的输出特征 [B, seq_len, d_model]
        :param previous_csi:    前 n 帧 CSI    [B, n_frame, n_subc, n_sym, n_tx, n_rx, 2]
        :return: 增强后的当前帧 CSI [B, n_subc, n_sym, n_tx, n_rx, 2]
        """
        B, seq_len, _ = encoder_features.shape
        _, n_frame, n_subc, n_sym, n_tx, n_rx, _ = previous_csi.shape
        # 添加 Query 的位置编码
        query = self.pos_query(encoder_features)
        # ============= 处理 Memory (previous_csi) =============
        # 展平历史 CSI 为 [B, n_frames, n_subc, n_sym, n_tx * n_rx * 2]
        memory = previous_csi.view(B, n_frame, n_subc, n_sym, -1)
        # 投影到 d_model 维度
        memory = self.memory_proj(memory)  # [B, n_frames, n_subc, n_sym, d_model]
        # 展平历史序列为 [B, seq_len_m, d_model]
        memory = memory.view(B, n_frame * n_subc * n_sym, self.d_model)
        memory = self.pos_memory(memory)
        
        # ============= 解码器 =============
        # 解码器输入 Query: [B, seq_len, d_model], Memory: [B, seq_len_m, d_model]
        enhanced_features = self.transformer_decoder(tgt=query, memory=memory)  # [B, seq_len, d_model]

        # 映射到 CSI 空间
        enhanced_csi = self.output_proj(enhanced_features)  # [B, seq_len, n_tx * n_rx * 2]

        # 恢复形状为 [B, n_subc, n_sym, n_tx, n_rx, 2]
        enhanced_csi = enhanced_csi.view(B, n_subc, n_sym, n_tx, n_rx, 2)
        return enhanced_csi


###############################################################################
# CSIFormer：同时包含 Encoder 和 Decoder，批维在前
###############################################################################
class CSIFormer(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=4, 
                 n_layers=4, 
                 n_tx=2, 
                 n_rx=2,
                 dim_feedforward=1024):
        """
        同时包含：
        1) CSIEncoder (编码器): 根据导频估计当前帧
        2) EnhancedCSIDecoder (解码器): 利用前 n 帧和当前帧初步估计进行增强
        :param d_model, nhead, n_layers: Transformer相关超参
        :param n_tx, n_rx: 发射/接收天线数
        :param n_frame: 前 n 帧参考数
        """
        super(CSIFormer, self).__init__()
        self.encoder = CSIEncoder(d_model, nhead, n_layers, n_rx, n_rx, dim_feedforward)
        self.decoder = EnhancedCSIDecoder(d_model, nhead, n_layers, n_tx, n_rx, dim_feedforward)


    def forward(self, csi_ls, previous_csi):
        """
        :param csi_ls: 当前帧的导频估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        :param previous_csi: 前 n 帧历史 CSI [B, n_frame, n_subc, n_sym, n_tx, n_rx, 2]
        :return: (csi_enc, csi_dec)
            csi_enc: 初步估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
            csi_dec: 增强估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        """
        # (1) 编码器：利用导频生成当前帧的初步 CSI 特征
        csi_enc = self.encoder(csi_ls)  # [B, seq_len, d_model]
        # (2) 解码器：结合前 n 帧的 CSI 与 csi_enc，输出增强后的 CSI
        csi_dec = self.decoder(csi_enc, previous_csi)  # [B, n_subc, n_sym, n_tx, n_rx, 2]
        return csi_dec

###############################################################################
# CSIFormer：同时包含 Encoder 和 Decoder，批维在前
###############################################################################
class CSIFormerStudent(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=2, 
                 n_layers=2, 
                 n_tx=2, 
                 n_rx=2,
                 dim_feedforward=256):
        """
        同时包含：
        1) CSIEncoder (编码器): 根据导频估计当前帧
        2) EnhancedCSIDecoder (解码器): 利用前 n 帧和当前帧初步估计进行增强
        :param d_model, nhead, n_layers: Transformer相关超参
        :param n_tx, n_rx: 发射/接收天线数
        :param n_frame: 前 n 帧参考数
        """
        super(CSIFormerStudent, self).__init__()
        self.encoder = CSIEncoder(d_model, nhead, n_layers, n_rx, n_rx, dim_feedforward)
        self.decoder = EnhancedCSIDecoder(d_model, nhead, n_layers, n_tx, n_rx, dim_feedforward)


    def forward(self, csi_ls, previous_csi):
        """
        :param csi_ls: 当前帧的导频估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        :param previous_csi: 前 n 帧历史 CSI [B, n_frame, n_subc, n_sym, n_tx, n_rx, 2]
        :return: (csi_enc, csi_dec)
            csi_enc: 初步估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
            csi_dec: 增强估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        """
        # (1) 编码器：利用导频生成当前帧的初步 CSI 特征
        csi_enc = self.encoder(csi_ls)  # [B, seq_len, d_model]
        # (2) 解码器：结合前 n 帧的 CSI 与 csi_enc，输出增强后的 CSI
        csi_dec = self.decoder(csi_enc, previous_csi)  # [B, n_subc, n_sym, n_tx, n_rx, 2]
        return csi_dec
    

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
        loss = torch.mean(torch.square(torch.sqrt(torch.square(diff[...,0]) + torch.square(diff[...,1]))))
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

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

teacher = CSIFormer()
print(f"Teacher Model total trainable parameters: {count_parameters(teacher)}")

student = CSIFormerStudent()
print(f"Student Model total trainable parameters: {count_parameters(student)}")

# 教师模型 hook（存为教师 encoder 的属性）
def teacher_encoder_hook(module, input, output):
    module.activation = output.detach()

# 学生模型 hook（存为学生 encoder 的属性）
def student_encoder_hook(module, input, output):
    module.activation = output  # 学生需要梯度

# 注册 hook
teacher.encoder.register_forward_hook(teacher_encoder_hook)
student.encoder.register_forward_hook(student_encoder_hook)

# 模型训练
def train_model(teacher, student, dataloader_train, dataloader_val, criterion, optimizer, scheduler, epochs, device, checkpoint_dir='./checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0

    criterion_val = ComplexMSELoss()

    teacher.to(device)
    teacher_path = os.path.join(checkpoint_dir, teacher.__class__.__name__ + '_v1_best.pth')
    if os.path.isfile(teacher_path):
        print(f"[INFO] Resuming training from '{teacher_path}'")
        checkpoint = torch.load(teacher_path, map_location=device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise Exception('something to wrong')

    student.to(device)
    # 查看是否有可用的最近 checkpoint
    latest_path = os.path.join(checkpoint_dir, student.__class__.__name__ + '_v1_latest.pth')
    best_path = os.path.join(checkpoint_dir, student.__class__.__name__ + '_v1_best.pth')
    if os.path.isfile(latest_path):
        print(f"[INFO] Resuming training from '{latest_path}'")
        checkpoint = torch.load(latest_path, map_location=device)

        # 加载模型、优化器、调度器状态
        student.load_state_dict(checkpoint['model_state_dict'])
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
        student.train()
        
        total_loss = 0
        for batch_idx, (csi_ls_train, pre_csi_train, csi_label) in enumerate(dataloader_train):

            csi_ls_train = csi_ls_train.to(device)
            pre_csi_train = pre_csi_train.to(device)
            csi_label = csi_label.to(device)

            with torch.no_grad():
                teacher_output = teacher(csi_ls_train, pre_csi_train)
            optimizer.zero_grad()
            student_output = student(csi_ls_train, pre_csi_train)

            joint_loss = criterion(student_output, teacher_output, student.encoder.activation, teacher.encoder.activation, csi_label)
            joint_loss.backward()
            optimizer.step()
            total_loss += joint_loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader_train)}, Loss: {joint_loss.item():.4f}")
        
        train_loss = total_loss / len(dataloader_train)
        # 学习率调度器步进（根据策略）
        if scheduler is not None:
            scheduler.step(train_loss)  # 对于 ReduceLROnPlateau 等需要传入指标的调度器

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader_train)}")

        # --------------------- Validate ---------------------
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (csi_ls_val, pre_csi_val, csi_label) in enumerate(dataloader_val):
                csi_ls_val = csi_ls_val.to(device)
                pre_csi_val = pre_csi_val.to(device)
                csi_label = csi_label.to(device)

                student_output = student(csi_ls_val, pre_csi_val)
                total_loss = criterion_val(student_output, csi_label)
                val_loss += total_loss.item()
        
        val_loss /= len(dataloader_val)
        print(f"Val Loss: {val_loss:.4f}")

        # --------------------- Checkpoint 保存 ---------------------
        # 1) 保存最新checkpoint（确保断点续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_loss': best_loss,
        }, latest_path)

        # 2) 如果当前验证集 Loss 最佳，则保存为 best.pth
        if val_loss < best_loss:
            best_loss = val_loss 
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_loss': best_loss,
            }, best_path)
            print(f"[INFO] Best model saved at epoch {epoch + 1}, val_loss={val_loss:.4f}")


print("load data")
data_train = hdf5storage.loadmat('/root/autodl-tmp/data/raw/trainDataV4.mat')
data_val = hdf5storage.loadmat('/root/autodl-tmp/data/raw/valDataV4.mat')
checkpoint_dir = '/root/autodl-tmp/checkpoints'
print("load done")

# 主函数执行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-3
epochs = 20
batch_size = 36
shuffle_flag = True

dataset_train = dataset_preprocess(data_train)
dataset_val = dataset_preprocess(data_val)
criterion = DistillationLoss()
optimizer = optim.Adam(student.parameters(), lr=lr)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle_flag, num_workers=4)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle_flag, num_workers=4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)


print('train model')
train_model(teacher, student, dataloader_train,dataloader_val, criterion, optimizer,scheduler, epochs, device, checkpoint_dir)



