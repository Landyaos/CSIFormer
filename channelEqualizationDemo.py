#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import hdf5storage
import torch.optim as optim
import gc


# ##### 数据集预处理
# 

# In[2]:


class CSIFormerDataset(Dataset):
    
    def __init__(self, tx_signal, rx_signal, csi, tx_pilot_mask, rx_pilot_mask, csi_window):
        """
        初始化数据集
        :param tx_signal: 发射导频信号 [data_size, n_subc, n_sym, n_tx, 2]
        :param rx_signal: 接收导频信号 [data_size, n_subc, n_sym, n_rx, 2]
        :param csi: CSI矩阵 [data_size, n_subc, n_sym, n_tx, n_rx, 2]
        
        """
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.csi = csi
        self.tx_pilot_mask = tx_pilot_mask
        self.rx_pilot_mask = rx_pilot_mask
        self.csi_window = csi_window

    def __len__(self):
        """返回数据集大小"""
        return self.csi.size(0)

    def __getitem__(self, idx):
        """
        返回单个样本
        :param idx: 样本索引
        :return: 发射导频、接收导频、CSI矩阵
        """
        tx_pilot = self.tx_signal[idx] * self.tx_pilot_mask    # [n_subc, n_sym, n_tx, 2]
        rx_pilot = self.tx_signal[idx] * self.rx_pilot_mask    # [n_subc, n_sym, n_rx, 2]
        csi_label = self.csi[idx]                              # [numSubc, n_sym, n_tx, n_rx, 2]
        tx_signal = self.tx_signal[idx]
        rx_signal = self.rx_signal[idx]
        
        if idx < self.csi_window:
            pre_csi = self.csi[idx].unsqueeze(0).repeat(self.csi_window,1,1,1,1,1)
        else:
            pre_csi = self.csi[idx-self.csi_window:idx]  
        return tx_pilot, rx_pilot, pre_csi, rx_signal, csi_label, tx_signal


# In[3]:


def dataset_preprocess(data, data_type = 'train', csi_step = 2):
    # 将数据转换为PyTorch张量
    if data_type == 'train':
        csi = torch.tensor(data['csiTrainData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
        rx_signal = torch.tensor(data['rxSignalTrainData'], dtype=torch.float32) # [data_size, n_subc, n_sym, n_rx, 2]
        tx_signal = torch.tensor(data['txSignalTrainData'], dtype=torch.float32) # [data_size, n_subc, n_sym, n_tx, 2]
    elif data_type == 'test':
        csi = torch.tensor(data['csiTestData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
        rx_signal = torch.tensor(data['rxSignalTestData'], dtype=torch.float32) # [data_size, n_subc, n_sym, n_rx, 2]
        tx_signal = torch.tensor(data['txSignalTestData'], dtype=torch.float32) # [data_size, n_subc, n_sym, n_tx, 2]
    elif data_type == 'val':
        csi = torch.tensor(data['csiValData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
        rx_signal = torch.tensor(data['rxSignalValData'], dtype=torch.float32) # [data_size, n_subc, n_sym, n_rx, 2]
        tx_signal = torch.tensor(data['txSignalValData'], dtype=torch.float32) # [data_size, n_subc, n_sym, n_tx, 2]
    else:
        exit(1)
    del data
    gc.collect()
    tx_pilot_mask = torch.zeros(tx_signal[0].shape)
    rx_pilot_mask = torch.zeros(rx_signal[0].shape)
    pilot_indices = torch.tensor([7, 8, 26, 27, 40, 41, 57, 58])-7
    tx_pilot_mask[pilot_indices,:,:,:] = 1
    rx_pilot_mask[pilot_indices,:,:,:] = 1
    return CSIFormerDataset(tx_signal, rx_signal, csi, tx_pilot_mask, rx_pilot_mask, csi_window=2)


# In[4]:


###############################################################################
# 第一部分：CSIFormer (编码器)
###############################################################################
class CSIEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=2, n_layers=1, n_tx=2, n_rx=2):
        """
        :param d_model: 输入特征维度
        :param nhead: 多头注意力头数
        :param n_layers: Transformer 层数
        :param n_tx: 发射天线数
        :param n_rx: 接收天线数
        """
        super(CSIEncoder, self).__init__()
        self.d_model = d_model
        self.num_tx = n_tx
        self.num_rx = n_rx

        # 线性层将输入映射到 d_model 维度
        self.input_proj = nn.Linear(n_tx * 2 + n_rx * 2, d_model)

        # Transformer 编码器 (batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # 输出层，预测 CSI 矩阵
        self.output_proj = nn.Linear(d_model, n_tx * n_rx * 2)

    def forward(self, tx_pilot_signal, rx_pilot_signal):
        """
        :param tx_pilot_signal: [B, n_subc, n_sym, n_tx, 2]
        :param rx_pilot_signal: [B, n_subc, n_sym, n_rx, 2]
        :return: 初步估计的 CSI [B, n_subc, n_sym, n_tx, n_rx, 2]
        """
        batch_size, n_subc, n_sym, _, _ = tx_pilot_signal.shape

        # 将发射导频和接收导频拼接为输入特征 [B, n_subc, n_sym, (n_tx+n_rx)*2]
        tx_pilot_signal = tx_pilot_signal.view(batch_size, n_subc, n_sym, -1)
        rx_pilot_signal = rx_pilot_signal.view(batch_size, n_subc, n_sym, -1)
        input_features = torch.cat([tx_pilot_signal, rx_pilot_signal], dim=-1)

        # 将输入特征映射到 d_model 维度 [B, n_subc, n_sym, d_model]
        input_features = self.input_proj(input_features)

        # 将 (n_subc, n_sym) “折叠” 成 seq_len，保持 batch 在第 0 维
        # 最终形状: [B, (n_subc*n_sym), d_model]
        seq_len = n_subc * n_sym
        input_features = input_features.view(batch_size, seq_len, self.d_model)

        # 通过 Transformer 编码器 (batch_first=True)
        # 结果也是 [B, seq_len, d_model]
        output = self.transformer_encoder(input_features)

        # 映射到输出维度 (n_tx*n_rx*2)，仍是 [B, seq_len, n_tx*n_rx*2]
        output = self.output_proj(output)

        # 调整输出形状为 [B, n_subc, n_sym, n_tx, n_rx, 2]
        output = output.view(batch_size, n_subc, n_sym, self.num_tx, self.num_rx, 2)

        return output


# In[5]:


###############################################################################
# 第二部分：EnhancedCSIDecoder (解码器)
###############################################################################
class EnhancedCSIDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=2, n_layers=1, n_tx=2, n_rx=2):
        """
        :param d_model: Decoder 嵌入维度
        :param nhead: 注意力头数
        :param n_layers: 解码器层数
        :param n_tx: 发射天线数
        :param n_rx: 接收天线数
        """
        super(EnhancedCSIDecoder, self).__init__()
        self.d_model = d_model
        self.num_tx = n_tx
        self.num_rx = n_rx

        # 输入映射层，将 CSI 转换到 d_model 维度
        self.input_proj = nn.Linear(n_tx * n_rx * 2, d_model)

        # Transformer 解码器 (batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=2048,
                batch_first=True  
            ),
            num_layers=n_layers
        )

        # 输出映射层，将 d_model 映射回原始 CSI 空间
        self.output_proj = nn.Linear(d_model, n_tx * n_rx * 2)

        # （可选）因为我们要把若干帧 memory 拼在一起，再把它映射到 d_model
        # 这里可以提前定义一个映射层，后面用到
        self.memory_proj = nn.Linear(d_model, d_model)

    def forward(self, current_est, previous_csi):
        """
        :param current_est:   当前帧初步估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        :param previous_csi:  前 n 帧 CSI    [B, n_frames, n_subc, n_sym, n_tx, n_rx, 2]
        :return: 增强后的当前帧 CSI [B, n_subc, n_sym, n_tx, n_rx, 2]
        """
        B, n_subc, n_sym, _, _, _ = current_est.shape

        # ============= 处理 Query (current_est) =============
        # 先展平成 [B, n_subc, n_sym, (n_tx*n_rx*2)]
        query = current_est.view(B, n_subc, n_sym, -1)
        # 投影到 d_model: [B, n_subc, n_sym, d_model]
        query = self.input_proj(query)
        # 折叠 (n_subc, n_sym) => seq_len
        # 最终: [B, seq_len_q, d_model]
        seq_len_q = n_subc * n_sym
        query = query.view(B, seq_len_q, self.d_model)

        # ============= 处理 Memory (previous_csi) =============
        # 形状: [B, n_frames, n_subc, n_sym, n_tx, n_rx, 2]
        _, n_frames, n_subc2, n_sym2, _, _, _ = previous_csi.shape
        assert n_subc == n_subc2 and n_sym == n_sym2, "子载波/符号数应与当前帧一致"

        # 可将前 n 帧合并，也可对每帧分别编码，这里演示简单合并:
        # 先变为 [B, n_frames, n_subc, n_sym, (n_tx*n_rx*2)]
        memory = previous_csi.view(B, n_frames, n_subc, n_sym, -1)
        # 投影到 d_model
        memory = self.input_proj(memory)  # [B, n_frames, n_subc, n_sym, d_model]

        # 将 (n_frames, n_subc, n_sym) 都折叠到 seq_len_m
        seq_len_m = n_frames * n_subc * n_sym
        memory = memory.view(B, seq_len_m, self.d_model)  # [B, seq_len_m, d_model]

        # （可选）若需要再投影/结合其他信息，这里可再做一次投影
        memory = self.memory_proj(memory)  # [B, seq_len_m, d_model]

        # ============= 解码器 =============
        # 由于 batch_first=True, query/memory 均为 [B, seq, d_model]
        enhanced = self.transformer_decoder(tgt=query, memory=memory)  # [B, seq_len_q, d_model]

        # 输出映射回 CSI 空间
        enhanced = self.output_proj(enhanced)  # [B, seq_len_q, n_tx*n_rx*2]
        # reshape 回 [B, n_subc, n_sym, n_tx, n_rx, 2]
        enhanced = enhanced.view(B, n_subc, n_sym, self.num_tx, self.num_rx, 2)

        return enhanced


# In[6]:


###############################################################################
# CSIFormer：同时包含 Encoder 和 Decoder，批维在前
###############################################################################
class CSIFormer(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=2, 
                 n_layers=1, 
                 n_tx=2, 
                 n_rx=2,
                 n_frames=2):
        """
        同时包含：
        1) CSIEncoder (编码器): 根据导频估计当前帧
        2) EnhancedCSIDecoder (解码器): 利用前 n 帧和当前帧初步估计进行增强
        :param d_model, nhead, n_layers: Transformer相关超参
        :param n_tx, n_rx: 发射/接收天线数
        :param n_frames: 前 n 帧参考数
        """
        super(CSIFormer, self).__init__()
        self.encoder = CSIEncoder(d_model, nhead, n_layers, n_tx, n_rx)
        self.decoder = EnhancedCSIDecoder(d_model, nhead, n_layers, n_tx, n_rx)
        self.n_frames = n_frames

    def forward(self, tx_pilot_signal, rx_pilot_signal, previous_csi):
        """
        :param tx_pilot_signal: [B, n_subc, n_sym, n_tx, 2]
        :param rx_pilot_signal: [B, n_subc, n_sym, n_rx, 2]
        :param previous_csi:    [B, n_frames, n_subc, n_sym, n_tx, n_rx, 2]
        :return: (csi_enc, csi_dec)
            csi_enc: 初步估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
            csi_dec: 增强估计 [B, n_subc, n_sym, n_tx, n_rx, 2]
        """
        # (1) 编码器：利用导频生成当前帧的初步CSI
        csi_enc = self.encoder(tx_pilot_signal, rx_pilot_signal)
        # (2) 解码器：结合前 n 帧的 CSI 与 csi_enc，输出增强后的 csi
        csi_dec = self.decoder(csi_enc, previous_csi)
        return csi_enc, csi_dec


# In[7]:


###############################################################################
# EqaulizerFormer: 信道均衡模型，TransformerEncoder组成
###############################################################################
class EqaulizerFormer(nn.Module):
    def __init__(self, d_model=256, nhead=4, n_layers=2, n_tx=2, n_rx=2):
        super(EqaulizerFormer, self).__init__()
        self.d_model = d_model
        self.n_tx = n_tx
        self.n_rx = n_rx

        # 将接收信号和 CSI 特征映射到 d_model 维度
        self.input_proj = nn.Linear(n_rx * 2 + n_tx * n_rx * 2, d_model)

        # Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # 输出层，映射到发射信号维度
        self.output_proj = nn.Linear(d_model, n_tx * 2)

    def forward(self, rx_signal, enhanced_csi):
        """
        :param rx_signal: 接收信号 [B, n_subc, n_sym, n_rx, 2]
        :param enhanced_csi: 增强的 CSI [B, n_subc, n_sym, n_tx, n_rx, 2]
        :return: 均衡后的信号 [B, n_subc, n_sym, n_tx, 2]
        """
        B, n_subc, n_sym, _, _ = rx_signal.shape

        # 展平接收信号和增强后的 CSI
        rx_signal = rx_signal.view(B, n_subc, n_sym, -1)
        enhanced_csi = enhanced_csi.view(B, n_subc, n_sym, -1)

        # 拼接输入特征
        input_features = torch.cat([rx_signal, enhanced_csi], dim=-1)

        # 投影到 d_model 维度
        input_features = self.input_proj(input_features)

        # Transformer 编码器
        seq_len = n_subc * n_sym
        input_features = input_features.view(B, seq_len, self.d_model)
        encoded_features = self.transformer_encoder(input_features)

        # 投影到输出信号维度
        output_signal = self.output_proj(encoded_features)
        output_signal = output_signal.view(B, n_subc, n_sym, self.n_tx, 2)

        return output_signal


# In[8]:


###############################################################################
# JointCEEQ: 联合信道估计与均衡模型
###############################################################################
class JointCEEQ(nn.Module):
    def __init__(self, d_model=256, nhead=2, n_layers=4, n_tx=2, n_rx=2, n_frames=2):
        super(JointCEEQ, self).__init__()
        self.estimate = CSIFormer(d_model, nhead, n_layers, n_tx, n_rx, n_frames)
        self.equalizer = EqaulizerFormer(d_model, nhead, n_layers, n_tx, n_rx)

    def forward(self, tx_pilot_signal, rx_pilot_signal, previous_csi, rx_signal):
        # 估计模块：结合导频信号和历史CSI增强估计 CSI
        csi_enc, csi_dec = self.estimate(tx_pilot_signal, rx_pilot_signal, previous_csi)

        # 均衡模块：结合接收信号和增强CSI 恢复信号
        equalized_signal = self.equalizer(rx_signal, csi_dec)
        return csi_enc, csi_dec, equalized_signal


# In[9]:


###############################################################################
# JointCEEQLoss: 联合信道估计与均衡模型联合损失函数
###############################################################################
class JointCEEQLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.9):
        """
        :param alpha: 第一部分损失的权重
        :param beta:  第二部分损失的权重
        """
        super(JointCEEQLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, csi_dec, csi_true, equalized_signal, tx_signal):
        """
        :param csi_enc: 第一部分(编码器)的输出
        :param csi_dec: 第二部分(解码器)的输出
        :param csi_true: 真实的目标CSI
        :return: (total_loss, loss_enc, loss_dec)
        """
        
        # 计算解码器的损失
        loss_dec = self.mse_loss(csi_dec, csi_true)
        
        # 计算均衡器的损失
        loss_equalized = self.mse_loss(equalized_signal, tx_signal)

        # 加权合并
        total_loss = self.alpha * loss_dec + self.beta * loss_equalized
        
        return total_loss, loss_dec, loss_equalized


# In[10]:


# 模型训练
def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, epochs, device, checkpoint_dir='/root/autodl-tmp/checkpoints'):
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
        for batch_idx, (tx_pilot_train, rx_pilot_train, pre_csi_train, rx_singal_train, csi_label, tx_signal_label) in enumerate(dataloader_train):
            tx_pilot_train = tx_pilot_train.to(device)
            rx_pilot_train = rx_pilot_train.to(device)
            pre_csi_train = pre_csi_train.to(device)
            rx_singal_train = rx_singal_train.to(device)
            csi_label = csi_label.to(device)
            tx_signal_label = tx_signal_label.to(device)

            optimizer.zero_grad()
            csi_enc, csi_dec, equalized_signal = model(tx_pilot_train, rx_pilot_train, pre_csi_train, rx_singal_train)
            joint_loss, _, _ = criterion(csi_dec, csi_label,equalized_signal, tx_signal_label)
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
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (tx_pilot_val, rx_pilot_val, pre_csi_val, rx_signal_val, csi_label, tx_signal_label) in enumerate(dataloader_val):
                tx_pilot_val = tx_pilot_val.to(device)
                rx_pilot_val = rx_pilot_val.to(device)
                pre_csi_val = pre_csi_val.to(device)
                rx_signal_val = rx_signal_val.to(device)
                csi_label = csi_label.to(device)
                tx_signal_label = tx_signal_label.to(device)

                csi_enc, csi_dec ,equalized_signal = model(tx_pilot_val, rx_pilot_val, pre_csi_val, rx_signal_val)
                total_loss, _, _ = criterion(csi_dec, csi_label,equalized_signal, tx_signal_label)
                val_loss += total_loss.item()
        
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


# In[11]:


# 模型评估
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader)}")


# In[12]:

print("load data")
data_train = hdf5storage.loadmat('/root/autodl-tmp/data/raw/trainData.mat')
data_val = hdf5storage.loadmat('/root/autodl-tmp/data/raw/valData.mat')

print("load done")
# In[13]:


# 主函数执行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-3
epochs = 30
batch_size = 90
shuffle_flag = False
model = JointCEEQ()
dataset_train = dataset_preprocess(data_train,'train')
dataset_val = dataset_preprocess(data_val,'val')
criterion = JointCEEQLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle_flag)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle_flag)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)


# In[ ]:
print('train model')

train_model(model, dataloader_train,dataloader_val, criterion, optimizer,scheduler, epochs, device)
# evaluate_model(model, dataloader, criterion, device)


# In[ ]:


# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model)}")


# In[ ]:




