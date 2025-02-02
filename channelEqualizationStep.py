# %%
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

# %%
###############################################################################
# 正弦/余弦位置编码
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
# EqaulizerFormer: 信道均衡模型
###############################################################################
class EqaulizerFormer(nn.Module):
    def __init__(self, d_model=256, nhead=8, n_layers=6, n_tx=2, n_rx=2, max_len=5000):
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
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=n_layers
        )
        # 位置编码模块
        self.pos_encoder = PositionalEncoding(d_model, max_len)
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

        # 拼接接收信号和 CSI 矩阵 [B, n_subc, n_sym, n_rx * 2 + n_tx * n_rx *2]
        input_features = torch.cat([rx_signal, enhanced_csi], dim=-1)

        # 投影到 d_model 维度
        input_features = self.input_proj(input_features)

        # Transformer 编码器
        input_features = input_features.view(B, n_subc * n_sym, self.d_model)
        # 添加位置编码
        input_features = self.pos_encoder(input_features)
        # ============= Transformer 编码器 =============
        # 提取全局特征 [B, seq_len, d_model]
        encoded_features = self.transformer_encoder(input_features)

        # 投影到输出信号维度
        output_signal = self.output_proj(encoded_features)
        output_signal = output_signal.view(B, n_subc, n_sym, self.n_tx, 2)
        return output_signal

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


###### 数据集预处理
class EqualizeDataset(Dataset):
    
    def __init__(self, tx_signal, rx_signal, csi):
        """
        初始化数据集
        :param tx_signal: 发射导频信号 [data_size, n_subc, n_sym, n_tx, 2]
        :param rx_signal: 接收导频信号 [data_size, n_subc, n_sym, n_rx, 2]
        :param csi: CSI矩阵 [data_size, n_subc, n_sym, n_tx, n_rx, 2]
        
        """
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.csi = csi

    def __len__(self):
        """返回数据集大小"""
        return self.csi.size(0)

    def __getitem__(self, idx):
        """
        返回单个样本
        :param idx: 样本索引
        :return: 发射导频、接收导频、CSI矩阵
        """

        csi_predict = self.csi[idx]                              # [numSubc, n_sym, n_tx, n_rx, 2]
        tx_signal = self.tx_signal[idx]
        rx_signal = self.rx_signal[idx]

        return rx_signal, csi_predict, tx_signal

def dataset_preprocess(data):
    # 将数据转换为PyTorch张量
    csi = torch.tensor(data['csiLabelData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    rx_signal = torch.tensor(data['rxSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_rx, 2]
    tx_signal = torch.tensor(data['txSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, 2]
    del data
    gc.collect()
    return EqualizeDataset(tx_signal, rx_signal, csi)

# %%
# 模型训练
def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, epochs, device, checkpoint_dir='./checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0
    model.to(device)
    # 查看是否有可用的最近 checkpoint
    latest_path = os.path.join(checkpoint_dir, model.__class__.__name__ + '_pro_latest.pth')
    best_path = os.path.join(checkpoint_dir, model.__class__.__name__ + '_pro_best.pth')

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
        for batch_idx, (rx_singal_train, csi_predict, tx_signal_label) in enumerate(dataloader_train):
            rx_singal_train = rx_singal_train.to(device)
            csi_predict = csi_predict.to(device)
            tx_signal_label = tx_signal_label.to(device)
            optimizer.zero_grad()
            equalized_signal = model(rx_singal_train, csi_predict)
            eq_loss = criterion(equalized_signal, tx_signal_label)
            eq_loss.backward()
            optimizer.step()
            total_loss += eq_loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader_train)}, Loss: {eq_loss.item():.4f}")
        
        train_loss = total_loss / len(dataloader_train)
        # 学习率调度器步进（根据策略）
        if scheduler is not None:
            scheduler.step(train_loss)  # 对于 ReduceLROnPlateau 等需要传入指标的调度器

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader_train)}")

        # --------------------- Validate ---------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (rx_signal_val, csi_predict, tx_signal_label) in enumerate(dataloader_val):
                rx_signal_val = rx_signal_val.to(device)
                csi_predict = csi_predict.to(device)
                tx_signal_label = tx_signal_label.to(device)
                equalized_signal = model(rx_signal_val, csi_predict)
                total_loss = criterion(equalized_signal, tx_signal_label)
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



# %%
print("load data")
data_train = hdf5storage.loadmat('/root/autodl-tmp/data/raw/trainData.mat')
data_val = hdf5storage.loadmat('/root/autodl-tmp/data/raw/valData.mat')
checkpoint_dir = '/root/autodl-tmp/checkpoints'
# checkpoint_dir = './checkpoints'
# data_train = hdf5storage.loadmat('./data/raw/trainData.mat')
# data_val = hdf5storage.loadmat('./data/raw/valData.mat')
print("load done")
# %%
# 主函数执行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-3
epochs = 1
batch_size = 10
shuffle_flag = True
model = EqaulizerFormer()
dataset_train = dataset_preprocess(data_train)
dataset_val = dataset_preprocess(data_val)
criterion = ComplexMSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle_flag, num_workers=4)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle_flag, num_workers=4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {count_parameters(model)}")

# %%
print('train model')
train_model(model, dataloader_train,dataloader_val, criterion, optimizer,scheduler, epochs, device)# evaluate_model(model, dataloader, criterion, device)


