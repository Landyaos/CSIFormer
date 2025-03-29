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

# --- Native PyTorch Modules based Blocks (from previous implementation) ---

class EncoderBlockNative(nn.Module):
    """
    Encoder Block using nn.MultiheadAttention and a custom CNN Pre-Network.
    (Assumes pre-norm convention as implemented before)
    """
    def __init__(self, embed_dim, num_heads, n_encoder_filters, kernel_size=3, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_encoder_filters = n_encoder_filters

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        # CNN Pre-Network
        self.pre_conv1 = nn.Conv1d(embed_dim, n_encoder_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.pre_conv2 = nn.Conv1d(n_encoder_filters, n_encoder_filters, kernel_size=kernel_size, padding=kernel_size // 2)

        # Residual Projection
        if embed_dim != n_encoder_filters:
            self.residual_proj = nn.Conv1d(embed_dim, n_encoder_filters, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        self.final_norm = nn.LayerNorm(n_encoder_filters) # Applied on feature dimension

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn_output

        residual_pre_cnn = x
        x_norm = self.norm2(x)
        x_permuted = x_norm.permute(0, 2, 1) # (B, embed_dim, S)

        x_conv = self.pre_conv1(x_permuted)
        x_conv = self.activation(x_conv)
        x_conv = self.pre_conv2(x_conv) # (B, n_encoder_filters, S)

        residual_permuted = residual_pre_cnn.permute(0, 2, 1)
        residual_projected = self.residual_proj(residual_permuted)

        x_out = residual_projected + x_conv

        # Apply final LayerNorm (needs permutation for channel dim)
        x_out_permuted = x_out.permute(0, 2, 1) # (B, S, n_encoder_filters)
        x_out_norm = self.final_norm(x_out_permuted)
        x_final = x_out_norm.permute(0, 2, 1) # (B, n_encoder_filters, S)
        return x_final

class DecoderBlockNative(nn.Module):
    """ Residual Convolutional Block for Decoder """
    def __init__(self, n_decoder_filters, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_decoder_filters, n_decoder_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_decoder_filters, n_decoder_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(n_decoder_filters) # Applied on feature dimension

    def forward(self, x):
        # x: (batch, n_decoder_filters, seq_len)
        residual = x
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = y + residual

        # Apply LayerNorm (needs permutation for channel dim)
        y_permuted = y.permute(0, 2, 1) # (B, S, F)
        y_norm = self.norm(y_permuted)
        y_final = y_norm.permute(0, 2, 1) # (B, F, S)
        return y_final

# --- Main Offline Channelformer Model (Alternative Reshape) ---

class ChannelformerMIMOOfflineAltReshape(nn.Module):
    def __init__(self, n_subc, n_sym, n_tx, n_rx, n_pilot_sym,
                 embed_dim=128, num_heads=8, n_encoder_filters=5,
                 n_decoder_filters=12, num_decoder_blocks=3,
                 encoder_kernel_size=3, decoder_kernel_size=5,
                 dropout=0.0):
        super().__init__()
        self.n_subc = n_subc
        self.n_sym = n_sym
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_pilot_sym = n_pilot_sym

        # Calculate sequence lengths based on the new scheme
        self.seq_len_pilot = n_subc * n_pilot_sym  # 72 * 2 = 144
        self.seq_len_full = n_subc * n_sym         # 72 * 14 = 1008

        # Calculate input feature dimension based on the new scheme
        self.input_feature_dim = n_tx * n_rx * 2  # 2 * 2 * 2 = 8
        # Calculate output feature dimension needed before final reshape
        self.output_feature_dim = n_tx * n_rx * 2 # 8

        # --- Input Projection ---
        # Project the combined Tx-Rx-Real/Imag features to embed_dim
        self.input_proj = nn.Linear(self.input_feature_dim, embed_dim)

        # --- Encoder ---
        # Input: (B, seq_len_pilot, embed_dim)
        # Output: (B, n_encoder_filters, seq_len_pilot)
        self.encoder = EncoderBlockNative(embed_dim, num_heads, n_encoder_filters,
                                          kernel_size=encoder_kernel_size, dropout=dropout)

        # --- Resizing Layer ---
        # Maps from pilot sequence length (144) to full sequence length (1008)
        # Project features first, then resize sequence length
        # resize_proj: Acts on feature dim (n_encoder_filters -> n_decoder_filters)
        # resize_seq: Acts on sequence dim (seq_len_pilot -> seq_len_full)
        self.resize_proj = nn.Linear(n_encoder_filters, n_decoder_filters)
        self.resize_seq = nn.Linear(self.seq_len_pilot, self.seq_len_full)

        # --- Decoder ---
        # Input: (B, n_decoder_filters, seq_len_full)
        # Output: (B, n_decoder_filters, seq_len_full)
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlockNative(n_decoder_filters, kernel_size=decoder_kernel_size)
              for _ in range(num_decoder_blocks)]
        )

        # --- Output Projection ---
        # Maps decoder features back to the combined Tx-Rx-Real/Imag dimension (8)
        # Use Conv1d acting on the sequence
        self.output_proj = nn.Conv1d(n_decoder_filters, self.output_feature_dim,
                                     kernel_size=decoder_kernel_size,
                                     padding=decoder_kernel_size // 2)


    def forward(self, csi_ls):
        # Input csi_ls: (batch, n_subc, n_pilot_sym, n_tx, n_rx, 2)
        # Example: (B, 72, 2, 2, 2, 2)
        batch_size = csi_ls.size(0)

        # 1. Reshape input according to the new scheme
        # Target shape: (B, seq_len_pilot, input_feature_dim) = (B, 144, 8)
        x = csi_ls.permute(0, 1, 2, 3, 4, 5).contiguous() # [B, 72, 2, 2, 2, 2]
        x = x.view(batch_size, self.n_subc, self.n_pilot_sym, self.input_feature_dim) # [B, 72, 2, 8]
        x = x.view(batch_size, self.seq_len_pilot, self.input_feature_dim) # [B, 144, 8]

        # 2. Input Projection
        # Input: (B, 144, 8) -> Output: (B, 144, embed_dim)
        x = self.input_proj(x)

        # 3. Encoder
        # Input: (B, 144, embed_dim) -> Output: (B, n_encoder_filters, 144)
        encoded_features = self.encoder(x)

        # 4. Resizing
        # Permute for linear layers: (B, 144, n_encoder_filters)
        x_resize = encoded_features.permute(0, 2, 1)
        # Project features: (B, 144, n_decoder_filters)
        x_resize = self.resize_proj(x_resize)
        # Permute for sequence resize: (B, n_decoder_filters, 144)
        x_resize = x_resize.permute(0, 2, 1)
        # Resize sequence: (B, n_decoder_filters, seq_len_full=1008)
        x_resized = self.resize_seq(x_resize)

        # 5. Decoder
        # Input: (B, n_decoder_filters, 1008) -> Output: (B, n_decoder_filters, 1008)
        decoded_features = self.decoder_blocks(x_resized)

        # 6. Output Projection
        # Input: (B, n_decoder_filters, 1008) -> Output: (B, output_feature_dim=8, 1008)
        output_seq = self.output_proj(decoded_features)

        # 7. Reshape Output
        # Reshape (B, 8, 1008) back to (B, n_subc, n_sym, n_tx, n_rx, 2)
        # seq_len_full = n_subc * n_sym = 72 * 14 = 1008
        # output_feature_dim = n_tx * n_rx * 2 = 8
        output = output_seq.view(batch_size, self.n_tx, self.n_rx, 2, self.n_subc, self.n_sym)
        # Permute dimensions to match label format: (B, n_subc, n_sym, n_tx, n_rx, 2)
        output = output.permute(0, 4, 5, 1, 2, 3).contiguous()

        return output

# --- Example Usage ---
if __name__ == '__main__':
    # MIMO-OFDM Parameters (Example)
    BATCH_SIZE = 4
    N_SUBC = 72
    N_SYM = 14
    N_TX = 2
    N_RX = 2
    N_PILOT_SYM = 2

    # Model Hyperparameters (Match paper/adjust as needed)
    EMBED_DIM = 128
    NUM_HEADS = 8
    N_ENCODER_FILTERS = 5
    N_DECODER_FILTERS = 12
    NUM_DECODER_BLOCKS = 3
    ENCODER_KERNEL_SIZE = 3
    DECODER_KERNEL_SIZE = 5
    DROPOUT = 0.0

    # Instantiate the model with the alternative reshape logic
    model = ChannelformerMIMOOfflineAltReshape(
        n_subc=N_SUBC, n_sym=N_SYM, n_tx=N_TX, n_rx=N_RX, n_pilot_sym=N_PILOT_SYM,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        n_encoder_filters=N_ENCODER_FILTERS, n_decoder_filters=N_DECODER_FILTERS,
        num_decoder_blocks=NUM_DECODER_BLOCKS,
        encoder_kernel_size=ENCODER_KERNEL_SIZE,
        decoder_kernel_size=DECODER_KERNEL_SIZE,
        dropout=DROPOUT
    )

    # Create dummy input data (LS estimates at pilot symbols)
    dummy_csi_ls = torch.randn(BATCH_SIZE, N_SUBC, N_PILOT_SYM, N_TX, N_RX, 2)

    # Pass data through the model
    print(f"Input shape: {dummy_csi_ls.shape}")
    estimated_csi = model(dummy_csi_ls)

    # Check output shape
    print(f"Output shape: {estimated_csi.shape}")
    expected_shape = (BATCH_SIZE, N_SUBC, N_SYM, N_TX, N_RX, 2)
    print(f"Expected shape: {expected_shape}")
    assert estimated_csi.shape == expected_shape

    # Print model summary (optional, requires torchinfo)
    try:
        from torchinfo import summary
        summary(model, input_data=dummy_csi_ls)
    except ImportError:
        print("\nInstall torchinfo for model summary: pip install torchinfo")
    except Exception as e:
         print(f"\nError during torchinfo summary: {e}")





def dataset_preprocess(data):
    # 将数据转换为PyTorch张量
    tx_signal = torch.tensor(data['txSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, 2]
    rx_signal = torch.tensor(data['rxSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_rx, 2]
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


    def forward(self, output, target):
        """
        复数信道估计的均方误差 (MSE) 损失函数。
        x_py: (batch_size, csi_matrix, 2)，估计值
        y_py: (batch_size, csi_matrix, 2)，真实值
        """
        diff = output - target  # 差值，形状保持一致
        loss = torch.mean(diff[..., 0]**2 + diff[..., 1]**2)  # 实部和虚部平方和
        return loss


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
    data_train = hdf5storage.loadmat('/root/autodl-tmp/data/raw/trainData.mat')
    data_val = hdf5storage.loadmat('/root/autodl-tmp/data/raw/valData.mat')
    checkpoint_dir = '/root/autodl-tmp/checkpoints'
    # checkpoint_dir = './checkpoints'
    # data_train = hdf5storage.loadmat('./data/raw/trainData.mat')
    # data_val = hdf5storage.loadmat('./data/raw/valData.mat')
    print("load done")

    dataset_train = dataset_preprocess(data_train)
    dataset_val = dataset_preprocess(data_val)

    model = DNNResEQWithAttention()
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