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


# --- Dataset (Provided by User) ---
class MIMODataset(Dataset):
    """
    MIMO-OFDM 数据集加载器 (保持不变)
    """
    def __init__(self, tx_signal, rx_signal, csi_ls, pilot_mask):
        self.data_size = tx_signal.shape[0]
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.csi_ls = csi_ls
        self.pilot_mask = pilot_mask

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # CSI, TxPilots (Tx * Mask), Rx, Target Tx
        return (
            self.csi_ls[idx],                  # [S, F, Nt, Nr, 2]
            self.tx_signal[idx] * self.pilot_mask, # [S, F, Nt, 2] - 已应用掩码
            self.rx_signal[idx],               # [S, F, Nr, 2]
            self.tx_signal[idx]                # [S, F, Nt, 2] - Target
        )


# DepthwiseSeparableConv2d remains the same conceptually,
# but we implement its logic directly inside ResNetBlock for clarity with multiplier.

class ResNetBlock(nn.Module):
    """
    Pre-activation ResNet Block based on Figure 3 and Table I.
    Uses Depthwise Separable Convolutions with depth_multiplier=2.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1,1), depth_multiplier=2): # Added depth_multiplier argument
        super().__init__()
        # Effective padding calculation for dilation
        eff_k_h = (kernel_size - 1) * dilation[0] + 1
        eff_k_w = (kernel_size - 1) * dilation[1] + 1
        padding = (eff_k_h // 2, eff_k_w // 2)

        # Intermediate channels after depthwise conv
        inter_channels = channels * depth_multiplier # Key change for multiplier=2

        # --- Separable Conv 1 ---
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        # Depthwise part: outputs inter_channels
        self.depthwise1 = nn.Conv2d(channels, inter_channels, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=channels, bias=False)
        # Pointwise part (1x1): projects inter_channels back to channels
        self.pointwise1 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

        # --- Separable Conv 2 ---
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        # Depthwise part: outputs inter_channels
        self.depthwise2 = nn.Conv2d(channels, inter_channels, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=channels, bias=False)
        # Pointwise part (1x1): projects inter_channels back to channels
        self.pointwise2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        residual = x

        # --- Block 1 ---
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.depthwise1(out)  # Out: [B, channels*multiplier, S, F]
        out = self.pointwise1(out) # Out: [B, channels, S, F] - Projects back

        # --- Block 2 ---
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.depthwise2(out)  # Out: [B, channels*multiplier, S, F]
        out = self.pointwise2(out) # Out: [B, channels, S, F] - Projects back

        out += residual # Add residual connection
        return out

class DeepRx_MIMO_Equalizer(nn.Module):
    """
    DeepRx adaptation for MIMO-OFDM Equalization.
    Outputs estimated transmit symbols.
    Input dimensions based on MIMODataset:
    - csi_ls:    [B, S, F, Nt, Nr, 2]
    - tx_pilots: [B, S, F, Nt, 2] (Tx Signal * Pilot Mask)
    - rx_signal: [B, S, F, Nr, 2]
    Output dimension:
    - tx_est:    [B, S, F, Nt, 2]
    Internal Tensor Format: [B, C, S, F]

    Uses ResNetBlocks with depth_multiplier=2 internally.
    """
    def __init__(self, n_tx=2, n_rx=2, num_blocks=11, channels=[64, 128, 256]):
        super().__init__()
        self.n_tx = n_tx
        self.n_rx = n_rx
        # Specify depth_multiplier consistent with the paper
        self.depth_multiplier = 2

        # Calculate input channels after flattening antennas and real/imag
        input_channels = (n_tx * n_rx * 2) + (n_tx * 2) + (n_rx * 2) # 8 + 4 + 4 = 16 for 2x2

        # --- Network Layers based on Table I ---
        self.conv_in = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # ResNet Blocks - Instantiate ResNetBlock with depth_multiplier=2
        self.resnet_blocks = nn.Sequential()
        current_channels = channels[0]

        # --- Block Definitions based on Table I Channel Progression ---

        # Block 0, 1 (dilation 1,1), channels[0]
        self.resnet_blocks.add_module("res0", ResNetBlock(current_channels, dilation=(1,1), depth_multiplier=self.depth_multiplier))
        self.resnet_blocks.add_module("res1", ResNetBlock(current_channels, dilation=(1,1), depth_multiplier=self.depth_multiplier))

        # Block 2 (transition to channels[1])
        self.resnet_blocks.add_module("proj2", nn.Sequential(
            nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[1], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[1]
        self.resnet_blocks.add_module("res2", ResNetBlock(current_channels, dilation=(2,3), depth_multiplier=self.depth_multiplier))

        # Block 3, 4 (dilation 2,3), channels[1]
        self.resnet_blocks.add_module("res3", ResNetBlock(current_channels, dilation=(2,3), depth_multiplier=self.depth_multiplier))
        self.resnet_blocks.add_module("res4", ResNetBlock(current_channels, dilation=(2,3), depth_multiplier=self.depth_multiplier))

        # Block 5 (transition to channels[2])
        self.resnet_blocks.add_module("proj5", nn.Sequential(
             nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[2], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[2]
        self.resnet_blocks.add_module("res5", ResNetBlock(current_channels, dilation=(3,6), depth_multiplier=self.depth_multiplier)) # Paper uses (3,6) here

        # Block 6 (transition back to channels[1])
        self.resnet_blocks.add_module("proj6", nn.Sequential(
             nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[1], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[1]
        self.resnet_blocks.add_module("res6", ResNetBlock(current_channels, dilation=(2,3), depth_multiplier=self.depth_multiplier))

        # Block 7, 8 (dilation 2,3), channels[1]
        self.resnet_blocks.add_module("res7", ResNetBlock(current_channels, dilation=(2,3), depth_multiplier=self.depth_multiplier))
        self.resnet_blocks.add_module("res8", ResNetBlock(current_channels, dilation=(2,3), depth_multiplier=self.depth_multiplier))

        # Block 9 (transition back to channels[0])
        self.resnet_blocks.add_module("proj9", nn.Sequential(
             nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[0], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[0]
        self.resnet_blocks.add_module("res9", ResNetBlock(current_channels, dilation=(1,1), depth_multiplier=self.depth_multiplier))

        # Block 10 (dilation 1,1), channels[0]
        self.resnet_blocks.add_module("res10", ResNetBlock(current_channels, dilation=(1,1), depth_multiplier=self.depth_multiplier))

        # Final Batch Norm and ReLU before output conv
        self.bn_out = nn.BatchNorm2d(current_channels)
        self.relu_out = nn.ReLU(inplace=True)

        # Output Convolution (Conv. Out)
        output_channels = n_tx * 2
        self.conv_out = nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, csi_ls, tx_pilots, rx_signal):
        """
        Forward pass.
        Input shapes:
        - csi_ls:    [B, S, F, Nt, Nr, 2]
        - tx_pilots: [B, S, F, Nt, 2]
        - rx_signal: [B, S, F, Nr, 2]
        """
        B, S, F, _, _, _ = csi_ls.shape

        # Reshape and Permute inputs to [B, C, S, F] format
        csi_ls_r = csi_ls.permute(0, 3, 4, 5, 1, 2).reshape(B, self.n_tx * self.n_rx * 2, S, F)
        tx_pilots_r = tx_pilots.permute(0, 3, 4, 1, 2).reshape(B, self.n_tx * 2, S, F)
        rx_signal_r = rx_signal.permute(0, 3, 4, 1, 2).reshape(B, self.n_rx * 2, S, F)

        x = torch.cat([csi_ls_r, tx_pilots_r, rx_signal_r], dim=1)

        x = self.conv_in(x)
        x = self.resnet_blocks(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.conv_out(x) # Output shape: [B, Nt*2, S, F]

        output = x.reshape(B, self.n_tx, 2, S, F).permute(0, 3, 4, 1, 2)

        return output

def dataset_preprocess(data):
    # 将数据转换为PyTorch张量
    pilot_mask = torch.zeros((256, 14, 2 , 2), dtype=torch.float32)
    indices_ant1 = torch.tensor([
        17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77,
        81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 133, 137,
        141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189,
        193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241
    ])-1

    indices_ant2 = torch.tensor([
        18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78,
        82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 130, 134,
        138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186,
        190, 194, 198, 202, 206, 210, 214, 218, 222, 226, 230, 234, 240
    ])-1

    indices = torch.tensor(
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 
        77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 
        131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 
        147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 
        163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 
        179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 
        195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 
        211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 
        227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241]
    )-1

    pilot_mask[indices_ant1,:,0,:] = 1
    pilot_mask[indices_ant2,:,1,:] = 1
    pilot_mask = pilot_mask[indices]
    print(pilot_mask.shape)

    csi_ls = torch.tensor(data['csiLSData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    tx_signal = torch.tensor(data['txSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, 2]
    rx_signal = torch.tensor(data['rxSignalData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_rx, 2]
    csi = torch.tensor(data['csiLabelData'], dtype=torch.float32) #[data_size, n_subc, n_sym, n_tx, n_rx, 2]
    
    del data
    gc.collect()
    return MIMODataset(tx_signal, rx_signal, csi_ls, pilot_mask)



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
        
# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        for batch_idx, (csi, pilot, rx_signal, tx_signal) in enumerate(dataloader_train):
            csi = csi.to(device)
            pilot = pilot.to(device)
            rx_signal = rx_signal.to(device)
            tx_signal = tx_signal.to(device)
            optimizer.zero_grad()
            output = model(csi, pilot, rx_signal)
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
            for batch_idx, (csi, pilot, rx_signal, tx_signal) in enumerate(dataloader_val):
                csi = csi.to(device)
                pilot = pilot.to(device)
                rx_signal = rx_signal.to(device)
                tx_signal = tx_signal.to(device)
                output = model(csi, pilot, rx_signal)
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
    # data_train = hdf5storage.loadmat('F:/dataset/valDataFinal.mat')
    # data_val = hdf5storage.loadmat('F:/dataset/valDataFinal.mat')
    # print("load done")

    # 主函数执行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lr = 1e-3
    epochs = 20
    batch_size = 36
    shuffle_flag = True
    model = DeepRx()
    print(f"Total trainable parameters: {count_parameters(model)}")
    print('train model')
    dataset_train = dataset_preprocess(data_train)
    dataset_val = dataset_preprocess(data_val)
    criterion = ComplexMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle_flag, num_workers=4)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle_flag, num_workers=4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)


    train_model(model, dataloader_train,dataloader_val, criterion, optimizer,scheduler, epochs, device, checkpoint_dir)




