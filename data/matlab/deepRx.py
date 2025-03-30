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

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution as used in the paper (implied) and Xception/MobileNet.
    Uses depth_multiplier=1 as a base, similar to standard MobileNet blocks.
    The paper mentions a multiplier of 2, which would mean doubling channels in depthwise.
    Here we implement the more standard version first.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        # Note: Paper uses (3,3) filters mostly. Dilation is applied.
        # Padding needs to be calculated to keep dimensions same: padding = (kernel_size - 1) * dilation // 2
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding = effective_kernel_size // 2

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResNetBlock(nn.Module):
    """
    Pre-activation ResNet Block based on Figure 3 and Table I.
    Uses Depthwise Separable Convolutions.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1,1)):
        super().__init__()
        # Effective padding calculation for dilation
        eff_k_h = (kernel_size - 1) * dilation[0] + 1
        eff_k_w = (kernel_size - 1) * dilation[1] + 1
        padding = (eff_k_h // 2, eff_k_w // 2)

        # Separable Conv 1 (includes BN -> ReLU -> Depthwise -> Pointwise(1x1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        # Paper uses depth_multiplier=2? Let's stick to 1 for now, standard separable conv
        # Depthwise part
        self.depthwise1 = nn.Conv2d(channels, channels, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=channels, bias=False)
        # Pointwise part (1x1) - Projects back to 'channels' dimension
        self.pointwise1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

        # Separable Conv 2
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        # Depthwise part
        self.depthwise2 = nn.Conv2d(channels, channels, kernel_size, stride=1,
                                    padding=padding, dilation=dilation, groups=channels, bias=False)
        # Pointwise part (1x1)
        self.pointwise2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.depthwise1(out)
        out = self.pointwise1(out) # Completes the first separable conv block implied in Fig 3

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.depthwise2(out)
        out = self.pointwise2(out) # Completes the second separable conv block

        out += residual # Add residual connection
        return out

class DeepRx(nn.Module):
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
    """
    def __init__(self, n_tx=2, n_rx=2, num_blocks=11, channels=[64, 128, 256]):
        super().__init__()
        self.n_tx = n_tx
        self.n_rx = n_rx

        # Calculate input channels after flattening antennas and real/imag
        # Input: csi_ls (Nt*Nr*2), tx_pilots (Nt*2), rx_signal (Nr*2)
        input_channels = (n_tx * n_rx * 2) + (n_tx * 2) + (n_rx * 2) # 8 + 4 + 4 = 16 for 2x2

        # --- Network Layers based on Table I ---
        # Initial Convolution (Conv. In)
        self.conv_in = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False) # 64 filters

        # ResNet Blocks
        # Dilation values from paper Table 1 for 11 blocks (adjust indices for 0-based)
        # Blocks 0,1: (1,1) -> channels[0] (64)
        # Blocks 2,3,4: (2,3) -> channels[1] (128)
        # Block 5: (3,6) -> channels[2] (256)
        # Block 6,7,8: (2,3) -> channels[1] (128)  <- paper goes back down here
        # Blocks 9,10: (1,1) -> channels[0] (64)
        # Note: Paper architecture goes 64->128->256->128->64. Let's follow that.

        self.resnet_blocks = nn.Sequential()
        current_channels = channels[0]

        # Block 0, 1 (dilation 1,1), channels[0]
        self.resnet_blocks.add_module("res0", ResNetBlock(current_channels, dilation=(1,1)))
        self.resnet_blocks.add_module("res1", ResNetBlock(current_channels, dilation=(1,1)))

        # Block 2 (transition to channels[1])
        self.resnet_blocks.add_module("proj2", nn.Sequential(
            nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[1], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[1]
        self.resnet_blocks.add_module("res2", ResNetBlock(current_channels, dilation=(2,3)))

        # Block 3, 4 (dilation 2,3), channels[1]
        self.resnet_blocks.add_module("res3", ResNetBlock(current_channels, dilation=(2,3)))
        self.resnet_blocks.add_module("res4", ResNetBlock(current_channels, dilation=(2,3)))

        # Block 5 (transition to channels[2])
        self.resnet_blocks.add_module("proj5", nn.Sequential(
             nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[2], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[2]
        self.resnet_blocks.add_module("res5", ResNetBlock(current_channels, dilation=(3,6))) # Paper uses (3,6) here

        # Block 6 (transition back to channels[1])
        self.resnet_blocks.add_module("proj6", nn.Sequential(
             nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[1], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[1]
        self.resnet_blocks.add_module("res6", ResNetBlock(current_channels, dilation=(2,3)))

        # Block 7, 8 (dilation 2,3), channels[1]
        self.resnet_blocks.add_module("res7", ResNetBlock(current_channels, dilation=(2,3)))
        self.resnet_blocks.add_module("res8", ResNetBlock(current_channels, dilation=(2,3)))

        # Block 9 (transition back to channels[0])
        self.resnet_blocks.add_module("proj9", nn.Sequential(
             nn.BatchNorm2d(current_channels), nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, channels[0], kernel_size=1, stride=1, bias=False) # Projection
        ))
        current_channels = channels[0]
        self.resnet_blocks.add_module("res9", ResNetBlock(current_channels, dilation=(1,1)))

        # Block 10 (dilation 1,1), channels[0]
        self.resnet_blocks.add_module("res10", ResNetBlock(current_channels, dilation=(1,1)))

        # Final Batch Norm and ReLU before output conv
        self.bn_out = nn.BatchNorm2d(current_channels)
        self.relu_out = nn.ReLU(inplace=True)

        # Output Convolution (Conv. Out)
        # Output should be the estimated tx_signal: [B, S, F, Nt, 2]
        # So, need Nt*2 output channels
        output_channels = n_tx * 2
        self.conv_out = nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True) # Use bias for output layer

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

        # Concatenate along the channel dimension
        x = torch.cat([csi_ls_r, tx_pilots_r, rx_signal_r], dim=1)

        # Pass through the network
        x = self.conv_in(x)
        x = self.resnet_blocks(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.conv_out(x) # Output shape: [B, Nt*2, S, F]

        # Reshape output to [B, S, F, Nt, 2]
        output = x.reshape(B, self.n_tx, 2, S, F).permute(0, 3, 4, 1, 2)

        return output 


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepRx().to(device)
    print('load model :', os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'))
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'), map_location=device)['model_state_dict'])
    print('load success.')
    return model


def infer(model, csi_ls, tx_signal, rx_signal):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot = torch.tensor(tx_signal, dtype=torch.float32) * pilot_mask
    pilot = torch.unsqueeze(pilot.to(device),0).contiguous()
    csi_ls = torch.unsqueeze(torch.tensor(csi_ls, dtype=torch.float32).to(device),0).contiguous()
    rx_signal = torch.unsqueeze(torch.tensor(rx_signal, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        equalized_signal = model(csi_ls, pilot, rx_signal)
    return np.asfortranarray(torch.squeeze(equalized_signal).cpu().numpy())


