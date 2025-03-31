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

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNResEQWithAttention().to(device)
    print('load model :', os.path.join('../../checkpoints', model.__class__.__name__ + '_v3.pth'))
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_v3.pth'), map_location=device)['model_state_dict'])
    print('load success.')
    return model


def infer(model, csi, rx_signal):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csi = torch.unsqueeze(torch.tensor(csi, dtype=torch.float32).to(device),0).contiguous()
    rx_signal = torch.unsqueeze(torch.tensor(rx_signal, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        equalized_signal = model(csi, rx_signal)
    return np.asfortranarray(torch.squeeze(equalized_signal).cpu().numpy())


