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
    def __init__(self, d_model=256, nhead=4, n_layers=4, n_tx=2, n_rx=2, max_len=7000):
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
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=n_layers
        )

        self.output_proj = nn.Linear(d_model, n_tx * n_rx * 2)

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

        # 映射到 CSI 空间
        csi_est = self.output_proj(output_features)  # [B, seq_len, n_tx * n_rx * 2]

        # 恢复形状为 [B, n_subc, n_sym, n_tx, n_rx, 2]
        csi_est = csi_est.view(B, n_subc, n_sym, n_tx, n_rx, 2)
        return csi_est


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSIEncoder().to(device)
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'), map_location=device)['model_state_dict'])
    print('load model path : ',os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'))
    return model


def infer(model, csi_ls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csi_ls = torch.unsqueeze(torch.tensor(csi_ls, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        csi_est = model(csi_ls)
    return np.asfortranarray(torch.squeeze(csi_est).cpu().numpy())

