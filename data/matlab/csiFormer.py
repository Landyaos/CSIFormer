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
    def __init__(self, d_model=256, nhead=4, n_layers=4, n_tx=2, n_rx=2, max_len=7000):
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
                dim_feedforward=1024,
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
                 n_rx=2):
        """
        同时包含：
        1) CSIEncoder (编码器): 根据导频估计当前帧
        2) EnhancedCSIDecoder (解码器): 利用前 n 帧和当前帧初步估计进行增强
        :param d_model, nhead, n_layers: Transformer相关超参
        :param n_tx, n_rx: 发射/接收天线数
        :param n_frame: 前 n 帧参考数
        """
        super(CSIFormer, self).__init__()
        self.encoder = CSIEncoder(d_model, nhead, n_layers, n_rx, n_rx)
        self.decoder = EnhancedCSIDecoder(d_model, nhead, n_layers, n_tx, n_rx)


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


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSIFormer().to(device)
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'), map_location=device)['model_state_dict'])
    print('load model path : ',os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'))
    return model


def infer(model, csi_ls, pre_csi):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csi_ls = torch.unsqueeze(torch.tensor(csi_ls, dtype=torch.float32).to(device),0).contiguous()
    pre_csi = torch.unsqueeze(torch.tensor(pre_csi, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        csi_est = model(csi_ls ,pre_csi)
    return np.asfortranarray(torch.squeeze(csi_est).cpu().numpy())

