import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import hdf5storage
import matplotlib.pyplot as plt
import torch.optim as optim
###############################################################################
# 第一部分：CSIFormer (编码器)
###############################################################################
class CSIEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=2, n_layers=4, n_tx=2, n_rx=2):
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

###############################################################################
# 第二部分：EnhancedCSIDecoder (解码器)
###############################################################################
class EnhancedCSIDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=2, n_layers=4, n_tx=2, n_rx=2):
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

###############################################################################
# CSIFormer：同时包含 Encoder 和 Decoder，批维在前
###############################################################################
class CSIFormer(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=2, 
                 n_layers=4, 
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
    
class CSIFormerLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8):
        """
        :param alpha: 第一部分损失的权重
        :param beta:  第二部分损失的权重
        """
        super(CSIFormerLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, csi_enc, csi_dec, csi_true):
        """
        :param csi_enc: 第一部分(编码器)的输出
        :param csi_dec: 第二部分(解码器)的输出
        :param csi_true: 真实的目标CSI
        :return: (total_loss, loss_enc, loss_dec)
        """
        # 计算编码器的损失
        loss_enc = self.mse_loss(csi_enc, csi_true)
        
        # 计算解码器的损失
        loss_dec = self.mse_loss(csi_dec, csi_true)
        
        # 加权合并
        total_loss = self.alpha * loss_enc + self.beta * loss_dec
        
        return total_loss, loss_enc, loss_dec

def load_model(model_name = 'JointCEEQ'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointCEEQ().to(device)
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_best.pth'), map_location=device)['model_state_dict'])
    return model


def infer3(model, tx_pilot, rx_pilot, pre_csi, rx_signal):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx_pilot = torch.unsqueeze(torch.tensor(tx_pilot, dtype=torch.float32).to(device),0).contiguous()
    rx_pilot = torch.unsqueeze(torch.tensor(rx_pilot, dtype=torch.float32).to(device),0).contiguous()
    pre_csi = torch.unsqueeze(torch.tensor(pre_csi, dtype=torch.float32).to(device),0).contiguous()
    rx_signal = torch.unsqueeze(torch.tensor(rx_signal, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        csi_enc, csi_dec, equalized_signal = model(tx_pilot, rx_pilot, pre_csi, rx_signal)
    equalized_signal = torch.squeeze(equalized_signal).cpu().numpy()
    return equalized_signal


