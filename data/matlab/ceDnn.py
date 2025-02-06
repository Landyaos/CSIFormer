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

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)

# 深度残差网络模型
class DNNResCELS(nn.Module):
    def __init__(self, hidden_dim=512, num_blocks=4, n_subc=224 ,n_sym=14 ,n_tx=2, n_rx=2):
        super().__init__()

        self.n_subc = n_subc
        self.n_sym = n_sym
        self.n_tx = n_tx
        self.n_rx = n_rx

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(n_subc * n_sym, hidden_dim),
            nn.ReLU()
        )
        
        # 残差块堆叠
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_dim*2)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, n_subc * n_sym)
        
    def forward(self, x):
        
        x = x.reshape(-1 ,self.n_subc * self.n_sym, self.n_tx * self.n_rx * 2)
        x = x.permute(0,2,1)
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        x = x.permute(0,2,1)
        x = x.reshape(-1, self.n_subc, self.n_sym, self.n_tx, self.n_rx, 2)
        return x

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


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNResCELS().to(device)
    print('load model :', os.path.join('../../checkpoints', model.__class__.__name__ + '_pro_latest.pth'))
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_pro_latest.pth'), map_location=device)['model_state_dict'])
    print('load success.')
    return model


def infer(model, csi_ls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csi_ls = torch.unsqueeze(torch.tensor(csi_ls, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        csi_est = model(csi_ls)
    return np.asfortranarray(torch.squeeze(csi_est).cpu().numpy())


