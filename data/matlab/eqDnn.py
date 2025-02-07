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
class DNNResEQ(nn.Module):
    def __init__(self, input_dim=12, output_dim=4, hidden_dim=128, num_blocks=4):
        super().__init__()
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 残差块堆叠
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_dim*2)
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, csi, rx_signal):
        csi = csi.reshape(*csi.shape[:3], -1)  # [batch_size, n_subc, n_sym, n_tx*n_rx*2]
        rx_signal = rx_signal.reshape(*rx_signal.shape[:3], -1) #[batch_size, n_subc, n_sym, n_rx*2]
        x = torch.cat([csi, rx_signal], dim=-1) # [batch_size, n_subc, n_sym, (n_tx*n_rx + n_rx)*2]
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        x = x.reshape(*x.shape[:3],2,2)
        return x

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNResEQ().to(device)
    print('load model :', os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'))
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'), map_location=device)['model_state_dict'])
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


