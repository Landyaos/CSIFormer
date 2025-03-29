#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import numpy as np

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

class Channelformer(nn.Module):
    def __init__(self, n_subc=224, n_sym=14, n_tx=2, n_rx=2, n_pilot_sym=14,
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



def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Channelformer().to(device)
    print('load model :', os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'))
    model.load_state_dict(torch.load(os.path.join('../../checkpoints', model.__class__.__name__ + '_v1_best.pth'), map_location=device)['model_state_dict'])
    print('load success.')
    return model


def infer(model, csi_ls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csi_ls = torch.unsqueeze(torch.tensor(csi_ls, dtype=torch.float32).to(device),0).contiguous()
    model.eval()
    with torch.no_grad():
        csi_est = model(csi_ls)
    return np.asfortranarray(torch.squeeze(csi_est).cpu().numpy())

