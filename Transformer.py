import torch
import torch.nn as nn


class DynamicGraphTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(DynamicGraphTransformer, self).__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, G_seq):
        seq_len = G_seq.size(0)
        src = G_seq.unsqueeze(0)

        if seq_len <= self.pos_encoder.size(1):
            pos_encoding = self.pos_encoder[:, :seq_len, :]
            src = src + pos_encoding

        output = self.transformer_encoder(src)
        return output.squeeze(0)

