import torch.nn as nn
import torch.nn.functional as F


class ClientModel(nn.Module):
    """
    入力: SwiFT 潜在ベクトル (B, 288)
    出力: pred (B, 2) と h (B, 128)

    encoder: 表現空間を構築する層（対照学習の対象）
    head:    行動予測ヘッド（acc_2bk, rt_2bk の回帰）
    """

    LATENT_DIM = 288

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.LATENT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def encode(self, z):
        return self.encoder(z)

    def forward(self, z):
        h = self.encode(z)
        return self.head(h), h
