import torch
import torch.nn as nn
import torch.nn.functional as F


class ClientModel(nn.Module):
    """
    Phase 1 モデル
    入力: SwiFT 潜在ベクトル (B, 288)
    出力: pred (B, 2) と h (B, 128)
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


class GateRAFModel(nn.Module):
    """
    Phase 2 モデル：GateRAF

    入力: h_q (N, hidden_dim), h_ret (N, hidden_dim)
      h_q   = バブル整地済み encoder の被験者平均表現（凍結）
      h_ret = メモリバンクから top-k 検索して集約した参照表現（凍結）
    出力: y_hat (N, 2), g (N, 1)

    g ≈ 1: 参照情報を積極的に取り込む
    g ≈ 0: 参照を棄却し自己完結で予測
    h_fused = h_q + g * h_ret
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.constant_(self.gate[2].bias, -3.0)

    def forward(self, h_q, h_ret):
        g = self.gate(torch.cat([h_q, h_ret], dim=-1))
        h_fused = h_q + g * h_ret
        return self.head(h_fused), g
