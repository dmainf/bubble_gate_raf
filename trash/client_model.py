import torch.nn as nn


class ClientModel(nn.Module):
    """
    クライアント側モデル（凍結エンコーダの後段）
    入力: SwiFT出力をpoolingした潜在ベクトル (B, 288)
    出力: [acc_2bk, rt_2bk] の予測 (B, 2)

    dist_match: 分布マッチング層 — バブル内でベクトル空間を整地する
    head: 行動予測ヘッド — 反応時間・正答率を回帰する
    """

    LATENT_DIM = 288  # SwiFT embed_dim * c_multiplier^3 = 36 * 8

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.dist_match = nn.Sequential(
            nn.Linear(self.LATENT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, z):
        return self.head(self.dist_match(z))
