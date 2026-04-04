import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from client_model import ClientModel


class Client:
    """
    クライアントシミュレーション

    高速化ポイント:
    1. 初期化時に全データをデバイスメモリにロード（DataLoaderを廃止）
    2. フルバッチ学習（40サンプル×288次元は極めて軽量）
    """

    def __init__(self, client_id, dataset, device="cpu"):
        self.client_id = client_id
        self.device = device

        z_list, acc_list, rt_list = [], [], []
        for sample in dataset:
            z_list.append(sample["z"])
            acc_list.append(sample["acc_2bk"])
            rt_list.append(sample["rt_2bk"])

        self.z = torch.stack(z_list).to(device)                                   # (N, 288)
        self.labels = torch.stack([
            torch.stack(acc_list), torch.stack(rt_list)
        ], dim=1).to(device)                                                       # (N, 2)

    def local_train(self, model_state, n_epochs=1, lr=1e-3):
        model = ClientModel()
        model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()})
        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(n_epochs):
            pred = model(self.z)
            loss = criterion(pred, self.labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        updates = {}
        for name, param in model.named_parameters():
            updates[name] = param.detach() - model_state[name].to(self.device)

        # dist_match 出力 h の統計量を計算（バブル更新に使用）
        with torch.no_grad():
            h = model.dist_match(self.z)   # (N, 128)
        mu = h.mean(dim=0)                 # (128,)
        sigma = h.std(dim=0)               # (128,)

        return updates, (mu, sigma), loss.item()
