import torch
from copy import deepcopy


class BubbleManager:
    """
    サーバー側バブル管理（CFLのサーバー役）

    バブル更新の根拠：
    - クライアントから送られてくる dist_match 出力 h の統計量 {μ, σ} を使い，
      2-Wasserstein 距離の近似 W2²(i,j) = ‖μ_i−μ_j‖² + ‖σ_i−σ_j‖² でクラスタを管理する

    split_threshold: バブル内の最大 W2² がこの値を超えたら分割
    merge_threshold: バブル間の W2² がこの値を下回ったら結合
    """

    def __init__(self, client_ids, model_state, split_threshold=1.0, merge_threshold=0.5):
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.bubbles = {0: {"clients": set(client_ids), "state": deepcopy(model_state)}}
        self.client_to_bubble = {c: 0 for c in client_ids}

    @property
    def n_bubbles(self):
        return len(self.bubbles)

    def bubble_summary(self):
        return {bid: len(b["clients"]) for bid, b in self.bubbles.items()}

    def get_model_state(self, client_id):
        bubble_id = self.client_to_bubble[client_id]
        return deepcopy(self.bubbles[bubble_id]["state"])

    def fedavg(self, bubble_id, client_updates):
        """バブル内 FedAvg"""
        state = self.bubbles[bubble_id]["state"]
        n = len(client_updates)
        new_state = {}
        for name in state:
            avg_delta = sum(upd[name] for upd in client_updates.values()) / n
            new_state[name] = state[name].to(avg_delta.device) + avg_delta
        self.bubbles[bubble_id]["state"] = new_state

    @staticmethod
    def _w2_sq(mu_a, sig_a, mu_b, sig_b):
        """2-Wasserstein 距離の二乗（ガウス近似）"""
        return ((mu_a - mu_b) ** 2).sum() + ((sig_a - sig_b) ** 2).sum()

    def split_bubbles(self, client_stats):
        """W2² 距離に基づいてバブルを分割

        client_stats: {cid: (mu, sigma)}  mu/sigma ∈ R^{128}
        """
        new_bubbles = {}
        new_client_to_bubble = {}
        new_id = 0

        for bubble_id, bubble in self.bubbles.items():
            clients = [c for c in bubble["clients"] if c in client_stats]
            if len(clients) <= 1:
                new_bubbles[new_id] = {"clients": set(bubble["clients"]), "state": bubble["state"]}
                for c in bubble["clients"]:
                    new_client_to_bubble[c] = new_id
                new_id += 1
                continue

            mus  = torch.stack([client_stats[c][0] for c in clients])   # (n, 128)
            sigs = torch.stack([client_stats[c][1] for c in clients])   # (n, 128)

            mu_diff  = mus.unsqueeze(1)  - mus.unsqueeze(0)             # (n, n, 128)
            sig_diff = sigs.unsqueeze(1) - sigs.unsqueeze(0)
            w2_sq = (mu_diff ** 2).sum(dim=2) + (sig_diff ** 2).sum(dim=2)  # (n, n)

            if w2_sq.max().item() <= self.split_threshold:
                new_bubbles[new_id] = {"clients": set(bubble["clients"]), "state": bubble["state"]}
                for c in bubble["clients"]:
                    new_client_to_bubble[c] = new_id
                new_id += 1
            else:
                group_a, group_b = self._bipartition(clients, w2_sq)
                for group in (group_a, group_b):
                    if not group:
                        continue
                    new_bubbles[new_id] = {"clients": set(group), "state": deepcopy(bubble["state"])}
                    for c in group:
                        new_client_to_bubble[c] = new_id
                    new_id += 1

        self.bubbles = new_bubbles
        self.client_to_bubble = new_client_to_bubble

    def merge_bubbles(self, client_stats):
        """W2² 距離に基づいてバブルを結合

        client_stats: {cid: (mu, sigma)}
        """
        bubble_ids = list(self.bubbles.keys())
        if len(bubble_ids) < 2:
            return

        # バブルごとの代表点（μ̄, σ̄）
        bubble_rep = {}
        for bid, bubble in self.bubbles.items():
            clients = [c for c in bubble["clients"] if c in client_stats]
            if clients:
                bubble_rep[bid] = (
                    torch.stack([client_stats[c][0] for c in clients]).mean(0),
                    torch.stack([client_stats[c][1] for c in clients]).mean(0),
                )

        to_merge = {}
        for i, bid_a in enumerate(bubble_ids):
            for bid_b in bubble_ids[i + 1:]:
                if bid_a not in bubble_rep or bid_b not in bubble_rep:
                    continue
                mu_a, sig_a = bubble_rep[bid_a]
                mu_b, sig_b = bubble_rep[bid_b]
                w2 = self._w2_sq(mu_a, sig_a, mu_b, sig_b).item()
                if w2 < self.merge_threshold:
                    to_merge[bid_b] = bid_a

        for src, dst in to_merge.items():
            if src not in self.bubbles or dst not in self.bubbles:
                continue
            self.bubbles[dst]["clients"].update(self.bubbles[src]["clients"])
            for c in self.bubbles[src]["clients"]:
                self.client_to_bubble[c] = dst
            del self.bubbles[src]

    def _bipartition(self, clients, w2_sq_matrix):
        """W2² が最大のペアを軸に2グループへ分割"""
        n = len(clients)
        i_max, j_max = 0, 1
        max_val = w2_sq_matrix[0, 1].item()
        for i in range(n):
            for j in range(i + 1, n):
                v = w2_sq_matrix[i, j].item()
                if v > max_val:
                    max_val, i_max, j_max = v, i, j

        group_a, group_b = [], []
        for k in range(n):
            if w2_sq_matrix[k, i_max].item() <= w2_sq_matrix[k, j_max].item():
                group_a.append(clients[k])
            else:
                group_b.append(clients[k])
        return group_a, group_b
