import torch
import torch.nn.functional as F


class BubbleManager:
    """
    プロトタイプベースのバブル管理（非連合学習版）

    サーバーが全クライアントの h = encoder(z).mean(0) を直接観測できる前提．

    バブル判定の基盤：
      各バブルの代表点 c_B = mean({ h_i | i ∈ B }) を管理し，
      コサイン距離でクラスタを分割・結合する．

    split_threshold: バブル内の最大コサイン距離がこの値を超えたら分割
    merge_threshold: バブル間プロトタイプのコサイン距離がこの値以下なら結合
    """

    def __init__(self, client_ids, split_threshold=0.3, merge_threshold=0.05):
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.bubbles = {0: set(client_ids)}
        self.client_to_bubble = {c: 0 for c in client_ids}
        self._next_id = 1

    @property
    def n_bubbles(self):
        return len(self.bubbles)

    def bubble_summary(self):
        return {bid: len(clients) for bid, clients in self.bubbles.items()}

    def get_prototypes(self, h_subject):
        """h_subject: {client_id: tensor(128,)} → {bubble_id: prototype tensor}"""
        result = {}
        for bid, clients in self.bubbles.items():
            vecs = [h_subject[c] for c in clients if c in h_subject]
            if vecs:
                result[bid] = torch.stack(vecs).mean(0)
        return result

    def infonce_loss(self, h_batch, prototypes, temperature=0.1):
        """
        PCL スタイルの InfoNCE 損失

        h_batch:    {client_id: tensor(128,)}  勾配あり（クエリ）
        prototypes: {bubble_id: tensor(128,)}  frozen（キー）

        ℒ = -log[ exp(sim(h_i, c_B) / τ) / Σ_{B'} exp(sim(h_i, c_B') / τ) ]
        """
        if len(prototypes) < 2:
            return h_batch[next(iter(h_batch))].new_tensor(0.0)

        bubble_ids = sorted(prototypes.keys())
        proto_stack = torch.stack([prototypes[b] for b in bubble_ids]).detach()
        proto_norm = F.normalize(proto_stack, dim=1)  # (K, 128)

        losses = []
        for cid, h in h_batch.items():
            bid = self.client_to_bubble.get(cid)
            if bid not in prototypes:
                continue
            h_norm = F.normalize(h.unsqueeze(0), dim=1)          # (1, 128)
            logits = (h_norm @ proto_norm.T).squeeze(0) / temperature  # (K,)
            label = torch.tensor([bubble_ids.index(bid)], device=h.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), label))

        return torch.stack(losses).mean() if losses else h_batch[next(iter(h_batch))].new_tensor(0.0)

    def reassign(self, h_subject):
        """最近傍プロトタイプに各クライアントを再割り当て"""
        prototypes = self.get_prototypes(h_subject)
        if len(prototypes) < 2:
            return

        bubble_ids = sorted(prototypes.keys())
        proto_norm = F.normalize(
            torch.stack([prototypes[b] for b in bubble_ids]), dim=1
        )

        new_bubbles = {bid: set() for bid in bubble_ids}
        for cid, h in h_subject.items():
            h_norm = F.normalize(h.unsqueeze(0), dim=1)
            nearest = bubble_ids[(h_norm @ proto_norm.T).squeeze(0).argmax().item()]
            new_bubbles[nearest].add(cid)
            self.client_to_bubble[cid] = nearest

        self.bubbles = {bid: clients for bid, clients in new_bubbles.items() if clients}

    def split_bubbles(self, h_subject):
        """バブル内の最大コサイン距離が split_threshold を超えたら2分割"""
        new_bubbles = {}
        new_assignments = {}

        for bid, clients in self.bubbles.items():
            active = [c for c in clients if c in h_subject]
            if len(active) < 2:
                new_bubbles[bid] = clients
                for c in clients:
                    new_assignments[c] = bid
                continue

            vecs = F.normalize(torch.stack([h_subject[c] for c in active]), dim=1)
            dist = 1 - vecs @ vecs.T  # コサイン距離行列 (n, n)

            if dist.triu(diagonal=1).max().item() <= self.split_threshold:
                new_bubbles[bid] = clients
                for c in clients:
                    new_assignments[c] = bid
            else:
                group_a, group_b = self._bipartition(active, dist)
                for group in (group_a, group_b):
                    if not group:
                        continue
                    new_id = self._next_id
                    self._next_id += 1
                    new_bubbles[new_id] = set(group)
                    for c in group:
                        new_assignments[c] = new_id

        self.bubbles = new_bubbles
        self.client_to_bubble = new_assignments

    def merge_bubbles(self, h_subject):
        """バブル間プロトタイプのコサイン距離が merge_threshold 以下なら結合"""
        prototypes = self.get_prototypes(h_subject)
        bubble_ids = list(self.bubbles.keys())
        if len(bubble_ids) < 2:
            return

        to_merge = {}
        for i, bid_a in enumerate(bubble_ids):
            for bid_b in bubble_ids[i + 1:]:
                if bid_a not in prototypes or bid_b not in prototypes:
                    continue
                a = F.normalize(prototypes[bid_a].unsqueeze(0), dim=1)
                b = F.normalize(prototypes[bid_b].unsqueeze(0), dim=1)
                dist = 1 - (a @ b.T).item()
                if dist < self.merge_threshold:
                    to_merge[bid_b] = bid_a

        for src, dst in to_merge.items():
            if src not in self.bubbles or dst not in self.bubbles:
                continue
            self.bubbles[dst].update(self.bubbles[src])
            for c in self.bubbles[src]:
                self.client_to_bubble[c] = dst
            del self.bubbles[src]

    def intra_dist_per_bubble(self, h_subject):
        """バブルごとの平均ペアコサイン距離 {bid: float}"""
        result = {}
        for bid, clients in self.bubbles.items():
            active = [c for c in clients if c in h_subject]
            if len(active) < 2:
                result[bid] = 0.0
                continue
            vecs = F.normalize(torch.stack([h_subject[c] for c in active]), dim=1)
            dist = 1 - vecs @ vecs.T
            n = len(active)
            result[bid] = float(dist.triu(diagonal=1).sum() / (n * (n - 1) / 2))
        return result

    def _bipartition(self, clients, dist_matrix):
        """コサイン距離が最大のペアを軸に2グループへ分割"""
        n = len(clients)
        i_max, j_max, max_val = 0, 1, dist_matrix[0, 1].item()
        for i in range(n):
            for j in range(i + 1, n):
                v = dist_matrix[i, j].item()
                if v > max_val:
                    max_val, i_max, j_max = v, i, j
        group_a, group_b = [], []
        for k in range(n):
            if dist_matrix[k, i_max].item() <= dist_matrix[k, j_max].item():
                group_a.append(clients[k])
            else:
                group_b.append(clients[k])
        return group_a, group_b
