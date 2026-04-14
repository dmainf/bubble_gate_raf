"""
バブル型クラスタ学習（対照学習版）メインループ

事前に precompute.py を実行して SwiFT 潜在ベクトルをキャッシュしておくこと．

アーキテクチャ:
  全被験者の z → encoder → h（サーバーが直接観測）
  ℒ = ℒ_MSE(acc, RT) + λ · ℒ_InfoNCE(h vs バブルプロトタイプ)
  バブル割り当ては h 空間の最近傍プロトタイプで更新する．
"""

import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from dataset import HCPWMLatentDataset, compute_label_stats
from client_model import ClientModel
from bubble_manager import BubbleManager


DATA_DIR = Path(__file__).parent / "data"
N_ROUNDS = 60
LOCAL_EPOCHS = 2
LR = 1e-3
LAMBDA_CON = 0.5
TEMPERATURE = 0.1
SPLIT_EVERY = 5
MERGE_EVERY = 10
REASSIGN_EVERY = 3
SPLIT_THRESHOLD = 0.12   # バブル内の最大コサイン距離がこれを超えたら分割
MERGE_THRESHOLD = 0.05   # バブル間プロトタイプのコサイン距離がこれ以下なら統合
ANNEALING_RATE = 1.15    # 分割のたびに閾値を拡大（過分割を抑制）
MIN_SPLIT_SIZE = 5       # 分割後の最小バブルサイズ


def get_subject_ids():
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "latent_cache"])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_data(device):
    subject_ids = get_subject_ids()
    n_train = int(len(subject_ids) * 0.8)
    label_stats = compute_label_stats(subject_ids[:n_train])

    subject_datasets = {}
    for sub in subject_ids:
        ds = HCPWMLatentDataset([sub], label_stats=label_stats)
        if len(ds) > 0:
            subject_datasets[sub] = ds

    client_ids = list(subject_datasets.keys())
    return subject_ids, label_stats, subject_datasets, client_ids


def load_all_tensors(subject_datasets, device):
    """全被験者の (z, labels) テンソルをデバイスに展開"""
    data = {}
    for sid, ds in subject_datasets.items():
        z_list, acc_list, rt_list = [], [], []
        for sample in ds:
            z_list.append(sample["z"])
            acc_list.append(sample["acc_2bk"])
            rt_list.append(sample["rt_2bk"])
        data[sid] = {
            "z": torch.stack(z_list).to(device),
            "labels": torch.stack([torch.stack(acc_list), torch.stack(rt_list)], dim=1).to(device),
        }
    return data



@torch.no_grad()
def compute_h_subject(model, all_data):
    """全被験者の h ベクトル（ウィンドウ平均）を返す: {sid: tensor(128,)}"""
    model.eval()
    result = {}
    for sid, d in all_data.items():
        h = model.encode(d["z"])
        result[sid] = h.mean(0)
    model.train()
    return result


def save_figures(data_dir, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    with open(data_dir / "bubble_history.json") as f:
        bubble_history = json.load(f)
    with open(data_dir / "loss_history.json") as f:
        loss_history = json.load(f)
    with open(data_dir / "final_bubbles.json") as f:
        final_bubbles = json.load(f)

    rounds = [h["round"] for h in bubble_history]
    mean_losses = [h["mean_loss"] for h in bubble_history]
    n_bubbles = [len(h["bubbles"]) for h in bubble_history]

    # 1. 損失曲線 + バブル数
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(rounds, mean_losses, color="steelblue", marker="o", markersize=4, label="Mean loss")
    ax2.step(rounds, n_bubbles, color="tomato", where="post", linewidth=2, label="# Bubbles")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Mean loss", color="steelblue")
    ax2.set_ylabel("Number of bubbles", color="tomato")
    ax1.tick_params(axis="y", colors="steelblue")
    ax2.tick_params(axis="y", colors="tomato")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Loss & Bubble count per round")
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_bubbles.png", dpi=150)
    plt.close()

    # 2. バブルサイズの推移
    all_bubble_ids = sorted({bid for h in bubble_history for bid in h["bubbles"]}, key=int)
    size_matrix = np.array([[len(h["bubbles"].get(bid, [])) for bid in all_bubble_ids]
                             for h in bubble_history]).T
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab20(np.linspace(0, 1, len(all_bubble_ids)))
    ax.stackplot(rounds, size_matrix, labels=[f"B{bid}" for bid in all_bubble_ids],
                 colors=colors, alpha=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Number of clients")
    ax.set_title("Bubble size evolution")
    ax.legend(loc="upper left", ncol=4, fontsize=7)
    plt.tight_layout()
    plt.savefig(figures_dir / "bubble_sizes.png", dpi=150)
    plt.close()

    # 3. 最終バブルの acc vs RT 散布図
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = cm.tab20(np.linspace(0, 1, len(final_bubbles)))
    for (bid, members), color in zip(final_bubbles.items(), colors):
        accs = [m["acc_2bk"] for m in members if m["acc_2bk"] is not None]
        rts  = [m["rt_2bk"]  for m in members if m["rt_2bk"]  is not None]
        if accs and rts:
            ax.scatter(accs, rts, label=f"B{bid} (n={len(members)})", color=color, s=60, alpha=0.85)
    ax.set_xlabel("2-back accuracy")
    ax.set_ylabel("Median RT (ms)")
    ax.set_title("Final bubbles: acc vs RT")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_bubbles_scatter.png", dpi=150)
    plt.close()

    # 4. バブルごとの平均損失曲線
    all_bubble_ids = sorted({bid for h in bubble_history for bid in h["bubbles"]}, key=int)
    colors = cm.tab20(np.linspace(0, 1, len(all_bubble_ids)))
    bubble_color = {bid: c for bid, c in zip(all_bubble_ids, colors)}

    bubble_losses = {}
    for h in bubble_history:
        r = h["round"]
        for bid, clients in h["bubbles"].items():
            ls = [loss_history[c][r - 1] for c in clients if c in loss_history]
            if ls:
                bubble_losses.setdefault(bid, {})[r] = sum(ls) / len(ls)

    fig, ax = plt.subplots(figsize=(10, 5))
    for bid in all_bubble_ids:
        if bid not in bubble_losses:
            continue
        rs = sorted(bubble_losses[bid])
        ls = [bubble_losses[bid][r] for r in rs]
        ax.plot(rs, ls, color=bubble_color[bid], linewidth=1.5,
                marker="o", markersize=3, label=f"B{bid}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean loss")
    ax.set_title("Per-bubble mean loss curves")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(figures_dir / "bubble_losses.png", dpi=150)
    plt.close()

    # 5. バブルごとのコサイン距離（intra_dist）推移
    all_dist_bids = sorted({bid for h in bubble_history for bid in h.get("intra_dist", {})}, key=int)
    colors_d = cm.tab20(np.linspace(0, 1, len(all_dist_bids)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for bid, color in zip(all_dist_bids, colors_d):
        rs = [h["round"] for h in bubble_history if bid in h.get("intra_dist", {})]
        vs = [h["intra_dist"][bid] for h in bubble_history if bid in h.get("intra_dist", {})]
        ax.plot(rs, vs, color=color, linewidth=1.5, marker="o", markersize=3, label=f"B{bid}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Intra-bubble cosine distance")
    ax.set_title("Intra-bubble cosine distance per round")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(figures_dir / "intra_dist.png", dpi=150)
    plt.close()

    print(f"  figures → {figures_dir}")


def run(subject_datasets, client_ids, label_stats, results_dir,
        split_threshold=SPLIT_THRESHOLD, merge_threshold=MERGE_THRESHOLD,
        lambda_con=LAMBDA_CON, temperature=TEMPERATURE,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR,
        verbose=True):
    """
    対照学習付きバブル型クラスタ学習の1実験を実行して結果を保存する．

    Returns:
        dict: {n_final_bubbles, final_loss, n_splits, n_merges}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    torch.manual_seed(42)
    np.random.seed(42)

    all_data = load_all_tensors(subject_datasets, device)
    model = ClientModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    server = BubbleManager(client_ids,
                           split_threshold=split_threshold,
                           merge_threshold=merge_threshold)

    loss_history = {sid: [] for sid in client_ids}
    bubble_history = []
    n_splits, n_merges = 0, 0

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n=== Round {round_idx + 1}/{n_rounds} ===")
            print(f"  Bubbles: {server.bubble_summary()}")

        # ラウンド先頭でプロトタイプを凍結（frozen keys）
        h_subject_frozen = compute_h_subject(model, all_data)
        prototypes = server.get_prototypes(h_subject_frozen)

        model.train()
        for epoch in range(local_epochs):
            # 全被験者のh を勾配ありで計算
            h_batch = {}
            task_losses = []
            for sid in client_ids:
                z = all_data[sid]["z"]
                h = model.encode(z)                # (N_win, 128)  grad あり
                pred = model.head(h)               # (N_win, 2)
                task_losses.append(criterion(pred, all_data[sid]["labels"]))
                h_batch[sid] = h.mean(0)           # (128,)        grad あり

            task_loss = torch.stack(task_losses).mean()
            con_loss = server.infonce_loss(h_batch, prototypes, temperature=temperature)

            loss = task_loss + lambda_con * con_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 評価（no_grad）
        model.eval()
        with torch.no_grad():
            for sid in client_ids:
                h = model.encode(all_data[sid]["z"])
                loss_val = criterion(model.head(h), all_data[sid]["labels"]).item()
                loss_history[sid].append(loss_val)

        round_mean_loss = sum(loss_history[sid][-1] for sid in client_ids) / len(client_ids)
        if verbose:
            print(f"  task_loss={task_loss.item():.4f}  con_loss={con_loss.item() if hasattr(con_loss, 'item') else con_loss:.4f}  eval_loss={round_mean_loss:.4f}")

        # h を更新してバブル管理
        h_subject = compute_h_subject(model, all_data)
        intra_dist = server.intra_dist_per_bubble(h_subject)

        if (round_idx + 1) % REASSIGN_EVERY == 0:
            server.reassign(h_subject)
            if verbose:
                print(f"  [Reassign] {server.bubble_summary()}")

        if (round_idx + 1) % SPLIT_EVERY == 0:
            before = server.n_bubbles
            server.split_bubbles(h_subject, min_size=MIN_SPLIT_SIZE)
            after = server.n_bubbles
            if after > before:
                n_splits += after - before
            server.split_threshold *= ANNEALING_RATE
            if verbose:
                print(f"  [Split] {before} → {after} bubbles (threshold={server.split_threshold/ANNEALING_RATE:.3f}→{server.split_threshold:.3f})")

        if (round_idx + 1) % MERGE_EVERY == 0:
            before = server.n_bubbles
            server.merge_bubbles(h_subject)
            after = server.n_bubbles
            if after < before:
                n_merges += before - after
            if verbose:
                print(f"  [Merge] {before} → {after} bubbles")

        bubble_history.append({
            "round": round_idx + 1,
            "bubbles": {str(bid): sorted(clients) for bid, clients in server.bubbles.items()},
            "mean_loss": round_mean_loss,
            "intra_dist": {str(bid): v for bid, v in intra_dist.items()},
        })

    # 予測
    model.eval()
    predictions = []
    with torch.no_grad():
        for sid in client_ids:
            h = model.encode(all_data[sid]["z"])
            pred = model.head(h).cpu()
            for i in range(pred.shape[0]):
                predictions.append({
                    "subject": sid,
                    "bubble": server.client_to_bubble[sid],
                    "pred_acc": float(pred[i, 0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                    "pred_rt":  float(pred[i, 1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
                    "true_acc": float(all_data[sid]["labels"][i, 0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                    "true_rt":  float(all_data[sid]["labels"][i, 1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
                })

    # 最終バブルの行動成績
    final_bubbles = {}
    for bid, clients in server.bubbles.items():
        members = []
        for cid in sorted(clients):
            path = DATA_DIR / cid / "MNINonLinear" / "Results" / "tfMRI_WM_LR" / "EVs" / "WM_Stats.csv"
            if path.exists():
                df = pd.read_csv(path)
                acc = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "ACC")]["Value"].mean()
                rt  = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "MEDIAN_RT")]["Value"].mean()
                members.append({"subject": cid,
                                 "acc_2bk": float(acc) if acc == acc else None,
                                 "rt_2bk":  float(rt)  if rt  == rt  else None})
            else:
                members.append({"subject": cid, "acc_2bk": None, "rt_2bk": None})
        final_bubbles[str(bid)] = members

    data_dir = results_dir / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    with open(data_dir / "bubble_history.json", "w") as f:
        json.dump(bubble_history, f, indent=2)
    with open(data_dir / "final_bubbles.json", "w") as f:
        json.dump(final_bubbles, f, indent=2)
    with open(data_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    torch.save(model.state_dict(), data_dir / "model.pt")

    save_figures(data_dir, results_dir)

    if verbose:
        print(f"\n=== Training complete ===")
        print(f"Final bubbles: {server.bubble_summary()}")
        for bid, members in final_bubbles.items():
            accs = [m["acc_2bk"] for m in members if m["acc_2bk"] is not None]
            rts  = [m["rt_2bk"]  for m in members if m["rt_2bk"]  is not None]
            if accs:
                print(f"  Bubble {bid} ({len(members)} clients): acc={sum(accs)/len(accs):.3f}, rt={sum(rts)/len(rts):.0f}ms")

    return {
        "n_final_bubbles": server.n_bubbles,
        "final_loss": bubble_history[-1]["mean_loss"],
        "n_splits": n_splits,
        "n_merges": n_merges,
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    subject_ids, label_stats, subject_datasets, client_ids = prepare_data(device)
    print(f"Found subjects: {len(subject_ids)}, Active clients: {len(client_ids)}")
    if not client_ids:
        print("No active clients. Run precompute.py first.")
        return

    print(f"Label stats (train only): acc={label_stats['acc_mean']:.3f}±{label_stats['acc_std']:.3f}, "
          f"rt={label_stats['rt_mean']:.1f}±{label_stats['rt_std']:.1f}ms")

    run(subject_datasets, client_ids, label_stats,
        results_dir=Path(__file__).parent / "results",
        verbose=True)


if __name__ == "__main__":
    main()
