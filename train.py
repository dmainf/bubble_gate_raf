"""
バブル型クラスタ連合学習（CFL）メインループ

事前にprecompute.pyを実行してSwiFT潜在ベクトルをキャッシュしておくこと．

アーキテクチャ（research_plan.txt セクション3参照）:
  クライアント: キャッシュ済み潜在ベクトル → ClientModel → Δw → サーバーへ送信
  サーバー: 勾配のコサイン類似度でバブルを管理 → バブル内FedAvg → モデルをクライアントへ返す
"""

import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from dataset import HCPWMLatentDataset, compute_label_stats
from client_model import ClientModel
from client import Client
from bubble_manager import BubbleManager


DATA_DIR = Path(__file__).parent / "data"
N_ROUNDS = 30
LOCAL_EPOCHS = 2
LR = 1e-3
SPLIT_EVERY = 5
MERGE_EVERY = 10
SPLIT_THRESHOLD = 1.0
MERGE_THRESHOLD = 2.0
ANNEALING_RATE = 2.0


def save_figures(results_dir, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    with open(results_dir / "bubble_history.json") as f:
        bubble_history = json.load(f)
    with open(results_dir / "loss_history.json") as f:
        loss_history = json.load(f)
    with open(results_dir / "final_bubbles.json") as f:
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

    # 3. 最終バブルのacc vs RT散布図
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

    # 4. バブルごとの平均損失曲線（ラウンドごとのバブル帰属を使用）
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

    # 5. バブルごとの W2² 推移
    all_w2_bids = sorted({bid for h in bubble_history for bid in h.get("intra_w2", {})}, key=int)
    colors_w2 = cm.tab20(np.linspace(0, 1, len(all_w2_bids)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for bid, color in zip(all_w2_bids, colors_w2):
        rs = [h["round"] for h in bubble_history if bid in h.get("intra_w2", {})]
        vs = [h["intra_w2"][bid] for h in bubble_history if bid in h.get("intra_w2", {})]
        ax.plot(rs, vs, color=color, linewidth=1.5, marker="o", markersize=3, label=f"B{bid}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Intra-bubble W2²")
    ax.set_title("Intra-bubble W2² per round (per bubble)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(figures_dir / "intra_w2.png", dpi=150)
    plt.close()

    print(f"  figures → {figures_dir}")


def get_subject_ids():
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "latent_cache"])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_data(device):
    """データセットの準備（sweep.pyと共有）"""
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


def run(subject_datasets, client_ids, label_stats, results_dir,
        split_threshold=SPLIT_THRESHOLD, merge_threshold=MERGE_THRESHOLD,
        annealing_rate=ANNEALING_RATE,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR,
        verbose=True):
    """
    CFLの1実験を実行して結果を保存する．
    sweep.pyからも呼ばれる．

    Returns:
        dict: {n_final_bubbles, final_loss, n_merges, n_splits}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    initial_state = {k: v.clone() for k, v in ClientModel().state_dict().items()}
    server = BubbleManager(client_ids, initial_state,
                           split_threshold=split_threshold,
                           merge_threshold=merge_threshold)
    clients = {sub: Client(sub, subject_datasets[sub], device=device) for sub in client_ids}

    loss_history = {}
    bubble_history = []
    n_splits, n_merges = 0, 0

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n=== Round {round_idx + 1}/{n_rounds} ===")
            print(f"  Bubbles: {server.bubble_summary()}")

        model_states = {cid: server.get_model_state(cid) for cid in client_ids}
        all_updates, all_stats = {}, {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(clients[cid].local_train, model_states[cid], local_epochs, lr): cid
                for cid in client_ids
            }
            for i, future in enumerate(as_completed(futures)):
                cid = futures[future]
                updates, stats, loss = future.result()
                all_updates[cid] = updates
                all_stats[cid] = stats
                loss_history.setdefault(cid, []).append(loss)
                if verbose:
                    print(f"  [{i+1}/{len(client_ids)}] {cid} loss={loss:.4f}")

        round_mean_loss = sum(loss_history[c][-1] for c in client_ids) / len(client_ids)
        if verbose:
            print(f"  --> Mean loss: {round_mean_loss:.4f}")

        for bubble_id, bubble in server.bubbles.items():
            bubble_updates = {c: all_updates[c] for c in bubble["clients"] if c in all_updates}
            if bubble_updates:
                server.fedavg(bubble_id, bubble_updates)

        if (round_idx + 1) % SPLIT_EVERY == 0:
            before = server.n_bubbles
            if round_idx > 0:
                server.split_threshold *= annealing_rate
            server.split_bubbles(all_stats)
            after = server.n_bubbles
            if after > before:
                n_splits += after - before
            if verbose:
                print(f"  [Split] threshold={server.split_threshold:.3f}, {before} → {after} bubbles")

        if (round_idx + 1) % MERGE_EVERY == 0:
            before = server.n_bubbles
            server.merge_bubbles(all_stats)
            after = server.n_bubbles
            if after < before:
                n_merges += before - after
                if verbose:
                    print(f"  [Merge] {before} → {after} bubbles")

        intra_w2 = server.intra_w2_per_bubble(all_stats)
        bubble_history.append({
            "round": round_idx + 1,
            "bubbles": {str(bid): sorted(b["clients"]) for bid, b in server.bubbles.items()},
            "mean_loss": round_mean_loss,
            "intra_w2": {str(bid): v for bid, v in intra_w2.items()},
        })

    # 予測
    predictions = []
    for cid in client_ids:
        model_state = server.get_model_state(cid)
        model = ClientModel()
        model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
        model = model.to(device)
        model.eval()
        loader = DataLoader(subject_datasets[cid], batch_size=8, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                z = batch["z"].to(device)
                pred = model(z).cpu()
                for i in range(pred.shape[0]):
                    predictions.append({
                        "subject": cid,
                        "bubble": server.client_to_bubble[cid],
                        "window_idx": int(batch["window_idx"][i]),
                        "pred_acc": float(pred[i, 0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                        "pred_rt":  float(pred[i, 1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
                        "true_acc": float(batch["acc_2bk"][i] * label_stats["acc_std"] + label_stats["acc_mean"]),
                        "true_rt":  float(batch["rt_2bk"][i]  * label_stats["rt_std"]  + label_stats["rt_mean"]),
                    })

    # 最終バブルの行動成績
    final_bubbles = {}
    for bid, bubble in server.bubbles.items():
        members = []
        for cid in sorted(bubble["clients"]):
            path = DATA_DIR / cid / "MNINonLinear" / "Results" / "tfMRI_WM_LR" / "EVs" / "WM_Stats.csv"
            if path.exists():
                df = pd.read_csv(path)
                acc = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "ACC")]["Value"].mean()
                rt = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "MEDIAN_RT")]["Value"].mean()
                members.append({"subject": cid,
                                 "acc_2bk": float(acc) if acc == acc else None,
                                 "rt_2bk":  float(rt)  if rt  == rt  else None})
            else:
                members.append({"subject": cid, "acc_2bk": None, "rt_2bk": None})
        final_bubbles[str(bid)] = members

    # 保存: JSON・ptはdata/サブディレクトリへ、グラフはresults_dir直下へ
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
    for bid, bubble in server.bubbles.items():
        torch.save(bubble["state"], data_dir / f"bubble_{bid}.pt")

    save_figures(data_dir, results_dir)

    if verbose:
        print(f"\n=== Training complete ===")
        print(f"Final bubble configuration: {server.bubble_summary()}")
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
