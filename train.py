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
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
sys.path.insert(0, str(Path(__file__).parent))

from dataset import HCPWMLatentDataset, compute_label_stats
from client_model import ClientModel, GateRAFModel
from bubble_manager import BubbleManager


DATA_DIR = Path(__file__).parent / "data"
N_ROUNDS = 30
LOCAL_EPOCHS = 2
LR = 1e-3
LAMBDA_CON = 0.5
TEMPERATURE = 0.1
SPLIT_EVERY = 5
MERGE_EVERY = 10
REASSIGN_EVERY = 3
SPLIT_THRESHOLD = 0.05   # バブル内プロトタイプ平均コサイン距離がこれを超えたら分割
MERGE_THRESHOLD = 0.05   # バブル間プロトタイプのコサイン距離がこれ以下なら統合
ANNEALING_RATE = 1.15    # 分割のたびに閾値を拡大（過分割を抑制）
TOP_K = 5                # Phase 2 メモリバンク top-k 件数
MIN_SPLIT_SIZE = TOP_K + 2  # 分割後の最小バブルサイズ: |B| > k+1 を保証（tex 条件と一致）
N_INIT_BUBBLES = 4      # Round 0 で L_con を有効にするための初期バブル数


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
    train_ids = subject_ids[:n_train]
    test_ids  = subject_ids[n_train:]
    label_stats = compute_label_stats(train_ids)

    train_datasets = {}
    for sub in train_ids:
        ds = HCPWMLatentDataset([sub], label_stats=label_stats)
        if len(ds) > 0:
            train_datasets[sub] = ds

    test_datasets = {}
    for sub in test_ids:
        ds = HCPWMLatentDataset([sub], label_stats=label_stats)
        if len(ds) > 0:
            test_datasets[sub] = ds

    return train_ids, test_ids, label_stats, train_datasets, test_datasets


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


def save_phase2_figures(data_dir, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    with open(data_dir / "loss_history.json") as f:
        loss_history = json.load(f)
    with open(data_dir / "train_predictions.json") as f:
        train_preds = json.load(f)
    test_preds = []
    test_path = data_dir / "test_predictions.json"
    if test_path.exists():
        with open(test_path) as f:
            test_preds = json.load(f)

    # 1. 学習損失曲線
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history,
            color="steelblue", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (standardized)")
    ax.set_title("Phase 2: Training loss")
    plt.tight_layout()
    plt.savefig(figures_dir / "phase2_loss.png", dpi=150)
    plt.close()

    # 2. Gate 値分布（訓練 vs テスト）
    fig, ax = plt.subplots(figsize=(7, 4))
    tr_gates = [p["gate_value"] for p in train_preds]
    ax.hist(tr_gates, bins=20, alpha=0.7, color="steelblue", label=f"Train (n={len(tr_gates)})")
    if test_preds:
        te_gates = [p["gate_value"] for p in test_preds]
        ax.hist(te_gates, bins=20, alpha=0.7, color="tomato", label=f"Test (n={len(te_gates)})")
    ax.set_xlabel("Gate value g")
    ax.set_ylabel("Count")
    ax.set_title("Phase 2: Gate value distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "phase2_gate_dist.png", dpi=150)
    plt.close()

    # 3. 予測精度散布図 (acc)
    fig, ax = plt.subplots(figsize=(6, 6))
    tr_true_acc = [p["true_acc"] for p in train_preds]
    tr_pred_acc = [p["pred_acc"] for p in train_preds]
    ax.scatter(tr_true_acc, tr_pred_acc, s=30, alpha=0.6, color="steelblue", label="Train")
    if test_preds:
        te_true_acc = [p["true_acc"] for p in test_preds]
        te_pred_acc = [p["pred_acc"] for p in test_preds]
        ax.scatter(te_true_acc, te_pred_acc, s=40, alpha=0.8, color="tomato", marker="^", label="Test")
    lim = [min(tr_true_acc + [p["true_acc"] for p in test_preds]), max(tr_true_acc + [p["true_acc"] for p in test_preds])]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("True acc")
    ax.set_ylabel("Pred acc")
    ax.set_title("Phase 2: Accuracy prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "phase2_pred_acc.png", dpi=150)
    plt.close()

    # 4. 予測精度散布図 (RT)
    fig, ax = plt.subplots(figsize=(6, 6))
    tr_true_rt = [p["true_rt"] for p in train_preds]
    tr_pred_rt = [p["pred_rt"] for p in train_preds]
    ax.scatter(tr_true_rt, tr_pred_rt, s=30, alpha=0.6, color="steelblue", label="Train")
    if test_preds:
        te_true_rt = [p["true_rt"] for p in test_preds]
        te_pred_rt = [p["pred_rt"] for p in test_preds]
        ax.scatter(te_true_rt, te_pred_rt, s=40, alpha=0.8, color="tomato", marker="^", label="Test")
    all_rt = tr_true_rt + [p["true_rt"] for p in test_preds]
    lim_rt = [min(all_rt), max(all_rt)]
    ax.plot(lim_rt, lim_rt, "k--", linewidth=1)
    ax.set_xlabel("True RT (ms)")
    ax.set_ylabel("Pred RT (ms)")
    ax.set_title("Phase 2: RT prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "phase2_pred_rt.png", dpi=150)
    plt.close()

    print(f"  figures → {figures_dir}")


def run(subject_datasets, client_ids, label_stats, results_dir,
        split_threshold=SPLIT_THRESHOLD, merge_threshold=MERGE_THRESHOLD,
        lambda_con=LAMBDA_CON, temperature=TEMPERATURE,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR,
        n_init_bubbles=N_INIT_BUBBLES, verbose=True):
    """
    Phase 1: 対照学習付きバブル型クラスタ学習の1実験を実行して結果を保存する．

    subject_datasets / client_ids は訓練被験者のみを渡すこと（データリーク防止）．

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
    server.initialize_random(n_init_bubbles)

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
            if accs and rts:
                print(f"  Bubble {bid} ({len(members)} clients): acc={sum(accs)/len(accs):.3f}, rt={sum(rts)/len(rts):.0f}ms")

    return {
        "n_final_bubbles": server.n_bubbles,
        "final_loss": bubble_history[-1]["mean_loss"],
        "n_splits": n_splits,
        "n_merges": n_merges,
    }


def evaluate_phase1_on_test(test_datasets, label_stats, phase1_dir, device=None):
    """Phase 1 の encoder+head でテスト被験者を評価する（比較実験 条件C 用）.

    GateRAF を使わずに Phase 1 モデルをそのまま test に適用する．
    各被験者の全ウィンドウ予測を平均して subject-level 予測とする．

    Returns:
        list[dict]: per-subject 予測結果
    """
    if device is None:
        device = get_device()
    model = ClientModel().to(device)
    model.load_state_dict(
        torch.load(Path(phase1_dir) / "data" / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    test_data = load_all_tensors(test_datasets, device)
    predictions = []
    with torch.no_grad():
        for sid in test_data:
            h = model.encode(test_data[sid]["z"])
            pred = model.head(h).mean(0)
            label = test_data[sid]["labels"][0]
            predictions.append({
                "subject": sid,
                "pred_acc": float(pred[0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                "pred_rt":  float(pred[1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
                "true_acc": float(label[0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                "true_rt":  float(label[1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
            })
    return predictions


def run_phase2(subject_datasets, client_ids, label_stats, results_dir,
               phase1_dir=None, test_datasets=None, top_k=TOP_K, n_epochs=50, lr=1e-3, verbose=True):
    """
    Phase 2: GateRAF 学習 + テスト評価

    Phase 1 の encoder を凍結したままメモリバンクを構築し，
    Gate 層と予測ヘッドのみをタスク損失で最適化する．
    メモリバンクは訓練被験者のみで構築するため，
    subject_datasets / client_ids には訓練被験者のみを渡すこと．

    テスト被験者（test_datasets）は最近傍バブル割り当て後に GateRAF で予測する：
      1. frozen encoder で h_bar_test を計算
      2. 訓練バブルのプロトタイプに最近傍割り当て
      3. 割り当てバブルの訓練被験者から top-k 検索
      4. GateRAF で予測

    エンコーダ凍結の根拠：encoder を更新するとクエリと
    メモリバンクの表現空間がずれ（表現ドリフト），コサイン類似度検索が無効化される．
    表現力の向上が必要な場合は Momentum Encoder（m ≈ 0.99）への移行を検討する．

    Returns:
        dict: {final_loss, gate_mean, test_mse_acc, test_mse_rt}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if phase1_dir is None:
        raise ValueError("phase1_dir must be specified (path to Phase 1 results directory)")
    phase1_data_dir = Path(phase1_dir) / "data"

    phase1_model = ClientModel().to(device)
    phase1_model.load_state_dict(
        torch.load(phase1_data_dir / "model.pt", map_location=device, weights_only=True)
    )
    phase1_model.eval()
    for p in phase1_model.parameters():
        p.requires_grad_(False)

    with open(phase1_data_dir / "final_bubbles.json") as f:
        final_bubbles_raw = json.load(f)

    all_data = load_all_tensors(subject_datasets, device)
    memory_bank = compute_h_subject(phase1_model, all_data)

    sid_to_bubble = {}
    for bid, members in final_bubbles_raw.items():
        for m in members:
            if m["subject"] in {s for s in client_ids}:
                sid_to_bubble[m["subject"]] = bid

    bubble_to_sids = {}
    for sid, bid in sid_to_bubble.items():
        bubble_to_sids.setdefault(bid, []).append(sid)

    @torch.no_grad()
    def compute_h_ret(sid):
        bid = sid_to_bubble.get(sid)
        if bid is None:
            return torch.zeros(128, device=device)
        candidates = [s for s in bubble_to_sids.get(bid, []) if s != sid and s in memory_bank]
        # tex 条件: |B| > k+1 のときのみ参照を使う（= candidates が top_k+1 件以上）
        if len(candidates) <= top_k:
            return torch.zeros(128, device=device)
        h_q_norm = F.normalize(memory_bank[sid].unsqueeze(0), dim=1)
        cands_norm = F.normalize(torch.stack([memory_bank[s] for s in candidates]), dim=1)
        sims = (h_q_norm @ cands_norm.T).squeeze(0)
        top_idx = sims.topk(top_k).indices.tolist()
        return torch.stack([memory_bank[candidates[i]] for i in top_idx]).mean(0)

    h_q_all   = {sid: memory_bank[sid].detach() for sid in client_ids}
    h_ret_all = {sid: compute_h_ret(sid)        for sid in client_ids}
    labels_all = {sid: all_data[sid]["labels"][0] for sid in client_ids}

    gate_model = GateRAFModel().to(device)
    optimizer = torch.optim.Adam(gate_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(n_epochs):
        gate_model.train()
        h_q_stack   = torch.stack([h_q_all[s]   for s in client_ids])
        h_ret_stack = torch.stack([h_ret_all[s]  for s in client_ids])
        y_stack     = torch.stack([labels_all[s] for s in client_ids])

        pred, g = gate_model(h_q_stack, h_ret_stack)
        loss = criterion(pred, y_stack)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [Phase2] Epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  gate_mean={g.mean().item():.3f}")

    gate_model.eval()
    with torch.no_grad():
        h_q_stack   = torch.stack([h_q_all[s]   for s in client_ids])
        h_ret_stack = torch.stack([h_ret_all[s]  for s in client_ids])
        y_stack     = torch.stack([labels_all[s] for s in client_ids])
        pred_stack, g_stack = gate_model(h_q_stack, h_ret_stack)

    predictions = []
    for i, sid in enumerate(client_ids):
        predictions.append({
            "subject": sid,
            "bubble": sid_to_bubble.get(sid, -1),
            "gate_value": float(g_stack[i, 0]),
            "pred_acc": float(pred_stack[i, 0] * label_stats["acc_std"] + label_stats["acc_mean"]),
            "pred_rt":  float(pred_stack[i, 1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
            "true_acc": float(y_stack[i, 0] * label_stats["acc_std"] + label_stats["acc_mean"]),
            "true_rt":  float(y_stack[i, 1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
        })

    # テスト被験者の推論
    # 1. frozen encoder で h_bar_test を計算
    # 2. 訓練バブルのプロトタイプへ最近傍割り当て
    # 3. 割り当てバブルの訓練被験者から top-k 検索
    # 4. GateRAF で予測
    test_predictions = []
    if test_datasets:
        test_data = load_all_tensors(test_datasets, device)
        test_sids = list(test_data.keys())

        train_bubble_ids = sorted(bubble_to_sids.keys())
        proto_vecs = torch.stack([
            F.normalize(
                torch.stack([memory_bank[s] for s in bubble_to_sids[bid]]).mean(0, keepdim=True),
                dim=1,
            ).squeeze(0)
            for bid in train_bubble_ids
        ])  # (K, 128)

        @torch.no_grad()
        def infer_test(sid):
            h_bar = phase1_model.encode(test_data[sid]["z"]).mean(0)
            h_bar_norm = F.normalize(h_bar.unsqueeze(0), dim=1)
            nearest_idx = (h_bar_norm @ proto_vecs.T).squeeze(0).argmax().item()
            assigned_bid = train_bubble_ids[nearest_idx]
            candidates = [s for s in bubble_to_sids[assigned_bid] if s in memory_bank]
            if len(candidates) > top_k:  # |B| > k+1（tex 条件と一致）
                cands_norm = F.normalize(torch.stack([memory_bank[s] for s in candidates]), dim=1)
                sims = (h_bar_norm @ cands_norm.T).squeeze(0)
                top_idx = sims.topk(top_k).indices.tolist()
                h_ret = torch.stack([memory_bank[candidates[i]] for i in top_idx]).mean(0)
            else:
                h_ret = torch.zeros(128, device=device)
            pred, g = gate_model(h_bar.unsqueeze(0), h_ret.unsqueeze(0))
            label = test_data[sid]["labels"][0]
            return pred.squeeze(0), g.squeeze(0), label, assigned_bid

        gate_model.eval()
        for sid in test_sids:
            pred_t, g_t, label_t, bid = infer_test(sid)
            test_predictions.append({
                "subject": sid,
                "assigned_bubble": bid,
                "gate_value": float(g_t[0]),
                "pred_acc": float(pred_t[0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                "pred_rt":  float(pred_t[1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
                "true_acc": float(label_t[0] * label_stats["acc_std"] + label_stats["acc_mean"]),
                "true_rt":  float(label_t[1] * label_stats["rt_std"]  + label_stats["rt_mean"]),
            })

    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gate_model.state_dict(), data_dir / "gate_model.pt")
    with open(data_dir / "train_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    with open(data_dir / "test_predictions.json", "w") as f:
        json.dump(test_predictions, f, indent=2)
    with open(data_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)

    save_phase2_figures(data_dir, results_dir)

    gate_mean = sum(p["gate_value"] for p in predictions) / len(predictions)
    result = {"final_loss": loss_history[-1], "gate_mean": gate_mean}

    if verbose:
        tr_mse_acc = sum((p["pred_acc"] - p["true_acc"]) ** 2 for p in predictions) / len(predictions)
        tr_mse_rt  = sum((p["pred_rt"]  - p["true_rt"])  ** 2 for p in predictions) / len(predictions)
        print(f"\n=== Phase 2 complete ===")
        print(f"  Train  MSE  acc={tr_mse_acc:.4f}, RT={tr_mse_rt:.1f} ms²  gate_mean={gate_mean:.3f}")
        if test_predictions:
            te_mse_acc = sum((p["pred_acc"] - p["true_acc"]) ** 2 for p in test_predictions) / len(test_predictions)
            te_mse_rt  = sum((p["pred_rt"]  - p["true_rt"])  ** 2 for p in test_predictions) / len(test_predictions)
            print(f"  Test   MSE  acc={te_mse_acc:.4f}, RT={te_mse_rt:.1f} ms²")
            result["test_mse_acc"] = te_mse_acc
            result["test_mse_rt"]  = te_mse_rt

    return result


def main():
    device = get_device()
    print(f"Device: {device}")

    train_ids, test_ids, label_stats, train_datasets, test_datasets = prepare_data(device)
    train_client_ids = list(train_datasets.keys())
    print(f"Train: {len(train_client_ids)}, Test: {len(test_datasets)}")
    if not train_client_ids:
        print("No active clients. Run precompute.py first.")
        return

    print(f"Label stats (train only): acc={label_stats['acc_mean']:.3f}±{label_stats['acc_std']:.3f}, "
          f"rt={label_stats['rt_mean']:.1f}±{label_stats['rt_std']:.1f}ms")

    results_dir = Path(__file__).parent / "results"
    phase1_dir  = results_dir / "phase1"
    phase2_dir  = results_dir / "phase2"

    run(train_datasets, train_client_ids, label_stats,
        results_dir=phase1_dir,
        verbose=True)

    print("\n--- Starting Phase 2: GateRAF ---")
    run_phase2(train_datasets, train_client_ids, label_stats,
               results_dir=phase2_dir,
               phase1_dir=phase1_dir,
               test_datasets=test_datasets,
               top_k=TOP_K,
               verbose=True)


if __name__ == "__main__":
    main()
