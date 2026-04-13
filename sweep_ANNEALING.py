"""
lambda_con（対照学習損失の重み）を複数の値で試すスイープスクリプト．
ラウンドごとの mean loss と mean intra-bubble cosine distance を折れ線でプロットする．
"""

import sys
import json
import tempfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from train import prepare_data, run, get_device

LAMBDA_CONS = [0.0, 0.1, 0.3, 0.5, 1.0]
SPLIT_THRESHOLD = 0.3
MERGE_THRESHOLD = 0.05


def main():
    device = get_device()
    print(f"Device: {device}")

    subject_ids, label_stats, subject_datasets, client_ids = prepare_data(device)
    print(f"Active clients: {len(client_ids)}")

    colors = cm.tab10(np.linspace(0, 1, len(LAMBDA_CONS)))
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    fig_w2,   ax_w2   = plt.subplots(figsize=(10, 5))
    fig_nb,   ax_nb   = plt.subplots(figsize=(10, 5))

    for lc, color in zip(LAMBDA_CONS, colors):
        print(f"\n--- lambda_con={lc} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                subject_datasets, client_ids, label_stats,
                results_dir=tmpdir,
                split_threshold=SPLIT_THRESHOLD,
                merge_threshold=MERGE_THRESHOLD,
                lambda_con=lc,
                verbose=False,
            )
            with open(Path(tmpdir) / "data" / "bubble_history.json") as f:
                bubble_history = json.load(f)

        rounds = [h["round"] for h in bubble_history]
        mean_losses = [h["mean_loss"] for h in bubble_history]
        ax_loss.plot(rounds, mean_losses, color=color, linewidth=1.5,
                     marker="o", markersize=3, label=f"lambda_con={lc}")

        mean_dists = [
            np.mean(list(h["intra_dist"].values())) if h.get("intra_dist") else 0.0
            for h in bubble_history
        ]
        ax_w2.plot(rounds, mean_dists, color=color, linewidth=1.5,
                   marker="o", markersize=3, label=f"lambda_con={lc}")

        n_bubbles = [len(h["bubbles"]) for h in bubble_history]
        ax_nb.step(rounds, n_bubbles, color=color, where="post", linewidth=1.5,
                   marker="o", markersize=3, label=f"lambda_con={lc}")

    out_dir = Path(__file__).parent / "sweep"
    out_dir.mkdir(exist_ok=True)

    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Mean loss")
    ax_loss.set_title("Mean loss per round (sweep over lambda_con)")
    ax_loss.legend(loc="upper right", fontsize=9)
    fig_loss.tight_layout()
    fig_loss.savefig(out_dir / "LAMBDA_loss.png", dpi=150)
    plt.close(fig_loss)

    ax_w2.set_xlabel("Round")
    ax_w2.set_ylabel("Mean intra-bubble cosine distance")
    ax_w2.set_title("Mean intra-bubble cosine distance per round (sweep over lambda_con)")
    ax_w2.legend(loc="upper right", fontsize=9)
    fig_w2.tight_layout()
    fig_w2.savefig(out_dir / "LAMBDA_dist.png", dpi=150)
    plt.close(fig_w2)

    ax_nb.set_xlabel("Round")
    ax_nb.set_ylabel("Number of bubbles")
    ax_nb.set_title("Number of bubbles per round (sweep over lambda_con)")
    ax_nb.legend(loc="upper right", fontsize=9)
    fig_nb.tight_layout()
    fig_nb.savefig(out_dir / "LAMBDA_nb.png", dpi=150)
    plt.close(fig_nb)

    print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
