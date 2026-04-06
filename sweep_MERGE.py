"""
merge_threshold を複数の値で試すスイープスクリプト．
結果は保存せず，ラウンドごとの mean loss を閾値ごとの折れ線でプロットする．
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

MERGE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
SPLIT_THRESHOLD = 1.0
ANNEALING_RATE = 2.0


def main():
    device = get_device()
    print(f"Device: {device}")

    subject_ids, label_stats, subject_datasets, client_ids = prepare_data(device)
    print(f"Active clients: {len(client_ids)}")

    colors = cm.tab10(np.linspace(0, 1, len(MERGE_THRESHOLDS)))
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    fig_w2,   ax_w2   = plt.subplots(figsize=(10, 5))
    fig_nb,   ax_nb   = plt.subplots(figsize=(10, 5))

    for mt, color in zip(MERGE_THRESHOLDS, colors):
        print(f"\n--- merge_threshold={mt} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                subject_datasets, client_ids, label_stats,
                results_dir=tmpdir,
                split_threshold=SPLIT_THRESHOLD,
                merge_threshold=mt,
                annealing_rate=ANNEALING_RATE,
                verbose=False,
            )
            with open(Path(tmpdir) / "data" / "bubble_history.json") as f:
                bubble_history = json.load(f)

        rounds = [h["round"] for h in bubble_history]
        mean_losses = [h["mean_loss"] for h in bubble_history]
        ax_loss.plot(rounds, mean_losses, color=color, linewidth=1.5,
                     marker="o", markersize=3, label=f"merge={mt}")

        mean_w2s = [
            np.mean(list(h["intra_w2"].values())) if h.get("intra_w2") else 0.0
            for h in bubble_history
        ]
        ax_w2.plot(rounds, mean_w2s, color=color, linewidth=1.5,
                   marker="o", markersize=3, label=f"merge={mt}")

        n_bubbles = [len(h["bubbles"]) for h in bubble_history]
        ax_nb.step(rounds, n_bubbles, color=color, where="post", linewidth=1.5,
                   marker="o", markersize=3, label=f"merge={mt}")

    out_dir = Path(__file__).parent / "sweep"
    out_dir.mkdir(exist_ok=True)

    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Mean loss")
    ax_loss.set_title("Mean loss per round (sweep over merge_threshold)")
    ax_loss.legend(loc="upper right", fontsize=9)
    fig_loss.tight_layout()
    fig_loss.savefig(out_dir / "MERGE_loss.png", dpi=150)
    plt.close(fig_loss)

    ax_w2.set_xlabel("Round")
    ax_w2.set_ylabel("Mean intra-bubble W2²")
    ax_w2.set_title("Mean intra-bubble W2² per round (sweep over merge_threshold)")
    ax_w2.legend(loc="upper right", fontsize=9)
    fig_w2.tight_layout()
    fig_w2.savefig(out_dir / "MERGE_w2.png", dpi=150)
    plt.close(fig_w2)

    ax_nb.set_xlabel("Round")
    ax_nb.set_ylabel("Number of bubbles")
    ax_nb.set_title("Number of bubbles per round (sweep over merge_threshold)")
    ax_nb.legend(loc="upper right", fontsize=9)
    fig_nb.tight_layout()
    fig_nb.savefig(out_dir / "MERGE_nb.png", dpi=150)
    plt.close(fig_nb)

    print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
