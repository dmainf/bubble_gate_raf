"""
merge_thresholdを複数の値で試すスイープスクリプト．
結果は results/merge_{threshold}/ に保存される．
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from train import prepare_data, run, get_device

MERGE_THRESHOLDS = [0.1, 0.3, 0.5, 0.8, 1.2, 2.0]
SPLIT_THRESHOLD = 1.0


def main():
    device = get_device()
    print(f"Device: {device}")

    subject_ids, label_stats, subject_datasets, client_ids = prepare_data(device)
    print(f"Active clients: {len(client_ids)}")
    print(f"\nSweeping merge_threshold: {MERGE_THRESHOLDS}")
    print(f"{'threshold':>12} {'final bubbles':>14} {'splits':>8} {'merges':>8} {'final loss':>12}")
    print("-" * 60)

    for mt in MERGE_THRESHOLDS:
        print(f"\n--- merge_threshold={mt} ---")
        result = run(
            subject_datasets, client_ids, label_stats,
            results_dir=Path(__file__).parent / "sweep" / f"merge_{mt}",
            split_threshold=SPLIT_THRESHOLD,
            merge_threshold=mt,
            verbose=False,
        )
        print(f"{mt:>12.1f} {result['n_final_bubbles']:>14} {result['n_splits']:>8} "
              f"{result['n_merges']:>8} {result['final_loss']:>12.4f}")

    print("\nDone. Results saved to results/merge_*/")


if __name__ == "__main__":
    main()
