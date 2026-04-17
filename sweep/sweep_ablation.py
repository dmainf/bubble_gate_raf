"""
比較実験（研究計画書 Section 4.1 対応）

GateRAF_only    : λ=0（対照学習なし）→ Phase 1 → Phase 2 GateRAF
                  → h 空間未整地で GateRAF を適用
Contrastive+RAF : λ>0（対照学習あり）→ Phase 1 → Phase 2 GateRAF   ← 本提案
                  → h 空間を整地した上で GateRAF を適用
Contrastive_only: λ>0（対照学習あり）→ Phase 1 のみ（GateRAF なし）
                  → 整地された h 空間で直接予測（参照なし）

検証仮説:
  Contrastive+RAF < GateRAF_only    : h 空間の整地が参照精度を高める
  Contrastive+RAF < Contrastive_only: バブル内参照が予測を向上させる
"""

import sys
import json
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import (prepare_data, run, run_phase2, evaluate_phase1_on_test,
                   get_device, TOP_K)

CONDITIONS = [
    {"label": "GateRAF_only",     "lambda_con": 0.0, "phase2": True},
    {"label": "Contrastive+RAF",  "lambda_con": 0.5, "phase2": True},
    {"label": "Contrastive_only", "lambda_con": 0.5, "phase2": False},
]

OUT_DIR = Path(__file__).parent / "ablation"


def subject_mse(preds, key_pred, key_true):
    """per-window 予測を被験者単位に集約して MSE を返す"""
    subs = defaultdict(list)
    for p in preds:
        subs[p["subject"]].append((p[key_pred] - p[key_true]) ** 2)
    return sum(sum(v) / len(v) for v in subs.values()) / len(subs)


def main():
    device = get_device()
    print(f"Device: {device}")

    train_ids, test_ids, label_stats, train_datasets, test_datasets = prepare_data(device)
    train_client_ids = list(train_datasets.keys())
    print(f"Train: {len(train_client_ids)}, Test: {len(test_datasets)}")
    if not train_client_ids:
        print("No active clients. Run precompute.py first.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    results = []

    for cond in CONDITIONS:
        print(f"\n=== 条件 {cond['label']} (λ={cond['lambda_con']}, phase2={cond['phase2']}) ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            phase1_dir = Path(tmpdir) / "phase1"

            run(train_datasets, train_client_ids, label_stats,
                results_dir=phase1_dir,
                lambda_con=cond["lambda_con"],
                verbose=False)

            if cond["phase2"]:
                phase2_dir = Path(tmpdir) / "phase2"
                run_phase2(train_datasets, train_client_ids, label_stats,
                           results_dir=phase2_dir,
                           phase1_dir=phase1_dir,
                           test_datasets=test_datasets,
                           top_k=TOP_K,
                           verbose=False)
                with open(phase2_dir / "data" / "test_predictions.json") as f:
                    test_preds = json.load(f)
                with open(phase2_dir / "data" / "train_predictions.json") as f:
                    train_preds = json.load(f)
            else:
                # 条件 C: Phase 1 encoder+head をテスト被験者に直接適用
                test_preds = evaluate_phase1_on_test(
                    test_datasets, label_stats, phase1_dir, device=device
                )
                with open(phase1_dir / "data" / "predictions.json") as f:
                    train_preds = json.load(f)

            result = {
                "label": cond["label"],
                "lambda_con": cond["lambda_con"],
                "phase2": cond["phase2"],
                "train_mse_acc": subject_mse(train_preds, "pred_acc", "true_acc"),
                "train_mse_rt":  subject_mse(train_preds, "pred_rt",  "true_rt"),
                "test_mse_acc":  subject_mse(test_preds,  "pred_acc", "true_acc"),
                "test_mse_rt":   subject_mse(test_preds,  "pred_rt",  "true_rt"),
            }
            results.append(result)
            print(f"  Train  acc={result['train_mse_acc']:.4f}  RT={result['train_mse_rt']:.1f} ms²")
            print(f"  Test   acc={result['test_mse_acc']:.4f}  RT={result['test_mse_rt']:.1f} ms²")

    with open(OUT_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _save_ablation_figures(results, OUT_DIR)
    print(f"\nDone. Results → {OUT_DIR}")

    # 仮説検証サマリ
    r = {rec["label"]: rec for rec in results}
    print("\n=== 仮説検証 ===")
    for metric, key in [("acc MSE", "test_mse_acc"), ("RT MSE", "test_mse_rt")]:
        raf  = r["GateRAF_only"][key]
        prop = r["Contrastive+RAF"][key]
        con  = r["Contrastive_only"][key]
        print(f"  {metric}: GateRAF_only={raf:.4f}  Contrastive+RAF={prop:.4f}  Contrastive_only={con:.4f}")
        print(f"    Contrastive+RAF < GateRAF_only    → 整地の効果: {'YES' if prop < raf  else 'NO'}")
        print(f"    Contrastive+RAF < Contrastive_only → 参照の効果: {'YES' if prop < con  else 'NO'}")


def _save_ablation_figures(results, out_dir):
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (key_tr, key_te, title, ylabel) in zip(axes, [
        ("train_mse_acc", "test_mse_acc", "Accuracy MSE by condition", "MSE (acc)"),
        ("train_mse_rt",  "test_mse_rt",  "RT MSE by condition",        "MSE (ms²)"),
    ]):
        ax.bar(x - w/2, [r[key_tr] for r in results], w,
               label="Train", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, [r[key_te] for r in results], w,
               label="Test",  color="tomato", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        best_idx = int(np.argmin([r[key_te] for r in results]))
        for i, tick in enumerate(ax.get_xticklabels()):
            if i == best_idx:
                tick.set_fontweight("bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "ablation_comparison.png", dpi=150)
    plt.close()
    print(f"  figure → {out_dir / 'ablation_comparison.png'}")


if __name__ == "__main__":
    main()
