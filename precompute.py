"""
SwiFTの潜在ベクトルを事前計算して保存する．
1回だけ実行すれば以降の学習でSwiFTのforward passが不要になる．

保存先: data/latent_cache/{subject_id}_{direction}.pt
形状:   (n_windows, 288)
"""

import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))

from dataset import HCPWMDataset, compute_label_stats, DIRECTIONS, DATA_DIR
from swift import load_swift_encoder

CACHE_DIR = DATA_DIR / "latent_cache"


def get_subject_ids():
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "latent_cache"])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    subject_ids = get_subject_ids()
    label_stats = compute_label_stats(subject_ids)

    print("Loading SwiFT encoder...")
    encoder = load_swift_encoder().to(device)
    encoder.eval()

    for i, sub in enumerate(subject_ids):
        for direction in DIRECTIONS:
            cache_path = CACHE_DIR / f"{sub}_{direction}.pt"
            if cache_path.exists():
                print(f"[{i+1}/{len(subject_ids)}] {sub} {direction} skip (already exists)")
                continue

            nii_path = DATA_DIR / sub / "MNINonLinear" / "Results" / f"tfMRI_WM_{direction}" / f"tfMRI_WM_{direction}.nii.gz"
            if not nii_path.exists():
                continue

            ds = HCPWMDataset([sub], label_stats=label_stats)
            ds_dir = [s for s in ds.samples if s[1] == direction]
            if not ds_dir:
                continue

            ds.preload()
            latents = []
            with torch.no_grad():
                loader = DataLoader(ds, batch_size=8, shuffle=False)
                for batch in loader:
                    fmri = batch["fmri"].to(device)
                    z = encoder(fmri).mean(dim=[2, 3, 4, 5]).cpu()  # (B, 288)
                    latents.append(z)
            ds.unload()

            latents = torch.cat(latents, dim=0)  # (n_windows_total, 288)
            torch.save(latents, cache_path)
            print(f"[{i+1}/{len(subject_ids)}] {sub} {direction}: {latents.shape} → {cache_path.name}")


if __name__ == "__main__":
    main()
