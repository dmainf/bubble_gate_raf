import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
DIRECTIONS = ["LR", "RL"]

# HCP volumetric data: 91×109×91 → crop to 96³
# SwiFT repo preprocessing.py より: [:, 14:-7, :, :] + padding
CROP = {
    "x": (slice(None), slice(None), slice(None)),
    "y": slice(14, -7),  # 109 → 88、その後paddingで96に
}


def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)
    return data


def crop_to_96(data):
    # data: (X, Y, Z, T) = (91, 109, 91, T)
    x, y, z, t = data.shape
    data = data[:, 14:102, :, :]  # y: 109 → 88、パディングで96へ
    data = torch.from_numpy(data)  # (91, 88, 91, T)
    # (X, Y, Z, T) → (1, X, Y, Z, T) → pad → (1, 96, 96, 96, T)
    data = data.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, 91, 88, 91)
    data = torch.nn.functional.pad(data.cpu(), (2, 3, 4, 4, 2, 3))  # (1, T, 96, 96, 96)
    data = data.squeeze(0).permute(1, 2, 3, 0)  # (96, 96, 96, T)
    return data


def z_norm(data):
    # data: (96, 96, 96, T)
    brain_mask = data != 0
    if brain_mask.sum() == 0:
        return data
    mean = data[brain_mask].mean()
    std = data[brain_mask].std()
    if std == 0:
        return data
    data = (data - mean) / std
    data[~brain_mask] = 0
    return data


def compute_label_stats(subject_ids):
    """訓練被験者のacc・rtの平均・標準偏差を計算する（テスト被験者を含めない）"""
    all_acc, all_rt = [], []
    for sub in subject_ids:
        path = DATA_DIR / sub / "MNINonLinear" / "Results" / "tfMRI_WM_LR" / "EVs" / "WM_Stats.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        acc = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "ACC")]["Value"].mean()
        rt = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "MEDIAN_RT")]["Value"].mean()
        all_acc.append(acc)
        all_rt.append(rt)
    acc_arr = np.array(all_acc, dtype=np.float32)
    rt_arr = np.array(all_rt, dtype=np.float32)
    return {
        "acc_mean": float(np.nanmean(acc_arr)), "acc_std": float(np.nanstd(acc_arr)) + 1e-8,
        "rt_mean": float(np.nanmean(rt_arr)),  "rt_std": float(np.nanstd(rt_arr)) + 1e-8,
    }


class HCPWMDataset(Dataset):
    def __init__(self, subject_ids, label_stats, window_size=20, stride=20):
        """
        label_stats: compute_label_stats(train_subject_ids) の結果を渡す
                     テスト被験者のデータをリークしないよう，訓練被験者のみで計算すること
        """
        self.window_size = window_size
        self.stride = stride
        self.label_stats = label_stats
        self.samples = []

        self._cache = {}  # {nii_path: tensor (96,96,96,T)} — preload/unloadで管理

        for sub in subject_ids:
            stats = self._load_stats(sub)
            if stats is None:
                continue
            for direction in DIRECTIONS:
                nii_path = DATA_DIR / sub / "MNINonLinear" / "Results" / f"tfMRI_WM_{direction}" / f"tfMRI_WM_{direction}.nii.gz"
                if not nii_path.exists():
                    continue
                n_timepoints = nib.load(str(nii_path)).shape[-1]
                for start in range(0, n_timepoints - window_size + 1, stride):
                    self.samples.append((sub, direction, start, stats, nii_path))

    def _load_stats(self, sub):
        path = DATA_DIR / sub / "MNINonLinear" / "Results" / "tfMRI_WM_LR" / "EVs" / "WM_Stats.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        acc_2bk = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "ACC")]["Value"].mean()
        rt_2bk = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "MEDIAN_RT")]["Value"].mean()
        if acc_2bk != acc_2bk or rt_2bk != rt_2bk:  # NaN check
            return None
        return {"acc_2bk": acc_2bk, "rt_2bk": rt_2bk}

    def preload(self):
        """local_train前に呼ぶ。全niiを1回だけ読み込んでキャッシュする"""
        paths = {nii_path for *_, nii_path in self.samples}
        for path in paths:
            if path not in self._cache:
                data = load_nii(str(path))
                data = crop_to_96(data)
                data = z_norm(data)
                self._cache[path] = data

    def unload(self):
        """local_train後に呼ぶ。メモリを解放する"""
        self._cache.clear()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sub, direction, start, stats, nii_path = self.samples[idx]

        if nii_path in self._cache:
            data = self._cache[nii_path]
        else:
            data = load_nii(str(nii_path))
            data = crop_to_96(data)
            data = z_norm(data)

        window = data[..., start:start + self.window_size]  # (96, 96, 96, 20)
        window = window.unsqueeze(0)             # (1, 96, 96, 96, 20)

        return {
            "fmri": window,
            "subject": sub,
            "direction": direction,
            "start_t": start,
            "acc_2bk": torch.tensor((stats["acc_2bk"] - self.label_stats["acc_mean"]) / self.label_stats["acc_std"], dtype=torch.float32),
            "rt_2bk": torch.tensor((stats["rt_2bk"] - self.label_stats["rt_mean"]) / self.label_stats["rt_std"], dtype=torch.float32),
        }


CACHE_DIR = DATA_DIR / "latent_cache"


class HCPWMLatentDataset(Dataset):
    """
    事前計算済みのSwiFT潜在ベクトルを使うDataset．
    precompute.pyを実行後に使用する．
    HCPWMDatasetと同じラベルを返すが，fMRIの代わりに(288,)の潜在ベクトルを返す．
    """

    def __init__(self, subject_ids, label_stats, window_size=20, stride=20):
        self.label_stats = label_stats
        self.samples = []  # (subject_id, direction, window_idx, stats)

        for sub in subject_ids:
            stats = self._load_stats(sub)
            if stats is None:
                continue
            for direction in DIRECTIONS:
                cache_path = CACHE_DIR / f"{sub}_{direction}.pt"
                if not cache_path.exists():
                    continue
                latents = torch.load(cache_path, weights_only=True)  # (n_windows, 288)
                for idx in range(latents.shape[0]):
                    self.samples.append((sub, direction, idx, stats, latents[idx]))

    def _load_stats(self, sub):
        path = DATA_DIR / sub / "MNINonLinear" / "Results" / "tfMRI_WM_LR" / "EVs" / "WM_Stats.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        acc_2bk = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "ACC")]["Value"].mean()
        rt_2bk = df[(df["ConditionName"].str.startswith("2BK")) & (df["Measure"] == "MEDIAN_RT")]["Value"].mean()
        if acc_2bk != acc_2bk or rt_2bk != rt_2bk:
            return None
        return {"acc_2bk": acc_2bk, "rt_2bk": rt_2bk}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sub, direction, window_idx, stats, z = self.samples[idx]
        return {
            "z": z,                                  # (288,) 潜在ベクトル
            "subject": sub,
            "direction": direction,
            "window_idx": window_idx,
            "acc_2bk": torch.tensor((stats["acc_2bk"] - self.label_stats["acc_mean"]) / self.label_stats["acc_std"], dtype=torch.float32),
            "rt_2bk": torch.tensor((stats["rt_2bk"] - self.label_stats["rt_mean"]) / self.label_stats["rt_std"], dtype=torch.float32),
        }
