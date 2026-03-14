import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.stats import t, ttest_rel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_LABELS = [
    "MN 230",
    "MN 236",
    "MN 242",
    "MN 408",
    "M 4329",
    "MN 384",
    "CI (gfp)",
    "S SAJV01",
    "A 25923",
    "NS",
    "VNS",
    "H2O",
]

CONTROL_ONLY_LABELS = ["VNS", "H2O"]
MAIN_TASK_LABELS = [label for label in RAW_LABELS if label not in CONTROL_ONLY_LABELS]
# Backward-compatible alias used by the rest of the experiment code.
LABELS = MAIN_TASK_LABELS

RAW_TO_MAIN_CLASS = {
    raw_idx: main_idx for main_idx, raw_idx in enumerate(i for i, name in enumerate(RAW_LABELS) if name in MAIN_TASK_LABELS)
}
MAIN_TASK_RAW_INDICES = sorted(RAW_TO_MAIN_CLASS.keys())

SPECTRAL_WINDOW_CM1_DEFAULT = (400.0, 1200.0)
SPECTRAL_WINDOW_CM1 = SPECTRAL_WINDOW_CM1_DEFAULT
PRIMARY_SAMPLES_PER_RAW_LABEL = 10
PRIMARY_DATASET_INFO: dict = {}

LOD_LEVELS = {
    0: ("high", 1.00, 0.008),
    1: ("mid", 0.60, 0.015),
    2: ("low", 0.30, 0.025),
}

SEEDS = [42 + i * 7 for i in range(30)]
SENSITIVITY_SEEDS = SEEDS[:10]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 64

TRAIN_REPEATS_PER_LOD = 3
EVAL_REPEATS_PER_LOD = 1
EXTRA_TRAIN_REPEATS_PER_LOD = 2

DUAL_LAMBDA_LOD = 0.5
DUAL_CLASS_WEIGHT_BY_LOD = (1.0, 1.2, 1.8)


@dataclass
class SplitData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_cls_train: np.ndarray
    y_cls_val: np.ndarray
    y_cls_test: np.ndarray
    y_lod_train: np.ndarray
    y_lod_val: np.ndarray
    y_lod_test: np.ndarray


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(drop)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + shortcut
        return self.act(x)


class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int = 256, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.block1 = ResidualMLPBlock(width)
        self.block2 = ResidualMLPBlock(width)
        self.head = nn.Sequential(
            nn.Linear(width, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class SingleHeadModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.encoder = SharedEncoder(input_dim=input_dim)
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        return self.cls_head(self.encoder(x))


class DualHeadModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_lod: int):
        super().__init__()
        self.encoder = SharedEncoder(input_dim=input_dim)
        self.cls_head = nn.Linear(128, num_classes)
        self.lod_head = nn.Linear(128, num_lod)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.cls_head(z), self.lod_head(z)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_primary_xlsx_path(base_dir: Path) -> Path:
    matches = list(base_dir.rglob("108.xlsx"))
    for p in matches:
        s = str(p).lower()
        if "bacteria2" in s and "van1-10" not in s:
            return p
    if not matches:
        raise FileNotFoundError("Cannot find 108.xlsx under workspace")
    return matches[0]


def resolve_aux_mat_path(base_dir: Path) -> Path:
    p = base_dir / "van1-10 in Emory" / "108" / "sheet-all.mat"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find auxiliary MAT file: {p}")
    return p


def flatten_numeric_tokens(obj) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.dtype != object:
        return arr.ravel()

    out = []
    for item in arr.ravel():
        if isinstance(item, np.ndarray):
            if item.size == 0:
                continue
            if item.dtype == object:
                out.extend(flatten_numeric_tokens(item).tolist())
            else:
                out.extend(np.asarray(item).ravel().tolist())
        else:
            out.append(item)
    return np.asarray(out)


def set_spectral_window(window_cm1: tuple[float, float] | None) -> None:
    """Set spectral feature window by Raman shift (cm^-1). Use None for full spectrum."""
    global SPECTRAL_WINDOW_CM1
    if window_cm1 is None:
        SPECTRAL_WINDOW_CM1 = None
        return
    lo, hi = float(window_cm1[0]), float(window_cm1[1])
    if hi <= lo:
        raise ValueError(f"Invalid spectral window: {window_cm1}")
    SPECTRAL_WINDOW_CM1 = (lo, hi)


def _spectral_window_mask(wavelengths: np.ndarray) -> np.ndarray:
    if SPECTRAL_WINDOW_CM1 is None:
        return np.ones_like(wavelengths, dtype=bool)
    lo, hi = SPECTRAL_WINDOW_CM1
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() < 10:
        raise RuntimeError(
            f"Too few spectral points remain after applying {lo}-{hi} cm^-1 window: {int(mask.sum())}"
        )
    return mask


def _filter_main_task_classes(x: np.ndarray, y_raw: np.ndarray):
    keep_mask = np.isin(y_raw, MAIN_TASK_RAW_INDICES)
    x_keep = x[keep_mask]
    y_keep_raw = y_raw[keep_mask]
    y_main = np.asarray([RAW_TO_MAIN_CLASS[int(v)] for v in y_keep_raw], dtype=np.int64)
    return x_keep.astype(np.float32), y_main, keep_mask


def load_primary_data(xlsx_path: Path):
    df = pd.read_excel(xlsx_path, sheet_name=1)
    wavelengths = df.iloc[:, 0].to_numpy(dtype=np.float32)
    spectra = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    x_full = spectra.T
    y_raw = np.repeat(np.arange(len(RAW_LABELS)), PRIMARY_SAMPLES_PER_RAW_LABEL)
    if x_full.shape[0] != y_raw.shape[0]:
        raise RuntimeError(f"Sample count mismatch: X={x_full.shape[0]}, labels={y_raw.shape[0]}")

    feat_mask = _spectral_window_mask(wavelengths)
    x_window = x_full[:, feat_mask]
    wavelengths_window = wavelengths[feat_mask]

    x_main, y_main, keep_mask = _filter_main_task_classes(x_window, y_raw)
    raw_counts = {RAW_LABELS[int(k)]: int(v) for k, v in zip(*np.unique(y_raw, return_counts=True))}
    feat_indices = np.where(feat_mask)[0]
    kept_main_counts = {LABELS[int(k)]: int(v) for k, v in zip(*np.unique(y_main, return_counts=True))}
    control_labels_present = [
        RAW_LABELS[int(y_raw[i])] for i in np.where(~keep_mask)[0]
    ]

    PRIMARY_DATASET_INFO.clear()
    PRIMARY_DATASET_INFO.update(
        {
            "raw_label_order": list(RAW_LABELS),
            "main_task_label_order": list(MAIN_TASK_LABELS),
            "control_only_labels_excluded_from_main_task": list(CONTROL_ONLY_LABELS),
            "n_raw_samples_before_main_task_filter": int(x_full.shape[0]),
            "n_main_task_samples_after_filter": int(x_main.shape[0]),
            "raw_class_counts_before_filter": raw_counts,
            "main_task_class_counts": kept_main_counts,
            "excluded_control_sample_count": int((~keep_mask).sum()),
            "excluded_control_sample_labels": sorted(set(control_labels_present)),
            "spectral_window_cm1": (
                [float(SPECTRAL_WINDOW_CM1[0]), float(SPECTRAL_WINDOW_CM1[1])]
                if SPECTRAL_WINDOW_CM1 is not None
                else None
            ),
            "spectral_window_mode": "cropped" if SPECTRAL_WINDOW_CM1 is not None else "full",
            "n_features_full": int(x_full.shape[1]),
            "n_features_after_window": int(x_main.shape[1]),
            "wavelength_min_full": float(np.min(wavelengths)),
            "wavelength_max_full": float(np.max(wavelengths)),
            "wavelength_min_window": float(np.min(wavelengths_window)),
            "wavelength_max_window": float(np.max(wavelengths_window)),
            "window_feature_index_range": [int(feat_indices.min()), int(feat_indices.max())],
            "window_feature_indices": feat_indices.astype(int).tolist(),
        }
    )
    return x_main.astype(np.float32), y_main.astype(np.int64), wavelengths_window.astype(np.float32)


def load_auxiliary_extra_samples(mat_path: Path, x_primary: np.ndarray, n_features_full: int | None = None):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    ans = mat.get("ans")
    if ans is None or not hasattr(ans, "data") or not hasattr(ans, "class"):
        raise RuntimeError(f"Unsupported MAT structure in {mat_path}")

    x_mat = np.asarray(ans.data, dtype=np.float32)
    if x_mat.ndim != 2:
        raise RuntimeError(f"Unexpected MAT data shape: {x_mat.shape}")

    if n_features_full is None:
        n_features_full = int(PRIMARY_DATASET_INFO.get("n_features_full", x_primary.shape[1]))

    if x_mat.shape[1] == n_features_full + 1:
        x_mat = x_mat[:, 1:]
    if x_mat.shape[1] != n_features_full:
        raise RuntimeError(f"Unexpected MAT feature count before windowing: {x_mat.shape[1]} vs expected {n_features_full}")

    # Apply the same spectral window used for the primary dataset (doc/MS logic: focus on 400-1200 cm^-1).
    window_feature_indices = PRIMARY_DATASET_INFO.get("window_feature_indices")
    if window_feature_indices:
        x_mat = x_mat[:, np.asarray(window_feature_indices, dtype=np.int64)]
    if x_mat.shape[1] != x_primary.shape[1]:
        raise RuntimeError(
            f"Feature mismatch between MAT and primary XLSX: {x_mat.shape[1]} vs {x_primary.shape[1]}"
        )

    class_tokens = flatten_numeric_tokens(getattr(ans, "class"))
    class_ids = []
    for v in class_tokens:
        try:
            class_ids.append(int(v))
        except Exception:
            continue
    y_mat = np.asarray(class_ids, dtype=np.int64)
    if y_mat.shape[0] != x_mat.shape[0]:
        raise RuntimeError(f"Class length mismatch in MAT: y={y_mat.shape[0]} x={x_mat.shape[0]}")
    if y_mat.min() >= 1:
        y_mat = y_mat - 1

    keep_mask = np.isin(y_mat, MAIN_TASK_RAW_INDICES)
    x_mat = x_mat[keep_mask]
    y_mat = y_mat[keep_mask]
    y_mat = np.asarray([RAW_TO_MAIN_CLASS[int(v)] for v in y_mat], dtype=np.int64)

    primary_keys = {row.tobytes() for row in np.asarray(x_primary, dtype=np.float32)}
    extra_idx = [i for i in range(x_mat.shape[0]) if x_mat[i].tobytes() not in primary_keys]
    x_extra = x_mat[extra_idx].astype(np.float32)
    y_extra = y_mat[extra_idx].astype(np.int64)
    return x_extra, y_extra


def split_raw_indices(y_cls: np.ndarray, seed: int):
    idx = np.arange(y_cls.shape[0])
    idx_train, idx_tmp = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=y_cls,
    )
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=0.50,
        random_state=seed,
        stratify=y_cls[idx_tmp],
    )
    return idx_train, idx_val, idx_test


def synthesize_lod_from_raw(x_raw: np.ndarray, y_cls: np.ndarray, seed: int, repeats_per_lod: int):
    rng = np.random.default_rng(seed)
    xs = []
    ys_cls = []
    ys_lod = []
    for i in range(x_raw.shape[0]):
        x0 = x_raw[i]
        sigma = max(float(np.std(x0)), 1e-6)
        n = x0.shape[0]
        drift_axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        for lod_id, (_, scale, noise_ratio) in LOD_LEVELS.items():
            for _ in range(repeats_per_lod):
                drift = drift_axis * rng.normal(0.0, 0.004 * sigma)
                noise = rng.normal(0.0, noise_ratio * sigma, size=n).astype(np.float32)
                x_new = x0 * scale + drift + noise
                xs.append(x_new.astype(np.float32))
                ys_cls.append(int(y_cls[i]))
                ys_lod.append(int(lod_id))
    return np.stack(xs), np.asarray(ys_cls, dtype=np.int64), np.asarray(ys_lod, dtype=np.int64)


def build_split(
    x_primary: np.ndarray,
    y_primary: np.ndarray,
    seed: int,
    x_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
    include_extra_train: bool = False,
) -> SplitData:
    idx_train, idx_val, idx_test = split_raw_indices(y_primary, seed=seed)

    x_train, y_train_cls, y_train_lod = synthesize_lod_from_raw(
        x_primary[idx_train], y_primary[idx_train], seed=seed + 11, repeats_per_lod=TRAIN_REPEATS_PER_LOD
    )
    x_val, y_val_cls, y_val_lod = synthesize_lod_from_raw(
        x_primary[idx_val], y_primary[idx_val], seed=seed + 13, repeats_per_lod=EVAL_REPEATS_PER_LOD
    )
    x_test, y_test_cls, y_test_lod = synthesize_lod_from_raw(
        x_primary[idx_test], y_primary[idx_test], seed=seed + 17, repeats_per_lod=EVAL_REPEATS_PER_LOD
    )

    if include_extra_train and x_extra is not None and y_extra is not None and x_extra.shape[0] > 0:
        x_aux, y_aux_cls, y_aux_lod = synthesize_lod_from_raw(
            x_extra, y_extra, seed=seed + 19, repeats_per_lod=EXTRA_TRAIN_REPEATS_PER_LOD
        )
        x_train = np.vstack([x_train, x_aux])
        y_train_cls = np.concatenate([y_train_cls, y_aux_cls])
        y_train_lod = np.concatenate([y_train_lod, y_aux_lod])

    scaler = StandardScaler()
    scaler.fit(x_train)

    return SplitData(
        x_train=scaler.transform(x_train).astype(np.float32),
        x_val=scaler.transform(x_val).astype(np.float32),
        x_test=scaler.transform(x_test).astype(np.float32),
        y_cls_train=y_train_cls,
        y_cls_val=y_val_cls,
        y_cls_test=y_test_cls,
        y_lod_train=y_train_lod,
        y_lod_val=y_val_lod,
        y_lod_test=y_test_lod,
    )


def train_single(split: SplitData, seed: int):
    set_seed(seed)
    model = SingleHeadModel(input_dim=split.x_train.shape[1], num_classes=len(LABELS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    x_tr = torch.from_numpy(split.x_train)
    y_tr = torch.from_numpy(split.y_cls_train).long()
    x_va = torch.from_numpy(split.x_val).to(DEVICE)

    best = None
    best_score = -1.0
    wait = 0

    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(x_tr.shape[0])
        for s in range(0, x_tr.shape[0], BATCH_SIZE):
            b = order[s : s + BATCH_SIZE]
            xb = x_tr[b].to(DEVICE)
            yb = y_tr[b].to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p = model(x_va).argmax(dim=1).cpu().numpy()
        f1 = f1_score(split.y_cls_val, p, average="macro")
        if f1 > best_score:
            best_score = f1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best is not None:
        model.load_state_dict(best)
    return model


def train_dual(split: SplitData, seed: int):
    set_seed(seed)
    model = DualHeadModel(input_dim=split.x_train.shape[1], num_classes=len(LABELS), num_lod=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_cls = nn.CrossEntropyLoss(reduction="none")
    ce_lod = nn.CrossEntropyLoss()
    lod_weight = torch.tensor(DUAL_CLASS_WEIGHT_BY_LOD, dtype=torch.float32, device=DEVICE)

    x_tr = torch.from_numpy(split.x_train)
    y_tr_cls = torch.from_numpy(split.y_cls_train).long()
    y_tr_lod = torch.from_numpy(split.y_lod_train).long()

    x_va = torch.from_numpy(split.x_val).to(DEVICE)

    best = None
    best_score = -1.0
    wait = 0

    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(x_tr.shape[0])
        for s in range(0, x_tr.shape[0], BATCH_SIZE):
            b = order[s : s + BATCH_SIZE]
            xb = x_tr[b].to(DEVICE)
            yb_cls = y_tr_cls[b].to(DEVICE)
            yb_lod = y_tr_lod[b].to(DEVICE)
            opt.zero_grad()
            cls_logits, lod_logits = model(xb)
            w = lod_weight[yb_lod]
            cls_loss = (ce_cls(cls_logits, yb_cls) * w).mean()
            loss = cls_loss + DUAL_LAMBDA_LOD * ce_lod(lod_logits, yb_lod)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p_cls = model(x_va)[0].argmax(dim=1).cpu().numpy()
        f1 = f1_score(split.y_cls_val, p_cls, average="macro")
        if f1 > best_score:
            best_score = f1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best is not None:
        model.load_state_dict(best)
    return model


def evaluate_single(model: SingleHeadModel, split: SplitData):
    model.eval()
    x_te = torch.from_numpy(split.x_test).to(DEVICE)
    with torch.no_grad():
        pred = model(x_te).argmax(dim=1).cpu().numpy()

    low_mask = split.y_lod_test == 2
    return {
        "accuracy": float(accuracy_score(split.y_cls_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(split.y_cls_test, pred)),
        "macro_f1": float(f1_score(split.y_cls_test, pred, average="macro")),
        "low_accuracy": float(accuracy_score(split.y_cls_test[low_mask], pred[low_mask])),
        "low_macro_f1": float(f1_score(split.y_cls_test[low_mask], pred[low_mask], average="macro")),
    }


def evaluate_dual(model: DualHeadModel, split: SplitData):
    model.eval()
    x_te = torch.from_numpy(split.x_test).to(DEVICE)
    with torch.no_grad():
        cls_logits, lod_logits = model(x_te)
        pred_cls = cls_logits.argmax(dim=1).cpu().numpy()
        pred_lod = lod_logits.argmax(dim=1).cpu().numpy()

    low_mask = split.y_lod_test == 2
    return {
        "accuracy": float(accuracy_score(split.y_cls_test, pred_cls)),
        "balanced_accuracy": float(balanced_accuracy_score(split.y_cls_test, pred_cls)),
        "macro_f1": float(f1_score(split.y_cls_test, pred_cls, average="macro")),
        "low_accuracy": float(accuracy_score(split.y_cls_test[low_mask], pred_cls[low_mask])),
        "low_macro_f1": float(f1_score(split.y_cls_test[low_mask], pred_cls[low_mask], average="macro")),
        "lod_macro_f1": float(f1_score(split.y_lod_test, pred_lod, average="macro")),
    }


def mean_ci(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size <= 1:
        return [mean, mean]
    sd = float(arr.std(ddof=1))
    h = float(t.ppf(0.975, df=arr.size - 1) * sd / math.sqrt(arr.size))
    return [mean - h, mean + h]


def paired_stats(a, b):
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    diff = db - da
    stat = ttest_rel(db, da)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    ci = mean_ci(diff)
    return {
        "mean_diff": mean_diff,
        "p_value": float(stat.pvalue),
        "t_stat": float(stat.statistic),
        "cohens_d_paired": d,
        "ci95_diff": [float(ci[0]), float(ci[1])],
        "wins": int((diff > 0).sum()),
        "n": int(diff.size),
    }


def run_experiment(
    x_primary: np.ndarray,
    y_primary: np.ndarray,
    x_extra: np.ndarray | None,
    y_extra: np.ndarray | None,
    include_extra_train: bool,
    seed_list: list[int],
):
    rows = []
    run_meta = []
    t_begin = time.time()
    for seed in seed_list:
        t_run = time.time()
        split = build_split(
            x_primary=x_primary,
            y_primary=y_primary,
            seed=seed,
            x_extra=x_extra,
            y_extra=y_extra,
            include_extra_train=include_extra_train,
        )

        baseline = train_single(split, seed=seed)
        m_base = evaluate_single(baseline, split)

        dual = train_dual(split, seed=seed)
        m_dual = evaluate_dual(dual, split)

        rows.append({"run_seed": seed, "model": "baseline_single_head", **m_base})
        rows.append({"run_seed": seed, "model": "conceptA_dual_head", **m_dual})
        run_meta.append(
            {
                "run_seed": seed,
                "n_train": int(split.x_train.shape[0]),
                "n_val": int(split.x_val.shape[0]),
                "n_test": int(split.x_test.shape[0]),
                "run_seconds": float(time.time() - t_run),
            }
        )

    df = pd.DataFrame(rows)
    base_df = df[df["model"] == "baseline_single_head"].sort_values("run_seed")
    dual_df = df[df["model"] == "conceptA_dual_head"].sort_values("run_seed")

    aggregate = {
        "baseline_macro_f1_mean": float(base_df["macro_f1"].mean()),
        "dual_macro_f1_mean": float(dual_df["macro_f1"].mean()),
        "baseline_low_macro_f1_mean": float(base_df["low_macro_f1"].mean()),
        "dual_low_macro_f1_mean": float(dual_df["low_macro_f1"].mean()),
        "baseline_accuracy_mean": float(base_df["accuracy"].mean()),
        "dual_accuracy_mean": float(dual_df["accuracy"].mean()),
        "baseline_balanced_accuracy_mean": float(base_df["balanced_accuracy"].mean()),
        "dual_balanced_accuracy_mean": float(dual_df["balanced_accuracy"].mean()),
        "dual_lod_macro_f1_mean": float(dual_df["lod_macro_f1"].mean()),
        "macro_f1_paired": paired_stats(base_df["macro_f1"].tolist(), dual_df["macro_f1"].tolist()),
        "low_macro_f1_paired": paired_stats(base_df["low_macro_f1"].tolist(), dual_df["low_macro_f1"].tolist()),
        "accuracy_paired": paired_stats(base_df["accuracy"].tolist(), dual_df["accuracy"].tolist()),
        "balanced_accuracy_paired": paired_stats(
            base_df["balanced_accuracy"].tolist(), dual_df["balanced_accuracy"].tolist()
        ),
        "baseline_macro_f1_ci95": mean_ci(base_df["macro_f1"].tolist()),
        "dual_macro_f1_ci95": mean_ci(dual_df["macro_f1"].tolist()),
        "baseline_low_macro_f1_ci95": mean_ci(base_df["low_macro_f1"].tolist()),
        "dual_low_macro_f1_ci95": mean_ci(dual_df["low_macro_f1"].tolist()),
    }

    return {
        "rows_df": df,
        "summary": {
            "setup": {
                "seed_list": seed_list,
                "split_strategy": "split_raw_then_augment",
                "include_extra_train": include_extra_train,
                "train_repeats_per_lod": TRAIN_REPEATS_PER_LOD,
                "eval_repeats_per_lod": EVAL_REPEATS_PER_LOD,
                "extra_train_repeats_per_lod": EXTRA_TRAIN_REPEATS_PER_LOD,
                "dual_lambda_lod": DUAL_LAMBDA_LOD,
                "dual_class_weight_by_lod": list(DUAL_CLASS_WEIGHT_BY_LOD),
            },
            "timing": {
                "total_seconds": float(time.time() - t_begin),
                "per_run": run_meta,
            },
            "aggregate": aggregate,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="AI-SCI Stage 3 Concept A experiment (10-strain fixed-concentration classification + weak-signal proxy robustness)."
    )
    p.add_argument("--out-subdir", default="", help="Optional subdirectory under ai_sci/stage3/outputs/")
    p.add_argument(
        "--spectrum-mode",
        choices=["full", "cropped"],
        default="cropped",
        help="Use full spectrum or 400-1200 cm^-1 cropped window.",
    )
    p.add_argument("--dual-lambda-lod", type=float, default=DUAL_LAMBDA_LOD, help="Dual-head auxiliary loss weight.")
    p.add_argument(
        "--dual-class-weight-by-lod",
        type=float,
        nargs=3,
        default=list(DUAL_CLASS_WEIGHT_BY_LOD),
        metavar=("W_HIGH", "W_MID", "W_LOW"),
        help="Per-proxy-LOD class loss weights.",
    )
    p.add_argument("--seed-count", type=int, default=len(SEEDS), help="Number of seeds for primary run.")
    p.add_argument(
        "--sensitivity-seed-count",
        type=int,
        default=len(SENSITIVITY_SEEDS),
        help="Number of seeds for sensitivity run with auxiliary train samples.",
    )
    p.add_argument("--skip-sensitivity-run", action="store_true", help="Skip auxiliary extra-train sensitivity run.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "ai_sci" / "stage3" / "outputs"
    if args.out_subdir:
        out_dir = out_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    global DUAL_LAMBDA_LOD, DUAL_CLASS_WEIGHT_BY_LOD
    DUAL_LAMBDA_LOD = float(args.dual_lambda_lod)
    DUAL_CLASS_WEIGHT_BY_LOD = tuple(float(v) for v in args.dual_class_weight_by_lod)
    if args.spectrum_mode == "full":
        set_spectral_window(None)
    else:
        set_spectral_window(SPECTRAL_WINDOW_CM1_DEFAULT)

    primary_seed_list = SEEDS[: max(1, min(len(SEEDS), int(args.seed_count)))]
    sensitivity_seed_list = SENSITIVITY_SEEDS[: max(1, min(len(SENSITIVITY_SEEDS), int(args.sensitivity_seed_count)))]

    xlsx_path = resolve_primary_xlsx_path(base_dir)
    aux_mat_path = resolve_aux_mat_path(base_dir)
    x_primary, y_primary, wavelengths = load_primary_data(xlsx_path)
    x_extra, y_extra = load_auxiliary_extra_samples(
        aux_mat_path,
        x_primary=x_primary,
        n_features_full=int(PRIMARY_DATASET_INFO.get("n_features_full", x_primary.shape[1])),
    )

    main_run = run_experiment(
        x_primary=x_primary,
        y_primary=y_primary,
        x_extra=x_extra,
        y_extra=y_extra,
        include_extra_train=False,
        seed_list=primary_seed_list,
    )
    aux_run = None
    if not args.skip_sensitivity_run:
        aux_run = run_experiment(
            x_primary=x_primary,
            y_primary=y_primary,
            x_extra=x_extra,
            y_extra=y_extra,
            include_extra_train=True,
            seed_list=sensitivity_seed_list,
        )

    main_run["rows_df"].to_csv(out_dir / "run_metrics.csv", index=False, encoding="utf-8")
    if aux_run is not None:
        aux_run["rows_df"].to_csv(out_dir / "run_metrics_aux_extra_train.csv", index=False, encoding="utf-8")

    summary = {
        "dataset": {
            "primary_xlsx_path": str(xlsx_path),
            "aux_mat_path": str(aux_mat_path),
            "n_primary_raw_samples_before_main_task_filter": int(
                PRIMARY_DATASET_INFO.get("n_raw_samples_before_main_task_filter", x_primary.shape[0])
            ),
            "n_primary_main_task_samples": int(x_primary.shape[0]),
            "n_primary_features": int(x_primary.shape[1]),
            "n_classes": int(len(np.unique(y_primary))),
            "main_task_label_order": list(LABELS),
            "control_only_labels_excluded_from_main_task": list(CONTROL_ONLY_LABELS),
            "main_task_definition": "10-strain classification at fixed concentration (1e-8); control/negative samples excluded from training",
            "weak_signal_proxy_definition": "synthetic attenuation+noise+drift perturbation used for weak-signal robustness evaluation (not real concentration/LOD labels)",
            "n_aux_extra_samples": int(x_extra.shape[0]),
            "aux_extra_class_distribution": {
                str(int(k)): int(v) for k, v in zip(*np.unique(y_extra, return_counts=True))
            },
            "synthetic_levels": {str(k): v[0] for k, v in LOD_LEVELS.items()},
            "wavelength_min": float(np.min(wavelengths)),
            "wavelength_max": float(np.max(wavelengths)),
            "spectral_window_cm1": (
                [float(SPECTRAL_WINDOW_CM1[0]), float(SPECTRAL_WINDOW_CM1[1])]
                if SPECTRAL_WINDOW_CM1 is not None
                else None
            ),
            "spectral_window_mode": "cropped" if SPECTRAL_WINDOW_CM1 is not None else "full",
            "primary_protocol_alignment": PRIMARY_DATASET_INFO,
        },
        "hardware": {
            "device": str(DEVICE),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "primary_result_no_extra_train": main_run["summary"],
        "sensitivity_result_with_extra_train": aux_run["summary"] if aux_run is not None else None,
        "total_elapsed_seconds": float(time.time() - t0),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if aux_run is not None:
        (out_dir / "summary_aux_extra_train.json").write_text(
            json.dumps(aux_run["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary["primary_result_no_extra_train"]["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
