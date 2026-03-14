#!/usr/bin/env python3

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
LABELS = MAIN_TASK_LABELS

PRIMARY_SAMPLES_PER_RAW_LABEL = 10

RAW_TO_MAIN_CLASS = {
    raw_idx: main_idx for main_idx, raw_idx in enumerate(i for i, name in enumerate(RAW_LABELS) if name in MAIN_TASK_LABELS)
}
MAIN_TASK_RAW_INDICES = sorted(RAW_TO_MAIN_CLASS.keys())

SPECTRAL_WINDOW_CM1_DEFAULT = (400.0, 1200.0)
SPECTRAL_WINDOW_CM1: tuple[float, float] | None = SPECTRAL_WINDOW_CM1_DEFAULT

# Filled by load_primary_data() for reporting/repro.
PRIMARY_DATASET_INFO: dict = {}


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
    if int(mask.sum()) < 10:
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


def resolve_primary_xlsx_path(base_dir: Path) -> Path:
    """Find `108.xlsx` in the workspace (preferring bacteria2 and excluding the Emory MAT folder)."""
    preferred = base_dir / "bacteria2" / "整理文档" / "108.xlsx"
    if preferred.exists():
        return preferred

    matches = list(base_dir.rglob("108.xlsx"))
    candidates: list[Path] = []
    for p in matches:
        s = str(p).lower()
        if "bacteria2" in s and "van1-10" not in s and ".aibin" not in s:
            candidates.append(p)
    if candidates:
        # Prefer the shortest path to avoid nested pullback mirrors.
        return sorted(candidates, key=lambda x: len(str(x)))[0]
    if not matches:
        raise FileNotFoundError("Cannot find 108.xlsx under workspace")
    return matches[0]


def load_primary_data(xlsx_path: Path):
    """Load Main10 (excluding controls) as (X, y, wavelengths_after_window)."""
    df = pd.read_excel(xlsx_path, sheet_name=1)
    wavelengths = df.iloc[:, 0].to_numpy(dtype=np.float32)
    spectra = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    x_full = spectra.T  # [samples, features]

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
    control_labels_present = [RAW_LABELS[int(y_raw[i])] for i in np.where(~keep_mask)[0]]

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
                [float(SPECTRAL_WINDOW_CM1[0]), float(SPECTRAL_WINDOW_CM1[1])] if SPECTRAL_WINDOW_CM1 is not None else None
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
