#!/usr/bin/env python3


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[repeat-holdout] run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=15)
    p.add_argument("--base-seed", type=int, default=42, help="Model/random seed used inside scripts.")
    p.add_argument("--stability-iters", type=int, default=30)
    p.add_argument("--crop-k", type=int, default=80)
    p.add_argument("--full-k", type=int, default=320)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "ai_sci" / "stage3" / "run_classical_baselines.py").exists():
        raise FileNotFoundError("run_classical_baselines.py not found")
    if not (repo_root / "ai_sci" / "stage3" / "run_stability_sparse_main10.py").exists():
        raise FileNotFoundError("run_stability_sparse_main10.py not found")

    for s in range(int(args.start_seed), int(args.start_seed) + int(args.n_seeds)):
        cropped_sub = f"repeat_holdout_main10/baseline_cropped_s{s}"
        full_sub = f"repeat_holdout_main10/baseline_full_s{s}"
        ss_crop_sub = f"repeat_holdout_main10/ss_cropped_s{s}"
        ss_full_sub = f"repeat_holdout_main10/ss_full_s{s}"
        split_file = f"ai_sci/stage3/outputs/{cropped_sub}/holdout_split_indices.json"

        _run(
            [
                sys.executable,
                "ai_sci/stage3/run_classical_baselines.py",
                "--eval-mode",
                "holdout",
                "--spectrum-mode",
                "cropped",
                "--models",
                "LDA",
                "--out-subdir",
                cropped_sub,
                "--test-frac",
                "0.2",
                "--val-frac",
                "0.2",
                "--split-seed",
                str(s),
                "--seed",
                str(args.base_seed),
            ]
        )
        _run(
            [
                sys.executable,
                "ai_sci/stage3/run_classical_baselines.py",
                "--eval-mode",
                "holdout",
                "--spectrum-mode",
                "full",
                "--models",
                "LDA",
                "--out-subdir",
                full_sub,
                "--split-file",
                split_file,
                "--seed",
                str(args.base_seed),
            ]
        )
        _run(
            [
                sys.executable,
                "ai_sci/stage3/run_stability_sparse_main10.py",
                "--eval-mode",
                "holdout",
                "--spectrum-mode",
                "cropped",
                "--out-subdir",
                ss_crop_sub,
                "--use-sg",
                "--use-snv",
                "--final-clf",
                "lda",
                "--stability-iters",
                str(args.stability_iters),
                "--subsample-frac",
                "0.7",
                "--l1-ratio",
                "0.7",
                "--k-list",
                str(args.crop_k),
                "--split-file",
                split_file,
                "--seed",
                str(args.base_seed),
            ]
        )
        _run(
            [
                sys.executable,
                "ai_sci/stage3/run_stability_sparse_main10.py",
                "--eval-mode",
                "holdout",
                "--spectrum-mode",
                "full",
                "--out-subdir",
                ss_full_sub,
                "--use-sg",
                "--use-snv",
                "--final-clf",
                "lda",
                "--stability-iters",
                str(args.stability_iters),
                "--subsample-frac",
                "0.7",
                "--l1-ratio",
                "0.7",
                "--k-list",
                str(args.full_k),
                "--split-file",
                split_file,
                "--seed",
                str(args.base_seed),
            ]
        )

    print("[repeat-holdout] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
