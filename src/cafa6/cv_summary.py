"""5-fold cross-validation summary: best validation IA-weighted Fmax within a fixed epoch budget per fold, mean and std.

Folds partition the train+val clusters (the held-out test clusters are never used), see `fold` in cafa6.train.
Usage: python -m cafa6.cv_summary [--prefix cv_35m_fold]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from cafa6.data import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cv_35m_fold")
    ap.add_argument("--max_epochs", type=int, default=4, help="same epoch budget for every fold")
    args = ap.parse_args()
    rows = []
    for run in sorted((ROOT / "runs").glob(f"{args.prefix}*")):
        log = [json.loads(l) for l in open(run / "train_log.jsonl")][:args.max_epochs]
        b = max(log, key=lambda r: r["val_Fw_mean"])
        rows.append({"fold": run.name, "best_epoch": b["epoch"], "epochs_run": len(log),
                     "MF": b["val_Fw_F"], "BP": b["val_Fw_P"], "CC": b["val_Fw_C"], "mean": b["val_Fw_mean"]})
    df = pd.DataFrame(rows).set_index("fold")
    stats = df[["MF", "BP", "CC", "mean"]].agg(["mean", "std"])
    out = ROOT / "runs" / "cv_summary"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "folds.tsv", sep="\t")
    stats.to_csv(out / "stats.tsv", sep="\t")
    print(df.round(4).to_string())
    print(stats.round(4).to_string())


if __name__ == "__main__":
    main()
