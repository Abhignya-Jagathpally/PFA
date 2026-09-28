"""Write pred_val.npy / pred_test.npy from a run's best saved weights (for runs stopped at a fixed epoch budget).

Usage: python -m cafa6.dump_preds --run abl_no_mcm   (set CAFA6_PROC for runs on another dataset)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cafa6.data import PROC, ROOT
from cafa6.explain import load_run
from cafa6.train import Batcher, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    model, cfg, vocab, tmap, tok = load_run(args.run)
    df = pd.read_parquet(PROC / "train.parquet")
    seqs = df.seq.tolist()
    taxa = df.taxon.map(lambda t: tmap.get(int(t), 0)).values
    batcher = Batcher(seqs, tok, cfg["max_len"], cfg["tokens_per_batch"] * 2, cfg["max_batch"] * 2)
    for split in ("val", "test"):
        idx = np.where(df.split == split)[0]
        np.save(ROOT / "runs" / args.run / f"pred_{split}.npy", predict(model, batcher, seqs, taxa, idx, "cuda"))
    (ROOT / "logs" / f"{args.run}.done").touch()
    print("dumped", args.run)


if __name__ == "__main__":
    main()
