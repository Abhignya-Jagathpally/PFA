"""Write pred_val.npy / pred_test.npy from a run's best saved weights (for runs stopped at a fixed epoch budget).

Usage: python -m cafa6.dump_preds --run abl_no_mcm [--splits val test] [--topk 500]
(set CAFA6_PROC for runs on another dataset). --topk K writes pred_{split}_top{K}.npz instead of the dense
matrix; read either format with cafa6.data.load_pred. Runs trained with knn_fusion get their retrieval inputs from
the same kNN files as in training (data.load_knn). The run's pred_id_sets (e.g. tv) are predicted too.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import PROC, ROOT, KnnBank, load_knn, save_pred
from cafa6.explain import load_run
from cafa6.train import Batcher, predict, prediction_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--topk", type=int, default=None, help="default: the run's pred_topk (0 = dense)")
    args = ap.parse_args()
    model, cfg, vocab, tmap, tok = load_run(args.run)
    df = pd.read_parquet(PROC / "train.parquet")
    seqs = df.seq.tolist()
    taxa = df.taxon.map(lambda t: tmap.get(int(t), 0)).values
    knn = None
    if cfg.get("knn_fusion"):
        Y = sp.load_npz(PROC / "Y.npz").tocsr()
        knn = KnnBank([load_knn(cfg, df, Y, s, np.where(df.split == s)[0], vocab.aspect.values) for s in args.splits])
    batcher = Batcher(seqs, tok, cfg["max_len"], cfg["tokens_per_batch"] * 2, cfg["max_batch"] * 2, knn=knn)
    top_k = cfg.get("pred_topk", 0) if args.topk is None else args.topk
    for split, idx in prediction_sets({**cfg, "pred_splits": args.splits}, df).items():
        save_pred(ROOT / "runs" / args.run, split, predict(model, batcher, seqs, taxa, idx, "cuda"), top_k)
    (ROOT / "logs" / f"{args.run}.done").touch()
    print("dumped", args.run)


if __name__ == "__main__":
    main()
