"""Step 8: Kaggle CAFA-6 submission for Test/testsuperset.fasta.

final score = MCM( g * DiamondScore(all labelled proteins) + (1 - g) * HiGO )
with the per-aspect gate g fit on the validation split (see cafa6.stack). Self-hits are excluded from
DIAMOND so test proteins that also appear in the training file are scored the same way as in validation.

Usage: python -m cafa6.submit --run dev_35m_lora [--top_k 500] [--min_score 0.02]
"""
from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from cafa6.data import PROC, ROOT
from cafa6.explain import load_run
from cafa6.homology import diamond_score, diamond_search, max_identity
from cafa6.stack import apply_gate, features, fit_gate, mcm_numpy
from cafa6.train import Batcher, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top_k", type=int, default=500)
    ap.add_argument("--min_score", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()
    out = ROOT / "runs" / args.run / "submission"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    anc = np.load(PROC / "go_ancestors.npy")
    ident_val = pd.read_parquet(PROC / "identity.parquet")
    model, cfg, vocab, tmap, tok = load_run(args.run)
    terms, aspect = vocab.term.values, vocab.aspect.values

    # gate fit on validation (identical to cafa6.stack)
    va = np.where(df.split == "val")[0]
    W = fit_gate(np.load(ROOT / "runs" / args.run / "pred_val.npy").astype(np.float32),
                 np.load(PROC / "diamond_val.npy").astype(np.float32), Y[va].toarray(),
                 features(ident_val, df.id.values[va]), aspect)
    json.dump({a: W[a].tolist() for a in W}, open(out / "gate.json", "w"))

    hits_path = PROC / "diamond_hits_kaggle_test.parquet"
    if hits_path.exists():
        hits = pd.read_parquet(hits_path)
    else:
        hits = diamond_search(df, test, max_hits=50, exclude_self=True)
        hits.to_parquet(hits_path)
    mi = max_identity(hits, test.id.values)
    ident = pd.DataFrame({"id": test.id.values, "max_pident": mi.values})

    batcher = Batcher(test.seq.tolist(), tok, cfg["max_len"], cfg["tokens_per_batch"] * 2, cfg["max_batch"] * 2)
    taxa = test.taxon.map(lambda t: tmap.get(int(t), 0)).values
    n_rows = 0
    with gzip.open(out / "submission.tsv.gz", "wt") as f:
        for s in range(0, len(test), args.chunk):
            idx = np.arange(s, min(s + args.chunk, len(test)))
            ids = test.id.values[idx]
            pm = predict(model, batcher, batcher.seqs, taxa, idx, "cuda").astype(np.float32)
            pk = diamond_score(hits, ids, df.id.values, Y)
            p = mcm_numpy(apply_gate(pm, pk, features(ident, ids), aspect, W), anc)
            k = min(args.top_k, p.shape[1])
            top = np.argpartition(-p, k - 1, axis=1)[:, :k]
            for r in range(len(idx)):
                cols = top[r][p[r, top[r]] >= args.min_score]
                cols = cols[np.argsort(-p[r, cols])]
                f.writelines(f"{ids[r]}\t{terms[c]}\t{p[r, c]:.3f}\n" for c in cols)
                n_rows += len(cols)
            print(f"{idx[-1] + 1}/{len(test)} proteins, {n_rows} rows", flush=True)
    info = {"run": args.run, "n_proteins": int(len(test)), "n_rows": int(n_rows), "top_k": args.top_k,
            "min_score": args.min_score, "frac_with_diamond_hit": float((mi > 0).mean())}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    (ROOT / "logs" / f"submit_{args.run}.done").touch()


if __name__ == "__main__":
    main()
