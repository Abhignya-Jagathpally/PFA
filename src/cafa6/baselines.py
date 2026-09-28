"""Baselines on the clustered val/test splits: naive frequency and DIAMOND kNN.

Usage: python -m cafa6.baselines
Writes runs/baselines/{metrics.tsv,summary.tsv}, data/proc/{diamond_*.npy,identity.parquet}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import PROC, ROOT
from cafa6.evaluate import dense_to_long, evaluate, summarize
from cafa6.homology import diamond_score, diamond_search, identity_bin, max_identity


def main():
    out = ROOT / "runs" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    terms = vocab.term.values
    tr = np.where(df.split == "train")[0]

    freq = np.asarray(Y[tr].mean(0)).ravel().astype(np.float32)
    hits_path = PROC / "diamond_hits_valtest.parquet"
    if hits_path.exists():
        hits = pd.read_parquet(hits_path)
    else:
        hits = diamond_search(df.iloc[tr], df[df.split != "train"])
        hits.to_parquet(hits_path)

    ident = []
    results = []
    for split in ("val", "test"):
        idx = np.where(df.split == split)[0]
        ids = df.id.values[idx]
        naive = np.tile(freq, (len(idx), 1))
        dmd = diamond_score(hits, ids, df.id.values[tr], Y[tr])
        np.save(PROC / f"diamond_{split}.npy", dmd.astype(np.float16))
        # blend used by DeepGOPlus-style systems: homology where available, prior elsewhere
        has_hit = dmd.sum(1) > 0
        blend = np.where(has_hit[:, None], 0.8 * dmd + 0.2 * naive, naive)
        preds = {"naive": dense_to_long(naive, ids, terms), "diamond_knn": dense_to_long(dmd, ids, terms),
                 "diamond+naive": dense_to_long(blend, ids, terms)}
        r = evaluate(preds, PROC / f"gt_{split}.tsv")
        r["split"] = split
        results.append(r)
        mi = max_identity(hits, ids)
        ident.append(pd.DataFrame({"id": ids, "split": split, "max_pident": mi.values,
                                   "id_bin": identity_bin(mi).astype(str).values}))
    res = pd.concat(results)
    res.to_csv(out / "metrics.tsv", sep="\t", index=False)
    ident = pd.concat(ident)
    ident.to_parquet(PROC / "identity.parquet")
    for s in ("val", "test"):
        print(f"== {s}")
        print(summarize(res[res.split == s]).round(4).to_string())
    print(ident.groupby(["split", "id_bin"]).size().unstack())
    summ = {s: summarize(res[res.split == s]).round(4).to_dict() for s in ("val", "test")}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    (ROOT / "logs" / "baselines.done").touch()


if __name__ == "__main__":
    main()
