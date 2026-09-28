"""Head-to-head with GO-GPT on its own temporal-holdout test set (paper Table S3 / S6 protocols).

Both models see the same training proteins and labels (wanglab/gogpt-training-data), GO 2023-01-01, CAFA-5 IA,
CAFA-evaluator (prop=max, norm=cafa). GO-GPT outputs binary term sets (score 1), HiGO outputs probabilities.
The homology gate is fit on the GO-GPT validation split and frozen before touching the test set.

Usage: CAFA6_PROC=data/proc_gogpt python -m cafa6.bench_gogpt --runs gogptbench_35m_lora
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import PROC, ROOT
from cafa6.evaluate import dense_to_long, evaluate, summarize
from cafa6.homology import identity_bin
from cafa6.stack import apply_gate, features, fit_gate, mcm_numpy

REF = ROOT / "runs" / "gogpt_cafa5" / "ref"
OBO, IA = REF / "go-basic_gogpt.obo", REF / "IA.txt"


def gogpt_preds(path_glob="preds_shard*.jsonl", sub=""):
    rows = []
    for f in sorted((ROOT / "runs" / "gogpt_cafa5" / sub).glob(path_glob)):
        for line in open(f):
            r = json.loads(line)
            for a in ("MF", "BP", "CC"):
                rows += [(r["protein_id"], t, 1.0) for t in r.get(a, [])]
    return pd.DataFrame(rows, columns=["id", "term", "score"]).drop_duplicates(["id", "term"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="bench_gogpt")
    args = ap.parse_args()
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    terms, aspect = vocab.term.values, vocab.aspect.values
    anc = np.load(PROC / "go_ancestors.npy")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    ident = pd.read_parquet(PROC / "identity.parquet")
    va, te = np.where(df.split == "val")[0], np.where(df.split == "test")[0]
    va_ids, te_ids = df.id.values[va], df.id.values[te]
    knn_va = np.load(PROC / "diamond_val.npy").astype(np.float32)
    knn_te = np.load(PROC / "diamond_test.npy").astype(np.float32)
    # 760 test proteins also occur in train/val (other aspects), so identity is looked up per split
    Xva, Xte = features(ident[ident.split == "val"], va_ids), features(ident[ident.split == "test"], te_ids)
    Yva = Y[va].toarray()

    preds = {"gogpt_beam": gogpt_preds(), "gogpt_greedy": gogpt_preds(sub="greedy")}
    freq = np.asarray(Y[np.where(df.split == "train")[0]].mean(0)).ravel().astype(np.float32)
    has = knn_te.sum(1) > 0
    dense = {"naive": np.tile(freq, (len(te), 1)),
             "diamond+naive": np.where(has[:, None], 0.8 * knn_te + 0.2 * freq, freq)}
    gates = {}
    for r in args.runs:
        pv = np.load(ROOT / "runs" / r / "pred_val.npy").astype(np.float32)
        pt = np.load(ROOT / "runs" / r / "pred_test.npy").astype(np.float32)
        W = fit_gate(pv, knn_va, Yva, Xva, aspect)
        gates[r] = {a: W[a].round(4).tolist() for a in W}
        dense[r] = pt
        dense[f"{r}+knn_gate"] = mcm_numpy(apply_gate(pt, knn_te, Xte, aspect, W), anc)
    for k, s in dense.items():
        preds[k] = dense_to_long(s, te_ids, terms)
    (out / "gates.json").write_text(json.dumps(gates, indent=2))

    gt = PROC / "gt_test.tsv"
    common = set(open(REF / "common_proteins.txt").read().split())
    subsets = {"S3_all": None, "S6_noknowledge": [p for p in te_ids if p not in common]}
    idt = ident[ident.split == "test"].set_index("id").loc[te_ids]
    bins = identity_bin(idt.max_pident)
    for b in ["no_hit", "<30%", "30-50%", ">=50%"]:
        subsets[f"identity {b}"] = list(idt.index[bins == b])
    tables = []
    for name, ids in subsets.items():
        res = evaluate(preds, gt, ids_subset=ids, obo=OBO, ia=IA, out_dir=out / name if ids is None else None)
        s = summarize(res)
        f = res.pivot(index="method", columns="aspect", values="Fmax")
        s = s.join(f.add_prefix("Fmax_"))
        s["subset"], s["n"] = name, len(te_ids) if ids is None else len(ids)
        tables.append(s)
        print(f"== {name} (n={s.n.iloc[0]})\n", s.round(4).to_string())
    pd.concat(tables).reset_index().to_csv(out / "summary.tsv", sep="\t", index=False)
    (ROOT / "logs" / f"{args.out}.done").touch()


if __name__ == "__main__":
    main()
