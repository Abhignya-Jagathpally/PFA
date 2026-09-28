"""Head-to-head on GO-GPT's own benchmark: rebuild HiGO's inputs from GO-GPT's training data.

Train/val = wanglab/gogpt-training-data (annotations up to Nov 2022, 133,492 proteins, already propagated),
test = the paper's temporal holdout (runs/gogpt_cafa5/ref/test.parquet, 8,630 proteins).
Ontology = GO 2023-01-01 and IA = CAFA-5 IA.txt shipped with BioReason-Pro, exactly as used for GO-GPT's scores.
Output mirrors data/proc so that `CAFA6_PROC=data/proc_gogpt python -m cafa6.train ...` works unchanged.

Usage: python -m cafa6.prepare_gogpt
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from huggingface_hub import hf_hub_download

from cafa6.data import ASPECT_ROOT, AMBIGUOUS, ROOT, ancestor_closure, load_go
from cafa6.homology import diamond_score, diamond_search, max_identity

OUT = ROOT / "data" / "proc_gogpt"
REF = ROOT / "runs" / "gogpt_cafa5" / "ref"


def as_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        return list(ast.literal_eval(x))
    return list(x)


def main(min_count: int = 30):
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for split, f in (("train", "data/train-00000-of-00001.parquet"), ("val", "data/validation-00000-of-00001.parquet")):
        p = hf_hub_download("wanglab/gogpt-training-data", f, repo_type="dataset", local_dir="/tmp/gogpt_train")
        d = pd.read_parquet(p, columns=["protein_id", "organism", "sequence", "go_bp", "go_mf", "go_cc"])
        d["split"] = split
        parts.append(d)
    te = pd.read_parquet(REF / "test.parquet", columns=["protein_id", "organism", "sequence", "go_bp", "go_mf", "go_cc"])
    te["split"] = "test"
    df = pd.concat(parts + [te], ignore_index=True).rename(columns={"protein_id": "id", "sequence": "seq"})
    df["length"] = df.seq.str.len()
    amb = df.seq.map(lambda s: sum(c in AMBIGUOUS for c in s) / max(len(s), 1))
    df = df[(df.split == "test") | ((amb <= 0.05) & (df.length >= 20))].reset_index(drop=True)
    orgs = sorted(df.organism.fillna("?").unique())
    df["taxon"] = df.organism.fillna("?").map({o: i + 1 for i, o in enumerate(orgs)}).astype(int)

    parents, aspect, alt = load_go(REF / "go-basic_gogpt.obo")
    closure = ancestor_closure(parents)
    labels = []
    for gb, gm, gc in zip(df.go_bp, df.go_mf, df.go_cc):
        s = set()
        for t in as_list(gb) + as_list(gm) + as_list(gc):
            t = alt.get(t, t)
            s.add(t); s |= closure.get(t, set())
        labels.append(s)
    tr = np.where(df.split == "train")[0]
    cnt = pd.Series([t for i in tr for t in labels[i]]).value_counts()
    roots = set(ASPECT_ROOT.values())
    ia = pd.read_csv(REF / "IA.txt", sep="\t", header=None, names=["term", "ia"]).set_index("term").ia
    vocab = [t for t, c in cnt.items() if c >= min_count and t not in roots and aspect.get(t) in ("F", "P", "C")]
    vocab_df = pd.DataFrame({"term": vocab, "aspect": [aspect[t] for t in vocab],
                             "count": [int(cnt[t]) for t in vocab], "ia": [float(ia.get(t, 0.0)) for t in vocab]})
    vocab_df = vocab_df.sort_values(["aspect", "count"], ascending=[True, False]).reset_index(drop=True)
    tidx = {t: i for i, t in enumerate(vocab_df.term)}
    r, c = zip(*[(i, tidx[t]) for i, s in enumerate(labels) for t in s if t in tidx])
    Y = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(df), len(tidx)))
    anc_pairs = [(tidx[a], tidx[b]) for a in tidx for b in closure.get(a, ()) if b in tidx]

    df["cluster"] = df.id
    df[["id", "taxon", "seq", "length", "split", "cluster", "organism"]].to_parquet(OUT / "train.parquet")
    vocab_df.to_csv(OUT / "vocab.tsv", sep="\t", index=False)
    sp.save_npz(OUT / "Y.npz", Y)
    np.save(OUT / "go_ancestors.npy", np.array(anc_pairs, dtype=np.int32))
    for s in ("val", "test"):
        idx = np.where(df.split == s)[0]
        rows = [(df.id[i], t) for i in idx for t in labels[i] if aspect.get(t) in ("F", "P", "C")]
        pd.DataFrame(rows).to_csv(OUT / f"gt_{s}.tsv", sep="\t", header=False, index=False)

    # homology kNN from the same training data (GO-GPT had these labels too), self hits excluded
    db = df[df.split == "train"]
    Ytr = Y[np.where(df.split == "train")[0]]
    ident = []
    for s in ("val", "test"):
        q = df[df.split == s]
        hits = diamond_search(db, q, threads=32)
        np.save(OUT / f"diamond_{s}.npy", diamond_score(hits, q.id.values, db.id.values, Ytr).astype(np.float16))
        ident.append(pd.DataFrame({"id": q.id.values, "split": s,
                                   "max_pident": max_identity(hits, q.id.values).values}))
    pd.concat(ident).to_parquet(OUT / "identity.parquet")
    rep = {"n": df.split.value_counts().to_dict(), "vocab": vocab_df.aspect.value_counts().to_dict(),
           "n_anc_pairs": len(anc_pairs), "n_organisms": len(orgs)}
    (OUT / "prepare_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
