"""Step 2: clean -> propagate -> cluster (MMseqs2) -> cluster-level split -> label space.

Usage: python -m cafa6.prepare [--min_id 0.3] [--min_count 30]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import (ASPECT_ROOT, PROC, RAW, ROOT, ancestor_closure, load_go, load_ia,
                        propagate, read_fasta)

MMSEQS = ROOT / "tools" / "mmseqs" / "bin" / "mmseqs"


def write_fasta(df: pd.DataFrame, path: Path):
    with open(path, "w") as f:
        for pid, s in zip(df.id, df.seq):
            f.write(f">{pid}\n{s}\n")


def mmseqs_cluster(df: pd.DataFrame, min_id: float, cov: float, threads: int) -> pd.Series:
    with tempfile.TemporaryDirectory(dir=PROC) as tmp:
        tmp = Path(tmp)
        write_fasta(df, tmp / "in.fasta")
        subprocess.run([str(MMSEQS), "easy-cluster", str(tmp / "in.fasta"), str(tmp / "clu"), str(tmp / "work"),
                        "--min-seq-id", str(min_id), "-c", str(cov), "--cov-mode", "0",
                        "--cluster-mode", "0", "--threads", str(threads), "-v", "1"], check=True)
        clu = pd.read_csv(tmp / "clu_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
    return clu.set_index("member").rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_id", type=float, default=0.3)
    ap.add_argument("--cov", type=float, default=0.8)
    ap.add_argument("--min_count", type=int, default=30, help="min propagated train annotations per term")
    ap.add_argument("--max_ambig", type=float, default=0.05)
    ap.add_argument("--min_len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args)}

    train = read_fasta(RAW / "Train" / "train_sequences.fasta")
    test = read_fasta(RAW / "Test" / "testsuperset.fasta")
    report["n_train_raw"], report["n_test"] = len(train), len(test)

    # 1) cleaning (train only; every test protein must still receive predictions)
    bad_ambig = train.ambig_frac > args.max_ambig
    bad_len = train.length < args.min_len
    report["dropped_ambig"], report["dropped_short"] = int(bad_ambig.sum()), int((bad_len & ~bad_ambig).sum())
    train = train[~bad_ambig & ~bad_len].reset_index(drop=True)
    report["n_train_clean"] = len(train)

    # 2) GO propagation (is_a + part_of, alt_id remapped)
    parents, aspect, alt = load_go()
    closure = ancestor_closure(parents)
    terms = pd.read_csv(RAW / "Train" / "train_terms.tsv", sep="\t")
    terms = terms[terms.EntryID.isin(set(train.id))]
    prop = propagate(terms, closure, alt)
    prop["aspect"] = prop.term.map(aspect)
    report["annotations_leaf"], report["annotations_propagated"] = len(terms), len(prop)
    train = train[train.id.isin(set(prop.id))].reset_index(drop=True)

    # 3) redundancy clustering -> cluster-level 80/10/10 split
    rep = mmseqs_cluster(train, args.min_id, args.cov, args.threads)
    train["cluster"] = train.id.map(rep)
    clusters = np.array(sorted(train.cluster.unique()), dtype=object)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(clusters)
    n = len(clusters)
    split_of = {c: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, c in enumerate(clusters)}
    train["split"] = train.cluster.map(split_of)
    report["n_clusters"] = int(n)
    report["split_sizes"] = train.split.value_counts().to_dict()

    # 4) length statistics for the tokenizer max length
    report["length_p50_p95_p99_max"] = [int(x) for x in np.percentile(train.length, [50, 95, 99, 100])]
    report["test_length_p95"] = int(np.percentile(test.length, 95))

    # 5) label space: terms with >= min_count propagated annotations among train-split proteins, roots excluded
    tr_ids = set(train.id[train.split == "train"])
    cnt = prop[prop.id.isin(tr_ids)].term.value_counts()
    roots = set(ASPECT_ROOT.values())
    vocab = [t for t, c in cnt.items() if c >= args.min_count and t not in roots and aspect.get(t) in ("F", "P", "C")]
    ia = load_ia()
    vocab_df = pd.DataFrame({"term": vocab, "aspect": [aspect[t] for t in vocab],
                             "count": [int(cnt[t]) for t in vocab], "ia": [ia.get(t, 0.0) for t in vocab]})
    vocab_df = vocab_df.sort_values(["aspect", "count"], ascending=[True, False]).reset_index(drop=True)
    tidx = {t: i for i, t in enumerate(vocab_df.term)}
    report["vocab_size"] = vocab_df.aspect.value_counts().to_dict()
    report["ia_mass_covered_by_vocab"] = float(
        prop[prop.id.isin(tr_ids)].term.map(lambda t: ia.get(t, 0.0) if t in tidx else 0.0).sum()
        / prop[prop.id.isin(tr_ids)].term.map(lambda t: ia.get(t, 0.0)).sum())

    # label matrix (proteins x vocab) and hierarchy edges within vocab (child -> nearest vocab ancestors)
    pidx = {p: i for i, p in enumerate(train.id)}
    pv = prop[prop.term.isin(tidx)]
    Y = sp.csr_matrix((np.ones(len(pv), np.float32), (pv.id.map(pidx).values, pv.term.map(tidx).values)),
                      shape=(len(train), len(tidx)))
    edges = [(tidx[c], tidx[p]) for c in tidx for p in parents.get(c, ()) if p in tidx]
    anc_pairs = [(tidx[c], tidx[a]) for c in tidx for a in closure.get(c, ()) if a in tidx]

    train.drop(columns=["ambig_frac"]).to_parquet(PROC / "train.parquet")
    test.drop(columns=["ambig_frac"]).to_parquet(PROC / "test.parquet")
    prop.to_parquet(PROC / "train_terms_propagated.parquet")
    vocab_df.to_csv(PROC / "vocab.tsv", sep="\t", index=False)
    sp.save_npz(PROC / "Y.npz", Y)
    np.save(PROC / "go_edges.npy", np.array(edges, dtype=np.int32))
    np.save(PROC / "go_ancestors.npy", np.array(anc_pairs, dtype=np.int32))
    report["n_parent_edges_in_vocab"], report["n_ancestor_pairs_in_vocab"] = len(edges), len(anc_pairs)

    # ground-truth files for cafaeval (propagated is fine; cafaeval re-propagates)
    for s in ("val", "test"):
        ids = set(train.id[train.split == s])
        terms[terms.EntryID.isin(ids)][["EntryID", "term"]].to_csv(PROC / f"gt_{s}.tsv", sep="\t", header=False, index=False)
    (PROC / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
