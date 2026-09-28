"""Precision / recall / F1 tables per aspect for the CAFA-6 clustered test and the GO-GPT temporal test.

Weighted metrics are IA-weighted precision/recall at each model's weighted-Fmax threshold per aspect;
the unweighted variant uses precision/recall at the unweighted-Fmax threshold. General = mean over MF/BP/CC
of P, R and F1; Kaggle = mean weighted F1 (identical to General F1, kept as an explicit column).

data.PROC is fixed at import time from CAFA6_PROC, so each bench runs in a subprocess with the right env
whenever the current process was started with a different one.

Usage: python -m cafa6.prf_table --bench cafa6|gogpt|all [--extra NAME=RUNDIR ...] [--extra-gogpt NAME=RUNDIR ...]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import PROC, ROOT
from cafa6.evaluate import dense_to_long, evaluate

OUT = ROOT / "runs" / "prf"
BENCH_PROC = {"cafa6": ROOT / "data" / "proc", "gogpt": ROOT / "data" / "proc_gogpt"}
BENCH_TITLE = {"cafa6": "CAFA-6 clustered test", "gogpt": "GO-GPT temporal test (S3_all)"}
ASPECTS3 = ("MF", "BP", "CC")


def _load_common():
    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    va, te = np.where(df.split == "val")[0], np.where(df.split == "test")[0]
    freq = np.asarray(Y[np.where(df.split == "train")[0]].mean(0)).ravel().astype(np.float32)
    return dict(df=df, terms=vocab.term.values, aspect=vocab.aspect.values, anc=np.load(PROC / "go_ancestors.npy"),
                ident=pd.read_parquet(PROC / "identity.parquet"), va_ids=df.id.values[va], te_ids=df.id.values[te],
                Yva=Y[va].toarray(), freq=freq,
                knn_va=np.load(PROC / "diamond_val.npy").astype(np.float32),
                knn_te=np.load(PROC / "diamond_test.npy").astype(np.float32))


def _gated(c, run, Xva, Xte):
    from cafa6.stack import apply_gate, fit_gate, mcm_numpy
    pv = np.load(ROOT / "runs" / run / "pred_val.npy").astype(np.float32)
    pt = np.load(ROOT / "runs" / run / "pred_test.npy").astype(np.float32)
    W = fit_gate(pv, c["knn_va"], c["Yva"], Xva, c["aspect"])
    return pt, mcm_numpy(apply_gate(pt, c["knn_te"], Xte, c["aspect"], W), c["anc"])


def _final(run):
    return np.load(ROOT / "runs" / run / "pred_test.npy").astype(np.float32)


def build_preds(bench: str, extras: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    """Ordered {row label: long-format predictions} for one bench."""
    from cafa6.stack import features
    c = _load_common()
    te_ids, freq, knn_te = c["te_ids"], c["freq"], c["knn_te"]
    has = knn_te.sum(1) > 0
    dense = {"Naive": np.tile(freq, (len(te_ids), 1)),
             "DIAMOND + naive": np.where(has[:, None], 0.8 * knn_te + 0.2 * freq, freq)}
    long = {}
    if bench == "cafa6":
        Xva, Xte = features(c["ident"], c["va_ids"]), features(c["ident"], te_ids)
        for run, label, glabel in [("base_35m_frozen_mlp", "Frozen ESM2 + MLP", "Frozen ESM2 + MLP + homology gate"),
                                   ("dev_35m_lora", "HiGO (LoRA)", "HiGO + homology gate")]:
            raw, gated = _gated(c, run, Xva, Xte)
            dense[label], dense[glabel] = raw, gated
        grounded = "grounded"
    else:
        from cafa6.bench_gogpt import gogpt_preds
        ident = c["ident"]
        Xva = features(ident[ident.split == "val"], c["va_ids"])
        Xte = features(ident[ident.split == "test"], te_ids)
        long["GO-GPT (beam)"] = gogpt_preds()
        if list((ROOT / "runs" / "gogpt_cafa5" / "greedy").glob("preds_shard*.jsonl")):
            long["GO-GPT (greedy)"] = gogpt_preds(sub="greedy")
        raw, gated = _gated(c, "gogptbench_35m_lora", Xva, Xte)
        dense["HiGO (LoRA)"], dense["HiGO + homology gate"] = raw, gated
        grounded = "grounded_gogpt"
    if (ROOT / "runs" / grounded / "pred_test.npy").exists():
        dense["HiGO-v2 grounded"] = _final(grounded)
    for name, run in extras:
        dense[name] = _final(run)
    out = {}
    for label in ("Naive", "DIAMOND + naive", "GO-GPT (beam)", "GO-GPT (greedy)"):
        if label in long:
            out[label] = long.pop(label)
        if label in dense:
            out[label] = dense_to_long(dense.pop(label), te_ids, c["terms"])
    for label, s in dense.items():
        assert s.shape == (len(te_ids), len(c["terms"])), (label, s.shape)
        out[label] = dense_to_long(s, te_ids, c["terms"])
    return out


def run_bench(bench: str, extras: list[tuple[str, str]], n_cpu: int) -> pd.DataFrame:
    preds = build_preds(bench, extras)
    labels = list(preds)
    files = {f"m{i:02d}": preds[k] for i, k in enumerate(labels)}  # labels contain spaces/'+'
    kw = {}
    if bench == "gogpt":
        from cafa6.bench_gogpt import IA, OBO
        kw = dict(obo=OBO, ia=IA)
    res = evaluate(files, PROC / "gt_test.tsv", n_cpu=n_cpu, **kw)
    res["method"] = res.method.map(dict(zip(files, labels)))
    tab = pd.DataFrame(index=pd.Index(labels, name="model"))
    for a in ASPECTS3:
        r = res[res.aspect == a].set_index("method")
        for col, src in [("P", "P_w"), ("R", "R_w"), ("F1", "Fmax_w"), ("tau", "tau_w"),
                         ("P_unw", "P"), ("R_unw", "R"), ("Fmax_unw", "Fmax")]:
            tab[f"{a}_{col}"] = r[src].reindex(labels)
    for col in ("P", "R", "F1", "P_unw", "R_unw", "Fmax_unw"):
        tab[f"General_{col}"] = tab[[f"{a}_{col}" for a in ASPECTS3]].mean(axis=1)
    tab["Kaggle"] = tab["General_F1"]
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / f"{bench}_test.tsv", sep="\t")
    print(f"== {bench}\n", tab.round(4).to_string())
    return tab


def _md(tab: pd.DataFrame, cols: list[str], suffix: str) -> str:
    groups = list(ASPECTS3) + ["General"]
    head = ["Model"] + [f"{g} {c}" for g in groups for c in cols] + (["Kaggle"] if not suffix else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m, r in tab.iterrows():
        src = [f"{g}_{c}{suffix}" for g in groups for c in cols]
        vals = [f"{r[s]:.3f}" for s in src] + ([f"{r['Kaggle']:.3f}"] if not suffix else [])
        lines.append("| " + " | ".join([m] + vals) + " |")
    return "\n".join(lines)


def write_markdown():
    parts = ["# Precision / recall / F1 per GO aspect\n",
             "Weighted = IA-weighted precision/recall/F1 at each model's weighted-Fmax threshold per aspect "
             "(CAFA-evaluator, prop=max, norm=cafa). General = mean over MF/BP/CC; Kaggle = mean weighted F1. "
             "Unweighted = precision/recall at the unweighted-Fmax threshold.\n"]
    for bench in ("cafa6", "gogpt"):
        f = OUT / f"{bench}_test.tsv"
        if not f.exists():
            continue
        tab = pd.read_csv(f, sep="\t", index_col=0)
        parts += [f"## {BENCH_TITLE[bench]}\n", "### IA-weighted (Precision, Recall, F1)\n",
                  _md(tab, ["P", "R", "F1"], ""), "",
                  "### Unweighted (Precision, Recall, Fmax)\n", _md(tab, ["P", "R", "Fmax"], "_unw"), ""]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    print((OUT / "tables.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["cafa6", "gogpt", "all"], default="all")
    ap.add_argument("--extra", action="append", default=[], help="NAME=RUNDIR, extra final row on the CAFA-6 bench")
    ap.add_argument("--extra-gogpt", action="append", default=[], help="NAME=RUNDIR, extra final row on GO-GPT bench")
    ap.add_argument("--n-cpu", type=int, default=16)
    args = ap.parse_args()
    extras = {"cafa6": [tuple(e.split("=", 1)) for e in args.extra],
              "gogpt": [tuple(e.split("=", 1)) for e in args.extra_gogpt]}
    for bench in (["cafa6", "gogpt"] if args.bench == "all" else [args.bench]):
        if PROC.resolve() == BENCH_PROC[bench].resolve():
            run_bench(bench, extras[bench], args.n_cpu)
            continue
        cmd = [sys.executable, "-m", "cafa6.prf_table", "--bench", bench, "--n-cpu", str(args.n_cpu)]
        cmd += [x for e in args.extra for x in ("--extra", e)] if bench == "cafa6" else \
               [x for e in args.extra_gogpt for x in ("--extra-gogpt", e)]
        env = dict(os.environ, CAFA6_PROC=str(BENCH_PROC[bench]),
                   PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])))
        subprocess.run(cmd, env=env, check=True, cwd=ROOT)
    write_markdown()


if __name__ == "__main__":
    main()
