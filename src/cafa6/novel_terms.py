"""Do the models recover test annotations that conventional homology transfer would have missed?

For every GO-GPT test protein and aspect, the "conventional" prediction is DIAMOND best-hit transfer: all GO terms
(closed under the true-path rule) of the top-bitscore training protein, the classic BLAST baseline of CAFA. Terms in
the test ground truth that this baseline does not contain are "beyond-homology" terms. For each model, at its own
per-aspect IA-weighted Fmax threshold (runs/test_look/look3_l3c0_twin/cafaeval/best_f_w.tsv), the script reports:
  recall_beyond     IA-weighted share of beyond-homology true terms the model predicts (pooled over proteins)
  prec_beyond       IA-weighted share of the model's predictions outside the best-hit set that are correct
  n_prot_beyond     proteins with at least one beyond-homology true term, and how many of them get >= 1 recovered
by identity bin of the nearest training protein. Descriptive only: frozen prediction files, no selection.

  python -m cafa6.novel_terms      -> runs/novel_terms/{summary.tsv, by_bin.tsv, report.md}
  python -m cafa6.novel_terms --out DIR    same files in DIR
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6 import knn_sharp as ks
from cafa6 import test_look as tl

ROOT = ks.ROOT
OUT = ROOT / "runs" / "novel_terms"
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:6.1f}s] {m}", flush=True)


def align(P: sp.csr_matrix, ids, order: np.ndarray, onto) -> sp.csr_matrix:
    """Rows of P reordered to `order` (missing proteins -> empty rows), max-propagated."""
    pos = {p: i for i, p in enumerate(ids)}
    rows = np.array([pos.get(p, -1) for p in order])
    P = onto.propagate_max(P.tocsr())
    out = P[np.where(rows >= 0, rows, 0)]
    return (sp.diags((rows >= 0).astype(float)) @ out).tocsr()


def main():
    global OUT
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", default=str(OUT), help="output directory (default runs/novel_terms)")
    OUT = Path(ap.parse_args().out)
    OUT.mkdir(parents=True, exist_ok=True)
    onto = ks.Ontology()
    sc = ks.Scorer(onto, ks.gt_frame("test"))
    ids = sc.ids
    G = sc.G.tocsr().astype(bool)
    log(f"test proteins with ground truth: {len(ids)}; true (protein, term) pairs after closure: {G.nnz}")

    go = ks.GO()
    kn = ks.build_knn(onto, go, ("test",))["test"]
    B, _ = kn.prior(np.inf, 1, 0, "all")            # labels of the top-bitscore training hit
    B = align(B, kn.ids, ids, onto) > 0
    log(f"best-hit transfer: {B.nnz} predicted pairs")
    ident = pd.Series(np.load(ROOT / "data/proc_gogpt_tv2/knn/test_knn.npz")["max_pident"].astype(float),
                      index=np.load(ROOT / "data/proc_gogpt_tv2/knn/test_knn.npz")["ids"].astype(str))
    bins = ks.ident_bins(ident.reindex(ids).fillna(0).values)
    bins.index = ids

    thr = pd.read_csv(ROOT / "runs/test_look/look3_l3c0_twin/cafaeval/best_f_w.tsv", sep="\t")
    ns_map = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
    thr["a"] = thr.ns.map(ns_map)
    files = {"HiGO-R2.tsv": "HiGO-R2", "GO-GPT_sample10.tsv": "GO-GPT ten-sample", "GO-GPT_beam.tsv": "GO-GPT beam"}
    models = {
        "HiGO-R2": tl.load_pred(ROOT / "runs/stacker_tv/l3_c0/pred_test_twin.npz", onto),
        "GO-GPT ten-sample": tl.load_long(ROOT / "runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz", onto),
    }
    Pg, gids = ks.gogpt_matrix(onto)
    models["GO-GPT beam"] = (gids, Pg)

    ia = onto.ia
    nsc = onto.ns
    rows_summary, rows_bin = [], []
    beyond = (G.astype(np.int8) - G.multiply(B).astype(np.int8)).astype(bool).tocsr()
    beyond.eliminate_zeros()
    for name, (mids, P) in models.items():
        P = align(P, mids, ids, onto)
        for k, a in enumerate(("MF", "BP", "CC")):
            tau = float(thr[(thr.filename.map(files) == name) & (thr.a == a)].tau.iloc[0])
            colmask = (nsc == k) & (ia > 0)
            D = (P >= tau).multiply(sp.csr_matrix(colmask[None, :].astype(float))).tocsr().astype(bool)
            Nk = beyond.multiply(sp.csr_matrix(colmask[None, :].astype(float))).tocsr().astype(bool)
            hit = D.multiply(Nk)                                  # recovered beyond-homology terms
            outside = (D.astype(np.int8) - D.multiply(B).astype(np.int8)).astype(bool).tocsr()
            outside_true = outside.multiply(G)
            w = lambda M: np.asarray(M.astype(float) @ ia).ravel()
            n_w, h_w, o_w, ot_w = w(Nk), w(hit), w(outside), w(outside_true)
            has_n = n_w > 0
            rec_any = np.asarray(hit.sum(1)).ravel() > 0
            for b in ["all"] + ks.BINS:
                m = np.ones(len(ids), bool) if b == "all" else (bins.values == b)
                r = {"model": name, "aspect": a, "bin": b, "tau": tau,
                     "recall_beyond": h_w[m].sum() / max(n_w[m].sum(), 1e-12),
                     "prec_beyond": ot_w[m].sum() / max(o_w[m].sum(), 1e-12),
                     "n_prot_beyond": int((has_n & m).sum()),
                     "n_prot_recovered": int((has_n & rec_any & m).sum())}
                (rows_summary if b == "all" else rows_bin).append(r)
            log(f"{name:18s} {a} tau {tau:.2f}: recall_beyond {rows_summary[-1]['recall_beyond']:.3f} "
                f"prec_beyond {rows_summary[-1]['prec_beyond']:.3f} "
                f"proteins {rows_summary[-1]['n_prot_recovered']}/{rows_summary[-1]['n_prot_beyond']}")
    S, Bb = pd.DataFrame(rows_summary), pd.DataFrame(rows_bin)
    S.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    Bb.to_csv(OUT / "by_bin.tsv", sep="\t", index=False)
    piv = Bb.pivot_table(index=["bin"], columns="model", values="recall_beyond", aggfunc="mean").reindex(ks.BINS)
    pivp = Bb.pivot_table(index=["bin"], columns="model", values="prec_beyond", aggfunc="mean").reindex(ks.BINS)
    nb = Bb[Bb.model == "HiGO-R2"].groupby("bin").n_prot_beyond.sum().reindex(ks.BINS)
    log("recall of beyond-homology terms (mean over aspects) by identity bin:\n" + piv.round(3).to_string())
    log("precision outside the best-hit set (mean over aspects) by identity bin:\n" + pivp.round(3).to_string())
    (OUT / "report.md").write_text(
        "# Beyond-homology GO terms on the GO-GPT temporal test\n\n" + (__doc__ or "") +
        "\n## Pooled, per aspect\n\n" + S.round(4).to_markdown(index=False) +
        "\n\n## Recall of beyond-homology terms by identity bin (mean over aspects)\n\n" +
        piv.round(3).to_markdown() + "\n\n## Precision outside the best-hit set by identity bin\n\n" +
        pivp.round(3).to_markdown() + "\n\n(protein-aspect counts with beyond-homology terms per bin: " +
        ", ".join(f"{k} {int(v)}" for k, v in nb.items()) + ")\n")
    log(f"done; {OUT}")


if __name__ == "__main__":
    main()
