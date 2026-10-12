"""L1: per-term post-hoc stacker over kNN evidence, fit on the temporal validation split (TV) only.

For every query and every GO term carried by any of its top-50 DIAMOND hits (the candidate set), the
stacker sees
  p_knn    the L0 prior from data/proc_gogpt_tv/knn/{split}_knn.npz (raw probability, not a clipped logit)
  s_best   best percent identity among hits carrying the term, and its gap to the query's max identity
  frac     fraction of hits carrying the term, among hits annotated in the term's aspect and among all hits
  n_t      number of hits carrying the term (log1p)
  max_id   the query's best identity, has_hit, number of hits
  IA, depth  the term's information accretion (CAFA-5) and its longest-path depth in GO 2023-01-01
  p_sharp  (variant "+sharp" only) a sharp prior, T=2, k=10, aspect-normalised, computed here
and outputs a per-term probability. One model per aspect: a logistic regression on standardised and
squared features, or a histogram gradient-boosting classifier. Scores are made hierarchy-consistent with
the max-constraint (MCM) before scoring, and proteins with no hit get no prediction, like L0.

Fit and evaluation: TV labels are the post-T0 GO-GPT labels (gt_tv.tsv) and the label database is the T0
snapshot, so neighbours never carry the held-out labels. TV scores come from 5-fold cross-validation
over TV proteins; the final model is refit on all of TV and applied to the random val split (scored) and
to the test split (written to runs/stacker_l1/pred_test.npz but NOT scored: the protocol allows no test
look for L1).

The HiGO feature (runs/gogptbench_35m_lora/pred_*.npy) is not used. That model trained on the TV
proteins' post-T0 labels, so its TV predictions would leak. L1-with-HiGO must wait for a HiGO run trained
on data/proc_gogpt_tv that also predicts the tv_ids.txt proteins.

With --tv v2 the same procedure runs on data/proc_gogpt_tv2 (L0 = runs/knn_sharp/chosen_v2.json, priors in
data/proc_gogpt_tv2/knn) and writes runs/stacker_l1_v2; the summary also reports the gain under the
test-like cell weights of knn_sharp.cell_weights.

Usage: python -m cafa6.stacker [--threads 48] [--tv v2]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cafa6 import knn_sharp as ks
from cafa6.data import ROOT
from cafa6.knn_sharp import (BINS, GO, Ontology, Scorer, build_knn, gt_frame, ident_bins, load_knn)
from cafa6.temporal_val import log

RUN = ROOT / "runs" / "stacker_l1"
BASE = ["p_knn", "s_best", "gap", "frac_asp", "frac_all", "log_n", "max_id", "has_hit", "log_hits", "ia", "depth"]
VARIANTS = {"logreg": (BASE, "logreg"), "gbm": (BASE, "gbm"), "gbm+sharp": (BASE + ["p_sharp"], "gbm"),
            "logreg_iaw": (BASE, "logreg_iaw"), "gbm_iaw": (BASE, "gbm_iaw")}


def features(onto: Ontology, kn, prior: dict, sharp: sp.csr_matrix) -> pd.DataFrame:
    """One row per (query, candidate term) with the evidence features."""
    m = kn.rank < 50
    q, s, pid = kn.q[m], kn.s[m], kn.pid[m]
    L = kn.L.tocsr()
    cnt = np.diff(L.indptr)[s]
    rq = np.repeat(q, cnt)
    rp = np.repeat(pid, cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    rt = L.indices[np.repeat(L.indptr[s], cnt) + off]
    nT = len(onto.terms)
    key = rq.astype(np.int64) * nT + rt
    o = np.argsort(key, kind="stable")
    key, rp = key[o], rp[o]
    start = np.r_[0, np.where(key[1:] != key[:-1])[0] + 1]
    uk = key[start]
    s_best = np.maximum.reduceat(rp, start)
    n_t = np.diff(np.r_[start, len(key)])
    qq, tt = uk // nT, uk % nT
    nhits = np.bincount(q, minlength=len(kn.ids))
    roots = np.array(kn.roots)
    ann = np.stack([np.bincount(q, weights=np.asarray(L[:, r].todense()).ravel()[s], minlength=len(kn.ids))
                    for r in roots], 1)
    asp_n = ann[qq, onto.ns[tt]]

    def lookup(P):
        P = P.tocoo()
        k2 = P.row.astype(np.int64) * nT + P.col
        o2 = np.argsort(k2)
        k2, v2 = k2[o2], P.data[o2]
        i = np.searchsorted(k2, uk)
        i = np.minimum(i, len(k2) - 1)
        return np.where(k2[i] == uk, v2[i], 0.0)

    # the contract prior is stored over the out-of-vocabulary-inclusive term list terms_full
    cols = np.array([onto.idx[t] for t in prior["terms_full"]])
    pf = prior["p_full"].tocoo()
    pos = {p: i for i, p in enumerate(kn.ids)}
    rmap = np.array([pos[p] for p in prior["ids"]])
    Pfull = sp.csr_matrix((pf.data, (rmap[pf.row], cols[pf.col])), shape=(len(kn.ids), nT))
    mx = kn.maxpid[qq]
    return pd.DataFrame({
        "q": qq, "t": tt, "ns": onto.ns[tt],
        "p_knn": lookup(Pfull), "p_sharp": lookup(sharp),
        "s_best": s_best / 100, "gap": (s_best - mx) / 100,
        "frac_asp": n_t / np.maximum(asp_n, 1), "frac_all": n_t / np.maximum(nhits[qq], 1),
        "log_n": np.log1p(n_t), "max_id": mx / 100, "has_hit": (nhits[qq] > 0).astype(float),
        "log_hits": np.log1p(nhits[qq]), "ia": onto.ia[tt], "depth": onto.depth[tt]})


def labels_for(F: pd.DataFrame, kn, sc: Scorer, onto: Ontology) -> tuple[np.ndarray, np.ndarray]:
    """y (term in the propagated ground truth) and a mask of rows whose aspect is scored for that protein."""
    nT = len(onto.terms)
    gi = np.array([sc.pos.get(p, -1) for p in kn.ids])
    g = gi[F.q.values]
    ok = g >= 0
    gk = np.where(ok, g.astype(np.int64) * nT + F.t.values, -1)
    y = np.isin(gk, sc.gkey) & ok
    scored = ok.copy()
    scored[ok] = sc.in_ns[F.ns.values[ok], g[ok]]
    return y.astype(np.int8), scored


def make_model(kind: str, threads: int):
    if kind.startswith("gbm"):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    return LogisticRegression(C=1.0, max_iter=2000)


def design(F: pd.DataFrame, cols: list, kind: str) -> np.ndarray:
    X = F[cols].values.astype(np.float64)
    if kind.startswith("logreg"):
        X = np.hstack([X, X ** 2, X[:, [0]] * X[:, [cols.index("max_id")]], np.log(X[:, [0]] + 1e-3)])
    return X


class AspectModel:
    def __init__(self, cols, kind, threads):
        self.cols, self.kind, self.threads, self.m = cols, kind, threads, {}

    def fit(self, F, y):
        for a in range(3):
            r = (F.ns.values == a)
            X = design(F[r], self.cols, self.kind)
            sc = StandardScaler().fit(X) if self.kind.startswith("logreg") else None
            # "_iaw" variants weight each candidate by its term's IA, as the metric does
            w = 0.1 + F.ia.values[r] if self.kind.endswith("_iaw") else None
            mdl = make_model(self.kind, self.threads).fit(sc.transform(X) if sc else X, y[r], sample_weight=w)
            self.m[a] = (sc, mdl)
        return self

    def predict(self, F):
        out = np.zeros(len(F))
        for a in range(3):
            r = np.where(F.ns.values == a)[0]
            if len(r):
                sc, mdl = self.m[a]
                X = design(F.iloc[r], self.cols, self.kind)
                out[r] = mdl.predict_proba(sc.transform(X) if sc else X)[:, 1]
        return out


def to_csr(F: pd.DataFrame, p: np.ndarray, n: int, onto: Ontology) -> sp.csr_matrix:
    P = sp.csr_matrix((p.astype(np.float32), (F.q.values, F.t.values)), shape=(n, len(onto.terms)))
    return onto.propagate_max(P)


def bin_row(sc: Scorer, cur, ident: pd.Series) -> dict:
    b = ident_bins(ident.reindex(sc.ids).fillna(0).values)
    b.index = sc.ids
    row = {"all": sc.fmax(cur)["mean"], **{a: sc.fmax(cur)[a] for a in ("MF", "BP", "CC")}}
    for name in BINS:
        ids = b.index[b == name]
        row[name] = sc.fmax(cur, subset=ids)["mean"] if len(ids) else np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tv", default="v1", choices=("v1", "v2"))
    a = ap.parse_args()
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    ks.use_tv(a.tv)
    run = RUN if a.tv == "v1" else ROOT / "runs" / "stacker_l1_v2"
    run.mkdir(parents=True, exist_ok=True)
    log(f"== L1 stacker (TV {a.tv})")
    onto, go = Ontology(), GO()
    knn = build_knn(onto, go, ("val", "tv", "test"))
    ident = pd.read_parquet(ks.TV_DIR / "identity.parquet")
    l0 = json.loads((ROOT / "runs" / "knn_sharp" / ("chosen.json" if a.tv == "v1" else "chosen_v2.json")).read_text())
    feats, scs = {}, {}
    for s in ("tv", "val", "test"):
        kn = knn[s]
        sharp, _ = kn.prior(2.0, 10, 0, "aspect")
        t = time.time()
        feats[s] = features(onto, kn, load_knn(s), sharp)
        log(f"{s}: {len(feats[s]):,} candidate rows for {len(kn.ids):,} queries ({time.time() - t:.0f}s)")
        if s != "test":
            scs[s] = Scorer(onto, gt_frame(s))

    F = feats["tv"]
    kn = knn["tv"]
    y, scored = labels_for(F, kn, scs["tv"], onto)
    log(f"TV training rows (scored aspects): {scored.sum():,}, positives {y[scored].sum():,} "
        f"({y[scored].mean():.3f}); GT coverage of candidates is the kNN recall ceiling")
    rng = np.random.default_rng(0)
    fold_of = rng.permutation(len(kn.ids)) % a.folds
    fq = fold_of[F.q.values]
    idm = {s: ident[ident.split == s].drop_duplicates("id").set_index("id").max_pident for s in ("tv", "val")}
    rows, bins, oof = [], [], {}
    # L0 reference on the same TV proteins
    P0 = load_knn("tv")
    cols = np.array([onto.idx[t] for t in P0["terms_full"]])
    pf = P0["p_full"].tocoo()
    Pl0 = sp.csr_matrix((pf.data, (pf.row, cols[pf.col])), shape=(len(kn.ids), len(onto.terms)))
    cur0 = scs["tv"].curves(Pl0, P0["ids"], closed=True)
    bins.append({"split": "tv", "method": f"L0 {l0['setting']}", **bin_row(scs["tv"], cur0, idm["tv"])})
    wtl = None
    if a.tv != "v1":
        wtl, _ = ks.cell_weights(pd.read_parquet(ks.TV_DIR / "cells.parquet"), scs["tv"].ids)
        bins[0]["testlike"] = scs["tv"].fmax(cur0, weights=wtl)["mean"]
    for name, (fc, kind) in VARIANTS.items():
        t = time.time()
        p = np.zeros(len(F))
        for f in range(a.folds):
            tr = (fq != f) & scored
            te = fq == f
            mdl = AspectModel(fc, kind, a.threads).fit(F[tr], y[tr])
            p[te] = mdl.predict(F[te])
        cur = scs["tv"].curves(to_csr(F, p, len(kn.ids), onto), kn.ids, closed=True)
        oof[name] = cur
        r = bin_row(scs["tv"], cur, idm["tv"])
        bins.append({"split": "tv", "method": f"L1 {name} (5-fold CV)", **r})
        bs = scs["tv"].bootstrap(cur, cur0, reps=1000)
        rows.append({"variant": name, "tv_cv_mean": r["all"], "MF": r["MF"], "BP": r["BP"], "CC": r["CC"],
                     "delta_vs_L0": bs["diff"], "ci_lo": bs["lo"], "ci_hi": bs["hi"]})
        if wtl is not None:
            bt = scs["tv"].bootstrap(cur, cur0, reps=500, weights=wtl)
            bins[-1]["testlike"] = scs["tv"].fmax(cur, weights=wtl)["mean"]
            rows[-1].update({"delta_testlike": bt["diff"], "tl_lo": bt["lo"], "tl_hi": bt["hi"]})
        log(f"L1 {name}: TV CV {r['all']:.4f} (L0 {bins[0]['all']:.4f}; diff {bs['diff']:+.4f} "
            f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}]) in {time.time() - t:.0f}s")
    summ = pd.DataFrame(rows).sort_values("tv_cv_mean", ascending=False)
    best = summ.iloc[0]
    keep = bool(best.delta_vs_L0 >= 0.005)
    remote_ok = {b: float(bins[[x["method"] for x in bins].index(f"L1 {best.variant} (5-fold CV)")][b] - bins[0][b])
                 for b in ("30-50", "<30")}
    chosen = {"variant": best.variant, "features": VARIANTS[best.variant][0], "model": VARIANTS[best.variant][1],
              "tv_cv_mean": float(best.tv_cv_mean), "l0_tv_mean": float(bins[0]["all"]),
              "delta_vs_L0": float(best.delta_vs_L0), "ci": [float(best.ci_lo), float(best.ci_hi)],
              "keep_rule": "keep if >= +0.005 over L0 on TV and remote bins not below the gate - 0.005 "
                           "(HiGO+gate has no fair TV prediction, so remote bins are compared to L0)",
              "keep": keep, "remote_bin_delta_vs_L0": remote_ok, "higo_feature": "not used (HiGO saw TV labels)",
              "l0_setting": l0["setting"], "tv_version": a.tv}
    if wtl is not None:
        chosen["delta_testlike"] = [float(best.delta_testlike), float(best.tl_lo), float(best.tl_hi)]

    # refit on all TV, apply to val (scored) and test (written, not scored)
    fc, kind = VARIANTS[best.variant]
    final = AspectModel(fc, kind, a.threads).fit(F[scored], y[scored])
    for s in ("val", "test"):
        Fs = feats[s]
        P = to_csr(Fs, final.predict(Fs), len(knn[s].ids), onto)
        Pk = P.copy()
        Pk.data[Pk.data < 0.01] = 0
        Pk.eliminate_zeros()
        used = np.unique(Pk.indices)
        Pu = Pk[:, used].tocsr()
        np.savez_compressed(run / f"pred_{s}.npz", ids=knn[s].ids.astype(str), data=Pu.data, indices=Pu.indices,
                            indptr=Pu.indptr, shape=np.array(Pu.shape), terms=np.array(onto.terms)[used].astype(str))
        if s == "val":
            cur = scs["val"].curves(P, knn["val"].ids, closed=True)
            bins.append({"split": "val", "method": f"L1 {best.variant} (fit on TV)", **bin_row(scs["val"], cur, idm["val"])})
            P0v = load_knn("val")
            cv = np.array([onto.idx[t] for t in P0v["terms_full"]])
            pv = P0v["p_full"].tocoo()
            Pv = sp.csr_matrix((pv.data, (pv.row, cv[pv.col])), shape=(len(P0v["ids"]), len(onto.terms)))
            bins.append({"split": "val", "method": f"L0 {l0['setting']}",
                         **bin_row(scs["val"], scs["val"].curves(Pv, P0v["ids"], closed=True), idm["val"])})
    summ.to_csv(run / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(bins).to_csv(run / "per_bin.tsv", sep="\t", index=False)
    (run / "chosen.json").write_text(json.dumps(chosen, indent=2))
    log("L1 summary:\n" + summ.round(4).to_string() + "\n" + pd.DataFrame(bins).round(4).to_string()
        + "\n" + json.dumps(chosen, indent=1))
    if a.tv == "v1":
        (ROOT / "logs" / "stacker_l1.done").touch()


if __name__ == "__main__":
    main()
