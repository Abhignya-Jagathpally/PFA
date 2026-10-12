"""L0: identity-sharpened kNN label transfer, plus an in-memory CAFA scorer, for the GO-GPT benchmark.

The kNN prior
-------------
For a query q with DIAMOND hits j (top k by bitscore, self excluded), each hit carries a bitscore b_j,
percent identity s_j, query coverage c_j (qcovhsp / 100) and the hit's full propagated label set y_j
(not restricted to the vocabulary). The weights and the prior are

    w_j = b_j * exp((s_j - max_i s_i) / T) * c_j ** gamma,      p_t = min(1, sum_j w_j y_jt / sum_j w_j).

T = inf gives the plain bitscore-weighted mean that `homology.diamond_score` computes. Because labels are
ancestor-closed, p is ancestor-closed too. Queries without a hit get no prediction (the method abstains).
An extra knob, norm="aspect", divides each aspect's scores only by the weight of the hits that have any
annotation in that aspect, so an unannotated close homolog no longer dilutes the prior to zero.

Label databases: val and test queries copy the full GO-GPT training labels; TV queries copy the T0 labels
built by `cafa6.temporal_val` (post-T0 gains stripped from every training protein), so a TV query cannot
copy an ortholog's annotation from the same 2022 paper.

The scorer
----------
`Scorer` reproduces cafaeval 1.3 as wired in `evaluate.evaluate` (prop=max, norm=cafa, thresholds
0.01..0.99, IA-weighted) in memory: scores are cut at 0.01, limited to the top 1,500 per protein and
rounded to 3 decimals as `dense_to_long` does, then max-propagated within each namespace. Precision is
averaged over proteins with at least one positive-IA prediction, recall over all proteins with ground
truth in the namespace. It keeps per-protein precision/recall curves so identity bins, protein weights
and paired bootstraps cost no rescoring. `python -m cafa6.knn_sharp --verify` checks it against
runs/bench_gogpt/summary.tsv.

Contract file: data/proc_gogpt_tv/knn/{split}_knn.npz, split in {val, tv, test}
------------------------------------------------------------------------------
Written with numpy.savez_compressed; read with `load_knn(split)`. Keys:
  ids          str[n]   query accessions; order = train.parquet rows of that split (val, test) or
                        tv_ids.txt (tv).
  p_data, p_indices, p_indptr, p_shape   CSR float32 [n, 7901]: p_knn over data/proc_gogpt/vocab.tsv
                        columns in file order (raw probabilities, no clipping or logit).
  pf_data, pf_indices, pf_indptr, pf_shape, terms_full   the same prior over every GO term reached by the
                        neighbours (out-of-vocabulary terms included); column j is terms_full[j].
  max_pident   float32[n]  best percent identity over all hits (0 when no hit).
  has_hit      bool[n]     at least one DIAMOND hit at e <= 1e-3.
  n_hits       int32[n]    number of hits used (after top-k).
  settings     str      JSON with T, k, gamma, label database and hit file.
`load_knn` returns a dict with `p` and `p_full` as scipy CSR matrices plus the arrays above.
knn/ holds the setting pre-registered on TV (runs/knn_sharp/chosen.json); `--write SETTING DIR` writes the
same files for another grid setting without scoring (data/proc_gogpt_tv/knn_T2_k10 = T=2, k=10, gamma 0).

Usage:
  python -m cafa6.knn_sharp --verify          scorer check against cafaeval numbers (no new test look)
  python -m cafa6.knn_sharp --grid            grid on val and TV, pre-registers the choice
  python -m cafa6.knn_sharp --final           one test look for the pre-registered choice, writes priors
  python -m cafa6.knn_sharp --write T2_k10_g0 data/proc_gogpt_tv/knn_T2_k10
  python -m cafa6.knn_sharp --grid-v2         grid on TV v2 with the test-like cell weights (V2_RULE),
                                              pre-registers runs/knn_sharp/chosen_v2.json; no test scoring
  python -m cafa6.knn_sharp --tv v2 --write T5_k50_g0_na data/proc_gogpt_tv2/knn
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import ROOT
from cafa6.temporal_val import GO, OUT as TV_DIR, OUT_V2, REF, SRC, full_labels, ident_bins, log, t0_labels

OBO, IA_FILE = REF / "go-basic_gogpt.obo", REF / "IA.txt"
KNN_DIR = TV_DIR / "knn"
TV_VERSION = "v1"


def use_tv(version: str):
    """Points every TV path at data/proc_gogpt_tv (v1) or data/proc_gogpt_tv2 (v2). v2 TV queries search the
    v2 database (split == "train" there); v1 keeps its original database (all GO-GPT training proteins)."""
    global TV_DIR, KNN_DIR, TV_VERSION
    TV_VERSION = version
    TV_DIR = {"v1": ROOT / "data" / "proc_gogpt_tv", "v2": OUT_V2}[version]
    KNN_DIR = TV_DIR / "knn"
RUN = ROOT / "runs" / "knn_sharp"
TEST_LOOKS = ROOT / "runs" / "test_looks.md"
NS = ("molecular_function", "biological_process", "cellular_component")
NS_SHORT = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
TAU = np.arange(0.01, 1, 0.01)
BINS = ["100", "95-99.9", "80-95", "50-80", "30-50", "<30", "no_hit"]
GRID = [(T, k, g, n) for T in (1.0, 2.0, 5.0, 10.0, np.inf) for k in (5, 10, 50) for g in (0, 1)
        for n in ("all", "aspect")]
# red-team test numbers (sharpened kNN, gamma 0, research_notes/.../red_team.md section 3.2)
RED_TEAM_TEST = {(1.0, 10): 0.698, (1.0, 50): 0.701, (2.0, 10): 0.698, (2.0, 50): 0.700,
                 (5.0, 10): 0.699, (5.0, 50): 0.701, (10.0, 10): 0.689, (10.0, 50): 0.691}


class Ontology:
    """The GO graph exactly as cafaeval's obo_parser builds it: one graph per namespace, is_a and part_of
    edges kept only inside a namespace, obsolete terms dropped, alt ids mapped to canonical ids."""

    def __init__(self, obo=OBO, ia_file=IA_FILE):
        terms, cur = {}, None
        for line in open(obo):
            kv = line.strip().split(": ")
            if len(kv) < 2:
                continue
            k, v = kv[0], ": ".join(kv[1:])
            if k == "id":
                if cur and not cur["obsolete"] and cur["ns"]:
                    terms[cur["id"]] = cur
                cur = {"id": v, "ns": None, "alt": [], "rel": [], "obsolete": False}
            elif cur is None:
                continue
            elif k == "alt_id":
                cur["alt"].append(v)
            elif k == "namespace" and v != "external":
                cur["ns"] = v
            elif k == "is_obsolete":
                cur["obsolete"] = True
            elif k == "is_a":
                cur["rel"].append(v.split("!")[0].strip())
            elif k == "relationship" and v.startswith("part_of"):
                cur["rel"].append(v.split()[1].strip())
        if cur and not cur["obsolete"] and cur["ns"]:
            terms[cur["id"]] = cur
        self.terms = [t for ns in NS for t in terms if terms[t]["ns"] == ns]
        self.idx = {t: i for i, t in enumerate(self.terms)}
        self.ns = np.array([NS.index(terms[t]["ns"]) for t in self.terms], np.int8)
        self.alt = defaultdict(list)
        for t in self.terms:
            for a in terms[t]["alt"]:
                self.alt[a].append(t)
        parents = {t: [p for p in terms[t]["rel"] if p in terms and terms[p]["ns"] == terms[t]["ns"]]
                   for t in self.terms}
        anc, depth = {}, {}

        def visit(t):
            if t in anc:
                return anc[t]
            s = set()
            d = 0
            for p in parents[t]:
                s.add(p)
                s |= visit(p)
                d = max(d, depth[p] + 1)
            anc[t], depth[t] = s, d
            return s

        import sys
        sys.setrecursionlimit(100000)
        for t in self.terms:
            visit(t)
        r, c = [], []
        for t in self.terms:
            i = self.idx[t]
            r.append(i); c.append(i)
            for a in anc[t]:
                r.append(i); c.append(self.idx[a])
        n = len(self.terms)
        # A[t, a] = 1 when a is t or an ancestor of t inside the namespace
        self.A = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
        self.depth = np.array([depth[t] for t in self.terms], np.float32)
        ia = pd.read_csv(ia_file, sep=r"\s+", header=None, names=["term", "ia"])
        ia = dict(zip(ia.term, ia.ia))
        self.ia = np.nan_to_num(np.array([ia.get(t, 0.0) or 0.0 for t in self.terms], np.float64),
                                nan=0, posinf=0, neginf=0)

    def canon(self, t) -> list[int]:
        if t in self.idx:
            return [self.idx[t]]
        return [self.idx[a] for a in self.alt.get(t, ())]

    def label_matrix(self, sets: list, close: bool = True) -> sp.csr_matrix:
        """Binary CSR [len(sets), n_terms] from term sets, closed within each namespace."""
        r, c = [], []
        for i, s in enumerate(sets):
            for t in s:
                for j in self.canon(t):
                    r.append(i); c.append(j)
        M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(sets), len(self.terms)))
        M.sum_duplicates()
        M.data[:] = 1
        if close:
            M = (M @ self.A).tocsr()
            M.data[:] = 1
        return M

    def propagate_max(self, P: sp.csr_matrix) -> sp.csr_matrix:
        """Max-constraint: every ancestor gets the max score of its descendants (cafaeval prop=max)."""
        P = P.tocsr()
        P.eliminate_zeros()
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        cnt = np.diff(self.A.indptr)[P.indices]
        rr = np.repeat(rows, cnt)
        vv = np.repeat(P.data, cnt)
        starts = self.A.indptr[P.indices]
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cc = self.A.indices[np.repeat(starts, cnt) + off]
        key = rr.astype(np.int64) * P.shape[1] + cc
        o = np.lexsort((-vv, key))
        key, vv = key[o], vv[o]
        first = np.r_[True, key[1:] != key[:-1]]
        key, vv = key[first], vv[first]
        return sp.csr_matrix((vv, (key // P.shape[1], key % P.shape[1])), shape=P.shape)


def to_submission(P: sp.csr_matrix, top_k: int = 1500, min_score: float = 0.01) -> sp.csr_matrix:
    """What dense_to_long + the TSV round trip do to scores: cut, top-k per protein, round to 3 decimals."""
    P = P.tocsr().astype(np.float64)
    keep = P.data >= min_score
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    rows, cols, vals = rows[keep], P.indices[keep], P.data[keep]
    if len(vals) and np.bincount(rows, minlength=P.shape[0]).max() > top_k:
        o = np.lexsort((-vals, rows))
        rows, cols, vals = rows[o], cols[o], vals[o]
        start = np.searchsorted(rows, rows, side="left")
        m = (np.arange(len(rows)) - start) < top_k
        rows, cols, vals = rows[m], cols[m], vals[m]
    return sp.csr_matrix((np.round(vals, 3), (rows, cols)), shape=P.shape)


class Scorer:
    """In-memory cafaeval (prop=max, norm=cafa, IA-weighted Fmax) with per-protein curves."""

    def __init__(self, onto: Ontology, gt: pd.DataFrame):
        self.o = onto
        self.ids = np.array(sorted(gt.id.unique()))
        self.pos = {p: i for i, p in enumerate(self.ids)}
        sets = gt.groupby("id").term.agg(set).reindex(self.ids)
        self.G = onto.label_matrix(list(sets), close=True)
        self.n = len(self.ids)
        tns = onto.ns[self.G.indices]
        rows = np.repeat(np.arange(self.n), np.diff(self.G.indptr))
        self.in_ns = np.zeros((3, self.n), bool)
        self.gtw = np.zeros((3, self.n))
        for k in range(3):
            m = tns == k
            self.in_ns[k, np.unique(rows[m])] = True
            self.gtw[k] = np.bincount(rows[m], weights=onto.ia[self.G.indices[m]], minlength=self.n)
        self.gkey = np.sort(rows.astype(np.int64) * len(onto.terms) + self.G.indices)

    def curves(self, P: sp.csr_matrix, ids, closed: bool = False) -> dict:
        """P: [len(ids), n_terms] scores in Ontology column order. Returns per-namespace arrays over the
        ground-truth proteins: prec, has (has a positive-IA prediction at tau) and rec, each [n, 99]."""
        ids = np.asarray(ids)
        m = np.array([p in self.pos for p in ids])
        P = P.tocsr()[np.where(m)[0]]
        gi = np.array([self.pos[p] for p in ids[m]], dtype=np.int64)
        # duplicate ids (none expected) would be max-combined by cafaeval; we keep the first
        P = to_submission(P)
        if not closed:
            P = self.o.propagate_max(P)
        rows = gi[np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))]
        cols, vals = P.indices, P.data
        w = self.o.ia[cols]
        nsc = self.o.ns[cols]
        key = rows * len(self.o.terms) + cols
        ing = np.isin(key, self.gkey, assume_unique=False)
        kk = np.searchsorted(TAU, vals, side="right")  # thresholds passed: tau[0..kk-1]
        out = {}
        for k in range(3):
            sel = (nsc == k) & (w > 0) & self.in_ns[k, rows]
            r, kq, ww, g = rows[sel], kk[sel], w[sel], ing[sel]
            C = np.bincount(r * 100 + kq, weights=ww, minlength=self.n * 100).reshape(self.n, 100)
            I = np.bincount(r * 100 + kq, weights=ww * g, minlength=self.n * 100).reshape(self.n, 100)
            predw = np.cumsum(C[:, ::-1], axis=1)[:, ::-1][:, 1:]
            interw = np.cumsum(I[:, ::-1], axis=1)[:, ::-1][:, 1:]
            has = predw > 0
            prec = np.divide(interw, predw, out=np.zeros_like(predw), where=has)
            gw = self.gtw[k][:, None]
            rec = np.divide(interw, gw, out=np.zeros_like(interw), where=gw > 0)
            idx = np.where(self.in_ns[k])[0]
            out[NS_SHORT[NS[k]]] = {"idx": idx, "prec": prec[idx].astype(np.float32),
                                    "has": has[idx], "rec": rec[idx].astype(np.float32)}
        return out

    def fmax(self, cur: dict, subset=None, weights=None) -> dict:
        """Fmax per namespace and their mean. subset: protein ids; weights: per-protein weights (dict or
        array over self.ids)."""
        wv = np.ones(self.n)
        if weights is not None:
            wv = np.asarray(weights, float)
        if subset is not None:
            mask = np.zeros(self.n)
            mask[[self.pos[p] for p in subset if p in self.pos]] = 1
            wv = wv * mask
        res = {}
        for a, c in cur.items():
            ww = wv[c["idx"]]
            ne = ww.sum()
            if ne <= 0:
                res[a] = np.nan
                continue
            nh = ww @ c["has"]
            P = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
            R = (ww @ c["rec"]) / ne
            F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
            F[nh <= 0] = 0
            res[a] = float(F.max())
        vals = [res[a] for a in ("MF", "BP", "CC") if not np.isnan(res.get(a, np.nan))]
        res["mean"] = float(np.mean(vals)) if vals else np.nan
        return res

    def bootstrap(self, curA: dict, curB: dict, reps: int = 1000, seed: int = 0, weights=None) -> dict:
        """Paired protein bootstrap of mean Fmax(A) - Fmax(B) over the union of namespaces."""
        rng = np.random.default_rng(seed)
        base = np.ones(self.n) if weights is None else np.asarray(weights, float)
        d = []
        for _ in range(reps):
            c = rng.multinomial(int(round(base.sum())), base / base.sum()).astype(float)
            d.append(self.fmax(curA, weights=c)["mean"] - self.fmax(curB, weights=c)["mean"])
        d = np.array(d)
        return {"diff": float(self.fmax(curA, weights=weights)["mean"] - self.fmax(curB, weights=weights)["mean"]),
                "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "reps": reps}


def load_hits(split: str) -> pd.DataFrame:
    if split == "tv":
        return pd.read_parquet(TV_DIR / "diamond_hits_tv.parquet")
    return pd.read_parquet(SRC / "diamond_hits_valtest.parquet")


def split_ids(split: str) -> np.ndarray:
    if split == "tv":
        return np.array(open(TV_DIR / "tv_ids.txt").read().split())
    df = pd.read_parquet(SRC / "train.parquet", columns=["id", "split"])
    return np.array(df.id[df.split == split].tolist(), dtype=str)


def gt_frame(split: str) -> pd.DataFrame:
    f = TV_DIR / "gt_tv.tsv" if split == "tv" else SRC / f"gt_{split}.tsv"
    return pd.read_csv(f, sep="\t", header=None, names=["id", "term"])


class KNN:
    """Hit table + label database for one query split, ready to evaluate any (T, k, gamma)."""

    def __init__(self, onto: Ontology, split: str, db_ids: np.ndarray, L_db: sp.csr_matrix):
        self.ids = split_ids(split)
        h = load_hits(split)
        h = h[h.q.isin(set(self.ids))]
        didx = {p: i for i, p in enumerate(db_ids)}
        h = h[h.s.isin(didx)].sort_values(["q", "bits"], ascending=[True, False])
        h = h.drop_duplicates(["q", "s"])
        qidx = {p: i for i, p in enumerate(self.ids)}
        self.q = h.q.map(qidx).values.astype(np.int64)
        self.s = h.s.map(didx).values.astype(np.int64)
        self.rank = h.groupby("q", sort=False).cumcount().values
        self.bits = h.bits.values
        self.pid = h.pident.values
        self.cov = np.clip(h.qcov.values / 100.0, 0, 1)
        self.maxpid = np.zeros(len(self.ids), np.float32)
        np.maximum.at(self.maxpid, self.q, self.pid.astype(np.float32))
        self.has_hit = np.bincount(self.q, minlength=len(self.ids)) > 0
        self.L = L_db
        self.roots = [onto.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
        self.term_ns = onto.ns
        self.split = split

    def prior(self, T: float, k: int, gamma: float, norm: str = "all") -> tuple[sp.csr_matrix, np.ndarray]:
        """norm="all" divides by the weight of every hit (the formula above); norm="aspect" divides, per
        aspect, only by the weight of hits that have any annotation in that aspect."""
        m = self.rank < k
        q, s, b, pid, cov = self.q[m], self.s[m], self.bits[m], self.pid[m], self.cov[m]
        mx = np.zeros(len(self.ids))
        np.maximum.at(mx, q, pid)
        w = b * (np.exp((pid - mx[q]) / T) if np.isfinite(T) else 1.0)
        if gamma:
            w = w * np.power(np.maximum(cov, 1e-3), gamma)
        W = sp.csr_matrix((w, (q, s)), shape=(len(self.ids), self.L.shape[0]))
        P = (W @ self.L).tocsr()
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        if norm == "aspect":
            D = (W @ self.L[:, self.roots]).toarray()
            den = D[rows, self.term_ns[P.indices]]
        else:
            den = np.asarray(W.sum(1)).ravel()[rows]
        P.data = np.minimum(P.data / np.maximum(den, 1e-12), 1.0).astype(np.float32)
        return P, np.bincount(q, minlength=len(self.ids))


def build_knn(onto: Ontology, go: GO, splits=("val", "tv", "test")) -> dict:
    df = pd.read_parquet(SRC / "train.parquet", columns=["id", "split"])
    db_ids = df.id.values[(df.split == "train").values]
    full = full_labels(go, "train")
    L_full = onto.label_matrix([full.get(p, set()) for p in db_ids])
    out = {}
    for s in splits:
        if s == "tv":
            t0 = t0_labels(go, full, out=TV_DIR)
            ids = db_ids
            if TV_VERSION != "v1":
                tdf = pd.read_parquet(TV_DIR / "train.parquet", columns=["id", "split"])
                ids = tdf.id.values[(tdf.split == "train").values]
            out[s] = KNN(onto, s, ids, onto.label_matrix([t0.get(p, set()) for p in ids]))
        else:
            out[s] = KNN(onto, s, db_ids, L_full)
    return out


# ---- grid (multiprocessing with fork; globals are inherited by workers) ----
_G: dict = {}


def _eval_setting(args):
    split, st = args
    kn, sc = _G["knn"][split], _G["scorer"][split]
    P, _ = kn.prior(*st)
    cur = sc.curves(P, kn.ids, closed=True)
    return split, st, cur


def bin_table(sc: Scorer, cur: dict, ident: pd.Series, weights=None) -> dict:
    b = ident_bins(ident.reindex(sc.ids).fillna(0).values)
    b.index = sc.ids
    row = {"all": sc.fmax(cur, weights=weights)["mean"]}
    for name in BINS:
        ids = b.index[b == name]
        row[name] = sc.fmax(cur, subset=ids)["mean"] if len(ids) else np.nan
    return row


def test_like_weights(sc: Scorer, ident: pd.Series, ident_test: pd.Series) -> np.ndarray:
    """Per-protein weights that give the TV identity-bin mix of the test set."""
    b = ident_bins(ident.reindex(sc.ids).fillna(0).values)
    bt = ident_bins(ident_test.values).value_counts(normalize=True)
    bv = b.value_counts(normalize=True)
    return b.map(lambda x: bt.get(x, 0.0) / bv[x]).values.astype(float)


def setting_name(T, k, g, n="all"):
    return f"T{'inf' if not np.isfinite(T) else int(T)}_k{k}_g{g}" + ("_na" if n == "aspect" else "")


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def run_grid(threads: int):
    RUN.mkdir(parents=True, exist_ok=True)
    log("== L0 grid on val and TV")
    onto, go = Ontology(), GO()
    knn = build_knn(onto, go, ("val", "tv"))
    _G["knn"] = knn
    _G["scorer"] = {s: Scorer(onto, gt_frame(s)) for s in ("val", "tv")}
    ident = pd.read_parquet(TV_DIR / "identity.parquet")
    idm = {s: ident[ident.split == s].drop_duplicates("id").set_index("id").max_pident for s in ("val", "tv", "test")}
    wtv = test_like_weights(_G["scorer"]["tv"], idm["tv"], idm["test"])
    tvdf = pd.read_parquet(TV_DIR / "train.parquet", columns=["id", "split"])
    nk = set(tvdf.id[tvdf.split == "tv"])
    jobs = [(s, g) for s in ("val", "tv") for g in GRID]
    t = time.time()
    with mp.get_context("fork").Pool(min(threads, len(jobs))) as pool:
        res = pool.map(_eval_setting, jobs)
    log(f"grid scored in {time.time() - t:.0f}s")
    rows, bins = [], []
    for split, st, cur in res:
        sc = _G["scorer"][split]
        f = sc.fmax(cur)
        r = {"split": split, "setting": setting_name(*st), "T": st[0], "k": st[1], "gamma": st[2], "norm": st[3],
             **{a: f[a] for a in ("MF", "BP", "CC")}, "mean": f["mean"]}
        if split == "tv":
            r["mean_testlike"] = sc.fmax(cur, weights=wtv)["mean"]
            r["mean_NK"] = sc.fmax(cur, subset=nk)["mean"]
            r["mean_LK"] = sc.fmax(cur, subset=[p for p in sc.ids if p not in nk])["mean"]
        rows.append(r)
        bins.append({"split": split, "setting": setting_name(*st), **bin_table(sc, cur, idm[split])})
    summ = pd.DataFrame(rows).sort_values(["split", "mean"], ascending=[True, False])
    summ.to_csv(RUN / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(bins).to_csv(RUN / "per_bin.tsv", sep="\t", index=False)

    # ranking agreement with the red-team test grid (gamma 0, k in {10, 50}, T in {1, 2, 5, 10})
    rt = {setting_name(T, k, 0): v for (T, k), v in RED_TEAM_TEST.items()}
    piv = summ.pivot(index="setting", columns="split", values="mean")
    tl = summ[summ.split == "tv"].set_index("setting").mean_testlike
    common = list(rt)
    agree = {"red_team_test": rt,
             "spearman_val_vs_test": spearman(piv.loc[common, "val"], [rt[c] for c in common]),
             "spearman_tv_vs_test": spearman(piv.loc[common, "tv"], [rt[c] for c in common]),
             "spearman_tvtestlike_vs_test": spearman(tl.loc[common], [rt[c] for c in common])}
    # bootstrap CI on TV for the spread of the overlap settings (best vs worst on TV)
    cur_tv = {setting_name(*st): cur for split, st, cur in res if split == "tv"}
    sc_tv = _G["scorer"]["tv"]
    tv_c = piv.loc[common, "tv"].sort_values()
    agree["tv_ci_best_vs_worst_overlap"] = sc_tv.bootstrap(cur_tv[tv_c.index[-1]], cur_tv[tv_c.index[0]], reps=1000)
    agree["tv_ci_best_vs_worst_overlap"]["pair"] = [tv_c.index[-1], tv_c.index[0]]
    log("ranking agreement: " + json.dumps(agree, indent=1))

    # pre-registered choice: best on TV; within 0.003 of the best, the fewest active knobs, then best TV.
    # Plan rule: when TV ranks the red-team settings against the test beyond its bootstrap CI, select on
    # the most test-like TV instead (here: TV reweighted to the test identity-bin mix).
    disagree = agree["spearman_tv_vs_test"] < 0 and agree["tv_ci_best_vs_worst_overlap"]["lo"] > 0
    crit = "mean_testlike" if disagree else "mean"
    tv = summ[summ.split == "tv"].copy()
    tv["knobs"] = np.isfinite(tv["T"]).astype(int) + (tv["gamma"] != 0).astype(int) + (tv["norm"] != "all").astype(int)
    best = tv[crit].max()
    cand = tv[tv[crit] >= best - 0.003].sort_values(["knobs", crit], ascending=[True, False])
    ch = cand.iloc[0]
    chosen = {"T": float(ch["T"]), "k": int(ch["k"]), "gamma": int(ch["gamma"]), "norm": ch["norm"],
              "setting": ch["setting"],
              "criterion": crit, "tv_mean": float(ch["mean"]), "tv_mean_testlike": float(ch["mean_testlike"]),
              "best_tv_setting": tv.sort_values(crit, ascending=False).iloc[0]["setting"], "best_tv": float(best),
              "rule": "max TV Fmax (criterion column: TV, or TV reweighted to the test identity mix when TV "
                      "ranks the red-team test grid the other way beyond its CI); among settings within 0.003 of "
                      "the best, fewest active knobs (finite T, gamma>0, aspect norm), then highest criterion",
              "ranking_agreement": agree, "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RUN / "chosen.json").write_text(json.dumps(chosen, indent=2, default=float))
    log("pre-registered L0 choice: " + json.dumps({k: chosen[k] for k in ("setting", "criterion", "tv_mean",
                                                                          "tv_mean_testlike", "best_tv_setting")}))
    with open(TEST_LOOKS, "a") as f:
        f.write(f"\n- {chosen['registered_at']} | L0 sharpened kNN | PRE-REGISTERED (no test yet): "
                f"{chosen['setting']} chosen on TV ({crit}) by rule in runs/knn_sharp/chosen.json\n")


MIN_CELL = 15
V2_RULE = ("criterion = TV v2 mean Fmax with every TV v2 protein weighted so that the cells (organism group "
           "zebrafish/rat/human/mouse/other x identity bin x best hit in the same organism) carry their test "
           f"shares; a cell with < {MIN_CELL} TV v2 proteins is merged over the same-organism flag, then over "
           "organism groups within its bin. Choice: among settings within 0.003 of the best criterion, fewest "
           "active knobs (finite T, gamma>0, aspect norm), then highest criterion. Test cell shares are dataset "
           "statistics (DIAMOND hits only); no test labels are used.")


def cell_weights(cells: pd.DataFrame, ids) -> tuple[np.ndarray, dict]:
    """Per-protein weights over `ids` (TV v2 scorer order) that give the test cell mix (rule V2_RULE)."""
    tv = cells[cells.split == "tv"].drop_duplicates("id").set_index("id").reindex(ids)
    te = cells[cells.split == "test"].drop_duplicates("id")
    key_tv = pd.Series(index=tv.index, dtype=object)
    key_te = pd.Series(index=te.id.values, dtype=object)
    report = {}
    for b in BINS:
        pool_tv, pool_te = [], []
        for g in sorted(set(te.org_group[te.bin == b]) | set(tv.org_group[tv.bin == b])):
            mtv = (tv.bin == b) & (tv.org_group == g)
            mte = ((te.bin == b) & (te.org_group == g)).values
            fine = all(((tv.best_same_org == s) & mtv).sum() >= MIN_CELL
                       for s in set(te.best_same_org[mte]))
            if fine and mte.any():
                key_tv[mtv] = [f"{g}|{b}|{s}" for s in tv.best_same_org[mtv]]
                key_te[mte] = [f"{g}|{b}|{s}" for s in te.best_same_org[mte]]
            elif mtv.sum() >= MIN_CELL:
                key_tv[mtv] = f"{g}|{b}"
                key_te[mte] = f"{g}|{b}"
            else:
                pool_tv.append(mtv); pool_te.append(mte)
        if pool_tv:
            mtv = np.logical_or.reduce(pool_tv)
            mte = np.logical_or.reduce(pool_te)
            key_tv[mtv] = f"*|{b}"
            key_te[mte] = f"*|{b}"
    ste = key_te.value_counts(normalize=True)
    ctv = key_tv.value_counts()
    w = key_tv.map(lambda k: ste.get(k, 0.0) / ctv[k] if isinstance(k, str) else 0.0).values.astype(float)
    w = w * len(w) / w.sum()
    lost = float(ste[~ste.index.isin(ctv.index)].sum())
    report = {"n_cells": int(len(ctv)), "test_mass_without_tv_cell": round(lost, 4),
              "effective_n": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
              "cells": {k: {"tv_n": int(ctv.get(k, 0)), "test_share": round(float(ste.get(k, 0)), 4)}
                        for k in sorted(set(ste.index) | set(ctv.index))}}
    return w, report


def run_grid_v2(threads: int):
    """L0 grid on TV v2 (no test scoring). The selection rule is appended to runs/test_looks.md before any
    TV v2 score is computed."""
    use_tv("v2")
    out = RUN / "v2"
    out.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOOKS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | L0 sharpened kNN v2 | PRE-REGISTERED RULE before "
                f"any TV v2 score (no test look): {V2_RULE}\n")
    log("== L0 grid on TV v2; rule: " + V2_RULE)
    onto, go = Ontology(), GO()
    knn = build_knn(onto, go, ("val", "tv"))
    _G["knn"] = knn
    _G["scorer"] = {s: Scorer(onto, gt_frame(s)) for s in ("val", "tv")}
    sc = _G["scorer"]["tv"]
    cells = pd.read_parquet(TV_DIR / "cells.parquet")
    w, wrep = cell_weights(cells, sc.ids)
    ctv = cells[cells.split == "tv"].drop_duplicates("id").set_index("id").reindex(sc.ids)
    ident = pd.read_parquet(TV_DIR / "identity.parquet")
    idm = {s: ident[ident.split == s].drop_duplicates("id").set_index("id").max_pident for s in ("val", "tv")}
    jobs = [(s, g) for s in ("val", "tv") for g in GRID]
    t = time.time()
    with mp.get_context("fork").Pool(min(threads, len(jobs))) as pool:
        res = pool.map(_eval_setting, jobs)
    log(f"v2 grid scored in {time.time() - t:.0f}s")
    hi = sc.ids[ctv.bin.isin(["100", "95-99.9"]).values]
    groups = {"dated": sc.ids[(ctv.stratum != "DA").values], "DA": sc.ids[(ctv.stratum == "DA").values]}
    rows, bins, cellrows, curs = [], [], [], {}
    for split, st, cur in res:
        s_ = _G["scorer"][split]
        f = s_.fmax(cur)
        name = setting_name(*st)
        r = {"split": split, "setting": name, "T": st[0], "k": st[1], "gamma": st[2], "norm": st[3],
             **{a: f[a] for a in ("MF", "BP", "CC")}, "mean": f["mean"]}
        if split == "tv":
            curs[name] = cur
            r["mean_testlike_v2"] = s_.fmax(cur, weights=w)["mean"]
            r["mean_hi_id"] = s_.fmax(cur, subset=hi)["mean"]
            for gname, ids in groups.items():
                r[f"mean_{gname}"] = s_.fmax(cur, subset=ids)["mean"]
            for (g, b), d in ctv.groupby(["org_group", "bin"]):
                if len(d) >= MIN_CELL:
                    cellrows.append({"setting": name, "org_group": g, "bin": b, "n": len(d),
                                     "fmax": s_.fmax(cur, subset=d.index.values)["mean"]})
        rows.append(r)
        bins.append({"split": split, "setting": name, **bin_table(s_, cur, idm[split])})
    summ = pd.DataFrame(rows).sort_values(["split", "mean"], ascending=[True, False])
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(bins).to_csv(out / "per_bin.tsv", sep="\t", index=False)
    pd.DataFrame(cellrows).to_csv(out / "per_cell.tsv", sep="\t", index=False)

    rt = {setting_name(T, k, 0): v for (T, k), v in RED_TEAM_TEST.items()}
    common = list(rt)
    tvs = summ[summ.split == "tv"].set_index("setting")
    vas = summ[summ.split == "val"].set_index("setting")
    pb = pd.DataFrame(bins)
    pbt = pb[pb.split == "tv"].set_index("setting")
    agree = {"spearman_vs_red_team_test": {
        "val": spearman(vas.loc[common, "mean"], [rt[c] for c in common]),
        "tv2": spearman(tvs.loc[common, "mean"], [rt[c] for c in common]),
        "tv2_testlike": spearman(tvs.loc[common, "mean_testlike_v2"], [rt[c] for c in common]),
        "tv2_hi_identity": spearman(tvs.loc[common, "mean_hi_id"], [rt[c] for c in common]),
        "tv2_DA": spearman(tvs.loc[common, "mean_DA"], [rt[c] for c in common]),
        "tv2_dated": spearman(tvs.loc[common, "mean_dated"], [rt[c] for c in common])},
        "spearman_tv2testlike_vs_val_all60": spearman(tvs["mean_testlike_v2"], vas.loc[tvs.index, "mean"]),
        "red_team_direction": "test: T in {1,2,5} beats T=10 by ~0.01 (k 10 and 50)"}
    # direction check: sharp (T2) minus flat (T10, Tinf) at k=10, gamma 0, norm all, with paired CIs
    dirs = {}
    for lo, hi_ in (("T2_k10_g0", "T10_k10_g0"), ("T2_k10_g0", "Tinf_k10_g0"), ("T2_k50_g0", "Tinf_k50_g0")):
        dirs[f"{lo} - {hi_}"] = {
            "testlike": sc.bootstrap(curs[lo], curs[hi_], reps=500, weights=w),
            "hi_identity_bins": {b: float(pbt.loc[lo, b] - pbt.loc[hi_, b]) for b in ("100", "95-99.9")},
            "low_identity_bins": {b: float(pbt.loc[lo, b] - pbt.loc[hi_, b]) for b in ("80-95", "50-80", "30-50")}}
    agree["sharp_minus_flat"] = dirs
    log("v2 ranking agreement: " + json.dumps(agree, indent=1, default=float))

    tv = summ[summ.split == "tv"].copy()
    tv["knobs"] = np.isfinite(tv["T"]).astype(int) + (tv["gamma"] != 0).astype(int) + (tv["norm"] != "all").astype(int)
    crit = "mean_testlike_v2"
    best = tv[crit].max()
    ch = tv[tv[crit] >= best - 0.003].sort_values(["knobs", crit], ascending=[True, False]).iloc[0]
    top = tv.sort_values(crit, ascending=False).iloc[0]
    v1 = json.loads((RUN / "chosen.json").read_text())
    chosen = {"T": float(ch["T"]), "k": int(ch["k"]), "gamma": int(ch["gamma"]), "norm": ch["norm"],
              "setting": ch["setting"], "criterion": crit, "rule": V2_RULE,
              "tv2_mean_testlike": float(ch[crit]), "tv2_mean": float(ch["mean"]),
              "best_setting": top["setting"], "best_criterion": float(best),
              "chosen_minus_v1_choice_testlike": sc.bootstrap(curs[ch["setting"]], curs[v1["setting"]], reps=500, weights=w),
              "chosen_minus_T2_k10_testlike": sc.bootstrap(curs[ch["setting"]], curs["T2_k10_g0"], reps=500, weights=w),
              "v1_choice": v1["setting"], "weights": wrep, "ranking_agreement": agree,
              "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RUN / "chosen_v2.json").write_text(json.dumps(chosen, indent=2, default=float))
    log("v2 L0 choice: " + json.dumps({k: chosen[k] for k in ("setting", "tv2_mean_testlike", "tv2_mean",
                                                              "best_setting", "best_criterion")}))
    with open(TEST_LOOKS, "a") as f:
        f.write(f"- {chosen['registered_at']} | L0 sharpened kNN v2 | PRE-REGISTERED (no test look): "
                f"{chosen['setting']} chosen on TV v2 ({crit}) by the rule above; runs/knn_sharp/chosen_v2.json\n")


def run_write(setting: str, out_dir: Path):
    """Writes contract files for another setting (e.g. T2_k10_g0) into out_dir, without any scoring."""
    T, k, g, n = next(st for st in GRID if setting_name(*st) == setting)
    onto, go = Ontology(), GO()
    for s, kn in build_knn(onto, go, ("val", "tv", "test")).items():
        P, nh = kn.prior(T, k, g, n)
        save_prior(s, kn, P, nh, onto, {"T": T, "k": k, "gamma": g, "norm": n, "setting": setting,
                                        "tv_version": TV_VERSION,
                                        "label_db": "T0 labels" if s == "tv" else "full GO-GPT training labels",
                                        "hits": "diamond_hits_tv.parquet" if s == "tv" else "diamond_hits_valtest.parquet",
                                        "note": "not selected on TV and not scored on test"}, out_dir)
    log(f"wrote {setting} priors to {out_dir}")


def save_prior(split: str, kn: KNN, P: sp.csr_matrix, n_hits, onto: Ontology, settings: dict, out_dir: Path | None = None):
    out_dir = KNN_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
    vcols = np.array([onto.idx.get(t, -1) for t in vocab.term])
    assert (vcols >= 0).all(), "vocab term missing from the ontology"
    Pv = P[:, vcols].tocsr().astype(np.float32)
    used = np.unique(P.indices)
    Pf = P[:, used].tocsr().astype(np.float32)
    np.savez_compressed(out_dir / f"{split}_knn.npz", ids=np.asarray(kn.ids, dtype=str),
                        p_data=Pv.data, p_indices=Pv.indices, p_indptr=Pv.indptr, p_shape=np.array(Pv.shape),
                        pf_data=Pf.data, pf_indices=Pf.indices, pf_indptr=Pf.indptr, pf_shape=np.array(Pf.shape),
                        terms_full=np.array(onto.terms)[used].astype(str),
                        max_pident=kn.maxpid.astype(np.float32), has_hit=kn.has_hit,
                        n_hits=np.asarray(n_hits, np.int32), settings=json.dumps(settings))


def load_knn(split: str, d: Path | None = None) -> dict:
    z = np.load((KNN_DIR if d is None else d) / f"{split}_knn.npz", allow_pickle=False)
    out = {k: z[k] for k in ("ids", "max_pident", "has_hit", "n_hits", "terms_full")}
    out["p"] = sp.csr_matrix((z["p_data"], z["p_indices"], z["p_indptr"]), shape=tuple(z["p_shape"]))
    out["p_full"] = sp.csr_matrix((z["pf_data"], z["pf_indices"], z["pf_indptr"]), shape=tuple(z["pf_shape"]))
    out["settings"] = json.loads(str(z["settings"]))
    return out


def gogpt_matrix(onto: Ontology, sub: str = "") -> tuple[sp.csr_matrix, np.ndarray]:
    rows = defaultdict(set)
    for f in sorted((ROOT / "runs" / "gogpt_cafa5" / sub).glob("preds_shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for a in ("MF", "BP", "CC"):
                rows[r["protein_id"]] |= set(r.get(a, []))
    ids = np.array(list(rows))
    return onto.label_matrix([rows[p] for p in ids], close=False), ids


def run_final(threads: int):
    chosen = json.loads((RUN / "chosen.json").read_text())
    T, k, g, nm = chosen["T"], chosen["k"], chosen["gamma"], chosen["norm"]
    log(f"== L0 final: {chosen['setting']} (pre-registered {chosen['registered_at']})")
    onto, go = Ontology(), GO()
    knn = build_knn(onto, go, ("val", "tv", "test"))
    ident = pd.read_parquet(TV_DIR / "identity.parquet")
    rows, bins, curs, scs = [], [], {}, {}
    for s in ("val", "tv", "test"):
        kn = knn[s]
        P, nh = kn.prior(T, k, g, nm)
        settings = {"T": T, "k": k, "gamma": g, "norm": nm, "setting": chosen["setting"],
                    "label_db": "T0 labels (t0_stripped removed)" if s == "tv" else "full GO-GPT training labels",
                    "hits": "diamond_hits_tv.parquet" if s == "tv" else "diamond_hits_valtest.parquet"}
        save_prior(s, kn, P, nh, onto, settings)
        sc = Scorer(onto, gt_frame(s))
        cur = sc.curves(P, kn.ids, closed=True)
        curs[s], scs[s] = cur, sc
        f = sc.fmax(cur)
        idm = ident[ident.split == s].drop_duplicates("id").set_index("id").max_pident
        rows.append({"split": s, "setting": chosen["setting"], **{a: f[a] for a in ("MF", "BP", "CC")}, "mean": f["mean"]})
        bins.append({"split": s, "setting": chosen["setting"], **bin_table(sc, cur, idm)})
    # same test look: GO-GPT released checkpoint (beam) in the same table, with a paired bootstrap
    Pg, gids = gogpt_matrix(onto)
    cg = scs["test"].curves(Pg, gids)
    fg = scs["test"].fmax(cg)
    idt = ident[ident.split == "test"].drop_duplicates("id").set_index("id").max_pident
    rows.append({"split": "test", "setting": "gogpt_beam (reference)", **{a: fg[a] for a in ("MF", "BP", "CC")},
                 "mean": fg["mean"]})
    bins.append({"split": "test", "setting": "gogpt_beam (reference)", **bin_table(scs["test"], cg, idt)})
    bs = scs["test"].bootstrap(curs["test"], cg, reps=1000)
    common = set(open(REF / "common_proteins.txt").read().split())
    s6 = [p for p in scs["test"].ids if p not in common]
    s6row = {"knn_S6": scs["test"].fmax(curs["test"], subset=s6)["mean"], "gogpt_S6": scs["test"].fmax(cg, subset=s6)["mean"]}
    out = pd.DataFrame(rows)
    out.to_csv(RUN / "final_summary.tsv", sep="\t", index=False)
    pd.DataFrame(bins).to_csv(RUN / "final_per_bin.tsv", sep="\t", index=False)
    res = {"chosen": chosen["setting"], "bootstrap_knn_minus_gogpt_test": bs, **s6row,
           "test_mean": float(out[(out.split == "test") & (out.setting == chosen["setting"])]["mean"].iloc[0])}
    (RUN / "final.json").write_text(json.dumps(res, indent=2))
    log("final L0:\n" + out.round(4).to_string() + "\n" + pd.DataFrame(bins).round(4).to_string() + "\n" + json.dumps(res))
    with open(TEST_LOOKS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | L0 sharpened kNN | TEST LOOK #1 for L0: "
                f"{chosen['setting']} -> test mean Fmax {res['test_mean']:.4f} (GO-GPT beam {fg['mean']:.4f}, "
                f"diff {bs['diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}], 1000 reps); no further selection\n")
    (ROOT / "logs" / "knn_l0.done").touch()


def run_verify():
    """Scorer check on predictions whose cafaeval scores are already in runs/bench_gogpt/summary.tsv."""
    from cafa6.evaluate import dense_to_long
    onto = Ontology()
    sc = Scorer(onto, gt_frame("test"))
    Pg, gids = gogpt_matrix(onto)
    f1 = sc.fmax(sc.curves(Pg, gids))
    te = split_ids("test")
    vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
    d = dense_to_long(np.load(ROOT / "runs" / "gogptbench_35m_lora" / "pred_test.npy").astype(np.float32),
                      te, vocab.term.values)
    tid = {p: i for i, p in enumerate(te)}
    Ph = sp.csr_matrix((d.score.values, (d.id.map(tid).values, d.term.map(onto.idx).values)),
                       shape=(len(te), len(onto.terms)))
    f2 = sc.fmax(sc.curves(Ph, te))
    ref = pd.read_csv(ROOT / "runs" / "bench_gogpt" / "summary.tsv", sep="\t")
    ref = ref[ref.subset == "S3_all"].set_index("method")
    msg = {"gogpt_beam": {"ours": f1, "cafaeval": ref.loc["gogpt_beam", ["MF", "BP", "CC", "kaggle_mean_Fw"]].to_dict()},
           "gogptbench_35m_lora": {"ours": f2, "cafaeval": ref.loc["gogptbench_35m_lora",
                                                                   ["MF", "BP", "CC", "kaggle_mean_Fw"]].to_dict()}}
    log("scorer verification (existing predictions, no selection): " + json.dumps(msg, indent=1))
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "scorer_check.json").write_text(json.dumps(msg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--write", nargs=2, metavar=("SETTING", "DIR"), help="extra contract files, no scoring")
    ap.add_argument("--tv", default="v1", choices=("v1", "v2"), help="TV version for --write")
    ap.add_argument("--grid-v2", action="store_true", help="grid on TV v2, pre-registers chosen_v2 (no test)")
    a = ap.parse_args()
    use_tv(a.tv)
    if a.grid_v2:
        run_grid_v2(a.threads)
    if a.write:
        run_write(a.write[0], ROOT / a.write[1])
    if a.verify:
        run_verify()
    if a.grid:
        run_grid(a.threads)
    if a.final:
        run_final(a.threads)


if __name__ == "__main__":
    main()
