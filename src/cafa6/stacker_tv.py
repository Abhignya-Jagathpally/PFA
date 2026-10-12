"""HiGO-R stacker fit on TV v2 with cross-fitting: HiGO + the pre-registered kNN prior -> per-term probabilities.

Why TV v2: every learned combiner fit on the random val split lost to the raw kNN prior on test, almost all of it
at >= 95% identity (runs/meanflow_pilot/notes.md section 3), because random val has much weaker identity-to-label
transfer than the temporal test. TV v2 (data/proc_gogpt_tv2) is a temporal holdout whose cells (organism group x
identity bin x same-organism best hit) can be reweighted to the test mix (knn_sharp.cell_weights, rule V2_RULE),
so the stacker is fit and selected there, never on val and never on test.

Inputs per (protein, candidate term)
------------------------------------
  HiGO     logit of the HiGO probability (pred_tv_all_top500.npz etc., read with data.load_pred), whether the term is
           in the protein's HiGO top-k, whether it is in the 7,901-term vocabulary; terms outside the top-k get the
           protein's k-th score (an upper bound), out-of-vocabulary terms get 1e-4
  kNN      T5_k50_g0_na prior from data/proc_gogpt_tv2/knn/{split}_knn.npz over the full ontology (p_full: dropping
           out-of-vocabulary kNN terms costs 0.010 test-like Fmax on TV v2), its logit, p > 0
  protein  max identity (raw + 9 RBF bumps), has_hit, log #hits, best hit in the same organism, organism group
           (cells.parquet), identity bin (gate of model c)
  term     IA (CAFA-5), longest-path depth, aspect; plus interactions logit x identity
Candidates = union of the HiGO top-k (--higo-topk, default 300) and every term the kNN prior reaches.

Models (every one starts exactly at the kNN prior: zero-initialised residual / weights that reproduce logit p_knn)
  a  fusion      per-term logistic fusion: logit = x . (w_aspect + delta_term), L2 1e-3 on delta (the control)
  b  twin        the discriminative twin of the MeanFlow pilot: shared per-term MLP over the row features, a learned
                 term embedding and a protein vector (MLP of protein features + p_knn- and HiGO-weighted bags of term
                 embeddings, i.e. label co-occurrence), BCE on logit(p_knn) + residual, residual output zero-init
  c  gated_twin  b with the prior logit replaced by a_{bin,aspect} logit(p_knn) + b_{bin,aspect} logit(p_higo)
                 + c_{bin,aspect} (init 1, 0, 0) over the 7 identity bins, plus the same residual

Cross-fitting
-------------
5 folds over TV v2 proteins, grouped into connected components of {protein - its best DIAMOND hit, protein - any
other TV protein it hits, shared sequence cluster} so near-duplicates never straddle folds. Test-like weights w
(chosen_v2 rule) enter as importance weights: proteins are sampled with probability ~ sqrt(w) and each scored row
carries loss weight sqrt(w) (unbiased for the w-weighted loss, lower variance than uniform sampling), times
IA / mean IA under --loss ia_bce (the default, as the metric weights terms by IA; it tied plain BCE on the val stand-in,
runs/stacker_tv/design_val_*, so the choice was not made on TV). Only rows in aspects where the protein has ground
truth are trained on. AdamW, linear warm-up over --warmup steps. Every --eval-every steps the held-out fold is predicted,
max-propagated over the GO graph (Ontology.propagate_max, i.e. MCM over go_ancestors and the out-of-vocabulary
terms) and scored with knn_sharp.Scorer under the test-like weights; training stops after --patience evaluations
without improvement on the held-out fold. Step 0 is the kNN prior itself.

Two pooled out-of-fold estimates are reported:
  es    each fold at its own best step on its own held-out curve (as requested; optimistic: 5 step choices on the
        data they are scored on)
  lofo  each fold at the step that maximises the mean held-out curve of the OTHER four folds (no fold's labels pick
        its own step). The selection rule uses lofo.

PRE-REGISTERED SELECTION RULE (also appended to runs/test_looks.md before any HiGO-R stacker score; no test look)
------------------------------------------------------------------------------------------------------------------
See SELECTION_RULE below; its text is the registration.

Outputs (--out): log.txt, sizes.json, folds.tsv, curves.tsv, summary.tsv, per_bin.tsv, result.json, and for full
runs with a HiGO run: pred_tv_oof_{model}.npz (cross-fitted) and pred_{val,test}_{model}.npz from a refit on all TV
at the median lofo step (val is scored, test is written but NOT scored).

Usage:
  python -m cafa6.stacker_tv --higo-run runs/l3_c1_tv2 --out runs/stacker_tv/l3_c1
  python -m cafa6.stacker_tv --higo-run none --out runs/stacker_tv/smoke_knn_only --smoke
  python -m cafa6.stacker_tv --split val --higo-run runs/gogptbench_35m_lora --out runs/stacker_tv/smoke_val --smoke
  python -m cafa6.stacker_tv --register-rule
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from cafa6 import knn_sharp as ks
from cafa6.data import ROOT, load_pred

TV2 = ROOT / "data" / "proc_gogpt_tv2"
KNN_DIR = Path(os.environ.get("CAFA6_KNN_DIR", TV2 / "knn"))  # e.g. TV2/knn_trainval: train + val references


def knn_dir(split: str) -> Path:
    """KNN_DIR when it has this split's prior, else the train-only TV2/knn (knn_trainval has no val prior)."""
    return KNN_DIR if (KNN_DIR / f"{split}_knn.npz").exists() else TV2 / "knn"
TEST_LOOKS = ROOT / "runs" / "test_looks.md"
ORGS = ("human", "mouse", "rat", "zebrafish", "other")
RBF_C = np.array([0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0], np.float32)
RBF_S = 0.06
EPS = 1e-4
MODELS = {"a": "fusion", "b": "twin", "c": "gated_twin"}
FUSION_LAMBDA = 1e-3
MIN_GAIN = 0.005
RULE_TAG = "HiGO-R TV stacker (stacker_tv.py)"
SELECTION_RULE = (
    "candidates = models a (per-term logistic fusion), b (twin refiner), c (identity-gated twin) of "
    "src/cafa6/stacker_tv.py, for each L3 HiGO run with TV v2 predictions (runs/l3_c0_tv2, runs/l3_c1_tv2), each fit "
    "on TV v2 with 5-fold cross-fitting grouped by best-hit / TV-link / cluster components, test-like weights "
    "(chosen_v2 rule) as sample weights, each fold's step count chosen on the other four folds' held-out test-like "
    "Fmax curves (lofo), pooled out-of-fold predictions max-propagated and scored with knn_sharp.Scorer. A candidate "
    f"is KEPT only if its cross-fitted TV v2 test-like mean Fmax minus that of the raw T5_k50_g0_na kNN prior is >= "
    f"{MIN_GAIN} AND the paired protein bootstrap 95% CI of that difference (1000 reps, test-like weights) excludes 0. "
    "Among kept candidates the highest cross-fitted test-like Fmax goes forward to a single test look; if none is "
    "kept, no stacker goes forward (the raw kNN prior stays). Smoke runs, val-split runs and HiGO-less runs are not "
    "selection runs.")
T_START = time.time()
LOG: Path | None = None


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')} {time.time() - T_START:7.0f}s] {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def register_rule():
    """Appends the selection rule to runs/test_looks.md once (no scores, no test look)."""
    txt = TEST_LOOKS.read_text() if TEST_LOOKS.exists() else ""
    if RULE_TAG in txt:
        log("selection rule already registered in runs/test_looks.md")
        return
    with open(TEST_LOOKS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {RULE_TAG} | PRE-REGISTERED RULE before any HiGO-R stacker "
                f"score on TV v2 (no test look): {SELECTION_RULE}\n")
    log("selection rule appended to runs/test_looks.md")


def pick_device(req: str) -> torch.device:
    """A GPU only if one has > 10 GB free (by nvidia-smi, so no CUDA context is opened on the busy GPUs)."""
    if req == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        free = {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    except Exception as e:  # noqa: BLE001
        log(f"nvidia-smi failed ({e}); using CPU")
        return torch.device("cpu")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = [int(x) for x in vis.split(",") if x.strip()] if vis else sorted(free)
    if req.startswith("cuda:"):
        phys = [phys[int(req.split(":")[1])]]
    best = max(phys, key=lambda g: free.get(g, 0))
    log(f"GPU free MiB: {free}; candidates {phys}")
    if free.get(best, 0) > 10240:
        return torch.device(f"cuda:{phys.index(best) if vis else best}")
    log("no GPU with > 10 GB free; using CPU")
    return torch.device("cpu")


def logit_np(p):
    p = np.clip(p, EPS, 1 - EPS)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def lookup(M: sp.csr_matrix, key: np.ndarray, nT: int) -> np.ndarray:
    """Values of CSR M (sorted indices) at flat keys row * nT + col (0 where absent)."""
    M = M.tocsr()
    M.sort_indices()
    if M.nnz == 0:
        return np.zeros(len(key), np.float32)
    mk = np.repeat(np.arange(M.shape[0], dtype=np.int64), np.diff(M.indptr)) * nT + M.indices
    i = np.minimum(np.searchsorted(mk, key), len(mk) - 1)
    return np.where(mk[i] == key, M.data[i], 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ data

def resolve_run(r: str | None) -> Path | None:
    if r is None or r.lower() == "none":
        return None
    p = Path(r)
    return p if p.is_absolute() else ROOT / p


def higo_matrix(run: Path, split: str, ids: np.ndarray, onto, topk: int):
    """HiGO top-k probabilities as CSR over ontology columns, rows aligned to ids, plus each protein's k-th score.
    split tv reads pred_tv_all* (tv_ids.txt order); val/test read pred_{split}* (train.parquet split order)."""
    name = "tv_all" if split == "tv" else split
    if split == "tv":
        rid = np.array(open(TV2 / "tv_ids.txt").read().split())
    else:
        df = pd.read_parquet(TV2 / "train.parquet", columns=["id", "split"])
        rid = df.id.values[(df.split == split).values].astype(str)
    M = load_pred(run, name, dense=False)
    D = M.toarray() if sp.issparse(M) else np.asarray(M)
    if D.shape[0] != len(rid):
        raise ValueError(f"{run}: pred_{name} has {D.shape[0]} rows, expected {len(rid)}")
    first = pd.Series(np.arange(len(rid)), index=rid)
    first = first[~first.index.duplicated()]
    miss = [p for p in ids if p not in first.index]
    if miss:
        raise KeyError(f"{len(miss)} proteins without HiGO predictions, e.g. {miss[:3]}")
    D = D[first.loc[ids].values].astype(np.float32)
    vocab = pd.read_csv(TV2 / "vocab.tsv", sep="\t")
    vcols = np.array([onto.idx[t] for t in vocab.term])
    k = min(topk, D.shape[1])
    idx = np.argpartition(-D, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(D, idx, 1)
    keep = val > 0
    rows = np.repeat(np.arange(len(ids)), k).reshape(len(ids), k)
    H = sp.csr_matrix((val[keep], (rows[keep], vcols[idx[keep]])), shape=(len(ids), len(onto.terms)))
    floor = np.where(keep.any(1), np.where(keep, val, np.inf).min(1), 0.0).astype(np.float32)
    return H, floor, vcols


def cells_for(split: str, ids: np.ndarray) -> pd.DataFrame:
    c = pd.read_parquet(TV2 / "cells.parquet")
    return c[c.split == split].drop_duplicates("id").set_index("id").reindex(ids)


def test_like_weights(split: str, sc: ks.Scorer) -> tuple[np.ndarray, dict]:
    """knn_sharp.cell_weights (chosen_v2 rule) over the scorer's proteins; a val split stands in for tv in smokes."""
    c = pd.read_parquet(TV2 / "cells.parquet")
    c = c[c.split.isin([split, "test"])].copy()
    c.loc[c.split == split, "split"] = "tv"
    return ks.cell_weights(c, sc.ids)


def build_split(split: str, onto, higo_run: Path | None, topk: int, with_gt: bool = True,
                loss: str = "ia_bce") -> dict:
    """Rows and features of one split; with_gt adds labels, the scored mask, test-like weights w and the row loss
    weights m = scored * sqrt(w) (* IA / mean IA for loss ia_bce)."""
    t0 = time.time()
    K = ks.load_knn(split, knn_dir(split))
    ids = K["ids"].astype(str)
    n, nT = len(ids), len(onto.terms)
    cols = np.array([onto.idx[t] for t in K["terms_full"]])
    pf = K["p_full"].tocoo()
    Pk = sp.csr_matrix((pf.data, (pf.row, cols[pf.col])), shape=(n, nT))
    Pk.sum_duplicates()
    pat = Pk.copy()
    pat.data[:] = 1
    H = None
    if higo_run is not None:
        H, floor, vcols = higo_matrix(higo_run, split, ids, onto, topk)
        hp = H.copy()
        hp.data[:] = 1
        pat = (pat + hp).tocsr()
    pat.sort_indices()
    pix = np.repeat(np.arange(n), np.diff(pat.indptr)).astype(np.int64)
    t = pat.indices.astype(np.int64)
    key = pix * nT + t
    pk = lookup(Pk, key, nT)
    mx = (K["max_pident"] / 100).astype(np.float32)
    if H is not None:
        ph = lookup(H, key, nT)
        in_top = (ph > 0).astype(np.float32)
        vm = np.zeros(nT, bool)
        vm[vcols] = True
        in_voc = vm[t].astype(np.float32)
        ph_eff = np.where(in_top > 0, ph, np.where(in_voc > 0, floor[pix], 0.0)).astype(np.float32)
        lh = logit_np(ph_eff)
    else:
        ph = in_top = in_voc = ph_eff = lh = np.zeros(len(t), np.float32)
    lk = logit_np(pk)
    asp = onto.ns[t].astype(np.int64)
    mr = mx[pix]
    X = np.stack([lk / 4, lh / 4, pk, in_top, in_voc, ph_eff, np.log1p(onto.ia[t]).astype(np.float32) / 3,
                  onto.depth[t] / 10, asp == 0, asp == 1, asp == 2, lk * mr / 4, lh * mr / 4, pk > 0, pk * ph_eff],
                 1).astype(np.float32)
    cl = cells_for(split, ids)
    org = cl.org_group.fillna("other").values
    same = cl.best_same_org.fillna(False).values.astype(np.float32)
    rbf = np.exp(-((mx[:, None] - RBF_C[None]) / RBF_S) ** 2) * K["has_hit"][:, None]
    Q = np.concatenate([np.stack([mx, K["has_hit"].astype(np.float32), same,
                                  np.log1p(K["n_hits"]).astype(np.float32) / 4], 1),
                        rbf, np.stack([org == o for o in ORGS], 1)], 1).astype(np.float32)
    ibin = pd.Series(ks.ident_bins(K["max_pident"])).map({b: i for i, b in enumerate(ks.BINS)}).values
    d = {"split": split, "ids": ids, "pix": pix, "t": t, "asp": asp, "X": X, "Q": Q, "pk": pk, "ia": onto.ia[t],
         "ph": (ph * in_top).astype(np.float32), "ptr": pat.indptr.astype(np.int64), "ibin": ibin.astype(np.int64),
         "bin_name": cl.bin.fillna("no_hit").values, "Pk": Pk, "higo": H is not None, "ph_eff": ph_eff}
    if with_gt:
        sc = ks.Scorer(onto, ks.gt_frame(split))
        gi = np.array([sc.pos.get(p, -1) for p in ids], np.int64)
        g = gi[pix]
        ok = g >= 0
        y = np.isin(np.where(ok, g * nT + t, -1), sc.gkey) & ok
        scored = ok.copy()
        scored[ok] = sc.in_ns[asp[ok], g[ok]]
        w_sc, wrep = test_like_weights(split, sc)
        w = np.where(gi >= 0, w_sc[np.maximum(gi, 0)], 0.0)
        m = scored * np.sqrt(w)[pix]
        if loss == "ia_bce":
            m = m * d["ia"] / max(float(d["ia"][scored].mean()), 1e-9)
        d.update(sc=sc, gi=gi, y=y.astype(np.float32), scored=scored, w=w, w_sc=w_sc, wrep=wrep,
                 m=m.astype(np.float32))
    sizes = {"split": split, "proteins": n, "rows": int(len(t)), "rows_per_protein_mean": round(len(t) / n, 1),
             "rows_per_protein_max": int(np.diff(pat.indptr).max()), "knn_rows": int((pk > 0).sum()),
             "higo_topk_rows": int(in_top.sum()), "both": int(((pk > 0) & (in_top > 0)).sum()),
             "feature_MB": round((X.nbytes + 4 * 8 * len(t)) / 2 ** 20, 1), "build_s": round(time.time() - t0, 1)}
    if with_gt:
        sizes.update(scored_rows=int(scored.sum()), positives=int(d["y"][scored].sum()),
                     pos_rate=round(float(d["y"][scored].mean()), 4),
                     gt_terms_covered=round(float(d["y"].sum() / max(sc.G.nnz, 1)), 4),
                     testlike_effective_n=wrep["effective_n"])
    d["sizes"] = sizes
    log(f"[{split}] " + json.dumps(sizes))
    return d


def make_folds(d: dict, n_folds: int, seed: int = 0) -> np.ndarray:
    """Fold per protein: components of best-hit / TV-to-TV hit / same-cluster links, greedily balanced by count."""
    ids = d["ids"]
    idset = set(ids)
    h = ks.load_hits(d["split"])
    h = h[h.q.isin(idset)].sort_values(["q", "bits"], ascending=[True, False])
    node = {p: i for i, p in enumerate(ids)}
    parent = list(range(len(ids)))

    def nid(x):
        if x not in node:
            node[x] = len(parent)
            parent.append(len(parent))
        return node[x]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(nid(a)), find(nid(b))
        if ra != rb:
            parent[ra] = rb

    best = h.drop_duplicates("q")
    for q, s in zip(best.q, best.s):
        union(q, "s:" + s if s not in idset else s)
    tt = h[h.s.isin(idset) & (h.s != h.q)]
    for q, s in zip(tt.q, tt.s):
        union(q, s)
    src = TV2 if d["split"] == "tv" else ks.SRC
    cl = pd.read_parquet(src / "train.parquet", columns=["id", "cluster"]).drop_duplicates("id").set_index("id").cluster
    for p, c in zip(ids, cl.reindex(ids).values):
        if isinstance(c, str):
            union(p, "c:" + c)
    comp = np.array([find(node[p]) for p in ids])
    uc, inv, cnt = np.unique(comp, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uc))
    order = order[np.argsort(-cnt[order], kind="stable")]
    load, fold_c = np.zeros(n_folds), np.zeros(len(uc), np.int64)
    for c in order:
        f = int(np.argmin(load))
        fold_c[c] = f
        load[f] += cnt[c]
    fold = fold_c[inv]
    log(f"folds: {len(uc)} components (largest {cnt.max()}, {int((cnt > 1).sum())} with >1 protein); per fold n "
        f"{np.bincount(fold, minlength=n_folds).tolist()}, test-like weight "
        f"{[round(float(d['w'][fold == f].sum()), 1) for f in range(n_folds)]}")
    d["component"] = inv
    return fold


class Data:
    """Row tensors on the training device and protein-batch gathering."""

    def __init__(self, d: dict, dev: torch.device):
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.dev = dev
        self.X, self.Q = T(d["X"]), T(d["Q"])
        self.t, self.asp = T(d["t"], torch.long), T(d["asp"], torch.long)
        self.pk, self.ph = T(d["pk"]), T(d["ph"])
        self.ptr, self.ibin = T(d["ptr"], torch.long), T(d["ibin"], torch.long)
        self.ptr_np = d["ptr"]
        if "y" in d:
            self.y = T(d["y"])
            self.m = T(d["m"])

    def rows_of(self, prots: np.ndarray) -> np.ndarray:
        s, e = self.ptr_np[prots], self.ptr_np[prots + 1]
        return np.concatenate([np.arange(a, b) for a, b in zip(s, e)]) if len(prots) else np.zeros(0, np.int64)

    def gather(self, prots: torch.Tensor, labels: bool = True) -> dict:
        starts = self.ptr[prots]
        lens = self.ptr[prots + 1] - starts
        off = torch.cumsum(lens, 0) - lens
        loc = torch.repeat_interleave(torch.arange(len(prots), device=self.dev), lens)
        rows = starts[loc] + torch.arange(int(lens.sum()), device=self.dev) - off[loc]
        b = {"X": self.X[rows], "t": self.t[rows], "asp": self.asp[rows], "pk": self.pk[rows], "ph": self.ph[rows],
             "loc": loc, "off": off, "Q": self.Q[prots], "ibin": self.ibin[prots], "B": len(prots)}
        if labels and hasattr(self, "y"):
            b["y"], b["m"] = self.y[rows], self.m[rows]
        return b


# ------------------------------------------------------------------------------------------------ models

class Fusion(nn.Module):
    """logit = z . (W_aspect + D_term), z = [lk, lh, pk, in_top, lk*id, lh*id, id, has_hit, same_org, 1] (lk, lh / 4)."""

    def __init__(self, nT: int):
        super().__init__()
        self.Wa = nn.Parameter(torch.zeros(3, 10))
        with torch.no_grad():
            self.Wa[:, 0] = 4.0
        self.D = nn.Embedding(nT, 10)
        nn.init.zeros_(self.D.weight)

    def forward(self, b):
        Qr = b["Q"][b["loc"]]
        z = torch.cat([b["X"][:, [0, 1, 2, 3, 11, 12]], Qr[:, :3], torch.ones_like(Qr[:, :1])], -1)
        self.dt = self.D(b["t"])
        return (z * (self.Wa[b["asp"]] + self.dt)).sum(-1)

    def penalty(self):
        return FUSION_LAMBDA * self.dt.pow(2).sum(-1).mean()


class Twin(nn.Module):
    """Residual on the prior logit: shared per-term MLP over [row features, protein features] + term embedding +
    protein vector g (MLP of protein features and p_knn-/HiGO-weighted bags of term embeddings). gated=True makes the
    prior a_{bin,asp} logit p_knn + b_{bin,asp} logit p_higo + c_{bin,asp} (model c)."""

    def __init__(self, nT: int, nf: int, nq: int, h: int = 64, d: int = 128, e: int = 32, gated: bool = False):
        super().__init__()
        self.E = nn.Embedding(nT, h)
        nn.init.normal_(self.E.weight, std=0.02)
        self.bagK = nn.EmbeddingBag(nT, e, mode="sum")
        self.bagH = nn.EmbeddingBag(nT, e, mode="sum")
        for bg in (self.bagK, self.bagH):
            nn.init.normal_(bg.weight, std=0.1)
        self.qnet = nn.Sequential(nn.Linear(nq + 2 * e, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.inp, self.gp = nn.Linear(nf + nq, h), nn.Linear(d, h)
        self.mid, self.out = nn.Linear(h, h), nn.Linear(h, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.gated = gated
        if gated:
            nb = len(ks.BINS)
            self.ga = nn.Parameter(torch.ones(nb, 3))
            self.gb = nn.Parameter(torch.zeros(nb, 3))
            self.gc = nn.Parameter(torch.zeros(nb, 3))

    def forward(self, b):
        loc, B = b["loc"], b["B"]
        nk = torch.zeros(B, device=loc.device).index_add_(0, loc, b["pk"])
        nh = torch.zeros(B, device=loc.device).index_add_(0, loc, b["ph"])
        bk = self.bagK(b["t"], b["off"], per_sample_weights=b["pk"]) / (nk + 1)[:, None]
        bh = self.bagH(b["t"], b["off"], per_sample_weights=b["ph"]) / (nh + 1)[:, None]
        g = self.qnet(torch.cat([b["Q"], bk, bh], -1))
        x = F.gelu(self.inp(torch.cat([b["X"], b["Q"][loc]], -1)) + self.E(b["t"]) + self.gp(g)[loc])
        x = F.gelu(self.mid(x)) + x
        res = self.out(x).squeeze(-1)
        lk, lh = b["X"][:, 0] * 4, b["X"][:, 1] * 4
        if self.gated:
            i = b["ibin"][loc] * 3 + b["asp"]
            return self.ga.view(-1)[i] * lk + self.gb.view(-1)[i] * lh + self.gc.view(-1)[i] + res
        return lk + res

    def penalty(self):
        return 0.0


def make_model(kind: str, d: dict) -> nn.Module:
    nT = int(d["nT"])
    if kind == "fusion":
        return Fusion(nT)
    return Twin(nT, d["X"].shape[1], d["Q"].shape[1], gated=(kind == "gated_twin"))


# ------------------------------------------------------------------------------------------------ scoring

class Ctx:
    """Scores row predictions of a split: cut at 0.01, max-propagate over GO (MCM), knn_sharp.Scorer."""

    def __init__(self, d: dict, onto):
        self.d, self.onto, self.sc = d, onto, d["sc"]
        self.n, self.nT = len(d["ids"]), len(onto.terms)

    def matrix(self, rows: np.ndarray, p: np.ndarray) -> sp.csr_matrix:
        p = np.asarray(p, np.float32)
        k = p >= 0.01
        P = sp.csr_matrix((p[k], (self.d["pix"][rows][k], self.d["t"][rows][k])), shape=(self.n, self.nT))
        return self.onto.propagate_max(P)

    def curves(self, rows, p):
        return self.sc.curves(self.matrix(rows, p), self.d["ids"], closed=False)

    def weights(self, prots=None, testlike=True) -> np.ndarray:
        w = self.d["w_sc"].copy() if testlike else np.ones(self.sc.n)
        if prots is not None:
            m = np.zeros(self.sc.n)
            gi = self.d["gi"][prots]
            m[gi[gi >= 0]] = 1
            w = w * m
        return w

    def fmax(self, cur, prots=None, testlike=True) -> dict:
        return self.sc.fmax(cur, weights=self.weights(prots, testlike))


def gpu_mem(dev) -> str:
    if dev.type != "cuda":
        return "cpu"
    s = f"{torch.cuda.max_memory_allocated(dev) / 2 ** 30:.2f} GB peak"
    torch.cuda.reset_peak_memory_stats(dev)
    return s


@torch.no_grad()
def predict(model, D: Data, prots: np.ndarray, chunk: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(prots), chunk):
        b = D.gather(torch.as_tensor(prots[i:i + chunk], device=D.dev), labels=False)
        out.append(torch.sigmoid(model(b)).float().cpu())
    model.train()
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def train_fold(kind: str, d: dict, D: Data, ctx: Ctx | None, tr: np.ndarray, te: np.ndarray, cfg, fold,
               steps: int | None = None):
    """Trains on proteins tr. With te / ctx: evaluates the held-out fold every cfg.eval_every steps (step 0 = prior),
    early-stops on its test-like Fmax, returns (curve rows, {step: held-out row predictions}). With steps: trains
    exactly that many steps and returns the model."""
    torch.manual_seed(cfg.seed + 101 * fold)
    model = make_model(kind, d).to(D.dev)
    lr = cfg.lr_fusion if kind == "fusion" else cfg.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0 if kind == "fusion" else 0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    sw = np.sqrt(d["w"][tr])
    tr, sw = tr[sw > 0], sw[sw > 0]
    probs = torch.as_tensor(sw / sw.sum(), dtype=torch.float32, device=D.dev)
    tr_t = torch.as_tensor(tr, device=D.dev)
    if te is not None:
        te_rows = D.rows_of(te)
        y, m = d["y"][te_rows], d["m"][te_rows]
    curve, preds, run_loss, t0 = [], {}, [], time.time()
    best, bad = -1.0, 0

    def evaluate(step):
        p = predict(model, D, te)
        cur = ctx.curves(te_rows, p)
        f = ctx.fmax(cur, prots=te)
        fu = ctx.fmax(cur, prots=te, testlike=False)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        hl = float(-(m * (y * np.log(pc) + (1 - y) * np.log1p(-pc))).sum() / max(m.sum(), 1e-9))
        tl = float(np.mean(run_loss)) if run_loss else float("nan")
        run_loss.clear()
        preds[step] = p.astype(np.float16)
        curve.append({"model": kind, "fold": fold, "step": step, "train_loss": tl, "heldout_loss": hl,
                      "fmax_testlike": f["mean"], "fmax": fu["mean"], "MF": f["MF"], "BP": f["BP"], "CC": f["CC"],
                      "elapsed_s": round(time.time() - t0, 1)})
        log(f"  {kind} fold {fold} step {step:5d}  train {tl:.5f}  heldout {hl:.5f}  test-like Fmax {f['mean']:.4f}"
            f"  (unweighted {fu['mean']:.4f})  {time.time() - t0:.0f}s")
        return f["mean"]

    n_steps = steps if steps is not None else cfg.steps
    if steps is None:
        best = evaluate(0)
    for step in range(1, n_steps + 1):
        idx = tr_t[torch.multinomial(probs, cfg.batch, replacement=True)]
        b = D.gather(idx)
        z = model(b)
        loss = (F.binary_cross_entropy_with_logits(z, b["y"], reduction="none") * b["m"]).sum() / \
            b["m"].sum().clamp_min(1e-8) + model.penalty()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())
        if steps is None and step % cfg.eval_every == 0:
            f = evaluate(step)
            if f > best + 1e-4:
                best, bad = f, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"  {kind} fold {fold}: early stop at step {step} (best {best:.4f})")
                    break
    if steps is not None:
        return model
    return curve, preds


def lofo_steps(curves: pd.DataFrame, n_folds: int) -> dict:
    """For each fold, the step maximising the mean held-out test-like Fmax of the other folds (curves carried
    forward after a fold's early stop), capped at the fold's own last evaluated step."""
    grid = np.sort(curves.step.unique())
    M = np.zeros((n_folds, len(grid)))
    last = {}
    for f in range(n_folds):
        c = curves[curves.fold == f].set_index("step").fmax_testlike
        M[f] = c.reindex(grid).ffill().values
        last[f] = int(c.index.max())
    out = {}
    for f in range(n_folds):
        others = np.delete(M, f, 0).mean(0)
        s = int(grid[int(np.argmax(others))])
        out[f] = min(s, last[f])
    return out


def bin_rows(ctx: Ctx, cur, name: str) -> dict:
    d = ctx.d
    r = {"method": name, "testlike": ctx.fmax(cur)["mean"], "all": ctx.fmax(cur, testlike=False)["mean"]}
    for b in ks.BINS:
        prots = np.where(d["bin_name"] == b)[0]
        r[b] = ctx.fmax(cur, prots=prots, testlike=False)["mean"] if len(prots) else np.nan
        r[f"n_{b}"] = len(prots)
    return r


def save_sparse(path: Path, P: sp.csr_matrix, ids, onto):
    P = P.tocsr()
    used = np.unique(P.indices)
    Pu = P[:, used].tocsr()
    np.savez_compressed(path, ids=np.asarray(ids).astype(str), data=Pu.data.astype(np.float16), indices=Pu.indices,
                        indptr=Pu.indptr, shape=np.array(Pu.shape), terms=np.array(onto.terms)[used].astype(str))


# ------------------------------------------------------------------------------------------------ main

def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--higo-run", default="none", help="run dir with pred_tv_all*/pred_val*/pred_test*, or none")
    ap.add_argument("--out", default=None, help="output dir, e.g. runs/stacker_tv/l3_c1")
    ap.add_argument("--smoke", action="store_true", help="short training (300 steps), 200 bootstrap reps, no refit")
    ap.add_argument("--split", default="tv", choices=("tv", "val"), help="val = stand-in smoke only, never selection")
    ap.add_argument("--models", default="a,b,c")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000, help="max steps per fold")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6, help="evaluations without held-out improvement")
    ap.add_argument("--batch", type=int, default=256, help="proteins per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-fusion", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=100, help="linear lr warm-up steps")
    ap.add_argument("--loss", default="ia_bce", choices=("ia_bce", "bce"),
                    help="ia_bce weights each row by its term's IA, as the metric does")
    ap.add_argument("--higo-topk", type=int, default=300)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:N (GPU still needs > 10 GB free)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-apply", action="store_true", help="skip the refit on all TV and val/test predictions")
    ap.add_argument("--refit-steps", type=int, default=None, help="override the median lofo step for the refit")
    ap.add_argument("--register-rule", action="store_true", help="append the selection rule to runs/test_looks.md")
    a = ap.parse_args()
    ks.use_tv("v2")
    if a.register_rule:
        register_rule()
        if a.out is None:
            return
    if a.out is None:
        ap.error("--out is required")
    if a.smoke:
        a.steps, a.eval_every, a.patience, a.boot = min(a.steps, 300), min(a.eval_every, 50), min(a.patience, 4), \
            min(a.boot, 200)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "log.txt"
    torch.set_num_threads(a.threads)
    higo = resolve_run(a.higo_run)
    selection_run = (not a.smoke) and a.split == "tv" and higo is not None
    log(f"== stacker_tv  split={a.split}  higo={higo}  out={out}  smoke={a.smoke}  selection_run={selection_run}")
    log("config " + json.dumps(vars(a)))
    if selection_run:
        register_rule()
    dev = pick_device(a.device)
    log(f"device {dev}")

    log("== stage 1: features")
    onto = ks.Ontology()
    d = build_split(a.split, onto, higo, a.higo_topk, loss=a.loss)
    d["nT"] = len(onto.terms)
    (out / "sizes.json").write_text(json.dumps(d["sizes"], indent=2))
    ctx = Ctx(d, onto)
    fold = make_folds(d, a.folds, a.seed)
    pd.DataFrame({"id": d["ids"], "fold": fold, "component": d["component"], "bin": d["bin_name"],
                  "w_testlike": d["w"]}).to_csv(out / "folds.tsv", sep="\t", index=False)
    D = Data(d, dev)
    log(f"tensors on {dev}: {gpu_mem(dev)}")

    log("== stage 2: references")
    all_rows = np.arange(len(d["t"]))
    cur_knn = ctx.curves(all_rows, d["pk"])
    refs = {"knn T5_k50_g0_na": cur_knn}
    if d["higo"]:
        refs[f"higo alone (top {a.higo_topk})"] = ctx.curves(all_rows, d["ph"])
    bins = [bin_rows(ctx, c, k) for k, c in refs.items()]
    for r in bins:
        log(f"  {r['method']:24s} test-like {r['testlike']:.4f}  unweighted {r['all']:.4f}")

    log("== stage 3: cross-fitting")
    kinds = [MODELS[k.strip()] for k in a.models.split(",")]
    all_curves, rows, oof, sel_steps = [], [], {}, {}
    for kind in kinds:
        tk = time.time()
        curves, preds = [], {}
        for f in range(a.folds):
            tf = time.time()
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            c, p = train_fold(kind, d, D, ctx, tr, te, a, f)
            curves += c
            preds[f] = p
            bf = max(c, key=lambda r: r["fmax_testlike"])
            log(f"  {kind} fold {f}: n_train {len(tr)} n_test {len(te)}; best held-out test-like Fmax "
                f"{bf['fmax_testlike']:.4f} at step {bf['step']} (prior {c[0]['fmax_testlike']:.4f}); "
                f"{time.time() - tf:.0f}s; GPU {gpu_mem(dev)}")
        cv = pd.DataFrame(curves)
        all_curves.append(cv)
        lo = lofo_steps(cv, a.folds)
        es = {f: int(cv[cv.fold == f].sort_values(["fmax_testlike", "step"], ascending=[False, True]).step.iloc[0])
              for f in range(a.folds)}
        sel_steps[kind] = {"lofo": lo, "es": es}
        for est, steps_f in (("lofo", lo), ("es", es)):
            p = np.zeros(len(d["t"]), np.float32)
            for f in range(a.folds):
                p[D.rows_of(np.where(fold == f)[0])] = preds[f][steps_f[f]]
            cur = ctx.curves(all_rows, p)
            oof[(kind, est)] = (p, cur)
            fm, fu = ctx.fmax(cur), ctx.fmax(cur, testlike=False)
            bs = ctx.sc.bootstrap(cur, cur_knn, reps=a.boot, weights=ctx.weights())
            rows.append({"model": kind, "estimate": est, "fmax_testlike": fm["mean"], "MF_tl": fm["MF"],
                         "BP_tl": fm["BP"], "CC_tl": fm["CC"], "fmax": fu["mean"],
                         "delta_vs_knn_testlike": bs["diff"], "ci_lo": bs["lo"], "ci_hi": bs["hi"],
                         "steps": json.dumps(steps_f)})
            bins.append(bin_rows(ctx, cur, f"{kind} ({est})"))
            log(f"  {kind} pooled OOF ({est}, steps {steps_f}): test-like {fm['mean']:.4f} (kNN "
                f"{bins[0]['testlike']:.4f}, diff {bs['diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}]); "
                f"unweighted {fu['mean']:.4f}")
        log(f"  {kind} done in {time.time() - tk:.0f}s")
        pd.concat(all_curves).to_csv(out / "curves.tsv", sep="\t", index=False)

    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    pb = pd.DataFrame(bins)
    pb.to_csv(out / "per_bin.tsv", sep="\t", index=False)
    lofo = summ[summ.estimate == "lofo"].copy()
    lofo["keep"] = (lofo.delta_vs_knn_testlike >= MIN_GAIN) & (lofo.ci_lo > 0)
    kept = lofo[lofo.keep].sort_values("fmax_testlike", ascending=False)
    res = {"split": a.split, "higo_run": str(higo), "smoke": a.smoke, "selection_run": selection_run,
           "rule": SELECTION_RULE, "knn_testlike": bins[0]["testlike"], "knn_unweighted": bins[0]["all"],
           "models": {r.model: {"lofo_testlike": r.fmax_testlike, "delta": r.delta_vs_knn_testlike,
                                "ci": [r.ci_lo, r.ci_hi], "keep": bool(r.keep)} for r in lofo.itertuples()},
           "steps": sel_steps, "chosen": kept.model.iloc[0] if len(kept) else None,
           "verdict": (f"KEEP {kept.model.iloc[0]}" if len(kept) else "no model kept: raw kNN stays")
           + ("" if selection_run else " (not a selection run)"), "sizes": d["sizes"],
           "runtime_s": round(time.time() - T_START)}
    log("summary:\n" + summ.drop(columns="steps").round(4).to_string(index=False))
    log("per identity bin (unweighted Fmax per bin; testlike = weighted overall):\n"
        + pb[["method", "testlike", "all"] + ks.BINS].round(4).to_string(index=False))
    log("verdict: " + res["verdict"])

    if higo is not None and a.split == "tv" and not a.smoke and not a.no_apply:
        log("== stage 4: OOF files, refit on all TV, val/test predictions (test NOT scored)")
        for kind in kinds:
            p, _ = oof[(kind, "lofo")]
            save_sparse(out / f"pred_tv_oof_{kind}.npz", ctx.matrix(all_rows, p), d["ids"], onto)
        dv = build_split("val", onto, higo, a.higo_topk, loss=a.loss)
        dt = build_split("test", onto, higo, a.higo_topk, with_gt=False)
        ctx_v = Ctx({**dv, "nT": len(onto.terms)}, onto)
        res["val"] = {"knn": ctx_v.fmax(ctx_v.curves(np.arange(len(dv["t"])), dv["pk"]), testlike=False)["mean"]}
        for kind in kinds:
            st = a.refit_steps if a.refit_steps is not None else int(np.median(list(sel_steps[kind]["lofo"].values())))
            res["steps"][kind]["refit"] = st
            if st == 0:
                log(f"  {kind}: median lofo step 0 = the kNN prior; no refit")
                continue
            tf = time.time()
            model = train_fold(kind, d, D, None, np.arange(len(d["ids"])), None, a, 99, steps=st)
            for s_, ds in (("val", dv), ("test", dt)):
                Ds = Data(ds, dev)
                p = predict(model, Ds, np.arange(len(ds["ids"])))
                P = Ctx({**ds, "sc": None}, onto).matrix(np.arange(len(ds["t"])), p)
                save_sparse(out / f"pred_{s_}_{kind}.npz", P, ds["ids"], onto)
                if s_ == "val":
                    res["val"][kind] = ctx_v.fmax(ctx_v.sc.curves(P, dv["ids"], closed=False), testlike=False)["mean"]
                del Ds
            log(f"  {kind}: refit {st} steps, wrote pred_val/pred_test; val (random split, not selection) "
                f"{res['val'][kind]:.4f} vs kNN {res['val']['knn']:.4f}; {time.time() - tf:.0f}s")
    res["runtime_s"] = round(time.time() - T_START)
    (out / "result.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done in {res['runtime_s']}s; outputs in {out}")


if __name__ == "__main__":
    main()
