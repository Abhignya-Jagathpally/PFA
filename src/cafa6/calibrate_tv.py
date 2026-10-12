"""Post-hoc calibration of TV v2 stacker outputs: one monotone score map per (aspect, identity bin).

Why: IA-weighted Fmax (cafaeval, knn_sharp.Scorer) uses one threshold per aspect across all proteins, so a score of
0.6 must mean the same thing for a 100%-identity protein and a 40%-identity one. The kNN prior and the stacker are
fit on pooled rows and need not be on a common scale across identity bins. A monotone map inside a bin cannot change
any protein's term ranking; it only moves each bin's score scale relative to the shared per-aspect threshold.
Methods (--method picks the candidate and the map written for val/test; both are always cross-fitted and reported):
  tb   temperature + bias on the logit, q = sigmoid(logit(p) / T + b), per (aspect, bin) -- temperature scaling
       (Guo et al., ICML 2017) with a Platt-style bias (Platt 1999), weighted BCE plus a small L2 on (log 1/T, b).
  iso  (default) isotonic regression per (aspect, bin) (Zadrozny & Elkan, KDD 2002), weighted least squares,
       sklearn IsotonicRegression: linear interpolation between knots, inputs outside the fitted range clipped
       (out_of_bounds='clip'), outputs clipped to ISO_CLIP = [0.001, 0.999] so a sparse end block cannot send a
       score to 0 (dropped by the 0.01 cut) or tie everything at 1. A group with fewer than --min-bin (40) training
       proteins in a given fit keeps the identity map. Self-check (every fold and the final map, raises on failure):
       each map is non-decreasing on a 2001-point grid and inside ISO_CLIP; within every (protein, aspect) no pair
       of terms swaps order (isotonic maps are non-strict, so new ties are allowed and counted); max-propagation
       afterwards only raises scores.
Row weights for both: IA x sqrt(test-like w) on the input's nonzero (protein, term) scores of proteins that have
ground truth in the aspect (the same row weighting as stacker_tv's ia_bce loss). After the map the matrix is max-propagated over the GO graph
(knn_sharp.Ontology.propagate_max, the MCM over go_ancestors that stacker_tv uses); monotone maps keep an
ancestor-closed input closed, so this is a safety step.

Bins: ks.BINS on the best DIAMOND identity (no_hit, <30, 30-50, 50-80, 80-95, 95-99.9, 100). Per aspect, a bin with
fewer than --min-bin (40) TV v2 proteins having ground truth in that aspect is merged into its smaller adjacent
neighbour, repeatedly; the grouping is fixed on all TV (counts only) and reused in every fold and for val/test.

Cross-fitting: the 5 protein folds of stacker_tv (stacker_tv.make_folds, seed 0: components of best-hit / TV-link /
cluster links; or the stacker run's own folds.tsv when --stacker-run is given). Fold f is calibrated with maps fit on
the other four; the pooled out-of-fold predictions are scored with knn_sharp.Scorer under the chosen_v2 test-like
weights (stacker_tv.test_like_weights) and compared with the uncalibrated input by a paired protein bootstrap
(Scorer.bootstrap, 1000 reps, test-like weights).

Inputs
  --stacker-run DIR --model M   DIR/pred_tv_oof_{M}.npz (cross-fitted TV v2, stacker_tv.save_sparse format: ids, data
                                float16, indices, indptr, shape, terms; already cut at 0.01 and max-propagated) and
                                DIR/pred_{val,test}_{M}.npz (refit on all TV). M defaults to result.json "chosen";
                                M may be a letter a/b/c or fusion/twin/gated_twin. If pred_{val,test}_{M}.npz is
                                missing (the stacker's median lofo step was 0, i.e. the kNN prior) the kNN prior is
                                used for val/test and this is logged.
  --stacker-run none            kNN-only stand-in: T5_k50_g0_na prior from data/proc_gogpt_tv2/knn/{tv,val,test}_knn.npz
                                (p_full over the whole ontology), cut at 0.01 and max-propagated as stacker_tv does.
  Identity per protein: max_pident in data/proc_gogpt_tv2/knn/{split}_knn.npz. Labels: gt_tv.tsv (TV v2), gt_val.tsv.

Outputs (runs/calibrate_tv/<name>/)
  log.txt          everything printed
  summary.tsv      method x scope (all test-like, all unweighted, each identity bin) x MF/BP/CC/mean, pooled OOF
  folds_scores.tsv held-out test-like Fmax per fold and method
  thresholds.tsv   per method, aspect and bin: the bin's own optimal threshold, the global (test-like) optimal
                   threshold, Fmax at each, and the gap; before (uncal) and after (tb, iso)
  maps.tsv         per fold (and 'all' = fit on all TV): aspect, group, T, bias, rows, proteins, isotonic knots
  result.json      rule, scores, bootstrap CIs, verdict
  pred_val.npz, pred_test.npz   the input's val/test predictions with the --method map fitted on all TV,
                   max-propagated, stacker_tv.save_sparse format. Val (random split) is scored for information;
                   test is written and NEVER scored.

Decision rules: CAL_RULE (tb, registered 2026-10-12 08:44:58) and ISO_RULE (iso, registered 08:47:32, before any
stacker output existed), both in runs/test_looks.md; no test look.

Usage:
  python -m cafa6.calibrate_tv --register-rule --method iso
  python -m cafa6.calibrate_tv --method iso --stacker-run none --name knn_standin_iso
  python -m cafa6.calibrate_tv --method iso --stacker-run runs/stacker_tv/l3_c0 --name l3_c0_iso
  python -m cafa6.calibrate_tv --method tb --stacker-run runs/stacker_tv/l3_c1 --model c --name l3_c1_c_tb
  scripts/chain_calibrate.sh l3_c0        (waits for runs/stacker_tv/l3_c0/result.json, then the iso run)
"""
from __future__ import annotations

import os
import sys


def _threads() -> str:
    a = sys.argv
    return a[a.index("--threads") + 1] if "--threads" in a and a.index("--threads") + 1 < len(a) else "16"


for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _threads()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from cafa6 import knn_sharp as ks  # noqa: E402
from cafa6 import stacker_tv as st  # noqa: E402
from cafa6.data import ROOT  # noqa: E402

TV2 = ROOT / "data" / "proc_gogpt_tv2"
KNN_DIR = TV2 / "knn"
OUT_ROOT = ROOT / "runs" / "calibrate_tv"
TEST_LOOKS = ROOT / "runs" / "test_looks.md"
ORDER = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
ASP = ("MF", "BP", "CC")
EPS = 1e-4
MIN_GAIN, MAX_PLAIN_DROP = 0.003, 0.002
MODEL_ALIAS = {"a": "fusion", "b": "twin", "c": "gated_twin"}
RULE_TAG = "TV calibration (calibrate_tv.py)"
CAL_RULE = (
    "candidate = per-(aspect, identity-bin) temperature + bias on the logit (Guo et al. ICML 2017 temperature "
    "scaling with a Platt-style bias) of src/cafa6/calibrate_tv.py, fit by IA x sqrt(test-like w) weighted BCE on the "
    "input's nonzero scores, bins no_hit/<30/30-50/50-80/80-95/95-99.9/100 merged per aspect to >= 40 TV v2 proteins "
    "with ground truth, cross-fitted over the stacker_tv protein folds (5 folds, best-hit/TV-link/cluster "
    "components), followed by GO max-propagation, scored with knn_sharp.Scorer. For a given input (the kNN-only "
    "stand-in, or a stacker_tv run's model) calibration is KEPT only if (i) its pooled cross-fitted TV v2 test-like "
    f"mean Fmax minus the uncalibrated input's is >= {MIN_GAIN}, (ii) the paired protein bootstrap 95% CI of that "
    "difference (1000 reps, test-like weights) excludes 0, and (iii) the plain (unweighted) TV v2 mean Fmax does not "
    f"drop by more than {MAX_PLAIN_DROP}. The isotonic variant is an upper reference only and is never kept. A kept "
    "calibration replaces its input in any later pre-registered test look but does not by itself trigger one; "
    "smoke runs are not decision runs.")
ISO_CLIP = (1e-3, 1 - 1e-3)
ISO_RULE_TAG = "TV calibration isotonic (calibrate_tv.py --method iso)"
ISO_RULE = (
    "candidate = per-(aspect, identity-bin) isotonic regression of src/cafa6/calibrate_tv.py (--method iso): sklearn "
    "IsotonicRegression (increasing, out_of_bounds='clip', linear interpolation between knots, map outputs clipped "
    f"to [{ISO_CLIP[0]}, {ISO_CLIP[1]:.3f}]), fit by IA x sqrt(test-like w) weighted least squares on the input's nonzero "
    "scores, the same merged bins as the temperature + bias rule (no_hit/<30/30-50/50-80/80-95/95-99.9/100 merged "
    "per aspect to >= 40 TV v2 proteins with ground truth; a group with < 40 training proteins in a given fit keeps "
    "the identity map), cross-fitted over the stacker_tv protein folds (5 folds, best-hit/TV-link/cluster "
    "components), followed by GO max-propagation, scored with knn_sharp.Scorer. For a given input calibration is "
    f"KEPT only if (i) its pooled cross-fitted TV v2 test-like mean Fmax minus the uncalibrated input's is >= {MIN_GAIN}, "
    "(ii) the paired protein bootstrap 95% CI of that difference (1000 reps, test-like weights) excludes 0, and "
    f"(iii) the plain (unweighted) TV v2 mean Fmax does not drop by more than {MAX_PLAIN_DROP}. The rule applies to "
    "the stacker outputs runs/stacker_tv/{l3_c0,l3_c1,l4a} (the stacker's kept model; a model passed with --model "
    "when the stacker kept none is diagnostic only) and to the kNN prior T5_k50_g0_na alone as the fallback "
    "pipeline. The kNN stand-in isotonic result (+0.0064 [+0.0045, +0.0079], runs/calibrate_tv/knn_standin, no "
    "output clip, 08:45) was observed BEFORE this registration and is not counted as confirmation; the kNN fallback "
    "verdict under this rule is recorded but flagged as not independently confirmed. A kept calibration replaces its "
    "input in any later pre-registered test look but does not by itself trigger one. The 08:44:58 temperature + "
    "bias rule stays as registered (its candidate is --method tb).")
RULES = {"tb": (RULE_TAG, CAL_RULE), "iso": (ISO_RULE_TAG, ISO_RULE)}
T_START = time.time()
LOG: Path | None = None


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')} {time.time() - T_START:7.1f}s] {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def register_rule(method: str):
    """Appends the method's rule to runs/test_looks.md once (no scores, no test look)."""
    tag, rule = RULES[method]
    txt = TEST_LOOKS.read_text() if TEST_LOOKS.exists() else ""
    if tag in txt:
        log(f"rule '{tag}' already registered in runs/test_looks.md")
        return
    what = "calibrated score on TV v2" if method == "tb" else "stacker output is scored"
    with open(TEST_LOOKS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {tag} | PRE-REGISTERED RULE before any {what} "
                f"(no test look): {rule}\n")
    log(f"rule '{tag}' appended to runs/test_looks.md")


# ------------------------------------------------------------------------------------------------ inputs

def load_sparse(path: Path, onto) -> tuple[np.ndarray, sp.csr_matrix]:
    """Reads a stacker_tv.save_sparse file into CSR over all ontology columns."""
    z = np.load(path, allow_pickle=False)
    Pu = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"])).tocoo()
    cols = np.array([onto.idx[t] for t in z["terms"]], np.int64)
    P = sp.csr_matrix((Pu.data, (Pu.row, cols[Pu.col])), shape=(Pu.shape[0], len(onto.terms)))
    return z["ids"].astype(str), P


def knn_prior(split: str, onto) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """T5_k50_g0_na p_full over ontology columns, cut at 0.01 and max-propagated (as stacker_tv.Ctx.matrix)."""
    K = ks.load_knn(split, KNN_DIR)
    cols = np.array([onto.idx[t] for t in K["terms_full"]])
    pf = K["p_full"].tocoo()
    P = sp.csr_matrix((pf.data, (pf.row, cols[pf.col])), shape=(len(K["ids"]), len(onto.terms)))
    P.sum_duplicates()
    P.data[P.data < 0.01] = 0
    return K["ids"].astype(str), onto.propagate_max(P), K["max_pident"].astype(np.float32)


def identity(split: str) -> pd.Series:
    K = ks.load_knn(split, KNN_DIR)
    return pd.Series(K["max_pident"].astype(np.float32), index=K["ids"].astype(str))


def resolve_model(run: Path, model: str | None) -> str:
    if model is None:
        res = json.loads((run / "result.json").read_text())
        model = res.get("chosen")
        if model is None:
            raise SystemExit(f"{run}/result.json has no kept model (chosen = null); pass --model explicitly")
        log(f"--model not given: using result.json chosen = {model}")
    return MODEL_ALIAS.get(model, model)


def load_inputs(a, onto) -> dict:
    """{split: (ids, P, max_pident)} for tv, val, test, plus a description."""
    out = {}
    if a.stacker_run.lower() == "none":
        for s in ("tv", "val", "test"):
            ids, P, mx = knn_prior(s, onto)
            out[s] = (ids, P, mx)
        out["desc"] = "kNN-only stand-in T5_k50_g0_na (data/proc_gogpt_tv2/knn)"
        return out
    run = st.resolve_run(a.stacker_run)
    model = resolve_model(run, a.model)
    out["model"] = model
    ids, P = load_sparse(run / f"pred_tv_oof_{model}.npz", onto)
    out["tv"] = (ids, P, identity("tv").reindex(ids).fillna(0).values.astype(np.float32))
    for s in ("val", "test"):
        f = run / f"pred_{s}_{model}.npz"
        if f.exists():
            ids, P = load_sparse(f, onto)
            out[s] = (ids, P, identity(s).reindex(ids).fillna(0).values.astype(np.float32))
        else:
            log(f"WARNING: {f} missing (stacker median lofo step 0 = kNN prior); using the kNN prior for {s}")
            out[s] = knn_prior(s, onto)
    out["desc"] = f"{run} model {model}"
    return out


# ------------------------------------------------------------------------------------------------ groups

def merge_bins(counts: dict, min_n: int) -> list[list[str]]:
    """Adjacent bins in ORDER; the smallest group under min_n joins its smaller neighbour until all are >= min_n."""
    groups = [[b] for b in ORDER]
    n = lambda g: sum(counts.get(b, 0) for b in g)  # noqa: E731
    while len(groups) > 1:
        small = [i for i, g in enumerate(groups) if n(g) < min_n]
        if not small:
            break
        i = min(small, key=lambda j: (n(groups[j]), j))
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        j = min(nb, key=lambda k: (n(groups[k]), k))
        lo, hi = sorted((i, j))
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
    return groups


class Groups:
    """Per-aspect bin -> group index; group names like '<30+30-50'."""

    def __init__(self, bins_tv: np.ndarray, has_gt: np.ndarray, min_n: int):
        self.map, self.names, self.counts = [], [], []
        for k in range(3):
            c = pd.Series(bins_tv[has_gt[:, k]]).value_counts().to_dict()
            gs = merge_bins(c, min_n)
            self.map.append({b: gi for gi, g in enumerate(gs) for b in g})
            self.names.append(["+".join(g) for g in gs])
            self.counts.append([int(sum(c.get(b, 0) for b in g)) for g in gs])
        self.ng = max(len(x) for x in self.names)

    def of(self, bins: np.ndarray) -> np.ndarray:
        """[n, 3] group index per protein and aspect."""
        return np.stack([pd.Series(bins).map(self.map[k]).values for k in range(3)], 1).astype(np.int64)


# ------------------------------------------------------------------------------------------------ maps

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def fit_platt(x, y, w, ridge):
    """q = sigmoid(exp(u) * x + b) minimising mean w-weighted BCE + ridge * (u^2 + b^2); returns (T, b)."""
    W = w.sum()

    def f(th):
        a, b = np.exp(th[0]), th[1]
        z = a * x + b
        r = w * (1.0 / (1.0 + np.exp(-z)) - y)
        loss = (w * (np.logaddexp(0, z) - y * z)).sum() / W + ridge * (th @ th)
        g = np.array([(r * x).sum() * a / W, r.sum() / W]) + 2 * ridge * th
        return loss, g

    r = minimize(f, np.zeros(2), jac=True, method="L-BFGS-B")
    return float(np.exp(-r.x[0])), float(r.x[1])


class Entries:
    """Nonzero (protein, term) scores of a split with aspect, group key, label and fit weight."""

    def __init__(self, P: sp.csr_matrix, ids, grp: np.ndarray, onto, ng: int, sc=None, w_prot=None):
        P = P.tocsr()
        P.eliminate_zeros()
        self.shape, self.ids = P.shape, ids
        self.row = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        self.col = P.indices.astype(np.int64)
        self.v = P.data.astype(np.float64)
        self.asp = onto.ns[self.col].astype(np.int64)
        self.key = self.asp * ng + grp[self.row, self.asp]
        if sc is not None:
            gi = np.array([sc.pos.get(p, -1) for p in ids], np.int64)
            g = gi[self.row]
            ok = g >= 0
            self.y = (np.isin(np.where(ok, g * len(onto.terms) + self.col, -1), sc.gkey) & ok).astype(np.float64)
            scored = ok.copy()
            scored[ok] = sc.in_ns[self.asp[ok], g[ok]]
            self.fw = scored * onto.ia[self.col] * np.sqrt(w_prot)[self.row]

    def matrix(self, v: np.ndarray, onto) -> sp.csr_matrix:
        v = np.asarray(v, np.float32)
        k = v >= 0.01
        P = sp.csr_matrix((v[k], (self.row[k], self.col[k])), shape=self.shape)
        return onto.propagate_max(P)


def fit_maps(E: Entries, prot_mask: np.ndarray, nkeys: int, ridge: float, min_prot: int):
    """Per group key: (T, b) and an isotonic map, from entries of proteins in prot_mask. A group with fewer than
    min_prot training proteins (with fit weight) keeps the identity map for isotonic; (T, b) needs 20 rows."""
    T, B, n_rows, n_prot, isos = np.ones(nkeys), np.zeros(nkeys), np.zeros(nkeys, int), np.zeros(nkeys, int), {}
    m = prot_mask[E.row] & (E.fw > 0)
    order = np.argsort(E.key[m], kind="stable")
    km, idx = E.key[m][order], np.where(m)[0][order]
    bounds = np.searchsorted(km, np.arange(nkeys + 1))
    for k in range(nkeys):
        sel = idx[bounds[k]:bounds[k + 1]]
        n_rows[k], n_prot[k] = len(sel), len(np.unique(E.row[sel]))
        if len(sel) < 20 or E.y[sel].sum() == 0 or E.y[sel].sum() == len(sel):
            continue
        T[k], B[k] = fit_platt(logit(E.v[sel]), E.y[sel], E.fw[sel], ridge)
        if n_prot[k] >= min_prot:
            isos[k] = IsotonicRegression(y_min=ISO_CLIP[0], y_max=ISO_CLIP[1], increasing=True,
                                         out_of_bounds="clip").fit(E.v[sel], E.y[sel], sample_weight=E.fw[sel])
    return {"T": T, "B": B, "rows": n_rows, "prots": n_prot, "iso": isos}


def apply_maps(E: Entries, maps: dict, method: str, sel: np.ndarray | None = None) -> np.ndarray:
    """Calibrated values of E's entries (only entries in sel are changed when given)."""
    v = E.v.copy()
    sel = np.ones(len(v), bool) if sel is None else sel
    if method == "tb":
        k = E.key[sel]
        z = logit(v[sel]) / maps["T"][k] + maps["B"][k]
        v[sel] = 1.0 / (1.0 + np.exp(-z))
    elif method == "iso":
        for k, ir in maps["iso"].items():
            s = sel & (E.key == k)
            if s.any():
                v[s] = np.clip(ir.predict(v[s]), *ISO_CLIP)
    return v


def self_check(E: Entries, v_new: np.ndarray, maps: dict, onto, label: str) -> dict:
    """Raises unless (1) every isotonic map is non-decreasing on a fine grid and stays inside ISO_CLIP, (2) within
    each (protein, aspect) -- one bin, one map -- no pair of terms swaps order (ties may appear: isotonic maps are
    non-strict), and (3) max-propagation of the calibrated matrix only adds or raises ancestor scores."""
    grid = np.linspace(0, 1, 2001)
    for k, ir in maps["iso"].items():
        g = np.clip(ir.predict(grid), *ISO_CLIP)
        if (np.diff(g) < -1e-12).any() or g.min() < ISO_CLIP[0] - 1e-12 or g.max() > ISO_CLIP[1] + 1e-12:
            raise AssertionError(f"[{label}] isotonic map {k} not monotone / outside the clip")
    o = np.lexsort((E.v, E.asp, E.row))
    same = (E.row[o][1:] == E.row[o][:-1]) & (E.asp[o][1:] == E.asp[o][:-1])
    dv_old, dv_new = np.diff(E.v[o]), np.diff(v_new[o])
    inv = same & (dv_old > 0) & (dv_new < -1e-9)
    if inv.any():
        raise AssertionError(f"[{label}] {int(inv.sum())} within-protein ranking inversions")
    ties = int((same & (dv_old > 0) & (np.abs(dv_new) <= 1e-9)).sum())
    P0 = E.matrix(v_new, onto)
    k = v_new >= 0.01
    lk = sp.csr_matrix((v_new[k].astype(np.float32), (E.row[k], E.col[k])), shape=E.shape)
    dd = (P0 - lk).tocsr().data
    if (dd < -1e-6).any():
        raise AssertionError(f"[{label}] propagation lowered {int((dd < -1e-6).sum())} scores")
    rep = {"maps_checked": len(maps["iso"]), "adjacent_pairs": int((same & (dv_old > 0)).sum()),
           "inversions": 0, "new_ties": ties, "entries_raised_by_propagation": int((dd > 1e-6).sum())}
    log(f"  self-check [{label}]: OK " + json.dumps(rep))
    return rep


# ------------------------------------------------------------------------------------------------ scoring helpers

def fcurve(sc: ks.Scorer, cur: dict, w: np.ndarray) -> dict:
    """Per-aspect F over ks.TAU with the arithmetic of Scorer.fmax (checked against it), to read off thresholds."""
    out, ref = {}, sc.fmax(cur, weights=w)
    for a, c in cur.items():
        ww = w[c["idx"]]
        ne = ww.sum()
        if ne <= 0:
            out[a] = None
            continue
        nh = ww @ c["has"]
        P = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
        R = (ww @ c["rec"]) / ne
        F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
        F[nh <= 0] = 0
        assert abs(F.max() - ref[a]) < 1e-12, (a, F.max(), ref[a])
        out[a] = F
    return out


def masked(w: np.ndarray, sc: ks.Scorer, gi_sel: np.ndarray) -> np.ndarray:
    m = np.zeros(sc.n)
    m[gi_sel[gi_sel >= 0]] = 1
    return w * m


# ------------------------------------------------------------------------------------------------ main

def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stacker-run", default="none", help="runs/stacker_tv/<run>, or none for the kNN-only stand-in")
    ap.add_argument("--model", default=None, help="a/b/c or fusion/twin/gated_twin; default result.json chosen")
    ap.add_argument("--name", default=None, help="output name under runs/calibrate_tv/ (default from the inputs)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bin", type=int, default=40, help="merge bins with fewer TV proteins (per aspect)")
    ap.add_argument("--ridge", type=float, default=1e-3, help="L2 on (log 1/T, bias) relative to mean BCE")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="200 bootstrap reps; not a decision run")
    ap.add_argument("--method", default="iso", choices=("tb", "iso"),
                    help="candidate map: tb = temperature + bias on the logit, iso = isotonic (default)")
    ap.add_argument("--register-rule", action="store_true", help="append --method's rule to runs/test_looks.md, exit")
    a = ap.parse_args()
    if a.threads > 16:
        ap.error("--threads must be <= 16 (shared CPU)")
    ks.use_tv("v2")
    if a.register_rule:
        register_rule(a.method)
        return
    if a.smoke:
        a.boot = min(a.boot, 200)
    name = a.name or (("knn_standin" if a.stacker_run.lower() == "none"
                       else Path(a.stacker_run).name + (f"_{a.model}" if a.model else "")) + f"_{a.method}")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "log.txt"
    LOG.write_text("")
    st.LOG = LOG
    np.random.seed(a.seed)
    decision = not a.smoke
    cand = a.method
    log(f"== calibrate_tv  method={cand}  stacker_run={a.stacker_run}  model={a.model}  out={out}  "
        f"decision_run={decision}")
    log("config " + json.dumps(vars(a)))
    register_rule(cand)

    log("== stage 1: inputs")
    t0 = time.time()
    onto = ks.Ontology()
    inp = load_inputs(a, onto)
    for s in ("tv", "val", "test"):
        ids, P, mx = inp[s]
        log(f"  {s}: {len(ids)} proteins x {P.shape[1]} terms, nnz {P.nnz} ({P.nnz / len(ids):.0f}/protein), "
            f"has_hit {(mx > 0).mean():.3f}")
    log(f"  input: {inp['desc']}  ({time.time() - t0:.1f}s)")

    log("== stage 2: labels, test-like weights, folds, bin groups")
    t0 = time.time()
    ids, P_tv, mx_tv = inp["tv"]
    sc = ks.Scorer(onto, ks.gt_frame("tv"))
    w_sc, wrep = st.test_like_weights("tv", sc)
    gi = np.array([sc.pos.get(p, -1) for p in ids], np.int64)
    w_ids = np.where(gi >= 0, w_sc[np.maximum(gi, 0)], 0.0)
    has_gt = np.stack([np.where(gi >= 0, sc.in_ns[k, np.maximum(gi, 0)], False) for k in range(3)], 1)
    log(f"  scorer: {sc.n} TV v2 proteins with ground truth ({len(ids)} in input); test-like effective n "
        f"{wrep['effective_n']}; with GT per aspect MF/BP/CC {has_gt.sum(0).tolist()}")
    d = {"ids": ids, "split": "tv", "w": w_ids}
    fold = st.make_folds(d, a.folds, a.seed)
    if a.stacker_run.lower() != "none":
        ff = st.resolve_run(a.stacker_run) / "folds.tsv"
        if ff.exists():
            sf = pd.read_csv(ff, sep="\t").set_index("id").fold.reindex(ids)
            agree = float((sf.values == fold).mean())
            log(f"  stacker folds.tsv agreement with make_folds: {agree:.4f}; using the stacker's folds")
            fold = sf.fillna(-1).values.astype(np.int64)
    bins_tv = ks.ident_bins(mx_tv).values
    G = Groups(bins_tv, has_gt, a.min_bin)
    for k in range(3):
        log(f"  {ASP[k]} groups: " + ", ".join(f"{n} (n={c})" for n, c in zip(G.names[k], G.counts[k])))
    nkeys = 3 * G.ng
    grp_tv = G.of(bins_tv)
    E = Entries(P_tv, ids, grp_tv, onto, G.ng, sc, w_ids)
    log(f"  entries: {len(E.v)} ({int((E.fw > 0).sum())} with fit weight > 0, positive rate "
        f"{E.y[E.fw > 0].mean():.4f}); per fold n {np.bincount(fold[fold >= 0], minlength=a.folds).tolist()} "
        f"({time.time() - t0:.1f}s)")

    def gname(k):
        asp, g = divmod(k, G.ng)
        return ASP[asp], (G.names[asp][g] if g < len(G.names[asp]) else None)

    log("== stage 3: cross-fitted maps")
    methods = ["uncal", "tb", "iso"]
    oof = {m: E.v.copy() for m in methods}
    map_rows, fold_rows = [], []
    for f in range(a.folds):
        tf = time.time()
        tr, te = fold != f, fold == f
        maps = fit_maps(E, tr, nkeys, a.ridge, a.min_bin)
        sel = te[E.row]
        for m in methods[1:]:
            oof[m][sel] = apply_maps(E, maps, m, sel)[sel]
        if cand == "iso":
            self_check(E, np.where(sel, oof["iso"], E.v), maps, onto, f"fold {f}")
        for k in range(nkeys):
            asp, gn = gname(k)
            if gn is None:
                continue
            map_rows.append({"fold": f, "aspect": asp, "group": gn, "T": maps["T"][k], "bias": maps["B"][k],
                             "rows": int(maps["rows"][k]), "proteins": int(maps["prots"][k]),
                             "iso_fitted": k in maps["iso"],
                             "iso_knots": len(maps["iso"][k].X_thresholds_) if k in maps["iso"] else 0})
        msg = "; ".join(f"{r['aspect']} {r['group']}: T={r['T']:.2f} b={r['bias']:+.2f} "
                        f"iso={'%d knots' % r['iso_knots'] if r['iso_fitted'] else 'identity'}"
                        for r in map_rows if r["fold"] == f)
        log(f"  fold {f}: fit on {int(tr.sum())} proteins, applied to {int(te.sum())} ({time.time() - tf:.1f}s)\n"
            f"    {msg}")
        wf = masked(w_sc, sc, gi[te])
        row = {"fold": f, "n_test": int(te.sum())}
        for m in methods:
            P = E.matrix(np.where(sel, oof[m], 0.0), onto)
            row[m] = sc.fmax(sc.curves(P, ids), weights=wf)["mean"]
        fold_rows.append(row)
        log(f"    held-out test-like Fmax: " + "  ".join(f"{m} {row[m]:.4f}" for m in methods))

    log("== stage 4: pooled OOF scores, bootstrap, thresholds")
    t0 = time.time()
    curs = {m: sc.curves(E.matrix(oof[m], onto), ids) for m in methods}
    ones = np.ones(sc.n)
    bin_sc = ks.ident_bins(identity("tv").reindex(sc.ids).fillna(0).values).values
    summ = []
    for m in methods:
        for scope, w in [("all_testlike", w_sc), ("all_unweighted", ones)] + \
                [(f"bin {b}", (bin_sc == b).astype(float)) for b in ORDER]:
            fm = sc.fmax(curs[m], weights=w)
            summ.append({"method": m, "scope": scope, "n": int((w > 0).sum()),
                         **{x: fm.get(x, np.nan) for x in ASP}, "mean": fm["mean"]})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    piv = summ.pivot(index="scope", columns="method", values="mean").reindex(
        ["all_testlike", "all_unweighted"] + [f"bin {b}" for b in ORDER])[methods]
    log("pooled OOF mean Fmax (bins unweighted):\n" + piv.round(4).to_string())
    boots = {}
    for m in methods[1:]:
        boots[m] = sc.bootstrap(curs[m], curs["uncal"], reps=a.boot, seed=a.seed, weights=w_sc)
        log(f"  {m} - uncal test-like: {boots[m]['diff']:+.4f} [{boots[m]['lo']:+.4f}, {boots[m]['hi']:+.4f}] "
            f"({a.boot} reps)")

    th_rows = []
    for m in methods:
        Fg = fcurve(sc, curs[m], w_sc)
        for x in ASP:
            if Fg[x] is None:
                continue
            tg = int(np.argmax(Fg[x]))
            th_rows.append({"method": m, "aspect": x, "bin": "all (test-like)", "n": int((w_sc > 0).sum()),
                            "tau_bin": ks.TAU[tg], "tau_global": ks.TAU[tg], "F_at_bin_tau": Fg[x][tg],
                            "F_at_global_tau": Fg[x][tg], "gap": 0.0})
            for b in ORDER:
                wb = w_sc * (bin_sc == b)
                if wb.sum() <= 0:
                    wb = (bin_sc == b).astype(float)
                if wb.sum() <= 0:
                    continue
                Fb = fcurve(sc, curs[m], wb)[x]
                if Fb is None:
                    continue
                tb = int(np.argmax(Fb))
                th_rows.append({"method": m, "aspect": x, "bin": b, "n": int((bin_sc == b).sum()),
                                "tau_bin": ks.TAU[tb], "tau_global": ks.TAU[tg], "F_at_bin_tau": Fb[tb],
                                "F_at_global_tau": Fb[tg], "gap": Fb[tb] - Fb[tg]})
    th = pd.DataFrame(th_rows)
    th.to_csv(out / "thresholds.tsv", sep="\t", index=False)
    tb = th[th.bin != "all (test-like)"].copy()
    tb["key"] = tb.aspect + " " + tb.bin
    show = tb.pivot(index="key", columns="method", values="tau_bin")[methods].join(
        tb.pivot(index="key", columns="method", values="gap")[methods], rsuffix="_gap")
    log("per-bin optimal threshold (test-like within bin) and F lost by using the global threshold:\n"
        + show.round(3).to_string())
    gl = th[th.bin == "all (test-like)"].pivot(index="aspect", columns="method", values="tau_global")[methods]
    log("global test-like optimal threshold per aspect:\n" + gl.to_string())
    log(f"  stage 4 took {time.time() - t0:.1f}s")

    si = summ.set_index(["method", "scope"])
    plain_u = float(piv.loc["all_unweighted", "uncal"])
    plain_p = float(piv.loc["all_unweighted", cand])
    b = boots[cand]
    keep = bool(b["diff"] >= MIN_GAIN and b["lo"] > 0 and plain_p - plain_u >= -MAX_PLAIN_DROP)
    knn_fallback = a.stacker_run.lower() == "none"
    verdict = (f"KEEP calibration ({cand})" if keep else "calibration not kept: uncalibrated input stays") + \
        ("" if decision else " (smoke, not a decision run)") + \
        (" (kNN fallback: observed before the isotonic registration, not independently confirmed)"
         if knn_fallback and cand == "iso" else "")
    log(f"rule: gain {b['diff']:+.4f} (>= {MIN_GAIN}?), CI lo {b['lo']:+.4f} (> 0?), plain change "
        f"{plain_p - plain_u:+.4f} (>= -{MAX_PLAIN_DROP}?) -> {verdict}")

    log(f"== stage 5: fit on all TV ({cand}), calibrate val/test (test NOT scored)")
    t0 = time.time()
    maps = fit_maps(E, np.ones(len(ids), bool), nkeys, a.ridge, a.min_bin)
    checks = {}
    if cand == "iso":
        checks["tv_all"] = self_check(E, apply_maps(E, maps, "iso"), maps, onto, "all-TV map on TV")
    for k in range(nkeys):
        asp, gn = gname(k)
        if gn is not None:
            map_rows.append({"fold": "all", "aspect": asp, "group": gn, "T": maps["T"][k], "bias": maps["B"][k],
                             "rows": int(maps["rows"][k]), "proteins": int(maps["prots"][k]),
                             "iso_fitted": k in maps["iso"],
                             "iso_knots": len(maps["iso"][k].X_thresholds_) if k in maps["iso"] else 0})
    mp_all = pd.DataFrame([r for r in map_rows if r["fold"] == "all"])
    log("maps fitted on all TV:\n" + mp_all.round(3).to_string(index=False))
    pd.DataFrame(map_rows).to_csv(out / "maps.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "folds_scores.tsv", sep="\t", index=False)
    val_scores = {}
    for s in ("val", "test"):
        ids_s, P_s, mx_s = inp[s]
        Es = Entries(P_s, ids_s, G.of(ks.ident_bins(mx_s).values), onto, G.ng)
        vc = apply_maps(Es, maps, cand)
        if cand == "iso":
            checks[s] = self_check(Es, vc, maps, onto, f"all-TV map on {s}")
        Pc = Es.matrix(vc, onto)
        st.save_sparse(out / f"pred_{s}.npz", Pc, ids_s, onto)
        log(f"  {s}: wrote pred_{s}.npz ({Pc.shape[0]} proteins, nnz {Pc.nnz})")
        if s == "val":
            scv = ks.Scorer(onto, ks.gt_frame("val"))
            val_scores = {"uncal": scv.fmax(scv.curves(P_s, ids_s))["mean"],
                          cand: scv.fmax(scv.curves(Pc, ids_s))["mean"]}
            log(f"  val (random split, information only, unweighted): uncal {val_scores['uncal']:.4f}  "
                f"{cand} {val_scores[cand]:.4f}")
    log(f"  stage 5 took {time.time() - t0:.1f}s")

    res = {"input": inp["desc"], "stacker_run": a.stacker_run, "model": inp.get("model"), "decision_run": decision,
           "method": cand, "rule": RULES[cand][1], "rule_tag": RULES[cand][0], "groups": {ASP[k]: dict(zip(G.names[k], G.counts[k])) for k in range(3)},
           "tv_testlike": {m: float(piv.loc["all_testlike", m]) for m in methods},
           "tv_unweighted": {m: float(piv.loc["all_unweighted", m]) for m in methods},
           "tv_testlike_aspects": {m: {x: float(si.loc[(m, "all_testlike"), x]) for x in ASP} for m in methods},
           "self_checks": checks, "knn_fallback_not_confirmation": knn_fallback and cand == "iso",
           "bootstrap_vs_uncal": boots, "per_fold": fold_rows, "val_unweighted": val_scores,
           "keep": keep and decision, "verdict": verdict, "weights": {"effective_n": wrep["effective_n"]},
           "runtime_s": round(time.time() - T_START)}
    (out / "result.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done in {res['runtime_s']}s; outputs in {out}")


if __name__ == "__main__":
    main()
