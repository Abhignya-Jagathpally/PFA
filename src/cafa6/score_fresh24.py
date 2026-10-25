"""Scoring of fresh24 look #1 exactly as registered in runs/fresh24_looks.md (registration 2026-10-19 18:40:00).

Sets (ground truth from data/fresh24; terms are unpropagated there, the scorer closes them)
  FR24-P            primary: no-knowledge proteins (gt_nk.tsv) with hpa_only == False in proteins.tsv
  NK-all            all no-knowledge proteins, HPA bulk load included
  LK                limited-knowledge proteins (gt_lk.tsv)
  FR24-P-noloads    FR24-P without the T. brucei HTP localisation load (rows of a Trypanosoma brucei protein with
                    evidence HTP assigned by UniProt) and the SynGO load (rows assigned by SynGO alone), taken from
                    gt_annotations_full.tsv; (protein, aspect) pairs left without ground truth drop out
  FR24-P-GOcurrent  FR24-P proteins scored against gt_nk_go_current.tsv with data/fresh24/ref/go-basic_current.obo
                    (predictions mapped to that release by Ontology.canon, IA from the same IA.txt; terms without
                    an IA value weigh 0 in the weighted metrics)
Every set is also broken down by identity bin of the nearest GO-GPT train+val protein (proteins.tsv bin) and by
organism group (cafa6.temporal_val.org_group: human, mouse, rat, zebrafish, other; plus, descriptively, every
organism with at least 100 proteins in the set).

Metrics (cafaeval protocol: prop=max, norm=cafa, tau = 0.01..0.99; scores cut at 0.01, top 1,500 per protein,
rounded to 3 decimals, max-propagated; GO runs/gogpt_cafa5/ref/go-basic_gogpt.obo, IA runs/gogpt_cafa5/ref/IA.txt;
the per-protein sums of cafa6.eval_suite.build_stats without the AUPR levels)
  Fmax_w  IA-weighted Fmax (headline)      Fmax  unweighted Fmax      Smin_w  IA-based S-min
  per aspect and their mean; F1 / F1_w per protein at the method's own Fmax / Fmax_w threshold (eval_suite).
  A protein without predictions in an aspect has recall 0 and no precision term (cafaeval).
  DeepGO-SE is scored on MF and BP only (its released CC models are faulty); its mean is not defined, the
  column Fmax_w_mean_MFBP gives the MF+BP mean for every row.

Statistics
  Paired protein bootstrap, 1,000 reps (seed 0), the same draws for every method, metric, bin and group; percentile
  95% intervals. Paired permutation (swap) tests, 10,000 reps (seed 1), Ontologue minus each method on
  {Fmax_w, Fmax, Smin_w} x {MF, BP, CC, mean} on FR24-P; Holm-Bonferroni over that whole family (every registered
  method other than Ontologue that has predictions; DeepGO-SE contributes MF and BP). Secondary sets: descriptive.

Outputs: runs/fresh24/{summary.tsv, per_bin.tsv, per_organism.tsv, pvalues.tsv, report.md}

  PYTHONPATH=src ../bin/python -m cafa6.score_fresh24 [--perm-reps 10000] [--sets FR24-P,...]
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("CAFA6_THREADS", "32")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from cafa6 import knn_sharp as ks  # noqa: E402
from cafa6.compare_paper import load_npz_pred  # noqa: E402
from cafa6.temporal_val import org_group  # noqa: E402

ROOT = ks.ROOT
FRESH = ROOT / "data" / "fresh24"
OUT = ROOT / "runs" / "fresh24"
PREDS = OUT / "preds"
ASPECTS = ("MF", "BP", "CC")
COLS = ASPECTS + ("mean",)
TAU = ks.TAU
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
B_REPS, B_SEED = 1000, 0
P_REPS, P_SEED = 10000, 1
REF = "ontologue"
TESTED = ("Fmax_w", "Fmax", "Smin_w")
KEYS = ("prec", "has", "rec", "tp", "fp", "fn")
SETS = ("FR24-P", "NK-all", "LK", "FR24-P-noloads", "FR24-P-GOcurrent")
METHODS = [
    ("ontologue", "Ontologue (served predictor)", ASPECTS),
    ("ontologue_knn", "Ontologue step 1a: kNN prior (train+val refs)", ASPECTS),
    ("ontologue_higo", "Ontologue step 1b: HiGO head alone", ASPECTS),
    ("blast_knn", "BLAST-KNN (training refs)", ASPECTS),
    ("blast_knn_trainval", "BLAST-KNN (train+val refs)", ASPECTS),
    ("diamondscore", "DIAMONDScore", ASPECTS),
    ("naive", "Naive", ASPECTS),
    ("gogpt_sample10", "GO-GPT ten-sample", ASPECTS),
    ("gogpt_beam", "GO-GPT beam", ASPECTS),
    ("sprof_go", "SPROF-GO", ASPECTS),
    ("deepgoplus", "DeepGOPlus 1.0.10", ASPECTS),
    ("proteinfer_e5", "ProteInfer (5-model ensemble)", ASPECTS),
    ("deepfri_cnn", "DeepFRI sequence CNN", ASPECTS),
    ("deepgose", "DeepGO-SE (MF, BP only)", ("MF", "BP")),
    ("bioreason_rl", "BioReason-Pro RL", ASPECTS),
    ("forecast_F1_ens", "Forecasting model F1 (3-seed ensemble)", ASPECTS),
    ("forecast_round2", "Forecasting model, round 2 (F1-650M ensemble)", ASPECTS),
]
# look #3 (runs/fresh24_looks.md, 2026-10-25 01:16): reference and comparison set
LOOK3_REF = "forecast_round2"
LOOK3_COMPARE = ("ontologue", "forecast_F1_ens", "sprof_go", "gogpt_sample10")
# look #2 (runs/fresh24_looks.md, 2026-10-24 22:45): reference and comparison set
LOOK2_REF = "forecast_F1_ens"
LOOK2_COMPARE = ("ontologue", "ontologue_knn", "gogpt_sample10", "sprof_go")
LABEL = {k: l for k, l, _ in METHODS}
ASP_OF = {k: a for k, _, a in METHODS}
# GO-GPT test rows of the same methods (runs/compare_paper/summary.tsv S3_all; components from runs/eval_suite)
TEST_KEY = {"ontologue": "higo_r2", "gogpt_sample10": "gogpt_sample10", "gogpt_beam": "gogpt_beam",
            "blast_knn": "blast_knn", "blast_knn_trainval": "blast_knn_trainval", "diamondscore": "diamondscore",
            "naive": "naive", "sprof_go": "sprof_go", "deepgoplus": "deepgoplus", "proteinfer_e5": "proteinfer_e5",
            "deepfri_cnn": "deepfri_cnn", "deepgose": "deepgose", "bioreason_rl": "bioreason_rl"}
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ data

def ground_truth() -> dict:
    """{set: (gt frame [id, term], ontology key)}."""
    p = pd.read_csv(FRESH / "proteins.tsv", sep="\t")
    prim = set(p.accession[~p.hpa_only.astype(bool)])
    nk = pd.read_csv(FRESH / "gt_nk.tsv", sep="\t")
    lk = pd.read_csv(FRESH / "gt_lk.tsv", sep="\t")
    nkc = pd.read_csv(FRESH / "gt_nk_go_current.tsv", sep="\t")
    full = pd.read_csv(FRESH / "gt_annotations_full.tsv", sep="\t")
    full = full.merge(p[["accession", "organism"]], left_on="protein", right_on="accession", how="left")
    tb = full.organism.fillna("").str.startswith("Trypanosoma brucei") & (full.evidence == "HTP") & \
        (full.assigned_by == "UniProt")
    syn = full.assigned_by == "SynGO"
    nol = full[(full.split == "nk") & full.protein.isin(prim) & ~tb & ~syn]
    fr = lambda d: d.rename(columns={"protein": "id"})[["id", "term"]].drop_duplicates()
    return {"FR24-P": (fr(nk[nk.protein.isin(prim)]), "gogpt"), "NK-all": (fr(nk), "gogpt"), "LK": (fr(lk), "gogpt"),
            "FR24-P-noloads": (fr(nol), "gogpt"),
            "FR24-P-GOcurrent": (fr(nkc[nkc.protein.isin(prim)]), "current")}


def processed(sc: ks.Scorer, ids, P: sp.csr_matrix, chunk: int = 2000) -> sp.csr_matrix:
    """Submission cut/round + max-propagation (as Scorer.curves), rows at the scorer's protein positions."""
    ids = np.asarray(ids)
    m = np.where([p in sc.pos for p in ids])[0]
    P = P.tocsr()[m]
    gi = np.array([sc.pos[p] for p in ids[m]], dtype=np.int64)
    parts = []
    for s in range(0, P.shape[0], chunk):
        parts.append(sc.o.propagate_max(ks.to_submission(P[s:s + chunk])))
    Q = (sp.vstack(parts) if parts else sp.csr_matrix((0, len(sc.o.terms)))).tocoo()
    return sp.csr_matrix((Q.data, (gi[Q.row], Q.col)), shape=(sc.n, len(sc.o.terms)))


def build_stats(sc: ks.Scorer, S: sp.csr_matrix) -> dict:
    """eval_suite.build_stats without the AUPR level arrays: per aspect and weighting, [n_a, 99] per-protein sums."""
    o = sc.o
    nT = len(o.terms)
    S = S.tocoo()
    r, c, v = S.row.astype(np.int64), S.col.astype(np.int64), S.data.astype(np.float64)
    ing = np.isin(r * nT + c, sc.gkey)
    kk = np.searchsorted(TAU, v, side="right")
    grows = np.repeat(np.arange(sc.n), np.diff(sc.G.indptr))
    gcols = sc.G.indices
    out = {}
    for k, a in enumerate(ASPECTS):
        idx = np.where(sc.in_ns[k])[0]
        nk = len(idx)
        loc = np.full(sc.n, -1, np.int64)
        loc[idx] = np.arange(nk)
        sel = (o.ns[c] == k) & (loc[r] >= 0)
        rl, kq, g, cc = loc[r[sel]], kk[sel], ing[sel], c[sel]
        gm = o.ns[gcols] == k
        st = {"idx": idx}
        for wn in ("u", "w"):
            wt = np.ones(len(cc)) if wn == "u" else o.ia[cc]
            gwt = np.ones(gm.sum()) if wn == "u" else o.ia[gcols[gm]]
            G = np.bincount(loc[grows[gm]], weights=gwt, minlength=nk)
            pos = wt > 0
            C = np.bincount(rl[pos] * 100 + kq[pos], weights=wt[pos], minlength=nk * 100).reshape(nk, 100).astype(np.float64)
            I = np.bincount(rl[pos] * 100 + kq[pos], weights=(wt * g)[pos], minlength=nk * 100).reshape(nk, 100).astype(np.float64)
            W = np.cumsum(C[:, ::-1], axis=1)[:, ::-1][:, 1:]
            T = np.cumsum(I[:, ::-1], axis=1)[:, ::-1][:, 1:]
            has = W > 0
            st[wn] = {"prec": np.divide(T, W, out=np.zeros_like(W), where=has), "has": has.astype(np.float64),
                      "rec": np.divide(T, G[:, None], out=np.zeros_like(T), where=G[:, None] > 0),
                      "tp": T, "fp": W - T, "fn": G[:, None] - T}
        out[a] = st
    return out


def first_best(x, maximize=True):
    b = x.max(-1, keepdims=True) if maximize else x.min(-1, keepdims=True)
    ok = (x >= b - 1e-12) if maximize else (x <= b + 1e-12)
    return ok.argmax(-1)


def metrics_from(s: dict) -> dict:
    """eval_suite.metrics_from restricted to Fmax, Smin, their thresholds and coverage."""
    ne = s["ne"][:, None]
    covu = s["u"]["has"] > 0
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for wn, suf in (("u", ""), ("w", "_w")):
            d = s[wn]
            P = np.where(d["has"] > 0, d["prec"] / np.where(d["has"] > 0, d["has"], 1), 0.0)
            R = d["rec"] / np.where(ne > 0, ne, 1)
            F = np.where(P + R > 0, 2 * P * R / np.where(P + R > 0, P + R, 1), 0.0)
            F = np.where(covu, F, 0.0)
            out["Fmax" + suf] = F.max(-1)
            jb = first_best(F)
            out["tau" + suf] = TAU[jb]
            out["cov" + suf] = np.take_along_axis(d["has"], jb[:, None], 1)[:, 0] / np.where(ne[:, 0] > 0, ne[:, 0], 1)
            Sv = np.sqrt((d["fn"] / np.where(ne > 0, ne, 1)) ** 2 + (d["fp"] / np.where(ne > 0, ne, 1)) ** 2)
            Sv = np.where(covu, Sv, np.inf)
            out["Smin" + suf] = np.where(np.isfinite(Sv.min(-1)), Sv.min(-1), np.nan)
    empty = s["ne"] <= 0
    for k in out:
        out[k] = np.where(empty, np.nan, out[k])
    return out


MET = ("Fmax_w", "Fmax", "Smin_w", "Smin", "tau_w", "tau", "cov_w")


def evaluate(st: dict, W: np.ndarray, mask: np.ndarray | None = None, aspects=ASPECTS) -> dict:
    """W [B, n] protein weights; mask [n] bool restricts to a protein group. -> {aspect: {metric: [B]}} + mean."""
    res = {}
    for a in ASPECTS:
        idx = st[a]["idx"]
        sel = np.arange(len(idx)) if mask is None else np.where(mask[idx])[0]
        Wk = W[:, idx[sel]]
        s = {"ne": Wk.sum(1), **{wn: {k: Wk @ st[a][wn][k][sel] for k in KEYS} for wn in ("u", "w")}}
        res[a] = metrics_from(s)
        if a not in aspects:
            res[a] = {k: np.full(W.shape[0], np.nan) for k in res[a]}
    res["mean"] = {m: np.mean([res[a][m] for a in ASPECTS], axis=0) for m in MET}
    res["mean_MFBP"] = {m: np.mean([res[a][m] for a in ("MF", "BP")], axis=0) for m in MET}
    return res


def per_protein_f1(st: dict, point: dict, n: int, aspects=ASPECTS) -> dict:
    """{wn: [n, 3]} F1 per protein and aspect at the method's own Fmax (u) / Fmax_w (w) threshold; NaN where the
    protein has no ground truth in the aspect."""
    out = {}
    for wn, suf in (("u", ""), ("w", "_w")):
        F = np.full((n, 3), np.nan)
        for k, a in enumerate(ASPECTS):
            if a not in aspects or not np.isfinite(point[a]["tau" + suf][0]):
                continue
            j = int(round(point[a]["tau" + suf][0] * 100)) - 1
            d = st[a][wn]
            p = np.where(d["has"][:, j] > 0, d["prec"][:, j], 0.0)
            r = d["rec"][:, j]
            F[st[a]["idx"], k] = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
        out[wn] = F
    return out


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[o]))
    out = np.empty(m)
    out[o] = adj
    return out


def perm_p(obs, null):
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--perm-reps", type=int, default=P_REPS)
    ap.add_argument("--boot-reps", type=int, default=B_REPS)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--look2", action="store_true", help="registered look #2: forecasting model vs 4 methods")
    ap.add_argument("--look3", action="store_true", help="registered look #3: round-2 forecasting model vs 4 methods")
    a = ap.parse_args()
    global REF
    if a.look2:
        REF = LOOK2_REF
    if a.look3:
        REF = LOOK3_REF
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    onto = {"gogpt": ks.Ontology()}
    onto["current"] = ks.Ontology(obo=FRESH / "ref" / "go-basic_current.obo")
    prot = pd.read_csv(FRESH / "proteins.tsv", sep="\t").set_index("accession")
    gts = ground_truth()
    avail = [k for k, _, _ in METHODS if (PREDS / f"{k}.npz").exists()]
    if a.look2:
        avail = [k for k in (LOOK2_REF,) + LOOK2_COMPARE if k in avail]
    elif a.look3:
        avail = [k for k in (LOOK3_REF,) + LOOK3_COMPARE if k in avail]
    else:
        avail = [k for k in avail if k not in (LOOK2_REF, LOOK3_REF)]
    log(f"methods with predictions: {avail}")
    raw = {}
    srows, brows, orows, prows = [], [], [], []
    for sname in a.sets.split(","):
        gt, ok = gts[sname]
        o = onto[ok]
        sc = ks.Scorer(o, gt)
        n = sc.n
        ids = sc.ids
        bins = prot["bin"].reindex(ids).astype(str).values
        orgs = prot.organism.reindex(ids)
        groups = {("org_group", g): (org_group(orgs).values == g) for g in ("human", "mouse", "rat", "zebrafish", "other")}
        vc = orgs.value_counts()
        for org in vc.index[vc >= 100]:
            groups[("organism", org)] = (orgs.values == org)
        log(f"== {sname}: {n} proteins; per aspect " + ", ".join(f"{x} {int(sc.in_ns[k].sum())}" for k, x in enumerate(ASPECTS)))
        rng = np.random.default_rng(B_SEED)
        BW = rng.multinomial(n, np.full(n, 1.0 / n), size=a.boot_reps).astype(np.float64)
        ones = np.ones((1, n))
        stats, point, boot, f1 = {}, {}, {}, {}
        for key in avail:
            if (key, ok) not in raw:
                raw[(key, ok)] = load_npz_pred(PREDS / f"{key}.npz", o)
            P, pids = raw[(key, ok)]
            S = processed(sc, pids, P)
            if key == "deepgose":
                keep = np.isin(o.ns[S.indices], [0, 1])
                S = S.tocsr()
                S.data = np.where(keep, S.data, 0)
                S.eliminate_zeros()
            stats[key] = build_stats(sc, S)
            npred = int((np.diff(S.tocsr().indptr) > 0).sum())
            point[key] = evaluate(stats[key], ones, aspects=ASP_OF[key])
            boot[key] = evaluate(stats[key], BW, aspects=ASP_OF[key])
            f1[key] = per_protein_f1(stats[key], point[key], n, ASP_OF[key])
            r = {"set": sname, "method": key, "label": LABEL[key], "n_proteins": n, "n_with_predictions": npred}
            for x in ASPECTS:
                r[f"n_{x}"] = int(sc.in_ns[ASPECTS.index(x)].sum())
            for x in COLS + ("mean_MFBP",):
                for m in ("Fmax_w", "Fmax", "Smin_w"):
                    r[f"{m}_{x}"] = float(point[key][x][m][0])
                    lo, hi = ci(boot[key][x][m])
                    r[f"{m}_{x}_lo"], r[f"{m}_{x}_hi"] = lo, hi
                if x in ASPECTS:
                    r[f"tau_w_{x}"] = float(point[key][x]["tau_w"][0])
                    r[f"cov_w_{x}"] = float(point[key][x]["cov_w"][0])
            srows.append(r)
            log(f"  {key:20s} pred {npred:6d}/{n}  Fw " + " ".join(f"{x} {point[key][x]['Fmax_w'][0]:.4f}" for x in COLS)
                + f"  Smin_w mean {point[key]['mean']['Smin_w'][0]:.3f}")
        # differences to Ontologue (bootstrap) on every set
        if REF in stats:
            for key in stats:
                if key == REF:
                    continue
                for x in COLS + ("mean_MFBP",):
                    for m in ("Fmax_w", "Fmax", "Smin_w"):
                        d = boot[REF][x][m] - boot[key][x][m]
                        lo, hi = ci(d)
                        row = [r for r in srows if r["set"] == sname and r["method"] == key][0]
                        row[f"diff_{m}_{x}"] = float(point[REF][x][m][0] - point[key][x][m][0])
                        row[f"diff_{m}_{x}_lo"], row[f"diff_{m}_{x}_hi"] = lo, hi
        # identity bins and organism groups
        for kind, gdefs, rows in (("bin", {("bin", b): bins == b for b in BINS}, brows), ("org", groups, orows)):
            for (gk, gname), mask in gdefs.items():
                if mask.sum() == 0:
                    continue
                gp = {key: evaluate(stats[key], ones, mask, ASP_OF[key]) for key in stats}
                gb = {key: evaluate(stats[key], BW, mask, ASP_OF[key]) for key in stats}
                for key in stats:
                    r = {"set": sname, "grouping": gk, "group": gname, "method": key, "label": LABEL[key],
                         "n_proteins": int(mask.sum())}
                    for k, x in enumerate(ASPECTS):
                        r[f"n_{x}"] = int((mask & sc.in_ns[k]).sum())
                        r[f"Fmax_w_{x}"] = float(gp[key][x]["Fmax_w"][0])
                    for x in ("mean", "mean_MFBP"):
                        r[f"Fmax_w_{x}"] = float(gp[key][x]["Fmax_w"][0])
                        r[f"Fmax_w_{x}_lo"], r[f"Fmax_w_{x}_hi"] = ci(gb[key][x]["Fmax_w"])
                    r["Fmax_mean"] = float(gp[key]["mean"]["Fmax"][0])
                    r["Smin_w_mean"] = float(gp[key]["mean"]["Smin_w"][0])
                    with np.errstate(invalid="ignore"):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            r["perprot_F1_mean"] = float(np.nanmean(np.nanmean(f1[key]["u"][mask], 0)))
                            r["perprot_F1w_mean"] = float(np.nanmean(np.nanmean(f1[key]["w"][mask], 0)))
                    if REF in stats and key != REF:
                        d = gb[REF]["mean"]["Fmax_w"] - gb[key]["mean"]["Fmax_w"]
                        r["diff_Fmax_w_mean"] = float(gp[REF]["mean"]["Fmax_w"][0] - gp[key]["mean"]["Fmax_w"][0])
                        r["diff_Fmax_w_mean_lo"], r["diff_Fmax_w_mean_hi"] = ci(d)
                    rows.append(r)
        log(f"  {sname}: bins and organism groups done")

        # permutation tests (registered family on FR24-P only)
        if sname == "FR24-P" and REF in stats and a.perm_reps > 0:
            prng = np.random.default_rng(P_SEED)
            chunk = 1000
            swaps = [prng.integers(0, 2, size=(min(chunk, a.perm_reps - s), n)).astype(np.float64)
                     for s in range(0, a.perm_reps, chunk)]
            for key in stats:
                if key == REF:
                    continue
                t = time.time()
                asp = ASP_OF[key]
                null = {(x, m): [] for x in COLS for m in TESTED}
                totA = {x: {wn: {k: stats[REF][x][wn][k].sum(0)[None] for k in KEYS} for wn in ("u", "w")} for x in ASPECTS}
                totB = {x: {wn: {k: stats[key][x][wn][k].sum(0)[None] for k in KEYS} for wn in ("u", "w")} for x in ASPECTS}
                for Sw in swaps:
                    res = {}
                    for x in ASPECTS:
                        idx = stats[REF][x]["idx"]
                        Swk = Sw[:, idx]
                        sa = {"ne": np.full(len(Sw), float(len(idx)))}
                        sb = {"ne": sa["ne"]}
                        for wn in ("u", "w"):
                            sa[wn], sb[wn] = {}, {}
                            for k in KEYS:
                                dlt = Swk @ (stats[key][x][wn][k] - stats[REF][x][wn][k])
                                sa[wn][k] = totA[x][wn][k] + dlt
                                sb[wn][k] = totB[x][wn][k] - dlt
                        ma, mb = metrics_from(sa), metrics_from(sb)
                        res[x] = {m: ma[m] - mb[m] for m in TESTED}
                    for m in TESTED:
                        for x in ASPECTS:
                            null[(x, m)].append(res[x][m])
                        null[("mean", m)].append(np.mean([res[x][m] for x in ASPECTS], 0))
                tests = [x for x in ASPECTS if x in asp] + (["mean"] if len(asp) == 3 else [])
                for x in tests:
                    for m in TESTED:
                        obs = float(point[REF][x][m][0] - point[key][x][m][0])
                        d = boot[REF][x][m] - boot[key][x][m]
                        lo, hi = ci(d)
                        prows.append({"set": sname, "comparison": f"{REF} - {key}", "method": key, "label": LABEL[key],
                                      "metric": m, "aspect": x, "diff": obs, "ci_lo": lo, "ci_hi": hi,
                                      "p_perm": perm_p(obs, np.concatenate(null[(x, m)]))})
                log(f"  permutation {REF} - {key}: {a.perm_reps} reps in {time.time() - t:.0f}s")
    summ = pd.DataFrame(srows)
    pv = pd.DataFrame(prows)
    if len(pv):
        pv["p_holm"] = holm(pv.p_perm.values)
        pv["family_size"] = len(pv)
        for _, r in pv.iterrows():
            m = (summ.set == r.set) & (summ.method == r.method)
            summ.loc[m, f"p_perm_{r.metric}_{r.aspect}"] = r.p_perm
            summ.loc[m, f"p_holm_{r.metric}_{r.aspect}"] = r.p_holm
    summ.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(brows).to_csv(out / "per_bin.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(orows).to_csv(out / "per_organism.tsv", sep="\t", index=False, float_format="%.4f")
    pv.to_csv(out / "pvalues.tsv", sep="\t", index=False, float_format="%.6g")
    write_report(summ, pd.DataFrame(brows), pd.DataFrame(orows), pv, out, a)
    log(f"wrote {out}/summary.tsv, per_bin.tsv, per_organism.tsv, pvalues.tsv, report.md")


# ------------------------------------------------------------------------------------------------ report

def f3(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write_report(summ, bins, orgs, pv, out, a):
    L = []
    w = L.append
    w("# fresh24 look #1: scores of the frozen methods\n")
    w("Registration: `runs/fresh24_looks.md` (2026-10-19 18:40:00). Scorer: `src/cafa6/score_fresh24.py` "
      "(cafaeval protocol, GO 2023-01-01, CAFA-5 IA). Predictions: `runs/fresh24/preds/` (how each was made: "
      "`runs/fresh24/methods.md`). This look selects nothing.\n")
    w(f"Bootstrap: {a.boot_reps} paired protein resamples (seed 0). Permutation tests: {a.perm_reps} swaps "
      f"(seed 1), Ontologue minus each method on {{Fmax_w, Fmax, Smin_w}} x {{MF, BP, CC, mean}} on FR24-P, Holm over "
      f"the whole family ({len(pv)} tests). For S-min lower is better, so a negative difference favours Ontologue.\n")
    p = summ[summ.set == "FR24-P"]
    if len(p):
        r0 = p.iloc[0]
        w(f"## FR24-P (primary): {int(r0.n_proteins):,} proteins (MF {int(r0.n_MF):,}, BP {int(r0.n_BP):,}, "
          f"CC {int(r0.n_CC):,})\n")
        rows = []
        for _, r in p.sort_values("Fmax_w_mean", ascending=False, na_position="last").iterrows():
            ph = r.get("p_holm_Fmax_w_mean", np.nan)
            if r.method == REF:
                d = "—"
            elif np.isfinite(r.get("diff_Fmax_w_mean", np.nan)):
                d = f"{r.diff_Fmax_w_mean:+.3f} [{r.diff_Fmax_w_mean_lo:+.3f}, {r.diff_Fmax_w_mean_hi:+.3f}], p_Holm {f3(ph, 4)}"
            else:
                d = (f"MF+BP mean {r.diff_Fmax_w_mean_MFBP:+.3f} [{r.diff_Fmax_w_mean_MFBP_lo:+.3f}, "
                     f"{r.diff_Fmax_w_mean_MFBP_hi:+.3f}] (not tested; MF p_Holm {f3(r.get('p_holm_Fmax_w_MF', np.nan), 4)}, "
                     f"BP p_Holm {f3(r.get('p_holm_Fmax_w_BP', np.nan), 4)})")
            rows.append({"method": r.label, "proteins with predictions": f"{int(r.n_with_predictions):,}",
                         "Fw MF": f3(r.Fmax_w_MF), "Fw BP": f3(r.Fmax_w_BP), "Fw CC": f3(r.Fmax_w_CC),
                         "Fw mean [95% CI]": (f"{r.Fmax_w_mean:.3f} [{r.Fmax_w_mean_lo:.3f}, {r.Fmax_w_mean_hi:.3f}]"
                                              if np.isfinite(r.Fmax_w_mean) else f"— (MF+BP {r.Fmax_w_mean_MFBP:.3f})"),
                         "F mean": f3(r.Fmax_mean), "S-min MF/BP/CC": "/".join(f3(r[f"Smin_w_{x}"], 2) for x in ASPECTS),
                         "S-min mean": f3(r.Smin_w_mean, 2), "Ontologue − method, Fw mean [95% CI]": d})
        w(md(rows, list(rows[0])) + "\n")
        w("Fw = IA-weighted Fmax, F = unweighted Fmax, S-min = IA-weighted S-min. Proteins without predictions count "
          "with recall 0.\n")
    if len(pv):
        w("## Permutation tests on FR24-P (Ontologue minus method; Holm over the family)\n")
        t = pv.copy()
        rows = []
        for key in t.method.unique():
            q = t[t.method == key]
            row = {"method": q.label.iloc[0]}
            for m in TESTED:
                for x in COLS:
                    z = q[(q.metric == m) & (q.aspect == x)]
                    row[f"{m} {x}"] = "—" if not len(z) else f"{z['diff'].iloc[0]:+.3f} (p {z.p_holm.iloc[0]:.4f})"
            rows.append(row)
        w(md(rows, list(rows[0])) + "\n")
        w(f"Smallest attainable permutation p: {1 / (1 + a.perm_reps):.1e}; Holm p values are those times up to "
          f"{len(pv)}. Tests with Holm p < 0.05: {int((pv.p_holm < 0.05).sum())} of {len(pv)}; of these, "
          f"in Ontologue's favour (higher F, lower S-min): "
          f"{int(((pv.p_holm < 0.05) & (np.where(pv.metric.str.startswith('Smin'), -pv['diff'], pv['diff']) > 0)).sum())}.\n")
    w("## Secondary sets (descriptive)\n")
    rows = []
    for sname in SETS[1:]:
        q = summ[summ.set == sname]
        if not len(q):
            continue
        for _, r in q.sort_values("Fmax_w_mean", ascending=False, na_position="last").iterrows():
            rows.append({"set": sname, "n": f"{int(r.n_proteins):,}", "method": r.label,
                         "Fw MF/BP/CC": "/".join(f3(r[f"Fmax_w_{x}"]) for x in ASPECTS), "Fw mean": f3(r.Fmax_w_mean),
                         "S-min mean": f3(r.Smin_w_mean, 2),
                         "Ontologue − method [95% CI]": "—" if r.method == REF or not np.isfinite(r.get("diff_Fmax_w_mean", np.nan))
                         else f"{r.diff_Fmax_w_mean:+.3f} [{r.diff_Fmax_w_mean_lo:+.3f}, {r.diff_Fmax_w_mean_hi:+.3f}]"})
    if rows:
        w(md(rows, list(rows[0])) + "\n")
    for sname in [s for s in SETS if s in set(bins.set)]:
        q = bins[bins.set == sname]
        w(f"## {sname}: Fw mean per identity bin to GO-GPT train+val (threshold chosen inside the bin)\n")
        n = q.groupby("group").n_proteins.first()
        piv = q.pivot_table(index="method", columns="group", values="Fmax_w_mean")
        cols = [b for b in BINS if b in piv.columns]
        order = summ[summ.set == sname].sort_values("Fmax_w_mean", ascending=False).method
        rows = [{"method": LABEL[m], **{f"{b} (n={int(n[b])})": f3(piv.loc[m, b]) for b in cols}} for m in order if m in piv.index]
        w(md(rows, list(rows[0])) + "\n")
        piv = q.pivot_table(index="method", columns="group", values="perprot_F1w_mean")
        rows = [{"method": LABEL[m], **{b: f3(piv.loc[m, b]) for b in cols}} for m in order if m in piv.index]
        w(f"Mean per-protein F1_w at each method's own Fmax_w threshold of the whole set ({sname}):\n")
        w(md(rows, list(rows[0])) + "\n")
        if sname != "FR24-P":
            continue
        qq = q[q.method == REF]
        rows = []
        for b in cols:
            z = qq[qq.group == b].iloc[0]
            rows.append({"bin": b, "proteins": int(z.n_proteins), "n MF/BP/CC": f"{z.n_MF}/{z.n_BP}/{z.n_CC}",
                         "Fw MF": f3(z.Fmax_w_MF), "Fw BP": f3(z.Fmax_w_BP), "Fw CC": f3(z.Fmax_w_CC),
                         "Fw mean [95% CI]": f"{z.Fmax_w_mean:.3f} [{z.Fmax_w_mean_lo:.3f}, {z.Fmax_w_mean_hi:.3f}]",
                         "F mean": f3(z.Fmax_mean), "S-min mean": f3(z.Smin_w_mean, 2),
                         "per-protein F1_w": f3(z.perprot_F1w_mean)})
        w("### Ontologue on FR24-P by identity bin\n")
        w(md(rows, list(rows[0])) + "\n")
    for sname in [s for s in ("FR24-P", "NK-all") if s in set(orgs.set)]:
        q = orgs[orgs.set == sname]
        w(f"## {sname}: Fw mean by organism group\n")
        q = q.assign(col=q.grouping + ":" + q.group)
        n = q.groupby("col").n_proteins.first()
        piv = q.pivot_table(index="method", columns="col", values="Fmax_w_mean")
        cols = [c for c in piv.columns if c.startswith("org_group")] + [c for c in piv.columns if c.startswith("organism")]
        order = summ[summ.set == sname].sort_values("Fmax_w_mean", ascending=False).method
        rows = [{"method": LABEL[m], **{f"{c.split(':', 1)[1]} (n={int(n[c])})": f3(piv.loc[m, c]) for c in cols}}
                for m in order if m in piv.index]
        w(md(rows, list(rows[0])) + "\n")
    # comparison with the GO-GPT test
    cpf = ROOT / "runs" / "compare_paper" / "summary.tsv"
    if cpf.exists():
        cp = pd.read_csv(cpf, sep="\t")
        cp = cp[cp.subset == "S3_all"].set_index("method")
        p = summ[summ.set == "FR24-P"].set_index("method")
        rows = []
        for key in p.index:
            tk = TEST_KEY.get(key)
            tv = float(cp.loc[tk, "Fw_mean"]) if tk in cp.index else np.nan
            rows.append({"method": LABEL[key], "GO-GPT test S3 Fw mean": f3(tv), "FR24-P Fw mean": f3(p.loc[key, "Fmax_w_mean"]),
                         "change": f3(p.loc[key, "Fmax_w_mean"] - tv) if np.isfinite(tv) else "—",
                         "GO-GPT test Fw MF/BP/CC": "/".join(f3(cp.loc[tk, f"Fw_{x}"]) for x in ASPECTS) if tk in cp.index else "—",
                         "FR24-P Fw MF/BP/CC": "/".join(f3(p.loc[key, f"Fmax_w_{x}"]) for x in ASPECTS)})
        w("## Against the GO-GPT test (runs/compare_paper/summary.tsv, S3_all, 8,630 proteins)\n")
        w("On the GO-GPT test Ontologue is the frozen HiGO-R2 file (the same pipeline; its kNN and HiGO parts were "
          "scored in runs/eval_suite, not in compare_paper).\n")
        w(md(rows, list(rows[0])) + "\n")
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
