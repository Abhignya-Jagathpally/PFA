"""Day-1 diagnostics on TV v2: (A) partial-coverage homologs as an error source, (B) abstention fallbacks.

Everything is scored on the temporal validation set TV v2 (data/proc_gogpt_tv2: gt_tv.tsv, cells.parquet) with
knn_sharp.Scorer (cafaeval reproduction) and the chosen_v2 test-like cell weights (stacker_tv.test_like_weights).
Nothing is scored on the GO-GPT test set.

Inputs
  kNN prior     data/proc_gogpt_tv2/knn_trainval/tv_knn.npz (T5_k50_g0_na, train + val references), p_full over the
                ontology, cut at 0.01 and max-propagated (as stacker_tv / calibrate_tv do)
  Ontologue     runs/stacker_tv/l3_c0/pred_tv_oof_twin.npz (cross-fitted twin stacker, stacker_tv.save_sparse format)
  HiGO head     runs/l3_c0_tv2/pred_tv_all_top500.npz (data.load_pred, via stacker_tv.higo_matrix)
  DIAMOND hits  data/proc_gogpt_tv2/knn_trainval/diamond_hits_tv_trainval.parquet (q, s, pident, bits, evalue, qcov);
                restricted to the reference db (v2 train + val) and the top 50 by bitscore, as the prior uses them

Definitions (pre-specified in logs/day1_analyses.log before scoring)
  best hit      the used hit with the highest pident (ties by bitscore); its pident is tv_knn max_pident
  qcov          best hit's query coverage (DIAMOND qcovhsp, %)
  tcov_est      estimated target coverage = min(100, qcov * len(query) / len(subject)); the hit table has no subject
                coordinates, so this assumes the aligned span has about the same length on both sequences
  wqcov         kNN-weight-averaged qcov over the used hits (w = bits * exp((pident - max pident) / 5))
  thresholds    per aspect, the tau maximising the whole-TV-v2 test-like Fmax of that predictor
  per-protein   P, R, F1 at that tau, IA-weighted (Scorer IA) and unweighted (IA = 1 for every non-root term)

A  TV v2 proteins with 50 <= best-hit identity < 95, grouped by qcov / tcov_est / wqcov into <50, 50-80, >=80.
   Per group: n, test-like share, cafa-style precision (mean over proteins with a prediction), recall, mean
   per-protein F1 (no prediction -> F1 0), FP IA mass, and the group's own Fmax. Then the share of the kNN
   false-positive IA mass in 50-95% that comes from best-hit qcov < 80%, and an oracle bound: whole-set test-like
   Fmax if every false positive of those proteins were removed (predicted mass set to the true-positive mass).
B  Ontologue at its per-aspect test-like Fmax thresholds: share of (protein, aspect) with no positive-IA term
   >= tau by identity bin, their recall shortfall, and two fallbacks for those pairs only (rules in the log):
   F1 HiGO top-3 terms of the aspect (IA > 0) at their HiGO score; F2 the top positive-IA Ontologue term of the
   aspect raised to tau. Mean Fmax before/after (test-like and plain) with paired protein bootstraps.
   Exploratory: thresholds and abstainer sets are in-sample on TV v2; this is not a registered selection.

Outputs: runs/day1_analyses/{coverage.tsv, abstention.tsv, fallbacks.tsv, summary.json} (report.md written by hand
from these numbers). Progress lines go to logs/day1_analyses.log.

Usage: OMP_NUM_THREADS=8 PYTHONPATH=src python -m cafa6.day1_analyses [--boot 1000] [--only A|B]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from cafa6 import knn_sharp as ks  # noqa: E402
from cafa6 import stacker_tv as st  # noqa: E402
from cafa6.data import ROOT  # noqa: E402

TV2 = ROOT / "data" / "proc_gogpt_tv2"
KNN_TV = TV2 / "knn_trainval"
HITS = KNN_TV / "diamond_hits_tv_trainval.parquet"
ONTOLOGUE = ROOT / "runs" / "stacker_tv" / "l3_c0" / "pred_tv_oof_twin.npz"
HIGO = ROOT / "runs" / "l3_c0_tv2"
OUT = ROOT / "runs" / "day1_analyses"
LOGF = ROOT / "logs" / "day1_analyses.log"
ASP = ("MF", "BP", "CC")
COV_EDGES = (50.0, 80.0)
COV_NAMES = ("<50", "50-80", ">=80")
TOPK_HIGO = 3
MIN_GAIN = 0.003
T0 = time.time()


def log(msg: str, file: bool = False):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:6.0f}s] {msg}", flush=True)
    if file:
        with open(LOGF, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [day1] {msg}\n")


# ------------------------------------------------------------------------------------------------ inputs

def load_sparse(path, onto) -> tuple[np.ndarray, sp.csr_matrix]:
    """stacker_tv.save_sparse file -> (ids, CSR over all ontology columns)."""
    z = np.load(path, allow_pickle=False)
    Pu = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"])).tocoo()
    cols = np.array([onto.idx[t] for t in z["terms"]], np.int64)
    return z["ids"].astype(str), sp.csr_matrix((Pu.data, (Pu.row, cols[Pu.col])), shape=(Pu.shape[0], len(onto.terms)))


def knn_prior(onto) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    K = ks.load_knn("tv", KNN_TV)
    cols = np.array([onto.idx[t] for t in K["terms_full"]])
    pf = K["p_full"].tocoo()
    P = sp.csr_matrix((pf.data, (pf.row, cols[pf.col])), shape=(len(K["ids"]), len(onto.terms)))
    P.sum_duplicates()
    P.data[P.data < 0.01] = 0
    return K["ids"].astype(str), onto.propagate_max(P), K["max_pident"].astype(np.float32)


def align(ids_src, P, ids):
    pos = pd.Series(np.arange(len(ids_src)), index=ids_src)
    pos = pos[~pos.index.duplicated()]
    return P[pos.reindex(ids).values.astype(np.int64)].tocsr()


def hit_features(ids: np.ndarray, maxpid: np.ndarray) -> pd.DataFrame:
    """Per TV protein: best-hit (max pident) qcov, estimated target coverage, kNN-weighted qcov."""
    h = pd.read_parquet(HITS)
    h = h[h.q.isin(set(ids))]
    tdf = pd.read_parquet(TV2 / "train.parquet", columns=["id", "split", "length"])
    db = set(tdf.id[tdf.split == "train"]) | set(tdf.id[tdf.split == "val"])
    h = h[h.s.isin(db)].sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
    h = h[h.groupby("q", sort=False).cumcount() < 50].copy()
    ln = pd.concat([pd.read_parquet(TV2 / "train.parquet", columns=["id", "length"]),
                    pd.read_parquet(ks.SRC / "train.parquet", columns=["id", "length"])]).drop_duplicates("id")
    ln = ln.set_index("id").length
    h["qlen"], h["slen"] = h.q.map(ln).values, h.s.map(ln).values
    h["tcov_est"] = np.minimum(100.0, h.qcov * h.qlen / h.slen)
    mx = h.groupby("q").pident.transform("max")
    h["w"] = h.bits * np.exp((h.pident - mx) / 5.0)
    best = h.sort_values(["q", "pident", "bits"], ascending=[True, False, False]).drop_duplicates("q").set_index("q")
    wq = (h.w * h.qcov).groupby(h.q).sum() / h.w.groupby(h.q).sum()
    f = pd.DataFrame({"best_pident": best.pident, "qcov": best.qcov, "tcov_est": best.tcov_est,
                      "len_ratio": best.slen / best.qlen, "wqcov": wq}).reindex(ids)
    d = np.abs(f.best_pident.fillna(0).values - maxpid)
    log(f"hit features: {f.qcov.notna().sum()} TV proteins with hits; best pident vs tv_knn max_pident "
        f"max |diff| {d.max():.3f} (mismatches > 0.05: {(d > 0.05).sum()}); missing lengths "
        f"{int(h.slen.isna().sum())} of {len(h)} hits")
    return f


# ------------------------------------------------------------------------------------------------ scoring

def unweighted_scorer(onto, gt) -> ks.Scorer:
    o2 = copy.copy(onto)
    ia = np.ones(len(onto.terms))
    ia[[onto.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]] = 0
    o2.ia = ia
    return ks.Scorer(o2, gt)


def raw_curves(sc: ks.Scorer, P: sp.csr_matrix, ids) -> dict:
    """Scorer.curves arithmetic, keeping predicted and true-positive IA mass [n_ns, 99] and GT mass per protein."""
    ids = np.asarray(ids)
    m = np.array([p in sc.pos for p in ids])
    P = ks.to_submission(P.tocsr()[np.where(m)[0]])
    gi = np.array([sc.pos[p] for p in ids[m]], dtype=np.int64)
    P = sc.o.propagate_max(P)
    rows = gi[np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))]
    cols, vals = P.indices, P.data
    w, nsc = sc.o.ia[cols], sc.o.ns[cols]
    ing = np.isin(rows * len(sc.o.terms) + cols, sc.gkey)
    kk = np.searchsorted(ks.TAU, vals, side="right")
    out = {}
    for k in range(3):
        sel = (nsc == k) & (w > 0) & sc.in_ns[k, rows]
        r, kq, ww, g = rows[sel], kk[sel], w[sel], ing[sel]
        C = np.bincount(r * 100 + kq, weights=ww, minlength=sc.n * 100).reshape(sc.n, 100)
        I = np.bincount(r * 100 + kq, weights=ww * g, minlength=sc.n * 100).reshape(sc.n, 100)
        idx = np.where(sc.in_ns[k])[0]
        out[ASP[k]] = {"idx": idx, "predw": np.cumsum(C[:, ::-1], 1)[:, ::-1][:, 1:][idx],
                       "interw": np.cumsum(I[:, ::-1], 1)[:, ::-1][:, 1:][idx], "gtw": sc.gtw[k][idx]}
    return out


def to_cur(raw: dict) -> dict:
    cur = {}
    for a, c in raw.items():
        has = c["predw"] > 0
        prec = np.divide(c["interw"], c["predw"], out=np.zeros_like(c["predw"]), where=has)
        gw = c["gtw"][:, None]
        rec = np.divide(c["interw"], gw, out=np.zeros_like(c["interw"]), where=gw > 0)
        cur[a] = {"idx": c["idx"], "prec": prec.astype(np.float32), "has": has, "rec": rec.astype(np.float32)}
    return cur


def thresholds(sc: ks.Scorer, cur: dict, w: np.ndarray) -> dict:
    """Index into ks.TAU of the whole-set weighted Fmax per aspect (Scorer.fmax arithmetic)."""
    ref, out = sc.fmax(cur, weights=w), {}
    for a, c in cur.items():
        ww = w[c["idx"]]
        nh = ww @ c["has"]
        P = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
        R = (ww @ c["rec"]) / ww.sum()
        F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
        F[nh <= 0] = 0
        assert abs(F.max() - ref[a]) < 1e-9
        out[a] = int(np.argmax(F))
    return out


def per_protein(raw: dict, tau: dict) -> dict:
    """{aspect: DataFrame indexed by scorer position: predw, tpw, fpw, gtw, has, P, R, F1} at tau."""
    out = {}
    for a, c in raw.items():
        t = tau[a]
        pw, iw, gw = c["predw"][:, t], c["interw"][:, t], c["gtw"]
        P = np.divide(iw, pw, out=np.zeros_like(pw), where=pw > 0)
        R = np.divide(iw, gw, out=np.zeros_like(iw), where=gw > 0)
        F = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P + R) > 0)
        out[a] = pd.DataFrame({"predw": pw, "tpw": iw, "fpw": pw - iw, "gtw": gw, "has": pw > 0,
                               "P": P, "R": R, "F1": F}, index=c["idx"])
    return out


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


# ------------------------------------------------------------------------------------------------ A

def analysis_a(ctx: dict) -> tuple[pd.DataFrame, dict]:
    sc, w, feat = ctx["sc"], ctx["w"], ctx["feat"]
    pid = feat.best_pident.reindex(sc.ids).fillna(0).values
    in_band = (pid >= 50) & (pid < 95)
    fsc = feat.reindex(sc.ids)
    groupings = {}
    for col in ("qcov", "tcov_est", "wqcov"):
        v = fsc[col].values
        groupings[col] = np.select([v < COV_EDGES[0], v < COV_EDGES[1]], COV_NAMES[:2], COV_NAMES[2])
    idb = np.where(pid < 80, "50-80", "80-95")
    log(f"A: {in_band.sum()} TV v2 proteins at 50-95% best-hit identity (test-like share "
        f"{w[in_band].sum() / w.sum():.3f}); qcov groups " +
        str(pd.Series(groupings['qcov'][in_band]).value_counts().to_dict()) + "; tcov_est groups " +
        str(pd.Series(groupings['tcov_est'][in_band]).value_counts().to_dict()), file=True)
    rows = []
    for pred in ("knn", "ontologue"):
        pp, ppu = ctx["pp"][pred], ctx["ppu"][pred]
        cur = ctx["cur"][pred]
        specs = [(g, None) for g in groupings] + [("qcov", "50-80"), ("qcov", "80-95")]
        for gcol, sub in specs:
            base = in_band & ((idb == sub) if sub else True)
            for gname in COV_NAMES + ("all",):
                mask = base & ((groupings[gcol] == gname) if gname != "all" else True)
                if gcol != "qcov" and gname == "all" and sub is None:
                    continue
                r0 = {"predictor": pred, "grouping": gcol if sub is None else f"qcov|id {sub}",
                      "group": gname, "n": int(mask.sum()), "testlike_share": w[mask].sum() / w.sum(),
                      "testlike_share_of_band": w[mask].sum() / w[in_band].sum()}
                if mask.sum() == 0:
                    continue
                fm = sc.fmax(cur, weights=mask.astype(float))
                fmt = sc.fmax(cur, weights=w * mask)
                acc = {}
                for a in ASP:
                    d, du = pp[a], ppu[a]
                    sel = mask[d.index.values]
                    dd, ddu = d[sel], du[sel]
                    if len(dd) == 0:
                        continue
                    ww = w[dd.index.values]
                    hp = dd.has.values
                    r = {**r0, "aspect": a, "n_aspect": len(dd), "pred_cov": hp.mean(),
                         "prec_ia": dd.P.values[hp].mean() if hp.any() else np.nan, "rec_ia": dd.R.mean(),
                         "f1_ia": dd.F1.mean(), "f1_ia_tl": wmean(dd.F1.values, ww),
                         "prec_u": ddu.P.values[ddu.has.values].mean() if ddu.has.any() else np.nan,
                         "rec_u": ddu.R.mean(), "f1_u": ddu.F1.mean(),
                         "fp_ia_mean": dd.fpw.mean(), "tp_ia_mean": dd.tpw.mean(), "gt_ia_mean": dd.gtw.mean(),
                         "group_fmax": fm[a], "group_fmax_tl": fmt[a]}
                    rows.append(r)
                    for k in ("pred_cov", "prec_ia", "rec_ia", "f1_ia", "f1_ia_tl", "prec_u", "rec_u", "f1_u",
                              "fp_ia_mean", "group_fmax", "group_fmax_tl"):
                        acc.setdefault(k, []).append(r[k])
                rows.append({**r0, "aspect": "mean", "n_aspect": np.nan,
                             **{k: float(np.nanmean(v)) for k, v in acc.items()}})
    cov = pd.DataFrame(rows)

    # false-positive IA mass decomposition and oracle bound
    summ = {"band_n": int(in_band.sum()), "band_testlike_share": float(w[in_band].sum() / w.sum())}
    for pred in ("knn", "ontologue"):
        pp = ctx["pp"][pred]
        s = {}
        for gcol in ("qcov", "tcov_est"):
            low = in_band & (groupings[gcol] != ">=80")
            for wname, wv in (("plain", np.ones(sc.n)), ("testlike", w)):
                fp_all = fp_band = fp_low = pr_band = pr_low = tp_band = tp_low = 0.0
                for a in ASP:
                    d = pp[a]
                    i = d.index.values
                    fp_all += float((d.fpw.values * wv[i]).sum())
                    fp_band += float((d.fpw.values * wv[i] * in_band[i]).sum())
                    fp_low += float((d.fpw.values * wv[i] * low[i]).sum())
                    pr_band += float((d.predw.values * wv[i] * in_band[i]).sum())
                    pr_low += float((d.predw.values * wv[i] * low[i]).sum())
                    tp_band += float((d.tpw.values * wv[i] * in_band[i]).sum())
                    tp_low += float((d.tpw.values * wv[i] * low[i]).sum())
                s[f"{gcol}<80|{wname}"] = {
                    "protein_share_of_band": float(wv[low].sum() / wv[in_band].sum()),
                    "fp_share_of_band_fp": fp_low / fp_band, "pred_mass_share_of_band": pr_low / pr_band,
                    "tp_share_of_band_tp": tp_low / tp_band,
                    "band_fp_share_of_all_fp": fp_band / fp_all, "low_fp_share_of_all_fp": fp_low / fp_all}
        # oracle: all FPs removed on (a) band proteins with qcov < 80, (b) band proteins with qcov >= 80
        raw = ctx["raw"][pred]
        base_tl = ctx["sc"].fmax(ctx["cur"][pred], weights=w)["mean"]
        base_pl = ctx["sc"].fmax(ctx["cur"][pred])["mean"]
        for tag, mask in (("qcov<80", in_band & (groupings["qcov"] != ">=80")),
                          ("qcov>=80", in_band & (groupings["qcov"] == ">=80")),
                          ("tcov_est<80", in_band & (groupings["tcov_est"] != ">=80")),
                          ("band_all", in_band)):
            r2 = {}
            for a, c in raw.items():
                mm = mask[c["idx"]][:, None]
                r2[a] = {**c, "predw": np.where(mm, c["interw"], c["predw"])}
            c2 = to_cur(r2)
            s[f"oracle_noFP_{tag}"] = {"testlike_gain": sc.fmax(c2, weights=w)["mean"] - base_tl,
                                       "plain_gain": sc.fmax(c2)["mean"] - base_pl, "n": int(mask.sum())}
        summ[pred] = s
        log(f"A [{pred}] FP IA mass: " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                                      for k, v in s.items()}), file=True)
    return cov, summ


# ------------------------------------------------------------------------------------------------ B

def analysis_b(ctx: dict, boot: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sc, w, onto = ctx["sc"], ctx["w"], ctx["onto"]
    ids, P = ctx["ids"], ctx["P"]["ontologue"]
    tau = ctx["tau"]["ontologue"]
    pp = ctx["pp"]["ontologue"]
    bins = ks.ident_bins(ctx["maxpid_sc"]).values
    log("B: Ontologue per-aspect test-like Fmax thresholds " + json.dumps({a: float(ks.TAU[t]) for a, t in tau.items()}),
        file=True)
    rows = []
    abst = {}
    for a in ASP:
        d = pp[a]
        ab = ~d.has.values
        abst[a] = d.index.values[ab]
        for b in ["all"] + ks.BINS[::-1]:
            sel = np.ones(len(d), bool) if b == "all" else bins[d.index.values] == b
            if sel.sum() == 0:
                continue
            i = d.index.values[sel]
            ww = w[i]
            rec_have = d.R.values[sel & ~ab]
            rows.append({"aspect": a, "bin": b, "n_with_gt": int(sel.sum()), "n_abstain": int((sel & ab).sum()),
                         "share_abstain": (sel & ab).sum() / sel.sum(),
                         "share_abstain_tl": wmean(ab[sel].astype(float), ww),
                         "testlike_share_of_aspect": ww.sum() / w[d.index.values].sum(),
                         "recall_ia_mean": d.R.values[sel].mean(),
                         "recall_ia_mean_predicting": rec_have.mean() if len(rec_have) else np.nan,
                         "recall_lost_vs_predicting": (sel & ab).sum() / sel.sum() * (rec_have.mean() if len(rec_have) else 0),
                         "abstainer_gt_ia_share": d.gtw.values[sel & ab].sum() / d.gtw.values[sel].sum()})
    ab_tab = pd.DataFrame(rows)
    show = ab_tab[["aspect", "bin", "n_with_gt", "n_abstain", "share_abstain", "share_abstain_tl",
                   "recall_ia_mean_predicting", "recall_lost_vs_predicting"]]
    log("B abstention by identity bin:\n" + show.round(4).to_string(index=False))

    # scorer positions -> rows of P (ids order)
    rpos = pd.Series(np.arange(len(ids)), index=ids)
    rpos = rpos[~rpos.index.duplicated()]
    ns = onto.ns
    iapos = onto.ia > 0
    Pc = P.tocsr().copy()
    Pc.sort_indices()

    # F2: raise the top positive-IA Ontologue term of each abstaining (protein, aspect) to tau_a
    add2_r, add2_c, add2_v, none2 = [], [], [], {}
    # F1: HiGO top-3 (IA > 0) of the aspect at the HiGO score
    H, _, _ = st.higo_matrix(HIGO, "tv", ids, onto, 500)
    H = H.tocsr()
    add1_r, add1_c, add1_v, above1 = [], [], [], {}
    for k, a in enumerate(ASP):
        t = float(ks.TAU[tau[a]])
        none2[a], above1[a] = 0, 0
        for gpos in abst[a]:
            r = int(rpos[sc.ids[gpos]])
            s, e = Pc.indptr[r], Pc.indptr[r + 1]
            c, v = Pc.indices[s:e], Pc.data[s:e]
            m = (ns[c] == k) & iapos[c]
            if m.any():
                j = np.argmax(np.where(m, v, -1))
                # +0.001 so the 3-decimal rounding of to_submission cannot land just below TAU[tau_a]
                add2_r.append(r); add2_c.append(c[j]); add2_v.append(min(1.0, max(t + 0.001, float(v[j]))))
            else:
                none2[a] += 1
            s, e = H.indptr[r], H.indptr[r + 1]
            c, v = H.indices[s:e], H.data[s:e]
            m = (ns[c] == k) & iapos[c]
            if m.any():
                o = np.argsort(-v[m])[:TOPK_HIGO]
                cc, vv = c[m][o], v[m][o]
                add1_r += [r] * len(cc); add1_c += list(cc); add1_v += list(vv)
                above1[a] += int((vv >= t).any())

    def with_adds(rr, cc, vv):
        A = sp.csr_matrix((np.asarray(vv, np.float32), (rr, cc)), shape=Pc.shape)
        return onto.propagate_max(Pc.maximum(A).tocsr())

    variants = {"F1_higo_top3": with_adds(add1_r, add1_c, add1_v), "F2_top1_floor": with_adds(add2_r, add2_c, add2_v)}
    log("B fallback coverage: F2 pairs without any positive-IA Ontologue term " + json.dumps(none2) +
        f"; F1 abstaining pairs whose HiGO top-{TOPK_HIGO} reaches tau " + json.dumps(above1) +
        " of " + json.dumps({a: int(len(abst[a])) for a in ASP}), file=True)
    cur0 = ctx["cur"]["ontologue"]
    f0, f0p = sc.fmax(cur0, weights=w), sc.fmax(cur0)
    frows = [{"variant": "ontologue_base", **{f"tl_{a}": f0[a] for a in ASP}, "tl_mean": f0["mean"],
              **{f"plain_{a}": f0p[a] for a in ASP}, "plain_mean": f0p["mean"]}]
    res = {}
    for name, Pv in variants.items():
        cv = sc.curves(Pv, ids)
        ft, fp = sc.fmax(cv, weights=w), sc.fmax(cv)
        bt = sc.bootstrap(cv, cur0, reps=boot, seed=0, weights=w)
        bp = sc.bootstrap(cv, cur0, reps=boot, seed=0)
        promising = bool(bt["diff"] >= MIN_GAIN and bt["lo"] > 0)
        frows.append({"variant": name, **{f"tl_{a}": ft[a] for a in ASP}, "tl_mean": ft["mean"],
                      **{f"plain_{a}": fp[a] for a in ASP}, "plain_mean": fp["mean"],
                      "tl_diff": bt["diff"], "tl_lo": bt["lo"], "tl_hi": bt["hi"],
                      "plain_diff": bp["diff"], "plain_lo": bp["lo"], "plain_hi": bp["hi"], "promising": promising})
        # fixed-threshold view: F at the base thresholds (no re-optimisation), test-like
        rv = raw_curves(sc, Pv, ids)
        ppv = per_protein(rv, tau)
        fixed = {}
        for a in ASP:
            d0, d1 = pp[a], ppv[a]
            sel = np.isin(d1.index.values, abst[a])
            fixed[a] = {"abstainers_now_predicting": int(d1.has.values[sel].sum()),
                        "abstainers_mean_P": float(d1.P.values[sel][d1.has.values[sel]].mean()) if d1.has.values[sel].any() else np.nan,
                        "abstainers_mean_R": float(d1.R.values[sel].mean()),
                        "abstainers_mean_R_before": float(d0.R.values[np.isin(d0.index.values, abst[a])].mean())}
        res[name] = {"testlike": bt, "plain": bp, "promising": promising, "at_base_tau": fixed,
                     "tl_after": ft["mean"], "plain_after": fp["mean"]}
        log(f"B {name}: test-like {f0['mean']:.4f} -> {ft['mean']:.4f} diff {bt['diff']:+.4f} "
            f"[{bt['lo']:+.4f}, {bt['hi']:+.4f}]; plain {f0p['mean']:.4f} -> {fp['mean']:.4f} diff {bp['diff']:+.4f} "
            f"[{bp['lo']:+.4f}, {bp['hi']:+.4f}] ({boot} reps); promising={promising}; at base tau "
            + json.dumps(fixed, default=float), file=True)
    res["base"] = {"tl": f0["mean"], "plain": f0p["mean"], "tau": {a: float(ks.TAU[t]) for a, t in tau.items()},
                   "f2_pairs_without_term": none2, "f1_pairs_reaching_tau": above1,
                   "abstaining_pairs": {a: int(len(abst[a])) for a in ASP}}
    return ab_tab, pd.DataFrame(frows), res


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--only", choices=("A", "B"), default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ks.use_tv("v2")
    log("START step 1: load inputs and score both predictors on TV v2", file=True)
    onto = ks.Ontology()
    gt = ks.gt_frame("tv")
    sc, scu = ks.Scorer(onto, gt), unweighted_scorer(onto, gt)
    w, wrep = st.test_like_weights("tv", sc)
    kid, Pk, mxk = knn_prior(onto)
    oid, Po = load_sparse(ONTOLOGUE, onto)
    ids = kid
    Po = align(oid, Po, ids)
    maxpid = pd.Series(mxk, index=kid)
    feat = hit_features(ids, mxk)
    log(f"scorer: {sc.n} TV v2 proteins with GT; test-like effective n {wrep['effective_n']}; "
        f"Ontologue ids == kNN ids: {bool((oid == kid).all()) if len(oid) == len(kid) else False}")
    ctx = {"onto": onto, "sc": sc, "scu": scu, "w": w, "ids": ids, "feat": feat,
           "maxpid_sc": maxpid.reindex(sc.ids).fillna(0).values, "P": {"knn": Pk, "ontologue": Po},
           "raw": {}, "cur": {}, "curu": {}, "tau": {}, "pp": {}, "ppu": {}}
    for name, P in ctx["P"].items():
        raw = raw_curves(sc, P, ids)
        cur = to_cur(raw)
        ref = sc.curves(P, ids)
        assert all(np.allclose(cur[x]["prec"], ref[x]["prec"], atol=1e-6) and
                   np.array_equal(cur[x]["has"], ref[x]["has"]) for x in ASP), "raw_curves != Scorer.curves"
        rawu = raw_curves(scu, P, ids)
        tau = thresholds(sc, cur, w)
        ctx["raw"][name], ctx["cur"][name], ctx["curu"][name], ctx["tau"][name] = raw, cur, to_cur(rawu), tau
        ctx["pp"][name], ctx["ppu"][name] = per_protein(raw, tau), per_protein(rawu, tau)
        ft, fp = sc.fmax(cur, weights=w), sc.fmax(cur)
        log(f"{name}: TV v2 test-like mean Fmax {ft['mean']:.4f} (MF {ft['MF']:.4f} BP {ft['BP']:.4f} CC "
            f"{ft['CC']:.4f}), plain {fp['mean']:.4f}; test-like tau " +
            json.dumps({x: float(ks.TAU[t]) for x, t in tau.items()}), file=True)
    log("END step 1", file=True)
    summary = {"inputs": {"knn": str(KNN_TV / "tv_knn.npz"), "ontologue": str(ONTOLOGUE), "higo": str(HIGO),
                          "hits": str(HITS)},
               "fmax": {n: {"testlike": sc.fmax(ctx["cur"][n], weights=w)["mean"], "plain": sc.fmax(ctx["cur"][n])["mean"],
                            "tau": {x: float(ks.TAU[t]) for x, t in ctx["tau"][n].items()}} for n in ctx["P"]}}
    if a.only in (None, "A"):
        log("START step 2: analysis A (coverage)", file=True)
        cov, sa = analysis_a(ctx)
        cov.to_csv(OUT / "coverage.tsv", sep="\t", index=False, float_format="%.5g")
        show = cov[(cov.aspect == "mean")][["predictor", "grouping", "group", "n", "testlike_share", "pred_cov",
                                            "prec_ia", "rec_ia", "f1_ia", "prec_u", "rec_u", "f1_u", "fp_ia_mean",
                                            "group_fmax"]]
        log("A coverage groups (mean over aspects):\n" + show.round(4).to_string(index=False))
        summary["A"] = sa
        log("END step 2: wrote runs/day1_analyses/coverage.tsv", file=True)
    if a.only in (None, "B"):
        log("START step 3: analysis B (abstention + fallbacks, rules pre-specified in this log)", file=True)
        ab_tab, fb, sb = analysis_b(ctx, a.boot)
        ab_tab.to_csv(OUT / "abstention.tsv", sep="\t", index=False, float_format="%.5g")
        fb.to_csv(OUT / "fallbacks.tsv", sep="\t", index=False, float_format="%.5g")
        log("B fallbacks:\n" + fb.round(4).to_string(index=False))
        summary["B"] = sb
        log("END step 3: wrote runs/day1_analyses/abstention.tsv, fallbacks.tsv", file=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done in {time.time() - T0:.0f}s", file=True)


if __name__ == "__main__":
    main()
