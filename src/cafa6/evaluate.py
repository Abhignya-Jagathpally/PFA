"""CAFA-evaluator wrapper: IA-weighted Fmax per aspect (the Kaggle CAFA-6 metric), plus strata."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from cafaeval.evaluation import cafa_eval

from cafa6.data import PROC, RAW

OBO = RAW / "Train" / "go-basic.obo"
IA = RAW / "IA.tsv"
NS_SHORT = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
logging.getLogger().setLevel(logging.WARNING)


def dense_to_long(scores: np.ndarray, ids, terms, top_k: int = 1500, min_score: float = 0.01) -> pd.DataFrame:
    """Keeps at most top_k terms per protein (Kaggle limit is 1500) with score >= min_score."""
    k = min(top_k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(ids)), k)
    cols = idx.ravel()
    vals = scores[rows, cols]
    keep = vals >= min_score
    terms = np.asarray(terms)
    ids = np.asarray(ids)
    return pd.DataFrame({"id": ids[rows[keep]], "term": terms[cols[keep]], "score": np.round(vals[keep], 3)})


def evaluate(preds: dict[str, pd.DataFrame], gt_file: Path | str, ids_subset=None, n_cpu: int = 16,
             out_dir: Path | None = None, obo: Path = OBO, ia: Path = IA) -> pd.DataFrame:
    """Returns one row per (method, aspect) with Fmax, weighted Fmax, Smin, coverage.

    `ids_subset` restricts both ground truth and predictions (used for homology/IA strata).
    """
    with tempfile.TemporaryDirectory(dir=PROC) as tmp:
        tmp = Path(tmp)
        (tmp / "pred").mkdir()
        gt = pd.read_csv(gt_file, sep="\t", header=None, names=["id", "term"])
        if ids_subset is not None:
            gt = gt[gt.id.isin(set(ids_subset))]
        gt.to_csv(tmp / "gt.tsv", sep="\t", header=False, index=False)
        for name, df in preds.items():
            d = df[df.id.isin(set(gt.id))]
            d[["id", "term", "score"]].to_csv(tmp / "pred" / f"{name}.tsv", sep="\t", header=False, index=False)
        df_all, best = cafa_eval(str(obo), str(tmp / "pred"), str(tmp / "gt.tsv"), ia=str(ia),
                                 norm="cafa", prop="max", n_cpu=n_cpu)
    rows = []
    fb = best["f"].reset_index().set_index(["filename", "ns"])
    sb = best["s"].reset_index().set_index(["filename", "ns"]) if "s" in best else None
    for (fname, ns, tau), r in best["f_w"].iterrows():
        f_row = fb.loc[(fname, ns)]
        rows.append({"method": fname.rsplit(".", 1)[0], "aspect": NS_SHORT.get(ns, ns),
                     "Fmax_w": float(r["f_w"]), "tau_w": float(tau),
                     "Fmax": float(f_row["f"]), "Smin": float(sb.loc[(fname, ns)]["s"]) if sb is not None else np.nan,
                     "cov": float(f_row["cov_max"])})
    res = pd.DataFrame(rows)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        res.to_csv(Path(out_dir) / "metrics.tsv", sep="\t", index=False)
    return res


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """Wide table: per-aspect weighted Fmax and the Kaggle score (mean over MF/BP/CC)."""
    w = res.pivot(index="method", columns="aspect", values="Fmax_w")
    w = w.reindex(columns=[c for c in ("MF", "BP", "CC") if c in w.columns])
    w["kaggle_mean_Fw"] = w.mean(axis=1)
    return w.sort_values("kaggle_mean_Fw", ascending=False)
