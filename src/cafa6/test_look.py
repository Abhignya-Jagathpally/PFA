"""One budgeted test look for a pre-registered HiGO-R2 pipeline on the GO-GPT temporal test.

Scores a single prediction file (stacker_tv.save_sparse format: ids, data, indices, indptr, shape, terms) with
knn_sharp.Scorer (in-memory cafaeval, prop=max, IA-weighted Fmax) next to the two GO-GPT references on the same
proteins, and runs a paired protein bootstrap of the difference against each:
  gogpt_sample10   runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz  (ten samples, term frequency as score)
  gogpt_beam       runs/gogpt_cafa5/preds_shard*.jsonl             (released checkpoint, beam search)
Per identity bin rows use the train-only best-hit identity (data/proc_gogpt_tv2/knn/test_knn.npz), the binning
GO-GPT test numbers have been reported with so far.

Outputs runs/test_look/<name>/{summary.tsv, per_bin.tsv, result.json}; appends one line to runs/test_looks.md.

  python -m cafa6.test_look --pred runs/stacker_tv/l3_c0/pred_test_twin.npz --name look3_l3c0_twin --look 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6 import knn_sharp as ks

ROOT = ks.ROOT
T0 = time.time()


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:7.1f}s] {msg}", flush=True)


def load_pred(path: Path, onto) -> tuple[np.ndarray, sp.csr_matrix]:
    z = np.load(path, allow_pickle=False)
    P = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    cols = np.array([onto.idx[t] for t in z["terms"]])
    P = P.tocoo()
    P = sp.csr_matrix((P.data, (P.row, cols[P.col])), shape=(P.shape[0], len(onto.terms)))
    return z["ids"].astype(str), P


def load_long(path: Path, onto) -> tuple[np.ndarray, sp.csr_matrix]:
    df = pd.read_csv(path, sep="\t", header=None, names=["id", "term", "score"])
    df = df[df.term.isin(onto.idx)]
    ids = np.array(sorted(df.id.unique()))
    pos = {p: i for i, p in enumerate(ids)}
    P = sp.csr_matrix((df.score.values.astype(np.float32), (df.id.map(pos).values, df.term.map(onto.idx).values)),
                      shape=(len(ids), len(onto.terms)))
    return ids, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--look", type=int, required=True, help="test look number for runs/test_looks.md")
    ap.add_argument("--desc", default="", help="pipeline description for the log line")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    out = ROOT / "runs" / "test_look" / a.name
    out.mkdir(parents=True, exist_ok=True)

    onto = ks.Ontology()
    sc = ks.Scorer(onto, ks.gt_frame("test"))
    log(f"GO-GPT test ground truth: {sc.n} proteins")
    ident = pd.Series(np.load(ROOT / "data/proc_gogpt_tv2/knn/test_knn.npz")["max_pident"].astype(float),
                      index=np.load(ROOT / "data/proc_gogpt_tv2/knn/test_knn.npz")["ids"].astype(str))

    preds = {}
    ids, P = load_pred(Path(a.pred), onto)
    preds[a.name] = (ids, P)
    log(f"{a.name}: {len(ids)} proteins, {P.nnz} scores from {a.pred}")
    ids, P = load_long(ROOT / "runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz", onto)
    preds["gogpt_sample10"] = (ids, P)
    log(f"gogpt_sample10: {len(ids)} proteins, {P.nnz} scores")
    P, ids = ks.gogpt_matrix(onto)
    preds["gogpt_beam"] = (ids, P)
    log(f"gogpt_beam: {len(ids)} proteins, {P.nnz} scores")

    curs, rows, bins = {}, [], []
    for k, (ids, P) in preds.items():
        curs[k] = sc.curves(P, ids)
        f = sc.fmax(curs[k])
        rows.append({"method": k, **{x: f[x] for x in ("MF", "BP", "CC")}, "mean": f["mean"]})
        bins.append({"method": k, **ks.bin_table(sc, curs[k], ident)})
        log(f"{k:16s} MF {f['MF']:.4f}  BP {f['BP']:.4f}  CC {f['CC']:.4f}  mean {f['mean']:.4f}")
    summ, per_bin = pd.DataFrame(rows), pd.DataFrame(bins)
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    per_bin.to_csv(out / "per_bin.tsv", sep="\t", index=False)
    log("per identity bin (train-only best hit):\n" + per_bin.round(4).to_string())

    res = {"name": a.name, "pred": a.pred, "look": a.look, "mean": rows[0]["mean"],
           "aspects": {x: rows[0][x] for x in ("MF", "BP", "CC")}}
    for ref in ("gogpt_sample10", "gogpt_beam"):
        bs = sc.bootstrap(curs[a.name], curs[ref], reps=a.reps)
        res[f"minus_{ref}"] = bs
        log(f"{a.name} - {ref}: {bs['diff']:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}] ({a.reps} paired protein reps)")
    (out / "result.json").write_text(json.dumps(res, indent=2))
    s10, bm = res["minus_gogpt_sample10"], res["minus_gogpt_beam"]
    with open(ks.TEST_LOOKS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | HiGO-R2 pipeline | TEST LOOK #{a.look}: {a.name} "
                f"({a.desc or a.pred}) -> test mean Fmax {res['mean']:.4f}; minus GO-GPT ten-sample "
                f"{s10['diff']:+.4f} [{s10['lo']:+.4f}, {s10['hi']:+.4f}], minus GO-GPT beam {bm['diff']:+.4f} "
                f"[{bm['lo']:+.4f}, {bm['hi']:+.4f}] ({a.reps} reps); runs/test_look/{a.name}\n")
    log(f"done; outputs in {out}")


if __name__ == "__main__":
    main()
