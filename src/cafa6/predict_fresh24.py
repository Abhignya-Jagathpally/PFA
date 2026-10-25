"""Predictions of the registered methods on fresh24 (data/fresh24/fresh24.fasta; registration runs/fresh24_looks.md).

Every file goes to runs/fresh24/preds/<name>.npz in the compare_paper format ({ids, data, indices, indptr, shape,
terms, meta}; read with compare_paper.load_npz_pred). Nothing is fitted or tuned here.

  ontologue      cafa6.predict_r2.Predictor exactly as the web app builds it (default references data/serve,
                 HiGO on cuda under bf16, stacker on CPU), called through Predictor.predict on chunks of the FASTA
                 with the organism given as the NCBI taxon of proteins.tsv (what a UniProt header's OX= gives).
                 The same calls also write the parts: ontologue_knn (step 1a, the sharpened kNN prior over the
                 train+val references) and ontologue_higo (step 1b, the HiGO head alone: its top 500 terms per
                 protein, as runs/l3_c0_tv2/pred_test_top500.npz). A spot-check runs predict() one protein per
                 request on 20 proteins and compares with the chunked rows.
  baselines      naive, blast_knn, diamondscore (GO-GPT training references) and blast_knn_trainval
                 (train + val references), the definitions of compare_paper.cmd_baselines, on DIAMOND hits of
                 fresh24 against each reference set (--more-sensitive -e 1e-3 -k 51, --dbsize = training letters).
  gogpt          runs/fresh24/gogpt_{sample10,beam}/ outputs of benchmarks/gogpt_eval.py -> gogpt_sample10.npz,
                 gogpt_beam.npz (term frequencies over the ten samples; beam sets with score 1).
  import-tsv     external (protein, GO id, score) rows -> preds/NAME.npz (compare_paper.cmd_import_tsv).

Usage (from cafa6/, PYTHONPATH=src):
  python -m cafa6.predict_fresh24 ontologue [--gpu 0] [--chunk 6100]
  python -m cafa6.predict_fresh24 baselines
  python -m cafa6.predict_fresh24 gogpt
  python -m cafa6.predict_fresh24 import-tsv --tsv X.tsv --name NAME [--meta JSON]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRESH = ROOT / "data" / "fresh24"
OUT = ROOT / "runs" / "fresh24"
PREDS = OUT / "preds"
WORK = Path("/dev/shm/fresh24_work")
LOGF = ROOT / "logs" / "fresh24_look1.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")


def _cp():
    from cafa6 import compare_paper as cp
    cp.PREDS = PREDS
    return cp


def records():
    import pandas as pd
    from cafa6.data import read_fasta
    fa = read_fasta(FRESH / "fresh24.fasta")
    pt = pd.read_csv(FRESH / "proteins.tsv", sep="\t", usecols=["accession", "taxon"])
    tax = dict(zip(pt.accession, pt.taxon.astype(int)))
    return [{"id": i, "seq": s, "taxon": tax[i]} for i, s in zip(fa.id, fa.seq)]


# ----------------------------------------------------------------------------------------------- Ontologue

def cmd_ontologue(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import numpy as np
    import scipy.sparse as sp
    from cafa6 import predict_r2 as pr2
    cp = _cp()
    recs = records()
    err = pr2.validate(recs)
    if err:
        raise SystemExit("invalid input: " + "; ".join(err[:5]))
    t0 = time.time()
    pr = pr2.Predictor(higo_device=pr2.resolve_device("auto"))
    log(f"ontologue: Predictor ready on {pr.higo_device} ({time.time() - t0:.0f}s); {len(recs)} proteins, "
        f"chunks of {a.chunk}")
    cap = {}
    knn_orig, higo_orig, fmt_orig = pr2.knn_prior, pr.higo_scores, pr._format

    def knn_cap(*args, **kw):
        r = knn_orig(*args, **kw)
        cap["Pk"] = r[0]
        return r

    def higo_cap(*args, **kw):
        r = higo_orig(*args, **kw)
        cap["Dh"] = r
        return r

    pr2.knn_prior, pr.higo_scores = knn_cap, higo_cap
    pr._format = lambda recs, *x: [{} for _ in recs]
    WORK.mkdir(parents=True, exist_ok=True)
    parts = {"ontologue": [], "ontologue_knn": [], "ontologue_higo": []}
    for c0 in range(0, len(recs), a.chunk):
        ck = WORK / f"ontologue_chunk{c0:06d}.npz"
        sub = recs[c0:c0 + a.chunk]
        if ck.exists():
            z = np.load(ck)
            mats = [sp.csr_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]), shape=tuple(z[f"{k}_shape"]))
                    for k in parts]
            log(f"ontologue: chunk {c0} loaded from {ck}")
        else:
            t = time.time()
            pr.predict(sub)
            P = pr.last_P.tocsr().astype(np.float32)
            Pk = cap["Pk"].tocsr().astype(np.float32)
            Dt = pr2.topk_csr(cap["Dh"].astype(np.float32), int(pr.cfg.get("pred_topk", 500)))
            r, c = np.nonzero(Dt)
            H = sp.csr_matrix((Dt[r, c], (r, pr.vcols[c])), shape=(len(sub), pr.nT))
            mats = [P, Pk, H]
            np.savez(ck, **{f"{k}_{f}": getattr(m, f) for k, m in zip(parts, mats) for f in ("data", "indices", "indptr")},
                     **{f"{k}_shape": np.array(m.shape) for k, m in zip(parts, mats)})
            log(f"ontologue: chunk {c0}-{c0 + len(sub)} done in {time.time() - t:.0f}s "
                f"(scores {P.nnz:,}, kNN {Pk.nnz:,}, HiGO {H.nnz:,})")
        for k, m in zip(parts, mats):
            parts[k].append(m)
    ids = [r["id"] for r in recs]
    meta = {"ontologue": {"method": "Ontologue (served predictor)", "def": "cafa6.predict_r2.Predictor.predict, "
                          "default data/serve references (GO-GPT train+val), HiGO cuda bf16, twin stacker on CPU, "
                          "propagated scores >= 0.01", "chunk": a.chunk, "higo_device": pr.higo_device},
            "ontologue_knn": {"method": "Ontologue step 1a: kNN prior", "def": "predict_r2.knn_prior T=5 k=50 "
                              "aspect-normalised over train+val references, captured from the same predict() calls"},
            "ontologue_higo": {"method": "Ontologue step 1b: HiGO head alone", "def": "HiGO l3_c0_tv2 probabilities "
                               "(cuda bf16), top 500 per protein (predict_r2.topk_csr), same predict() calls"}}
    for k in parts:
        cp.save_pred(k, sp.vstack(parts[k]).tocsr(), ids, pr.onto, meta[k])
    log(f"ontologue: wrote preds/ontologue*.npz ({time.time() - t0:.0f}s)")

    # spot-check: predict() one protein per request (the web app path) vs the chunked rows
    pr2.knn_prior, pr.higo_scores, pr._format = knn_orig, higo_orig, fmt_orig
    pr.verbose = False
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(recs), 20, replace=False))
    Pall = sp.vstack(parts["ontologue"]).tocsr()
    res = []
    for i in pick:
        out = pr.predict([recs[i]])[0]
        row = Pall[i]
        new = sp.csr_matrix(([s["score"] for s in out["scores"]],
                             ([0] * len(out["scores"]), [pr.onto.idx[s["term"]] for s in out["scores"]])),
                            shape=(1, pr.nT))
        ref = row.copy()
        ref.data = np.round(ref.data, 4)
        d = abs(new - ref)
        res.append({"id": recs[i]["id"], "len": len(recs[i]["seq"]), "n_terms_single": len(out["scores"]),
                    "n_terms_chunked": int(row.nnz), "max_abs_diff": float(d.max()) if d.nnz else 0.0,
                    "same_terms": bool(set(new.indices) == set(row.indices))})
    (OUT / "ontologue_spotcheck.json").write_text(json.dumps(res, indent=1))
    mx = max(r["max_abs_diff"] for r in res)
    log(f"ontologue: spot-check 20 proteins, single-request predict() vs chunked rows (4-decimal output): "
        f"max abs diff {mx:.2e}, same term sets {sum(r['same_terms'] for r in res)}/20")


# ----------------------------------------------------------------------------------------------- baselines

def diamond(db_fa: Path, db: Path, q_fa: Path, out: Path, dbsize: int, threads: int = 64, k: int = 51):
    import subprocess
    if not Path(str(db) + ".dmnd").exists():
        subprocess.run([str(ROOT / "tools/diamond"), "makedb", "--in", str(db_fa), "-d", str(db), "--quiet",
                        "-p", str(threads)], check=True)
    subprocess.run([str(ROOT / "tools/diamond"), "blastp", "-d", str(db), "-q", str(q_fa), "-o", str(out),
                    "--more-sensitive", "-e", "1e-3", "-k", str(k), "-p", str(threads), "--quiet",
                    "--tmpdir", str(out.parent), "--dbsize", str(dbsize),
                    "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)


def cmd_baselines(a):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from cafa6.knn_sharp import SRC
    from cafa6.temporal_val import GO, full_labels
    cp = _cp()
    onto = cp.ontology()
    WORK.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split"])
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dbsize = int(tr.seq.str.len().sum())
    q_fa = WORK / "fresh24.fa"
    recs = records()
    with open(q_fa, "w") as f:
        for r in recs:
            f.write(f">{r['id']}\n{r['seq']}\n")
    te = np.array([r["id"] for r in recs])
    hits = {}
    for name, refs in (("train", tr), ("trainval", pd.concat([tr, va]))):
        hf = WORK / f"hits_{name}.tsv"
        if not hf.exists():
            fa = WORK / f"refs_{name}.fa"
            with open(fa, "w") as f:
                for pid, s in zip(refs.id, refs.seq):
                    f.write(f">{pid}\n{s}\n")
            t = time.time()
            diamond(fa, WORK / f"refs_{name}", q_fa, hf, dbsize)
            log(f"baselines: DIAMOND fresh24 vs {name} references ({len(refs):,}) in {time.time() - t:.0f}s")
        h = pd.read_csv(hf, sep="\t", header=None, names=["q", "s", "pident", "bits", "evalue", "qcov"])
        hits[name] = h[h.q != h.s].sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
        log(f"baselines: {name} hits {len(hits[name]):,} for {hits[name].q.nunique():,}/{len(te):,} queries")

    db_ids = tr.id.values
    full = full_labels(GO(), "train")
    L = onto.label_matrix([full.get(p, set()) for p in db_ids])
    qi = {p: i for i, p in enumerate(te)}
    di = {p: i for i, p in enumerate(db_ids)}

    cnt = np.asarray(L.sum(0)).ravel()
    roots = [onto.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    n_asp = np.asarray(L[:, roots].sum(0)).ravel()
    freq = cnt / n_asp[onto.ns]
    keep = np.where(freq >= 0.005)[0]
    row = sp.csr_matrix((freq[keep].astype(np.float32), (np.zeros(len(keep), int), keep)), shape=(1, len(onto.terms)))
    cp.save_pred("naive", sp.vstack([row] * len(te)).tocsr(), te, onto,
                 {"method": "naive", "def": "per-aspect term frequency among training proteins annotated in the aspect"})

    h = hits["train"]
    q, s = h.q.map(qi).values, h.s.map(di).values
    bits, pid = h.bits.values, h.pident.values
    order = np.lexsort((-bits, -pid, q))
    qq, ss, pp = q[order], s[order], pid[order]
    first = np.r_[True, qq[1:] != qq[:-1]]
    W = sp.csr_matrix((pp[first] / 100.0, (qq[first], ss[first])), shape=(len(te), len(db_ids)))
    cp.save_pred("blast_knn", (W @ L).tocsr(), te, onto,
                 {"method": "BLAST-KNN", "def": "compare_paper: best hit by pident (ties: bitscore) among DIAMOND hits "
                  "e<=1e-3 to GO-GPT training proteins; all propagated terms of the hit, score = pident/100"})
    h50 = h.groupby("q", sort=False).head(50)
    W = sp.csr_matrix((h50.bits.values, (h50.q.map(qi).values, h50.s.map(di).values)), shape=(len(te), len(db_ids)))
    den = np.asarray(W.sum(1)).ravel()
    P = (W @ L).tocsr()
    P.data = (P.data / np.repeat(np.maximum(den, 1e-12), np.diff(P.indptr))).astype(np.float32)
    cp.save_pred("diamondscore", P, te, onto,
                 {"method": "DIAMONDScore", "def": "sum_j bits_j y_jt / sum_j bits_j over DIAMOND hits (top 50, e<=1e-3) "
                  "to GO-GPT training proteins"})

    full_v = full_labels(GO(), "val")
    ref = {**{p: full[p] for p in db_ids if p in full}, **full_v}
    h = hits["trainval"]
    h = h[h.s.isin(set(ref))].sort_values(["q", "pident", "bits"], ascending=[True, False, False]).drop_duplicates("q")
    Lb = sp.diags(h.pident.values / 100.0) @ onto.label_matrix([ref[x] for x in h.s])
    Lb = Lb.tocoo()
    rows = h.q.map(qi).values
    P = sp.csr_matrix((Lb.data, (rows[Lb.row], Lb.col)), shape=(len(te), len(onto.terms)))
    cp.save_pred("blast_knn_trainval", P, te, onto,
                 {"method": "BLAST-KNN (train+val references)", "def": "as blast_knn, references = GO-GPT train + val"})
    log("baselines: wrote naive, blast_knn, diamondscore, blast_knn_trainval")


# ----------------------------------------------------------------------------------------------- GO-GPT

def cmd_gogpt(a):
    import gzip
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    cp = _cp()
    onto = cp.ontology()
    d = OUT / "gogpt_sample10"
    rows = []
    for f in sorted(d.rglob("samples_shard*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            for asp in cp.ASPECTS:
                runs = r[asp]
                cnt = {}
                for terms in runs:
                    for t in set(terms):
                        cnt[t] = cnt.get(t, 0) + 1
                rows += [(r["protein_id"], t, round(c / len(runs), 3)) for t, c in cnt.items()]
    if rows:
        df = pd.DataFrame(rows, columns=["id", "term", "score"]).drop_duplicates(["id", "term"])
        ids = np.array(sorted(df.id.unique()))
        pos = {p: i for i, p in enumerate(ids)}
        r_, c_, v_ = [], [], []
        for p, t, s in zip(df.id, df.term, df.score):
            for j in onto.canon(t):
                r_.append(pos[p]); c_.append(j); v_.append(s)
        M = sp.csr_matrix((np.array(v_, np.float32), (r_, c_)), shape=(len(ids), len(onto.terms)))
        M.sum_duplicates()
        cp.save_pred("gogpt_sample10", M, ids, onto,
                     {"method": "GO-GPT ten-sample frequency", "def": "released checkpoint, T 0.7, top-k 20, 10 samples "
                      "per aspect, score = fraction of samples with the term (runs/gogpt_sample10 settings)"})
    sets = {}
    for f in sorted((OUT / "gogpt_beam").rglob("preds_shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            sets[r["protein_id"]] = {t for k in cp.ASPECTS for t in r.get(k, [])}
    if sets:
        M, ids = cp.sets_matrix(sets, onto)
        cp.save_pred("gogpt_beam", M, ids, onto, {"method": "GO-GPT beam 5", "def": "released predictor setting "
                                                  "(beam 5, length penalty 0.3), score 1.0"})


def cmd_import(a):
    cp = _cp()
    cp.cmd_import_tsv(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ontologue")
    o.add_argument("--gpu", default="0")
    o.add_argument("--chunk", type=int, default=6100)
    sub.add_parser("baselines")
    sub.add_parser("gogpt")
    t = sub.add_parser("import-tsv")
    t.add_argument("--tsv", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--cols", type=int, nargs=3, default=[0, 1, 2])
    t.add_argument("--header", action="store_true")
    t.add_argument("--meta", default="")
    a = ap.parse_args()
    {"ontologue": cmd_ontologue, "baselines": cmd_baselines, "gogpt": cmd_gogpt, "import-tsv": cmd_import}[a.cmd](a)


if __name__ == "__main__":
    main()
