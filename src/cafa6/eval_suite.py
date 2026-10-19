"""Evaluation suite for the final HiGO-R2 test predictions (test look #3) against GO-GPT on GO-GPT's temporal test.

Every prediction file below was frozen before this script was written. The script only describes them: nothing
is tuned, selected or changed on test results, and nothing is appended to runs/test_looks.md.

Rows (same 8,630 test proteins, ground truth data/proc_gogpt/gt_test.tsv)
  HiGO-R2            runs/stacker_tv/l3_c0/pred_test_twin.npz (look-3 pipeline: frozen ESM2-35M HiGO head + train+val
                     sharpened kNN, twin stacker fit on TV v2, no calibration)
  GO-GPT ten-sample  runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz (score = fraction of 10 samples with the term)
  GO-GPT beam        runs/gogpt_cafa5/preds_shard*.jsonl (released predictor, beam search; binary sets, score 1)
  [component] kNN prior, train+val refs   data/proc_gogpt_tv2/knn_trainval/test_knn.npz (p_full, all GO terms)
  [component] HiGO head alone             runs/l3_c0_tv2/pred_test_top500.npz (MCM output, top 500 per protein)
The two components are inputs of HiGO-R2 that happen to exist as frozen test files, not separate models.

Common protocol (cafaeval 1.3 as wired in cafa6.evaluate: prop=max, norm=cafa, tau = 0.01..0.99 step 0.01)
  Scores go through what dense_to_long and the TSV round trip do (scores < 0.01 dropped, top 1,500 per protein,
  rounded to 3 decimals), then max-propagation inside each namespace of go-basic_gogpt.obo (is_a + part_of).
  Ground truth is ancestor-closed. For aspect a only the n_a proteins with ground truth in a count
  (MF 2080, BP 5819, CC 3440); a protein without any prediction in a has precision undefined and recall 0.
  "mean" is the unweighted mean of the MF, BP and CC values. Two term weightings are used throughout:
  unweighted (u: every term in the namespace weighs 1, roots included, as cafaeval toi with orphans) and
  IA-weighted (w: term weight = IA from runs/gogpt_cafa5/ref/IA.txt; IA = 0 terms drop out).

Metric definitions (for one aspect; W_i(tau) = weight of protein i's terms with score >= tau, T_i(tau) = weight of
those that are in its ground truth, G_i = weight of its ground truth, n(tau) = proteins with W_i(tau) > 0)
  Fmax        precision P(tau) = mean over the n(tau) covered proteins of T_i/W_i, recall R(tau) = sum_i T_i/G_i / n_a,
              Fmax = max_tau 2PR/(P+R). Fmax_w is the same with IA weights (the CAFA/Kaggle metric).
  cov         n(tau) / n_a at the Fmax threshold (cov_w: proteins with positive-IA predictions at the Fmax_w tau).
  Rmax        R(0.01), the highest recall a row reaches on the threshold grid (the recall ceiling of its PR curve).
  Fmicro      micro-averaged F: P = sum T / sum W, R = sum T / sum G over all n_a proteins, maximised over tau.
  Smin_w      ru(tau) = sum_i (G_i - T_i) / n_a, mi(tau) = sum_i (W_i - T_i) / n_a with IA weights,
              S = sqrt(ru^2 + mi^2), Smin_w = min over tau with unweighted coverage > 0 (cafaeval drops tau rows
              without coverage). Smin (no suffix) is the same with unit weights, which is the column cafaeval's
              best_s.tsv minimises (column s); Smin_w is the CAFA information-content S-min.
  AUPR_micro  term-centric micro AUPR: every (protein, term) pair with the protein in the n_a proteins and the term in
              the aspect's namespace (all namespace terms, after max-propagation) is one item; positives = all
              ground-truth pairs; pairs without a prediction score 0. Average precision over the distinct scores,
              AP = sum_s (R(s) - R(s')) P(s) with s running down over the distinct score values (the 3-decimal
              grid, plus 0) and s' the next higher value; all pairs tied at a score enter together (one operating
              point per distinct score, the convention of sklearn.average_precision_score). The score-0 point
              (everything predicted, recall 1, precision = base rate) is included, so unpredicted positives count.
              AUPR_micro_w weighs every pair by the term's IA (roots and other IA = 0 terms drop out).
  AUPR_prot   protein-centric PR area: the CAFA averaged curve (R(tau), P(tau)) of Fmax over tau = 0.01..0.99,
              AP = sum_j (R(tau_j) - R(tau_j+1)) P(tau_j) with R(tau_100) = 0 and P = 0 where n(tau) = 0 (R(tau)
              is non-increasing in tau, so this is the step area from recall 0 to R(0.01); the curve is not
              extrapolated beyond R(0.01)). AUPR_prot_w uses the IA-weighted curve.
              GO-GPT beam has only score 1, so its micro curve has two points (score 1, score 0) and its protein
              curve is one point: AUPR_prot = R*P and AUPR_micro ~ R*P plus the base-rate tail. These rows are
              reported but flagged; they measure one operating point, not a ranking.
  F1_i        per-protein F1 at the model's own per-aspect Fmax threshold (unweighted F1 at the Fmax tau, IA-weighted
              F1 at the Fmax_w tau): 2 p_i r_i / (p_i + r_i), p_i = T_i/W_i (0 when W_i = 0), r_i = T_i/G_i.
              Its mean over the n_a proteins is not Fmax (Fmax averages precision only over covered proteins).
  identity    max percent identity of the protein's DIAMOND hits (e <= 1e-3) against the GO-GPT training proteins
              (data/proc_gogpt_tv2/knn/test_knn.npz max_pident) and, as a second view, against train + val
              (knn_trainval/test_knn.npz; HiGO-R2 uses val proteins as kNN references). No hit = 0.
              Bins: no_hit, <30, 30-50, 50-80, 80-95, 95-99.9, 100 (cafa6.temporal_val.ident_bins).
  Spearman    rank correlation (average ranks for ties) between F1_i and identity over the n_a proteins of an aspect;
              "mean" = mean of the three aspect correlations.
  violations  on the raw prediction matrix (before cut, rounding or propagation, after mapping terms to the
              ontology): pairs (protein, child c, direct parent p) over the is_a/part_of edges inside one namespace,
              counted where score(c) > 0; a violation is score(c) > score(p) + 1e-6 (a missing parent scores 0).
              Rate = violations / counted pairs; also the share of proteins with any nonzero score that have >= 1
              violation. A second rate counts only pairs whose parent is in the model's own output vocabulary.
  retrieval   GO-relation retrieval from learned term vectors: for each vocabulary term, rank all other vocabulary
              terms by cosine similarity; targets = its direct parents + children (is_a, part_of, same namespace in
              the full GO graph) that are in the vocabulary, or all terms within 2 undirected hops of it (parents,
              children, grandparents, grandchildren, siblings, co-parents). Precision@k = hits in the top k / k,
              Recall@k = hits / targets, averaged over terms with >= 1 target. Vectors: HiGO term_emb e_t (raw
              parameter), HiGO effective query q_t = e_t + mean_{a in anc(t)} e_a (HiGO.queries(), what the model
              uses), the same ancestor-mean operator on untrained N(0, 0.02^2) vectors (5 seeds; how much the
              construction alone gives), a random ranking (expected values), and GO-GPT's decoder token
              embeddings (wte; lm_head too when untied). Each model in its own vocabulary and both on the common one.

Statistics
  Bootstrap: 1,000 paired protein resamples (seed 0; the n = 8,630 ground-truth proteins drawn with replacement,
  the same draws for every row, metric and aspect), percentile 95% CI. Every metric is recomputed from the
  resampled proteins; per-protein F1 uses the full-data threshold. Differences are HiGO-R2 minus a GO-GPT row on the
  same draw.
  Tests: paired permutation (swap) test, 10,000 reps (seed 1): each protein's two predictions (HiGO-R2 and the GO-GPT
  row) are exchanged with probability 1/2 and the metric difference is recomputed; two-sided
  p = (1 + #{|d*| >= |d|}) / (1 + 10,000). Holm-Bonferroni and Benjamini-Hochberg are applied within each family
  (columns p_holm, q_bh) and over both families together (p_holm_all, q_bh_all).
  Primary family (56): {Fmax, Fmax_w, Smin_w, AUPR_micro, AUPR_micro_w, AUPR_prot, AUPR_prot_w} x {MF, BP, CC, mean}
  x {vs ten-sample, vs beam}. Secondary family (44): mean-over-aspects per-protein F1 in each of the 7 train-only
  identity bins x {F1, F1_w} x 2 references (28), and the Spearman(F1, train-only identity) difference x {F1, F1_w}
  x {MF, BP, CC, mean} x 2 references (16).

Cross-check: Fmax, Fmax_w, Fmicro, Fmicro_w, Smin (count-based) and their thresholds are compared with the official
cafaeval output runs/test_look/look3_l3c0_twin/cafaeval/best_*.tsv; s_w is compared at cafaeval's best_s tau. The
HiGO-R2 file given to cafaeval there had scores rounded to 4 decimals, so HiGO-R2 is checked twice: in the
submission format used for every table (3 decimals; must agree to 1e-3) and as a replica of that 4-decimal file
(which isolates the scorer).

Outputs: runs/eval_suite/{headline,auprc,smin,identity_bins,spearman,consistency,retrieval,differences,pvalues,
crosscheck,summary}.tsv and report.md.

--extended (outputs in runs/eval_suite_v2/, same file names and columns; the default run is unchanged)
  Adds the methods of runs/compare_paper, read with cafa6.compare_paper.load_method (the loader behind
  runs/compare_paper/summary.tsv): GO-GPT greedy (runs/gogpt_cafa5/greedy), BioReason-Pro RL
  (preds/bioreason_rl.npz: GO ids over the whole response, the 0.649 S3 row; not the _final variant),
  BLAST-KNN with training and with train+val references, DIAMONDScore and Naive. Every non-component row is a
  comparison: primary family {7 metrics} x {MF, BP, CC, mean} x 8 comparisons (224), secondary family
  {7 bins x 2 F1 + 4 Spearman x 2 F1} x 8 (176). Rows whose raw scores are all one value get the AUPR flag.
  Cross-check: Fmax_w, Smin_w, Fmax and the Fmax_w threshold of every row with a compare_paper key against
  runs/compare_paper/summary.tsv (S3_all, 4 decimals; required < 1e-3). Retrieval adds BioReason-Pro's GO
  graph encoder (external/BioReason-Pro-github GOGraphEncoderUnified with go_encoder.pt of the RL checkpoint):
  per-term output of GAT x3 + output projection (the input of its cross-attention reducer), the last GAT layer,
  the precomputed term embeddings it starts from, and the same encoder at fresh initialisation (5 seeds).

  CAFA6_THREADS=24 PYTHONPATH=src ../bin/python -m cafa6.eval_suite
  CAFA6_THREADS=24 PYTHONPATH=src ../bin/python -m cafa6.eval_suite --extended
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("CAFA6_THREADS", "24")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from cafa6 import knn_sharp as ks
from cafa6 import test_look as tl
from cafa6.data import load_pred as load_run_pred

ROOT = ks.ROOT
OUT = ROOT / "runs" / "eval_suite"
TV2 = ROOT / "data" / "proc_gogpt_tv2"
CAFAEVAL_DIR = ROOT / "runs" / "test_look" / "look3_l3c0_twin" / "cafaeval"
HIGO_RUN = ROOT / "runs" / "l3_c0_tv2"
GOGPT_SNAP = Path.home() / ".cache/huggingface/hub/models--wanglab--gogpt/snapshots/56dc8801f62c0bbd8092c9c0718afd129d2be6bc"
ASPECTS = ("MF", "BP", "CC")
COLS = ASPECTS + ("mean",)
TAU = ks.TAU
NLVL = 1001
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
B_REPS, B_SEED = 1000, 0
P_REPS, P_SEED = 10000, 1
REFS = ("GO-GPT ten-sample", "GO-GPT beam")
MAIN = "HiGO-R2"
MODELS = (MAIN, "GO-GPT ten-sample", "GO-GPT beam", "[component] kNN prior (train+val refs)",
          "[component] HiGO head alone")
CAFAEVAL_NAME = {MAIN: "HiGO-R2.tsv", "GO-GPT ten-sample": "GO-GPT_sample10.tsv", "GO-GPT beam": "GO-GPT_beam.tsv"}
PRIMARY = ("Fmax", "Fmax_w", "Smin_w", "AUPR_micro", "AUPR_micro_w", "AUPR_prot", "AUPR_prot_w")
REPORTED = PRIMARY + ("Smin", "Fmicro", "Fmicro_w", "cov", "cov_w", "Rmax", "Rmax_w")
LOWER_BETTER = {"Smin", "Smin_w"}
FLAGS = {"GO-GPT beam": "beam scores are binary: one operating point, not a ranking"}
T0 = time.time()

OUT_EXT = ROOT / "runs" / "eval_suite_v2"
COMPARE = ROOT / "runs" / "compare_paper"
EXT_ROWS = (("GO-GPT greedy", "gogpt_greedy"), ("BioReason-Pro RL", "bioreason_rl"),
            ("BLAST-KNN (training refs)", "blast_knn"), ("BLAST-KNN (train+val refs)", "blast_knn_trainval"),
            ("DIAMONDScore", "diamondscore"), ("Naive", "naive"), ("SPROF-GO", "sprof_go"),
            ("DeepGOPlus", "deepgoplus"), ("ProteInfer (5-model ensemble)", "proteinfer_e5"), ("DeepGO-SE", "deepgose"),
            ("DeepFRI (sequence CNN)", "deepfri_cnn"), ("InterLabelGO+ (leaked)", "interlabelgo_plus"))
# scored on every metric but kept out of the paired-test families: its released training labels contain 96.1% of the
# test answers (runs/compare_paper/extra_models.md)
NO_TEST = {"InterLabelGO+ (leaked)"}
CP_KEY = {MAIN: "higo_r2", "GO-GPT ten-sample": "gogpt_sample10", "GO-GPT beam": "gogpt_beam", **dict(EXT_ROWS)}
BR_REPO = ROOT / "external" / "BioReason-Pro-github"
BR_CKPT = Path("/dev/shm/cmp_paper_bioreason_rl")
BR_EMB = Path("/dev/shm/cmp_paper_go_emb/go_embeddings")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ loading

def gogpt_vocab(onto) -> np.ndarray:
    tok = json.load(open(GOGPT_SNAP / "go_tokenizer.json"))["token_to_id"]
    cols = sorted({j for t in tok if t.startswith("GO:") for j in onto.canon(t)})
    return np.array(cols, dtype=np.int64)


def load_extended(onto, gv: np.ndarray) -> dict:
    """compare_paper rows via compare_paper.load_method, so the scores are those of runs/compare_paper/summary.tsv."""
    from cafa6 import compare_paper as cp
    out = {}
    for name, key in EXT_ROWS:
        P, ids = cp.load_method(key, onto)
        if P is None:
            log(f"  {name}: no prediction file for compare_paper key {key}; row skipped")
            continue
        if key.startswith("gogpt"):
            voc = gv
        elif key.startswith("bioreason"):
            # free-text GO ids: the model can emit any ontology term
            voc = np.arange(len(onto.terms))
        else:
            z = np.load(cp.PREDS / f"{key}.npz", allow_pickle=False)
            voc = np.unique([c[0] for c in map(onto.canon, z["terms"]) if c]).astype(np.int64)
        out[name] = (ids, P, voc)
    return out


def load_models(onto, extended: bool = False) -> dict:
    """name -> (ids, raw CSR over ontology columns, output-vocabulary column indices)."""
    out = {}
    z = np.load(ROOT / "runs/stacker_tv/l3_c0/pred_test_twin.npz", allow_pickle=False)
    ids, P = tl.load_pred(ROOT / "runs/stacker_tv/l3_c0/pred_test_twin.npz", onto)
    out[MAIN] = (ids, P, np.array([onto.idx[t] for t in z["terms"]]))
    gv = gogpt_vocab(onto)
    ids, P = tl.load_long(ROOT / "runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz", onto)
    out["GO-GPT ten-sample"] = (ids, P, gv)
    P, ids = ks.gogpt_matrix(onto)
    out["GO-GPT beam"] = (ids, P, gv)
    K = ks.load_knn("test", TV2 / "knn_trainval")
    cols = np.array([onto.idx[t] for t in K["terms_full"]])
    pf = K["p_full"].tocoo()
    Pk = sp.csr_matrix((pf.data, (pf.row, cols[pf.col])), shape=(len(K["ids"]), len(onto.terms)))
    out["[component] kNN prior (train+val refs)"] = (K["ids"].astype(str), Pk, cols)
    df = pd.read_parquet(TV2 / "train.parquet", columns=["id", "split"])
    rid = df.id.values[(df.split == "test").values].astype(str)
    M = load_run_pred(HIGO_RUN, "test", dense=False).tocoo()
    vocab = pd.read_csv(TV2 / "vocab.tsv", sep="\t")
    vcols = np.array([onto.idx[t] for t in vocab.term])
    Ph = sp.csr_matrix((M.data, (M.row, vcols[M.col])), shape=(M.shape[0], len(onto.terms)))
    out["[component] HiGO head alone"] = (rid, Ph, vcols)
    if extended:
        out.update(load_extended(onto, gv))
    for k, (ids, P, v) in out.items():
        P = P.tocsr()
        P.sum_duplicates()
        P.eliminate_zeros()
        out[k] = (np.asarray(ids).astype(str), P, v)
        log(f"  {k:40s} {len(ids):5d} proteins, {P.nnz:9d} raw nonzero scores, output vocabulary {len(v)} terms")
    return out


def direct_edges(onto, obo=ks.OBO) -> tuple[np.ndarray, np.ndarray]:
    """(child, parent) column pairs of the direct is_a / part_of edges inside one namespace, as Ontology keeps them."""
    pairs, cur, obsolete = [], None, False
    rel = []

    def flush():
        if cur is not None and not obsolete and cur in onto.idx:
            for p in rel:
                if p in onto.idx and onto.ns[onto.idx[p]] == onto.ns[onto.idx[cur]]:
                    pairs.append((onto.idx[cur], onto.idx[p]))

    for line in open(obo):
        kv = line.strip().split(": ")
        if line.startswith("[Typedef]"):
            flush()
            cur, rel, obsolete = None, [], False
            continue
        if len(kv) < 2:
            continue
        k, v = kv[0], ": ".join(kv[1:])
        if k == "id":
            flush()
            cur, rel, obsolete = v, [], False
        elif cur is None:
            continue
        elif k == "is_obsolete":
            obsolete = True
        elif k == "is_a":
            rel.append(v.split("!")[0].strip())
        elif k == "relationship" and v.startswith("part_of"):
            rel.append(v.split()[1].strip())
    flush()
    e = np.unique(np.array(pairs, dtype=np.int64), axis=0)
    return e[:, 0], e[:, 1]


# ------------------------------------------------------------------------------------------------ ontology consistency

def consistency(P: sp.csr_matrix, child: np.ndarray, parent: np.ndarray, onto, vocab_cols: np.ndarray) -> list[dict]:
    nT = len(onto.terms)
    E = sp.csr_matrix((np.ones(len(child)), (child, parent)), shape=(nT, nT))
    P = P.tocsr().astype(np.float64)
    P.sort_indices()
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr)).astype(np.int64)
    cols, vals = P.indices.astype(np.int64), P.data
    keys = rows * nT + cols
    cnt = np.diff(E.indptr)[cols]
    r2, c2, v2 = np.repeat(rows, cnt), np.repeat(cols, cnt), np.repeat(vals, cnt)
    starts = np.repeat(E.indptr[cols], cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    par = E.indices[starts + off].astype(np.int64)
    kp = r2 * nT + par
    pos = np.minimum(np.searchsorted(keys, kp), len(keys) - 1)
    found = keys[pos] == kp
    ps = np.where(found, vals[pos], 0.0)
    counted = v2 > 0
    viol = counted & (v2 > ps + 1e-6)
    invoc = np.zeros(nT, bool)
    invoc[vocab_cols] = True
    pin = invoc[par]
    nprot_any = len(np.unique(rows[vals > 0]))
    res = []
    for k, a in enumerate(ASPECTS + ("all",)):
        m = counted if a == "all" else counted & (onto.ns[c2] == k)
        v = viol & m
        prot_with = len(np.unique(r2[v]))
        prot_any = nprot_any if a == "all" else len(np.unique(rows[(vals > 0) & (onto.ns[cols] == k)]))
        res.append({"aspect": a, "pairs": int(m.sum()), "violations": int(v.sum()),
                    "rate": float(v.sum() / max(m.sum(), 1)),
                    "proteins_scored": prot_any, "proteins_with_violation": prot_with,
                    "share_proteins": float(prot_with / max(prot_any, 1)),
                    "mean_gap": float((v2[v] - ps[v]).mean()) if v.any() else 0.0,
                    "parent_missing_share": float((~found[v]).mean()) if v.any() else 0.0,
                    "pairs_parent_in_vocab": int((m & pin).sum()),
                    "rate_parent_in_vocab": float((v & pin).sum() / max((m & pin).sum(), 1))})
    return res


# ------------------------------------------------------------------------------------------------ per-protein statistics

def cafaeval_file_replica(P: sp.csr_matrix) -> sp.csr_matrix:
    """The HiGO-R2 TSV that runs/test_look/look3_l3c0_twin/cafaeval was computed from: scores rounded to 4 decimals
    (written as text, read back as float64), kept if >= 0.01, no top-k cut."""
    P = P.tocsr().copy()
    P.data = np.round(P.data, 4).astype(str).astype(np.float64)
    P.data[P.data < 0.01] = 0
    P.eliminate_zeros()
    return P


def processed(sc: ks.Scorer, ids, P: sp.csr_matrix, replica: bool = False) -> sp.csr_matrix:
    """Submission cut/round + max-propagation, rows placed at the scorer's protein positions (as Scorer.curves).
    replica=True replaces the submission step by cafaeval_file_replica (cross-check only)."""
    ids = np.asarray(ids)
    m = np.array([p in sc.pos for p in ids])
    assert len(set(ids[m])) == m.sum(), "duplicate protein ids"
    P = P.tocsr()[np.where(m)[0]]
    P = cafaeval_file_replica(P) if replica else ks.to_submission(P)
    P = sc.o.propagate_max(P).tocoo()
    gi = np.array([sc.pos[p] for p in ids[m]], dtype=np.int64)
    return sp.csr_matrix((P.data, (gi[P.row], P.col)), shape=(sc.n, P.shape[1]))


def build_stats(sc: ks.Scorer, S: sp.csr_matrix) -> dict:
    """Per aspect and weighting, per-protein additive arrays: every metric is a function of their protein sums.
    prec/has/rec/tp/fp/fn are [n_a, 99] over tau; lvl_tp/lvl_fp are [n_a, 1001] pair weights per score level
    (level = round(1000 * score), level 0 = unpredicted pairs of the namespace)."""
    o = sc.o
    nT = len(o.terms)
    S = S.tocoo()
    r, c, v = S.row.astype(np.int64), S.col.astype(np.int64), S.data.astype(np.float64)
    ing = np.isin(r * nT + c, sc.gkey)
    kk = np.searchsorted(TAU, v, side="right")
    lvl = np.clip(np.rint(v * 1000).astype(np.int64), 1, 1000)
    grows = np.repeat(np.arange(sc.n), np.diff(sc.G.indptr))
    gcols = sc.G.indices
    out = {}
    for k, a in enumerate(ASPECTS):
        idx = np.where(sc.in_ns[k])[0]
        nk = len(idx)
        loc = np.full(sc.n, -1, np.int64)
        loc[idx] = np.arange(nk)
        sel = (o.ns[c] == k) & (loc[r] >= 0)
        rl, kq, g, lv, cc = loc[r[sel]], kk[sel], ing[sel], lvl[sel], c[sel]
        gm = o.ns[gcols] == k
        st = {"idx": idx}
        for wn in ("u", "w"):
            wt = np.ones(len(cc)) if wn == "u" else o.ia[cc]
            gwt = np.ones(gm.sum()) if wn == "u" else o.ia[gcols[gm]]
            G = np.bincount(loc[grows[gm]], weights=gwt, minlength=nk)
            tot = float((o.ns == k).sum()) if wn == "u" else float(o.ia[o.ns == k].sum())
            pos = wt > 0
            C = np.bincount(rl[pos] * 100 + kq[pos], weights=wt[pos], minlength=nk * 100).reshape(nk, 100)
            I = np.bincount(rl[pos] * 100 + kq[pos], weights=(wt * g)[pos], minlength=nk * 100).reshape(nk, 100)
            W = np.cumsum(C[:, ::-1], axis=1)[:, ::-1][:, 1:]
            T = np.cumsum(I[:, ::-1], axis=1)[:, ::-1][:, 1:]
            has = W > 0
            prec = np.divide(T, W, out=np.zeros_like(W), where=has)
            rec = np.divide(T, G[:, None], out=np.zeros_like(T), where=G[:, None] > 0)
            TPl = np.bincount(rl * NLVL + lv, weights=wt * g, minlength=nk * NLVL).reshape(nk, NLVL)
            FPl = np.bincount(rl * NLVL + lv, weights=wt * (~g), minlength=nk * NLVL).reshape(nk, NLVL)
            TPl[:, 0] = G - TPl.sum(1)
            FPl[:, 0] = tot - G - FPl.sum(1)
            st[wn] = {"prec": prec, "has": has.astype(np.float64), "rec": rec, "tp": T, "fp": W - T,
                      "fn": G[:, None] - T, "lvl_tp": TPl, "lvl_fp": FPl}
        out[a] = st
    return out


KEYS = ("prec", "has", "rec", "tp", "fp", "fn", "lvl_tp", "lvl_fp")


def summed(st: dict, Wk: np.ndarray) -> dict:
    """Wk [B, n_a] protein weights -> {wn: {key: [B, ...]}, 'ne': [B]}."""
    return {"ne": Wk.sum(1), **{wn: {key: Wk @ st[wn][key] for key in KEYS} for wn in ("u", "w")}}


def first_best(x: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Index of the first tau whose value is within 1e-12 of the best, as cafaeval's idxmax/idxmin on exact ties
    (summation order makes tied thresholds differ by ~1e-16 here)."""
    b = x.max(-1, keepdims=True) if maximize else x.min(-1, keepdims=True)
    ok = (x >= b - 1e-12) if maximize else (x <= b + 1e-12)
    return ok.argmax(-1)


def metrics_from(s: dict) -> dict:
    """Metric values [B] from protein sums (see the module docstring)."""
    ne = s["ne"][:, None]
    covu = s["u"]["has"] > 0
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for wn, suf in (("u", ""), ("w", "_w")):
            d = s[wn]
            P = np.where(d["has"] > 0, d["prec"] / np.where(d["has"] > 0, d["has"], 1), 0.0)
            R = d["rec"] / ne
            F = np.where(P + R > 0, 2 * P * R / np.where(P + R > 0, P + R, 1), 0.0)
            F = np.where(covu, F, 0.0)
            out["Fmax" + suf] = F.max(-1)
            jb = first_best(F)
            out["tau" + suf] = TAU[jb]
            out["cov" + suf] = np.take_along_axis(d["has"], jb[:, None], 1)[:, 0] / ne[:, 0]
            out["Rmax" + suf] = R[:, 0]
            Rn = np.concatenate([R[:, 1:], np.zeros((R.shape[0], 1))], 1)
            out["AUPR_prot" + suf] = ((R - Rn) * P).sum(-1)
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            pm = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), 0.0)
            rm = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)
            fm = np.where(pm + rm > 0, 2 * pm * rm / np.where(pm + rm > 0, pm + rm, 1), 0.0)
            out["Fmicro" + suf] = np.where(covu, fm, 0.0).max(-1)
            Sv = np.sqrt((fn / ne) ** 2 + (fp / ne) ** 2)
            Sv = np.where(covu, Sv, np.inf)
            out["Smin" + suf] = Sv.min(-1)
            out["tau_s" + suf] = TAU[first_best(Sv, maximize=False)]
            TP, FP = d["lvl_tp"][:, ::-1], d["lvl_fp"][:, ::-1]
            TPc, FPc = np.cumsum(TP, -1), np.cumsum(FP, -1)
            tot = TPc[:, -1:]
            Pm = np.where(TPc + FPc > 0, TPc / np.where(TPc + FPc > 0, TPc + FPc, 1), 0.0)
            out["AUPR_micro" + suf] = (TP / np.where(tot > 0, tot, 1) * Pm).sum(-1)
    return out


def eval_weights(stats: dict, W: np.ndarray) -> dict:
    """W [B, n] protein weights over the scorer's proteins -> {aspect: {metric: [B]}} plus 'mean'."""
    res = {a: metrics_from(summed(stats[a], W[:, stats[a]["idx"]])) for a in ASPECTS}
    res["mean"] = {m: np.mean([res[a][m] for a in ASPECTS], axis=0) for m in REPORTED}
    return res


def per_protein_f1(st: dict, j: int, wn: str) -> np.ndarray:
    d = st[wn]
    p = np.where(d["has"][:, j] > 0, d["prec"][:, j], 0.0)
    r = d["rec"][:, j]
    return np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)


# ------------------------------------------------------------------------------------------------ statistics helpers

def ci(x) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[o]))
    out = np.empty(m)
    out[o] = adj
    return out


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind="stable")
    q = np.minimum.accumulate((m * p[o] / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def perm_p(obs: float, null: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


def spearman_rows(f1: np.ndarray, ident: np.ndarray) -> float:
    a, b = rankdata(f1), rankdata(ident)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman_batch(F: np.ndarray, rid: np.ndarray) -> np.ndarray:
    """Rows of F [R, n] against fixed identity ranks rid [n]."""
    Rf = rankdata(F, axis=1)
    Rf = Rf - Rf.mean(1, keepdims=True)
    ri = rid - rid.mean()
    den = np.sqrt((Rf ** 2).sum(1) * (ri ** 2).sum())
    return np.where(den > 0, (Rf @ ri) / np.where(den > 0, den, 1), np.nan)


def md(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


# ------------------------------------------------------------------------------------------------ GO retrieval

def retrieval_scores(X: np.ndarray, cols: np.ndarray, adj1: sp.csr_matrix, adj2: sp.csr_matrix,
                     ks_=(1, 5, 10)) -> dict:
    """X [V, d] vectors of the terms at ontology columns cols. Ranks the other V-1 terms by cosine similarity."""
    X = X.astype(np.float32)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    V = len(cols)
    A1 = adj1[cols][:, cols].tocsr()
    A2 = adj2[cols][:, cols].tocsr()
    kmax = max(ks_)
    top = np.empty((V, kmax), np.int64)
    for s in range(0, V, 2048):
        sim = Xn[s:s + 2048] @ Xn.T
        sim[np.arange(sim.shape[0]), np.arange(s, s + sim.shape[0])] = -np.inf
        part = np.argpartition(-sim, kmax, axis=1)[:, :kmax]
        ps = np.take_along_axis(sim, part, 1)
        o = np.lexsort((part, -ps), axis=1)
        top[s:s + sim.shape[0]] = np.take_along_axis(part, o, 1)
    res = {}
    for name, A in (("direct", A1), ("2hop", A2)):
        nt = np.diff(A.indptr)
        q = np.where(nt > 0)[0]
        hit = np.zeros((V, kmax), bool)
        for i in q:
            hit[i] = np.isin(top[i], A.indices[A.indptr[i]:A.indptr[i + 1]])
        res[f"{name}_queries"] = int(len(q))
        res[f"{name}_mean_targets"] = float(nt[q].mean())
        for k in ks_:
            h = hit[q, :k].sum(1)
            res[f"{name}_P@{k}"] = float((h / k).mean())
            res[f"{name}_R@{k}"] = float((h / nt[q]).mean())
            res[f"{name}_random_P@{k}"] = float((nt[q] / (V - 1)).mean())
            res[f"{name}_random_R@{k}"] = float(k / (V - 1))
    return res


def higo_vectors(onto) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    """term_emb, effective queries (model.queries()), vocab columns, row-normalised ancestor-mean operator."""
    import torch
    import yaml
    from cafa6.model import build_higo, load_trainable
    torch.set_num_threads(int(_THREADS))
    cfg = yaml.safe_load(open(HIGO_RUN / "config.yaml"))
    vocab = pd.read_csv(TV2 / "vocab.tsv", sep="\t")
    anc = np.load(TV2 / "go_ancestors.npy")
    tmap = json.load(open(HIGO_RUN / "taxa.json"))
    sd = torch.load(HIGO_RUN / "trainable_weights.pt", map_location="cpu")
    E = sd["term_emb"].float().numpy()
    nV = len(vocab)
    deg = np.bincount(anc[:, 0], minlength=nV).astype(np.float64)
    Aop = sp.csr_matrix((1.0 / np.maximum(deg[anc[:, 0]], 1), (anc[:, 0], anc[:, 1])), shape=(nV, nV))
    q_np = E + Aop @ E
    how = "numpy e + A e (build_higo failed)"
    try:
        model = build_higo(cfg, vocab, anc, len(tmap) + 1, None)
        load_trainable(model, sd)
        with torch.no_grad():
            q = model.queries().float().numpy()
        err = float(np.abs(q - q_np).max())
        how = "cafa6.model.build_higo + load_trainable, HiGO.queries()"
    except Exception as ex:  # backbone not loadable offline
        log(f"  build_higo failed ({ex!r}); using e + A e from the weights")
        q, err = q_np, 0.0
    vcols = np.array([onto.idx[t] for t in vocab.term])
    return E, q, vcols, Aop, err, how


def gogpt_vectors(onto) -> tuple[dict, np.ndarray, str]:
    import torch
    ck = torch.load(GOGPT_SNAP / "model.ckpt", map_location="cpu", mmap=True, weights_only=True)
    sd = ck["state_dict"]
    wte = sd["model.transformer.wte.weight"].float().numpy()
    head = sd["model.lm_head.weight"].float().numpy()
    tok = json.load(open(GOGPT_SNAP / "go_tokenizer.json"))["token_to_id"]
    rows, cols, seen = [], [], set()
    for t, i in sorted(tok.items(), key=lambda x: x[1]):
        if not t.startswith("GO:"):
            continue
        c = onto.canon(t)
        if len(c) == 1 and c[0] not in seen:
            rows.append(i); cols.append(c[0]); seen.add(c[0])
    rows = np.array(rows)
    tied = bool(np.array_equal(wte, head))
    vecs = {"GO-GPT token embedding (wte)": wte[rows]}
    if not tied:
        vecs["GO-GPT output embedding (lm_head)"] = head[rows]
    note = (f"{len(tok)} tokens, {len(rows)} GO tokens mapped to a unique non-obsolete term of go-basic_gogpt.obo; "
            f"lm_head {'tied to' if tied else 'separate from'} wte; dim {wte.shape[1]}")
    return vecs, np.array(cols), note


def bioreason_retrieval(onto, adj1, adj2, common: np.ndarray, seeds: int = 5) -> tuple[list, dict]:
    """Per-term vectors of BioReason-Pro's GO graph encoder (the authors' GOGraphEncoderUnified, RL checkpoint
    weights go_encoder.pt), on CPU. The encoder runs GAT x3 over the obonet graph of the authors' go-basic.obo
    (all relation types) on the precomputed 2560-d term embeddings, projects back to 2560-d and reduces the
    43k term vectors to 200 with cross-attention; only those 200 reach the LLM. The reduced output is
    recomputed and compared with the cached go_embedding.pt of the checkpoint."""
    import importlib.util

    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(_THREADS))
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("bioreason_go_graph_encoder",
                                                  BR_REPO / "bioreason2/models/go_graph_encoder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    enc = mod.GOGraphEncoderUnified(str(BR_REPO / "bioreason2/dataset/go-basic.obo"), str(BR_EMB), hidden_dim=512,
                                    num_gat_layers=3, num_heads=8, num_reduced_embeddings=200, embedding_dim=2560,
                                    embeddings_load_to="cpu")
    enc.load_state_dict(torch.load(BR_CKPT / "go_encoder.pt", map_location="cpu", weights_only=True), strict=True)
    enc.eval()
    x0 = enc._get_precomputed_embeddings(enc.all_go_terms).float()
    ei = enc._get_unified_edge_index()
    rel = pd.Series([k for _, _, k in enc.go_graph.edges(keys=True)]).value_counts()
    log(f"  BioReason-Pro encoder built in {time.time() - t0:.0f}s: {len(enc.all_go_terms)} terms, {ei.shape[1]} "
        f"edges ({', '.join(f'{k} {v}' for k, v in rel.items())})")

    def forward():
        with torch.no_grad():
            x = x0
            for layer in enc.gat_layers:
                x = F.gelu(layer(x, ei))
            return x, enc.output_projection(x)

    h, y = forward()
    with torch.no_grad():
        red = enc.cross_attention_reducer(y.unsqueeze(0))[0]
    cached = torch.load(BR_CKPT / "go_embedding.pt", map_location="cpu", weights_only=True).float()
    red_rel = float((red - cached).norm() / cached.norm())
    red_cos = float(F.cosine_similarity(red, cached, dim=1).min())
    log(f"  reduced 200 x 2560 output vs cached go_embedding.pt: relative L2 error {red_rel:.2e}, "
        f"min row cosine {red_cos:.5f}")
    rows, cols, seen = [], [], set()
    for i, t in enumerate(enc.all_go_terms):
        c = onto.canon(t)
        if len(c) == 1 and c[0] not in seen:
            rows.append(i); cols.append(c[0]); seen.add(c[0])
    rows, cols = np.array(rows), np.array(cols)
    cpos = {c: i for i, c in enumerate(cols)}
    csel = np.array([cpos[c] for c in common if c in cpos])
    vocs = (("own", None, f"BioReason-Pro vocab ({len(cols)})"),
            ("common", csel, f"common ({len(common)})" if len(csel) == len(common)
             else f"common with BioReason-Pro ({len(csel)})"))
    vecs = {"BioReason-Pro GO encoder output per term (GAT x3 + projection, 2560-d)": y.numpy()[rows],
            "BioReason-Pro last GAT layer per term (512-d)": h.numpy()[rows],
            "BioReason-Pro precomputed GO term embedding (encoder input, frozen)": x0.numpy()[rows]}
    out = []
    for _, sel, vname in vocs:
        cc = cols if sel is None else cols[sel]
        for name, X in vecs.items():
            t = time.time()
            out.append({"vectors": name, "vocabulary": vname, "V": len(cc),
                        **retrieval_scores(X if sel is None else X[sel], cc, adj1, adj2)})
            log(f"    {name[:60]:60s} {vname:32s} direct_P@1 {out[-1]['direct_P@1']:.4f} ({time.time() - t:.0f}s)")
    base = {v: [] for _, _, v in vocs}
    for seed in range(seeds):
        torch.manual_seed(seed)
        for layer in enc.gat_layers:
            layer.reset_parameters()
        enc._initialize_weights()
        X = forward()[1].numpy()[rows]
        for _, sel, vname in vocs:
            base[vname].append(retrieval_scores(X if sel is None else X[sel], cols if sel is None else cols[sel],
                                                adj1, adj2))
        log(f"    untrained encoder seed {seed}: direct_P@1 own {base[vocs[0][2]][-1]['direct_P@1']:.4f}")
    for _, sel, vname in vocs:
        bdf = pd.DataFrame(base[vname])
        V = len(cols) if sel is None else len(sel)
        out.append({"vectors": f"untrained BioReason-Pro encoder, fresh init ({seeds} seeds, mean)", "vocabulary": vname,
                    "V": V, **bdf.mean().to_dict()})
        out.append({"vectors": f"untrained BioReason-Pro encoder ({seeds} seeds, sd)", "vocabulary": vname, "V": V,
                    **bdf.std(ddof=1).to_dict()})
    info = {"br_terms": len(enc.all_go_terms), "br_mapped": int(len(cols)), "br_edges": int(ei.shape[1]),
            "br_relations": {str(k): int(v) for k, v in rel.items()}, "br_reduced_rel_err": red_rel,
            "br_reduced_min_cos": red_cos, "br_common": int(len(csel)), "br_seconds": time.time() - t0}
    log(f"  BioReason-Pro retrieval done in {info['br_seconds']:.0f}s")
    return out, info


def run_retrieval(onto, child, parent, extended: bool = False) -> tuple[pd.DataFrame, dict]:
    nT = len(onto.terms)
    D = sp.csr_matrix((np.ones(len(child)), (child, parent)), shape=(nT, nT))
    adj1 = ((D + D.T) > 0).astype(np.float32).tocsr()
    adj2 = ((adj1 + adj1 @ adj1) > 0).astype(np.float32).tolil()
    adj2.setdiag(0)
    adj2 = adj2.tocsr()
    adj2.eliminate_zeros()
    adj1.setdiag(0)
    adj1.eliminate_zeros()
    E, q, hcols, Aop, err, how = higo_vectors(onto)
    log(f"  HiGO vectors via {how}; max |queries() - (e + A e)| = {err:.2e}")
    gvecs, gcols, gnote = gogpt_vectors(onto)
    log(f"  GO-GPT: {gnote}")
    common = np.intersect1d(hcols, gcols)
    hpos = {c: i for i, c in enumerate(hcols)}
    gpos = {c: i for i, c in enumerate(gcols)}
    hsel = np.array([hpos[c] for c in common])
    gsel = np.array([gpos[c] for c in common])
    rows = []

    def add(name, X, cols, vocab_name):
        r = retrieval_scores(X, cols, adj1, adj2)
        rows.append({"vectors": name, "vocabulary": vocab_name, "V": len(cols), **r})

    for vname, sel_h, sel_g in (("own", None, None), ("common", hsel, gsel)):
        hc = hcols if sel_h is None else hcols[sel_h]
        hv = f"HiGO vocab ({len(hcols)})" if sel_h is None else f"common ({len(common)})"
        add("HiGO effective query q = e + mean_anc(e)", q if sel_h is None else q[sel_h], hc, hv)
        add("HiGO raw term_emb e", E if sel_h is None else E[sel_h], hc, hv)
        base = []
        for seed in range(5):
            e0 = np.random.default_rng(seed).normal(0, 0.02, E.shape)
            q0 = e0 + Aop @ e0
            base.append(retrieval_scores(q0 if sel_h is None else q0[sel_h], hc, adj1, adj2))
        bdf = pd.DataFrame(base)
        rows.append({"vectors": "untrained N(0,0.02^2) e through the same ancestor-mean (5 seeds, mean)",
                     "vocabulary": hv, "V": len(hc), **bdf.mean().to_dict()})
        rows.append({"vectors": "untrained ancestor-mean (5 seeds, sd)", "vocabulary": hv, "V": len(hc),
                     **bdf.std(ddof=1).to_dict()})
        gc = gcols if sel_g is None else gcols[sel_g]
        gvn = f"GO-GPT vocab ({len(gcols)})" if sel_g is None else f"common ({len(common)})"
        for gname, X in gvecs.items():
            add(gname, X if sel_g is None else X[sel_g], gc, gvn)
    info = {"higo_how": how, "higo_query_check_maxabs": err, "gogpt": gnote, "common_vocab": int(len(common)),
            "direct_edges": int(len(child))}
    if extended:
        brow, binfo = bioreason_retrieval(onto, adj1, adj2, common)
        rows += brow
        info.update(binfo)
    return pd.DataFrame(rows), info


CP_SCORES = "vs runs/compare_paper/summary.tsv S3_all (knn_sharp.Scorer, 4 decimals)"


def crosscheck_compare_paper(point: dict) -> list[dict]:
    """Fmax_w, Smin_w, Fmax (and the Fmax_w threshold) of every row with a compare_paper key against
    runs/compare_paper/summary.tsv, S3_all; the `cafaeval` column holds the compare_paper value."""
    ref = pd.read_csv(COMPARE / "summary.tsv", sep="\t")
    ref = ref[ref.subset == "S3_all"].set_index("method")
    out = []
    for m in MODELS:
        key = CP_KEY.get(m)
        if key is None or key not in ref.index:
            continue
        r = ref.loc[key]
        for met, col in (("Fmax_w", "Fw"), ("Smin_w", "Smin"), ("Fmax", "F")):
            for asp in COLS:
                v, rv = float(point[m][asp][met][0]), float(r[f"{col}_{asp}"])
                row = {"model": m, "scores": CP_SCORES, "metric": met, "aspect": asp, "cafaeval": rv,
                       "eval_suite": v, "abs_diff": abs(v - rv)}
                if met == "Fmax_w" and asp != "mean":
                    row["tau_cafaeval"] = round(float(r[f"tau_w_{asp}"]), 2)
                    row["tau_eval_suite"] = round(float(point[m][asp]["tau_w"][0]), 2)
                out.append(row)
    return out


# ------------------------------------------------------------------------------------------------ main

def main():
    global OUT, MODELS, REFS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--perm-reps", type=int, default=P_REPS)
    ap.add_argument("--boot-reps", type=int, default=B_REPS)
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--extended", action="store_true",
                    help="add the runs/compare_paper methods as rows and comparisons; outputs in runs/eval_suite_v2")
    a = ap.parse_args()
    if a.extended:
        OUT = OUT_EXT
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"== eval_suite: descriptive metrics of frozen test predictions (look #3); threads {_THREADS}")
    if a.extended:
        log(f"  extended mode: compare_paper methods added as rows and comparisons; outputs in {OUT}")

    log("== stage 1: ontology, ground truth, prediction files")
    onto = ks.Ontology()
    sc = ks.Scorer(onto, ks.gt_frame("test"))
    n_a = {a_: int(sc.in_ns[k].sum()) for k, a_ in enumerate(ASPECTS)}
    log(f"  {sc.n} ground-truth proteins; per aspect {n_a}")
    models = load_models(onto, extended=a.extended)
    levels = {}
    if a.extended:
        MODELS = MODELS + tuple(m for m, _ in EXT_ROWS if m in models)
        REFS = REFS + tuple(m for m, _ in EXT_ROWS if m in models and m not in NO_TEST)
        for m in MODELS:
            ids, P, _ = models[m]
            nz = np.diff(P.indptr) > 0
            mx = np.maximum.reduceat(P.data, P.indptr[:-1][nz]) if P.nnz else np.zeros(0)
            mn = np.minimum.reduceat(P.data, P.indptr[:-1][nz]) if P.nnz else np.zeros(0)
            levels[m] = {"distinct": int(len(np.unique(P.data))), "one_score_share": float((mx == mn).mean())}
            if m in REFS and m not in FLAGS and levels[m]["distinct"] == 1:
                FLAGS[m] = f"{m} scores are binary: one operating point, not a ranking"
            log(f"  {m:40s} distinct raw scores {levels[m]['distinct']:7d}; proteins whose scores are all one value "
                f"{levels[m]['one_score_share']:.3f}" + ("  [AUPR flagged: binary]" if m in FLAGS else ""))
        log(f"  {len(MODELS)} rows, {len(REFS)} comparisons with {MAIN}: {', '.join(REFS)}")
    child, parent = direct_edges(onto)
    log(f"  {len(child)} direct is_a/part_of edges inside namespaces")

    log("== stage 2: ontology consistency on the raw files")
    crow = []
    for name, (ids, P, voc) in models.items():
        for r in consistency(P, child, parent, onto, voc):
            crow.append({"model": name, **r})
        r = crow[-1]
        log(f"  {name:40s} violation rate {r['rate']:.4f} of {r['pairs']} pairs; proteins with >=1: "
            f"{r['share_proteins']:.3f}; parent-in-vocab rate {r['rate_parent_in_vocab']:.4f}")
    cons = pd.DataFrame(crow)
    cons.to_csv(OUT / "consistency.tsv", sep="\t", index=False)

    log("== stage 3: per-protein statistics (submission cut + max-propagation)")
    stats = {}
    for name, (ids, P, _) in models.items():
        t = time.time()
        stats[name] = build_stats(sc, processed(sc, ids, P))
        log(f"  {name:40s} {time.time() - t:5.1f}s")

    log("== stage 4: point estimates and cafaeval cross-check")
    point = {m: eval_weights(stats[m], np.ones((1, sc.n))) for m in MODELS}
    for m in MODELS:
        p = point[m]
        log(f"  {m:40s} " + "  ".join(f"{x} {p['mean'][x][0]:.4f}" for x in ("Fmax", "Fmax_w", "Smin_w",
                                                                              "AUPR_micro", "AUPR_prot_w")))
    cx = []
    best = {k: pd.read_csv(CAFAEVAL_DIR / f"best_{k}.tsv", sep="\t") for k in ("f", "f_w", "s", "f_micro", "f_micro_w")}
    nsmap = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
    checks = (("f", "f", "Fmax", "tau"), ("f_w", "f_w", "Fmax_w", "tau_w"), ("f_micro", "f_micro", "Fmicro", None),
              ("f_micro_w", "f_micro_w", "Fmicro_w", None), ("s", "s", "Smin", "tau_s"))
    ids, P, _ = models[MAIN]
    replica = build_stats(sc, processed(sc, ids, P, replica=True))
    check_rows = [(m, fname, stats[m], point[m], "submission format (3 decimals)") for m, fname in CAFAEVAL_NAME.items()]
    check_rows.insert(1, (MAIN, CAFAEVAL_NAME[MAIN], replica, eval_weights(replica, np.ones((1, sc.n))),
                          "cafaeval input file replica (4 decimals)"))
    for m, fname, st_m, pt_m, proto in check_rows:
        for bk, col, ours, tcol in checks:
            b = best[bk][best[bk].filename == fname]
            for _, r in b.iterrows():
                asp = nsmap[r["ns"]]
                v = float(pt_m[asp][ours][0])
                row = {"model": m, "scores": proto, "metric": ours, "aspect": asp, "cafaeval": float(r[col]),
                       "eval_suite": v, "abs_diff": abs(v - float(r[col]))}
                if tcol:
                    row["tau_cafaeval"] = round(float(r["tau"]), 2)
                    row["tau_eval_suite"] = round(float(pt_m[asp][tcol][0]), 2)
                cx.append(row)
        b = best["s"][best["s"].filename == fname]
        for _, r in b.iterrows():
            asp = nsmap[r["ns"]]
            st = st_m[asp]["w"]
            j = int(round(float(r["tau"]) * 100)) - 1
            ne = len(st_m[asp]["idx"])
            sw = float(np.sqrt((st["fn"][:, j].sum() / ne) ** 2 + (st["fp"][:, j].sum() / ne) ** 2))
            cx.append({"model": m, "scores": proto, "metric": "s_w at best_s tau", "aspect": asp,
                       "cafaeval": float(r["s_w"]), "eval_suite": sw, "abs_diff": abs(sw - float(r["s_w"])),
                       "tau_cafaeval": round(float(r["tau"]), 2), "tau_eval_suite": round(float(r["tau"]), 2)})
    if a.extended:
        cx += crosscheck_compare_paper(point)
    cx = pd.DataFrame(cx)
    cx.to_csv(OUT / "crosscheck.tsv", sep="\t", index=False)
    for proto, g in cx.groupby("scores", sort=False):
        fm = g[g.metric.isin(["Fmax", "Fmax_w"])]
        t = g.dropna(subset=["tau_cafaeval"])
        log(f"  cross-check [{proto}]: max |diff| Fmax/Fmax_w {fm.abs_diff.max():.2e}, all checked metrics "
            f"{g.abs_diff.max():.2e}; thresholds equal {int((t.tau_cafaeval == t.tau_eval_suite).sum())}/{len(t)}")
    fm = cx[cx.metric.isin(["Fmax", "Fmax_w"])]
    assert fm.abs_diff.max() < 1e-3, "Fmax cross-check against cafaeval failed"
    if a.extended:
        g = cx[(cx.scores == CP_SCORES) & cx.metric.isin(["Fmax_w", "Smin_w"])]
        for m, gm in g.groupby("model", sort=False):
            log(f"  compare_paper check {m:32s} max |diff| Fmax_w {gm[gm.metric == 'Fmax_w'].abs_diff.max():.1e}, "
                f"Smin_w {gm[gm.metric == 'Smin_w'].abs_diff.max():.1e}")
        assert g.abs_diff.max() < 1e-3, "Fmax_w / Smin_w cross-check against runs/compare_paper/summary.tsv failed"

    log(f"== stage 5: paired protein bootstrap ({a.boot_reps} reps, seed {B_SEED})")
    rng = np.random.default_rng(B_SEED)
    bidx = rng.integers(0, sc.n, size=(a.boot_reps, sc.n))
    BW = np.stack([np.bincount(r, minlength=sc.n) for r in bidx]).astype(np.float64)
    boot = {}
    for m in MODELS:
        t = time.time()
        boot[m] = eval_weights(stats[m], BW)
        log(f"  {m:40s} {time.time() - t:5.1f}s")
    srow, drow = [], []
    for m in MODELS:
        for asp in COLS:
            for met in REPORTED:
                lo, hi = ci(boot[m][asp][met])
                srow.append({"model": m, "metric": met, "aspect": asp, "value": float(point[m][asp][met][0]),
                             "lo": lo, "hi": hi})
    for ref in REFS:
        for asp in COLS:
            for met in REPORTED:
                d = boot[MAIN][asp][met] - boot[ref][asp][met]
                lo, hi = ci(d)
                drow.append({"comparison": f"{MAIN} - {ref}", "metric": met, "aspect": asp,
                             "diff": float(point[MAIN][asp][met][0] - point[ref][asp][met][0]), "lo": lo, "hi": hi})
    summ = pd.DataFrame(srow)
    diffs = pd.DataFrame(drow)

    log("== stage 6: identity bins and Spearman (per-protein F1 at each model's own Fmax threshold)")
    idents = {}
    for view, d in (("train", TV2 / "knn"), ("train+val", TV2 / "knn_trainval")):
        z = np.load(d / "test_knn.npz", allow_pickle=False)
        s = pd.Series(z["max_pident"].astype(float), index=z["ids"].astype(str))
        idents[view] = s.reindex(sc.ids).fillna(0.0).values
    bins = {v: ks.ident_bins(x).values for v, x in idents.items()}
    f1 = {m: {asp: {wn: per_protein_f1(stats[m][asp], int(round(point[m][asp]["tau" + suf][0] * 100)) - 1, wn)
                    for wn, suf in (("u", ""), ("w", "_w"))} for asp in ASPECTS} for m in MODELS}
    brow = []
    bin_boot = {}
    for view in idents:
        for m in MODELS:
            for wn in ("u", "w"):
                per_asp_pt, per_asp_bs, per_asp_cov = {}, {}, {}
                for asp in ASPECTS:
                    idx = stats[m][asp]["idx"]
                    bb = bins[view][idx]
                    Ind = np.stack([(bb == b).astype(float) for b in BINS], 1)
                    x = f1[m][asp][wn]
                    jt = int(round(point[m][asp]["tau" if wn == "u" else "tau_w"][0] * 100)) - 1
                    hasj = stats[m][asp][wn]["has"][:, jt]
                    num = BW[:, idx] @ (Ind * x[:, None])
                    den = BW[:, idx] @ Ind
                    with np.errstate(invalid="ignore", divide="ignore"):
                        per_asp_bs[asp] = num / den
                        per_asp_pt[asp] = (Ind * x[:, None]).sum(0) / Ind.sum(0)
                        per_asp_cov[asp] = (Ind * hasj[:, None]).sum(0) / Ind.sum(0)
                    for j, b in enumerate(BINS):
                        lo, hi = ci(per_asp_bs[asp][:, j][np.isfinite(per_asp_bs[asp][:, j])]) if Ind[:, j].sum() else (np.nan, np.nan)
                        brow.append({"identity": view, "model": m, "f1": "F1" if wn == "u" else "F1_w", "aspect": asp,
                                     "bin": b, "n": int(Ind[:, j].sum()), "mean_f1": float(per_asp_pt[asp][j]),
                                     "lo": lo, "hi": hi, "covered": float(per_asp_cov[asp][j])})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mpt = np.nanmean([per_asp_pt[x] for x in ASPECTS], 0)
                    mbs = np.nanmean([per_asp_bs[x] for x in ASPECTS], 0)
                    mcv = np.nanmean([per_asp_cov[x] for x in ASPECTS], 0)
                bin_boot[(view, m, wn)] = mbs
                nprot = [(bins[view] == b).sum() for b in BINS]
                for j, b in enumerate(BINS):
                    lo, hi = ci(mbs[:, j][np.isfinite(mbs[:, j])])
                    brow.append({"identity": view, "model": m, "f1": "F1" if wn == "u" else "F1_w", "aspect": "mean",
                                 "bin": b, "n": int(nprot[j]), "mean_f1": float(mpt[j]), "lo": lo, "hi": hi,
                                 "covered": float(mcv[j])})
    binsdf = pd.DataFrame(brow)
    for ref in REFS:
        for wn in ("u", "w"):
            d = bin_boot[("train", MAIN, wn)] - bin_boot[("train", ref, wn)]
            bt = binsdf[(binsdf.identity == "train") & (binsdf.aspect == "mean") & (binsdf.f1 == ("F1" if wn == "u" else "F1_w"))]
            for j, b in enumerate(BINS):
                v = float(bt[(bt.model == MAIN) & (bt.bin == b)].mean_f1.iloc[0] - bt[(bt.model == ref) & (bt.bin == b)].mean_f1.iloc[0])
                lo, hi = ci(d[:, j][np.isfinite(d[:, j])])
                drow.append({"comparison": f"{MAIN} - {ref}", "metric": f"bin {b} mean-over-aspects "
                             + ("F1" if wn == "u" else "F1_w") + " (train identity)", "aspect": "mean",
                             "diff": v, "lo": lo, "hi": hi})
    binsdf.to_csv(OUT / "identity_bins.tsv", sep="\t", index=False)
    log("  bins written; mean-over-aspects F1 (train identity):\n" + binsdf[(binsdf.identity == "train") & (binsdf.aspect == "mean") & (binsdf.f1 == "F1")]
        .pivot(index="model", columns="bin", values="mean_f1")[BINS].round(4).to_string())

    sprow, sp_boot = [], {}
    for view in idents:
        for m in MODELS:
            for wn in ("u", "w"):
                pt, bs = {}, {}
                for asp in ASPECTS:
                    idx = stats[m][asp]["idx"]
                    x, y = f1[m][asp][wn], idents[view][idx]
                    pt[asp] = spearman_rows(x, y)
                    loc = np.full(sc.n, -1)
                    loc[idx] = np.arange(len(idx))
                    vals = []
                    for r in bidx:
                        li = loc[r]
                        li = li[li >= 0]
                        vals.append(spearman_rows(x[li], y[li]))
                    bs[asp] = np.array(vals)
                pt["mean"] = float(np.mean([pt[x] for x in ASPECTS]))
                bs["mean"] = np.mean([bs[x] for x in ASPECTS], 0)
                sp_boot[(view, m, wn)] = bs
                for asp in COLS:
                    lo, hi = ci(bs[asp])
                    sprow.append({"identity": view, "model": m, "f1": "F1" if wn == "u" else "F1_w", "aspect": asp,
                                  "spearman": pt[asp], "lo": lo, "hi": hi})
        log(f"  Spearman bootstrap done for {view} identity")
    spdf = pd.DataFrame(sprow)
    for ref in REFS:
        for wn in ("u", "w"):
            for asp in COLS:
                d = sp_boot[("train", MAIN, wn)][asp] - sp_boot[("train", ref, wn)][asp]
                q = spdf[(spdf.identity == "train") & (spdf.f1 == ("F1" if wn == "u" else "F1_w")) & (spdf.aspect == asp)]
                v = float(q[q.model == MAIN].spearman.iloc[0] - q[q.model == ref].spearman.iloc[0])
                lo, hi = ci(d)
                drow.append({"comparison": f"{MAIN} - {ref}", "metric": "Spearman(" + ("F1" if wn == "u" else "F1_w")
                             + ", train identity)", "aspect": asp, "diff": v, "lo": lo, "hi": hi})
    spdf.to_csv(OUT / "spearman.tsv", sep="\t", index=False)
    diffs = pd.DataFrame(drow)

    log(f"== stage 7: paired permutation tests ({a.perm_reps} reps, seed {P_SEED}), Holm and BH")
    prng = np.random.default_rng(P_SEED)
    chunk = 1000
    swaps = [prng.integers(0, 2, size=(min(chunk, a.perm_reps - s), sc.n)).astype(np.float64)
             for s in range(0, a.perm_reps, chunk)]
    prow = []
    for ref in REFS:
        t = time.time()
        null = {(asp, met): [] for asp in COLS for met in PRIMARY}
        totA = {asp: summed(stats[MAIN][asp], np.ones((1, len(stats[MAIN][asp]["idx"])))) for asp in ASPECTS}
        totB = {asp: summed(stats[ref][asp], np.ones((1, len(stats[ref][asp]["idx"])))) for asp in ASPECTS}
        for Sw in swaps:
            res = {}
            for asp in ASPECTS:
                idx = stats[MAIN][asp]["idx"]
                Swk = Sw[:, idx]
                sa = {"ne": np.full(len(Sw), float(len(idx)))}
                sb = {"ne": sa["ne"]}
                for wn in ("u", "w"):
                    sa[wn], sb[wn] = {}, {}
                    for key in KEYS:
                        dlt = Swk @ (stats[ref][asp][wn][key] - stats[MAIN][asp][wn][key])
                        sa[wn][key] = totA[asp][wn][key] + dlt
                        sb[wn][key] = totB[asp][wn][key] - dlt
                ma, mb = metrics_from(sa), metrics_from(sb)
                res[asp] = {met: ma[met] - mb[met] for met in PRIMARY}
            for met in PRIMARY:
                for asp in ASPECTS:
                    null[(asp, met)].append(res[asp][met])
                null[("mean", met)].append(np.mean([res[x][met] for x in ASPECTS], 0))
        for asp in COLS:
            for met in PRIMARY:
                obs = float(point[MAIN][asp][met][0] - point[ref][asp][met][0])
                prow.append({"family": "primary", "comparison": f"{MAIN} - {ref}", "metric": met, "aspect": asp,
                             "diff": obs, "p_raw": perm_p(obs, np.concatenate(null[(asp, met)])),
                             "flag": FLAGS.get(ref, "") if met.startswith("AUPR") else ""})
        log(f"  primary vs {ref}: {time.time() - t:.0f}s")
        t = time.time()
        for wn in ("u", "w"):
            lab = "F1" if wn == "u" else "F1_w"
            num_null = []
            for Sw in swaps:
                per = []
                for asp in ASPECTS:
                    idx = stats[MAIN][asp]["idx"]
                    bb = bins["train"][idx]
                    Ind = np.stack([(bb == b).astype(float) for b in BINS], 1)
                    xa, xb = f1[MAIN][asp][wn], f1[ref][asp][wn]
                    Swk = Sw[:, idx]
                    dA = (Ind * xa[:, None]).sum(0) + Swk @ (Ind * (xb - xa)[:, None])
                    dB = (Ind * xb[:, None]).sum(0) - Swk @ (Ind * (xb - xa)[:, None])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        per.append((dA - dB) / Ind.sum(0))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    num_null.append(np.nanmean(per, 0))
            num_null = np.concatenate(num_null)
            bt = binsdf[(binsdf.identity == "train") & (binsdf.aspect == "mean") & (binsdf.f1 == lab)]
            for j, b in enumerate(BINS):
                obs = float(bt[(bt.model == MAIN) & (bt.bin == b)].mean_f1.iloc[0] - bt[(bt.model == ref) & (bt.bin == b)].mean_f1.iloc[0])
                prow.append({"family": "secondary", "comparison": f"{MAIN} - {ref}",
                             "metric": f"bin {b} mean-over-aspects {lab} (train identity)", "aspect": "mean",
                             "diff": obs, "p_raw": perm_p(obs, num_null[:, j]), "flag": ""})
            sp_null = {}
            for asp in ASPECTS:
                idx = stats[MAIN][asp]["idx"]
                rid = rankdata(idents["train"][idx])
                xa, xb = f1[MAIN][asp][wn], f1[ref][asp][wn]
                vals = []
                for Sw in swaps:
                    Swk = Sw[:, idx]
                    A_ = xa + Swk * (xb - xa)
                    B_ = xb - Swk * (xb - xa)
                    vals.append(spearman_batch(A_, rid) - spearman_batch(B_, rid))
                sp_null[asp] = np.concatenate(vals)
            sp_null["mean"] = np.mean([sp_null[x] for x in ASPECTS], 0)
            q = spdf[(spdf.identity == "train") & (spdf.f1 == lab)]
            for asp in COLS:
                qa = q[q.aspect == asp]
                obs = float(qa[qa.model == MAIN].spearman.iloc[0] - qa[qa.model == ref].spearman.iloc[0])
                prow.append({"family": "secondary", "comparison": f"{MAIN} - {ref}",
                             "metric": f"Spearman({lab}, train identity)", "aspect": asp, "diff": obs,
                             "p_raw": perm_p(obs, sp_null[asp]), "flag": ""})
        log(f"  secondary vs {ref}: {time.time() - t:.0f}s")
    pv = pd.DataFrame(prow)
    for fam in ("primary", "secondary"):
        m_ = pv.family == fam
        pv.loc[m_, "p_holm"] = holm(pv.loc[m_, "p_raw"].values)
        pv.loc[m_, "q_bh"] = bh(pv.loc[m_, "p_raw"].values)
    pv["p_holm_all"] = holm(pv.p_raw.values)
    pv["q_bh_all"] = bh(pv.p_raw.values)
    try:
        from statsmodels.stats.multitest import multipletests
        m_ = pv.family == "primary"
        assert np.allclose(multipletests(pv.loc[m_, "p_raw"], method="holm")[1], pv.loc[m_, "p_holm"])
        assert np.allclose(multipletests(pv.loc[m_, "p_raw"], method="fdr_bh")[1], pv.loc[m_, "q_bh"])
        log("  Holm and BH match statsmodels.multipletests")
    except ImportError:
        pass
    pv = pv.merge(diffs[["comparison", "metric", "aspect", "lo", "hi"]], on=["comparison", "metric", "aspect"], how="left")
    pv = pv[["family", "comparison", "metric", "aspect", "diff", "lo", "hi", "p_raw", "p_holm", "q_bh", "p_holm_all",
             "q_bh_all", "flag"]]
    pv.to_csv(OUT / "pvalues.tsv", sep="\t", index=False)
    diffs.to_csv(OUT / "differences.tsv", sep="\t", index=False)
    log(f"  {len(pv)} tests ({(pv.family == 'primary').sum()} primary, {(pv.family == 'secondary').sum()} secondary); "
        f"primary Holm < 0.05: {int(((pv.family == 'primary') & (pv.p_holm < 0.05)).sum())}")

    if a.skip_retrieval:
        retr, rinfo = pd.DataFrame(), {}
    else:
        log("== stage 8: GO-relation retrieval from learned term vectors")
        retr, rinfo = run_retrieval(onto, child, parent, extended=a.extended)
        retr.to_csv(OUT / "retrieval.tsv", sep="\t", index=False)
        log("  " + retr[["vectors", "vocabulary", "direct_P@1", "direct_R@10", "2hop_P@10"]].round(4).to_string()
            .replace("\n", "\n  "))

    log("== stage 9: tables and report")
    head = summ[summ.metric.isin(["Fmax", "Fmax_w", "Fmicro", "Fmicro_w", "cov", "cov_w"])]
    head.to_csv(OUT / "headline.tsv", sep="\t", index=False)
    summ[summ.metric.str.startswith(("AUPR", "Rmax"))].to_csv(OUT / "auprc.tsv", sep="\t", index=False)
    summ[summ.metric.str.startswith("Smin")].to_csv(OUT / "smin.tsv", sep="\t", index=False)
    summ_all = pd.concat([summ.assign(kind="model"),
                          diffs.rename(columns={"comparison": "model", "diff": "value"}).assign(kind="difference")])
    summ_all.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    if a.extended:
        write_report_extended(summ, diffs, pv, binsdf, spdf, cons, cx, retr, rinfo, n_a, a, levels)
        log_extended_summary(summ, pv, cons, cx)
    else:
        write_report(summ, diffs, pv, binsdf, spdf, cons, cx, retr, rinfo, n_a, a)
    log(f"done; outputs in {OUT}")


def write_report(summ, diffs, pv, binsdf, spdf, cons, cx, retr, rinfo, n_a, a):
    def wide(metric, models=MODELS):
        rows = []
        for m in models:
            q = summ[(summ.model == m) & (summ.metric == metric)].set_index("aspect")
            rows.append({"row": m, **{c: f"{q.loc[c, 'value']:.4f} [{q.loc[c, 'lo']:.4f}, {q.loc[c, 'hi']:.4f}]"
                                      for c in COLS}})
        return pd.DataFrame(rows)

    def dwide(metric):
        rows = []
        for ref in REFS:
            q = pv[(pv.comparison == f"{MAIN} - {ref}") & (pv.metric == metric)].set_index("aspect")
            if len(q) == 0:
                q = diffs[(diffs.comparison == f"{MAIN} - {ref}") & (diffs.metric == metric)].set_index("aspect")
                rows.append({"difference": f"{MAIN} - {ref}",
                             **{c: f"{q.loc[c, 'diff']:+.4f} [{q.loc[c, 'lo']:+.4f}, {q.loc[c, 'hi']:+.4f}]" for c in COLS}})
            else:
                rows.append({"difference": f"{MAIN} - {ref}",
                             **{c: f"{q.loc[c, 'diff']:+.4f} [{q.loc[c, 'lo']:+.4f}, {q.loc[c, 'hi']:+.4f}] "
                                   f"(Holm p {q.loc[c, 'p_holm']:.4f})" for c in COLS}})
        return pd.DataFrame(rows)

    def val(m, metric, asp="mean"):
        return float(summ[(summ.model == m) & (summ.metric == metric) & (summ.aspect == asp)].value.iloc[0])

    def dv(ref, metric, asp="mean"):
        q = pv[(pv.comparison == f"{MAIN} - {ref}") & (pv.metric == metric) & (pv.aspect == asp)]
        if len(q) == 0:
            q = diffs[(diffs.comparison == f"{MAIN} - {ref}") & (diffs.metric == metric) & (diffs.aspect == asp)]
        return q.iloc[0]

    def dtxt(ref, metric, asp="mean", f=".4f"):
        r = dv(ref, metric, asp)
        return f"{r['diff']:+{f}} [{r['lo']:+{f}}, {r['hi']:+{f}}]"

    def sides(ref, metric, keys=ASPECTS):
        """Aspects whose bootstrap CI of HiGO-R2 minus ref lies above 0 / below 0 / includes 0."""
        up, down, mid = [], [], []
        for k in keys:
            r = dv(ref, metric, k)
            (up if r["lo"] > 0 else down if r["hi"] < 0 else mid).append(k)
        return up, down, mid

    def bin_sides(ref, lab, view="train"):
        up, down, mid = [], [], []
        for b in BINS:
            r = dv(ref, f"bin {b} mean-over-aspects {lab} (train identity)", "mean")
            (up if r["lo"] > 0 else down if r["hi"] < 0 else mid).append(f"{b} ({r['diff']:+.3f})")
        return up, down, mid

    def lst(x):
        return ", ".join(x) if x else "none"

    L = []
    w = L.append
    w("# HiGO-R2 vs GO-GPT: evaluation suite on GO-GPT's temporal test\n")
    w("Descriptive metrics of prediction files that were frozen before this suite existed (test look #3 in "
      "`runs/test_looks.md`). Nothing was tuned, selected or changed based on these numbers, and the suite records "
      "no new test look. Generated by `src/cafa6/eval_suite.py` (definitions in its module docstring); all numbers "
      "below come from the tsv files next to this report.\n")
    w(f"Ground truth: `data/proc_gogpt/gt_test.tsv`, 8,630 proteins (MF {n_a['MF']}, BP {n_a['BP']}, CC {n_a['CC']} "
      "with annotations). Ontology `runs/gogpt_cafa5/ref/go-basic_gogpt.obo`, IA `runs/gogpt_cafa5/ref/IA.txt`, "
      "cafaeval protocol (prop=max, norm=cafa, tau 0.01..0.99). Brackets are paired protein bootstrap 95% CIs "
      f"({a.boot_reps} reps, seed 0). \"mean\" is the mean of the three aspects.\n")
    w("Rows: **HiGO-R2** (`runs/stacker_tv/l3_c0/pred_test_twin.npz`), **GO-GPT ten-sample** "
      "(`runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz`), **GO-GPT beam** (`runs/gogpt_cafa5/preds_shard*.jsonl`). "
      "The two `[component]` rows are inputs of HiGO-R2 that exist as frozen test files (the train+val kNN prior and "
      "the HiGO head alone); they are not separate models and are not part of any test family.\n")

    w("## 0. Cross-check against the official cafaeval output\n")
    sub = cx[cx.scores == "submission format (3 decimals)"]
    rep = cx[cx.scores != "submission format (3 decimals)"]
    fm, fr = sub[sub.metric.isin(["Fmax", "Fmax_w"])], rep[rep.metric.isin(["Fmax", "Fmax_w"])]
    t = sub.dropna(subset=["tau_cafaeval"])
    gsub = sub[sub.model != MAIN]
    w(f"Fmax and IA-weighted Fmax of the three main rows were recomputed in memory and compared with "
      f"`runs/test_look/look3_l3c0_twin/cafaeval/best_f.tsv` and `best_f_w.tsv`. **Check passed: max absolute "
      f"difference {fm.abs_diff.max():.1e} over {len(fm)} values (required < 1e-3).** The GO-GPT rows match to "
      f"{gsub.abs_diff.max():.0e} on every checked value. The HiGO-R2 differences (up to {fm.abs_diff.max():.1e}) "
      "come from the input file, not the scorer: the TSV given to cafaeval for HiGO-R2 had its scores rounded to 4 "
      "decimals, while this suite (like `cafa6.test_look`, which reported look #3) uses the submission format of "
      "`cafa6.evaluate.dense_to_long` (3 decimals, top 1,500 per protein; no protein has more than 1,448 scores >= "
      "0.01, so only the rounding differs). Scoring a replica of that 4-decimal file reproduces cafaeval to "
      f"{fr.abs_diff.max():.0e} for Fmax/Fmax_w and {rep.abs_diff.max():.0e} over all {len(rep)} checked HiGO-R2 "
      "values, including micro-F (`best_f_micro*.tsv`), the count-based S of `best_s.tsv` and s_w at its "
      "threshold. All tables below use the 3-decimal submission format for every row. Under that format the "
      f"selected thresholds agree with cafaeval in {int((t.tau_cafaeval == t.tau_eval_suite).sum())} of {len(t)} "
      "cases; the others are HiGO-R2 aspects where the 3- vs 4-decimal rounding moves the best threshold by one "
      "or a few steps. Full table: `crosscheck.tsv`.\n")
    w(md(cx[cx.metric.isin(["Fmax", "Fmax_w"])][["model", "scores", "metric", "aspect", "cafaeval", "eval_suite",
                                                   "abs_diff", "tau_cafaeval", "tau_eval_suite"]], ".6g"))
    w("")

    w("## 1. Fmax and IA-weighted Fmax\n")
    w("Unweighted Fmax (every namespace term weighs 1):\n")
    w(md(wide("Fmax")))
    w("\nIA-weighted Fmax (the CAFA/Kaggle metric):\n")
    w(md(wide("Fmax_w")))
    w("\nDifferences (HiGO-R2 minus the GO-GPT row, bootstrap CI, Holm-adjusted permutation p over the primary "
      "family):\n")
    w(md(dwide("Fmax")))
    w("")
    w(md(dwide("Fmax_w")))
    w("\nCoverage at the Fmax_w threshold (share of annotated proteins with at least one positive-IA prediction "
      "there; `cov` at the unweighted threshold is in `headline.tsv`):\n")
    w(md(wide("cov_w")))
    w("")
    parts = []
    for met, lab in (("Fmax_w", "IA-weighted Fmax"), ("Fmax", "unweighted Fmax")):
        for ref in REFS:
            up, down, mid = sides(ref, met)
            parts.append(f"{lab} vs {ref}: mean {dtxt(ref, met)}; CI above 0 in {lst(up)}, below 0 in {lst(down)}, "
                         f"including 0 in {lst(mid)}")
    w(f"What it shows: HiGO-R2 has a mean IA-weighted Fmax of {val(MAIN, 'Fmax_w'):.4f} against "
      f"{val(REFS[0], 'Fmax_w'):.4f} (ten-sample) and {val(REFS[1], 'Fmax_w'):.4f} (beam). Per comparison: "
      + "; ".join(parts) + f". The IA-weighted margin ({dv(REFS[0], 'Fmax_w')['diff']:+.4f} vs ten-sample) is larger than the "
      f"unweighted one ({dv(REFS[0], 'Fmax')['diff']:+.4f}), i.e. HiGO-R2's advantage is mostly on specific "
      f"(high-IA) terms; on unweighted Fmax the CI against ten-sample excludes 0 only in "
      f"{lst(sides(REFS[0], 'Fmax')[0] + sides(REFS[0], 'Fmax')[1])}. "
      + (f"The train+val kNN prior alone has a mean Fmax_w of {val(MODELS[3], 'Fmax_w'):.4f}, "
         f"{val(MODELS[3], 'Fmax_w') - val(MAIN, 'Fmax_w'):+.4f} relative to HiGO-R2, so the stacker's gain over its "
         "own prior that was measured on TV v2 does not appear on this test (descriptive only; no decision depends "
         "on it; the component rows have no paired test here). "
         if val(MODELS[3], 'Fmax_w') >= val(MAIN, 'Fmax_w') else
         f"The train+val kNN prior alone has a mean Fmax_w of {val(MODELS[3], 'Fmax_w'):.4f}. ")
      + "What it does "
      "not show: Fmax is one operating point per aspect, chosen on the test set itself for every row (the CAFA "
      "protocol, also used by the GO-GPT paper), and precision is averaged only over covered proteins. HiGO-R2 "
      f"abstains on more proteins (coverage at its Fmax_w threshold: MF {val(MAIN, 'cov_w', 'MF'):.1%}, BP "
      f"{val(MAIN, 'cov_w', 'BP'):.1%}, CC {val(MAIN, 'cov_w', 'CC'):.1%}, against "
      f"{val(REFS[0], 'cov_w', 'MF'):.1%} / {val(REFS[0], 'cov_w', 'BP'):.1%} / {val(REFS[0], 'cov_w', 'CC'):.1%} "
      "for ten-sample), and in precision that costs nothing; it costs only through recall.\n")

    w("## 2. AUPRC\n")
    w("Term-centric micro AUPR (all (protein, term) pairs of the aspect, unpredicted pairs scored 0, ties grouped by "
      "distinct score; unweighted, roots included):\n")
    w(md(wide("AUPR_micro")))
    w("\nIA-weighted term-centric micro AUPR (pairs weighted by the term's IA):\n")
    w(md(wide("AUPR_micro_w")))
    w("\nProtein-centric PR area (step area under the CAFA averaged curve over tau = 0.01..0.99), unweighted and "
      "IA-weighted:\n")
    w(md(wide("AUPR_prot")))
    w("")
    w(md(wide("AUPR_prot_w")))
    w("\nDifferences:\n")
    for met in ("AUPR_micro", "AUPR_micro_w", "AUPR_prot", "AUPR_prot_w"):
        w(f"{met}:\n")
        w(md(dwide(met)))
        w("")
    w("Handling of ties and zeros: scores are first rounded to 3 decimals and cut at 0.01 (the submission format "
      "cafaeval reads), so all scores lie on a 0.001 grid. In the micro AUPR every distinct score value is one "
      "operating point and all pairs tied at that value enter together; the last point (score 0) predicts every "
      "remaining pair of the aspect, so ground-truth pairs a model never scores still count against its recall. "
      "The protein-centric area uses the 99 thresholds of cafaeval and stops at the recall reached at tau = 0.01 "
      "(no extrapolation to recall 1).\n")
    w("**Flag: GO-GPT beam.** Beam predictions are sets with score 1, so its micro curve has two operating points "
      "(score 1 and score 0) and its protein curve one point; its AUPR values are close to precision x recall at "
      "that point. They are listed for completeness, but they measure one operating point rather than a ranking, "
      "and the beam rows of the AUPR tests in the p-value table carry the same flag. The ten-sample scores take "
      "only 11 values (0, 0.1, ..., 1), so its curves have at most 11 points.\n")
    w("Recall ceiling R(0.01) of the protein-centric curve (unweighted, then IA-weighted): the area of the "
      "protein-centric AUPR can only accumulate up to this recall.\n")
    w(md(wide("Rmax")))
    w("")
    w(md(wide("Rmax_w")))
    w(f"\nWhat it shows: how well each row ranks pairs (micro) or how its averaged precision-recall trade-off looks "
      f"over all thresholds (protein-centric), not only at the Fmax point. HiGO-R2 is ahead on every AUPR variant "
      f"and aspect, by {dv(REFS[0], 'AUPR_micro_w')['diff']:+.3f} (micro, IA-weighted, mean) over ten-sample. These "
      "gaps are much larger than the Fmax gaps because the GO-GPT rows have coarse scores (11 levels for "
      "ten-sample, 1 for beam) and a lower recall ceiling (see the R(0.01) table: GO-GPT scores only the terms "
      "that appear in its generated sets, so every other pair stays at score 0, while HiGO-R2 and the kNN prior "
      "give low scores to many more terms). AUPR therefore mixes ranking quality with score resolution and with how many pairs a row scores at "
      "all; it does not say that GO-GPT's top predictions are worse. What it does not show: the unweighted micro "
      "AUPR is dominated by frequent shallow terms (it includes the namespace roots, which every annotated protein "
      "has), which is why the IA-weighted version is also given; none of these curves is calibrated across "
      "proteins.\n")

    w("## 3. S-min\n")
    w("IA-weighted S-min (sqrt(ru^2 + mi^2) in information content, minimised over tau; lower is better):\n")
    w(md(wide("Smin_w")))
    w("\nDifferences (negative = HiGO-R2 lower, i.e. better):\n")
    w(md(dwide("Smin_w")))
    w("\nCount-based S-min (unit term weights; the quantity cafaeval's `best_s.tsv` minimises in column `s`, and the "
      "`Smin` column of `cafa6.evaluate.evaluate`):\n")
    w(md(wide("Smin")))
    lowest = [asp for asp in COLS if min(MODELS, key=lambda m: val(m, "Smin_w", asp)) == MAIN]
    w("\nWhat it shows: the remaining uncertainty and misinformation (in bits of IA per protein) at the best "
      "threshold; it rewards specific correct terms and penalises specific wrong ones more than Fmax does. "
      f"HiGO-R2 has the lowest S-min_w of all five rows in {lst(lowest)}; against ten-sample the mean difference "
      f"is {dtxt(REFS[0], 'Smin_w')} and against beam {dtxt(REFS[1], 'Smin_w')}, largest in BP "
      f"({dtxt(REFS[0], 'Smin_w', 'BP')} and {dtxt(REFS[1], 'Smin_w', 'BP')}). The count-based S-min orders HiGO-R2 "
      "and the two GO-GPT rows the same way in every aspect (in MF the kNN prior is slightly lower on it). What it "
      "does not show: S-min is not normalised by the amount of annotation, so aspects (and proteins) with many deep "
      "annotations dominate, and the mean over aspects mixes different scales.\n")

    w("## 4. Generalisation by sequence identity to the nearest training protein\n")
    w("Per-protein F1 at each row's own per-aspect Fmax threshold (F1 at the unweighted Fmax tau, F1_w at the "
      "IA-weighted Fmax tau; a protein with no prediction at that threshold has F1 = 0), averaged within identity "
      "bins and then over the three aspects. n = proteins in the bin (any aspect); per-aspect values and n are in "
      "`identity_bins.tsv`.\n")
    for view, txt in (("train", "Identity to GO-GPT training proteins only (`data/proc_gogpt_tv2/knn/test_knn.npz`; "
                                "the binning used for GO-GPT numbers so far):"),
                      ("train+val", "Identity to training + validation proteins (`knn_trainval/test_knn.npz`; "
                                    "HiGO-R2 uses val proteins as kNN references, so this is its effective nearest "
                                    "reference):")):
        w(txt + "\n")
        for lab in ("F1", "F1_w"):
            q = binsdf[(binsdf.identity == view) & (binsdf.aspect == "mean") & (binsdf.f1 == lab)]
            rows = []
            nrow = {"row": "n proteins", **{b: str(int(q[(q.model == MAIN) & (q.bin == b)].n.iloc[0])) for b in BINS}}
            for m in MODELS:
                qq = q[q.model == m].set_index("bin")
                rows.append({"row": m, **{b: f"{qq.loc[b, 'mean_f1']:.3f} [{qq.loc[b, 'lo']:.3f}, {qq.loc[b, 'hi']:.3f}]"
                                          for b in BINS}})
            w(f"{lab}:\n")
            w(md(pd.DataFrame([nrow] + rows)))
            w("")
    w("Per-aspect n per bin (train identity):\n")
    q = binsdf[(binsdf.identity == "train") & (binsdf.model == MAIN) & (binsdf.f1 == "F1")]
    w(md(q.pivot(index="aspect", columns="bin", values="n")[BINS].reset_index(), ".0f"))
    w("\nBin differences HiGO-R2 minus GO-GPT (mean-over-aspects, train identity, bootstrap CI, Holm p within the "
      "secondary family):\n")
    rows = []
    for ref in REFS:
        for lab in ("F1", "F1_w"):
            r = {"difference": f"{MAIN} - {ref}", "f1": lab}
            for b in BINS:
                qq = pv[(pv.comparison == f"{MAIN} - {ref}") & (pv.metric == f"bin {b} mean-over-aspects {lab} (train identity)")].iloc[0]
                r[b] = f"{qq['diff']:+.3f} [{qq['lo']:+.3f}, {qq['hi']:+.3f}] p={qq['p_holm']:.3f}"
            rows.append(r)
    w(md(pd.DataFrame(rows)))
    parts = []
    for ref in REFS:
        for lab in ("F1", "F1_w"):
            up, down, mid = bin_sides(ref, lab)
            parts.append(f"{lab} vs {ref}: HiGO-R2 ahead (CI above 0) in {lst(up)}; behind (CI below 0) in "
                         f"{lst(down)}; CI includes 0 in {lst(mid)}")
    hb = binsdf[(binsdf.identity == "train") & (binsdf.aspect == "mean") & (binsdf.f1 == "F1")]
    hw = binsdf[(binsdf.identity == "train") & (binsdf.aspect == "mean") & (binsdf.f1 == "F1_w")]
    hi_share = hb[(hb.model == MAIN) & hb.bin.isin(["95-99.9", "100"])].n.sum() / hb[hb.model == MAIN].n.sum()
    knn_nohit = hb[(hb.model == MODELS[3]) & (hb.bin == "no_hit")].mean_f1.iloc[0]

    def cov_bin(m, b, h=hw):
        return float(h[(h.model == m) & (h.bin == b)].covered.iloc[0])

    w("\nShare of proteins with at least one positive-IA prediction at the row's Fmax_w threshold, per bin "
      "(mean over aspects, train identity):\n")
    w(md(pd.DataFrame([{"row": m, **{b: f"{cov_bin(m, b):.3f}" for b in BINS}} for m in MODELS])))
    w("\nWhat it shows: where the margin over GO-GPT comes from (train-only identity, mean over aspects). "
      + "; ".join(parts) + f". HiGO-R2's lead is concentrated in the >= 95% bins, which hold {hi_share:.0%} of "
      "the test proteins; below 80% identity the point estimates favour GO-GPT, and the largest gap is on "
      "proteins without any DIAMOND hit, where HiGO-R2 relies on the HiGO head alone (the kNN prior abstains: its "
      f"no-hit F1 is {knn_nohit:.3f}) and at its global Fmax_w threshold predicts anything for only "
      f"{cov_bin(MAIN, 'no_hit'):.1%} of these proteins (GO-GPT ten-sample: {cov_bin(REFS[0], 'no_hit'):.1%}). "
      "With train+val "
      "identity the bins shift a little (val proteins become the nearest reference for some test proteins) but "
      "the pattern is the same. What it does not show: per-protein F1 at a global threshold is not Fmax; the "
      "low-identity bins are small (see n; < 30% has under 100 proteins), so their CIs are wide; and identity to "
      "the nearest training protein is only one notion of novelty (the test is temporal, so a protein at 100% "
      "identity can still gain new annotations).\n")

    w("## 5. Dependence on homology (Spearman correlation of per-protein F1 with nearest-training identity)\n")
    rows = []
    for view in ("train", "train+val"):
        for lab in ("F1", "F1_w"):
            for m in MODELS:
                qq = spdf[(spdf.identity == view) & (spdf.f1 == lab) & (spdf.model == m)].set_index("aspect")
                rows.append({"identity": view, "f1": lab, "row": m,
                             **{c: f"{qq.loc[c, 'spearman']:.3f} [{qq.loc[c, 'lo']:.3f}, {qq.loc[c, 'hi']:.3f}]" for c in COLS}})
    w(md(pd.DataFrame(rows)))
    w("\nDifferences in Spearman correlation (train identity, HiGO-R2 minus GO-GPT, Holm p within the secondary "
      "family):\n")
    rows = []
    for ref in REFS:
        for lab in ("F1", "F1_w"):
            q = pv[(pv.comparison == f"{MAIN} - {ref}") & (pv.metric == f"Spearman({lab}, train identity)")].set_index("aspect")
            rows.append({"difference": f"{MAIN} - {ref}", "f1": lab,
                         **{c: f"{q.loc[c, 'diff']:+.3f} [{q.loc[c, 'lo']:+.3f}, {q.loc[c, 'hi']:+.3f}] p={q.loc[c, 'p_holm']:.3f}" for c in COLS}})
    w(md(pd.DataFrame(rows)))
    def rho(m, lab="F1_w", view="train", asp="mean"):
        return float(spdf[(spdf.identity == view) & (spdf.f1 == lab) & (spdf.model == m) & (spdf.aspect == asp)].spearman.iloc[0])

    w(f"\nWhat it shows: a larger correlation means per-protein quality tracks homology more closely. With F1_w "
      f"and train-only identity the mean correlation is {rho(MAIN):.3f} for HiGO-R2, {rho(REFS[0]):.3f} for "
      f"ten-sample and {rho(REFS[1]):.3f} for beam (HiGO-R2 minus ten-sample "
      f"{dtxt(REFS[0], 'Spearman(F1_w, train identity)', f='.3f')}); the train+val kNN prior, a purely "
      f"homology-driven method, is at {rho(MODELS[3]):.3f} and the HiGO head alone at {rho(MODELS[4]):.3f}. So "
      "HiGO-R2 depends on homology about as much as its kNN prior and clearly more than GO-GPT, whose quality is "
      "flatter across identity (it is also trained on the proteins a sequence search would find, so it is not "
      "homology-free either). What it does not show: it is a rank correlation over all proteins, dominated by "
      "the large high-identity bins (most test proteins have a >= 80% hit), and a higher correlation together with "
      "a higher mean is not a weakness by itself; the bin table above is the more direct view of low-identity "
      "behaviour.\n")

    w("## 6. Ontology consistency of the raw predictions\n")
    w("Violation = score(child) > score(direct parent) + 1e-6 over is_a/part_of edges inside a namespace, counted "
      "over (protein, child, parent) triples with score(child) > 0, on the raw files (before the evaluator's cut, "
      "rounding and max-propagation; terms mapped to go-basic_gogpt.obo). `share_proteins` = proteins with >= 1 "
      "violation among proteins with any score; `parent_missing_share` = share of violations where the parent has no "
      "score at all; `rate_parent_in_vocab` restricts to parents in the row's own output vocabulary.\n")
    c = cons[cons.aspect == "all"][["model", "pairs", "violations", "rate", "share_proteins", "mean_gap",
                                     "parent_missing_share", "rate_parent_in_vocab"]]
    w(md(c))
    w("\nPer aspect:\n")
    w(md(cons.pivot(index="model", columns="aspect", values="rate")[list(ASPECTS)].reindex(list(MODELS)).reset_index()))
    ca = cons[cons.aspect == "all"].set_index("model")
    w(f"\nWhat it shows: whether a file can be used as is by a consumer that expects the true-path rule (a parent "
      f"scored at least as high as its child). HiGO-R2 has {int(ca.loc[MAIN, 'violations'])} violations in "
      f"{int(ca.loc[MAIN, 'pairs']):,} pairs (its stacker output is max-propagated before it is saved), as does the "
      f"kNN prior ({int(ca.loc[MODELS[3], 'violations'])}; labels are ancestor-closed). GO-GPT ten-sample violates "
      f"{ca.loc[REFS[0], 'rate']:.2%} of pairs, in {ca.loc[REFS[0], 'share_proteins']:.1%} of proteins, "
      f"{ca.loc[REFS[0], 'parent_missing_share']:.0%} of them with the parent never generated; beam violates "
      f"{ca.loc[REFS[1], 'rate']:.2%} of pairs in {ca.loc[REFS[1], 'share_proteins']:.1%} of proteins (a generated "
      f"term without its parent). The HiGO head alone shows {ca.loc[MODELS[4], 'rate']:.2%}, all of it from parents "
      f"outside its {7901}-term vocabulary (rate with in-vocabulary parents: "
      f"{ca.loc[MODELS[4], 'rate_parent_in_vocab']:.4f}); its MCM layer makes it consistent inside the vocabulary. "
      "What it does not show: the cafaeval scores above are unaffected, because the evaluator max-propagates every "
      "file before scoring; and a binary set that omits a parent (beam) and a score slightly above its parent's "
      "count the same here, so `mean_gap` is given as well.\n")

    w("## 7. GO-relation retrieval from learned term vectors\n")
    if len(retr):
        w(f"Vectors: HiGO term parameters from `runs/l3_c0_tv2/trainable_weights.pt` ({rinfo['higo_how']}; "
          f"max |queries() - (e + A e)| = {rinfo['higo_query_check_maxabs']:.1e}); GO-GPT from the released "
          f"checkpoint ({rinfo['gogpt']}). Targets come from {rinfo['direct_edges']} direct is_a/part_of edges of "
          f"go-basic_gogpt.obo; the common vocabulary has {rinfo['common_vocab']} terms. Cosine similarity, all "
          "other vocabulary terms (any aspect) ranked; mean over terms with >= 1 target in the vocabulary. The random "
          "row is the expected value of a uniformly random ranking. The 'untrained ... ancestor-mean' rows apply "
          "HiGO's query construction q = e + mean_anc(e) to untrained N(0, 0.02^2) vectors (the model's init, 5 "
          "seeds): the share of retrieval that the hierarchy-built operator gives before any learning.\n")
        cols = ["vectors", "vocabulary", "V", "direct_queries", "direct_mean_targets"] + \
               [f"direct_{m}@{k}" for m in ("P", "R") for k in (1, 5, 10)]
        rr = retr[~retr.vectors.str.contains("sd\\)")]
        rnd = rr.groupby("vocabulary", sort=False).first().reset_index()
        rnd_rows = []
        for _, r in rnd.iterrows():
            rnd_rows.append({"vectors": "random ranking (expected)", "vocabulary": r["vocabulary"], "V": r["V"],
                             "direct_queries": r["direct_queries"], "direct_mean_targets": r["direct_mean_targets"],
                             **{f"direct_{m}@{k}": r[f"direct_random_{m}@{k}"] for m in ("P", "R") for k in (1, 5, 10)},
                             **{f"2hop_{m}@{k}": r[f"2hop_random_{m}@{k}"] for m in ("P", "R") for k in (1, 5, 10)},
                             "2hop_queries": r["2hop_queries"], "2hop_mean_targets": r["2hop_mean_targets"]})
        tab = pd.concat([rr, pd.DataFrame(rnd_rows)], ignore_index=True)
        tab = tab.sort_values("vocabulary", kind="stable")
        for c_ in ("V", "direct_queries", "2hop_queries"):
            tab[c_] = tab[c_].round().astype(int)
        sd = retr[retr.vectors.str.contains("sd\\)")]
        sdmax = float(sd[[c_ for c_ in sd.columns if "@" in c_ and "random" not in c_]].max().max())
        w(f"The untrained rows are means over 5 seeds; the largest seed sd of any P@k / R@k is {sdmax:.4f} "
          "(`retrieval.tsv`).\n")
        w("Direct parents + children:\n")
        w(md(tab[cols]))
        cols2 = ["vectors", "vocabulary", "2hop_queries", "2hop_mean_targets"] + \
                [f"2hop_{m}@{k}" for m in ("P", "R") for k in (1, 5, 10)]
        w("\nWithin 2 undirected hops:\n")
        w(md(tab[cols2]))
        cm = tab[tab.vocabulary.str.startswith("common")].set_index("vectors")

        def g(v, k):
            return float(cm.loc[v, k])

        vq, ve = "HiGO effective query q = e + mean_anc(e)", "HiGO raw term_emb e"
        vu = "untrained N(0,0.02^2) e through the same ancestor-mean (5 seeds, mean)"
        vg, vr = "GO-GPT token embedding (wte)", "random ranking (expected)"
        w(f"\nWhat it shows: whether terms that are adjacent in GO end up close in each model's term space. On the "
          f"common vocabulary, direct-neighbour P@1 is {g(vq, 'direct_P@1'):.3f} for HiGO's effective query, "
          f"{g(ve, 'direct_P@1'):.3f} for its raw term_emb, {g(vu, 'direct_P@1'):.3f} for untrained vectors through "
          f"the same ancestor-mean, {g(vg, 'direct_P@1'):.3f} for GO-GPT's token embedding and "
          f"{g(vr, 'direct_P@1'):.4f} for a random ranking (R@10: {g(vq, 'direct_R@10'):.3f}, "
          f"{g(ve, 'direct_R@10'):.3f}, {g(vu, 'direct_R@10'):.3f}, {g(vg, 'direct_R@10'):.3f}, "
          f"{g(vr, 'direct_R@10'):.4f}). For HiGO the effective query shares the ancestor mean with every term "
          "below the same ancestors, so part of the neighbourhood structure is built in; the untrained row measures "
          "that part, and the learned rows should be read against it, not only against random. "
          + ("Training adds to it (learned q above untrained q at P@1), and " if g(vq, 'direct_P@1') > g(vu, 'direct_P@1')
             else "Learned q is not above untrained q at P@1; ")
          + "the raw e_t (evaluated without the ancestor mean, though trained through it) scores "
          + ("above" if g(ve, 'direct_P@1') > g(vu, 'direct_P@1') else "below") + " the untrained construction. "
          "GO-GPT's token embeddings, learned from term co-occurrence and order in annotation sequences, are "
          + ("above" if g(vg, 'direct_P@1') > g(vr, 'direct_P@1') else "not above") + " random and "
          + ("below" if g(vg, 'direct_P@1') < g(ve, 'direct_P@1') else "above") + " HiGO's raw term_emb. "
          "What it does not show: retrieval of GO neighbours is not "
          "what either model is trained for (HiGO scores terms against a protein; GO-GPT predicts the next term "
          "token), so these numbers say nothing about prediction quality, and the vocabularies differ (compare "
          "the 'common' rows).\n")
    else:
        w("Skipped (--skip-retrieval).\n")

    w("## 8. Multiple testing\n")
    w(f"Paired permutation tests (swap HiGO-R2 and the GO-GPT row per protein with probability 1/2, "
      f"{a.perm_reps} reps, seed 1; two-sided p = (1 + #|d*| >= |d|) / (1 + reps), so the smallest attainable p "
      f"is {1 / (1 + a.perm_reps):.1e}). Holm-Bonferroni (`p_holm`) and Benjamini-Hochberg (`q_bh`) within each "
      "family; `p_holm_all` / `q_bh_all` over all 100 tests together.\n")
    w("Primary family (56 tests): {Fmax, Fmax_w, Smin_w, AUPR_micro, AUPR_micro_w, AUPR_prot, AUPR_prot_w} x "
      "{MF, BP, CC, mean} x {HiGO-R2 vs GO-GPT ten-sample, HiGO-R2 vs GO-GPT beam}.\n")
    w("Secondary family (44 tests): mean-over-aspects per-protein F1 and F1_w in each of the 7 train-identity bins x "
      "2 references (28), and the difference in Spearman(F1 or F1_w, train identity) for MF, BP, CC and mean x 2 "
      "references (16).\n")
    pr = pv[pv.family == "primary"].copy()
    w(md(pr[["comparison", "metric", "aspect", "diff", "lo", "hi", "p_raw", "p_holm", "q_bh", "flag"]], ".4g"))
    w("")
    sc_ = pv[pv.family == "secondary"].copy()
    w(md(sc_[["comparison", "metric", "aspect", "diff", "lo", "hi", "p_raw", "p_holm", "q_bh"]], ".4g"))
    np_ = int((pr.p_holm < 0.05).sum())
    ns_ = pr[pr.p_holm >= 0.05]
    nsl = [f"{r.metric} {r.aspect} vs {r.comparison.split(' - ')[1]} ({r['diff']:+.4f}, Holm p {r.p_holm:.3f})"
           for _, r in ns_.iterrows()]
    w(f"\nWhat it shows: {np_} of {len(pr)} primary comparisons have Holm-adjusted p < 0.05 "
      f"({int((pr.q_bh < 0.05).sum())} with BH q < 0.05). Not significant after Holm: {lst(nsl)}. In the secondary "
      f"family {int((sc_.p_holm < 0.05).sum())} of {len(sc_)} pass Holm and {int((sc_.q_bh < 0.05).sum())} pass BH; "
      "the significant bin differences have both signs (HiGO-R2 behind at low identity, ahead at >= 95%), and the "
      "Spearman differences say HiGO-R2 is more homology-dependent. All primary comparisons with Holm p < 0.05 "
      "favour HiGO-R2 " + ("(no primary comparison favours a GO-GPT row significantly). "
                           if not ((pr.p_holm < 0.05) & (((pr.metric.isin(LOWER_BETTER)) & (pr['diff'] > 0)) |
                                                         ((~pr.metric.isin(LOWER_BETTER)) & (pr['diff'] < 0)))).any()
                           else "except those listed with the opposite sign in the table. ")
      + "What it does "
      "not show: the tests condition on the 8,630 test proteins (protein-level resampling, not a new temporal "
      "split); a permutation p of the minimum value only says the observed difference is larger than every "
      "swapped one; and the metrics within a family are strongly correlated, so Holm is conservative here.\n")

    w("## Files\n")
    w("`headline.tsv` (Fmax, Fmax_w, micro-F), `auprc.tsv`, `smin.tsv`, `identity_bins.tsv`, `spearman.tsv`, "
      "`consistency.tsv`, `retrieval.tsv`, `differences.tsv` (bootstrap CIs of HiGO-R2 minus each GO-GPT row), "
      "`pvalues.tsv`, `crosscheck.tsv`, `summary.tsv` (all model rows and differences in long format).\n")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


def favours_main(r) -> bool:
    return (r["diff"] < 0) if r["metric"] in LOWER_BETTER else (r["diff"] > 0)


def test_counts(pv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ref in REFS:
        q = pv[pv.comparison == f"{MAIN} - {ref}"]
        r = {"comparison": f"{MAIN} - {ref}"}
        for fam in ("primary", "secondary"):
            f = q[q.family == fam]
            sig = f[f.p_holm < 0.05]
            r[f"{fam} tests"] = len(f)
            r[f"{fam} Holm < 0.05"] = len(sig)
            r[f"{fam} BH < 0.05"] = int((f.q_bh < 0.05).sum())
            r[f"{fam} Holm(all) < 0.05"] = int((f.p_holm_all < 0.05).sum())
            if fam == "primary":
                r["primary Holm < 0.05 favouring HiGO-R2"] = int(sum(favours_main(x) for _, x in sig.iterrows()))
        rows.append(r)
    return pd.DataFrame(rows)


def log_extended_summary(summ, pv, cons, cx):
    def val(m, metric):
        return float(summ[(summ.model == m) & (summ.metric == metric) & (summ.aspect == "mean")].value.iloc[0])

    ca = cons[cons.aspect == "all"].set_index("model")
    cnt = test_counts(pv).set_index("comparison")
    log("  per-row means (Fmax_w, AUPR_micro_w, AUPR_prot_w, Smin_w), violation rate, Holm < 0.05 primary/secondary:")
    for m in MODELS:
        c = f"{MAIN} - {m}"
        sig = (f"{cnt.loc[c, 'primary Holm < 0.05']}/{cnt.loc[c, 'primary tests']} primary, "
               f"{cnt.loc[c, 'secondary Holm < 0.05']}/{cnt.loc[c, 'secondary tests']} secondary") if c in cnt.index else ""
        log(f"    {m:40s} {val(m, 'Fmax_w'):.4f} {val(m, 'AUPR_micro_w'):.4f} {val(m, 'AUPR_prot_w'):.4f} "
            f"{val(m, 'Smin_w'):7.4f}  viol {ca.loc[m, 'rate']:.4f}  {sig}")
    g = cx[cx.scores == CP_SCORES]
    t = g.dropna(subset=["tau_cafaeval"])
    log(f"  compare_paper cross-check: {len(g)} values, max |diff| Fmax_w {g[g.metric == 'Fmax_w'].abs_diff.max():.1e}, "
        f"Smin_w {g[g.metric == 'Smin_w'].abs_diff.max():.1e}, Fmax {g[g.metric == 'Fmax'].abs_diff.max():.1e}; "
        f"Fmax_w thresholds equal {int((t.tau_cafaeval == t.tau_eval_suite).sum())}/{len(t)}")
    pr, se = pv[pv.family == "primary"], pv[pv.family == "secondary"]
    log(f"  Holm < 0.05: primary {int((pr.p_holm < 0.05).sum())}/{len(pr)}, secondary {int((se.p_holm < 0.05).sum())}/"
        f"{len(se)}; over all {len(pv)} tests {int((pv.p_holm_all < 0.05).sum())}")


def write_report_extended(summ, diffs, pv, binsdf, spdf, cons, cx, retr, rinfo, n_a, a, levels):
    def val(m, metric, asp="mean"):
        return float(summ[(summ.model == m) & (summ.metric == metric) & (summ.aspect == asp)].value.iloc[0])

    def wide(metric, models=MODELS, f=".4f"):
        rows = []
        for m in models:
            q = summ[(summ.model == m) & (summ.metric == metric)].set_index("aspect")
            rows.append({"row": m, **{c: f"{q.loc[c, 'value']:{f}} [{q.loc[c, 'lo']:{f}}, {q.loc[c, 'hi']:{f}}]"
                                      for c in COLS}})
        return pd.DataFrame(rows)

    def dv(ref, metric, asp="mean"):
        q = pv[(pv.comparison == f"{MAIN} - {ref}") & (pv.metric == metric) & (pv.aspect == asp)]
        if len(q) == 0:
            q = diffs[(diffs.comparison == f"{MAIN} - {ref}") & (diffs.metric == metric) & (diffs.aspect == asp)]
        return q.iloc[0]

    def dwide(metric):
        rows = []
        for ref in REFS:
            r = {"difference": f"{MAIN} - {ref}"}
            for c in COLS:
                q = dv(ref, metric, c)
                r[c] = f"{q['diff']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}]" + (
                    f" (Holm p {q['p_holm']:.4f})" if "p_holm" in q.index else "")
            rows.append(r)
        return pd.DataFrame(rows)

    def dtxt(ref, metric, asp="mean", f=".4f"):
        r = dv(ref, metric, asp)
        return f"{r['diff']:+{f}} [{r['lo']:+{f}}, {r['hi']:+{f}}]"

    def sides(ref, metric, keys=ASPECTS):
        up, down, mid = [], [], []
        for k in keys:
            r = dv(ref, metric, k)
            (up if r["lo"] > 0 else down if r["hi"] < 0 else mid).append(k)
        return up, down, mid

    def lst(x):
        return ", ".join(x) if x else "none"

    def mds(df):
        return df.to_markdown(index=False, disable_numparse=True)

    def ranking(metric, lower=False, models=MODELS):
        o = sorted(models, key=lambda m: val(m, metric), reverse=not lower)
        return "; ".join(f"{m} {val(m, metric):.4f}" for m in o)

    def per_ref(metric):
        out = []
        for ref in REFS:
            up, down, mid = sides(ref, metric)
            s = [f"{lab} {', '.join(x)}" for lab, x in (("above 0 in", up), ("below 0 in", down),
                                                        ("includes 0 in", mid)) if x]
            out.append(f"vs {ref} {dtxt(ref, metric)} (CI {'; '.join(s)})")
        return "; ".join(out)

    flagged = [m for m in REFS if m in FLAGS]
    cps = pd.read_csv(COMPARE / "summary.tsv", sep="\t")
    gauth = float(cps[(cps.subset == "S3_all") & (cps.method == "gogpt_authors")].Fw_mean.iloc[0])
    ca = cons[cons.aspect == "all"].set_index("model")
    cnt = test_counts(pv)
    cntc = cnt.set_index("comparison")
    L = []
    w = L.append
    w("# HiGO-R2 vs GO-GPT, BioReason-Pro and the compare_paper baselines: evaluation suite on GO-GPT's temporal "
      "test (extended)\n")
    w("Generated by `python -m cafa6.eval_suite --extended` (`src/cafa6/eval_suite.py`; metric definitions, "
      "statistics and scoring protocol in its module docstring, identical to `runs/eval_suite/report.md`). The "
      "default run and its outputs in `runs/eval_suite/` are unchanged; this directory adds the methods scored in "
      "`runs/compare_paper/` as rows and as comparisons. Every prediction file was frozen before this run; nothing "
      "was tuned, selected or changed on these numbers, and no test look is recorded. All numbers below come from "
      "the tsv files next to this report.\n")
    w(f"Ground truth: `data/proc_gogpt/gt_test.tsv`, 8,630 proteins (MF {n_a['MF']}, BP {n_a['BP']}, CC "
      f"{n_a['CC']} with annotations); ontology `runs/gogpt_cafa5/ref/go-basic_gogpt.obo`, IA "
      "`runs/gogpt_cafa5/ref/IA.txt`, cafaeval protocol (prop=max, norm=cafa, tau 0.01..0.99), scores in the "
      "3-decimal submission format. Brackets are paired protein bootstrap 95% CIs "
      f"({a.boot_reps} reps, seed 0, the same draws for every row). \"mean\" is the mean of the three aspects.\n")
    w("Rows:\n")
    w("- **HiGO-R2** `runs/stacker_tv/l3_c0/pred_test_twin.npz`; **GO-GPT ten-sample** "
      "`runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz`; **GO-GPT beam** `runs/gogpt_cafa5/preds_shard*.jsonl`; the "
      "two `[component]` rows (inputs of HiGO-R2, not models; no tests): as in `runs/eval_suite`.")
    w("- **GO-GPT greedy** `runs/gogpt_cafa5/greedy/preds_shard*.jsonl` (released checkpoint, greedy decoding, our "
      "generation; sets with score 1).")
    w("- **BioReason-Pro RL** `runs/compare_paper/preds/bioreason_rl.npz`: released `wanglab/bioreason-pro-rl`, "
      "the authors' `eval.py` with the published-benchmark settings of `scripts/sh_eval.sh` (greedy, "
      "ask_all_go_aspects=False, force_uniprot_summary=False); GO ids are taken from the whole response with "
      "score 1, as `evals/cafa_evals.py` does. In `cafa6/compare_paper.py` this is method key `bioreason_rl`, the "
      "row of `runs/compare_paper/summary.tsv` with S3 IA-weighted Fmax mean 0.6490. `bioreason_rl_final.npz` "
      "(final-answer block only, 0.6478) and the label-independent prompt run (0.6471) are variants and are not "
      "rows here.")
    w("- **BLAST-KNN (training refs)** `preds/blast_knn.npz` (paper section 4.4.2: best DIAMOND hit by percent "
      "identity among GO-GPT training proteins, all its propagated terms with score identity / 100); **BLAST-KNN "
      "(train+val refs)** `preds/blast_knn_trainval.npz` (same rule, training + validation proteins as references, "
      "the reference set of HiGO-R2's kNN); **DIAMONDScore** `preds/diamondscore.npz` (bitscore-weighted label mean "
      "over the top 50 hits); **Naive** `preds/naive.npz` (per-aspect training term frequency, the same vector for "
      "every protein).\n")
    w("The seven compare_paper rows (and, for the cross-check, nothing else) are read with "
      "`cafa6.compare_paper.load_method`, the loader behind `runs/compare_paper/summary.tsv`. Not included as rows: "
      "the GO-GPT authors' `go_pred` file, the two BioReason-Pro variants, and the BioReason-Pro SFT catalogue "
      "(1,213 test proteins only).\n")
    w("Score resolution of the raw files (distinct score values; share of proteins with predictions whose scores "
      "are all one value) and the AUPR flag:\n")
    w(mds(pd.DataFrame([{"row": m, "distinct scores": levels[m]["distinct"],
                        "proteins with one score value": f"{levels[m]['one_score_share']:.3f}",
                        "AUPR flag": "binary: one operating point" if m in FLAGS else ""} for m in MODELS])))
    w("")

    w("## Overview\n")
    rows = []
    for m in MODELS:
        c = f"{MAIN} - {m}"
        rows.append({"row": m, "Fmax_w": f"{val(m, 'Fmax_w'):.4f}", "Fmax": f"{val(m, 'Fmax'):.4f}",
                     "AUPR_micro_w": f"{val(m, 'AUPR_micro_w'):.4f}" + (" *" if m in FLAGS else ""),
                     "AUPR_prot_w": f"{val(m, 'AUPR_prot_w'):.4f}" + (" *" if m in FLAGS else ""),
                     "Smin_w": f"{val(m, 'Smin_w'):.3f}", "violation rate": f"{ca.loc[m, 'rate']:.4f}",
                     "primary Holm < 0.05": "—" if c not in cntc.index else
                     f"{cntc.loc[c, 'primary Holm < 0.05']}/{cntc.loc[c, 'primary tests']}",
                     "secondary Holm < 0.05": "—" if c not in cntc.index else
                     f"{cntc.loc[c, 'secondary Holm < 0.05']}/{cntc.loc[c, 'secondary tests']}"})
    w(mds(pd.DataFrame(rows)))
    w("\nMeans over MF, BP, CC. `*` = binary scores, AUPR measures one operating point. Test counts are tests of "
      "HiGO-R2 against that row with Holm-adjusted p < 0.05 within the family (section 8); the component rows "
      "have no tests.\n")

    w("## 0. Cross-checks\n")
    sub = cx[cx.scores == "submission format (3 decimals)"]
    rep = cx[cx.scores == "cafaeval input file replica (4 decimals)"]
    cpx = cx[cx.scores == CP_SCORES]
    fm = sub[sub.metric.isin(["Fmax", "Fmax_w"])]
    w(f"Against the official cafaeval output (`runs/test_look/look3_l3c0_twin/cafaeval/best_*.tsv`, the three "
      f"original rows, as in `runs/eval_suite`): max |diff| of Fmax / Fmax_w {fm.abs_diff.max():.1e} in the "
      f"submission format and {rep[rep.metric.isin(['Fmax', 'Fmax_w'])].abs_diff.max():.0e} for the HiGO-R2 "
      "4-decimal replica (the HiGO-R2 file given to cafaeval was rounded to 4 decimals; see the v1 report).\n")
    t = cpx.dropna(subset=["tau_cafaeval"])
    ok = cpx[cpx.metric.isin(["Fmax_w", "Smin_w"])].abs_diff.max() < 1e-3
    w(f"Against `runs/compare_paper/summary.tsv` (S3_all, written with 4 decimals by `knn_sharp.Scorer`, the "
      f"in-memory cafaeval copy): IA-weighted Fmax, IA-weighted S-min and unweighted Fmax for MF, BP, CC and the "
      f"mean, for every row that has a compare_paper key. **Check {'passed' if ok else 'FAILED'}: max |diff| "
      f"Fmax_w {cpx[cpx.metric == 'Fmax_w'].abs_diff.max():.1e}, Smin_w {cpx[cpx.metric == 'Smin_w'].abs_diff.max():.1e}"
      f" (required < 1e-3), Fmax {cpx[cpx.metric == 'Fmax'].abs_diff.max():.1e}; Fmax_w thresholds equal in "
      f"{int((t.tau_cafaeval == t.tau_eval_suite).sum())} of {len(t)} aspect rows.** Differences of this size are "
      "the 4-decimal rounding of summary.tsv. In `crosscheck.tsv` these rows have `scores` = "
      f"\"{CP_SCORES}\" and the compare_paper value in the `cafaeval` column.\n")
    rows = []
    for m, g in cpx.groupby("model", sort=False):
        tt = g.dropna(subset=["tau_cafaeval"])
        rows.append({"row": m, "Fmax_w mean (summary.tsv)": f"{g[(g.metric == 'Fmax_w') & (g.aspect == 'mean')].cafaeval.iloc[0]:.4f}",
                     "Fmax_w mean (suite)": f"{g[(g.metric == 'Fmax_w') & (g.aspect == 'mean')].eval_suite.iloc[0]:.4f}",
                     "Smin_w mean (summary.tsv)": f"{g[(g.metric == 'Smin_w') & (g.aspect == 'mean')].cafaeval.iloc[0]:.4f}",
                     "Smin_w mean (suite)": f"{g[(g.metric == 'Smin_w') & (g.aspect == 'mean')].eval_suite.iloc[0]:.4f}",
                     "max abs diff Fmax_w": f"{g[g.metric == 'Fmax_w'].abs_diff.max():.1e}",
                     "max abs diff Smin_w": f"{g[g.metric == 'Smin_w'].abs_diff.max():.1e}",
                     "max abs diff Fmax": f"{g[g.metric == 'Fmax'].abs_diff.max():.1e}",
                     "tau_w equal": f"{int((tt.tau_cafaeval == tt.tau_eval_suite).sum())}/{len(tt)}"})
    w(mds(pd.DataFrame(rows)))
    w("")

    w("## 1. Fmax and IA-weighted Fmax\n")
    w("Unweighted Fmax:\n")
    w(md(wide("Fmax")))
    w("\nIA-weighted Fmax (the CAFA/Kaggle metric):\n")
    w(md(wide("Fmax_w")))
    w("\nDifferences HiGO-R2 minus each row (bootstrap CI; Holm-adjusted permutation p within the primary family):\n")
    w("Fmax_w:\n")
    w(md(dwide("Fmax_w")))
    w("\nFmax:\n")
    w(md(dwide("Fmax")))
    w("\nCoverage at the Fmax_w threshold (share of annotated proteins with at least one positive-IA prediction "
      "there):\n")
    w(md(wide("cov_w")))
    w(f"\nWhat it shows: mean IA-weighted Fmax by row: {ranking('Fmax_w')}. HiGO-R2 minus each comparison row, "
      f"IA-weighted: {per_ref('Fmax_w')}. Unweighted: {per_ref('Fmax')}. The BioReason-Pro row "
      f"({val('BioReason-Pro RL', 'Fmax_w'):.4f}) scores at the level of the GO-GPT authors' predictions that its "
      f"prompt contains ({gauth:.4f} in `runs/compare_paper/summary.tsv`) and below the released GO-GPT "
      "checkpoint runs here. BLAST-KNN abstains on proteins without a DIAMOND hit, so its "
      "coverage stays below the hit rate (table above) and its precision is averaged over fewer proteins. What it "
      "does not show: Fmax is one operating point per aspect, chosen on the test set for every row (the CAFA "
      "protocol), and precision averages only over covered proteins.\n")

    w("## 2. AUPRC\n")
    for met, txt in (("AUPR_micro", "Term-centric micro AUPR (unweighted, roots included)"),
                     ("AUPR_micro_w", "IA-weighted term-centric micro AUPR"),
                     ("AUPR_prot", "Protein-centric PR area (unweighted)"),
                     ("AUPR_prot_w", "Protein-centric PR area (IA-weighted)")):
        w(f"{txt}:\n")
        w(md(wide(met)))
        w("")
    w("Differences:\n")
    for met in ("AUPR_micro", "AUPR_micro_w", "AUPR_prot", "AUPR_prot_w"):
        w(f"{met}:\n")
        w(md(dwide(met)))
        w("")
    w("Recall ceiling R(0.01) of the protein-centric curve (unweighted, then IA-weighted):\n")
    w(md(wide("Rmax")))
    w("")
    w(md(wide("Rmax_w")))
    one = [m for m in MODELS if m not in FLAGS and levels[m]["one_score_share"] > 0.99]
    w(f"\n**Flag: {lst(flagged)}.** These rows have binary scores (one distinct value in the raw file), so their "
      "micro curve has two operating points (score 1 and score 0) and their protein curve one; their AUPR values "
      "are close to precision x recall at that point and measure one operating point, not a ranking. The AUPR "
      "tests against them carry the flag in `pvalues.tsv`. "
      + (f"Not flagged but close to it: {lst(one)} give every term of a protein the same score (BLAST-KNN: the "
         "identity of the best hit), so within a protein there is no ranking; the curve points come only from "
         "differences between proteins. " if one else "")
      + "Naive gives the same vector to every protein, so it ranks terms but not proteins.\n")
    w(f"What it shows: mean IA-weighted micro AUPR by row: {ranking('AUPR_micro_w')}. Mean IA-weighted "
      f"protein-centric AUPR: {ranking('AUPR_prot_w')}. HiGO-R2 minus each row on AUPR_micro_w: "
      f"{per_ref('AUPR_micro_w')}. As in the v1 report, AUPR mixes ranking quality with score resolution and with "
      "how many pairs a row scores at all (recall ceiling); rows with coarse or binary scores are at a structural "
      "disadvantage that says nothing about their top predictions.\n")

    w("## 3. S-min\n")
    w("IA-weighted S-min (lower is better):\n")
    w(md(wide("Smin_w", f=".3f")))
    w("\nDifferences (negative = HiGO-R2 lower):\n")
    w(md(dwide("Smin_w")))
    w("\nCount-based S-min (unit term weights):\n")
    w(md(wide("Smin", f=".3f")))
    w(f"\nWhat it shows: mean IA-weighted S-min by row (lowest first): {ranking('Smin_w', lower=True)}. HiGO-R2 "
      f"minus each row: {per_ref('Smin_w')}. What it does not show: S-min is not normalised by annotation depth, "
      "so BP dominates the mean.\n")

    w("## 4. Per-protein F1 by sequence identity to the nearest training protein\n")
    w("Per-protein F1 at each row's own per-aspect Fmax threshold (F1 at the unweighted, F1_w at the IA-weighted "
      "one; no prediction = 0), averaged within identity bins and then over the three aspects; n = proteins in the "
      "bin (any aspect).\n")
    for view, txt in (("train", "Identity to GO-GPT training proteins only (`data/proc_gogpt_tv2/knn/test_knn.npz`):"),
                      ("train+val", "Identity to training + validation proteins (`knn_trainval/test_knn.npz`):")):
        w(txt + "\n")
        for lab in ("F1", "F1_w"):
            q = binsdf[(binsdf.identity == view) & (binsdf.aspect == "mean") & (binsdf.f1 == lab)]
            rows = [{"row": "n proteins", **{b: str(int(q[(q.model == MAIN) & (q.bin == b)].n.iloc[0])) for b in BINS}}]
            for m in MODELS:
                qq = q[q.model == m].set_index("bin")
                rows.append({"row": m, **{b: f"{qq.loc[b, 'mean_f1']:.3f} [{qq.loc[b, 'lo']:.3f}, {qq.loc[b, 'hi']:.3f}]"
                                          for b in BINS}})
            w(f"{lab}:\n")
            w(md(pd.DataFrame(rows)))
            w("")
    w("Bin differences HiGO-R2 minus each row (mean over aspects, train identity, bootstrap CI, Holm p within the "
      "secondary family):\n")
    rows = []
    for ref in REFS:
        for lab in ("F1", "F1_w"):
            r = {"difference": f"{MAIN} - {ref}", "f1": lab}
            for b in BINS:
                qq = dv(ref, f"bin {b} mean-over-aspects {lab} (train identity)")
                r[b] = f"{qq['diff']:+.3f} [{qq['lo']:+.3f}, {qq['hi']:+.3f}] p={qq['p_holm']:.3f}"
            rows.append(r)
    w(md(pd.DataFrame(rows)))
    hw = binsdf[(binsdf.identity == "train") & (binsdf.aspect == "mean") & (binsdf.f1 == "F1_w")]
    w("\nShare of proteins with at least one positive-IA prediction at the row's Fmax_w threshold, per bin (mean "
      "over aspects, train identity):\n")
    w(mds(pd.DataFrame([{"row": m, **{b: f"{float(hw[(hw.model == m) & (hw.bin == b)].covered.iloc[0]):.3f}"
                                     for b in BINS}} for m in MODELS])))
    parts = []
    for ref in REFS:
        up, down = [], []
        for b in BINS:
            r = dv(ref, f"bin {b} mean-over-aspects F1_w (train identity)")
            if r["p_holm"] < 0.05:
                (up if r["diff"] > 0 else down).append(f"{b} ({r['diff']:+.3f})")
        s = [f"{lab} {', '.join(x)}" for lab, x in (("HiGO-R2 ahead in", up), ("behind in", down)) if x]
        parts.append(f"vs {ref}: " + ("; ".join(s) if s else "no bin passes"))
    nohit = sorted(MODELS, key=lambda m: -float(hw[(hw.model == m) & (hw.bin == "no_hit")].mean_f1.iloc[0]))
    w("\nWhat it shows: bins where the F1_w difference passes Holm in the secondary family (train identity): "
      + "; ".join(parts) + ". F1_w on proteins without a DIAMOND hit to a training protein: "
      + "; ".join(f"{m} {float(hw[(hw.model == m) & (hw.bin == 'no_hit')].mean_f1.iloc[0]):.3f}" for m in nohit)
      + ". The homology rows (BLAST-KNN, DIAMONDScore, the kNN prior) have nothing to transfer there (the "
      "train+val BLAST-KNN and kNN prior can still hit a val protein), while the generative rows and Naive predict "
      "for every protein. What it does not show: the low-identity bins are small (see n), so their CIs are wide, "
      "and per-protein F1 at a global threshold is not Fmax.\n")

    w("## 5. Spearman correlation of per-protein F1 with nearest-training identity\n")
    rows = []
    for view in ("train", "train+val"):
        for lab in ("F1", "F1_w"):
            for m in MODELS:
                qq = spdf[(spdf.identity == view) & (spdf.f1 == lab) & (spdf.model == m)].set_index("aspect")
                rows.append({"identity": view, "f1": lab, "row": m,
                             **{c: f"{qq.loc[c, 'spearman']:.3f} [{qq.loc[c, 'lo']:.3f}, {qq.loc[c, 'hi']:.3f}]"
                                for c in COLS}})
    w(md(pd.DataFrame(rows)))
    w("\nDifferences in Spearman correlation (train identity, HiGO-R2 minus each row, Holm p within the secondary "
      "family):\n")
    rows = []
    for ref in REFS:
        for lab in ("F1", "F1_w"):
            r = {"difference": f"{MAIN} - {ref}", "f1": lab}
            for c in COLS:
                q = dv(ref, f"Spearman({lab}, train identity)", c)
                r[c] = f"{q['diff']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] p={q['p_holm']:.3f}"
            rows.append(r)
    w(md(pd.DataFrame(rows)))

    def rho(m):
        return float(spdf[(spdf.identity == "train") & (spdf.f1 == "F1_w") & (spdf.model == m)
                          & (spdf.aspect == "mean")].spearman.iloc[0])

    o = sorted(MODELS, key=rho, reverse=True)
    w("\nWhat it shows: mean Spearman(F1_w, train identity) by row: " + "; ".join(f"{m} {rho(m):.3f}" for m in o)
      + ". A larger value means per-protein quality tracks homology more closely; Naive has no protein-specific "
      "input, so its correlation comes only from which proteins are easy (shallow, frequent annotations). What it "
      "does not show: the correlation is dominated by the large high-identity bins, and a higher correlation "
      "together with a higher mean is not a weakness by itself.\n")

    w("## 6. Ontology consistency of the raw predictions\n")
    w("Violation = score(child) > score(direct parent) + 1e-6 over is_a/part_of edges inside a namespace, counted "
      "over (protein, child, parent) triples with score(child) > 0, on the raw files. Output vocabulary for "
      "`rate_parent_in_vocab`: the GO-GPT tokenizer for the GO-GPT rows, every ontology term for BioReason-Pro "
      "(free-text GO ids), the terms present in the file for the compare_paper baselines.\n")
    w(md(cons[cons.aspect == "all"][["model", "pairs", "violations", "rate", "share_proteins", "mean_gap",
                                      "parent_missing_share", "rate_parent_in_vocab"]]))
    w("\nPer aspect:\n")
    w(md(cons.pivot(index="model", columns="aspect", values="rate")[list(ASPECTS)].reindex(list(MODELS)).reset_index()))
    zero = [m for m in MODELS if int(ca.loc[m, "violations"]) == 0]
    nz = sorted([m for m in MODELS if m not in zero], key=lambda m: -ca.loc[m, "rate"])
    w(f"\nWhat it shows: rows with no violation: {lst(zero)} (propagated label sets or max-propagated outputs). "
      "Rows with violations, highest rate first: "
      + "; ".join(f"{m} {ca.loc[m, 'rate']:.2%} of pairs, {ca.loc[m, 'share_proteins']:.1%} of proteins, "
                  f"{ca.loc[m, 'parent_missing_share']:.0%} with the parent unscored" for m in nz)
      + ". For the binary rows (GO-GPT greedy and beam, BioReason-Pro) a violation is a listed term whose parent "
      "is not listed. The cafaeval scores are unaffected, because every file is max-propagated before scoring.\n")

    w("## 7. GO-relation retrieval from learned term vectors\n")
    if len(retr):
        w("HiGO and GO-GPT rows are those of `runs/eval_suite` (same vectors, vocabularies and targets). "
          f"Targets come from {rinfo['direct_edges']} direct is_a/part_of edges of go-basic_gogpt.obo; cosine "
          "similarity, all other vocabulary terms ranked; mean over terms with >= 1 target.\n")
        if "br_terms" in rinfo:
            rel = ", ".join(f"{k} {v}" for k, v in rinfo["br_relations"].items())
            w(f"BioReason-Pro has term vectors inside its GO graph encoder, so it is added: the authors' "
              "`GOGraphEncoderUnified` (`external/BioReason-Pro-github/bioreason2/models/go_graph_encoder.py`) "
              f"was built on CPU with the RL checkpoint's `go_encoder.pt` and the precomputed term embeddings "
              f"(`/dev/shm/cmp_paper_go_emb/go_embeddings`, {rinfo['br_terms']} terms, 2560-d). It runs 3 GAT "
              f"layers over the obonet graph of the authors' go-basic.obo ({rinfo['br_edges']} edges, all relation "
              f"types: {rel}), then a 2-layer projection back to 2560-d, then a cross-attention reducer to 200 "
              "vectors; only those 200 vectors reach the LLM, so per-term vectors exist only inside the encoder. "
              "Recomputing the 200 reduced vectors from the loaded weights reproduces the checkpoint's cached "
              f"`go_embedding.pt` to relative L2 error {rinfo['br_reduced_rel_err']:.1e} (min row cosine "
              f"{rinfo['br_reduced_min_cos']:.4f}), so the encoder is loaded as the model uses it. Rows: the "
              "per-term encoder output (input of the reducer), the last GAT layer (512-d), the precomputed input "
              "embeddings (frozen, not trained by BioReason-Pro), and the same encoder at fresh initialisation on "
              f"the same inputs (5 seeds; what the graph propagation gives without training). {rinfo['br_mapped']} "
              "terms map to a unique term of go-basic_gogpt.obo; on the common vocabulary of HiGO and GO-GPT "
              f"BioReason-Pro has {rinfo['br_common']} of {rinfo['common_vocab']} terms. Run time "
              f"{rinfo['br_seconds'] / 60:.1f} min on CPU.\n")
        cols = ["vectors", "vocabulary", "V", "direct_queries", "direct_mean_targets"] + \
               [f"direct_{m}@{k}" for m in ("P", "R") for k in (1, 5, 10)]
        rr = retr[~retr.vectors.str.contains("sd\\)")]
        rnd = rr.groupby("vocabulary", sort=False).first().reset_index()
        rnd_rows = []
        for _, r in rnd.iterrows():
            rnd_rows.append({"vectors": "random ranking (expected)", "vocabulary": r["vocabulary"], "V": r["V"],
                             "direct_queries": r["direct_queries"], "direct_mean_targets": r["direct_mean_targets"],
                             **{f"direct_{m}@{k}": r[f"direct_random_{m}@{k}"] for m in ("P", "R") for k in (1, 5, 10)},
                             **{f"2hop_{m}@{k}": r[f"2hop_random_{m}@{k}"] for m in ("P", "R") for k in (1, 5, 10)},
                             "2hop_queries": r["2hop_queries"], "2hop_mean_targets": r["2hop_mean_targets"]})
        tab = pd.concat([rr, pd.DataFrame(rnd_rows)], ignore_index=True)
        tab = tab.sort_values("vocabulary", kind="stable")
        for c_ in ("V", "direct_queries", "2hop_queries"):
            tab[c_] = tab[c_].round().astype(int)
        sd = retr[retr.vectors.str.contains("sd\\)")]
        sdmax = float(sd[[c_ for c_ in sd.columns if "@" in c_ and "random" not in c_]].max().max())
        w(f"Untrained rows are means over 5 seeds; the largest seed sd of any P@k / R@k is {sdmax:.4f} "
          "(`retrieval.tsv`).\n")
        w("Direct parents + children:\n")
        w(md(tab[cols]))
        cols2 = ["vectors", "vocabulary", "2hop_queries", "2hop_mean_targets"] + \
                [f"2hop_{m}@{k}" for m in ("P", "R") for k in (1, 5, 10)]
        w("\nWithin 2 undirected hops:\n")
        w(md(tab[cols2]))
        if "br_terms" in rinfo:
            cm = tab[tab.vocabulary == f"common ({rinfo['common_vocab']})"].set_index("vectors")

            def g(v, k="direct_P@1"):
                return float(cm.loc[v, k]) if v in cm.index else float("nan")

            names = (("HiGO effective query q = e + mean_anc(e)", "HiGO effective query"),
                     ("untrained N(0,0.02^2) e through the same ancestor-mean (5 seeds, mean)", "HiGO construction untrained"),
                     ("GO-GPT token embedding (wte)", "GO-GPT wte"),
                     ("BioReason-Pro GO encoder output per term (GAT x3 + projection, 2560-d)", "BioReason-Pro encoder output"),
                     ("BioReason-Pro last GAT layer per term (512-d)", "BioReason-Pro last GAT layer"),
                     ("BioReason-Pro precomputed GO term embedding (encoder input, frozen)", "BioReason-Pro input embedding"),
                     ("untrained BioReason-Pro encoder, fresh init (5 seeds, mean)", "BioReason-Pro encoder untrained"),
                     ("random ranking (expected)", "random"))
            own = tab[tab.vocabulary == f"BioReason-Pro vocab ({rinfo['br_mapped']})"].set_index("vectors")
            bo, bu = names[3][0], names[6][0]
            bsd = sd[(sd.vocabulary == f"BioReason-Pro vocab ({rinfo['br_mapped']})")].iloc[0]
            w(f"\nWhat it shows (common vocabulary, direct neighbours, P@1 / R@10): "
              + "; ".join(f"{lab} {g(v):.3f} / {g(v, 'direct_R@10'):.3f}" for v, lab in names) + ". "
              f"On its own {rinfo['br_mapped']}-term vocabulary BioReason-Pro's encoder output has direct P@1 "
              f"{float(own.loc[bo, 'direct_P@1']):.3f} against {float(own.loc[bu, 'direct_P@1']):.3f} for the "
              f"untrained encoder (seed sd {float(bsd['direct_P@1']):.3f}) and "
              f"{float(own.loc[names[5][0], 'direct_P@1']):.3f} for its input embeddings. So almost all of its "
              "neighbourhood structure is there before training: the precomputed inputs already place GO "
              "neighbours close, and three rounds of message passing over the GO graph (is_a, part_of and the "
              "regulates relations) average each term with its neighbours whatever the weights are. The same holds, "
              "less strongly, for HiGO's ancestor-mean construction (untrained row above). What it does not show: "
              "none of these models is trained for GO-neighbour retrieval, BioReason-Pro's LLM never sees these "
              "per-term vectors (only the 200 reduced ones), and the numbers say nothing about prediction quality.\n")
    else:
        w("Skipped (--skip-retrieval).\n")

    w("## 8. Multiple testing\n")
    npr, nse = int((pv.family == "primary").sum()), int((pv.family == "secondary").sum())
    w(f"Paired permutation tests ({a.perm_reps} reps, seed 1; two-sided; smallest attainable p "
      f"{1 / (1 + a.perm_reps):.1e}), Holm-Bonferroni (`p_holm`) and Benjamini-Hochberg (`q_bh`) within each "
      f"family, `p_holm_all` / `q_bh_all` over all {len(pv)} tests.\n")
    w(f"Primary family ({npr} tests): {{Fmax, Fmax_w, Smin_w, AUPR_micro, AUPR_micro_w, AUPR_prot, AUPR_prot_w}} x "
      f"{{MF, BP, CC, mean}} x {{HiGO-R2 vs each of the {len(REFS)} comparison rows: {', '.join(REFS)}}}.\n")
    w(f"Secondary family ({nse} tests): mean-over-aspects F1 and F1_w in the 7 train-identity bins (14) and the "
      f"Spearman(F1 or F1_w, train identity) difference for MF, BP, CC and mean (8), x {len(REFS)} comparison "
      "rows.\n")
    w("Counts per comparison:\n")
    w(md(cnt))
    pr, se = pv[pv.family == "primary"], pv[pv.family == "secondary"]
    sig = pr[pr.p_holm < 0.05]
    against = [f"{r.metric} {r.aspect} vs {r.comparison.split(' - ')[1]} ({r['diff']:+.4f})"
               for _, r in sig.iterrows() if not favours_main(r)]
    nsig = [f"{r.metric} {r.aspect} vs {r.comparison.split(' - ')[1]} ({r['diff']:+.4f}, Holm p {r.p_holm:.3f})"
            for _, r in pr[pr.p_holm >= 0.05].iterrows()]
    w(f"\nWhat it shows: {len(sig)} of {len(pr)} primary tests have Holm p < 0.05 ({int((pr.q_bh < 0.05).sum())} "
      f"with BH q < 0.05; {int((pr.p_holm_all < 0.05).sum())} with Holm over all {len(pv)} tests). Significant "
      f"primary differences in favour of the comparison row: {lst(against)}. Not significant after Holm "
      f"({len(nsig)}): " + (lst(nsig) if len(nsig) <= 30 else "see the table below") + ". "
      f"Secondary family: {int((se.p_holm < 0.05).sum())} of {len(se)} pass Holm, "
      f"{int((se.q_bh < 0.05).sum())} pass BH. What it does not show: the tests condition on the 8,630 test "
      "proteins; a p at the minimum only says the observed difference exceeds every swapped one; metrics within a "
      "family are strongly correlated, so Holm is conservative.\n")
    w("Primary family:\n")
    w(md(pr[["comparison", "metric", "aspect", "diff", "lo", "hi", "p_raw", "p_holm", "q_bh", "flag"]], ".4g"))
    w("\nSecondary family:\n")
    w(md(se[["comparison", "metric", "aspect", "diff", "lo", "hi", "p_raw", "p_holm", "q_bh"]], ".4g"))
    w("")
    w("## Files\n")
    w("Same names and columns as `runs/eval_suite/`: `headline.tsv`, `auprc.tsv`, `smin.tsv`, `identity_bins.tsv`, "
      "`spearman.tsv`, `consistency.tsv`, `retrieval.tsv`, `differences.tsv` (bootstrap CIs of HiGO-R2 minus each "
      "comparison row), `pvalues.tsv`, `crosscheck.tsv` (cafaeval rows as in v1, plus the compare_paper rows), "
      "`summary.tsv`; `log.txt` is the console output of the run.\n")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
