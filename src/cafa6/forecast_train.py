"""Annotation-forecast model (idea 1): arms F1 'forecast' and F0 'full-label' of the design pre-registered in
runs/test_looks.md (2026-10-19 19:15:00), on the data of data/forecast (README there).

Model (both arms, identical architecture and budget)
- Frozen ESM2-35M HiGO head (entmax pooling, hierarchical queries, MCM, taxon, IA-weighted MCLoss), built from
  configs/l3_c0_tv2.yaml and initialised from runs/l3_c0_tv2/trainable_weights.pt (taxon embedding included; the
  GO-GPT organism codes behind runs/l3_c0_tv2/taxa.json are looked up by accession, else by NCBI taxon).
- Residual on the input-time kNN prior:  z_t = a_aspect(t) * L_t + z_t^HiGO,  L_t = logit(clip(p_knn_t, 1e-3)).
  a is one learned scalar per aspect, init 1. Where the prior has nothing to say (no DIAMOND hit, or no
  neighbour annotated in that aspect, i.e. an empty prior row in the aspect) L_t is the term-prior logit
  logit(clip(f_t, 1e-3)), f_t = term frequency in the arm's training targets. On 30% of training batches
  (Bernoulli per batch, own RNG) the prior is withheld: L = term-prior logit for every protein in the batch.
- Output p = MCM(sigmoid(z)) over the 7,901-term HiGO vocabulary. At prediction time the kNN prior terms outside
  the vocabulary pass through with their raw prior score (p_full), as the stacker does; Scorer max-propagates.

Rows and targets
- One training example per protein (the head predicts all three aspects at once); its label row is the union of
  the protein's three (protein, aspect) rows.
- F1 target = input_terms + target_gained_terms (labels held after T). F0 target = input_terms (labels at the input
  snapshot). Input terms are NOT a model input: as a feature they would let F0 learn the identity map, so both arms
  see only sequence, taxon and the prior. F0 drops proteins without any input term (aspect_status 'nk' in all
  aspects), as the usual task trains only on annotated proteins; F1 keeps every train protein (an empty target is a
  forecast of "nothing arrives"). Empty aspects of kept proteins are all-negative in both arms, as in HiGO training.
- Prior: the period's input-time prior (data/forecast/{A,B}_knn_prior.npz). For F0 it is leave-self-out because
  the DIAMOND references exclude the query accession (checked by `verify`).
- Training data: period A train split only (B is evaluation only).

Budget: configs/l3_c0_tv2.yaml (15 epochs, head_lr 1e-3, AdamW cosine, warm-up 6%, wd 0.01 with wd_exclude, token
budget 16,384, max_len 1,022, taxon dropout 0.1, bf16 autocast) with fast_entmax on GPU. No early stopping: the
weights after the last epoch are used (the per-epoch valid numbers are printed, never used for a choice).

Evaluation (knn_sharp.Scorer protocol: top 1,500 per protein, cut 0.01, 3 decimals, max-propagation, CAFA-5 IA,
GO 2023-01-01). Ground truth of a (protein, aspect) row = its gained terms; rows without a gain are not scored.
Predicted terms that are input terms of the protein are removed after propagation (CAFA partial-knowledge rule),
which changes nothing on rows with an empty input aspect. Row sets: nk = aspect empty at input (aspect_status nk or
lk; the registered selection set on A valid), nk_only = no input term in any aspect, known = input terms in the
aspect, all = every row; per identity bin; B valid also on its TV v2 proteins (in_tv2).

fresh24 predictions (look #2, runs/fresh24_looks.md 2026-10-24 22:45; writes predictions only, scores nothing):
- Queries: data/fresh24/fresh24.fasta (24,141 proteins), NCBI taxon from data/fresh24/proteins.tsv. No input terms.
- Prior: the training-time definition. DIAMOND as build_forecast.cmd_diamond (--more-sensitive -e 1e-3 -k 60,
  --dbsize = GO-GPT train letters) against the served references data/serve/refs.dmnd (GO-GPT train + val, 133,054),
  best HSP per pair, then build_forecast.make_knn + knn_sharp.KNN.prior T5_k50_g0_na (aspect-normalised) over the
  served label matrix data/serve/labels.npz, which is checked to equal the S0 labels used for period A. The look-1
  prior (runs/fresh24/preds/ontologue_knn.npz) used -k 51; `predict` recomputes it from the look-1 hits as a check
  of the code path and reports how far the -k 60 prior is from it.
- Taxon: the training rule (most common GO-GPT organism code among example proteins with the same NCBI taxon),
  then data/serve/taxa.tsv (NCBI -> GO-GPT organism, as the served predictor resolves it), else 'other'.
- Scores: per seed, MCM probabilities over the vocabulary; F1_ens = logit mean of seeds 0, 1, 2; kNN prior terms
  outside the vocabulary pass through with their prior score; max-propagated over the GO graph, scores >= 0.01, in
  the compare_paper npz layout (ids, data, indices, indptr, shape, terms, meta).
- Spot-check first: 20 period-A valid proteins through the same path must reproduce runs/forecast/F1_s*/pred_A_valid.npy
  within 1e-3; otherwise nothing is written.

Usage:
  python -m cafa6.forecast_train verify
  python -m cafa6.forecast_train train --arm F1 --seed 0 [--device cuda] [--smoke 500] [--epochs N]
  python -m cafa6.forecast_train report [--smoke]
  python -m cafa6.forecast_train predict [--device cuda:0] [--threads 16]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer

from cafa6.data import ROOT, save_pred
from cafa6.knn_sharp import NS, NS_SHORT, TAU, Ontology, Scorer, to_submission
from cafa6.model import ASPECT_INDEX, build_higo, load_trainable
from cafa6.train import Batcher, build_optimizer, fmt_dur, iter_train_batches, param_groups, predict

FC = ROOT / "data" / "forecast"
GOGPT = ROOT / "data" / "proc_gogpt_tv2"
INIT = ROOT / "runs" / "l3_c0_tv2"
CFG = ROOT / "configs" / "l3_c0_tv2.yaml"
OUT = ROOT / "runs" / "forecast"
LOG = ROOT / "logs" / "forecast_train.log"
ASP = {"MFO": "F", "BPO": "P", "CCO": "C"}
NS_OF = {"MFO": "MF", "BPO": "BP", "CCO": "CC"}
EPS = 1e-3
WITHHOLD = 0.30
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]


def plog(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- data
class Period:
    """One period's examples as a protein table (row order = sorted accessions = prior file order)."""

    def __init__(self, name: str, vocab: pd.DataFrame, keep_proteins=None):
        self.name = name
        ex = pd.read_parquet(FC / f"{name}_examples.parquet")
        if keep_proteins is not None:
            ex = ex[ex.protein.isin(keep_proteins)]
        self.ex = ex.reset_index(drop=True)
        z = np.load(FC / f"{name}_knn_prior.npz", allow_pickle=False)
        pids = z["ids"].astype(str)
        prot = ex.drop_duplicates("protein").set_index("protein")
        self.ids = np.array(sorted(prot.index))
        pos = {p: i for i, p in enumerate(pids)}
        loc = np.array([pos[p] for p in self.ids])
        self.seqs = prot.loc[self.ids, "sequence"].tolist()
        self.ncbi = prot.loc[self.ids, "taxon"].values
        self.split = prot.loc[self.ids, "split"].values
        P = sp.csr_matrix((z["p_data"], z["p_indices"], z["p_indptr"]), shape=tuple(z["p_shape"]))
        Pf = sp.csr_matrix((z["pf_data"], z["pf_indices"], z["pf_indptr"]), shape=tuple(z["pf_shape"]))
        self.P, self.Pf = P[loc].tocsr(), Pf[loc].tocsr()
        self.terms_full = z["terms_full"].astype(str)
        self.has_hit = z["has_hit"][loc].astype(bool)
        self.max_pident = z["max_pident"][loc].astype(np.float32)
        tid = {t: i for i, t in enumerate(vocab.term)}
        self.row = {p: i for i, p in enumerate(self.ids)}
        n, T = len(self.ids), len(vocab)

        def mat(col_lists):
            r, c = [], []
            for pid, terms in col_lists:
                i = self.row[pid]
                for t in terms:
                    j = tid.get(t)
                    if j is not None:
                        r.append(i); c.append(j)
            M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, T))
            M.sum_duplicates(); M.data[:] = 1
            return M

        self.Y_in = mat(zip(ex.protein, ex.input_terms))
        self.Y_gain = mat(zip(ex.protein, ex.target_gained_terms))
        n_in = ex.groupby("protein").n_input.sum().reindex(self.ids).values
        self.any_input = n_in > 0


def taxon_indices(periods) -> np.ndarray:
    """HiGO taxon index per protein: GO-GPT organism code by accession (proc_gogpt train.parquet), else the most
    common code among example proteins with the same NCBI taxon, mapped through runs/l3_c0_tv2/taxa.json (0 = other)."""
    gg = pd.read_parquet(GOGPT / "train.parquet", columns=["id", "taxon"])
    code = dict(zip(gg.id, gg.taxon))
    tmap = {int(k): v for k, v in json.load(open(INIT / "taxa.json")).items()}
    pairs = pd.DataFrame([(t, code[p]) for per in periods for p, t in zip(per.ids, per.ncbi) if p in code],
                         columns=["ncbi", "code"])
    ncbi2code = pairs.groupby("ncbi").code.agg(lambda s: s.value_counts().index[0]).to_dict()
    out = []
    for per in periods:
        c = [code.get(p, ncbi2code.get(t, -1)) for p, t in zip(per.ids, per.ncbi)]
        out.append(np.array([tmap.get(int(x), 0) for x in c], np.int64))
    return out


class PriorLogits:
    """Batcher-compatible retrieval source: batch(rows, cap) -> {'L': B x T prior logits}; cap == 0 withholds."""

    def __init__(self, P: sp.csr_matrix, has_hit: np.ndarray, aspect: np.ndarray, term_freq: np.ndarray):
        self.P, self.has_hit = P, has_hit
        self.cols = [np.where(aspect == a)[0] for a in ("F", "P", "C")]
        f = np.clip(term_freq, EPS, 1 - EPS)
        self.L0 = np.log(f / (1 - f)).astype(np.float32)

    def batch(self, rows, cap=None) -> dict:
        rows = np.asarray(rows)
        L = np.broadcast_to(self.L0, (len(rows), len(self.L0))).copy()
        if cap == 0:
            return {"L": L}
        p = self.P[rows].toarray()
        lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).astype(np.float32)
        for cols in self.cols:
            use = self.has_hit[rows] & (p[:, cols] > 0).any(1)
            L[np.ix_(use, cols)] = lp[np.ix_(use, cols)]
        return {"L": L}


class ForecastHiGO(nn.Module):
    """z = a_aspect * L + z_HiGO; L comes in as knn['L'] (prior logit or term-prior logit)."""

    def __init__(self, higo, aspect: np.ndarray, a_init: float = 1.0):
        super().__init__()
        self.higo, self.n_terms = higo, higo.n_terms
        self.a = nn.Parameter(torch.full((3,), float(a_init)))
        self.register_buffer("asp", torch.as_tensor([ASPECT_INDEX[a] for a in aspect], dtype=torch.long))

    def forward(self, input_ids, attention_mask, res_mask, taxon=None, knn=None, **kw):
        z, _ = self.higo(input_ids, attention_mask, res_mask, taxon)
        return z.float() + self.a[self.asp][None] * knn["L"].float(), None

    def predict_proba(self, z):
        return self.higo.predict_proba(z)

    def loss(self, z, y, ia_weight=True):
        return self.higo.loss(z, y, ia_weight=ia_weight)


# ----------------------------------------------------------------------------- evaluation
class RowScorer:
    """knn_sharp.Scorer over the gained terms of one period's rows, with input-term removal after propagation and
    per-(protein, namespace) row masks."""

    def __init__(self, onto: Ontology, per: Period, split: str, vocab: pd.DataFrame):
        self.o, self.per = onto, per
        ex = per.ex[(per.ex.split == split)]
        self.ex = ex
        gt = pd.DataFrame([(p, t) for p, ts in zip(ex.protein, ex.target_gained_terms) for t in ts],
                          columns=["id", "term"])
        self.sc = Scorer(onto, gt)
        self.prot = per.ids[per.split == split]
        self.prow = np.array([per.row[p] for p in self.prot])
        # input terms of every scored protein, in ontology columns (excluded after propagation)
        r, c = [], []
        pos = {p: i for i, p in enumerate(self.prot)}
        for p, ts in zip(ex.protein, ex.input_terms):
            for t in ts:
                for j in onto.canon(t):
                    r.append(pos[p]); c.append(j)
        self.excl = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(self.prot), len(onto.terms)))
        self.excl.sum_duplicates(); self.excl.data[:] = 1
        self.vcols = np.array([onto.idx[t] for t in vocab.term])
        fcols = np.array([onto.idx.get(t, -1) for t in per.terms_full])
        oov = (fcols >= 0) & ~np.isin(fcols, self.vcols)
        self.f_sel, self.f_cols = np.where(oov)[0], fcols[oov]
        self.f_vsel = np.where(np.isin(fcols, self.vcols))[0]
        self.f_vcols = fcols[self.f_vsel]
        # row masks per namespace over sc.ids
        self.masks = {}
        ns_key = {"MFO": "MF", "BPO": "BP", "CCO": "CC"}
        for name, sel in self.row_sets(ex).items():
            m = {}
            for asp, ns in ns_key.items():
                ok = set(ex.protein[sel & (ex.aspect == asp)])
                m[ns] = np.array([p in ok for p in self.sc.ids])
            self.masks[name] = m
        bins = ex.drop_duplicates("protein").set_index("protein").bin.reindex(self.sc.ids).values
        self.bins = bins

    @staticmethod
    def row_sets(ex) -> dict:
        s = {"nk": (ex.n_input == 0).values, "nk_only": (ex.aspect_status == "nk").values,
             "known": (ex.aspect_status == "known").values, "all": np.ones(len(ex), bool)}
        if "in_tv2" in ex:
            s["tv2_nk"] = (ex.in_tv2 & (ex.n_input == 0)).values
            s["tv2_all"] = ex.in_tv2.values
        return s

    def full(self, pv: np.ndarray | None) -> sp.csr_matrix:
        """Scores over the ontology for the scored proteins: model vocab scores (or the kNN prior on the vocab when
        pv is None) plus the out-of-vocabulary kNN prior terms."""
        n, T = len(self.prot), len(self.o.terms)
        Pf = self.per.Pf[self.prow]
        oov = Pf[:, self.f_sel].tocoo()
        r, c, v = [oov.row], [self.f_cols[oov.col]], [oov.data]
        if pv is None:
            vv = Pf[:, self.f_vsel].tocoo()
            r.append(vv.row); c.append(self.f_vcols[vv.col]); v.append(vv.data)
        else:
            pv = np.asarray(pv, np.float32)
            rr, cc = np.nonzero(pv >= 0.005)
            r.append(rr); c.append(self.vcols[cc]); v.append(pv[rr, cc])
        M = sp.csr_matrix((np.concatenate(v).astype(np.float64), (np.concatenate(r), np.concatenate(c))),
                          shape=(n, T))
        M.sum_duplicates()
        return M

    def curves(self, P: sp.csr_matrix) -> dict:
        """Scorer.curves with the input terms removed after max-propagation."""
        sc, o = self.sc, self.o
        ids = self.prot
        m = np.array([p in sc.pos for p in ids])
        P = to_submission(P.tocsr()[np.where(m)[0]])
        P = o.propagate_max(P)
        X = self.excl[np.where(m)[0]]
        P = (P - P.multiply(X)).tocsr()
        P.eliminate_zeros()
        gi = np.array([sc.pos[p] for p in ids[m]], dtype=np.int64)
        rows = gi[np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))]
        cols, vals = P.indices, P.data
        w = o.ia[cols]
        nsc = o.ns[cols]
        key = rows * len(o.terms) + cols
        ing = np.isin(key, sc.gkey)
        kk = np.searchsorted(TAU, vals, side="right")
        out = {}
        for k in range(3):
            sel = (nsc == k) & (w > 0) & sc.in_ns[k, rows]
            r, kq, ww, g = rows[sel], kk[sel], w[sel], ing[sel]
            C = np.bincount(r * 100 + kq, weights=ww, minlength=sc.n * 100).reshape(sc.n, 100)
            I = np.bincount(r * 100 + kq, weights=ww * g, minlength=sc.n * 100).reshape(sc.n, 100)
            predw = np.cumsum(C[:, ::-1], axis=1)[:, ::-1][:, 1:]
            interw = np.cumsum(I[:, ::-1], axis=1)[:, ::-1][:, 1:]
            has = predw > 0
            prec = np.divide(interw, predw, out=np.zeros_like(predw), where=has)
            gw = sc.gtw[k][:, None]
            rec = np.divide(interw, gw, out=np.zeros_like(interw), where=gw > 0)
            idx = np.where(sc.in_ns[k])[0]
            out[NS_SHORT[NS[k]]] = {"idx": idx, "prec": prec[idx].astype(np.float32), "has": has[idx],
                                    "rec": rec[idx].astype(np.float32)}
        return out

    def fmax(self, cur: dict, rows: str, weights=None, bin_=None) -> dict:
        res, n = {}, {}
        for a, c in cur.items():
            ww = self.masks[rows][a].astype(float)
            if bin_ is not None:
                ww = ww * (self.bins == bin_)
            if weights is not None:
                ww = ww * weights
            ww = ww[c["idx"]]
            ne = ww.sum()
            n[a] = int((ww > 0).sum())
            if ne <= 0:
                res[a] = np.nan
                continue
            nh = ww @ c["has"]
            Pr = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
            R = (ww @ c["rec"]) / ne
            F = np.divide(2 * Pr * R, Pr + R, out=np.zeros(99), where=(Pr + R) > 0)
            F[nh <= 0] = 0
            res[a] = float(F.max())
        vals = [res[a] for a in ("MF", "BP", "CC") if not np.isnan(res.get(a, np.nan))]
        res["mean"] = float(np.mean(vals)) if vals else np.nan
        res["n"] = n
        return res

    def bootstrap(self, curA, curB, rows: str, reps: int = 1000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        inset = np.zeros(self.sc.n, bool)
        for m in self.masks[rows].values():
            inset |= m
        idx = np.where(inset)[0]
        d = []
        for _ in range(reps):
            w = np.bincount(rng.choice(idx, len(idx)), minlength=self.sc.n).astype(float)
            d.append(self.fmax(curA, rows, w)["mean"] - self.fmax(curB, rows, w)["mean"])
        return {"diff": self.fmax(curA, rows)["mean"] - self.fmax(curB, rows)["mean"],
                "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "reps": reps}


# ----------------------------------------------------------------------------- training
def setup(smoke: int = 0):
    vocab = pd.read_csv(GOGPT / "vocab.tsv", sep="\t")
    keepA = keepB = None
    if smoke:
        a = pd.read_parquet(FC / "A_examples.parquet", columns=["protein", "split", "n_input", "has_gain"])
        rng = np.random.default_rng(0)
        tr = a[a.split == "train"].protein.unique()
        va = a[(a.split == "valid")].protein.unique()
        # half of the smoke valid proteins have a gain in an empty aspect, so the nk set is not empty
        nkv = a[(a.split == "valid") & (a.n_input == 0) & a.has_gain].protein.unique()
        n_tr = max(smoke // 3, 8)
        keepA = set(rng.choice(tr, n_tr, replace=False)) | set(rng.choice(nkv, 30, replace=False)) \
            | set(rng.choice(va, 30, replace=False))
        b = pd.read_parquet(FC / "B_examples.parquet", columns=["protein", "split"])
        keepB = set(rng.choice(b[b.split == "valid"].protein.unique(), 60, replace=False))
    A, B = Period("A", vocab, keepA), Period("B", vocab, keepB)
    tA, tB = taxon_indices([A, B])
    return vocab, A, B, tA, tB


PLM_TAG = {"facebook/esm2_t33_650M_UR50D": "650M"}


def run_name(arm, seed, plm=None):
    return f"{arm}_{PLM_TAG[plm]}_s{seed}" if plm else f"{arm}_s{seed}"


def run_dir(arm, seed, smoke, plm=None):
    return (ROOT / "runs" / "forecast_smoke" if smoke else OUT) / run_name(arm, seed, plm)


def save_ckpt(path, model, opt, sched, epoch, wrng, elapsed):
    state = {"model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
             "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "wrng": wrng.bit_generator.state,
             "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "elapsed": elapsed}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(arm: str, seed: int, device: str, smoke: int = 0, epochs: int | None = None, plm: str | None = None):
    """plm (round 2): another frozen encoder, run in bf16; the head is then freshly initialised (term_bias at the
    logit of the arm's training term frequency, as term_bias_init: prior in the recipe) instead of loaded from
    runs/l3_c0_tv2, whose head only fits the 35M hidden size. Everything else is unchanged."""
    assert arm in ("F1", "F0")
    t_start = time.time()
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    cfg = yaml.safe_load(open(CFG))
    on_gpu = device.startswith("cuda")
    cfg.update(fast_entmax=on_gpu, seed=seed)
    if plm:
        cfg.update(plm=plm, plm_bf16=on_gpu)
    if epochs is not None:
        cfg["epochs"] = epochs
    run = run_dir(arm, seed, smoke, plm)
    run.mkdir(parents=True, exist_ok=True)
    tag = f"{run.name}{' smoke' if smoke else ''}"
    vocab, A, B, tA, tB = setup(smoke)
    aspect = vocab.aspect.values
    Y = (A.Y_in + A.Y_gain) if arm == "F1" else A.Y_in.copy()
    Y = Y.tocsr(); Y.data[:] = 1
    tr = np.where(A.split == "train")[0]
    if arm == "F0":
        tr = tr[A.any_input[tr]]
    va = np.where(A.split == "valid")[0]
    freq = np.asarray(Y[tr].mean(0)).ravel()
    exA_tr = A.ex[A.ex.protein.isin(A.ids[tr])]
    info = {"arm": arm, "seed": seed, "device": device, "train_proteins": int(len(tr)),
            "train_rows": int(len(exA_tr)), "train_positive_pairs": int(Y[tr].sum()),
            "train_pairs_per_protein": float(Y[tr].sum() / max(len(tr), 1)),
            "train_gain_rows": int(exA_tr.has_gain.sum()), "valid_proteins": int(len(va)),
            "B_valid_proteins": int((B.split == "valid").sum()), "n_terms": len(vocab),
            "prior_hit_share_train": float(A.has_hit[tr].mean()), "epochs": cfg["epochs"],
            "head_lr": cfg["head_lr"], "withhold": WITHHOLD}
    plog(f"TRAIN {tag} START {json.dumps(info)}")
    (run / "config.json").write_text(json.dumps({**cfg, **info}, indent=2))

    tok = AutoTokenizer.from_pretrained(cfg["plm"])
    priorA = PriorLogits(A.P, A.has_hit, aspect, freq)
    priorB = PriorLogits(B.P, B.has_hit, aspect, freq)
    batcher = Batcher(A.seqs, tok, cfg["max_len"], cfg["tokens_per_batch"], cfg["max_batch"], knn=priorA, seed=seed)
    anc = np.load(GOGPT / "go_ancestors.npy")
    higo = build_higo(cfg, vocab, anc, len(json.load(open(INIT / "taxa.json"))) + 1)
    if plm:
        f = np.clip(freq, 1e-4, 1 - 1e-4)
        higo.term_bias.data.copy_(torch.as_tensor(np.log(f / (1 - f)), dtype=torch.float32))
    else:
        load_trainable(higo, torch.load(INIT / "trainable_weights.pt", map_location="cpu"))
    model = ForecastHiGO(higo, aspect).to(device)
    groups = param_groups(higo, cfg) + [{"params": [model.a], "lr": cfg["head_lr"], "weight_decay": 0.0}]
    all_params = [p for g in groups for p in g["params"]]
    steps_per_epoch = len(batcher.batches(tr, True))
    total = steps_per_epoch * cfg["epochs"]
    opt, sched, _ = build_optimizer(cfg, groups, all_params, total)
    plog(f"TRAIN {tag} steps/epoch {steps_per_epoch}, total {total}, head params "
         f"{sum(p.numel() for p in all_params):,}")

    onto = Ontology()
    rsA = RowScorer(onto, A, "valid", vocab)
    wrng = np.random.default_rng(1000 + seed)
    ckpt = run / "checkpoint_last.pt"
    start_ep, elapsed0 = 0, 0.0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict({**model.state_dict(), **ck["model"]})
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, elapsed0 = ck["epoch"], ck["elapsed"]
        wrng.bit_generator.state = ck["wrng"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        plog(f"TRAIN {tag} resumed at epoch {start_ep}")
    t0 = time.time() - elapsed0
    log = open(run / "train_log.jsonl", "a")
    make = lambda b: batcher.train_tensors(b, Y, tA, pin=on_gpu)
    for ep in range(start_ep, cfg["epochs"]):
        model.train()
        ep_t0 = last_plog = time.time()
        tot, nb, n_with = 0.0, 0, 0
        batches = batcher.batches(tr, True)
        for bi, (b, ids, att, res, y, tx, n_tok) in iter_train_batches(batches, 0, make, on_gpu):
            ids, att, res, y, tx = (t.to(device, non_blocking=on_gpu) for t in (ids, att, res, y, tx))
            tx = torch.where(torch.rand_like(tx, dtype=torch.float) < cfg.get("taxon_dropout", 0.1), 0, tx)
            withhold = wrng.random() < WITHHOLD
            n_with += withhold
            kn = batcher.retrieval(b, 0 if withhold else None, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu):
                z, _ = model(ids, att, res, tx, knn=kn)
            loss = model.loss(z.float(), y, ia_weight=cfg.get("ia_weight", True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(loss.detach()); nb += 1
            if bi % 200 == 0:
                print(f"ep{ep} step{bi}/{len(batches)} loss {tot / nb:.5f} lr {sched.get_last_lr()[0]:.2e} "
                      f"a {[round(float(x), 3) for x in model.a.detach()]} {time.time() - t0:.0f}s", flush=True)
            if time.time() - last_plog > 600:
                last_plog = time.time()
                plog(f"TRAIN {tag} epoch {ep + 1} step {bi}/{len(batches)} loss {tot / nb:.5f} "
                     f"({fmt_dur(time.time() - ep_t0)} into the epoch)")
        train_s = time.time() - ep_t0
        pv = predict(model, batcher, A.seqs, tA, va, device, bf16=on_gpu).astype(np.float32)
        cur = rsA.curves(rsA.full(pv))
        fnk, fall = rsA.fmax(cur, "nk"), rsA.fmax(cur, "all")
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "withheld_batches": int(n_with), "batches": nb,
               "a": [float(x) for x in model.a], "A_valid_nk": fnk, "A_valid_all": fall,
               "train_s": train_s, "val_s": time.time() - ep_t0 - train_s, "elapsed_s": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        plog(f"TRAIN {tag} epoch {ep + 1}/{cfg['epochs']} loss {rec['train_loss']:.5f} | a F/P/C "
             f"{'/'.join(f'{x:.3f}' for x in rec['a'])} | A valid nk Fmax MF {fnk['MF']:.4f} BP {fnk['BP']:.4f} "
             f"CC {fnk['CC']:.4f} mean {fnk['mean']:.4f} | all-rows mean {fall['mean']:.4f} | withheld "
             f"{n_with}/{nb} | train {fmt_dur(train_s)} val {fmt_dur(rec['val_s'])}")
        save_ckpt(ckpt, model, opt, sched, ep + 1, wrng, time.time() - t0)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
               run / "trainable_weights.pt")
    # predictions with the final weights: A valid (prior A), B valid (prior B)
    pA = predict(model, batcher, A.seqs, tA, va, device, bf16=on_gpu)
    np.save(run / "pred_A_valid.npy", pA)
    np.savetxt(run / "ids_A_valid.txt", A.ids[va], fmt="%s")
    vb = np.where(B.split == "valid")[0]
    bB = Batcher(B.seqs, tok, cfg["max_len"], cfg["tokens_per_batch"], cfg["max_batch"], knn=priorB, seed=seed)
    pB = predict(model, bB, B.seqs, tB, vb, device, bf16=on_gpu)
    np.save(run / "pred_B_valid.npy", pB)
    np.savetxt(run / "ids_B_valid.txt", B.ids[vb], fmt="%s")
    info.update(train_time_s=time.time() - t0, total_s=time.time() - t_start, a=[float(x) for x in model.a])
    (run / "info.json").write_text(json.dumps(info, indent=2))
    plog(f"TRAIN {tag} END a F/P/C {'/'.join(f'{x:.3f}' for x in info['a'])}, wall {fmt_dur(info['total_s'])}; "
         f"wrote {run}")


# ----------------------------------------------------------------------------- report
def logit_mean(ps):
    zs = [np.log(np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6))
          for p in ps]
    return (1 / (1 + np.exp(-np.mean(zs, 0)))).astype(np.float32)


def report(smoke: bool = False, seeds=(0, 1, 2), reps: int = 1000):
    base = ROOT / "runs" / "forecast_smoke" if smoke else OUT
    (base / "preds").mkdir(parents=True, exist_ok=True)
    vocab, A, B, _, _ = setup(500 if smoke else 0)
    onto = Ontology()
    rs = {"A": RowScorer(onto, A, "valid", vocab), "B": RowScorer(onto, B, "valid", vocab)}
    preds = {}
    for per in ("A", "B"):
        for arm in ("F1", "F0"):
            ps = []
            for s in seeds:
                d = base / f"{arm}_s{s}"
                if not (d / f"pred_{per}_valid.npy").exists():
                    continue
                ids = np.loadtxt(d / f"ids_{per}_valid.txt", dtype=str, ndmin=1)
                assert (ids == rs[per].prot).all(), f"{d}: id order differs"
                p = np.load(d / f"pred_{per}_valid.npy")
                preds[(per, f"{arm}_s{s}")] = p
                ps.append(p)
            if ps:
                preds[(per, f"{arm}_ens")] = logit_mean(ps)
        preds[(per, "kNN")] = None
        np.savetxt(base / "preds" / f"ids_{per}_valid.txt", rs[per].prot, fmt="%s")
    sets = {"A": ["nk", "nk_only", "known", "all"], "B": ["nk", "nk_only", "known", "all", "tv2_nk", "tv2_all"]}
    rows, bins, curs = [], [], {}
    for (per, m), p in preds.items():
        cur = rs[per].curves(rs[per].full(p))
        curs[(per, m)] = cur
        for st in sets[per]:
            f = rs[per].fmax(cur, st)
            rows.append({"period": per, "rows": st, "method": m, "MF": f["MF"], "BP": f["BP"], "CC": f["CC"],
                         "mean": f["mean"], "n_MF": f["n"]["MF"], "n_BP": f["n"]["BP"], "n_CC": f["n"]["CC"]})
        for st in ("nk", "all"):
            for b in BINS:
                f = rs[per].fmax(cur, st, bin_=b)
                bins.append({"period": per, "rows": st, "bin": b, "method": m, "mean": f["mean"],
                             "MF": f["MF"], "BP": f["BP"], "CC": f["CC"], "n": sum(f["n"].values())})
    S = pd.DataFrame(rows)
    S.to_csv(base / "summary.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(bins).to_csv(base / "per_bin.tsv", sep="\t", index=False, float_format="%.4f")
    for (per, m), p in preds.items():
        if p is not None:
            save_pred(base / "preds", f"{m}_{per}_valid", p.astype(np.float16), 500)

    def val(per, st, m):
        r = S[(S.period == per) & (S.rows == st) & (S.method == m)]
        return float(r["mean"].iloc[0]) if len(r) else np.nan

    have = [s for s in seeds if ("A", f"F1_s{s}") in preds and ("A", f"F0_s{s}") in preds]
    knnA = val("A", "nk", "kNN")
    seed_tab = []
    for s in have:
        f1, f0 = val("A", "nk", f"F1_s{s}"), val("A", "nk", f"F0_s{s}")
        seed_tab.append({"seed": s, "F1": f1, "F0": f0, "kNN": knnA, "F1-kNN": f1 - knnA, "F1-F0": f1 - f0,
                         "B_F1": val("B", "nk", f"F1_s{s}"), "B_F0": val("B", "nk", f"F0_s{s}"),
                         "B_kNN": val("B", "nk", "kNN")})
    T = pd.DataFrame(seed_tab)
    m1 = float(T.F1.mean()) if len(T) else np.nan
    m0 = float(T.F0.mean()) if len(T) else np.nan
    d_knn, d_f0 = m1 - knnA, m1 - m0
    c1 = d_knn >= 0.010 and bool((T["F1-kNN"] > 0).all())
    c2 = d_f0 >= 0.005 and bool((T["F1-F0"] > 0).all())
    complete = len(have) == len(seeds)
    kept = c1 and c2 and complete
    boots = {}
    for per in ("A", "B"):
        if (per, "F1_ens") in curs:
            for other in ("kNN", "F0_ens"):
                if (per, other) in curs:
                    boots[(per, other)] = rs[per].bootstrap(curs[(per, "F1_ens")], curs[(per, other)], "nk", reps)
    res = {"seeds": have, "F1_mean": m1, "F0_mean": m0, "kNN": knnA, "F1-kNN": d_knn, "F1-F0": d_f0,
           "cond_i": c1, "cond_ii": c2, "complete": complete, "kept": kept,
           "bootstrap_ens_nk": {f"{p}:F1_ens-{o}": v for (p, o), v in boots.items()}}
    (base / "rule.json").write_text(json.dumps(res, indent=2))
    write_md(base, S, pd.DataFrame(bins), T, res, smoke)
    plog(f"REPORT{' smoke' if smoke else ''}: A valid nk F1 mean {m1:.4f} F0 mean {m0:.4f} kNN {knnA:.4f}; "
         f"F1-kNN {d_knn:+.4f} F1-F0 {d_f0:+.4f}; cond (i) {c1} (ii) {c2}; seeds {have} -> "
         f"{'KEPT' if kept else 'NOT KEPT'}")
    return res


# ----------------------------------------------------------------------------- round 2 selection
R2_FAMILIES = {"a": (None, "F1-35M ensemble"), "b": ("facebook/esm2_t33_650M_UR50D", "F1-650M ensemble")}
R2_THETAS = (80, 90, 95, 98)
R2_MARGIN = 0.003


def round2_candidates(families_ok):
    """(name, family, theta, simplicity group) in the registered order (a) < (a)+routing < (b) < (b)+routing."""
    out = []
    for g, fam in enumerate(("a", "b")):
        if fam not in families_ok:
            continue
        out.append((fam, fam, None, 2 * g))
        out += [(f"{fam}+route{t}", fam, t, 2 * g + 1) for t in R2_THETAS]
    return out


def round2(seeds=(0, 1, 2), base: Path | None = None):
    """Registered criterion (runs/test_looks.md 2026-10-25 01:15:00): mean of IA-weighted Fmax over all rows of A valid
    and all rows of B valid, seed ensembles; routing = kNN prior (p_full) for proteins with best input-time hit
    identity >= theta. Choice: simplest within 0.003 of the best; theta does not order candidates inside a group."""
    t0 = time.time()
    base = base or OUT / "round2"
    base.mkdir(parents=True, exist_ok=True)
    plog(f"ROUND2 selection START -> {base.relative_to(ROOT)}")
    vocab, A, B, _, _ = setup(0)
    onto = Ontology()
    rs = {"A": RowScorer(onto, A, "valid", vocab), "B": RowScorer(onto, B, "valid", vocab)}
    pers = {"A": A, "B": B}
    ens, ok, dropped, per_seed = {}, [], {}, []
    for fam, (plm, _) in R2_FAMILIES.items():
        dirs = [OUT / run_name("F1", s, plm) for s in seeds]
        miss = [str(d.name) for d in dirs for per in ("A", "B") if not (d / f"pred_{per}_valid.npy").exists()]
        if miss:
            dropped[fam] = f"missing predictions: {sorted(set(miss))}"
            continue
        ok.append(fam)
        for per in ("A", "B"):
            ps = []
            for d in dirs:
                ids = np.loadtxt(d / f"ids_{per}_valid.txt", dtype=str, ndmin=1)
                assert (ids == rs[per].prot).all(), f"{d}: id order differs"
                ps.append(np.load(d / f"pred_{per}_valid.npy"))
                cur = rs[per].curves(rs[per].full(ps[-1]))
                per_seed.append({"family": fam, "run": d.name, "period": per,
                                 "all": rs[per].fmax(cur, "all")["mean"], "nk": rs[per].fmax(cur, "nk")["mean"]})
            ens[(fam, per)] = rs[per].full(logit_mean(ps)).tocsr()
    knn = {per: rs[per].full(None).tocsr() for per in ("A", "B")}
    rows = []

    def score(per, M):
        cur = rs[per].curves(M)
        return {st: rs[per].fmax(cur, st) for st in ("all", "nk")}

    for per in ("A", "B"):
        f = score(per, knn[per])
        rows.append({"candidate": "kNN prior alone (reference, not a candidate)", "period": per, **{
            f"{st}_{k}": f[st][k] for st in f for k in ("MF", "BP", "CC", "mean")}, "routed_share": 1.0})
    for name, fam, theta, grp in round2_candidates(ok):
        for per in ("A", "B"):
            M = ens[(fam, per)]
            route = np.zeros(len(rs[per].prot), bool)
            if theta is not None:
                route = pers[per].max_pident[rs[per].prow] >= theta
                M = (sp.diags((~route).astype(float)) @ M + sp.diags(route.astype(float)) @ knn[per]).tocsr()
            f = score(per, M)
            rows.append({"candidate": name, "family": fam, "theta": theta, "group": grp, "period": per,
                         **{f"{st}_{k}": f[st][k] for st in f for k in ("MF", "BP", "CC", "mean")},
                         "routed_share": float(route.mean())})
        plog(f"ROUND2 scored {name} ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(rows)
    by = {per: R[R.period == per].set_index("candidate") for per in ("A", "B")}
    tab = pd.DataFrame([{"candidate": n, "family": fam, "theta": np.nan if t is None else float(t), "group": grp,
                         **{f"{v}_{per}": by[per].loc[n, v] for per in ("A", "B")
                            for v in ("all_mean", "nk_mean", "routed_share")}}
                        for n, fam, t, grp in round2_candidates(ok)])
    tab["criterion"] = (tab.all_mean_A + tab.all_mean_B) / 2
    tab["secondary_nk"] = (tab.nk_mean_A + tab.nk_mean_B) / 2
    best = float(tab.criterion.max())
    tab["within_margin"] = tab.criterion >= best - R2_MARGIN - 1e-12
    elig = tab[tab.within_margin]
    g = int(elig.group.min())
    pick = elig[elig.group == g].sort_values("criterion", ascending=False).iloc[0]
    kref = R[R.candidate.str.startswith("kNN")].set_index("period")
    tab.to_csv(base / "selection.tsv", sep="\t", index=False, float_format="%.4f")
    R.to_csv(base / "selection_detail.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(per_seed).to_csv(base / "per_seed.tsv", sep="\t", index=False, float_format="%.4f")
    theta = None if pd.isna(pick.theta) else float(pick.theta)
    choice = {"registration": "runs/test_looks.md 2026-10-25 01:15:00", "families_trained": ok, "dropped": dropped,
              "best_criterion": best, "best_candidate": str(tab.loc[tab.criterion.idxmax(), "candidate"]),
              "margin": R2_MARGIN, "eligible": elig.candidate.tolist(),
              "chosen": {"candidate": str(pick.candidate), "family": str(pick.family),
                         "plm": R2_FAMILIES[pick.family][0] or "facebook/esm2_t12_35M_UR50D", "theta": theta,
                         "criterion": float(pick.criterion), "A_all": float(pick.all_mean_A),
                         "B_all": float(pick.all_mean_B), "secondary_nk": float(pick.secondary_nk)},
              "kNN_reference": {"A_all": float(kref.loc["A", "all_mean"]), "B_all": float(kref.loc["B", "all_mean"]),
                                "criterion": float((kref.loc["A", "all_mean"] + kref.loc["B", "all_mean"]) / 2)},
              "rule": "simplest candidate within 0.003 of the best criterion; order (a) < (a)+routing < (b) < "
                      "(b)+routing; theta does not order candidates inside a group; ties by higher criterion"}
    (base / "choice.json").write_text(json.dumps(choice, indent=2))
    write_round2_md(base, tab, R, pd.DataFrame(per_seed), choice)
    plog(f"ROUND2 selection END: best {choice['best_candidate']} {best:.4f}; eligible {choice['eligible']}; CHOSEN "
         f"{pick.candidate} (criterion {pick.criterion:.4f}) in {fmt_dur(time.time() - t0)}")
    return choice


def write_round2_md(base, tab, R, PS, choice):
    def f(v):
        return "" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))

    c = choice["chosen"]
    L = ["# Forecasting model, round 2: encoder size and routing", "",
         "Registration: runs/test_looks.md, entry 2026-10-25 01:15:00. Code: `src/cafa6/forecast_train.py` (`train --plm`, "
         "`round2`, `predict --round2`). Log: `logs/forecast_train.log`.", "",
         "Criterion = mean of the IA-weighted Fmax (mean over MF/BP/CC) on all rows of A valid and all rows of B valid "
         "(gained terms; input terms removed from predictions after propagation), seed ensembles (logit mean of seeds "
         "0, 1, 2). Routing sends every aspect of a protein whose best input-time hit identity is >= theta to the "
         "input-time kNN prior. No-knowledge rows (empty aspect at input) are secondary.", "",
         f"- families trained: {choice['families_trained']}; dropped: {choice['dropped'] or 'none'}",
         f"- best criterion: {choice['best_candidate']} {choice['best_criterion']:.4f}; eligible within "
         f"{choice['margin']}: {', '.join(choice['eligible'])}",
         f"- **chosen: {c['candidate']}** (encoder {c['plm']}, theta {c['theta']}), criterion {c['criterion']:.4f} "
         f"(A all {c['A_all']:.4f}, B all {c['B_all']:.4f}; secondary nk {c['secondary_nk']:.4f})",
         f"- kNN prior alone (reference): criterion {choice['kNN_reference']['criterion']:.4f} "
         f"(A all {choice['kNN_reference']['A_all']:.4f}, B all {choice['kNN_reference']['B_all']:.4f})", "",
         "## Selection table", "",
         "| candidate | group | criterion | A all | B all | A nk | B nk | secondary nk | routed A | routed B | within 0.003 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.to_dict("records"):
        L.append(f"| {r['candidate']} | {int(r['group'])} | {f(r['criterion'])} | {f(r['all_mean_A'])} | "
                 f"{f(r['all_mean_B'])} | {f(r['nk_mean_A'])} | {f(r['nk_mean_B'])} | {f(r['secondary_nk'])} | "
                 f"{f(r['routed_share_A'])} | {f(r['routed_share_B'])} | {'yes' if r['within_margin'] else ''} |")
    L += ["", "## Per aspect (all rows and nk rows)", "",
          "| candidate | period | all MF | all BP | all CC | all mean | nk MF | nk BP | nk CC | nk mean |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R.to_dict("records"):
        L.append(f"| {r['candidate']} | {r['period']} | " + " | ".join(
            f(r[f"{st}_{k}"]) for st in ("all", "nk") for k in ("MF", "BP", "CC", "mean")) + " |")
    L += ["", "## Single seeds (no routing; descriptive)", "", "| run | period | all mean | nk mean |", "|---|---|---|---|"]
    for r in PS.to_dict("records"):
        L.append(f"| {r['run']} | {r['period']} | {f(r['all'])} | {f(r['nk'])} |")
    (base / "report.md").write_text("\n".join(L) + "\n")


def write_md(base, S, Bn, T, res, smoke):
    def cell(v, c):
        if c == "seed" or c.startswith("n"):
            return "" if pd.isna(v) else str(int(v))
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)

    def tab(df, cols):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df[cols].to_dict("records"):
            out.append("| " + " | ".join(cell(r[c], c) for c in cols) + " |")
        return "\n".join(out)

    L = [f"# Forecasting model (idea 1): arms F1 'forecast' and F0 'full-label'{' (SMOKE, not a decision)' if smoke else ''}",
         "", "Design and keep rule: runs/test_looks.md, entry 2026-10-19 19:15:00. Code: `src/cafa6/forecast_train.py` "
         "(choices that the registration leaves open are in its docstring). Log: `logs/forecast_train.log`.", "",
         "## Keep rule (A valid, rows with an empty aspect at input, mean IA-weighted Fmax over MF/BP/CC)", "",
         f"- seeds evaluated: {res['seeds']}",
         f"- F1 seed mean {res['F1_mean']:.4f}, F0 seed mean {res['F0_mean']:.4f}, kNN prior alone {res['kNN']:.4f}",
         f"- (i) F1 - kNN = {res['F1-kNN']:+.4f} (needs >= +0.010, every seed > 0): {'pass' if res['cond_i'] else 'fail'}",
         f"- (ii) F1 - F0 = {res['F1-F0']:+.4f} (needs >= +0.005, every seed > 0): {'pass' if res['cond_ii'] else 'fail'}",
         f"- outcome: **{'F1 KEPT' if res['kept'] else 'F1 NOT KEPT'}**", "", "Per seed (A valid nk; B valid nk secondary):", "",
         tab(T, ["seed", "F1", "F0", "kNN", "F1-kNN", "F1-F0", "B_F1", "B_F0", "B_kNN"]) if len(T) else "(no seeds)", "",
         "Paired protein bootstrap of the seed-averaged (logit-mean) ensembles on nk rows (descriptive, not part of the rule):", ""]
    for k, v in res["bootstrap_ens_nk"].items():
        L.append(f"- {k}: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] ({v['reps']} reps)")
    L += ["", "## Summary (mean = mean over namespaces; n = scored rows per namespace)", ""]
    for per in ("A", "B"):
        for st in S[S.period == per].rows.unique():
            d = S[(S.period == per) & (S.rows == st)]
            L += [f"### {per} valid, rows {st}", "", tab(d, ["method", "MF", "BP", "CC", "mean", "n_MF", "n_BP", "n_CC"]), ""]
    L += ["## Per identity bin (ensembles and kNN; seed rows in per_bin.tsv)", ""]
    for per in ("A", "B"):
        for st in ("nk", "all"):
            d = Bn[(Bn.period == per) & (Bn.rows == st) & Bn.method.isin(["kNN", "F1_ens", "F0_ens"])]
            if len(d):
                pv = d.pivot(index="bin", columns="method", values="mean").reindex(BINS)
                nn_ = d[d.method == "kNN"].set_index("bin").n.reindex(BINS)
                pv["n_rows"] = nn_
                pv = pv.reset_index()
                L += [f"### {per} valid, rows {st}", "", tab(pv, list(pv.columns)), ""]
    (base / "report.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- verify
def verify():
    h = pd.read_parquet(FC / "work" / "hits.parquet", columns=["q", "s", "pident", "qcov"])
    vocab = pd.read_csv(GOGPT / "vocab.tsv", sep="\t")
    A = Period("A", vocab)
    self_hits = int((h.q == h.s).sum())
    hA = h[h.q.isin(set(A.ids))]
    ident = hA[(hA.pident >= 100) & (hA.qcov >= 95)].q.nunique()
    gg = pd.read_parquet(GOGPT / "train.parquet", columns=["id", "split"])
    in_ref = np.isin(A.ids, gg.id[gg.split != "test"].values)
    plog(f"VERIFY leave-self-out: hits.parquet {len(h):,} rows, self hits (q == s) {self_hits}; A proteins that are "
         f"references themselves {int(in_ref.sum()):,} of {len(A.ids):,}; A proteins with a 100%-identity, >=95% "
         f"coverage hit to another accession {ident:,} (kept: different accession, as in every kNN prior here)")
    z = np.load(FC / "A_knn_prior.npz")
    mx = hA.groupby("q").pident.max().reindex(A.ids).fillna(0).values
    plog(f"VERIFY A prior max_pident equals max hit pident over non-self hits: "
         f"{bool(np.allclose(mx, A.max_pident, atol=0.05))} (max abs diff {np.abs(mx - A.max_pident).max():.3f})")
    tA, = taxon_indices([A])
    plog(f"VERIFY taxon map: A proteins with a known HiGO taxon index {float((tA > 0).mean()):.3f}")
    return self_hits == 0


# ----------------------------------------------------------------------------- fresh24 predictions (look #2)
F24 = ROOT / "data" / "fresh24"
F24_PREDS = ROOT / "runs" / "fresh24" / "preds"
F24_WORK = OUT / "fresh24_work"
SERVE = ROOT / "data" / "serve"
LOOK1_HITS = Path("/dev/shm/fresh24_work/hits_trainval.tsv")
SPOT_TOL = 1e-3


def read_fasta(path: Path):
    ids, seqs, cur = [], [], []
    for line in open(path):
        if line.startswith(">"):
            if ids:
                seqs.append("".join(cur))
            ids.append(line[1:].split()[0]); cur = []
        else:
            cur.append(line.strip())
    if ids:
        seqs.append("".join(cur))
    return np.array(ids), seqs


def served_labels(onto: Ontology):
    z = np.load(SERVE / "labels.npz", allow_pickle=False)
    assert list(z["terms"]) == list(onto.terms), "data/serve/labels.npz columns differ from the ontology"
    L = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return z["ids"].astype(str), L


def check_served_labels(onto: Ontology, ids, L) -> dict:
    """The served label matrix vs S0 as build_forecast.cmd_prior builds it for period A (same references)."""
    from cafa6.build_forecast import ROOTS
    from cafa6.temporal_val import GO, SRC, full_labels
    go = GO()
    src = pd.read_parquet(SRC / "train.parquet", columns=["id", "split"])
    db_ids = np.array(sorted(set(src.id[src.split.isin(["train", "val"])])))
    S0 = {**full_labels(go, "train"), **full_labels(go, "val")}
    ref = np.array([p for p in db_ids if S0.get(p, set()) - ROOTS])
    L0 = onto.label_matrix([S0[p] for p in ref]).tocsr()
    pos = {p: i for i, p in enumerate(ids)}
    same_set = set(ref) == set(ids)
    Ls = L[[pos[p] for p in ref]] if same_set else None
    diff = int(abs(Ls - L0).sum()) if same_set else -1
    return {"served_refs": int(len(ids)), "S0_refs_with_nonroot_label": int(len(ref)), "same_reference_set": same_set,
            "label_pairs_S0": int(L0.nnz), "label_pairs_served": int(L.nnz), "differing_label_pairs": diff}


def diamond_k60(q_fa: Path, out: Path, threads: int):
    import subprocess
    dbsize = json.loads((SERVE / "meta.json").read_text())["dbsize"]
    subprocess.run([str(ROOT / "tools" / "diamond"), "blastp", "-d", str(SERVE / "refs"), "-q", str(q_fa),
                    "-o", str(out), "--more-sensitive", "-e", "1e-3", "-k", "60", "-p", str(threads), "--quiet",
                    "--tmpdir", str(out.parent), "--dbsize", str(dbsize),
                    "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)
    return dbsize


def read_hits(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path, sep="\t", header=None, names=["q", "s", "pident", "bits", "evalue", "qcov"],
                    dtype={"q": str, "s": str})
    return h[h.q != h.s].sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])


def knn_prior(onto, qids, hits, ref_ids, L):
    """build_forecast.cmd_prior's prior (make_knn + KNN.prior T5_k50_g0_na) over arbitrary queries."""
    from cafa6.build_forecast import KNN, make_knn
    kn = make_knn(onto, qids, hits, ref_ids, L)
    P, nh = kn.prior(KNN["T"], KNN["k"], KNN["gamma"], KNN["norm"])
    return P.tocsr(), kn.maxpid, kn.has_hit, nh


def full_scores(onto, vcols, Pf, f_terms, pv) -> sp.csr_matrix:
    """RowScorer.full for arbitrary proteins: vocab scores pv plus the out-of-vocabulary prior terms of Pf."""
    fcols = np.array([onto.idx.get(t, -1) for t in f_terms])
    oov = (fcols >= 0) & ~np.isin(fcols, vcols)
    f_sel, f_cols = np.where(oov)[0], fcols[oov]
    o = Pf[:, f_sel].tocoo()
    pv = np.asarray(pv, np.float32)
    rr, cc = np.nonzero(pv >= 0.005)
    M = sp.csr_matrix((np.concatenate([o.data, pv[rr, cc]]).astype(np.float64),
                       (np.concatenate([o.row, rr]), np.concatenate([f_cols[o.col], vcols[cc]]))),
                      shape=(Pf.shape[0], len(onto.terms)))
    M.sum_duplicates()
    return M


def write_cp_npz(path: Path, P: sp.csr_matrix, ids, onto, meta: dict):
    """compare_paper.save_pred layout, max-propagated, scores >= 0.01."""
    P = onto.propagate_max(P.tocsr()).tocsr().astype(np.float32)
    P.data[P.data < 0.01] = 0
    P.eliminate_zeros()
    used = np.unique(P.indices)
    Q = P[:, used].tocsr()
    np.savez_compressed(path, ids=np.asarray(ids, dtype=str), data=Q.data, indices=Q.indices, indptr=Q.indptr,
                        shape=np.array(Q.shape), terms=np.array(onto.terms)[used].astype(str), meta=json.dumps(meta))
    return Q.nnz


def load_forecast_model(arm: str, seed: int, vocab, device: str, plm: str | None = None):
    cfg = yaml.safe_load(open(CFG))
    cfg.update(fast_entmax=device.startswith("cuda"), seed=seed)
    if plm:
        cfg.update(plm=plm, plm_bf16=device.startswith("cuda"))
    anc = np.load(GOGPT / "go_ancestors.npy")
    higo = build_higo(cfg, vocab, anc, len(json.load(open(INIT / "taxa.json"))) + 1)
    model = ForecastHiGO(higo, vocab.aspect.values)
    w = torch.load(OUT / run_name(arm, seed, plm) / "trainable_weights.pt", map_location="cpu")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(w) == trainable, f"weights/model mismatch: {sorted(set(w) ^ trainable)[:5]}"
    model.load_state_dict({**model.state_dict(), **w})
    return model.to(device).eval(), cfg


def predict_seed(model, cfg, tok, seqs, taxa, P_vocab, has_hit, freq, aspect, device) -> np.ndarray:
    prior = PriorLogits(P_vocab, has_hit, aspect, freq)
    b = Batcher(seqs, tok, cfg["max_len"], cfg["tokens_per_batch"], cfg["max_batch"], knn=prior, seed=0)
    return predict(model, b, seqs, taxa, np.arange(len(seqs)), device, bf16=device.startswith("cuda"))


def fresh24_taxa(ids, periods):
    """Training rule of taxon_indices for proteins outside GO-GPT, then the served NCBI -> organism map."""
    gg = pd.read_parquet(GOGPT / "train.parquet", columns=["id", "taxon"])
    code = dict(zip(gg.id, gg.taxon))
    tmap = {int(k): v for k, v in json.load(open(INIT / "taxa.json")).items()}
    pairs = pd.DataFrame([(t, code[p]) for per in periods for p, t in zip(per.ids, per.ncbi) if p in code],
                         columns=["ncbi", "code"])
    ncbi2code = pairs.groupby("ncbi").code.agg(lambda s: s.value_counts().index[0]).to_dict()
    st = pd.read_csv(SERVE / "taxa.tsv", sep="\t").dropna(subset=["ncbi", "gogpt_taxon"])
    serve = dict(zip(st.ncbi.astype(int), st.gogpt_taxon.astype(int)))
    pt = pd.read_csv(F24 / "proteins.tsv", sep="\t").set_index("accession").taxon
    ncbi = pt.reindex(ids).values
    out, src = [], {"training_rule": 0, "served_map": 0, "other": 0}
    disagree = 0
    for t in ncbi:
        t = int(t)
        c1, c2 = ncbi2code.get(t), serve.get(t)
        if c1 is not None and c2 is not None and tmap.get(int(c1), 0) != tmap.get(int(c2), 0):
            disagree += 1
        if c1 is not None:
            src["training_rule"] += 1; c = c1
        elif c2 is not None:
            src["served_map"] += 1; c = c2
        else:
            src["other"] += 1; c = -1
        out.append(tmap.get(int(c), 0))
    out = np.array(out, np.int64)
    src.update(known_taxon_index=int((out > 0).sum()), rule_vs_served_disagree=disagree)
    return out, src


def predict_fresh24(device: str, threads: int, plm: str | None = None, theta: float | None = None,
                    out_name: str | None = None, cand: dict | None = None):
    """Default: look #2 files (forecast_F1_s{0,1,2}.npz, forecast_F1_ens.npz). With out_name (round 2): only the
    seed ensemble of the chosen family (plm), proteins with best hit identity >= theta routed to the kNN prior,
    written to out_name.npz + out_name.json; the spot-check goes to runs/forecast/round2."""
    t_all = time.time()
    F24_WORK.mkdir(parents=True, exist_ok=True)
    fam = run_name("F1", "{s}", plm).replace("_s{s}", "")
    plog(f"PREDICT fresh24 {fam}_ens{f' routed at identity >= {theta}' if theta else ''} START on {device} "
         f"({'round 2 -> ' + out_name if out_name else 'look #2'}; predictions only, no scoring)")
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(min(1.0, 20 * 2**30 / torch.cuda.get_device_properties(device).total_memory), device)
    vocab = pd.read_csv(GOGPT / "vocab.tsv", sep="\t")
    aspect = vocab.aspect.values
    onto = Ontology()
    vcols = np.array([onto.idx[t] for t in vocab.term])
    tok = AutoTokenizer.from_pretrained(plm or yaml.safe_load(open(CFG))["plm"])

    # 1. period A: F1 term frequencies (term-prior logit) and the spot-check inputs, exactly as in train()
    t = time.time()
    A, B = Period("A", vocab), Period("B", vocab)
    tA, _ = taxon_indices([A, B])
    Y = (A.Y_in + A.Y_gain).tocsr(); Y.data[:] = 1
    tr = np.where(A.split == "train")[0]
    freq = np.asarray(Y[tr].mean(0)).ravel()
    va = np.where(A.split == "valid")[0]
    rng = np.random.default_rng(0)
    long_ = va[np.array([len(A.seqs[j]) > 1022 for j in va])]
    pick = np.r_[rng.choice(long_, min(3, len(long_)), replace=False),
                 rng.choice(np.setdiff1d(va, long_), 20 - min(3, len(long_)), replace=False)]
    plog(f"PREDICT period A loaded ({len(tr):,} train proteins for the F1 term prior; spot-check on 20 A valid "
         f"proteins, {min(3, len(long_))} longer than 1,022) in {time.time() - t:.0f}s")

    # 2. spot-check: the predict path on 20 A valid proteins vs the saved training-time predictions
    t = time.time()
    models, spot = {}, {}
    va_ids = list(np.loadtxt(OUT / run_name("F1", 0, plm) / "ids_A_valid.txt", dtype=str))
    vpos = {p: i for i, p in enumerate(va_ids)}
    for s in (0, 1, 2):
        models[s] = load_forecast_model("F1", s, vocab, device, plm)
        ref = np.load(OUT / run_name("F1", s, plm) / "pred_A_valid.npy", mmap_mode="r")
        pr = predict_seed(*models[s], tok, [A.seqs[j] for j in pick], tA[pick], A.P[pick], A.has_hit[pick], freq,
                          aspect, device)
        r = np.asarray(ref[[vpos[A.ids[j]] for j in pick]], np.float32)
        d = np.abs(pr.astype(np.float32) - r)
        spot[run_name("F1", s, plm)] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                            "pairs_over_tol": int((d > SPOT_TOL).sum())}
    ok = all(v["max_abs_diff"] <= SPOT_TOL for v in spot.values())
    spot_path = OUT / "round2" / "spotcheck_fresh24.json" if out_name else F24_WORK / "spotcheck.json"
    spot_path.write_text(json.dumps({"proteins": [str(A.ids[j]) for j in pick], "tol": SPOT_TOL,
                                                         "result": spot, "pass": ok}, indent=2))
    plog(f"PREDICT spot-check (20 A valid proteins x 7,901 terms, tol {SPOT_TOL}): "
         + "; ".join(f"{k} max |diff| {v['max_abs_diff']:.2e}" for k, v in spot.items())
         + f" -> {'PASS' if ok else 'FAIL'} ({time.time() - t:.0f}s)")
    if not ok:
        raise SystemExit("spot-check failed; nothing written")

    # 3. fresh24 prior at the training-time definition, checked against the look-1 prior
    t = time.time()
    ids, seqs = read_fasta(F24 / "fresh24.fasta")
    q_fa = F24_WORK / "fresh24.fa"
    with open(q_fa, "w") as f:
        for p, s in zip(ids, seqs):
            f.write(f">{p}\n{s}\n")
    ref_ids, L = served_labels(onto)
    lab_check = check_served_labels(onto, ref_ids, L)
    plog(f"PREDICT served labels vs period-A S0 labels: {json.dumps(lab_check)} ({time.time() - t:.0f}s)")
    assert lab_check["same_reference_set"] and lab_check["differing_label_pairs"] == 0, "served labels differ from S0"
    hits_path = F24_WORK / "hits_k60.tsv"
    t = time.time()
    if not hits_path.exists():
        dbsize = diamond_k60(q_fa, hits_path, threads)
        plog(f"PREDICT DIAMOND fresh24 ({len(ids):,}) vs served references, --more-sensitive -e 1e-3 -k 60 "
             f"--dbsize {dbsize:,}, {threads} threads: {time.time() - t:.0f}s")
    else:
        plog(f"PREDICT reusing the -k 60 DIAMOND hits of look #2 ({hits_path.relative_to(ROOT)})")
    h60 = read_hits(hits_path)
    P60, maxpid, has_hit, nh = knn_prior(onto, ids, h60, ref_ids, L)
    chk = {"queries": int(len(ids)), "with_hit_k60": int(has_hit.sum()), "hits_k60_rows": int(len(h60))}
    z = np.load(F24_PREDS / "ontologue_knn.npz", allow_pickle=False)
    assert (z["ids"].astype(str) == ids).all(), "ontologue_knn.npz id order differs from fresh24.fasta"
    Q = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).tocoo()
    qc = np.array([onto.idx[t] for t in z["terms"]])
    P1 = sp.csr_matrix((Q.data, (Q.row, qc[Q.col])), shape=P60.shape)
    if LOOK1_HITS.exists():
        P51, *_ = knn_prior(onto, ids, read_hits(LOOK1_HITS), ref_ids, L)
        d = abs(P51 - P1)
        chk["look1_hits_recomputed_vs_ontologue_knn_max_abs_diff"] = float(d.max()) if d.nnz else 0.0
    d = abs(P60 - P1).tocsr()
    rows_diff = np.unique(d.tocoo().row[d.tocoo().data > 1e-6])
    chk.update(k60_vs_ontologue_knn_max_abs_diff=float(d.max()) if d.nnz else 0.0,
               k60_vs_ontologue_knn_proteins_differing=int(len(rows_diff)),
               k60_vs_ontologue_knn_pairs_differing_gt_0_01=int((d.data > 0.01).sum()),
               k60_nnz=int(P60.nnz), ontologue_knn_nnz=int(P1.nnz))
    if not out_name:
        (F24_WORK / "prior_check.json").write_text(json.dumps(chk, indent=2))
    plog(f"PREDICT fresh24 prior T5_k50_g0_na (training-time -k 60) vs look-1 ontologue_knn.npz (-k 51): "
         f"{json.dumps(chk)} ({time.time() - t:.0f}s)")
    used = np.unique(P60.indices)
    Pf, f_terms = P60[:, used].tocsr().astype(np.float32), np.array(onto.terms)[used]
    Pv = P60[:, vcols].tocsr().astype(np.float32)
    prior_path = F24_WORK / ("fresh24_knn_prior.npz" if not out_name else "fresh24_knn_prior_round2.npz")
    np.savez_compressed(prior_path, ids=ids, p_data=Pv.data, p_indices=Pv.indices,
                        p_indptr=Pv.indptr, p_shape=np.array(Pv.shape), pf_data=Pf.data, pf_indices=Pf.indices,
                        pf_indptr=Pf.indptr, pf_shape=np.array(Pf.shape), terms_full=f_terms.astype(str),
                        max_pident=maxpid.astype(np.float32), has_hit=has_hit, n_hits=np.asarray(nh, np.int32),
                        settings=json.dumps({"T": 5.0, "k": 50, "gamma": 0, "norm": "aspect",
                                             "setting": "T5_k50_g0_na", "diamond": "--more-sensitive -e 1e-3 -k 60",
                                             "references": "data/serve (GO-GPT train+val, S0 labels)"}))

    # 4. per-seed predictions and the logit-mean ensemble
    taxa, tsrc = fresh24_taxa(ids, [A, B])
    plog(f"PREDICT fresh24 taxon indices: {json.dumps(tsrc)}")
    meta0 = {"look": "fresh24 look #2 (runs/fresh24_looks.md 2026-10-24 22:45)",
             "prior": "T5_k50_g0_na aspect-normalised, DIAMOND --more-sensitive -e 1e-3 -k 60 vs data/serve train+val "
                      "references (S0 labels); runs/forecast/fresh24_work/fresh24_knn_prior.npz",
             "inputs": "sequence, taxon, prior; no input terms", "device": device,
             "scores": "MCM probabilities over the 7,901-term vocabulary + out-of-vocabulary kNN prior terms, "
                       "max-propagated, >= 0.01", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if out_name:
        z2 = np.load(F24_WORK / "fresh24_knn_prior.npz")
        P2 = sp.csr_matrix((z2["pf_data"], z2["pf_indices"], z2["pf_indptr"]), shape=tuple(z2["pf_shape"]))
        same = list(z2["terms_full"]) == list(f_terms) and abs(P2 - Pf).sum() == 0 \
            and np.array_equal(z2["max_pident"], maxpid.astype(np.float32))
        plog(f"PREDICT round-2 prior identical to the look-2 prior file (fresh24_knn_prior.npz): {same}")
        ps, timing = [], {}
        for s in (0, 1, 2):
            t = time.time()
            ps.append(predict_seed(*models[s], tok, seqs, taxa, Pv, has_hit, freq, aspect, device))
            timing[run_name("F1", s, plm)] = time.time() - t
            plog(f"PREDICT {run_name('F1', s, plm)}: {len(ids):,} proteins in {timing[run_name('F1', s, plm)]:.0f}s; "
                 f"peak GPU {torch.cuda.max_memory_allocated(device) / 2**30 if device.startswith('cuda') else 0:.1f} GiB")
        t = time.time()
        M = full_scores(onto, vcols, Pf, f_terms, logit_mean(ps)).tocsr()
        route = (maxpid >= theta) if theta is not None else np.zeros(len(ids), bool)
        if route.any():
            keep = sp.diags((~route).astype(np.float64))
            M = (keep @ M + sp.diags(route.astype(np.float64)) @ P60.astype(np.float64)).tocsr()
        desc = {**meta0, "look": "fresh24 look #3 candidate (runs/test_looks.md 2026-10-25 01:15:00, round 2)",
                "method": f"forecast model {fam}_ens (logit mean of seeds 0, 1, 2)"
                          + (f", proteins with best hit identity >= {theta} routed to the kNN prior" if theta else ""),
                "weights": f"runs/forecast/{fam}_s{{0,1,2}}", "theta": theta, "routed_proteins": int(route.sum()),
                "routing_identity": "max pident over the -k 60 DIAMOND hits vs data/serve train+val references",
                "candidate": cand or {}, "taxon": tsrc, "timing_s": timing}
        nnz = write_cp_npz(F24_PREDS / f"{out_name}.npz", M, ids, onto, desc)
        (F24_PREDS / f"{out_name}.json").write_text(json.dumps({**desc, "file": f"runs/fresh24/preds/{out_name}.npz",
                                                                "scores": nnz}, indent=2))
        plog(f"PREDICT round 2: wrote runs/fresh24/preds/{out_name}.npz ({nnz:,} scores, {nnz / len(ids):.0f} per "
             f"protein; {int(route.sum()):,} proteins routed to kNN) and {out_name}.json in {time.time() - t:.0f}s; "
             f"total wall {fmt_dur(time.time() - t_all)}")
        return
    ps, timing = [], {}
    for s in (0, 1, 2):
        t = time.time()
        p = predict_seed(*models[s], tok, seqs, taxa, Pv, has_hit, freq, aspect, device)
        timing[f"F1_s{s}"] = time.time() - t
        ps.append(p)
        nnz = write_cp_npz(F24_PREDS / f"forecast_F1_s{s}.npz", full_scores(onto, vcols, Pf, f_terms, p), ids, onto,
                           {**meta0, "method": f"forecast model F1 seed {s}", "weights": f"runs/forecast/F1_s{s}"})
        plog(f"PREDICT F1_s{s}: {len(ids):,} proteins in {timing[f'F1_s{s}']:.0f}s; wrote "
             f"runs/fresh24/preds/forecast_F1_s{s}.npz ({nnz:,} scores, {nnz / len(ids):.0f} per protein); "
             f"peak GPU {torch.cuda.max_memory_allocated(device) / 2**30 if device.startswith('cuda') else 0:.1f} GiB")
    t = time.time()
    ens = logit_mean(ps)
    nnz = write_cp_npz(F24_PREDS / "forecast_F1_ens.npz", full_scores(onto, vcols, Pf, f_terms, ens), ids, onto,
                       {**meta0, "method": "forecast model F1_ens (logit mean of seeds 0, 1, 2)",
                        "weights": "runs/forecast/F1_s{0,1,2}"})
    plog(f"PREDICT F1_ens: wrote runs/fresh24/preds/forecast_F1_ens.npz ({nnz:,} scores, {nnz / len(ids):.0f} per "
         f"protein) in {time.time() - t:.0f}s; total wall {fmt_dur(time.time() - t_all)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "report", "verify", "predict", "round2"])
    ap.add_argument("--round2", action="store_true", help="predict: the candidate in runs/forecast/round2/choice.json")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--arm", default="F1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--plm", default=None, help="round 2: other frozen encoder, e.g. facebook/esm2_t33_650M_UR50D")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.arm, a.seed, a.device, a.smoke, a.epochs, a.plm)
    elif a.cmd == "report":
        report(bool(a.smoke), reps=a.reps)
    elif a.cmd == "predict" and a.round2:
        c = json.loads((OUT / "round2" / "choice.json").read_text())["chosen"]
        predict_fresh24(a.device, a.threads, R2_FAMILIES[c["family"]][0], c["theta"], "forecast_round2", c)
    elif a.cmd == "predict":
        predict_fresh24(a.device, a.threads)
    elif a.cmd == "round2":
        round2()
    else:
        verify()


if __name__ == "__main__":
    main()
