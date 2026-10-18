"""HiGO-R2 GO-term prediction for new protein sequences: the look-3 pipeline (runs/test_looks.md, test look #3;
research_notes/HiGO-R loop/HiGO-R2_model.md section 0), on CPU.

Pipeline per protein
  1. DIAMOND blastp against the GO-GPT train + val reference proteins (data/serve/refs.dmnd, 133,054 sequences) with
     the settings of the benchmark hits: --more-sensitive, e <= 1e-3, -k 51, --dbsize = letters of the 119,749
     training proteins (as knn_trainval.run_diamond, so e-values and the 1e-3 cut match), best HSP per pair, top 50
     subjects by bitscore.
  2. Sharpened kNN prior T5_k50_g0_na with aspect normalisation (knn_sharp.KNN.prior; runs/knn_sharp/chosen_v2.json):
     w_j = bits_j * exp((pident_j - max pident) / 5) over the 50 hits, labels = each reference's full propagated GO
     set, each aspect divided by the weight of the hits annotated in that aspect.
  3. HiGO network runs/l3_c0_tv2 (frozen facebook/esm2_t12_35M_UR50D, entmax per-term pooling, hierarchical queries,
     MCM, taxon embedding over 67 GO-GPT organisms + "other"); sequences longer than 1,022 residues are cut into
     half-overlapping windows whose logits are max-pooled (train.predict). Only the top 300 HiGO terms enter step 4.
  4. Twin stacker (stacker_tv model b, refit on all of TV v2 for 800 steps; runs/stacker_tv/l3_c0/model_twin.pt) over
     the union of the HiGO top 300 and every kNN term. Protein features: max identity, has hit, number of hits,
     organism group (human / mouse / rat / zebrafish / other, from the taxon) and whether the top-bitscore hit among
     the training references comes from the same organism (as data/proc_gogpt_tv2/cells.parquet defines it).
  5. Scores >= 0.01, max-propagated over the GO graph (knn_sharp.Ontology.propagate_max).

Inputs
  --fasta FILE   protein FASTA. The organism of a record is read from 'OX=<NCBI taxon id>' in its header (UniProt
                 style), else from --taxon. Residues: the 20 standard amino acids plus X, B, Z, U, O; '*' and
                 whitespace are dropped.
  --taxon N      NCBI taxon id for records without OX= (9606 human, 10090 mouse, 10116 rat, 7955 zebrafish, ...).
                 A taxon outside the GO-GPT data (or none) gets the generic taxon embedding, organism group "other"
                 and no same-organism hit.
  data/serve/    reference data, built once by --build (about 2 min, ~160 MB): refs.fa, refs.dmnd, refs.parquet
                 (id, split, organism), labels.npz (reference x GO label matrix), taxa.tsv (organism <-> NCBI taxon).

Outputs (--out DIR)
  predictions.tsv   protein, go_term, score, aspect, term_name: every propagated score >= 0.01
  predictions.json  per protein: length, taxon, organism, max identity and identity bin, number of hits, the 5
                    nearest references, the terms above the per-aspect thresholds and all scores
  report.md         per protein: terms above MF 0.28 / BP 0.42 / CC 0.45 grouped by aspect with names, and the 5
                    nearest reference hits with identity

Usage
  python -m cafa6.predict_r2 --build                                  build data/serve (once)
  python -m cafa6.predict_r2 --fasta in.fa --taxon 9606 --out out_dir
  python -m cafa6.predict_r2 --self-test                              20 GO-GPT test proteins end to end vs
                                                                      runs/stacker_tv/l3_c0/pred_test_twin.npz
  python -m cafa6.predict_r2 --verify-stacker                         saved stacker on the stored test features vs
                                                                      pred_test_twin.npz (all 8,630 test proteins)
The web app (cafa6.webapp) imports Predictor and keeps one instance loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from cafa6.data import ROOT  # noqa: E402

SERVE = ROOT / "data" / "serve"
SRC = ROOT / "data" / "proc_gogpt"
TV2 = ROOT / "data" / "proc_gogpt_tv2"
HIGO_RUN = ROOT / "runs" / "l3_c0_tv2"
STACKER_DIR = ROOT / "runs" / "stacker_tv" / "l3_c0"
STACKER = STACKER_DIR / "model_twin.pt"
DIAMOND = ROOT / "tools" / "diamond"
THRESH = {"MF": 0.28, "BP": 0.42, "CC": 0.45}
ASPECTS = ("MF", "BP", "CC")
AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
KNN_K, KNN_T = 50, 5.0
MIN_SCORE = 0.01
TOL = 1e-3
T_START = time.time()


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T_START:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ input

def clean_seq(s: str) -> str:
    return re.sub(r"[\s*]", "", s).upper()


def parse_fasta(text: str, default_taxon: int | None = None) -> list[dict]:
    """[{id, seq, taxon, header}] from FASTA text; a bare sequence without '>' becomes one record 'query'."""
    recs, cur = [], None
    lines = text.splitlines()
    if not any(l.startswith(">") for l in lines):
        s = clean_seq(text)
        return [{"id": "query", "seq": s, "taxon": default_taxon, "header": ""}] if s else []
    for line in lines:
        if line.startswith(">"):
            hdr = line[1:].strip()
            tok = hdr.split()[0] if hdr else f"seq{len(recs) + 1}"
            m = re.match(r"^(?:sp|tr)\|([^|]+)\|", tok)
            ox = re.search(r"\bOX=(\d+)", hdr)
            cur = {"id": m.group(1) if m else tok, "seq": "", "header": hdr,
                   "taxon": int(ox.group(1)) if ox else default_taxon}
            recs.append(cur)
        elif cur is not None:
            cur["seq"] += clean_seq(line)
    return recs


def validate(recs: list[dict], max_seqs: int | None = None, max_len: int | None = None) -> list[str]:
    """Problems with the input (empty list = valid)."""
    err = []
    if not recs:
        err.append("no sequence found")
    if max_seqs is not None and len(recs) > max_seqs:
        err.append(f"{len(recs)} sequences; at most {max_seqs} per request")
    seen = set()
    for r in recs:
        if not r["seq"]:
            err.append(f"{r['id']}: empty sequence")
            continue
        bad = sorted(set(r["seq"]) - AA)
        if bad:
            err.append(f"{r['id']}: characters outside the amino-acid alphabet: {''.join(bad)}")
        if max_len is not None and len(r["seq"]) > max_len:
            err.append(f"{r['id']}: {len(r['seq'])} residues; at most {max_len} per sequence")
        if r["id"] in seen:
            err.append(f"{r['id']}: duplicate record id")
        seen.add(r["id"])
    return err


# ------------------------------------------------------------------------------------------------ build

def build_serve(threads: int = 16):
    """data/serve: train + val reference FASTA and DIAMOND db, reference table, label matrix, taxon map."""
    from cafa6 import knn_sharp as ks
    from cafa6.temporal_val import GO, full_labels
    SERVE.mkdir(parents=True, exist_ok=True)
    t = time.time()
    df = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split", "organism", "taxon"])
    tr, va = df[df.split == "train"], df[df.split == "val"]
    refs = pd.concat([tr, va], ignore_index=True)
    with open(SERVE / "refs.fa", "w") as f:
        for pid, s in zip(refs.id, refs.seq):
            f.write(f">{pid}\n{s}\n")
    subprocess.run([str(DIAMOND), "makedb", "--in", str(SERVE / "refs.fa"), "-d", str(SERVE / "refs"), "--quiet",
                    "-p", str(threads)], check=True)
    refs[["id", "split", "organism", "taxon"]].assign(length=refs.seq.str.len()).to_parquet(
        SERVE / "refs.parquet", index=False)
    log(f"refs.fa / refs.dmnd: {len(tr):,} train + {len(va):,} val references ({time.time() - t:.0f}s)")
    # labels as knn_trainval.priors builds them for test queries: full propagated GO-GPT labels of train and val
    onto, go = ks.Ontology(), GO()
    tfull = full_labels(go, "train")
    vfull = {p: v for p, v in full_labels(go, "val").items() if p in set(va.id)}
    L = onto.label_matrix([tfull.get(p, set()) for p in tr.id] + [vfull.get(p, set()) for p in va.id]).tocsr()
    np.savez_compressed(SERVE / "labels.npz", data=L.data.astype(np.float32), indices=L.indices, indptr=L.indptr,
                        shape=np.array(L.shape), terms=np.array(onto.terms).astype(str), ids=np.array(refs.id.tolist(), dtype=str))
    # organism name <-> NCBI taxon id through UniProt (data/grounding/uniprot_features.parquet)
    u = pd.read_parquet(ROOT / "data" / "grounding" / "uniprot_features.parquet", columns=["id", "taxon"])
    m = df.merge(u.rename(columns={"taxon": "ncbi"}), on="id", how="left").dropna(subset=["ncbi"])
    m["ncbi"] = m.ncbi.astype(int)
    tax = (m.groupby(["organism", "ncbi"]).size().rename("n").reset_index()
           .sort_values(["organism", "n"], ascending=[True, False]).drop_duplicates("organism"))
    gtx = df.drop_duplicates("organism").set_index("organism").taxon
    tax["gogpt_taxon"] = tax.organism.map(gtx).values
    tax.to_csv(SERVE / "taxa.tsv", sep="\t", index=False)
    meta = {"dbsize": int(tr.seq.str.len().sum()), "n_train": len(tr), "n_val": len(va),
            "diamond": "--more-sensitive -e 1e-3 -k 51 --dbsize <train letters>; top 50 by bitscore",
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (SERVE / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"labels.npz {L.shape} nnz {L.nnz:,}; taxa.tsv {len(tax):,} organisms; done in {time.time() - t:.0f}s")


# ------------------------------------------------------------------------------------------------ pipeline pieces

def knn_prior(hits: pd.DataFrame, qids: list, didx: dict, L: sp.csr_matrix, roots: list, term_ns: np.ndarray,
              L_roots: sp.csr_matrix | None = None):
    """knn_sharp.KNN + KNN.prior(T=5, k=50, gamma=0, norm='aspect') for one batch of queries.
    hits: [q, s, pident, bits, ...] already deduplicated per (q, s). L_roots: L[:, roots], if precomputed."""
    qidx = {p: i for i, p in enumerate(qids)}
    h = hits.assign(_q=hits.q.map(qidx), _s=hits.s.map(didx)).dropna(subset=["_q", "_s"])
    h = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
    q = h._q.values.astype(np.int64)
    s = h._s.values.astype(np.int64)
    rank = h.groupby("q", sort=False).cumcount().values
    bits, pid = h.bits.values, h.pident.values
    n = len(qids)
    maxpid = np.zeros(n, np.float32)
    np.maximum.at(maxpid, q, pid.astype(np.float32))
    has_hit = np.bincount(q, minlength=n) > 0
    m = rank < KNN_K
    q, s, b, pid = q[m], s[m], bits[m], pid[m]
    mx = np.zeros(n)
    np.maximum.at(mx, q, pid)
    w = b * np.exp((pid - mx[q]) / KNN_T)
    W = sp.csr_matrix((w, (q, s)), shape=(n, L.shape[0]))
    P = (W @ L).tocsr()
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    D = (W @ (L[:, roots] if L_roots is None else L_roots)).toarray()
    den = D[rows, term_ns[P.indices]]
    P.data = np.minimum(P.data / np.maximum(den, 1e-12), 1.0).astype(np.float32)
    return P.astype(np.float32), maxpid, has_hit, np.bincount(q, minlength=n).astype(np.int32)


def topk_csr(D: np.ndarray, top_k: int = 500) -> np.ndarray:
    """What data.save_pred(top_k) + data.load_pred keep: scores strictly above each row's k-th largest score."""
    s32 = D.astype(np.float32)
    t = s32.shape[1]
    kth = np.partition(s32, t - top_k, axis=1)[:, t - top_k][:, None]
    return np.where(s32 > kth, s32, 0.0).astype(np.float32)


def higo_matrix(D: np.ndarray, vcols: np.ndarray, nT: int, topk: int):
    """stacker_tv.higo_matrix on an in-memory [n, 7901] HiGO matrix (rows already aligned)."""
    n = D.shape[0]
    k = min(topk, D.shape[1])
    idx = np.argpartition(-D, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(D, idx, 1)
    keep = val > 0
    rows = np.repeat(np.arange(n), k).reshape(n, k)
    H = sp.csr_matrix((val[keep], (rows[keep], vcols[idx[keep]])), shape=(n, nT))
    floor = np.where(keep.any(1), np.where(keep, val, np.inf).min(1), 0.0).astype(np.float32)
    return H, floor


def stack_features(Pk: sp.csr_matrix, H: sp.csr_matrix, floor: np.ndarray, vcols: np.ndarray, max_pident,
                   has_hit, n_hits, org_group, same_org, onto) -> dict:
    """The feature rows of stacker_tv.build_split (with_gt=False) from in-memory inputs."""
    from cafa6 import knn_sharp as ks
    from cafa6.stacker_tv import ORGS, RBF_C, RBF_S, logit_np, lookup
    n, nT = Pk.shape
    Pk = Pk.tocsr().copy()
    Pk.sum_duplicates()
    pat = Pk.copy()
    pat.data[:] = 1
    hp = H.copy()
    hp.data[:] = 1
    pat = (pat + hp).tocsr()
    pat.sort_indices()
    pix = np.repeat(np.arange(n), np.diff(pat.indptr)).astype(np.int64)
    t = pat.indices.astype(np.int64)
    key = pix * nT + t
    pk = lookup(Pk, key, nT)
    mx = (np.asarray(max_pident, np.float32) / 100).astype(np.float32)
    ph = lookup(H, key, nT)
    in_top = (ph > 0).astype(np.float32)
    vm = np.zeros(nT, bool)
    vm[vcols] = True
    in_voc = vm[t].astype(np.float32)
    ph_eff = np.where(in_top > 0, ph, np.where(in_voc > 0, floor[pix], 0.0)).astype(np.float32)
    lh = logit_np(ph_eff)
    lk = logit_np(pk)
    asp = onto.ns[t].astype(np.int64)
    mr = mx[pix]
    X = np.stack([lk / 4, lh / 4, pk, in_top, in_voc, ph_eff, np.log1p(onto.ia[t]).astype(np.float32) / 3,
                  onto.depth[t] / 10, asp == 0, asp == 1, asp == 2, lk * mr / 4, lh * mr / 4, pk > 0, pk * ph_eff],
                 1).astype(np.float32)
    org = np.asarray(org_group, dtype=object)
    same = np.asarray(same_org, dtype=bool).astype(np.float32)
    hh = np.asarray(has_hit, bool)
    rbf = np.exp(-((mx[:, None] - RBF_C[None]) / RBF_S) ** 2) * hh[:, None]
    Q = np.concatenate([np.stack([mx, hh.astype(np.float32), same,
                                  np.log1p(np.asarray(n_hits)).astype(np.float32) / 4], 1),
                        rbf, np.stack([org == o for o in ORGS], 1)], 1).astype(np.float32)
    ibin = pd.Series(ks.ident_bins(np.asarray(max_pident))).map({b: i for i, b in enumerate(ks.BINS)}).values
    return {"ids": np.arange(n), "pix": pix, "t": t, "asp": asp, "X": X, "Q": Q, "pk": pk,
            "ph": (ph * in_top).astype(np.float32), "ptr": pat.indptr.astype(np.int64), "ibin": ibin.astype(np.int64)}


def load_stacker(path: Path = STACKER, device: str = "cpu"):
    from cafa6.stacker_tv import Twin
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["kind"] not in ("twin", "gated_twin"):
        raise ValueError(f"{path}: model kind {ck['kind']} is not a twin model")
    i = ck["init"]
    model = Twin(i["nT"], i["nf"], i["nq"], gated=ck["kind"] == "gated_twin")
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def apply_stacker(model, feats: dict, onto, device: str = "cpu") -> sp.csr_matrix:
    """Row probabilities of the stacker -> cut at 0.01 -> max-propagated CSR [n, n_terms] (stacker_tv.Ctx.matrix)."""
    from cafa6.stacker_tv import Data, predict
    n = int(len(feats["ptr"]) - 1)
    D = Data(feats, torch.device(device))
    p = predict(model, D, np.arange(n))
    p = np.asarray(p, np.float32)
    k = p >= MIN_SCORE
    P = sp.csr_matrix((p[k], (feats["pix"][k], feats["t"][k])), shape=(n, len(onto.terms)))
    return onto.propagate_max(P)


def read_pred_npz(path: Path, onto) -> tuple[np.ndarray, sp.csr_matrix]:
    """A stacker_tv.save_sparse file as (ids, CSR over ontology columns, float32 of the stored float16)."""
    z = np.load(path)
    M = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    cols = np.array([onto.idx[t] for t in z["terms"]])
    M = M.tocoo()
    return z["ids"].astype(str), sp.csr_matrix((M.data, (M.row, cols[M.col])), shape=(M.shape[0], len(onto.terms)))


def compare(P_new: sp.csr_matrix, P_ref: sp.csr_matrix) -> dict:
    """Max abs difference after float16 rounding of the new scores (absent = 0), plus the same with entries present
    in only one file counted as max(0, score - 0.01) (the other side is then below the 0.01 cut)."""
    def flat(M):
        M = sp.csr_matrix(M).tocoo()
        k = M.row.astype(np.int64) * M.shape[1] + M.col
        return pd.Series(M.data.astype(np.float32), index=k)

    a = flat(P_new).astype(np.float16).astype(np.float32)
    b = flat(P_ref)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    both = j.a.notna() & j.b.notna()
    d_both = float((j.a[both] - j.b[both]).abs().max()) if both.any() else 0.0
    only = j[~both].max(axis=1)
    raw = float(max(d_both, only.max() if len(only) else 0.0))
    d_only = float(np.maximum(only.values - MIN_SCORE, 0).max()) if len(only) else 0.0
    return {"max_abs_diff_absent_as_0": raw, "max_abs_diff_shared_entries": d_both,
            "n_entries_new": int(len(a)), "n_entries_ref": int(len(b)), "n_only_in_one": int(len(only)),
            "max_excess_over_cut_only_in_one": d_only, "max_abs_diff": max(d_both, d_only)}


# ------------------------------------------------------------------------------------------------ predictor

class Predictor:
    """Loads the reference data, HiGO network, stacker and ontology once; predict() runs the five stages."""

    def __init__(self, threads: int | None = None, diamond_threads: int | None = None, verbose: bool = True,
                 higo_device: str = "cpu"):
        import yaml
        from transformers import AutoTokenizer

        from cafa6 import knn_sharp as ks
        from cafa6.model import build_higo, load_trainable
        from cafa6.train import Batcher
        from cafa6.temporal_val import org_group
        self.verbose = verbose
        t0 = time.time()
        missing = [f for f in ("refs.dmnd", "refs.parquet", "labels.npz", "taxa.tsv", "meta.json")
                   if not (SERVE / f).exists()]
        if missing:
            raise FileNotFoundError(f"data/serve is incomplete ({missing}); run python -m cafa6.predict_r2 --build")
        if not STACKER.exists():
            raise FileNotFoundError(f"{STACKER} missing; refit with python -m cafa6.stacker_tv ... --save-model "
                                    f"runs/stacker_tv/l3_c0 (see docs/serving.md)")
        ncpu = os.cpu_count() or 4
        self.threads = threads or min(16, ncpu)
        self.diamond_threads = diamond_threads or min(32, ncpu)
        torch.set_num_threads(self.threads)
        self._org_group = org_group
        self.meta = json.loads((SERVE / "meta.json").read_text())

        self.onto = ks.Ontology()
        self.nT = len(self.onto.terms)
        self.roots = [self.onto.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
        self.names = self._go_names()
        z = np.load(SERVE / "labels.npz")
        if list(z["terms"]) != list(self.onto.terms):
            raise RuntimeError("data/serve/labels.npz was built for another ontology column order; rebuild it")
        self.L = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        self.L_roots = self.L[:, self.roots].tocsr()
        self.refs = pd.read_parquet(SERVE / "refs.parquet")
        self.db_ids = self.refs.id.values.astype(str)
        self.didx = {p: i for i, p in enumerate(self.db_ids)}
        self.ref_org = dict(zip(self.refs.id, self.refs.organism))
        self.ref_split = dict(zip(self.refs.id, self.refs.split))
        tax = pd.read_csv(SERVE / "taxa.tsv", sep="\t")
        self.org_taxid = dict(zip(tax.organism, tax.ncbi))
        tx = tax.sort_values("n", ascending=False).drop_duplicates("ncbi")
        self.taxid_org = dict(zip(tx.ncbi.astype(int), tx.organism))
        self.org_gogpt = dict(zip(tax.organism, tax.gogpt_taxon))
        allorg = pd.read_parquet(SRC / "train.parquet", columns=["organism", "taxon"]).drop_duplicates("organism")
        self.org_gogpt.update(dict(zip(allorg.organism, allorg.taxon)))
        log(f"references: {len(self.db_ids):,} ({self.meta['n_train']:,} train + {self.meta['n_val']:,} val), "
            f"label matrix nnz {self.L.nnz:,}, {len(self.taxid_org):,} NCBI taxa mapped ({time.time() - t0:.1f}s)")

        t1 = time.time()
        self.cfg = yaml.safe_load(open(HIGO_RUN / "config.yaml"))
        self.vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
        self.tmap = {int(k): v for k, v in json.load(open(HIGO_RUN / "taxa.json")).items()}
        self.higo = build_higo(self.cfg, self.vocab, np.load(SRC / "go_ancestors.npy"), len(self.tmap) + 1, None)
        load_trainable(self.higo, torch.load(HIGO_RUN / "trainable_weights.pt", map_location="cpu"))
        self.higo.eval()
        self.higo_device = "cpu"
        self.set_higo_device(higo_device)
        self.tok = AutoTokenizer.from_pretrained(self.cfg["plm"])
        self._Batcher = Batcher
        self.vcols = np.array([self.onto.idx[t] for t in self.vocab.term])
        log(f"HiGO {HIGO_RUN.name}: {self.cfg['plm']} ({self.cfg['adapter']}), {len(self.vocab):,} terms, "
            f"{len(self.tmap)} taxa + other, on {self.higo_device}, CPU threads {self.threads} "
            f"({time.time() - t1:.1f}s)")

        t2 = time.time()
        self.stacker, ck = load_stacker(STACKER)
        if ck["terms"] != list(self.onto.terms):
            raise RuntimeError(f"{STACKER}: term order differs from the ontology")
        self.topk = int(ck["higo_topk"])
        log(f"stacker {STACKER.relative_to(ROOT)}: {ck['kind']}, refit {ck['refit_steps']} steps, HiGO top "
            f"{self.topk} ({time.time() - t2:.1f}s); ready in {time.time() - t0:.1f}s")

    def _go_names(self) -> dict:
        from cafa6.knn_sharp import OBO
        names, cur = {}, None
        for line in open(OBO):
            if line.startswith("id: "):
                cur = line[4:].strip()
            elif line.startswith("name: ") and cur:
                names[cur] = line[6:].strip()
                cur = None
        return names

    # -------------------------------------------------------------------------------------------- stages

    def resolve_organism(self, rec: dict) -> dict:
        """NCBI taxon / organism name -> GO-GPT organism name, HiGO taxon index, organism group."""
        org = rec.get("organism")
        taxid = rec.get("taxon")
        if org is None and taxid is not None:
            org = self.taxid_org.get(int(taxid))
        if org is not None and taxid is None:
            taxid = self.org_taxid.get(org)
        g = self.org_gogpt.get(org) if org is not None else None
        tax_idx = self.tmap.get(int(g), 0) if g is not None else 0
        group = str(self._org_group(pd.Series([org]))[0])
        return {"taxon": None if taxid is None else int(taxid), "organism": org, "higo_taxon": int(tax_idx),
                "org_group": group}

    def diamond(self, recs: list[dict], exclude_self: bool = False) -> pd.DataFrame:
        with tempfile.TemporaryDirectory(dir=SERVE) as tmp:
            tmp = Path(tmp)
            with open(tmp / "q.fa", "w") as f:
                for i, r in enumerate(recs):
                    f.write(f">q{i}\n{r['seq']}\n")
            subprocess.run([str(DIAMOND), "blastp", "-d", str(SERVE / "refs"), "-q", str(tmp / "q.fa"),
                            "-o", str(tmp / "hits.tsv"), "--more-sensitive", "-e", "1e-3", "-k", str(KNN_K + 1),
                            "-p", str(self.diamond_threads), "--quiet", "--tmpdir", str(tmp),
                            "--dbsize", str(self.meta["dbsize"]),
                            "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"],
                           check=True)
            h = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None,
                            names=["q", "s", "pident", "bits", "evalue", "qcov"], dtype={"q": str, "s": str})
        h["q"] = h.q.str[1:].astype(int).map(lambda i: recs[i]["id"])
        if exclude_self:
            h = h[h.q != h.s]
        h = h.sort_values(["q", "bits"], ascending=[True, False], kind="stable").drop_duplicates(["q", "s"])
        return h.groupby("q", sort=False).head(KNN_K).reset_index(drop=True)

    def set_higo_device(self, device: str):
        """cpu (fp32, the serving default) or cuda[:N] (bf16 autocast, as the benchmark predictions were made)."""
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device available")
        self.higo.to(device)
        self.higo_device = device

    def higo_scores(self, recs: list[dict], taxa: list[int]) -> np.ndarray:
        """[n, 7901] float16 HiGO probabilities through train.predict: fp32 on CPU, bf16 autocast on CUDA."""
        from cafa6.train import predict
        seqs = [r["seq"] for r in recs]
        b = self._Batcher(seqs, self.tok, self.cfg["max_len"], self.cfg["tokens_per_batch"],
                          self.cfg["max_batch"])
        cuda = self.higo_device.startswith("cuda")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # train.predict's CUDA autocast context is a no-op on CPU
            return predict(self.higo, b, seqs, np.asarray(taxa), np.arange(len(seqs)), self.higo_device, bf16=cuda)

    def same_org_best_hit(self, hits: pd.DataFrame, qids: list, orgs: list) -> np.ndarray:
        """cells.parquet's best_same_org: the top-bitscore hit among the training references is from the query's
        organism (benchmark cells were built from the train-only hits)."""
        h = hits[hits.s.map(self.ref_split).eq("train")]
        top = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates("q").set_index("q").s
        out = []
        for q, o in zip(qids, orgs):
            s = top.get(q)
            out.append(bool(s is not None and o is not None and self.ref_org.get(s) == o))
        return np.array(out, bool)

    def predict(self, recs: list[dict], exclude_self: bool = False, hits: pd.DataFrame | None = None,
                higo: np.ndarray | None = None) -> list[dict]:
        """recs: [{id, seq, taxon (NCBI id or None) | organism (GO-GPT organism name)}]. exclude_self drops hits
        whose reference accession equals the query id (the benchmark convention; used by --self-test). hits / higo
        reuse DIAMOND hits or [n, 7901] HiGO scores computed elsewhere (self-test decomposition only)."""
        t0 = time.time()
        qids = [r["id"] for r in recs]
        res = [self.resolve_organism(r) for r in recs]
        n = len(recs)
        t = time.time()
        if hits is None:
            hits = self.diamond(recs, exclude_self)
        self.last_hits = hits
        td = time.time() - t
        self._log(f"[1/5] DIAMOND: {len(hits):,} hits for {hits.q.nunique()}/{n} queries ({td:.1f}s)")
        t = time.time()
        Pk, maxpid, has_hit, n_hits = knn_prior(hits, qids, self.didx, self.L, self.roots, self.onto.ns,
                                                self.L_roots)
        tk = time.time() - t
        self._log(f"[2/5] kNN prior T5_k50_g0_na: {Pk.nnz:,} term scores ({tk:.2f}s)")
        t = time.time()
        if higo is None:
            Dh = self.higo_scores(recs, [r["higo_taxon"] for r in res]).astype(np.float32)
            src = f"{self.higo_device}{', bf16' if self.higo_device.startswith('cuda') else ', fp32'}"
        else:
            Dh, src = np.asarray(higo, np.float32), "precomputed"
        H, floor = higo_matrix(topk_csr(Dh, int(self.cfg.get("pred_topk", 500))), self.vcols, self.nT, self.topk)
        th = time.time() - t
        self._log(f"[3/5] HiGO ({src}, {sum(len(r['seq']) for r in recs):,} residues): top {self.topk} terms "
                  f"({th:.1f}s)")
        t = time.time()
        same = self.same_org_best_hit(hits, qids, [r["organism"] for r in res])
        feats = stack_features(Pk, H, floor, self.vcols, maxpid, has_hit, n_hits, [r["org_group"] for r in res],
                               same, self.onto)
        P = apply_stacker(self.stacker, feats, self.onto)
        ts = time.time() - t
        self._log(f"[4/5] twin stacker: {len(feats['t']):,} candidate rows ({ts:.2f}s)")
        t = time.time()
        out = self._format(recs, res, hits, P, maxpid, has_hit, n_hits, same)
        tf = time.time() - t
        total = time.time() - t0
        self._log(f"[5/5] propagated scores >= {MIN_SCORE}: {P.nnz:,} ({tf:.2f}s); total {total:.1f}s for {n} "
                  f"protein(s)")
        for o in out:
            o["timing_s"] = {"diamond": round(td, 2), "knn": round(tk, 3), "higo": round(th, 2),
                             "stacker": round(ts, 3), "format": round(tf, 3), "total_request": round(total, 2)}
        self.last_P = P
        return out

    def _log(self, msg):
        if self.verbose:
            log(msg)

    def _format(self, recs, res, hits, P, maxpid, has_hit, n_hits, same) -> list[dict]:
        from cafa6.temporal_val import ident_bins
        bins = ident_bins(maxpid).values
        P = P.tocsr()
        out = []
        hg = {q: d for q, d in hits.groupby("q", sort=False)}
        for i, r in enumerate(recs):
            a, b = P.indptr[i], P.indptr[i + 1]
            cols, vals = P.indices[a:b], P.data[a:b]
            o = np.argsort(-vals, kind="stable")
            scores = [{"term": self.onto.terms[c], "score": round(float(v), 4), "aspect": ASPECTS[self.onto.ns[c]],
                       "name": self.names.get(self.onto.terms[c], "")} for c, v in zip(cols[o], vals[o])]
            above = {asp: [s for s in scores if s["aspect"] == asp and s["score"] >= THRESH[asp]] for asp in ASPECTS}
            h = hg.get(r["id"])
            near = []
            if h is not None:
                for row in h.head(5).itertuples():
                    near.append({"id": row.s, "organism": self.ref_org.get(row.s), "split": self.ref_split.get(row.s),
                                 "pident": float(row.pident), "bits": float(row.bits), "evalue": float(row.evalue),
                                 "qcov": float(row.qcov)})
            out.append({"id": r["id"], "length": len(r["seq"]), **res[i], "max_identity": float(maxpid[i]),
                        "identity_bin": str(bins[i]), "has_hit": bool(has_hit[i]), "n_hits": int(n_hits[i]),
                        "best_train_hit_same_organism": bool(same[i]), "nearest": near, "thresholds": THRESH,
                        "above_threshold": above, "scores": scores})
        return out


# ------------------------------------------------------------------------------------------------ writers

def to_tsv(results: list[dict]) -> str:
    lines = ["protein\tgo_term\tscore\taspect\tterm_name"]
    for r in results:
        for s in r["scores"]:
            lines.append(f"{r['id']}\t{s['term']}\t{s['score']:.4f}\t{s['aspect']}\t{s['name']}")
    return "\n".join(lines) + "\n"


def low_identity_note(r: dict) -> str | None:
    if r["identity_bin"] in ("<30", "no_hit"):
        return ("No reference above 30% identity: accuracy is much lower here (look-3 IA-weighted Fmax 0.451 below "
                "30% identity, 0.342 without a hit, against 0.718 over the whole test set).")
    return None


def to_markdown(results: list[dict]) -> str:
    out = ["# HiGO-R2 GO-term predictions", "",
           f"Terms at or above the per-aspect thresholds (MF {THRESH['MF']}, BP {THRESH['BP']}, CC {THRESH['CC']}); "
           "every score >= 0.01 is in predictions.tsv.", ""]
    for r in results:
        org = r["organism"] or "unknown organism"
        out += [f"## {r['id']}", "",
                f"{r['length']} residues; taxon {r['taxon'] if r['taxon'] is not None else 'not given'} ({org}); "
                f"best identity {r['max_identity']:.1f}% (bin {r['identity_bin']}), {r['n_hits']} hits used.", ""]
        note = low_identity_note(r)
        if note:
            out += [f"> {note}", ""]
        for asp in ASPECTS:
            terms = r["above_threshold"][asp]
            out.append(f"### {asp} ({len(terms)} terms >= {THRESH[asp]})")
            out.append("")
            if terms:
                out += ["| GO term | name | score |", "|---|---|---|"]
                out += [f"| {s['term']} | {s['name']} | {s['score']:.3f} |" for s in terms]
            else:
                out.append("none")
            out.append("")
        out += ["### Nearest references", ""]
        if r["nearest"]:
            out += ["| reference | organism | set | identity % | coverage % | bitscore | e-value |",
                    "|---|---|---|---|---|---|---|"]
            out += [f"| {h['id']} | {h['organism']} | {h['split']} | {h['pident']:.1f} | {h['qcov']:.0f} | "
                    f"{h['bits']:.0f} | {h['evalue']:.1e} |" for h in r["nearest"]]
        else:
            out.append("no DIAMOND hit at e <= 1e-3")
        out.append("")
    return "\n".join(out)


def write_outputs(results: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.tsv").write_text(to_tsv(results))
    (out / "predictions.json").write_text(json.dumps(results, indent=1))
    (out / "report.md").write_text(to_markdown(results))


# ------------------------------------------------------------------------------------------------ checks

def verify_stacker() -> dict:
    """Saved twin model on the stored test features (knn_trainval/test_knn.npz, l3_c0_tv2 pred_test_top500,
    cells.parquet) through this module's feature and inference code, against pred_test_twin.npz. No labels."""
    from cafa6 import knn_sharp as ks
    from cafa6.data import load_pred
    t = time.time()
    onto = ks.Ontology()
    model, ck = load_stacker(STACKER)
    K = ks.load_knn("test", Path(ck["knn_dir"]))
    ids = K["ids"].astype(str)
    cols = np.array([onto.idx[x] for x in K["terms_full"]])
    pf = K["p_full"].tocoo()
    Pk = sp.csr_matrix((pf.data, (pf.row, cols[pf.col])), shape=(len(ids), len(onto.terms)))
    df = pd.read_parquet(TV2 / "train.parquet", columns=["id", "split"])
    rid = df.id.values[(df.split == "test").values].astype(str)
    assert list(rid) == list(ids), "test row order differs between train.parquet and the kNN file"
    D = load_pred(HIGO_RUN, "test", dense=False).toarray().astype(np.float32)
    vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
    vcols = np.array([onto.idx[x] for x in vocab.term])
    H, floor = higo_matrix(D, vcols, len(onto.terms), int(ck["higo_topk"]))
    c = pd.read_parquet(TV2 / "cells.parquet")
    c = c[c.split == "test"].drop_duplicates("id").set_index("id").reindex(ids)
    feats = stack_features(Pk, H, floor, vcols, K["max_pident"], K["has_hit"], K["n_hits"],
                           c.org_group.fillna("other").values, c.best_same_org.fillna(False).values, onto)
    P = apply_stacker(model, feats, onto)
    rids, R = read_pred_npz(STACKER_DIR / "pred_test_twin.npz", onto)
    assert list(rids) == list(ids)
    res = {"proteins": len(ids), **compare(P, R), "seconds": round(time.time() - t, 1)}
    log("saved stacker vs pred_test_twin.npz (all test proteins): " + json.dumps(res))
    return res


def deviation(P_new: sp.csr_matrix, P_ref: sp.csr_matrix, ns: np.ndarray) -> dict:
    """Size of the differences beyond the max: share of entries within TOL, 99th percentile, threshold flips
    (ns: aspect index per ontology column)."""
    def flat(M):
        M = sp.csr_matrix(M).tocoo()
        return pd.Series(M.data.astype(np.float32), index=M.row.astype(np.int64) * M.shape[1] + M.col)
    a = flat(P_new).astype(np.float16).astype(np.float32)
    b = flat(P_ref)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
    d = (j.a - j.b).abs()
    thr = np.array([THRESH[x] for x in ASPECTS])[ns[(j.index.values % len(ns))]]
    flips = int(((j.a.values >= thr) != (j.b.values >= thr)).sum())
    return {"share_within_tol": round(float((d <= TOL).mean()), 4), "p99_abs_diff": round(float(d.quantile(0.99)), 5),
            "mean_abs_diff": round(float(d.mean()), 6), "threshold_decision_flips": flips,
            "decisions_above_threshold_ref": int((j.b.values >= thr).sum())}


def self_test(n: int = 20, seed: int = 0, out: Path | None = None, use_cuda: bool = True) -> dict:
    """n GO-GPT test proteins from their sequences end to end against their rows of pred_test_twin.npz (self hits,
    reference accession == query accession, dropped as in the benchmark; no labels read). Three runs share one
    DIAMOND search:
      cpu          the serving path: HiGO on CPU in fp32
      cpu_stored   the same, with the HiGO rows stored in runs/l3_c0_tv2/pred_test_top500.npz (checks DIAMOND, kNN,
                   organism features, stacker and propagation of the new-sequence path)
      cuda_bf16    HiGO on a GPU under bf16 autocast, the precision the benchmark predictions were made with
    PASS = cpu_stored and (when a GPU is present) cuda_bf16 within TOL. The cpu run differs by the fp32-vs-bf16
    numerics of the HiGO network; its deviation is reported."""
    from cafa6.data import load_pred
    pr = Predictor()
    df = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split", "organism"])
    te = df[df.split == "test"].reset_index(drop=True)
    pick = np.sort(np.random.default_rng(seed).choice(len(te), n, replace=False))
    recs = [{"id": te.id[i], "seq": te.seq[i], "organism": te.organism[i]} for i in pick]
    rids, R = read_pred_npz(STACKER_DIR / "pred_test_twin.npz", pr.onto)
    pos = {p: i for i, p in enumerate(rids)}
    R = R[[pos[r["id"]] for r in recs]]
    res = {"proteins": n, "seed": seed, "ids": [r["id"] for r in recs], "lengths": [len(r["seq"]) for r in recs],
           "tolerance": TOL, "runs": {}}

    log("== self-test run cpu (serving path)")
    t = time.time()
    results = pr.predict(recs, exclude_self=True)
    dt = time.time() - t
    hits = pr.last_hits
    res["runs"]["cpu"] = {"predict_seconds": round(dt, 1), **compare(pr.last_P, R), **deviation(pr.last_P, R, pr.onto.ns),
                          "per_protein_max_abs_diff": {r["id"]: compare(pr.last_P[i], R[i])["max_abs_diff"]
                                                       for i, r in enumerate(recs)}}
    log("== self-test run cpu_stored (stored HiGO rows)")
    tdf = pd.read_parquet(TV2 / "train.parquet", columns=["id", "split"])
    hid = tdf.id.values[(tdf.split == "test").values]
    hpos = {p: i for i, p in enumerate(hid)}
    Dst = load_pred(HIGO_RUN, "test", dense=False)[[hpos[r["id"]] for r in recs]].toarray()
    pr.predict(recs, exclude_self=True, hits=hits, higo=Dst)
    res["runs"]["cpu_stored"] = compare(pr.last_P, R)
    if use_cuda and torch.cuda.is_available():
        log("== self-test run cuda_bf16")
        pr.set_higo_device("cuda")
        pr.predict(recs, exclude_self=True, hits=hits)
        res["runs"]["cuda_bf16"] = compare(pr.last_P, R)
        pr.set_higo_device("cpu")
    else:
        res["runs"]["cuda_bf16"] = "skipped (no GPU)"
    must = [k for k in ("cpu_stored", "cuda_bf16") if isinstance(res["runs"][k], dict)]
    res["pass"] = bool(all(res["runs"][k]["max_abs_diff"] <= TOL for k in must))
    for k, v in res["runs"].items():
        log(f"  {k:10s} " + (f"max abs diff {v['max_abs_diff']:.2e}" if isinstance(v, dict) else v)
            + (f"  (entries within {TOL}: {v['share_within_tol']:.1%}, p99 {v['p99_abs_diff']}, threshold flips "
               f"{v['threshold_decision_flips']}/{v['decisions_above_threshold_ref']})" if k == "cpu" else ""))
    if out is not None:
        write_outputs(results, out)
        (out / "self_test.json").write_text(json.dumps(res, indent=1))
    log(f"self-test {'PASSED' if res['pass'] else 'FAILED'} ({' and '.join(must)} within {TOL})")
    return res


def resolve_device(device: str) -> str:
    """auto = cuda (bf16, bit-for-bit the benchmarked HiGO scores) when a GPU is visible, else cpu (fp32; up to 0.16
    away from the benchmarked scores on single entries, see the self-test)."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="auto", help="HiGO device: auto | cpu | cuda[:N]")
    ap.add_argument("--fasta", help="input FASTA")
    ap.add_argument("--taxon", type=int, default=None, help="NCBI taxon id for records without OX=")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for torch (default min(16, cores))")
    ap.add_argument("--diamond-threads", type=int, default=None, help="default min(32, cores)")
    ap.add_argument("--exclude-self", action="store_true",
                    help="drop hits to a reference with the query's accession (benchmark convention)")
    ap.add_argument("--build", action="store_true", help="build data/serve (reference db, labels, taxa)")
    ap.add_argument("--self-test", action="store_true", help="20 GO-GPT test proteins end to end vs pred_test_twin")
    ap.add_argument("--verify-stacker", action="store_true", help="saved stacker on stored test features")
    ap.add_argument("--no-cuda", action="store_true", help="self-test without the GPU bf16 run")
    a = ap.parse_args()
    if a.build:
        build_serve()
    if a.verify_stacker:
        verify_stacker()
    if a.self_test:
        r = self_test(out=Path(a.out) if a.out else None, use_cuda=not a.no_cuda)
        if not r["pass"]:
            raise SystemExit(1)
    if a.fasta:
        if not a.out:
            ap.error("--out is required with --fasta")
        recs = parse_fasta(Path(a.fasta).read_text(), a.taxon)
        err = validate(recs)
        if err:
            raise SystemExit("invalid input:\n  " + "\n  ".join(err))
        log(f"{len(recs)} sequence(s) from {a.fasta}")
        pr = Predictor(threads=a.threads, diamond_threads=a.diamond_threads, higo_device=resolve_device(a.device))
        results = pr.predict(recs, exclude_self=a.exclude_self)
        write_outputs(results, Path(a.out))
        log(f"wrote {a.out}/predictions.tsv, predictions.json, report.md")
    if not (a.build or a.verify_stacker or a.self_test or a.fasta):
        ap.print_help()


if __name__ == "__main__":
    main()
