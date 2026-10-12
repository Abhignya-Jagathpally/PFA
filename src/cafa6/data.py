"""CAFA-6 data loading, cleaning, GO propagation and label-space construction."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import obonet
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = Path(os.environ.get("CAFA6_PROC", ROOT / "data" / "proc"))

ASPECT_ROOT = {"F": "GO:0003674", "P": "GO:0008150", "C": "GO:0005575"}
NAMESPACE_ASPECT = {"molecular_function": "F", "biological_process": "P", "cellular_component": "C"}
AMBIGUOUS = set("XBZUOJ")
PROP_RELS = ("is_a", "part_of")


def read_fasta(path: Path) -> pd.DataFrame:
    """Parses both the UniProt-style train FASTA and the `>ID taxon` test FASTA."""
    rows, pid, tax, buf = [], None, None, []

    def flush():
        if pid is not None:
            rows.append((pid, tax, "".join(buf)))

    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                flush()
                head = line[1:].strip()
                if "|" in head.split()[0]:
                    pid = head.split("|")[1]
                    m = re.search(r"OX=(\d+)", head)
                    tax = int(m.group(1)) if m else -1
                else:
                    parts = head.split()
                    pid = parts[0]
                    tax = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
                buf = []
            else:
                buf.append(line.strip())
        flush()
    df = pd.DataFrame(rows, columns=["id", "taxon", "seq"])
    df["length"] = df.seq.str.len()
    df["ambig_frac"] = df.seq.map(lambda s: sum(c in AMBIGUOUS for c in s) / max(len(s), 1))
    return df


def load_go(obo_path: Path = RAW / "Train" / "go-basic.obo"):
    """Returns (parents: term -> set of direct is_a/part_of parents, aspect: term -> F/P/C)."""
    g = obonet.read_obo(obo_path)
    parents: dict[str, set[str]] = defaultdict(set)
    for child, parent, rel in g.edges(keys=True):
        if rel in PROP_RELS:
            parents[child].add(parent)
    aspect = {n: NAMESPACE_ASPECT.get(d.get("namespace"), "?") for n, d in g.nodes(data=True)}
    alt = {}
    for n, d in g.nodes(data=True):
        for a in d.get("alt_id", []):
            alt[a] = n
    return parents, aspect, alt


def ancestor_closure(parents: dict[str, set[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def anc(t):
        if t in memo:
            return memo[t]
        out = set()
        for p in parents.get(t, ()):
            out.add(p)
            out |= anc(p)
        memo[t] = out
        return out

    import sys
    sys.setrecursionlimit(100000)
    for t in list(parents):
        anc(t)
    return memo


def propagate(terms: pd.DataFrame, closure: dict[str, set[str]], alt: dict[str, str]) -> pd.DataFrame:
    """Expands leaf annotations to all ancestors (true-path rule), same as cafaeval `-prop`."""
    out = defaultdict(set)
    for pid, t in zip(terms.EntryID, terms.term):
        t = alt.get(t, t)
        out[pid].add(t)
        out[pid] |= closure.get(t, set())
    rows = [(p, t) for p, ts in out.items() for t in ts]
    return pd.DataFrame(rows, columns=["id", "term"])


def load_ia(path: Path = RAW / "IA.tsv") -> dict[str, float]:
    ia = pd.read_csv(path, sep="\t", header=None, names=["term", "ia"])
    return dict(zip(ia.term, ia.ia))


# ----------------------------------------------------------------------------- predictions on disk
def save_pred(run_dir: Path, split: str, scores: np.ndarray, top_k: int = 0) -> Path:
    """Dense `pred_{split}.npy` (top_k=0, the historical format) or at most top_k scores per protein as CSR arrays in
    `pred_{split}_top{k}.npz` (data float16, indices uint16/int32, indptr int64, shape), about 15x smaller.
    Scores are kept strictly above each protein's k-th largest score: with MCM output a parent never scores below
    its child, so the kept set stays ancestor-closed even when scores tie at the cut."""
    if not top_k:
        np.save(run_dir / f"pred_{split}.npy", scores)
        return run_dir / f"pred_{split}.npy"
    import scipy.sparse as sp
    n, t = scores.shape
    k = min(top_k, t)
    s32 = scores.astype(np.float32)
    kth = np.partition(s32, t - k, axis=1)[:, t - k][:, None] if k < t else np.zeros((n, 1), np.float32)
    keep = (s32 > kth) | ((k == t) & (s32 > 0))
    m = sp.csr_matrix(np.where(keep, s32, 0.0))
    path = run_dir / f"pred_{split}_top{k}.npz"
    np.savez_compressed(path, data=m.data.astype(np.float16), indices=m.indices.astype(np.uint16 if t < 65536 else np.int32),
                        indptr=m.indptr.astype(np.int64), shape=np.array([n, t]))
    return path


def load_pred(run_dir: Path, split: str, dense: bool = True):
    """Reads whichever prediction format a run wrote; dense=False returns a scipy CSR (float32) for top-k files."""
    import scipy.sparse as sp
    run_dir = Path(run_dir)
    if (run_dir / f"pred_{split}.npy").exists():
        return np.load(run_dir / f"pred_{split}.npy")
    files = sorted(run_dir.glob(f"pred_{split}_top*.npz"))
    if not files:
        raise FileNotFoundError(f"no pred_{split}.npy or pred_{split}_top*.npz in {run_dir}")
    z = np.load(files[0])
    m = sp.csr_matrix((z["data"].astype(np.float32), z["indices"].astype(np.int64), z["indptr"]),
                      shape=tuple(z["shape"]))
    return m.toarray().astype(np.float16) if dense else m


# ----------------------------------------------------------------------------- retrieval prior (HiGO-R)
# Files under <PROC>/knn/ (override with the `knn_dir` config key):
#   {split}_knn.npz      contract written by knn_sharp.py, used for val/test (and train if no hits file). CSR p_knn
#                        over the vocab as data/indices/indptr/shape (optionally prefixed "p_" or "p_knn_"), a
#                        per-protein max identity (max_identity | max_id | max_pident; percent values are rescaled
#                        to [0, 1]), optional has_hit, optional ids (row order; without it rows follow the split's
#                        row order in train.parquet), optional CSR s_star / frac / count as {name}_data/_indices/...
#   train_hits.parquet   train-vs-train hits [q, s, pident, bits, qcov] (protein ids, percent units; DIAMOND
#                        -k 50 output as from homology.diamond_search). Self hits are dropped here by accession.
#                        Needed for the per-batch identity cap during training.
#   {split}_hits.parquet same layout for val/test against the train split (only for knn_eval_source: hits).
KNN_CAPS = [[0, 0.25], [30, 0.25], [50, 0.20], [70, 0.15], [None, 0.15]]


def _csr(z, prefixes, shape_hint=None):
    import scipy.sparse as sp
    for pre in prefixes:
        if f"{pre}data" in z and f"{pre}indptr" in z:
            shape = tuple(z[f"{pre}shape"]) if f"{pre}shape" in z else shape_hint
            return sp.csr_matrix((z[f"{pre}data"].astype(np.float32), z[f"{pre}indices"], z[f"{pre}indptr"]),
                                 shape=shape)
    return None


class KnnPrior:
    """Per-protein retrieval inputs for HiGO-R, addressed by global row of train.parquet.

    batch(rows, cap) returns numpy arrays p (B x T), max_id (B, fraction), has_hit (B) and s_star / frac / count
    (B x T, or None when the source lacks them). cap=0 means no retrieval, cap=c keeps hits with identity <= c
    percent, cap=None keeps all hits. Caps other than 0/None need a hits table."""

    def __init__(self, n_terms: int):
        self.n_terms, self.row = n_terms, {}

    # -- from a hits table: sharpened kNN computed on the fly
    @classmethod
    def from_hits(cls, hits: pd.DataFrame, df: pd.DataFrame, Y, query_rows, db_rows, T: float = 2.0,
                  top_k: int = 10, gamma: float = 0.0, norm: str = "all", aspect=None):
        """Same prior as knn_sharp.KNN.prior: top_k hits by bitscore, w = bits * exp((s - max s)/T) * cov^gamma;
        norm="aspect" divides each aspect by the weight of the hits annotated in that aspect (needs `aspect`)."""
        self = cls(Y.shape[1])
        self.kind, self.Y, self.T, self.top_k, self.gamma, self.norm = "hits", Y, T, top_k, gamma, norm
        if norm == "aspect":
            assert aspect is not None, "norm='aspect' needs the per-term aspect array"
            self.aspect_cols = [np.where(np.asarray(aspect) == a)[0] for a in ("F", "P", "C")]
        ids = df.id.values
        qmap = {ids[r]: int(r) for r in query_rows}
        smap = {}
        for r in db_rows:
            smap.setdefault(ids[r], int(r))
        h = hits[(hits.q != hits.s) & hits.q.isin(qmap) & hits.s.isin(smap)]
        qcov = h.qcov.values if "qcov" in h else np.full(len(h), 100.0)
        h = pd.DataFrame({"q": h.q.map(qmap).values, "s": h.s.map(smap).values, "pident": h.pident.values,
                          "bits": h.bits.values, "qcov": qcov}).sort_values(["q", "bits"], ascending=[True, False])
        h = h.drop_duplicates(["q", "s"])                  # DIAMOND can report several HSPs per subject
        self.q, self.s = h.q.values, h.s.values
        self.pident, self.bits, self.qcov = (h[c].values.astype(np.float32) for c in ("pident", "bits", "qcov"))
        starts = np.searchsorted(self.q, np.asarray(sorted(qmap.values())), "left")
        ends = np.searchsorted(self.q, np.asarray(sorted(qmap.values())), "right")
        self.row = {int(r): (int(a), int(b)) for r, a, b in zip(sorted(qmap.values()), starts, ends)}
        return self

    # -- from the knn_sharp.py contract file
    @classmethod
    def from_contract(cls, path: Path, df: pd.DataFrame, split_rows, n_terms: int):
        self = cls(n_terms)
        self.kind = "contract"
        z = np.load(path, allow_pickle=False)
        self.P = _csr(z, ("p_knn_", "p_", ""))
        if self.P is None:
            raise KeyError(f"{path}: no CSR p_knn arrays (data/indices/indptr/shape) found; keys {list(z.keys())}")
        n = self.P.shape[0]
        if self.P.shape[1] != n_terms:
            raise ValueError(f"{path}: p_knn has {self.P.shape[1]} columns, vocab has {n_terms}")
        mid = next((z[k] for k in ("max_identity", "max_id", "max_pident") if k in z), None)
        if mid is None:
            raise KeyError(f"{path}: needs a per-protein max identity (max_identity | max_id | max_pident)")
        mid = mid.astype(np.float32)
        self.max_id = mid / 100.0 if mid.max(initial=0) > 1.0 else mid
        self.has_hit = z["has_hit"].astype(bool) if "has_hit" in z else (np.diff(self.P.indptr) > 0) | (mid > 0)
        self.extra = {f: _csr(z, (f"{f}_",), (n, n_terms)) for f in ("s_star", "frac", "count")}
        split_rows = np.asarray(split_rows)
        if "ids" in z:
            pos = {pid: i for i, pid in enumerate(z["ids"].astype(str))}
            missing = [r for r in split_rows if df.id.values[r] not in pos]
            if missing:
                raise KeyError(f"{path}: {len(missing)} of {len(split_rows)} proteins missing, e.g. "
                               f"{df.id.values[missing[0]]}")
            self.row = {int(r): pos[df.id.values[r]] for r in split_rows}
        else:
            if n != len(split_rows):
                raise ValueError(f"{path}: {n} rows without an ids array, but the split has {len(split_rows)}")
            self.row = {int(r): i for i, r in enumerate(split_rows)}
        return self

    def covers(self, r) -> bool:
        return int(r) in self.row

    def batch(self, rows, cap=None) -> dict:
        B, T = len(rows), self.n_terms
        out = {"p": np.zeros((B, T), np.float32), "max_id": np.zeros(B, np.float32), "has_hit": np.zeros(B, bool)}
        if self.kind == "contract":
            if cap not in (None, 0):
                raise ValueError("identity caps other than 0 / None need a hits table (train_hits.parquet)")
            for f in ("s_star", "frac", "count"):
                out[f] = None if self.extra[f] is None else np.zeros((B, T), np.float32)
            if cap == 0:
                return out
            loc = np.array([self.row[int(r)] for r in rows])
            out["p"] = self.P[loc].toarray()
            out["max_id"], out["has_hit"] = self.max_id[loc], self.has_hit[loc]
            for f, m in self.extra.items():
                if m is not None:
                    out[f] = m[loc].toarray()
            return out
        for f in ("s_star", "frac", "count"):
            out[f] = np.zeros((B, T), np.float32)
        if cap == 0:
            return out
        for i, r in enumerate(rows):
            a, b = self.row[int(r)]
            if a == b:
                continue
            pid = self.pident[a:b]
            keep = np.nonzero(pid <= cap)[0] if cap is not None else np.arange(b - a)
            if len(keep) == 0:
                continue
            out["max_id"][i] = pid[keep].max() / 100.0
            sel = keep[:self.top_k] + a                     # hits are sorted by bitscore within a query
            p_sel = self.pident[sel]
            w = self.bits[sel] * (np.exp((p_sel - p_sel.max()) / self.T) if np.isfinite(self.T) else 1.0)
            if self.gamma:
                w = w * np.clip(self.qcov[sel] / 100.0, 1e-3, 1.0) ** self.gamma
            Ys = self.Y[self.s[sel]].toarray()
            if self.norm == "aspect":
                for cols in self.aspect_cols:
                    den = w[Ys[:, cols].any(1)].sum()
                    if den > 0:
                        out["p"][i, cols] = np.minimum(1.0, w @ Ys[:, cols] / den)
            else:
                out["p"][i] = np.minimum(1.0, w @ Ys / w.sum())
            out["count"][i] = Ys.sum(0)
            out["frac"][i] = out["count"][i] / len(sel)
            out["s_star"][i] = (Ys * (p_sel[:, None] / 100.0)).max(0)
            out["has_hit"][i] = True
        return out


class KnnBank:
    """Dispatches rows to whichever KnnPrior covers them; a row no source covers is an error, not a silent no-hit."""

    def __init__(self, sources):
        self.sources = [s for s in sources if s is not None]

    def batch(self, rows, cap=None) -> dict:
        rows = [int(r) for r in rows]
        groups = {}
        for i, r in enumerate(rows):
            src = next((k for k, s in enumerate(self.sources) if s.covers(r)), None)
            if src is None:
                raise KeyError(f"no kNN source covers row {r}")
            groups.setdefault(src, []).append(i)
        if len(groups) == 1:
            (k, _), = groups.items()
            return self.sources[k].batch(rows, cap)
        T = self.sources[0].n_terms
        out = {"p": np.zeros((len(rows), T), np.float32), "max_id": np.zeros(len(rows), np.float32),
               "has_hit": np.zeros(len(rows), bool)}
        for k, idx in groups.items():
            part = self.sources[k].batch([rows[i] for i in idx], cap)
            for f, v in part.items():
                if v is None:
                    out[f] = None
                    continue
                if f not in out:
                    out[f] = np.zeros((len(rows),) + v.shape[1:], v.dtype)
                if out[f] is not None:
                    out[f][idx] = v
        return out


def load_knn(cfg: dict, df: pd.DataFrame, Y, split: str, rows, aspect=None) -> KnnPrior:
    """kNN source for one split. Train uses train_hits.parquet when present (identity caps possible), else the
    contract file; val/test/tv use the contract file unless knn_eval_source is 'hits', in which case the hits come
    from {split}_hits.parquet or the benchmark's DIAMOND files (diamond_hits_valtest / diamond_hits_tv)."""
    kdir = Path(cfg.get("knn_dir") or PROC / "knn")
    db_rows = np.where(df.split.isin(cfg.get("knn_db_splits", ["train"])))[0]
    kw = dict(T=float(cfg.get("knn_T", 2.0)), top_k=cfg.get("knn_topk", 10), gamma=cfg.get("knn_gamma", 0.0),
              norm=cfg.get("knn_norm", "all"), aspect=aspect)
    npz_path = kdir / f"{split}_knn.npz"
    hits_paths = [kdir / f"{split}_hits.parquet"]
    if split != "train":
        hits_paths += [PROC / f"diamond_hits_{split}.parquet", PROC / "diamond_hits_valtest.parquet"]
    hits_path = next((p for p in hits_paths if p.exists()), hits_paths[0])
    use_hits = hits_path.exists() and (split == "train" or cfg.get("knn_eval_source", "contract") == "hits")
    if use_hits:
        return KnnPrior.from_hits(pd.read_parquet(hits_path), df, Y, rows, db_rows, **kw)
    if npz_path.exists():
        return KnnPrior.from_contract(npz_path, df, rows, Y.shape[1])
    raise FileNotFoundError(f"knn_fusion needs {npz_path} (knn_sharp.py contract) or {hits_path}; neither exists")
