"""DIAMOND homology: label transfer (DiamondScore, DeepGOPlus) and max-identity strata."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import PROC, ROOT

DIAMOND = ROOT / "tools" / "diamond"


def _write(df, path):
    with open(path, "w") as f:
        for pid, s in zip(df.id, df.seq):
            f.write(f">{pid}\n{s}\n")


def diamond_search(db_df: pd.DataFrame, q_df: pd.DataFrame, threads: int = 64, max_hits: int = 50,
                   exclude_self: bool = True) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(dir=PROC) as tmp:
        tmp = Path(tmp)
        _write(db_df, tmp / "db.fa")
        _write(q_df, tmp / "q.fa")
        subprocess.run([str(DIAMOND), "makedb", "--in", str(tmp / "db.fa"), "-d", str(tmp / "db"), "--quiet"], check=True)
        subprocess.run([str(DIAMOND), "blastp", "-d", str(tmp / "db"), "-q", str(tmp / "q.fa"), "-o", str(tmp / "hits.tsv"),
                        "--more-sensitive", "-e", "1e-3", "-k", str(max_hits), "-p", str(threads), "--quiet",
                        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)
        hits = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None,
                           names=["q", "s", "pident", "bits", "evalue", "qcov"])
    if exclude_self:
        hits = hits[hits.q != hits.s]
    return hits


def diamond_score(hits: pd.DataFrame, q_ids, db_ids, Y_db: sp.csr_matrix) -> np.ndarray:
    """score(q, t) = sum_j bits_j * y_jt / sum_j bits_j over hits j of q."""
    qidx = {p: i for i, p in enumerate(q_ids)}
    didx = {p: i for i, p in enumerate(db_ids)}
    h = hits[hits.q.isin(qidx) & hits.s.isin(didx)]
    W = sp.csr_matrix((h.bits.values.astype(np.float32), (h.q.map(qidx).values, h.s.map(didx).values)),
                      shape=(len(q_ids), len(db_ids)))
    W.sum_duplicates()
    norm = np.asarray(W.sum(1)).ravel()
    norm[norm == 0] = 1
    S = (W @ Y_db).toarray() / norm[:, None]
    return S.astype(np.float32)


def max_identity(hits: pd.DataFrame, q_ids) -> pd.Series:
    """Best percent identity to any training protein (0 when no DIAMOND hit at e<=1e-3)."""
    m = hits.groupby("q").pident.max()
    return pd.Series(q_ids, index=q_ids).map(m).fillna(0.0)


def identity_bin(pid: pd.Series) -> pd.Series:
    return pd.cut(pid, [-1, 0, 30, 50, 101], labels=["no_hit", "<30%", "30-50%", ">=50%"])
