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
