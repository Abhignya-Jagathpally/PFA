"""Temporal validation split (TV) for the GO-GPT benchmark, built from dated GOA annotations.

GO-GPT's own validation split is a random 10% of its November 2022 training proteins, and it ranks
homology settings almost in reverse of the temporal test. This module carves a CAFA-style temporal
holdout out of GO-GPT's training proteins instead.

Definitions
-----------
T0 = 2021-11-30 and the label cutoff is 2022-11-30 (GO-GPT trained on GOA November 2022).
The only dated annotations we hold are `data/grounding/goa_reference.parquet`, a 2026 GOA release for
ten reference proteomes (human, mouse, rat, Arabidopsis, fly, budding and fission yeast, worm, E. coli,
M. tuberculosis). There is no zebrafish, which is 45% of the GO-GPT test set.

Evidence codes: EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC plus the high-throughput codes HTP, HDA, HMP, HGI,
HEP. The paper lists only the first eight, but GO-GPT's training labels contain 98.8% of the GOA HDA
annotations dated before the cutoff, and 16,666 of the 16,821 training proteins that have terms supported
only by high-throughput codes carry those terms in their labels. Its label filter therefore kept them.

For every GO-GPT training protein in that GOA file and every aspect, with L = its GO-GPT labels in the
aspect (already propagated over is_a and part_of in GO 2023-01-01):
  pre = closure of its annotations dated <= T0, win = closure of those dated in (T0, cutoff].
  Post-T0 terms are L & win - pre. They are stripped from the training labels of every protein, so
  the label matrix approximates a T0 snapshot for the ten proteomes.
  The protein-aspect enters TV when it has no annotation dated <= T0, at least one dated in the window,
  and every term of L is implied by the window annotations (L is a subset of win). The last condition
  guards against GOA's habit of re-dating old annotations: if part of L cannot be explained by window
  annotations, that knowledge may predate T0 and the aspect is not used.
  A TV protein is no-knowledge (NK) when all of its labelled aspects are TV aspects, otherwise
  limited-knowledge (LK).

Outputs in data/proc_gogpt_tv (same layout as data/proc_gogpt):
  train.parquet   copy of the source with split = "tv" for NK TV proteins (they have no labels at T0,
                  so they must not be trained on as all-negative) and a boolean column `tv`.
                  LK TV proteins keep split = "train" with their TV aspects emptied, as GO-GPT's real
                  LK test proteins were in its training set with their other aspects.
  Y.npz           the T0 label matrix over the unchanged vocabulary (rows aligned with train.parquet).
  tv_ids.txt      TV protein accessions.
  gt_tv.tsv       CAFA ground truth (id, term) of the TV aspects, propagated, roots included, like gt_test.
  t0_stripped.parquet  every (id, term, aspect, kind) removed from the labels; kind is "tv" for TV aspects
                  and "pk" for post-T0 gains in aspects that already had pre-T0 knowledge.
  diamond_hits_tv.parquet  TV queries against the 119,749 training proteins, self excluded
                  (DIAMOND --more-sensitive, e <= 1e-3, top 50 after removing self).
  identity.parquet val/test rows of the source plus tv rows, [id, split, max_pident].
  tv_report.json  counts, organism mix and identity distribution next to the test.
  Symlinked unchanged: vocab.tsv, go_ancestors.npy, gt_val.tsv, gt_test.tsv, diamond_hits_valtest.parquet,
  diamond_val.npy, diamond_test.npy, prepare_report.json.

Using it for training: `CAFA6_PROC=data/proc_gogpt_tv python -m cafa6.train ... pred_id_sets='{tv: tv_ids.txt}'`.
The trainer trains on split == "train" and early-stops on split == "val", so NK TV proteins are left out and
LK TV proteins train on their T0 labels; pred_id_sets makes it also write pred_tv. The kNN database for TV
queries is the training split with T0 labels; val and test queries may use full labels because those are
legitimately known at the cutoff. With knn_fusion, the training-time prior is computed from hits with
knn_T / knn_topk / knn_gamma / knn_norm, so either set knn_eval_source: hits (diamond_hits_tv.parquet and
diamond_hits_valtest.parquet are here) or point knn_dir at contract files written with the same setting
(knn/ holds the TV-selected T=inf k=50, knn_T2_k10/ holds T=2 k=10). Identity-capped training also needs
knn/train_hits.parquet (train-vs-train DIAMOND), which this module does not build.

Caveat measured by knn_sharp: TV does not rank kNN sharpening like the GO-GPT test. The test's >=95%
identity bins are mostly same-organism duplicate accessions (94% of the 100% bin), where copying the top hit
pays; TV's close hits are mostly cross-species orthologs, where it does not.

TV v2 (`--v2`, data/proc_gogpt_tv2, same layout; v1 is untouched)
------------------------------------------------------------------
Two strata. Dated: the v1 rule on a wider dated GOA (goa_reference + EBI zebrafish proteome GAF + ZFIN and
RGD gene GAFs mapped to every UniProt accession of the gene; data/grounding/goa_tv2). Zebrafish adds little:
its window annotations fall on genes annotated before T0. DA (duplicate accession, see `da_stratum`):
the test's >=95% identity bins are further accessions of already-annotated genes whose annotations predate
T0, which no date rule can admit, so one accession per same-organism >=80% identity cluster is held out with
its full labels as ground truth. Extra files: cells.parquet [id, split, max_pident, bin, best_same_org,
org_group, stratum] for tv/val/test (test rows from DIAMOND hits only), train.parquet column tv_stratum
(dated_NK, dated_LK, DA), knn/train_hits.parquet (symlink to the train-vs-train DIAMOND). TV queries'
hits are their train-vs-train hits restricted to the v2 database (split == "train").

Usage: python -m cafa6.temporal_val [--threads 48] [--v2]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import ASPECT_ROOT, ROOT, ancestor_closure, load_go

SRC = ROOT / "data" / "proc_gogpt"
OUT = ROOT / "data" / "proc_gogpt_tv"
REF = ROOT / "runs" / "gogpt_cafa5" / "ref"
OBO = REF / "go-basic_gogpt.obo"
GOA = ROOT / "data" / "grounding" / "goa_reference.parquet"
GOGPT_TRAIN = ROOT / "external" / "gogpt_tmp_backup" / "gogpt_train" / "data"
DIAMOND = ROOT / "tools" / "diamond"
LOG = ROOT / "logs" / "tv_knn.log"

T0, CUTOFF = 20211130, 20221130
EXP_CODES = ("EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC")
HTP_CODES = ("HTP", "HDA", "HMP", "HGI", "HEP")
ASPECTS = ("F", "P", "C")
SYMLINKS = ("vocab.tsv", "go_ancestors.npy", "gt_val.tsv", "gt_test.tsv", "diamond_hits_valtest.parquet",
            "diamond_val.npy", "diamond_test.npy", "prepare_report.json")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def _as_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        return list(ast.literal_eval(x))
    return list(x)


class GO:
    """GO 2023-01-01 with the same closure prepare_gogpt used (is_a + part_of, alt ids remapped)."""

    def __init__(self, obo=OBO):
        parents, self.aspect, self.alt = load_go(obo)
        self.closure = ancestor_closure(parents)

    def close(self, terms) -> set:
        out = set()
        for t in terms:
            t = self.alt.get(t, t)
            out.add(t)
            out |= self.closure.get(t, set())
        return out


def full_labels(go: GO, split: str = "train") -> dict[str, set]:
    """GO-GPT's full propagated label sets (not restricted to the vocabulary) for one split."""
    if split == "test":
        d = pd.read_parquet(REF / "test.parquet", columns=["protein_id", "go_bp", "go_mf", "go_cc"])
    else:
        f = "train-00000-of-00001.parquet" if split == "train" else "validation-00000-of-00001.parquet"
        d = pd.read_parquet(GOGPT_TRAIN / f, columns=["protein_id", "go_bp", "go_mf", "go_cc"])
    return {p: go.close(_as_list(b) + _as_list(m) + _as_list(c))
            for p, b, m, c in zip(d.protein_id, d.go_bp, d.go_mf, d.go_cc)}


def t0_labels(go: GO, labels: dict[str, set], out: Path = OUT) -> dict[str, set]:
    """Training labels at T0: the full labels minus every stripped pair in t0_stripped.parquet."""
    st = pd.read_parquet(out / "t0_stripped.parquet")
    drop: dict[str, set] = {}
    for p, t in zip(st.id, st.term):
        drop.setdefault(p, set()).add(t)
    return {p: (s - drop[p]) if p in drop else s for p, s in labels.items()}


def build_split(go: GO, labels: dict[str, set], train_ids: set, g: pd.DataFrame | None = None):
    if g is None:
        g = pd.read_parquet(GOA, columns=["id", "go", "evidence", "date"])
    g = g[g.evidence.isin(EXP_CODES + HTP_CODES) & g.id.isin(train_ids)].copy()
    g["go"] = g.go.astype(str).map(lambda t: go.alt.get(t, t))
    g["aspect"] = g.go.map(lambda t: go.aspect.get(t, "?"))
    g = g[g.aspect.isin(ASPECTS)]
    g["period"] = np.where(g.date <= T0, "pre", np.where(g.date <= CUTOFF, "win", "post"))
    log(f"GOA experimental+HTP rows on GO-GPT training proteins: {len(g):,} ({g.id.nunique():,} proteins); "
        f"periods {g.period.value_counts().to_dict()}")

    roots = set(ASPECT_ROOT.values())
    stripped, tv_rows, stats = [], [], Counter()
    for (pid, asp), d in g.groupby(["id", "aspect"], sort=False):
        L = {t for t in labels.get(pid, ()) if go.aspect.get(t) == asp}
        pre = go.close(d.go[d.period == "pre"])
        win = go.close(d.go[d.period == "win"])
        if not win:
            continue
        gain = (L & win) - pre
        if not L - roots:
            stats["win_but_no_gogpt_label"] += 1
            continue
        if pre:
            if gain:
                stripped += [(pid, t, asp, "pk") for t in gain]
                stats["pk_aspects"] += 1
            continue
        if L <= win:
            stripped += [(pid, t, asp, "tv") for t in L]
            tv_rows.append((pid, asp))
            stats["tv_aspects"] += 1
        else:
            # knowledge we cannot date: keep it, but still strip what the window explains
            if gain:
                stripped += [(pid, t, asp, "pk") for t in gain]
            stats["rejected_unexplained_labels"] += 1
            stats["rejected_cover_ge_0.9"] += int(len(L & win) >= 0.9 * len(L))
    log(f"aspect-level counts: {dict(stats)}")
    return pd.DataFrame(stripped, columns=["id", "term", "aspect", "kind"]), \
        pd.DataFrame(tv_rows, columns=["id", "aspect"]), dict(stats)


def run_diamond(db: pd.DataFrame, q: pd.DataFrame, threads: int, k: int = 50) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        tmp = Path(tmp)
        for df, name in ((db, "db.fa"), (q, "q.fa")):
            with open(tmp / name, "w") as f:
                for pid, s in zip(df.id, df.seq):
                    f.write(f">{pid}\n{s}\n")
        subprocess.run([str(DIAMOND), "makedb", "--in", str(tmp / "db.fa"), "-d", str(tmp / "db"), "--quiet"],
                       check=True)
        subprocess.run([str(DIAMOND), "blastp", "-d", str(tmp / "db"), "-q", str(tmp / "q.fa"),
                        "-o", str(tmp / "hits.tsv"), "--more-sensitive", "-e", "1e-3", "-k", str(k + 1),
                        "-p", str(threads), "--quiet", "--tmpdir", str(tmp),
                        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)
        h = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None, names=["q", "s", "pident", "bits", "evalue", "qcov"])
    h = h[h.q != h.s]
    # DIAMOND may report several HSPs per pair; keep the best one, then the top k subjects by bits
    h = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
    return h.groupby("q", sort=False).head(k).reset_index(drop=True)


def ident_bins(x: pd.Series) -> pd.Series:
    """The seven identity bins of the plan, on max percent identity (0 = no hit)."""
    x = np.asarray(x, dtype=float)
    out = np.select([x <= 0, x < 30, x < 50, x < 80, x < 95, x < 100],
                    ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9"], "100")
    return pd.Series(out)


# ---------------------------------------------------------------------------------------------- TV v2
OUT_V2 = ROOT / "data" / "proc_gogpt_tv2"
GOA_V2 = ROOT / "data" / "grounding" / "goa_tv2"
TRAIN_HITS = OUT / "knn" / "train_hits.parquet"
ORG_GROUPS = {"Danio rerio": "zebrafish", "Rattus norvegicus": "rat", "Homo sapiens": "human",
              "Mus musculus": "mouse"}
DA_MIN_PIDENT, DA_CAP, DA_SEED = 80.0, 2000, 0


def org_group(organism: pd.Series) -> pd.Series:
    return organism.fillna("?").str.split(" (", regex=False).str[0].map(ORG_GROUPS).fillna("other")


def _read_gaf(path: Path, db: str) -> pd.DataFrame:
    """[obj, go, evidence, date] of the non-NOT rows of one GAF whose column 1 is `db`."""
    import gzip
    rows = []
    with (gzip.open(path, "rt", errors="replace") if str(path).endswith(".gz") else open(path, errors="replace")) as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 14 or p[0] != db or "NOT" in p[3].split("|"):
                continue
            if p[6] in EXP_CODES + HTP_CODES:
                rows.append((p[1], p[4], p[6], int(p[13]) if p[13].isdigit() else 0))
    return pd.DataFrame(rows, columns=["obj", "go", "evidence", "date"])


def dated_goa_v2(train_ids: set) -> tuple[pd.DataFrame, dict]:
    """Experimental + HTP annotations with dates, on GO-GPT training accessions:
    goa_reference.parquet (10 reference proteomes) + the EBI zebrafish proteome GAF + ZFIN's gene GAF mapped
    to every UniProt accession of the gene (zfin.org/downloads/uniprot.txt) + RGD's gene GAF mapped through
    GENES_RAT.txt UNIPROT_ID. Current accessions are mapped back to GO-GPT's 2022 accessions through
    uniprot_features.parquet (id -> primary_acc)."""
    parts, stats = [], {}
    g = pd.read_parquet(GOA, columns=["id", "go", "evidence", "date"])
    g = g[g.evidence.isin(EXP_CODES + HTP_CODES)]
    parts.append(g.assign(src="goa_reference")[["id", "go", "evidence", "date", "src"]].astype({"evidence": str}))
    d = _read_gaf(GOA_V2 / "D_rerio.goa", "UniProtKB").rename(columns={"obj": "id"})
    parts.append(d.assign(src="goa_zebrafish_proteome"))
    z = _read_gaf(GOA_V2 / "zfin.gaf.gz", "ZFIN")
    zm = pd.read_csv(GOA_V2 / "zfin_uniprot.txt", sep="\t", header=None, names=["obj", "so", "sym", "id"], dtype=str)
    parts.append(z.merge(zm[["obj", "id"]], on="obj")[["id", "go", "evidence", "date"]].assign(src="zfin"))
    r = _read_gaf(GOA_V2 / "rgd.gaf.gz", "RGD")
    rg = pd.read_csv(GOA_V2 / "GENES_RAT.txt", sep="\t", comment="#", dtype=str, usecols=["GENE_RGD_ID", "UNIPROT_ID"])
    rg = rg.dropna().assign(id=lambda x: x.UNIPROT_ID.str.split(";")).explode("id")
    rg = rg.rename(columns={"GENE_RGD_ID": "obj"})[["obj", "id"]]
    parts.append(r.merge(rg, on="obj")[["id", "go", "evidence", "date"]].assign(src="rgd"))
    g = pd.concat(parts, ignore_index=True)
    uf = pd.read_parquet(ROOT / "data" / "grounding" / "uniprot_features.parquet", columns=["id", "primary_acc"])
    uf = uf[(uf.primary_acc != "") & (uf.primary_acc != uf.id) & uf.id.isin(train_ids)]
    sec = uf.groupby("primary_acc").id.agg(list)
    direct = g[g.id.isin(train_ids)]
    via = g[~g.id.isin(train_ids) & g.id.isin(sec.index)].copy()
    via["id"] = via.id.map(sec)
    via = via.explode("id")
    g = pd.concat([direct, via], ignore_index=True)
    stats["rows_by_source"] = g.src.value_counts().to_dict()
    stats["proteins_by_source"] = g.groupby("src").id.nunique().to_dict()
    stats["rows_via_secondary_accession"] = int(len(via))
    g = g.drop_duplicates(["id", "go", "evidence", "date"])
    return g[["id", "go", "evidence", "date"]], stats


def da_stratum(df_tr: pd.DataFrame, hits: pd.DataFrame, exclude: set) -> pd.DataFrame:
    """Duplicate-accession (DA) holdout, the mechanism behind the test's high-identity bins: GO-GPT test
    proteins at >=95% identity are mostly further UniProt accessions of a gene whose other accession was
    already annotated (97% same ZFIN gene in zebrafish), and their annotations predate T0, so no date rule
    can admit them. Rule (fixed before any v2 scoring): same-organism training pairs at >= 80% identity form
    a graph over training proteins outside the dated TV; in every connected component of size >= 2 one
    member, drawn at random (seed 0), is held out; at most 2,000 components per organism group (random,
    seed 0). The other members stay in the database with their T0 labels; the held-out accession's ground
    truth is its full GO-GPT label set at the cutoff, as for a test protein."""
    from scipy.sparse.csgraph import connected_components
    org = df_tr.set_index("id").organism.fillna("?")
    nodes = np.array(sorted(set(df_tr.id) - exclude))
    pos = {p: i for i, p in enumerate(nodes)}
    h = hits[(hits.pident >= DA_MIN_PIDENT) & hits.q.isin(pos) & hits.s.isin(pos)]
    h = h[h.q.map(org).values == h.s.map(org).values]
    A = sp.coo_matrix((np.ones(len(h)), (h.q.map(pos).values, h.s.map(pos).values)), shape=(len(nodes),) * 2)
    _, comp = connected_components(A, directed=False)
    size = np.bincount(comp)
    rng = np.random.default_rng(DA_SEED)
    c = pd.DataFrame({"id": nodes, "comp": comp, "size": size[comp]})
    c = c[c["size"] >= 2]
    c["group"] = org_group(c.id.map(org)).values
    pick = c.groupby("comp").id.agg(lambda s: s.values[rng.integers(len(s))])
    out = pd.DataFrame({"comp": pick.index, "id": pick.values})
    out["group"] = org_group(out.id.map(org)).values
    out["comp_size"] = out.comp.map(c.groupby("comp").size()).values
    keep = []
    for gname, d in out.groupby("group", sort=True):
        keep.append(d if len(d) <= DA_CAP else d.iloc[np.sort(rng.choice(len(d), DA_CAP, replace=False))])
    return pd.concat(keep, ignore_index=True)


def best_hit_cells(hits: pd.DataFrame, ids, org: pd.Series) -> pd.DataFrame:
    """[id, max_pident, bin, best_same_org] for query ids; best hit = top bitscore, bin on max identity."""
    h = hits[hits.q.isin(set(ids))]
    mx = h.groupby("q").pident.max()
    top = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates("q").set_index("q").s
    ids = pd.Index(ids)
    out = pd.DataFrame({"id": ids, "max_pident": mx.reindex(ids).fillna(0.0).values})
    out["bin"] = ident_bins(out.max_pident.values).values
    out["best_same_org"] = (top.reindex(ids).map(org).values == org.reindex(ids).values) & out.max_pident.gt(0).values
    out["org_group"] = org_group(org.reindex(ids)).values
    return out


def main_v2():
    """TV v2: the strict dated rule of v1 on a wider dated GOA (zebrafish, whole-taxon rat) plus the DA
    stratum. Writes data/proc_gogpt_tv2 in the v1 layout; v1 is untouched."""
    OUT_V2.mkdir(parents=True, exist_ok=True)
    log("== P2 TV v2 build")
    go = GO()
    df = pd.read_parquet(SRC / "train.parquet")
    tr_mask = (df.split == "train").values
    train_ids = set(df.id[tr_mask])
    labels = {p: s for p, s in full_labels(go, "train").items() if p in train_ids}
    g, gstats = dated_goa_v2(train_ids)
    log(f"dated experimental+HTP rows on GO-GPT training proteins: {gstats}")
    stripped, tv, stats = build_split(go, labels, train_ids, g)
    tv_asp = tv.groupby("id").aspect.agg(set).to_dict()
    lab_asp = {p: {go.aspect.get(t) for t in labels[p]} & set(ASPECTS) for p in tv_asp}
    kind = {p: "dated_NK" if lab_asp[p] <= tv_asp[p] else "dated_LK" for p in tv_asp}

    hits = pd.read_parquet(TRAIN_HITS)
    df_tr = df[tr_mask][["id", "organism"]]
    da = da_stratum(df_tr, hits, set(kind))
    roots = set(ASPECT_ROOT.values())
    da_rows = []
    for p in da.id:
        for a in ASPECTS:
            L = {t for t in labels[p] if go.aspect.get(t) == a}
            if L - roots:
                da_rows += [(p, t, a, "da") for t in L]
    da_strip = pd.DataFrame(da_rows, columns=["id", "term", "aspect", "kind"])
    for p in da.id:
        kind[p] = "DA"
    stripped = pd.concat([stripped, da_strip], ignore_index=True)
    stripped.to_parquet(OUT_V2 / "t0_stripped.parquet", index=False)
    tv_ids = sorted(kind)
    (OUT_V2 / "tv_ids.txt").write_text("\n".join(tv_ids) + "\n")
    gt = stripped[stripped.kind.isin(["tv", "da"])][["id", "term"]].drop_duplicates().sort_values(["id", "term"])
    gt.to_csv(OUT_V2 / "gt_tv.tsv", sep="\t", header=False, index=False)
    kc = Counter(kind.values())
    log(f"TV v2 proteins {len(tv_ids):,} {dict(kc)}; dated aspects {len(tv):,}; GT pairs {len(gt):,}; "
        f"stripped {stripped.kind.value_counts().to_dict()}")

    vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
    tidx = {t: i for i, t in enumerate(vocab.term)}
    Y = sp.load_npz(SRC / "Y.npz").tocsr().astype(np.float32)
    row_of = {p: i for i, p in zip(np.where(tr_mask)[0], df.id.values[tr_mask])}
    m = stripped.term.isin(tidx).values
    D = sp.csr_matrix((np.ones(m.sum(), np.float32), (stripped.id[m].map(row_of).values,
                                                     stripped.term[m].map(tidx).values)), shape=Y.shape)
    Y0 = Y - Y.multiply(D > 0)
    Y0.eliminate_zeros()
    sp.save_npz(OUT_V2 / "Y.npz", Y0.tocsr())
    out_df = df.copy()
    out_df["tv"] = tr_mask & out_df.id.isin(set(tv_ids)).values
    out_df["tv_stratum"] = np.where(out_df.tv, out_df.id.map(kind).fillna(""), "")
    moved = out_df.tv & out_df.tv_stratum.isin(["dated_NK", "DA"])
    out_df.loc[moved, "split"] = "tv"
    out_df.to_parquet(OUT_V2 / "train.parquet", index=False)
    log(f"Y.npz nnz {Y.nnz:,} -> {Y0.nnz:,}; moved to split tv: {int(moved.sum()):,}")
    for f in SYMLINKS:
        if not (OUT_V2 / f).exists():
            os.symlink(os.path.relpath(SRC / f, OUT_V2), OUT_V2 / f)
    (OUT_V2 / "knn").mkdir(exist_ok=True)
    if not (OUT_V2 / "knn" / "train_hits.parquet").exists():
        os.symlink(os.path.relpath(TRAIN_HITS, OUT_V2 / "knn"), OUT_V2 / "knn" / "train_hits.parquet")

    # TV queries keep their train-vs-train hits against the v2 database (split == "train" in v2)
    db = set(out_df.id[out_df.split == "train"])
    th = hits[hits.q.isin(set(tv_ids)) & hits.s.isin(db)].reset_index(drop=True)
    th.to_parquet(OUT_V2 / "diamond_hits_tv.parquet", index=False)
    org = df.drop_duplicates("id").set_index("id").organism
    cells_tv = best_hit_cells(th, tv_ids, org).assign(split="tv")
    cells_tv["stratum"] = cells_tv.id.map(kind).values
    vt = pd.read_parquet(SRC / "diamond_hits_valtest.parquet")
    cells = [cells_tv]
    for s in ("val", "test"):
        cells.append(best_hit_cells(vt, df.id[df.split == s].values, org).assign(split=s, stratum=""))
    cells = pd.concat(cells, ignore_index=True)
    cells.to_parquet(OUT_V2 / "cells.parquet", index=False)
    ident = pd.read_parquet(SRC / "identity.parquet")
    ident = pd.concat([ident, cells_tv[["id", "split", "max_pident"]]], ignore_index=True)
    ident.to_parquet(OUT_V2 / "identity.parquet", index=False)

    v1 = best_hit_cells(pd.read_parquet(OUT / "diamond_hits_tv.parquet"), open(OUT / "tv_ids.txt").read().split(), org)
    order = ["100", "95-99.9", "80-95", "50-80", "30-50", "<30", "no_hit"]

    def comp(c):
        return {"n": int(len(c)),
                "organism_group": {k: round(float(v), 4) for k, v in c.org_group.value_counts(normalize=True).items()},
                "identity_bins": {b: round(float((c.bin == b).mean()), 4) for b in order},
                "same_org_best_hit_by_bin": {b: (round(float(c.best_same_org[c.bin == b].mean()), 3)
                                                 if (c.bin == b).any() else None) for b in order}}
    rep = {"T0": T0, "cutoff": CUTOFF, "evidence_codes": list(EXP_CODES + HTP_CODES),
           "dated_sources": gstats, "aspect_stats": stats, "strata": dict(kc),
           "da_rule": {"min_pident": DA_MIN_PIDENT, "cap_per_group": DA_CAP, "seed": DA_SEED,
                       "per_group": da.group.value_counts().to_dict(),
                       "component_size_median": float(da.comp_size.median())},
           "gt_pairs": int(len(gt)), "Y_nnz_before_after": [int(Y.nnz), int(Y0.nnz)],
           "tv2": comp(cells_tv), "tv2_dated": comp(cells_tv[cells_tv.stratum != "DA"]),
           "tv2_DA": comp(cells_tv[cells_tv.stratum == "DA"]), "tv1": comp(v1),
           "test": comp(cells[cells.split == "test"]), "val": comp(cells[cells.split == "val"]),
           "diamond_hits_tv": {"n_hits": int(len(th)), "queries_with_hit": int(th.q.nunique())}}
    (OUT_V2 / "tv_report.json").write_text(json.dumps(rep, indent=2))
    log(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--v2", action="store_true", help="build data/proc_gogpt_tv2")
    args = ap.parse_args()
    if args.v2:
        return main_v2()
    OUT.mkdir(parents=True, exist_ok=True)
    log("== P2 temporal validation build")
    go = GO()
    df = pd.read_parquet(SRC / "train.parquet")
    tr_mask = (df.split == "train").values
    train_ids = set(df.id[tr_mask])
    labels = full_labels(go, "train")
    labels = {p: s for p, s in labels.items() if p in train_ids}
    log(f"GO-GPT training proteins with full labels: {len(labels):,}")

    stripped, tv, stats = build_split(go, labels, train_ids)
    stripped.to_parquet(OUT / "t0_stripped.parquet", index=False)
    tv_ids = sorted(tv.id.unique())
    tv_set = set(tv_ids)
    # NK when every labelled aspect of the protein is a TV aspect
    tv_asp = tv.groupby("id").aspect.agg(set).to_dict()
    lab_asp = {p: {go.aspect.get(t) for t in labels[p]} & set(ASPECTS) for p in tv_ids}
    kind = {p: "NK" if lab_asp[p] <= tv_asp[p] else "LK" for p in tv_ids}
    (OUT / "tv_ids.txt").write_text("\n".join(tv_ids) + "\n")
    gt = stripped[stripped.kind == "tv"][["id", "term"]].sort_values(["id", "term"])
    gt.to_csv(OUT / "gt_tv.tsv", sep="\t", header=False, index=False)
    log(f"TV proteins {len(tv_ids):,} (NK {sum(v == 'NK' for v in kind.values()):,}, "
        f"LK {sum(v == 'LK' for v in kind.values()):,}); TV aspects {len(tv):,}; GT pairs {len(gt):,}; "
        f"stripped pairs {len(stripped):,} ({stripped.kind.value_counts().to_dict()})")

    # T0 label matrix over the unchanged vocabulary
    vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
    tidx = {t: i for i, t in enumerate(vocab.term)}
    Y = sp.load_npz(SRC / "Y.npz").tocsr().astype(np.float32)
    row_of = {p: i for i, p in zip(np.where(tr_mask)[0], df.id.values[tr_mask])}
    rr = [row_of[p] for p, t in zip(stripped.id, stripped.term) if t in tidx]
    cc = [tidx[t] for t in stripped.term if t in tidx]
    D = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=Y.shape)
    D.data[:] = 1.0
    Y0 = Y - Y.multiply(D > 0)
    Y0.eliminate_zeros()
    sp.save_npz(OUT / "Y.npz", Y0.tocsr())
    log(f"Y.npz: nnz {Y.nnz:,} -> {Y0.nnz:,}")

    out_df = df.copy()
    out_df["tv"] = tr_mask & out_df.id.isin(tv_set).values
    nk_rows = out_df.tv.values & out_df.id.map(lambda p: kind.get(p) == "NK").values
    out_df.loc[nk_rows, "split"] = "tv"
    out_df.to_parquet(OUT / "train.parquet", index=False)
    for f in SYMLINKS:
        dst = OUT / f
        if not dst.exists():
            os.symlink(os.path.relpath(SRC / f, OUT), dst)

    # DIAMOND: TV queries against the full GO-GPT training split (TV proteins included, self excluded)
    hits_path = OUT / "diamond_hits_tv.parquet"
    if not hits_path.exists():
        t = time.time()
        q = df[tr_mask & df.id.isin(tv_set).values][["id", "seq"]]
        hits = run_diamond(df[tr_mask][["id", "seq"]], q, args.threads)
        hits.to_parquet(hits_path, index=False)
        log(f"DIAMOND TV vs train: {len(hits):,} hits for {hits.q.nunique():,}/{len(q):,} queries "
            f"in {time.time() - t:.0f}s")
    hits = pd.read_parquet(hits_path)
    mx = hits.groupby("q").pident.max()
    ident = pd.read_parquet(SRC / "identity.parquet")
    ident_tv = pd.DataFrame({"id": tv_ids, "split": "tv", "max_pident": [float(mx.get(p, 0.0)) for p in tv_ids]})
    ident = pd.concat([ident, ident_tv], ignore_index=True)
    ident.to_parquet(OUT / "identity.parquet", index=False)

    # report: TV vs test composition
    org = df[tr_mask].set_index("id").organism
    te = df[df.split == "test"]
    def mix(s):
        v = s.fillna("?").value_counts(normalize=True)
        return {k: round(float(x), 4) for k, x in v.head(12).items()}
    def bins(x):
        v = ident_bins(pd.Series(x)).value_counts(normalize=True)
        return {k: round(float(v.get(k, 0.0)), 4) for k in ["100", "95-99.9", "80-95", "50-80", "30-50", "<30", "no_hit"]}
    gt_asp = gt.term.map(lambda t: go.aspect.get(t))
    n_per_asp = gt.assign(a=gt_asp).groupby("a").id.nunique().to_dict()
    test_orgs = set(te.organism.dropna())
    rep = {
        "T0": T0, "cutoff": CUTOFF, "evidence_codes": list(EXP_CODES + HTP_CODES),
        "evidence_note": "GO-GPT training labels contain 98.8% of GOA HDA annotations dated <= cutoff and the "
                         "HTP-only terms of 16,666/16,821 proteins, so high-throughput codes are included.",
        "n_tv_proteins": len(tv_ids), "n_nk": sum(v == "NK" for v in kind.values()),
        "n_lk": sum(v == "LK" for v in kind.values()), "n_tv_aspects": len(tv),
        "tv_proteins_per_aspect": {a: int(n_per_asp.get(a, 0)) for a in ASPECTS},
        "gt_pairs": len(gt), "mean_gt_terms_per_protein_aspect": round(len(gt) / max(len(tv), 1), 2),
        "stripped_pairs": stripped.kind.value_counts().to_dict(),
        "aspect_stats": stats,
        "organism_mix_tv": mix(org.reindex(tv_ids)), "organism_mix_test": mix(te.organism),
        "tv_share_in_test_organisms": round(float(org.reindex(tv_ids).isin(test_orgs).mean()), 4),
        "identity_bins_tv": bins(ident_tv.max_pident.values),
        "identity_bins_test": bins(ident[ident.split == "test"].max_pident.values),
        "identity_bins_val": bins(ident[ident.split == "val"].max_pident.values),
        "zebrafish_gap": "goa_reference.parquet has no Danio rerio; zebrafish is "
                         f"{float((te.organism.fillna('').str.startswith('Danio')).mean()):.1%} of the test and 0% of TV. "
                         "A dated ZFIN GAF would close the gap.",
        "train_split_changes": {"nk_moved_to_split_tv": int(nk_rows.sum()),
                                "lk_kept_in_train_with_t0_labels": int((out_df.tv & (out_df.split == "train")).sum())},
        "Y_nnz_before_after": [int(Y.nnz), int(Y0.nnz)],
        "diamond_hits_tv": {"n_hits": int(len(hits)), "queries_with_hit": int(hits.q.nunique())},
    }
    (OUT / "tv_report.json").write_text(json.dumps(rep, indent=2))
    log(json.dumps(rep, indent=2))
    (ROOT / "logs" / "tv.done").touch()


if __name__ == "__main__":
    main()
