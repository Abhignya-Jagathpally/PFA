"""Sharpened kNN prior with GO-GPT's validation proteins added to the reference set (train + val).

The L0 prior (knn_sharp, T5_k50_g0_na) copies labels from GO-GPT training proteins only. GO-GPT's validation
split (13,305 proteins) is a random 10% of the same November 2022 snapshot, so its labels are known at the
cutoff too. This module measures whether adding those proteins as references would help.

DIAMOND: test and TV v2 queries against the val proteins with the settings of temporal_val.run_diamond
(--more-sensitive, e <= 1e-3, -k 51, self accessions removed, best HSP per pair, top 50 by bitscore) and
--dbsize set to the letters of the 119,749-protein training database, so e-values (and the 1e-3 cut) match
the existing train hits. The val-reference hits are merged with the existing hits and the prior keeps the top
k by bitscore over the union.

Label databases: test queries copy full labels from train and val. TV v2 queries copy T0 labels from the v2
database (train minus TV) and from val proteins, whose labels are stripped to T0 with the TV v2 dated GOA
rule (temporal_val.dated_goa_v2 + build_split, both "tv" and "pk" pairs removed), as for training references.

Stages (each writes to runs/knn_trainval/ and logs/knn_trainval.log):
  --diamond   val-reference hits for test + TV v2 queries (runs/knn_trainval/hits_valref.parquet)
  --stats     label-free best-hit identity changes for test and TV v2 queries
  --validity  accession / sequence overlap checks, val T0 stripping counts
  --register  appends the pre-registered keep rule to runs/test_looks.md (no test look)
  --tv        scores baseline vs train+val on TV v2 (test-like v2 weights), applies the rule
  --write     writes data/proc_gogpt_tv2/knn_trainval/{tv,test}_knn.npz and the merged hit tables (only if kept)

Usage: python -m cafa6.knn_trainval --diamond --stats --validity [--threads 24]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cafa6.knn_sharp as ks
import cafa6.temporal_val as tvm
from cafa6.data import ROOT
from cafa6.temporal_val import DIAMOND, GO, OUT_V2, SRC, best_hit_cells, full_labels, ident_bins, t0_labels

RUN = ROOT / "runs" / "knn_trainval"
OUT = OUT_V2 / "knn_trainval"
LOG = ROOT / "logs" / "knn_trainval.log"
HITS_VAL = RUN / "hits_valref.parquet"
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
RULE = ("'kNN reference set = train + val' (GO-GPT validation proteins added as references to the sharpened kNN "
        "prior T5_k50_g0_na; TV v2 queries copy val labels stripped to T0 by the TV v2 dated GOA rule; DIAMOND "
        "--more-sensitive e<=1e-3 top 50 by bitscore over the merged train+val hits) is KEPT if its TV v2 "
        "test-like mean Fmax (chosen_v2 cell weights, knn_sharp.Scorer) minus the train-only prior's is >= 0.003, "
        "the paired protein bootstrap 95% CI of that difference (1000 reps, seed 0, test-like weights) excludes 0, "
        "and the plain (unweighted) TV v2 mean Fmax is not lower by more than 0.002. Weights recomputed from "
        "train+val cells are a diagnostic only. Keeping it writes data/proc_gogpt_tv2/knn_trainval/{tv,test}_knn.npz "
        "but does not by itself trigger a test look.")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


tvm.log = log  # build_split logs through the module global; keep its messages in this run's log


def source() -> pd.DataFrame:
    return pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split", "organism"])


def tv_ids() -> list[str]:
    return open(OUT_V2 / "tv_ids.txt").read().split()


def organisms(df: pd.DataFrame) -> pd.Series:
    return df.drop_duplicates("id").set_index("id").organism


# ------------------------------------------------------------------------------------------------ DIAMOND
def run_diamond(threads: int, k: int = 50):
    RUN.mkdir(parents=True, exist_ok=True)
    df = source()
    db = df[df.split == "val"]
    tv = set(tv_ids())
    q = pd.concat([df[df.split == "test"], df[(df.split == "train") & df.id.isin(tv)]]).drop_duplicates("id")
    dbsize = int(df.seq[df.split == "train"].str.len().sum())
    log(f"== DIAMOND val references: {len(q):,} queries (test {int((q.split == 'test').sum()):,}, TV v2 "
        f"{int((q.split == 'train').sum()):,}) vs {len(db):,} val proteins, --dbsize {dbsize:,} (train letters), "
        f"{threads} threads")
    t = time.time()
    with tempfile.TemporaryDirectory(dir=RUN) as tmp:
        tmp = Path(tmp)
        for d, name in ((db, "db.fa"), (q, "q.fa")):
            with open(tmp / name, "w") as f:
                for pid, s in zip(d.id, d.seq):
                    f.write(f">{pid}\n{s}\n")
        subprocess.run([str(DIAMOND), "makedb", "--in", str(tmp / "db.fa"), "-d", str(tmp / "db"), "--quiet"],
                       check=True)
        subprocess.run([str(DIAMOND), "blastp", "-d", str(tmp / "db"), "-q", str(tmp / "q.fa"),
                        "-o", str(tmp / "hits.tsv"), "--more-sensitive", "-e", "1e-3", "-k", str(k + 1),
                        "-p", str(threads), "--quiet", "--tmpdir", str(tmp), "--dbsize", str(dbsize),
                        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)
        h = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None, names=["q", "s", "pident", "bits", "evalue", "qcov"])
    n_self = int((h.q == h.s).sum())
    h = h[h.q != h.s]
    h = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
    h = h.groupby("q", sort=False).head(k).reset_index(drop=True)
    h.to_parquet(HITS_VAL, index=False)
    log(f"val-reference hits: {len(h):,} for {h.q.nunique():,}/{len(q):,} queries; self-accession hits removed "
        f"{n_self:,}; {time.time() - t:.0f}s -> {HITS_VAL}")


def merged_hits(split: str) -> pd.DataFrame:
    """Existing hits of the split's queries plus their val-reference hits."""
    base = pd.read_parquet(OUT_V2 / ("diamond_hits_tv.parquet" if split == "tv" else "diamond_hits_valtest.parquet"))
    ids = set(tv_ids()) if split == "tv" else set(source().query("split == 'test'").id)
    base = base[base.q.isin(ids)]
    hv = pd.read_parquet(HITS_VAL)
    hv = hv[hv.q.isin(ids)]
    cols = ["q", "s", "pident", "bits", "evalue", "qcov"]
    return pd.concat([base[cols], hv[cols]], ignore_index=True)


# ------------------------------------------------------------------------------------------------ stats
def transition(before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.Series]:
    b = before.set_index("id")
    a = after.set_index("id").reindex(b.index)
    tab = pd.crosstab(pd.Categorical(b.bin, BINS), pd.Categorical(a.bin, BINS), dropna=False)
    tab.index.name, tab.columns.name = "before", "after"
    imp = a.max_pident > b.max_pident + 1e-9
    n = len(b)
    summ = {"n": n, "improved": int(imp.sum()), "improved_share": round(float(imp.mean()), 4),
            "bin_changed": int((a.bin != b.bin).sum()),
            "into_ge95": int(((b.max_pident < 95) & (a.max_pident >= 95)).sum()),
            "into_100": int(((b.max_pident < 100) & (a.max_pident >= 100)).sum()),
            "ge95_before": int((b.max_pident >= 95).sum()), "ge95_after": int((a.max_pident >= 95).sum()),
            "100_before": int((b.max_pident >= 100).sum()), "100_after": int((a.max_pident >= 100).sum()),
            "no_hit_before": int((b.max_pident <= 0).sum()), "no_hit_after": int((a.max_pident <= 0).sum()),
            "best_same_org_before": int(b.best_same_org.sum()), "best_same_org_after": int(a.best_same_org.sum())}
    return tab, summ, imp


def run_stats():
    log("== label-free identity statistics (DIAMOND hits only)")
    df = source()
    org = organisms(df)
    val_ids = set(df.id[df.split == "val"])
    hv = pd.read_parquet(HITS_VAL)
    out = {}
    for split in ("test", "tv"):
        ids = tv_ids() if split == "tv" else df.id[df.split == "test"].tolist()
        base = merged_hits(split)
        base = base[~base.s.isin(val_ids)]
        before = best_hit_cells(base, ids, org)
        after = best_hit_cells(merged_hits(split), ids, org)
        tab, summ, imp = transition(before, after)
        # same organism: the val hit that sets the new best identity (top bitscore among the val hits at that identity)
        h = hv[hv.q.isin(set(np.asarray(ids)[imp.values]))]
        top = h.sort_values(["q", "pident", "bits"], ascending=[True, False, False]).drop_duplicates("q")
        same = top.q.map(org).values == top.s.map(org).values
        top = top.assign(same_org=same, after_bin=ident_bins(top.pident.values).values)
        so = top.set_index("q").same_org
        b, a = before.set_index("id").max_pident, after.set_index("id").max_pident
        summ["improved_by_same_org_val_hit"] = int(same.sum())
        summ["into_ge95_same_org"] = int(so.reindex(b.index[(b < 95) & (a >= 95)]).fillna(False).sum())
        summ["into_100_same_org"] = int(so.reindex(b.index[(b < 100) & (a >= 100)]).fillna(False).sum())
        summ["queries_with_val_hit"] = int(hv[hv.q.isin(set(ids))].q.nunique())
        summ["improved_same_org_by_after_bin"] = top.groupby("after_bin").same_org.agg(["size", "sum"]).reindex(
            BINS).fillna(0).astype(int).rename(columns={"size": "n", "sum": "same_org"}).to_dict("index")
        tab.to_csv(RUN / f"stats_{split}_transition.tsv", sep="\t")
        before.merge(after, on="id", suffixes=("_before", "_after")).to_parquet(RUN / f"stats_{split}_cells.parquet",
                                                                                  index=False)
        marg = pd.DataFrame({"before": tab.sum(axis=1), "after": tab.sum(axis=0)})
        marg["before_share"] = (marg.before / len(before)).round(4)
        marg["after_share"] = (marg.after / len(before)).round(4)
        imp_by_bin = pd.Series(imp.values, index=before.bin.values).groupby(level=0).agg(["size", "sum"]).reindex(BINS)
        marg["improved_from_bin"] = imp_by_bin["sum"].fillna(0).astype(int).values
        marg.to_csv(RUN / f"stats_{split}_bins.tsv", sep="\t")
        out[split] = summ
        log(f"{split}: bins before/after and number improved per before-bin\n{marg.to_string()}\n"
            f"transition (rows before, cols after)\n{tab.to_string()}\n{json.dumps(summ, indent=1)}")
    (RUN / "stats.json").write_text(json.dumps(out, indent=2))


# ------------------------------------------------------------------------------------------------ validity
def val_t0_labels(go: GO, val_full: dict) -> tuple[dict, dict]:
    vids = set(val_full)
    g, gstats = tvm.dated_goa_v2(vids)
    stripped, tv_rows, stats = tvm.build_split(go, val_full, vids, g)
    drop: dict[str, set] = {}
    for p, t in zip(stripped.id, stripped.term):
        drop.setdefault(p, set()).add(t)
    lab = {p: (s - drop[p]) if p in drop else s for p, s in val_full.items()}
    rep = {"dated_rows": gstats, "aspect_stats": stats, "stripped_pairs": stripped.kind.value_counts().to_dict(),
           "val_proteins_with_stripped_terms": int(stripped.id.nunique()),
           "val_proteins_left_without_labels": int(sum(1 for p in lab if not lab[p] and val_full[p]))}
    return lab, rep


def run_validity():
    log("== validity checks")
    df = source()
    s = {k: df[df.split == k] for k in ("train", "val", "test")}
    tvset = set(tv_ids())
    vid, tid = set(s["val"].id), set(s["test"].id)
    vseq = set(s["val"].seq)
    rep = {"label_snapshot": "GO-GPT val and train both come from wanglab/gogpt-training-data (validation and train "
                             "parquet of the same release, annotations up to Nov 2022, random 90/10 split); the test "
                             "set holds annotations gained Mar 2023 - Feb 2024 (prepare_gogpt.py, REPORT.md 4.1).",
           "val_test_accession_overlap": len(vid & tid),
           "val_test_sequence_overlap_test_proteins": int(s["test"].seq.isin(vseq).sum()),
           "val_test_sequence_overlap_same_accession": int(s["test"][s["test"].seq.isin(vseq) & s["test"].id.isin(vid)].shape[0]),
           "train_test_accession_overlap": len(set(s["train"].id) & tid),
           "train_test_sequence_overlap_test_proteins": int(s["test"].seq.isin(set(s["train"].seq)).sum()),
           "val_tv_accession_overlap": len(vid & tvset),
           "val_tv_sequence_overlap_tv_proteins": int(s["train"][s["train"].id.isin(tvset)].seq.isin(vseq).sum())}
    hv = pd.read_parquet(HITS_VAL)
    rep["val_hits_with_self_accession"] = int((hv.q == hv.s).sum())
    rep["val_hits_of_overlapping_test_accessions"] = int(hv.q.isin(vid & tid).sum())
    # labels stored for the 87 shared accessions: GO-GPT's val labels are pre-cutoff by construction; self hits are
    # dropped, so these proteins are never their own reference (same convention as the 673 shared with train)
    go = GO()
    vfull = full_labels(go, "val")
    vfull = {p: v for p, v in vfull.items() if p in vid}
    _, t0rep = val_t0_labels(go, vfull)
    rep["val_t0_stripping"] = t0rep
    (RUN / "validity.json").write_text(json.dumps(rep, indent=2, default=str))
    log("validity: " + json.dumps(rep, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ TV v2
def register():
    with open(ks.TEST_LOOKS, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | kNN train+val references (knn_trainval.py) | "
                f"PRE-REGISTERED RULE before any TV v2 score (no test look): {RULE}\n")
    log("registered: " + RULE)


def make_knn(onto, split: str, db_ids: np.ndarray, L, hits: pd.DataFrame) -> ks.KNN:
    orig = ks.load_hits
    ks.load_hits = lambda s: hits
    try:
        return ks.KNN(onto, split, db_ids, L)
    finally:
        ks.load_hits = orig


def priors(onto, go, splits=("tv",)) -> dict:
    """{split: (baseline KNN, train+val KNN)} for the chosen v2 setting."""
    ks.use_tv("v2")
    df = source()
    val_ids = df.id.values[(df.split == "val").values]
    vfull = {p: v for p, v in full_labels(go, "val").items() if p in set(val_ids)}
    out = {}
    base = ks.build_knn(onto, go, splits)
    for s in splits:
        kb = base[s]
        if s == "tv":
            tdf = pd.read_parquet(OUT_V2 / "train.parquet", columns=["id", "split"])
            tr_ids = tdf.id.values[(tdf.split == "train").values]
            t0 = t0_labels(go, full_labels(go, "train"), out=OUT_V2)
            vlab, _ = val_t0_labels(go, vfull)
            labs = [t0.get(p, set()) for p in tr_ids] + [vlab.get(p, set()) for p in val_ids]
        else:
            tr_ids = df.id.values[(df.split == "train").values]
            tfull = full_labels(go, "train")
            labs = [tfull.get(p, set()) for p in tr_ids] + [vfull.get(p, set()) for p in val_ids]
        db = np.concatenate([tr_ids, val_ids])
        out[s] = (kb, make_knn(onto, s, db, onto.label_matrix(labs), merged_hits(s)))
    return out


def run_tv():
    ch = json.loads((ks.RUN / "chosen_v2.json").read_text())
    st = (ch["T"], ch["k"], ch["gamma"], ch["norm"])
    log(f"== TV v2: {ch['setting']} train-only vs train+val references")
    onto, go = ks.Ontology(), GO()
    kb, kn = priors(onto, go, ("tv",))["tv"]
    sc = ks.Scorer(onto, ks.gt_frame("tv"))
    cells = pd.read_parquet(OUT_V2 / "cells.parquet")
    w, wrep = ks.cell_weights(cells, sc.ids)
    Pb, _ = kb.prior(*st)
    Pn, _ = kn.prior(*st)
    # the baseline must reproduce the stored contract prior
    ref = ks.load_knn("tv", OUT_V2 / "knn")
    assert list(ref["ids"]) == list(kb.ids)
    vocab = pd.read_csv(SRC / "vocab.tsv", sep="\t")
    vcols = np.array([onto.idx[t] for t in vocab.term])
    dmax = float(abs(Pb[:, vcols].tocsr().astype(np.float32) - ref["p"]).max())
    log(f"baseline vs data/proc_gogpt_tv2/knn/tv_knn.npz: max abs diff {dmax:.2e}")
    cb, cn = sc.curves(Pb, kb.ids, closed=True), sc.curves(Pn, kn.ids, closed=True)
    fb, fn = sc.fmax(cb), sc.fmax(cn)
    fbw, fnw = sc.fmax(cb, weights=w), sc.fmax(cn, weights=w)
    bs = sc.bootstrap(cn, cb, reps=1000, seed=0, weights=w)
    bs_plain = sc.bootstrap(cn, cb, reps=1000, seed=0)
    # diagnostic: weights from train+val cells for both TV and test
    st_tv = pd.read_parquet(RUN / "stats_tv_cells.parquet")
    st_te = pd.read_parquet(RUN / "stats_test_cells.parquet")
    strata = cells[cells.split == "tv"].set_index("id").stratum
    def after_cells(d, split):
        c = d[["id", "max_pident_after", "bin_after", "best_same_org_after", "org_group_after"]].copy()
        c.columns = ["id", "max_pident", "bin", "best_same_org", "org_group"]
        return c.assign(split=split, stratum=c.id.map(strata).fillna("") if split == "tv" else "")
    w2, w2rep = ks.cell_weights(pd.concat([after_cells(st_tv, "tv"), after_cells(st_te, "test")]), sc.ids)
    bs2 = sc.bootstrap(cn, cb, reps=1000, seed=0, weights=w2)
    ctv = cells[cells.split == "tv"].drop_duplicates("id").set_index("id").reindex(sc.ids)
    changed = set(st_tv.id[st_tv.max_pident_after > st_tv.max_pident_before + 1e-9])
    sub = {"DA": sc.ids[(ctv.stratum == "DA").values], "dated": sc.ids[(ctv.stratum != "DA").values],
           "best_identity_improved": [p for p in sc.ids if p in changed],
           "best_identity_unchanged": [p for p in sc.ids if p not in changed]}
    rows = [{"prior": name, **{a: f[a] for a in ("MF", "BP", "CC")}, "mean": f["mean"], "mean_testlike_v2": fw["mean"],
             "mean_testlike_trainval_cells": sc.fmax(c, weights=w2)["mean"],
             **{f"mean_{g}": sc.fmax(c, subset=ids)["mean"] for g, ids in sub.items()},
             **{f"bin_{b}": v for b, v in ks.bin_table(sc, c, ctv.max_pident).items() if b != "all"}}
            for name, f, fw, c in (("train_only", fb, fbw, cb), ("train_val", fn, fnw, cn))]
    tab = pd.DataFrame(rows)
    tab.to_csv(RUN / "tv_summary.tsv", sep="\t", index=False)
    gain = bs["diff"]
    plain_drop = fb["mean"] - fn["mean"]
    keep = bool(gain >= 0.003 and bs["lo"] > 0 and plain_drop <= 0.002)
    res = {"setting": ch["setting"], "rule": RULE, "baseline_matches_contract_max_abs_diff": dmax,
           "testlike_v2": {"train_only": fbw["mean"], "train_val": fnw["mean"], "bootstrap": bs},
           "plain": {"train_only": fb["mean"], "train_val": fn["mean"], "bootstrap": bs_plain},
           "diagnostic_testlike_trainval_cells": {"bootstrap": bs2, "weights": {k: w2rep[k] for k in
                                                                                 ("n_cells", "effective_n", "test_mass_without_tv_cell")}},
           "n_tv_scored": int(sc.n), "n_tv_best_identity_improved": len(sub["best_identity_improved"]),
           "criteria": {"gain_ge_0.003": gain >= 0.003, "ci_excludes_0": bs["lo"] > 0,
                        "plain_not_lower_by_gt_0.002": plain_drop <= 0.002},
           "kept": keep, "scored_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RUN / "tv_result.json").write_text(json.dumps(res, indent=2, default=float))
    log("TV v2 result:\n" + tab.round(4).T.to_string() + "\n" + json.dumps(res, indent=1, default=float))
    log(f"VERDICT: {'KEPT' if keep else 'NOT KEPT'} (test-like gain {gain:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}], "
        f"plain {fn['mean'] - fb['mean']:+.4f})")
    with open(ks.TEST_LOOKS, "a") as f:
        f.write(f"- {res['scored_at']} | kNN train+val references | TV v2 result (no test look): test-like gain "
                f"{gain:+.4f} [{bs['lo']:+.4f}, {bs['hi']:+.4f}], plain {fn['mean'] - fb['mean']:+.4f} -> "
                f"{'KEPT' if keep else 'NOT KEPT'}; runs/knn_trainval/tv_result.json\n")


def run_write():
    res = json.loads((RUN / "tv_result.json").read_text())
    if not res["kept"]:
        log("rule not met: no knn_trainval contract files written")
        return
    ch = json.loads((ks.RUN / "chosen_v2.json").read_text())
    st = (ch["T"], ch["k"], ch["gamma"], ch["norm"])
    onto, go = ks.Ontology(), GO()
    OUT.mkdir(parents=True, exist_ok=True)
    for s, (_, kn) in priors(onto, go, ("tv", "test")).items():
        P, nh = kn.prior(*st)
        hf = f"diamond_hits_{s}_trainval.parquet"
        merged_hits(s).to_parquet(OUT / hf, index=False)
        ks.save_prior(s, kn, P, nh, onto, {
            "T": st[0], "k": st[1], "gamma": st[2], "norm": st[3], "setting": ch["setting"], "tv_version": "v2",
            "reference_set": "train + val",
            "label_db": ("T0 labels of the v2 database + val labels stripped to T0 (TV v2 dated GOA rule)" if s == "tv"
                         else "full GO-GPT training + validation labels"),
            "hits": f"knn_trainval/{hf}", "note": "kept by the knn_trainval TV v2 rule; not scored on test"}, OUT)
        log(f"wrote {OUT / f'{s}_knn.npz'} ({len(kn.ids):,} queries, {int(kn.has_hit.sum()):,} with a hit) and {OUT / hf}")


def main():
    ap = argparse.ArgumentParser()
    for s in ("diamond", "stats", "validity", "register", "tv", "write"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.diamond:
        run_diamond(min(a.threads, 24))
    if a.stats:
        run_stats()
    if a.validity:
        run_validity()
    if a.register:
        register()
    if a.tv:
        run_tv()
    if a.write:
        run_write()


if __name__ == "__main__":
    main()
