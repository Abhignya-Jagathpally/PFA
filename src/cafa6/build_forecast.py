"""Annotation-forecast training data: (sequence, labels of the protein and its homologues at time T) ->
(GO terms the protein gained after T).

Periods
-------
A  input T = GO-GPT snapshot S0 (labels of November 2022, S0_DATE), targets from GOA UniProt release 219
   (2024-02-09, data/fresh24/source/exp_r219.tsv.gz).
   A_gogpt: GO-GPT train+val proteins whose accession is in release 219. Input = S0 labels (full GO-GPT labels,
     propagated over is_a + part_of in GO 2023-01-01 as temporal_val.full_labels does). Target = closure of the
     release 219 terms minus S0, per aspect, aspect roots dropped. S0 is ancestor-closed, so a term is gained only
     if neither it nor any descendant was in S0.
   A_extra: release 219 proteins outside GO-GPT (accession, UniProt secondary-accession links and sequence) with
     rows dated (S0_DATE, R219_DATE]. Input = closure of their release 219 rows dated <= S0_DATE; target = closure
     of rows dated in the window minus the input. Sequences from UniProt REST; entries whose sequence was modified
     after R219_DATE, or whose accession is no longer primary, are dropped.
   Release 219 rows: UniProtKB, evidence EXP IDA IPI IMP IGI IEP TAS IC HTP HDA HMP HGI HEP, no NOT qualifier, no
   IPI to GO:0005515 (the fresh24 rules); terms mapped to GO 2023-01-01 (alt ids remapped, obsolete or new terms
   dropped).
B  input T = T0 (2021-11-30), targets within the GO-GPT snapshot (cutoff 2022-11-30): temporal_val.build_split
   (the TV v2 rule) on GO-GPT train+val with the TV v2 dated sources (temporal_val.dated_goa_v2) plus the dated
   release 219 rows (NOT removed, protein binding kept, as in TV). Gained = stripped "tv" + "pk" pairs; input =
   closure(S0 minus stripped); target = stripped minus input.
Every period: one row per (protein, aspect) over MFO/BPO/CCO for the selected proteins = all proteins with a gain
plus a random sample (seed 0) of NOGAIN_RATIO x as many covered proteins without a gain.

Exclusions: GO-GPT test accessions, fresh24 accessions (plus UniProt secondary/primary links of both) and every
protein whose sequence md5 equals a GO-GPT test or fresh24 sequence.

Retrieval as of the input time: DIAMOND (--more-sensitive -e 1e-3 -k 60, --dbsize = GO-GPT train letters) of all
example proteins against GO-GPT train+val; self hits and references without a non-root label at the input time
removed; identity-sharpened kNN prior knn_sharp T5_k50_g0_na over the period's reference labels (A: S0, B: T0
labels by the rule above). Files A_knn_prior.npz / B_knn_prior.npz use the knn_sharp contract layout.

Splits: MMseqs2 easy-cluster (50% identity, 80% coverage) over all example sequences; whole clusters go to
valid (seed 0, ~VALID_FRAC of proteins), so A and B put a protein in the same split.

Usage: python -m cafa6.build_forecast labels extra diamond cluster prior stats [--threads 16]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import cafa6.temporal_val as tvm
from cafa6.data import ASPECT_ROOT, ROOT
from cafa6.temporal_val import GO, SRC, full_labels, ident_bins

OUT = ROOT / "data" / "forecast"
WORK = OUT / "work"
LOG = ROOT / "logs" / "forecast_data.log"
STATS_MD = ROOT / "runs" / "forecast_data_stats.md"
FRESH = ROOT / "data" / "fresh24"
R219 = FRESH / "source" / "exp_r219.tsv.gz"
SEC_AC = FRESH / "source" / "sec_ac.txt.gz"
DIAMOND = ROOT / "tools" / "diamond"
MMSEQS = ROOT / "tools" / "mmseqs" / "bin" / "mmseqs"

T0, S0_DATE, R219_DATE = 20211130, 20221130, 20240209
EVIDENCE = ("EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP")
PROTEIN_BINDING = "GO:0005515"
ASPECTS = {"F": "MFO", "P": "BPO", "C": "CCO"}
ROOTS = set(ASPECT_ROOT.values())
NOGAIN_RATIO, SEED, VALID_FRAC = 2, 0, 0.1
KNN = dict(T=5.0, k=50, gamma=0, norm="aspect", setting="T5_k50_g0_na")
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
GAF_COLS = ["acc", "qual", "term", "ev", "aspect", "taxon", "date", "by"]


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


tvm.log = log


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def rep_update(**kw):
    p = WORK / "report.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    r.update(kw)
    p.write_text(json.dumps(r, indent=1, default=str))


def report() -> dict:
    return json.loads((WORK / "report.json").read_text())


def secondary_links() -> tuple[dict, dict]:
    """sec -> [primaries], primary -> [secs] from UniProt sec_ac.txt (used only to widen exclusions)."""
    sec, started = {}, False
    for line in gzip.open(SEC_AC, "rt"):
        if line.startswith("_"):
            started = True
            continue
        p = line.split()
        if started and len(p) == 2:
            sec.setdefault(p[0], []).append(p[1])
    sec_of = {}
    for s, ps in sec.items():
        for p in ps:
            sec_of.setdefault(p, []).append(s)
    return sec, sec_of


def linked(accs: set, sec: dict, sec_of: dict) -> set:
    out = set(accs)
    for a in accs:
        out.update(sec.get(a, ()))
        out.update(sec_of.get(a, ()))
    return out


def load_r219(go: GO, protein_binding: bool) -> pd.DataFrame:
    """Release 219 experimental UniProtKB rows, NOT removed, terms in GO 2023-01-01 with the GAF aspect."""
    df = pd.read_csv(R219, sep="\t", names=GAF_COLS, dtype=str, keep_default_na=False)
    n0 = len(df)
    df = df[~df.qual.str.contains("NOT") & df.ev.isin(EVIDENCE)]
    if not protein_binding:
        df = df[~((df.ev == "IPI") & (df.term == PROTEIN_BINDING))]
    df = df.assign(term=df.term.map(lambda t: go.alt.get(t, t)), date=df.date.astype(int))
    ok = df.term.map(go.aspect.get).values == df.aspect.values
    log(f"release 219: {n0:,} rows -> {len(df):,} after NOT/evidence{'' if protein_binding else '/protein-binding'} "
        f"filter; {int((~ok).sum()):,} rows with terms not in GO 2023-01-01 (or aspect mismatch) dropped; "
        f"dates {df.date.min()}..{df.date.max()}")
    return df[ok].reset_index(drop=True)


def closer(go: GO):
    cache = {}

    def c(t):
        if t not in cache:
            cache[t] = frozenset({t} | go.closure.get(t, set()))
        return cache[t]
    return c


# ------------------------------------------------------------------------------------------------ labels
def cmd_labels(a):
    WORK.mkdir(parents=True, exist_ok=True)
    log("== labels: period A (S0 2022-11 -> release 219) and period B (T0 2021-11-30 -> 2022-11-30)")
    go = GO()
    cl = closer(go)
    src = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split", "organism"])
    sec, sec_of = secondary_links()
    test_acc = set(src.id[src.split == "test"])
    fr = pd.read_csv(FRESH / "proteins.tsv", sep="\t", usecols=["accession"]).accession
    fr_seq = pd.read_parquet(FRESH / "work" / "seqs.parquet")
    fr_seq = fr_seq[fr_seq.accession.isin(set(fr))]
    fresh_acc = set(fr)
    excl_acc = linked(test_acc | fresh_acc, sec, sec_of)
    excl_md5 = set(src.seq[src.split == "test"].map(md5)) | set(fr_seq.seq.map(md5))
    tvl = src[src.split.isin(["train", "val"])].drop_duplicates("id")
    tvl = tvl[~tvl.id.isin(test_acc)]  # 760 accessions are both test and train/val
    m5 = tvl.seq.map(md5)
    drop_acc = tvl.id.isin(excl_acc)
    drop_seq = m5.isin(excl_md5) & ~drop_acc
    log(f"GO-GPT train+val {src.split.isin(['train', 'val']).sum():,} rows; accessions also in test "
        f"{int(src[src.split.isin(['train', 'val'])].id.isin(test_acc).sum()):,}; of the rest, linked to test/fresh24 "
        f"by accession or sec_ac {int(drop_acc.sum()):,}, identical sequence to test/fresh24 {int(drop_seq.sum()):,}")
    pool = tvl[~drop_acc & ~drop_seq]
    pool_ids = set(pool.id)
    labels = {**full_labels(go, "train"), **full_labels(go, "val")}
    allref = set(src.id[src.split.isin(["train", "val"])])
    labels = {p: s for p, s in labels.items() if p in allref}
    log(f"example pool (GO-GPT train+val after exclusions): {len(pool):,}; S0 label sets {len(labels):,}")
    rep_update(exclusions={"gogpt_trainval_rows": int(src.split.isin(["train", "val"]).sum()),
                           "accession_in_gogpt_test": int(src[src.split.isin(["train", "val"])].id.isin(test_acc).sum()),
                           "linked_accession_test_or_fresh24": int(drop_acc.sum()),
                           "identical_sequence_test_or_fresh24": int(drop_seq.sum()), "pool": len(pool)})

    def by_aspect(terms) -> dict:
        d = {"F": set(), "P": set(), "C": set()}
        for t in terms:
            x = go.aspect.get(t)
            if x in d:
                d[x].add(t)
        return d

    rng = np.random.default_rng(SEED)

    def select(per_prot: dict, covered: list, tag: str) -> list:
        gain = sorted(p for p in covered if any(per_prot[p][x][1] for x in "FPC"))
        gs = set(gain)
        nog = sorted(p for p in covered if p not in gs)
        n = min(len(nog), NOGAIN_RATIO * len(gain))
        pick = sorted(rng.choice(nog, n, replace=False).tolist()) if n else []
        log(f"[{tag}] covered proteins {len(covered):,}: with a gain {len(gain):,}, without {len(nog):,}; "
            f"sampled {len(pick):,} without (ratio {len(pick) / max(len(gain), 1):.2f})")
        return gain + pick

    # ---------------- period A, GO-GPT proteins
    r = load_r219(go, protein_binding=False)
    taxon = r.drop_duplicates("acc").set_index("acc").taxon.str.split("|").str[0].str.replace("taxon:", "").astype(int)
    ra = r[r.acc.isin(pool_ids)]
    covered = sorted(set(ra.acc))
    not_in_219 = len(pool_ids - set(covered))
    log(f"[A_gogpt] pool proteins with release 219 rows (same accession): {len(covered):,}; without: {not_in_219:,} "
        f"(not used: gain/no-gain unknown)")
    perA, extraA = {}, {}
    lost = Counter()
    for pid, d in ra.groupby("acc", sort=False):
        S0 = by_aspect(labels.get(pid, ()))
        rows = {"F": [], "P": [], "C": []}
        for t, dt, by in zip(d.term, d.date, d.by):
            rows[go.aspect[t]].append((t, dt, by))
        perA[pid], extraA[pid] = {}, {}
        for x in "FPC":
            R = set().union(*(cl(t) for t, _, _ in rows[x])) if rows[x] else set()
            late = set().union(*(cl(t) for t, dt, _ in rows[x] if dt > S0_DATE)) if rows[x] else set()
            g = (R - S0[x]) - ROOTS
            lost[x] += len((S0[x] - R) - ROOTS) if R else 0
            by = sorted({b for t, _, b in rows[x] if cl(t) & g})
            perA[pid][x] = (sorted(S0[x] - ROOTS), sorted(g))
            extraA[pid][x] = (sorted(g & late), ",".join(by), max((dt for _, dt, _ in rows[x]), default=0))
    selA = select(perA, covered, "A_gogpt")

    # ---------------- period A, release 219 proteins outside GO-GPT
    gogpt_all = set(src.id)
    gogpt_linked = linked(gogpt_all, sec, sec_of)
    cand = r[~r.acc.isin(gogpt_linked) & ~r.acc.isin(excl_acc)]
    win_acc = set(cand.acc[(cand.date > S0_DATE) & (cand.date <= R219_DATE)])
    cand = cand[cand.acc.isin(win_acc)]
    perX, extraX = {}, {}
    for pid, d in cand.groupby("acc", sort=False):
        perX[pid], extraX[pid] = {}, {}
        for x in "FPC":
            dx = d[d.aspect.values == x]
            pre = set().union(*(cl(t) for t in dx.term[dx.date <= S0_DATE])) if len(dx) else set()
            win = set().union(*(cl(t) for t in dx.term[(dx.date > S0_DATE) & (dx.date <= R219_DATE)])) if len(dx) else set()
            g = (win - pre) - ROOTS
            by = sorted({b for t, dt, b in zip(dx.term, dx.date, dx.by) if dt > S0_DATE and cl(t) & g})
            perX[pid][x] = (sorted(pre - ROOTS), sorted(g))
            extraX[pid][x] = (sorted(g), ",".join(by), int(dx.date.max()) if len(dx) else 0)
    selX = sorted(p for p in perX if any(perX[p][x][1] for x in "FPC"))
    n_pre = sum(any(perX[p][x][0] for x in "FPC") for p in selX)
    log(f"[A_extra] release 219 accessions outside GO-GPT (and not linked to GO-GPT/test/fresh24 by sec_ac) with "
        f"rows dated in ({S0_DATE}, {R219_DATE}]: {len(win_acc):,}; with a gained term {len(selX):,} "
        f"({n_pre:,} of them with rows dated <= {S0_DATE})")

    rowsA = []
    for pid in selA:
        for x in "FPC":
            inp, tg = perA[pid][x]
            late, by, mxd = extraA[pid][x]
            rowsA.append((pid, "A_gogpt", x, inp, tg, late, by, mxd))
    for pid in selX:
        for x in "FPC":
            inp, tg = perX[pid][x]
            late, by, mxd = extraX[pid][x]
            rowsA.append((pid, "A_extra", x, inp, tg, late, by, mxd))
    A = pd.DataFrame(rowsA, columns=["protein", "source", "aspect", "input_terms", "target_gained_terms",
                                     "target_dated_after_input", "target_assigned_by", "max_row_date"])
    A["taxon"] = A.protein.map(taxon).fillna(-1).astype(int)
    A.to_parquet(WORK / "A_labels.parquet", index=False)
    log(f"[A] rows {len(A):,} ({A.protein.nunique():,} proteins); lost S0 terms absent from release 219 closure "
        f"(not targets): {dict(lost)}")

    # ---------------- period B
    trv = set(labels)
    g2, gstats = tvm.dated_goa_v2(trv)
    rb = load_r219(go, protein_binding=True)
    tx = rb.drop_duplicates("acc").set_index("acc").taxon.str.split("|").str[0].str.replace("taxon:", "").astype(int)
    org = src.drop_duplicates("id").set_index("id").organism
    org_tx = pd.DataFrame({"o": org.reindex(tx.index), "t": tx}).dropna().groupby("o").t.agg(lambda s: s.mode()[0])
    taxon_b = pd.Series({p: tx.get(p, org_tx.get(org.get(p), -1)) for p in trv})
    rb = rb[rb.acc.isin(trv)].rename(columns={"acc": "id", "term": "go", "ev": "evidence"})[["id", "go", "evidence", "date"]]
    g = pd.concat([g2.assign(date=g2.date.astype(int)), rb], ignore_index=True).drop_duplicates()
    log(f"[B] dated rows: TV v2 sources {len(g2):,} ({g2.id.nunique():,} proteins), release 219 {len(rb):,} "
        f"({rb.id.nunique():,} proteins); union {len(g):,} ({g.id.nunique():,} proteins)")
    stripped, tvrows, bstats = tvm.build_split(go, labels, trv, g)
    drop = stripped.groupby("id").term.agg(set).to_dict()
    lab_t0 = {}
    for p, s in labels.items():
        lab_t0[p] = go.close(s - drop[p]) if p in drop else s
    pd.DataFrame([(p, t) for p, s in lab_t0.items() for t in s], columns=["id", "term"]).to_parquet(
        WORK / "B_t0_labels.parquet", index=False)
    stripped.to_parquet(WORK / "B_stripped.parquet", index=False)
    gd = g[g.date <= tvm.CUTOFF]
    covB = sorted(set(gd.id) & pool_ids)
    tv2 = set(open(tvm.OUT_V2 / "tv_ids.txt").read().split())
    perB = {}
    for p in covB:
        S0 = by_aspect(labels.get(p, ()))
        I = by_aspect(lab_t0[p])
        dp = by_aspect(drop.get(p, ()))
        perB[p] = {x: (sorted(I[x] - ROOTS), sorted((dp[x] - I[x]) - ROOTS)) for x in "FPC"}
    selB = select(perB, covB, "B")
    kind = stripped.groupby(["id", "aspect"]).kind.agg(lambda s: ",".join(sorted(set(s)))).to_dict()
    B = pd.DataFrame([(p, "B_gogpt", x, *perB[p][x], kind.get((p, x), "")) for p in selB for x in "FPC"],
                     columns=["protein", "source", "aspect", "input_terms", "target_gained_terms", "tv_kind"])
    B["in_tv2"] = B.protein.isin(tv2)
    B["taxon"] = B.protein.map(taxon_b).fillna(-1).astype(int)
    B.to_parquet(WORK / "B_labels.parquet", index=False)
    log(f"[B] stripped pairs {stripped.kind.value_counts().to_dict()}; aspect stats {bstats}; rows {len(B):,} "
        f"({B.protein.nunique():,} proteins)")

    # leakage bookkeeping for B's reference labels: kept T0 terms supported by window rows only
    gg = g.assign(go=g.go.map(lambda t: go.alt.get(t, t)))
    pre = {p: go.close(s) for p, s in gg[gg.date <= T0].groupby("id").go}
    win = {p: go.close(s) for p, s in gg[(gg.date > T0) & (gg.date <= tvm.CUTOFF)].groupby("id").go}
    n_ref, n_win_only, n_dated = 0, 0, 0
    for p, s in lab_t0.items():
        s = s - ROOTS
        n_ref += len(s)
        w, q = win.get(p, set()), pre.get(p, set())
        n_win_only += len((s & w) - q)
        n_dated += len(s & q)
    rep_update(A_label_counts={"covered": len(covered), "pool_not_in_219": not_in_219, "selected": len(selA),
                               "extra_window_accessions": len(win_acc), "extra_with_gain": len(selX),
                               "extra_with_pre_rows": n_pre, "lost_terms": dict(lost)},
               B_label_counts={"dated_sources": gstats, "dated_rows_union": len(g), "covered": len(covB),
                               "selected": len(selB), "stripped": stripped.kind.value_counts().to_dict(),
                               "aspect_stats": bstats},
               B_reference_dating={"reference_terms_nonroot": n_ref, "supported_by_row_dated_le_T0": n_dated,
                                   "window_support_only_kept_by_closure": n_win_only})
    log(f"[B] reference T0 labels: {n_ref:,} non-root terms; {n_dated:,} supported by a row dated <= {T0}; "
        f"{n_win_only:,} have only window support but are implied by an undated kept descendant")


# ------------------------------------------------------------------------------------------------ extra
def cmd_extra(a):
    A = pd.read_parquet(WORK / "A_labels.parquet")
    accs = sorted(set(A.protein[A.source == "A_extra"]))
    log(f"== extra: UniProt REST for {len(accs):,} A_extra accessions")
    rows, release = [], None
    fields = "accession,organism_id,organism_name,length,sequence,date_sequence_modified,sequence_version"
    for i in range(0, len(accs), 400):
        chunk = accs[i:i + 400]
        url = (f"https://rest.uniprot.org/uniprotkb/accessions?accessions={','.join(chunk)}"
               f"&format=json&fields={fields}&size=400")
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    release = r.headers.get("X-UniProt-Release"), r.headers.get("X-UniProt-Release-Date")
                    d = json.load(r)
                break
            except Exception as e:  # noqa: BLE001
                log(f"uniprot batch {i // 400}: {e}; retry {attempt + 1}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError("uniprot fetch failed")
        for e in d["results"]:
            o = e.get("organism", {})
            rows.append({"primary": e["primaryAccession"], "taxon": o.get("taxonId"),
                         "organism": o.get("scientificName"), "seq": e["sequence"]["value"],
                         "seq_modified": e.get("entryAudit", {}).get("lastSequenceUpdateDate", ""),
                         "reviewed": "Swiss-Prot" in e.get("entryType", "")})
        if (i // 400) % 10 == 0:
            log(f"uniprot: {min(i + 400, len(accs)):,}/{len(accs):,} requested, {len(rows):,} entries")
        time.sleep(0.5)
    up = pd.DataFrame(rows).drop_duplicates("primary")
    got = up[up.primary.isin(set(accs))].copy()
    missing = len(set(accs) - set(got.primary))
    late = got.seq_modified.str.replace("-", "").fillna("0").astype(int) > R219_DATE
    src = pd.read_parquet(SRC / "train.parquet", columns=["seq", "split"])
    fr = pd.read_parquet(FRESH / "work" / "seqs.parquet")
    excl = set(src.seq[src.split == "test"].map(md5)) | set(fr.seq.map(md5))
    got["md5"] = got.seq.map(md5)
    same = got.md5.isin(excl)
    same_trv = got.md5.isin(set(src.seq[src.split != "test"].map(md5)))
    keep = got[~late & ~same]
    log(f"UniProt {release}: {len(got):,} returned as primary, {missing:,} not primary (dropped); sequence "
        f"modified after {R219_DATE}: {int(late.sum()):,} dropped; identical to GO-GPT test/fresh24: "
        f"{int((same & ~late).sum()):,} dropped; kept {len(keep):,} ({int(same_trv[~late & ~same].sum()):,} identical "
        f"to a GO-GPT train/val sequence, kept)")
    keep[["primary", "taxon", "organism", "seq", "seq_modified", "reviewed"]].to_parquet(WORK / "extra_seqs.parquet",
                                                                                       index=False)
    rep_update(extra_fetch={"requested": len(accs), "uniprot_release": release, "not_primary": missing,
                            "seq_modified_after_r219": int(late.sum()),
                            "identical_test_or_fresh24": int((same & ~late).sum()), "kept": len(keep),
                            "identical_gogpt_trainval_kept": int(same_trv[~late & ~same].sum())})


# ------------------------------------------------------------------------------------------------ sequences
def example_proteins() -> pd.DataFrame:
    """protein, seq, organism, source for every example protein of A and B."""
    src = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split", "organism"])
    src = src[src.split.isin(["train", "val"])].drop_duplicates("id")
    A = pd.read_parquet(WORK / "A_labels.parquet", columns=["protein", "source"])
    B = pd.read_parquet(WORK / "B_labels.parquet", columns=["protein"])
    ids = set(A.protein[A.source == "A_gogpt"]) | set(B.protein)
    g = src[src.id.isin(ids)].rename(columns={"id": "protein"})[["protein", "seq", "organism", "split"]]
    g = g.rename(columns={"split": "gogpt_split"})
    x = pd.read_parquet(WORK / "extra_seqs.parquet").rename(columns={"primary": "protein"})
    x = x[["protein", "seq", "organism"]].assign(gogpt_split="")
    return pd.concat([g, x], ignore_index=True)


def cmd_diamond(a):
    src = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split"])
    db = src[src.split.isin(["train", "val"])].drop_duplicates("id")
    dbsize = int(src.seq[src.split == "train"].str.len().sum())
    q = example_proteins()
    log(f"== DIAMOND: {len(q):,} example proteins vs {len(db):,} GO-GPT train+val, --more-sensitive -e 1e-3 -k 60 "
        f"--dbsize {dbsize:,}, {a.threads} threads")
    t = time.time()
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        tmp = Path(tmp)
        for d, idc, name in ((db, "id", "db.fa"), (q, "protein", "q.fa")):
            with open(tmp / name, "w") as f:
                for pid, s in zip(d[idc], d.seq):
                    f.write(f">{pid}\n{s}\n")
        subprocess.run([str(DIAMOND), "makedb", "--in", str(tmp / "db.fa"), "-d", str(tmp / "db"), "--quiet"],
                       check=True)
        subprocess.run([str(DIAMOND), "blastp", "-d", str(tmp / "db"), "-q", str(tmp / "q.fa"),
                        "-o", str(tmp / "hits.tsv"), "--more-sensitive", "-e", "1e-3", "-k", "60",
                        "-p", str(a.threads), "--quiet", "--tmpdir", str(tmp), "--dbsize", str(dbsize),
                        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)
        h = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None, names=["q", "s", "pident", "bits", "evalue", "qcov"])
    n_self = int((h.q == h.s).sum())
    h = h[h.q != h.s].sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
    h.reset_index(drop=True).to_parquet(WORK / "hits.parquet", index=False)
    log(f"DIAMOND done in {time.time() - t:.0f}s: {len(h):,} hits for {h.q.nunique():,}/{len(q):,} queries; "
        f"self hits removed {n_self:,}")


def cmd_cluster(a):
    q = example_proteins().drop_duplicates("protein")
    log(f"== MMseqs2 easy-cluster of {len(q):,} example sequences (--min-seq-id 0.5 -c 0.8 --cov-mode 0), "
        f"{a.threads} threads")
    t = time.time()
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        tmp = Path(tmp)
        with open(tmp / "q.fa", "w") as f:
            for pid, s in zip(q.protein, q.seq):
                f.write(f">{pid}\n{s}\n")
        subprocess.run([str(MMSEQS), "easy-cluster", str(tmp / "q.fa"), str(tmp / "res"), str(tmp / "mm"),
                        "--min-seq-id", "0.5", "-c", "0.8", "--cov-mode", "0", "--threads", str(a.threads),
                        "-v", "1"], check=True)
        c = pd.read_csv(tmp / "res_cluster.tsv", sep="\t", header=None, names=["mmseqs_rep", "protein"])
    # MMseqs2 clusters merged over every DIAMOND example-example hit at >= 50% identity (any coverage), so no
    # detected >= 50% pair crosses the split
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    h = pd.read_parquet(WORK / "hits.parquet")
    pos = {p: i for i, p in enumerate(c.protein)}
    e = h[h.q.isin(pos) & h.s.isin(pos) & (h.pident >= 50)]
    r = np.r_[np.arange(len(c)), e.q.map(pos).values]
    cc = np.r_[c.mmseqs_rep.map(pos).values, e.s.map(pos).values]
    _, lab = connected_components(sp.coo_matrix((np.ones(len(r)), (r, cc)), shape=(len(c),) * 2), directed=False)
    c["rep"] = c.groupby(lab).protein.transform("min").values
    log(f"MMseqs2 {c.mmseqs_rep.nunique():,} clusters; merged over {len(e):,} DIAMOND >= 50% identity edges -> "
        f"{c.rep.nunique():,} components")
    sizes = c.rep.value_counts()
    reps = np.array(sorted(sizes.index))
    rng = np.random.default_rng(SEED)
    rng.shuffle(reps)
    target, n, valid = VALID_FRAC * len(c), 0, set()
    for r in reps:
        if n >= target:
            break
        valid.add(r)
        n += sizes[r]
    c["split"] = np.where(c.rep.isin(valid), "valid", "train")
    c.to_parquet(WORK / "clusters.parquet", index=False)
    log(f"MMseqs2 done in {time.time() - t:.0f}s: {c.rep.nunique():,} clusters (largest {sizes.max():,}, singletons "
        f"{int((sizes == 1).sum()):,}); valid {int((c.split == 'valid').sum()):,} proteins in {len(valid):,} "
        f"clusters, train {int((c.split == 'train').sum()):,}")


# ------------------------------------------------------------------------------------------------ prior
def make_knn(onto, ids, hits, db_ids, L_db):
    """A knn_sharp.KNN over arbitrary query ids and hit table (same arrays as KNN.__init__)."""
    from cafa6.knn_sharp import KNN
    kn = object.__new__(KNN)
    kn.ids = np.asarray(ids, dtype=str)
    didx = {p: i for i, p in enumerate(db_ids)}
    qidx = {p: i for i, p in enumerate(kn.ids)}
    h = hits[hits.q.isin(qidx) & hits.s.isin(didx)].sort_values(["q", "bits"], ascending=[True, False])
    h = h.drop_duplicates(["q", "s"])
    kn.q = h.q.map(qidx).values.astype(np.int64)
    kn.s = h.s.map(didx).values.astype(np.int64)
    kn.rank = h.groupby("q", sort=False).cumcount().values
    kn.bits, kn.pid = h.bits.values, h.pident.values
    kn.cov = np.clip(h.qcov.values / 100.0, 0, 1)
    kn.maxpid = np.zeros(len(kn.ids), np.float32)
    np.maximum.at(kn.maxpid, kn.q, kn.pid.astype(np.float32))
    kn.has_hit = np.bincount(kn.q, minlength=len(kn.ids)) > 0
    kn.L = L_db
    kn.roots = [onto.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    kn.term_ns = onto.ns
    kn.split = "forecast"
    return kn


def cmd_prior(a):
    from cafa6.knn_sharp import Ontology, save_prior
    log("== kNN priors at the input time (knn_sharp T5_k50_g0_na) and example tables")
    onto, go = Ontology(), GO()
    src = pd.read_parquet(SRC / "train.parquet", columns=["id", "split"])
    db_ids = np.array(sorted(set(src.id[src.split.isin(["train", "val"])])))
    S0 = {**full_labels(go, "train"), **full_labels(go, "val")}
    t0 = pd.read_parquet(WORK / "B_t0_labels.parquet").groupby("id").term.agg(set).to_dict()
    hits = pd.read_parquet(WORK / "hits.parquet")
    prot = example_proteins().drop_duplicates("protein").set_index("protein")
    cl = pd.read_parquet(WORK / "clusters.parquet").set_index("protein")
    for per, lab in (("A", S0), ("B", t0)):
        ref = np.array([p for p in db_ids if lab.get(p, set()) - ROOTS])
        L = onto.label_matrix([lab[p] for p in ref])
        ex = pd.read_parquet(WORK / f"{per}_labels.parquet")
        if per == "A":
            keep = (ex.source == "A_gogpt") | ex.protein.isin(prot.index)
            log(f"[A] A_extra rows without a kept sequence dropped: {int((~keep).sum()):,}")
            ex = ex[keep].reset_index(drop=True)
        ids = np.array(sorted(ex.protein.unique()))
        kn = make_knn(onto, ids, hits, ref, L)
        P, nh = kn.prior(KNN["T"], KNN["k"], KNN["gamma"], KNN["norm"])
        tmp = WORK / f"prior_{per}"
        save_prior(per, kn, P, nh, onto, {**KNN, "period": per,
                                         "label_db": "S0 = GO-GPT train+val labels (2022-11)" if per == "A" else
                                         "T0 labels: GO-GPT train+val labels minus terms dated after 2021-11-30",
                                         "reference": f"GO-GPT train+val with a non-root label at the input time "
                                                      f"({len(ref):,}), self excluded",
                                         "hits": "data/forecast/work/hits.parquet"}, tmp)
        (tmp / f"{per}_knn.npz").rename(OUT / f"{per}_knn_prior.npz")
        tmp.rmdir()
        mp = pd.Series(kn.maxpid, index=ids)
        ex["sequence"] = ex.protein.map(prot.seq)
        ex["organism"] = ex.protein.map(prot.organism)
        if per == "A":
            xt = pd.read_parquet(WORK / "extra_seqs.parquet").set_index("primary")
            m = ex.source == "A_extra"
            ex.loc[m, "taxon"] = ex.protein[m].map(xt.taxon).fillna(ex.taxon[m]).astype(int)
        ex["aspect"] = ex.aspect.map(ASPECTS)
        has_in = ex.groupby("protein").input_terms.agg(lambda s: any(len(x) for x in s))
        ex["aspect_status"] = np.where(ex.input_terms.map(len) > 0, "known",
                                       np.where(ex.protein.map(has_in), "lk", "nk"))
        ex["n_input"] = ex.input_terms.map(len)
        ex["n_target"] = ex.target_gained_terms.map(len)
        ex["has_gain"] = ex.n_target > 0
        ex["best_identity"] = ex.protein.map(mp).astype(float)
        ex["bin"] = ident_bins(ex.best_identity.values).values
        ex["split"] = ex.protein.map(cl.split)
        ex["cluster"] = ex.protein.map(cl.rep)
        ex["n_hits_used"] = ex.protein.map(pd.Series(nh, index=ids)).astype(int)
        lead = ["protein", "sequence", "taxon", "aspect", "input_terms", "target_gained_terms", "split",
                "best_identity", "bin"]
        ex = ex[lead + [c for c in ex.columns if c not in lead]]
        ex.to_parquet(OUT / f"{per}_examples.parquet", index=False)
        log(f"[{per}] {len(ex):,} rows / {len(ids):,} proteins -> {per}_examples.parquet; prior nnz {P.nnz:,}; "
            f"queries with a hit {int(kn.has_hit.sum()):,}; references {len(ref):,}")


# ------------------------------------------------------------------------------------------------ stats
def cmd_stats(a):
    import scipy.sparse as sp
    log("== stats and leakage checks")
    rep = report()
    src = pd.read_parquet(SRC / "train.parquet", columns=["id", "seq", "split"])
    fr = pd.read_parquet(FRESH / "work" / "seqs.parquet")
    fr_ids = set(pd.read_csv(FRESH / "proteins.tsv", sep="\t", usecols=["accession"]).accession)
    sec, sec_of = secondary_links()
    test_acc = set(src.id[src.split == "test"])
    test_md5 = set(src.seq[src.split == "test"].map(md5))
    fr_md5 = set(fr.seq[fr.accession.isin(fr_ids)].map(md5))
    go = GO()
    checks, rec_rows, fa_rows = {}, [], []
    E = {p: pd.read_parquet(OUT / f"{p}_examples.parquet") for p in ("A", "B")}
    for per, ex in E.items():
        pr = ex.drop_duplicates("protein")
        m5 = pr.sequence.map(md5)
        ids = set(pr.protein)
        z = np.load(OUT / f"{per}_knn_prior.npz")
        prior_ids = set(z["ids"].tolist())
        hits = pd.read_parquet(WORK / "hits.parquet")
        hs = set(hits.s[hits.q.isin(ids)])
        c = {"proteins": len(ids),
             "fresh24_accession_overlap": len(ids & fr_ids), "fresh24_linked_accession_overlap":
                 len(ids & linked(fr_ids, sec, sec_of)), "fresh24_sequence_overlap": int(m5.isin(fr_md5).sum()),
             "gogpt_test_accession_overlap": len(ids & test_acc), "gogpt_test_linked_accession_overlap":
                 len(ids & linked(test_acc, sec, sec_of)), "gogpt_test_sequence_overlap": int(m5.isin(test_md5).sum()),
             "prior_ids_equal_example_proteins": prior_ids == ids,
             "prior_ids_overlap_fresh24": len(prior_ids & fr_ids),
             "hit_references_in_fresh24": len(hs & fr_ids), "hit_references_in_gogpt_test_only":
                 len(hs & (test_acc - set(src.id[src.split != "test"]))),
             "max_row_date": int(ex.max_row_date.max()) if "max_row_date" in ex else None}
        clu = pd.read_parquet(WORK / "clusters.parquet")
        sp_ = clu.set_index("protein").split
        tr_c = set(clu.rep[clu.split == "train"])
        va_c = set(clu.rep[clu.split == "valid"])
        c["clusters_in_both_splits"] = len(tr_c & va_c)
        # residual similarity: valid example proteins with a >= 50% identity, >= 80% coverage hit to a train one
        h = hits[hits.q.isin(ids) & hits.s.isin(ids)]
        h = h[(h.q.map(sp_) == "valid") & (h.s.map(sp_) == "train")]
        c["valid_with_ge50id_cov80_hit_to_train_example"] = int(h[(h.pident >= 50) & (h.qcov >= 80)].q.nunique())
        c["valid_with_ge50id_any_cov_hit_to_train_example"] = int(h[h.pident >= 50].q.nunique())
        c["valid_proteins"] = int((pr.split == "valid").sum())
        checks[per] = c

        # retrieval-recoverable share of gained terms
        P = sp.csr_matrix((z["pf_data"], z["pf_indices"], z["pf_indptr"]), shape=tuple(z["pf_shape"]))
        col = {t: j for j, t in enumerate(z["terms_full"].tolist())}
        row = {p: i for i, p in enumerate(z["ids"].tolist())}
        terms = np.asarray(z["terms_full"])
        tasp = np.array([ASPECTS.get(go.aspect.get(t), "?") for t in terms])
        for r in ex.itertuples(index=False):
            pv = P[row[r.protein]]
            m = (pv.data >= 0.5) & (tasp[pv.indices] == r.aspect)
            new = set(terms[pv.indices[m]]) - set(r.input_terms) - ROOTS
            fa_rows.append((per, r.has_gain, len(new), len(new & set(r.target_gained_terms))))
        g = ex[ex.has_gain]
        for r in g.itertuples(index=False):
            i = row[r.protein]
            pv = P[i]
            d = dict(zip(pv.indices, pv.data))
            sc = np.array([d.get(col.get(t, -1), 0.0) for t in r.target_gained_terms], np.float32)
            rec_rows.append((per, r.source, r.aspect, r.bin, r.aspect_status, r.split, len(sc),
                             int((sc > 0).sum()), int((sc >= 0.1).sum()), int((sc >= 0.3).sum()),
                             int((sc >= 0.5).sum()), bool((sc > 0).all()), bool((sc >= 0.5).all())))
    rec = pd.DataFrame(rec_rows, columns=["period", "source", "aspect", "bin", "status", "split", "n", "gt0", "ge01",
                                          "ge03", "ge05", "all_gt0", "all_ge05"])
    rec.to_parquet(WORK / "recoverable.parquet", index=False)
    fa = pd.DataFrame(fa_rows, columns=["period", "has_gain", "new05", "hit05"])
    fa_tab = fa.groupby(["period", "has_gain"]).agg(
        rows=("new05", "size"), rows_with_new_term=("new05", lambda s: float((s > 0).mean())),
        mean_new_terms=("new05", "mean"), new_terms=("new05", "sum"), new_terms_gained=("hit05", "sum")).reset_index()
    fa_tab["precision"] = fa_tab.new_terms_gained / fa_tab.new_terms.clip(lower=1)

    # B reference dating check
    checks["B"]["reference_terms_dated_le_T0"] = rep["B_reference_dating"]["supported_by_row_dated_le_T0"]
    checks["B"]["reference_terms_window_only_kept_by_closure"] = rep["B_reference_dating"][
        "window_support_only_kept_by_closure"]
    checks["B"]["reference_terms_nonroot"] = rep["B_reference_dating"]["reference_terms_nonroot"]
    rep_update(checks=checks)
    log("leakage checks: " + json.dumps(checks, indent=1, default=str))

    def md(df):
        cols = list(df.columns)
        out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else
                                  (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |"
                for r in df.itertuples(index=False)]
        return "\n".join(out)

    def share(d):
        n = d.n.sum()
        return pd.Series({"examples": len(d), "gained_terms": int(n),
                          "p>0": d.gt0.sum() / n, "p>=0.1": d.ge01.sum() / n, "p>=0.3": d.ge03.sum() / n,
                          "p>=0.5": d.ge05.sum() / n, "all_gained_p>0": d.all_gt0.mean(),
                          "all_gained_p>=0.5": d.all_ge05.mean()})

    def grp(d, keys):
        o = d.groupby(keys).apply(share, include_groups=False).reset_index()
        for k in ("examples", "gained_terms"):
            o[k] = o[k].astype(int)
        return o

    lines = ["# Annotation-forecast data: statistics", "",
             "Built by `python -m cafa6.build_forecast` (rules in `data/forecast/README.md`). Rows are "
             "(protein, aspect) examples; a gain example has at least one gained term. Terms are propagated, "
             "aspect roots excluded.", ""]
    for per, ex in E.items():
        pr = ex.drop_duplicates("protein")
        gp = ex.groupby("protein").has_gain.any()
        lines += [f"## Period {per}", "",
                  f"Proteins {len(pr):,} (with a gain {int(gp.sum()):,}, without {int((~gp).sum()):,}, ratio without/with "
                  f"{(~gp).sum() / max(gp.sum(), 1):.2f}); rows {len(ex):,} (gain rows {int(ex.has_gain.sum()):,}). "
                  f"Split: train {int((pr.split == 'train').sum()):,} / valid {int((pr.split == 'valid').sum()):,} "
                  f"proteins.", ""]
        t = ex.groupby(["source", "split", "aspect"]).agg(rows=("protein", "size"), gain_rows=("has_gain", "sum"),
                                                          gained_terms=("n_target", "sum"),
                                                          mean_input=("n_input", "mean")).reset_index()
        t["gain_rows"] = t.gain_rows.astype(int)
        lines += ["### Rows per source, split, aspect", "", md(t), ""]
        t = ex.groupby(["aspect", "aspect_status"]).agg(rows=("protein", "size"), gain_rows=("has_gain", "sum"),
                                                        gained_terms=("n_target", "sum")).reset_index()
        t["gain_rows"] = t.gain_rows.astype(int)
        lines += ["### Gains by aspect status (nk: no input term in any aspect; lk: none in this aspect; "
                  "known: input terms in this aspect)", "", md(t), ""]
        t = pr.groupby("bin").agg(proteins=("protein", "size")).reindex(BINS).fillna(0).astype(int).reset_index()
        t["with_gain"] = t.bin.map(ex[ex.has_gain].drop_duplicates("protein").bin.value_counts()).fillna(0).astype(int)
        lines += ["### Identity bins (best DIAMOND identity to references at the input time)", "", md(t), ""]
        org = ex.assign(org=ex.organism.fillna("?").str.split(" (", regex=False).str[0])
        po = org.drop_duplicates("protein").org.value_counts().head(12)
        go_ = org[org.has_gain].drop_duplicates("protein").org.value_counts()
        t = pd.DataFrame({"organism": po.index, "proteins": po.values,
                          "with_gain": [int(go_.get(o, 0)) for o in po.index]})
        lines += ["### Top organisms", "", md(t), ""]
        q = ex[ex.has_gain].n_target
        qq = q.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_dict()
        hist = pd.cut(q, [0, 1, 2, 5, 10, 20, 50, 100, 1e9], labels=["1", "2", "3-5", "6-10", "11-20", "21-50",
                                                                    "51-100", ">100"]).value_counts().sort_index()
        lines += ["### Gained terms per gain row (propagated)", "",
                  f"mean {q.mean():.1f}; quantiles " + ", ".join(f"{k:g}: {v:g}" for k, v in qq.items()), "",
                  md(pd.DataFrame({"gained terms": hist.index.astype(str), "rows": hist.values})), ""]
        qp = ex[ex.has_gain].groupby("protein").n_target.sum()
        lines += [f"Per protein (all aspects): mean {qp.mean():.1f}, median {qp.median():.0f}.", ""]
        if per == "A":
            g = ex[ex.has_gain]
            late = g.target_dated_after_input.map(len).sum() / g.n_target.sum()
            by = g.target_assigned_by.str.split(",").explode().value_counts().head(10)
            lines += [f"A_gogpt + A_extra gained terms supported by a release 219 row dated after {S0_DATE}: "
                      f"{late:.1%} (the rest are supported only by rows dated before, i.e. absent from S0 for "
                      f"filtering or late-import reasons).",
                      "", "Assigning groups of the rows behind gained terms (rows counted once per example): " +
                      ", ".join(f"{k} {v:,}" for k, v in by.items()), ""]
        r = rec[rec.period == per]
        lines += ["### Retrieval-recoverable share of gained terms (kNN prior at the input time)", "",
                  "Share of gained (protein, term) pairs whose prior score is > 0 / >= 0.1 / >= 0.3 / >= 0.5, and "
                  "share of gain rows whose gained terms all have p > 0 / p >= 0.5.", "",
                  md(grp(r.assign(all="all"), ["all"])), "", md(grp(r, ["source", "aspect"])), "",
                  md(grp(r, ["aspect", "status"])), "", md(grp(r, ["bin"])), ""]
    lines += ["## What retrieval proposes beyond the input (kNN prior >= 0.5, same aspect, not in input_terms)", "",
              "Rows with and without a gain; precision = share of the proposed new terms that are gained terms. "
              "This is the false-alarm side of label transfer that a forecast model has to learn to suppress.", "",
              md(fa_tab), ""]
    lines += ["## Leakage checks", "", "```", json.dumps(checks, indent=1, default=str), "```", "",
              "## Build counts", "", "```", json.dumps({k: v for k, v in rep.items() if k != "checks"}, indent=1,
                                                       default=str), "```", ""]
    STATS_MD.write_text("\n".join(lines))
    log(f"wrote {STATS_MD}")
    print("\n".join(lines))


def cmd_verify(a):
    """Re-derives the support of every target and input term from the dated rows of the output examples."""
    log("== verify: dates behind every input and target term")
    go = GO()
    cl = closer(go)
    S0 = {**full_labels(go, "train"), **full_labels(go, "val")}
    out = {}
    A = pd.read_parquet(OUT / "A_examples.parquet")
    r = load_r219(go, protein_binding=False)
    r = r[r.acc.isin(set(A.protein))]
    rows = {k: list(zip(d.term, d.date)) for k, d in r.groupby(["acc", "aspect"])}
    inv = Counter()
    for e in A.itertuples(index=False):
        x = {"MFO": "F", "BPO": "P", "CCO": "C"}[e.aspect]
        rr = rows.get((e.protein, x), [])
        tg, inp = set(e.target_gained_terms), set(e.input_terms)
        inv["rows"] += 1
        inv["target_input_overlap"] += len(tg & inp)
        inv["target_wrong_aspect"] += sum(go.aspect.get(t) != x for t in tg)
        inv["input_wrong_aspect"] += sum(go.aspect.get(t) != x for t in inp)
        if e.source == "A_gogpt":
            sup = set().union(*(cl(t) for t, _ in rr)) if rr else set()
            inv["gogpt_target_in_S0"] += len(tg & S0.get(e.protein, set()))
            inv["gogpt_input_not_S0"] += len(inp - S0.get(e.protein, set()))
        else:
            sup = set().union(*(cl(t) for t, d in rr if S0_DATE < d <= R219_DATE)) if rr else set()
            pre = set().union(*(cl(t) for t, d in rr if d <= S0_DATE)) if rr else set()
            inv["extra_input_not_dated_le_S0"] += len(inp - pre)
            inv["extra_target_implied_by_pre"] += len(tg & pre)
        inv["target_without_support"] += len(tg - sup)
        used = [d for t, d in rr if cl(t) & (tg | inp)]
        inv["max_support_date"] = max(inv["max_support_date"], max(used, default=0))
    out["A"] = dict(inv)

    B = pd.read_parquet(OUT / "B_examples.parquet")
    ids = set(B.protein)
    g2, _ = tvm.dated_goa_v2(ids)
    rb = load_r219(go, protein_binding=True)
    rb = rb[rb.acc.isin(ids)].rename(columns={"acc": "id", "term": "go", "ev": "evidence"})
    g = pd.concat([g2.assign(date=g2.date.astype(int))[["id", "go", "date"]], rb[["id", "go", "date"]]])
    g = g.assign(go=g.go.map(lambda t: go.alt.get(t, t)))
    pre = {p: go.close(s) for p, s in g[g.date <= T0].groupby("id").go}
    wrows = {p: list(zip(d.go, d.date)) for p, d in g[(g.date > T0) & (g.date <= tvm.CUTOFF)].groupby("id")}
    inv = Counter()
    for e in B.itertuples(index=False):
        tg, inp = set(e.target_gained_terms), set(e.input_terms)
        s0 = S0.get(e.protein, set())
        wr = wrows.get(e.protein, [])
        win = set().union(*(cl(t) for t, _ in wr)) if wr else set()
        inv["rows"] += 1
        inv["target_input_overlap"] += len(tg & inp)
        inv["target_not_in_S0"] += len(tg - s0)
        inv["input_not_in_S0"] += len(inp - s0)
        inv["target_without_window_row"] += len(tg - win)
        inv["target_implied_by_row_le_T0"] += len(tg & pre.get(e.protein, set()))
        inv["input_window_only"] += len((inp & win) - pre.get(e.protein, set()))
        used = [d for t, d in wr if cl(t) & tg]
        inv["max_target_support_date"] = max(inv["max_target_support_date"], max(used, default=0))
        inv["min_target_support_date"] = min(inv.get("min_target_support_date", 99999999), min(used, default=99999999))
    out["B"] = dict(inv)
    rep_update(verify=out)
    log("verify: " + json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["labels", "extra", "diamond", "cluster", "prior", "stats", "verify"])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    for s in a.steps:
        globals()[f"cmd_{s}"](a)


if __name__ == "__main__":
    main()
