"""fresh24: a temporal benchmark of experimental GO annotations added after 2024-02-29.

Construction (details and numbers in data/fresh24/README.md):
  1. Stream the current GOA UniProt GAF and the archived release 219 (2024-02-09, the last release before
     2024-03) once each, with parallel HTTP range requests reassembled in order (EBI serves ~1 MB/s per
     connection), through `pigz -dc | grep | mawk`. Only UniProtKB rows with an experimental or
     high-throughput experimental evidence code are kept (EVIDENCE below). Nothing but the filtered rows is
     written; the compressed bytes pass through bounded chunk files in /dev/shm.
  2. Prior knowledge at the cutoff for (protein, aspect) = an experimental annotation (non-NOT, excluding
     IPI to GO:0005515) in release 219, or one dated <= 2024-02-29 in the current file.
     New terms = current non-NOT experimental annotations dated > 2024-02-29 that were absent from 219.
     no-knowledge (nk): no prior experimental annotation in any aspect.
     limited-knowledge (lk): no prior annotation in this aspect, some in another aspect.
  3. Terms are mapped to runs/gogpt_cafa5/ref/go-basic_gogpt.obo (alt ids -> canonical, obsolete dropped).
  4. Proteins in GO-GPT train/val/test or TV v2 (by accession or identical sequence) are removed.
  5. Sequences/organisms from UniProt REST; DIAMOND (--more-sensitive -e 1e-3 -k 51) against GO-GPT
     train+val for label-free identity statistics.

Usage:
  python -m cafa6.build_fresh24 fetch            # stream both GAFs (~45 min) -> data/fresh24/source/
  python -m cafa6.build_fresh24 build            # candidates, uniprot fetch, disjointness, outputs
  python -m cafa6.build_fresh24 diamond stats    # identity statistics, runs/fresh24_stats.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/dev/shm/fresh24_work")  # range-request chunks and DIAMOND temp files only
OUT = ROOT / "data" / "fresh24"
SRC_DIR = OUT / "source"  # filtered experimental GAF rows, stream metadata, sec_ac.txt
INTER = OUT / "work"  # UniProt sequences, protein table before identity, DIAMOND hits
LOG = ROOT / "logs" / "fresh_benchmark.log"
STATS_MD = ROOT / "runs" / "fresh24_stats.md"
OBO = ROOT / "runs" / "gogpt_cafa5" / "ref" / "go-basic_gogpt.obo"
GOGPT = ROOT / "data" / "proc_gogpt" / "train.parquet"
TV2_IDS = ROOT / "data" / "proc_gogpt_tv2" / "tv_ids.txt"
DIAMOND = ROOT / "tools" / "diamond"

GOA_BASE = "https://ftp.ebi.ac.uk/pub/databases/GO/goa"
SOURCES = {
    "current": f"{GOA_BASE}/UNIPROT/goa_uniprot_all.gaf.gz",
    "r219": f"{GOA_BASE}/old/UNIPROT/goa_uniprot_all.gaf.219.gz",
}
SEC_AC_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/sec_ac.txt"
CUTOFF = "20240229"
EVIDENCE = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP"]
PROTEIN_BINDING = "GO:0005515"
ASPECTS = {"F": "MFO", "P": "BPO", "C": "CCO"}
ID_BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------------------------- fetch
def head(url: str) -> dict:
    r = subprocess.run(["curl", "-sfI", url], capture_output=True, text=True, check=True)
    h = {}
    for line in r.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            h[k.strip().lower()] = v.strip()
    return h


def stream_ranges(url: str, sink, size: int, tmp: Path, conns: int, chunk: int, tag: str) -> str:
    """Write the bytes of `url` to `sink` in order using `conns` parallel range requests; returns md5."""
    tmp.mkdir(parents=True, exist_ok=True)
    n = (size + chunk - 1) // chunk
    window = 2 * conns

    def fetch(i: int) -> Path:
        a, b = i * chunk, min(size, (i + 1) * chunk) - 1
        p = tmp / f"part_{i:06d}"
        for attempt in range(40):
            r = subprocess.run(["curl", "-sf", "--connect-timeout", "30", "--speed-time", "60",
                                "--speed-limit", "20000", "-r", f"{a}-{b}", "-o", str(p), "-w", "%{http_code}",
                                url], capture_output=True, text=True)
            if r.returncode == 0 and p.exists() and p.stat().st_size == b - a + 1:
                return p
            log(f"[{tag}] chunk {i} attempt {attempt + 1}: curl exit {r.returncode} http {r.stdout.strip()} "
                f"bytes {p.stat().st_size if p.exists() else 0}")
            time.sleep(min(60, 5 * (attempt + 1)))
        raise RuntimeError(f"{tag}: chunk {i} failed after retries")

    md5 = hashlib.md5()
    t0, done = time.time(), 0
    with ThreadPoolExecutor(conns) as ex:
        futs = {i: ex.submit(fetch, i) for i in range(min(n, window))}
        for i in range(n):
            p = futs.pop(i).result()
            data = p.read_bytes()
            p.unlink()
            if i + window < n:
                futs[i + window] = ex.submit(fetch, i + window)
            md5.update(data)
            sink.write(data)
            done += len(data)
            if i % 16 == 0 or i == n - 1:
                el = time.time() - t0
                log(f"[{tag}] {done / 1e9:.2f}/{size / 1e9:.2f} GB  {done / 1e6 / max(el, 1e-9):.1f} MB/s  "
                    f"elapsed {el / 60:.1f} min")
    return md5.hexdigest()


def gaf_header(url: str) -> list[str]:
    r = subprocess.run(f"curl -sf -r 0-2000000 '{url}' | gzip -dc 2>/dev/null | grep '^!' | head -40",
                       shell=True, capture_output=True, text=True)
    return r.stdout.splitlines()


def fetch_one(tag: str, conns: int, chunk_mb: int):
    url = SOURCES[tag]
    WORK.mkdir(parents=True, exist_ok=True)
    h = head(url)
    size = int(h["content-length"])
    hdr = gaf_header(url)
    log(f"[{tag}] {url} size={size} last-modified={h.get('last-modified')} conns={conns} chunk={chunk_mb}MB")
    for line in hdr:
        if any(k in line.lower() for k in ("date", "version", "generated")):
            log(f"[{tag}] header {line}")
    out = WORK / f"exp_{tag}.tsv"
    ev = " ".join(EVIDENCE)
    # fixed-string prefilter (grep -P on the ~1e9 lines was the bottleneck); mawk checks column 7 exactly
    grep_pat = " ".join(f"-e '\t{e}\t'" for e in EVIDENCE)
    # keep: acc, qualifier, GO id, evidence, aspect, taxon, date, assigned_by (NOT and protein binding are
    # kept here and filtered in `build` so that they can be counted)
    awk = ("BEGIN{n=split(\"%s\",a,\" \"); for(i=1;i<=n;i++) ev[a[i]]=1}"
           " $1==\"UniProtKB\" && ($7 in ev) {print $2,$4,$5,$7,$9,$13,$14,$15}") % ev
    cmd = f"pigz -dc | LC_ALL=C grep -aF {grep_pat} | LC_ALL=C mawk -F'\\t' -v OFS='\\t' '{awk}' > '{out}'"
    proc = subprocess.Popen(["bash", "-c", "set -o pipefail; " + cmd], stdin=subprocess.PIPE)
    md5 = stream_ranges(url, proc.stdin, size, WORK / f"chunks_{tag}", conns, chunk_mb << 20, tag)
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{tag}: filter pipeline exit {rc}")
    rows = int(subprocess.run(["wc", "-l", str(out)], capture_output=True, text=True).stdout.split()[0])
    ref = None
    r = subprocess.run(["curl", "-sf", url + ".md5"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        ref = r.stdout.split()[0]
    meta = {"url": url, "bytes": size, "last_modified": h.get("last-modified"), "md5_streamed": md5,
            "md5_published": ref, "header": hdr, "experimental_rows": rows}
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    (SRC_DIR / f"meta_{tag}.json").write_text(json.dumps(meta, indent=1))
    subprocess.run(f"pigz -c '{out}' > '{SRC_DIR}/exp_{tag}.tsv.gz' && rm '{out}'", shell=True, check=True)
    log(f"[{tag}] done: {rows} experimental UniProtKB rows -> {SRC_DIR}/exp_{tag}.tsv.gz; md5 streamed {md5} "
        f"published {ref} {'MATCH' if ref == md5 else ('n/a' if ref is None else 'MISMATCH')}")


def cmd_fetch(a):
    ths = [threading.Thread(target=fetch_one, args=(t, a.conns, a.chunk_mb)) for t in a.sources]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(f"curl -sf '{SEC_AC_URL}' | pigz -c > '{SRC_DIR}/sec_ac.txt.gz'", shell=True, check=True)
    log(f"sec_ac.txt from {SEC_AC_URL} -> {SRC_DIR}/sec_ac.txt.gz")


# ---------------------------------------------------------------------------------------------- build
COLS = ["acc", "qual", "term", "ev", "aspect", "taxon", "date", "by"]


def load_exp(tag: str, rep: dict):
    import pandas as pd
    df = pd.read_csv(SRC_DIR / f"exp_{tag}.tsv.gz", sep="\t", names=COLS, dtype=str, keep_default_na=False)
    is_not = df.qual.str.contains("NOT")
    is_pb = (df.ev == "IPI") & (df.term == PROTEIN_BINDING) & ~is_not
    keep = df[~is_not & ~is_pb].copy()
    rep[tag] = {"rows": len(df), "rows_NOT": int(is_not.sum()), "rows_IPI_protein_binding": int(is_pb.sum()),
                "rows_kept": len(keep), "proteins_kept": int(keep.acc.nunique()),
                "date_min": keep.date.min(), "date_max": keep.date.max(),
                "rows_by_evidence": keep.ev.value_counts().to_dict()}
    log(f"[{tag}] {len(df):,} experimental rows; NOT {int(is_not.sum()):,}; IPI protein binding "
        f"{int(is_pb.sum()):,}; kept {len(keep):,} rows / {keep.acc.nunique():,} proteins; dates "
        f"{keep.date.min()}..{keep.date.max()}")
    return keep


def secondary_map() -> dict:
    """secondary accession -> list of current primary accessions (UniProt sec_ac.txt)."""
    m, started = {}, False
    import gzip
    for line in gzip.open(SRC_DIR / "sec_ac.txt.gz", "rt"):
        if line.startswith("_"):
            started = True
            continue
        p = line.split()
        if started and len(p) == 2:
            m.setdefault(p[0], []).append(p[1])
    return m


def uniprot_fetch(accs: list[str], batch: int = 400) -> "pd.DataFrame":
    """accession, primary accession, taxon, organism, length, sequence for current UniProtKB entries."""
    import pandas as pd
    import urllib.request
    rows, release = [], None
    fields = "accession,organism_id,organism_name,length,sequence"
    for i in range(0, len(accs), batch):
        chunk = accs[i:i + batch]
        url = (f"https://rest.uniprot.org/uniprotkb/accessions?accessions={','.join(chunk)}"
               f"&format=json&fields={fields}&size={batch}")
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    release = r.headers.get("X-UniProt-Release"), r.headers.get("X-UniProt-Release-Date")
                    d = json.load(r)
                break
            except Exception as e:  # noqa: BLE001 - retry any transport error
                log(f"uniprot batch {i // batch}: {e}; retry {attempt + 1}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"uniprot batch {i // batch} failed")
        for r in d["results"]:
            o = r.get("organism", {})
            rows.append({"primary": r["primaryAccession"], "reviewed": "Swiss-Prot" in r.get("entryType", ""),
                         "taxon": o.get("taxonId"),
                         "organism": o.get("scientificName"),
                         # UniProt TSV 'Organism' string, the form stored in GO-GPT train.parquet
                         "organism_full": o.get("scientificName", "") + "".join(
                             f" ({x})" for x in [o.get("commonName")] + o.get("synonyms", []) if x),
                         "length": r["sequence"]["length"],
                         "seq": r["sequence"]["value"]})
        if (i // batch) % 10 == 0:
            log(f"uniprot: {min(i + batch, len(accs)):,}/{len(accs):,} requested, {len(rows):,} entries")
        time.sleep(0.5)
    return pd.DataFrame(rows).drop_duplicates("primary"), release


def map_terms(gt, obo: Path, label: str):
    """Rows of gt (acc, term, aspect, date) with terms replaced by canonical ids of `obo`; unmapped dropped."""
    from cafa6.knn_sharp import NS, Ontology
    onto = Ontology(obo)
    asp_of_ns = {"molecular_function": "MFO", "biological_process": "BPO", "cellular_component": "CCO"}
    ns_of = {t: asp_of_ns[NS[onto.ns[i]]] for i, t in enumerate(onto.terms)}
    canon = {t: [onto.terms[j] for j in onto.canon(t)] for t in gt.term.unique()}
    m = gt.assign(cterm=gt.term.map(canon)).explode("cterm")
    miss = m[m.cterm.isna()]
    m = m.dropna(subset=["cterm"])
    alt = int((m.cterm != m.term).sum())
    wrong_ns = m.cterm.map(ns_of) != m.aspect
    m = m[~wrong_ns]
    m = m.assign(term=m.cterm).drop(columns="cterm")
    join = lambda s: ",".join(sorted(set(s)))  # noqa: E731
    out = (m.groupby(["acc", "term", "aspect", "split"], as_index=False)
           .agg(date=("date", "min"), ev=("ev", join), by=("by", join)).sort_values(["acc", "aspect", "term"]))
    info = {"obo": str(obo.relative_to(ROOT)), "data_version": next(
        (l.split(": ", 1)[1].strip() for l in open(obo) if l.startswith("data-version")), None),
        "terms_in": int(gt.term.nunique()), "terms_not_in_release": int(miss.term.nunique()),
        "rows_not_in_release": len(miss), "rows_alt_id_mapped": alt, "rows_namespace_mismatch": int(wrong_ns.sum()),
        "rows_out": len(out), "proteins_out": int(out.acc.nunique())}
    log(f"[{label}] {obo.name} ({info['data_version']}): {info['terms_not_in_release']:,}/{info['terms_in']:,} new "
        f"terms not in release ({len(miss):,} rows), {alt:,} rows via alt_id, {int(wrong_ns.sum())} namespace "
        f"mismatches; {len(out):,} rows / {out.acc.nunique():,} proteins")
    return out, info, sorted(miss.term.unique())


def cmd_build(a):
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"cutoff": CUTOFF, "evidence_codes": EVIDENCE,
           "excluded": "qualifier containing NOT; IPI annotations to GO:0005515 (protein binding)",
           "sources": {t: json.loads((SRC_DIR / f"meta_{t}.json").read_text()) for t in SOURCES}}
    for t in SOURCES:
        rep["sources"][t].pop("header", None)
    cur, old = load_exp("current", rep), load_exp("r219", rep)
    gen = rep["sources"]["current"].get("last_modified")
    sec = secondary_map()
    log(f"sec_ac.txt: {len(sec):,} secondary accessions")

    # prior knowledge at the cutoff: release 219 (accessions also mapped to current primaries) + dated rows
    old_pa = old[["acc", "aspect"]].drop_duplicates()
    old_sec = old_pa[old_pa.acc.isin(sec)]
    old_mapped = old_sec.assign(acc=old_sec.acc.map(sec)).explode("acc")
    prior_219 = set(map(tuple, pd.concat([old_pa, old_mapped])[["acc", "aspect"]].values))
    dated = cur[cur.date <= CUTOFF]
    prior_dated = set(map(tuple, dated[["acc", "aspect"]].drop_duplicates().values))
    prior = prior_219 | prior_dated
    prior_acc = {p for p, _ in prior}
    log(f"prior (protein, aspect): release 219 {len(prior_219):,} (incl. {len(old_mapped):,} via secondary->primary "
        f"mapping), current dated <= {CUTOFF} {len(prior_dated):,}, union {len(prior):,}; proteins {len(prior_acc):,}")
    rep["prior"] = {"pairs_release219": len(prior_219), "pairs_release219_via_secondary": len(old_mapped),
                    "pairs_dated": len(prior_dated), "pairs_union": len(prior), "proteins": len(prior_acc),
                    "pairs_dated_not_in_219": len(prior_dated - prior_219),
                    "pairs_219_not_dated_in_current": len(prior_219 - prior_dated)}

    late = cur[cur.date > CUTOFF]
    key = list(zip(late.acc, late.aspect))
    in219 = pd.Series([k in prior_219 for k in key], index=late.index)
    indated = pd.Series([k in prior_dated for k in key], index=late.index)
    new = late[~in219 & ~indated]
    old_trip = set(zip(old.acc, old.aspect, old.term))
    reset = sum((x, y, z) in old_trip for x, y, z in zip(late.acc, late.aspect, late.term))
    rep["late_rows"] = {"rows_dated_after_cutoff": len(late), "rows_triple_already_in_219": int(reset),
                        "rows_pair_in_219_only": int((in219 & ~indated).sum()),
                        "rows_pair_dated_before": int(indated.sum()), "rows_new_pair": len(new)}
    log(f"rows dated > {CUTOFF}: {len(late):,}; same (protein, aspect, term) already in release 219 {reset:,}; "
        f"(protein, aspect) known only from release 219 {int((in219 & ~indated).sum()):,}; known from dates "
        f"{int(indated.sum()):,}; new-pair rows {len(new):,} ({new.acc.nunique():,} proteins)")

    new = new.assign(aspect=new.aspect.map({"F": "MFO", "P": "BPO", "C": "CCO"}),
                     date=pd.to_datetime(new.date, format="%Y%m%d").dt.strftime("%Y-%m-%d"))
    new["split"] = ["lk" if p in prior_acc else "nk" for p in new.acc]
    rep["new_by_evidence"] = new.ev.value_counts().to_dict()
    rep["new_by_assigned_by"] = new.by.value_counts().head(20).to_dict()
    gt = new[["acc", "term", "aspect", "date", "split", "ev", "by", "taxon"]]

    # disjointness with GO-GPT train/val/test and TV v2: accession (with secondary mapping) and sequence
    src = pd.read_parquet(GOGPT, columns=["id", "seq", "split", "taxon", "organism"])
    used = set(src.id) | set(open(TV2_IDS).read().split())
    used_primary = used | {p for s in used if s in sec for p in sec[s]}
    sec_of = {}
    for s, ps in sec.items():
        for p in ps:
            sec_of.setdefault(p, []).append(s)
    cand = sorted(gt.acc.unique())
    drop_acc = {p for p in cand if p in used_primary or any(s in used for s in sec_of.get(p, ()))}
    log(f"candidates {len(cand):,}; in GO-GPT/TV v2 by accession (incl. secondary) {len(drop_acc):,}")
    cand = [p for p in cand if p not in drop_acc]

    up, release = uniprot_fetch(cand)
    rep["uniprot_release"] = release
    req = set(cand)
    got = up[up.primary.isin(req)]
    moved = up[~up.primary.isin(req)]
    missing = req - set(got.primary)
    log(f"UniProt {release}: {len(got):,} entries returned under the requested accession, {len(moved):,} under "
        f"another primary accession (dropped), {len(missing):,} requested accessions not returned as primary")
    used_md5 = set(src.seq.map(lambda s: hashlib.md5(s.encode()).hexdigest()))
    got = got.assign(md5=got.seq.map(lambda s: hashlib.md5(s.encode()).hexdigest()))
    same_seq = got.md5.isin(used_md5)
    log(f"identical sequence to a GO-GPT/TV v2 protein: {int(same_seq.sum()):,} removed")
    got = got[~same_seq]
    rep["disjoint"] = {"candidates": int(gt.acc.nunique()), "removed_accession": len(drop_acc),
                       "not_current_primary": len(missing) + len(moved), "removed_identical_sequence": int(same_seq.sum()),
                       "kept": len(got)}
    gt = gt[gt.acc.isin(set(got.primary))]

    # GO term mapping: the scorer's release (main GT) and the current GO release (second GT)
    gt_main, info_main, miss_main = map_terms(gt, OBO, "gogpt release")
    cur_obo = OUT / "ref" / "go-basic_current.obo"
    gt_cur, info_cur, miss_cur = map_terms(gt, cur_obo, "current GO")
    rep["go_mapping"] = {"gogpt": info_main, "current": info_cur}
    (OUT / "terms_not_in_gogpt_release.txt").write_text("\n".join(miss_main) + "\n")

    keep = set(gt_main.acc)
    gt_cur = gt_cur[gt_cur.acc.isin(keep)]
    cols = ["acc", "term", "aspect", "date"]
    hdr = ["protein", "term", "aspect", "date"]
    for split in ("nk", "lk"):
        gt_main[gt_main.split == split][cols].to_csv(OUT / f"gt_{split}.tsv", sep="\t", index=False, header=hdr)
        gt_cur[gt_cur.split == split][cols].to_csv(OUT / f"gt_{split}_go_current.tsv", sep="\t", index=False,
                                                    header=hdr)
    gt_main.rename(columns={"acc": "protein", "ev": "evidence", "by": "assigned_by"})[
        ["protein", "term", "aspect", "date", "split", "evidence", "assigned_by"]].to_csv(
        OUT / "gt_annotations_full.tsv", sep="\t", index=False)
    prot = got[got.primary.isin(keep)].rename(columns={"primary": "accession"}).sort_values("accession")
    join = lambda s: ",".join(sorted(set(",".join(s).split(","))))  # noqa: E731
    asp = gt_main.groupby(["acc", "split"]).aspect.agg(join).unstack()
    prot["nk_aspects"] = prot.accession.map(asp.get("nk", pd.Series(dtype=str))).fillna("")
    prot["lk_aspects"] = prot.accession.map(asp.get("lk", pd.Series(dtype=str))).fillna("")
    prot["assigned_by"] = prot.accession.map(gt_main.groupby("acc").by.agg(join))
    prot["evidence"] = prot.accession.map(gt_main.groupby("acc").ev.agg(join))
    prot["hpa_only"] = prot.assigned_by.eq("HPA")
    dup = prot.md5.duplicated(keep=False)
    log(f"Swiss-Prot {int(prot.reviewed.sum()):,} / TrEMBL {int((~prot.reviewed).sum()):,}; HPA-only proteins "
        f"{int(prot.hpa_only.sum()):,}; proteins sharing an identical sequence with another fresh24 protein "
        f"{int(dup.sum()):,} ({prot.md5[dup].nunique():,} sequences)")
    rep["composition"] = {"swissprot": int(prot.reviewed.sum()), "trembl": int((~prot.reviewed).sum()),
                          "hpa_only": int(prot.hpa_only.sum()), "hpa_only_nk": int((prot.hpa_only &
                                                                                    prot.nk_aspects.ne("")).sum()),
                          "intra_identical_proteins": int(dup.sum()),
                          "assigned_by_rows": gt_main.by.str.split(",").explode().value_counts().head(15).to_dict()}
    with open(OUT / "fresh24.fasta", "w") as f:
        for p, s in zip(prot.accession, prot.seq):
            f.write(f">{p}\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n")
    INTER.mkdir(parents=True, exist_ok=True)
    prot.drop(columns=["seq"]).to_csv(INTER / "proteins_base.tsv", sep="\t", index=False)
    prot[["accession", "seq"]].to_parquet(INTER / "seqs.parquet", index=False)

    counts = {}
    for split in ("nk", "lk"):
        g = gt_main[gt_main.split == split]
        counts[split] = {x: {"proteins": int(g[g.aspect == x].acc.nunique()), "terms": int((g.aspect == x).sum())}
                         for x in ("MFO", "BPO", "CCO")}
        counts[split]["any"] = {"proteins": int(g.acc.nunique()), "terms": len(g)}
        log(f"split {split}: " + ", ".join(f"{x} {v['proteins']:,} proteins / {v['terms']:,} terms"
                                          for x, v in counts[split].items()))
    rep["counts"] = counts
    rep["proteins_total"] = len(prot)
    rep["dates_new"] = {"min": gt_main.date.min(), "max": gt_main.date.max(),
                        "by_year": gt_main.date.str[:4].value_counts().sort_index().to_dict()}
    rep["current_last_modified"] = gen
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"wrote {OUT}/fresh24.fasta ({len(prot):,} proteins), gt_nk.tsv, gt_lk.tsv, *_go_current.tsv, build_report.json")


# ---------------------------------------------------------------------------------------------- diamond
def cmd_diamond(a):
    import pandas as pd
    import tempfile
    src = pd.read_parquet(GOGPT, columns=["id", "seq", "split"])
    db = src[src.split.isin(["train", "val"])].drop_duplicates("id")
    dbsize = int(src.seq[src.split == "train"].str.len().sum())
    q = pd.read_parquet(INTER / "seqs.parquet")
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"DIAMOND: {len(q):,} fresh24 queries vs {len(db):,} GO-GPT train+val proteins, --more-sensitive -e 1e-3 "
        f"-k 51, --dbsize {dbsize:,} (train letters, as knn_trainval), {a.threads} threads")
    t = time.time()
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        tmp = Path(tmp)
        for d, idc, name in ((db, "id", "db.fa"), (q, "accession", "q.fa")):
            with open(tmp / name, "w") as f:
                for pid, s in zip(d[idc], d.seq):
                    f.write(f">{pid}\n{s}\n")
        subprocess.run([str(DIAMOND), "makedb", "--in", str(tmp / "db.fa"), "-d", str(tmp / "db"), "--quiet"],
                       check=True)
        subprocess.run([str(DIAMOND), "blastp", "-d", str(tmp / "db"), "-q", str(tmp / "q.fa"),
                        "-o", str(tmp / "hits.tsv"), "--more-sensitive", "-e", "1e-3", "-k", "51",
                        "-p", str(a.threads), "--quiet", "--tmpdir", str(tmp), "--dbsize", str(dbsize),
                        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "evalue", "qcovhsp"], check=True)
        h = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None, names=["q", "s", "pident", "bits", "evalue", "qcov"])
    h = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates(["q", "s"])
    h.to_parquet(INTER / "hits.parquet", index=False)
    log(f"DIAMOND done in {time.time() - t:.0f}s: {len(h):,} hits for {h.q.nunique():,}/{len(q):,} queries")


# ---------------------------------------------------------------------------------------------- stats
def cmd_stats(a):
    import pandas as pd
    from cafa6.temporal_val import ident_bins
    prot = pd.read_csv(INTER / "proteins_base.tsv", sep="\t", keep_default_na=False)
    h = pd.read_parquet(INTER / "hits.parquet")
    # GO-GPT `taxon` is an internal index; organisms are compared on the UniProt organism string
    src = pd.read_parquet(GOGPT, columns=["id", "organism"]).drop_duplicates("id").set_index("id").organism
    mx = h.groupby("q").pident.max()
    top = h.sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates("q").set_index("q")
    prot["best_identity"] = prot.accession.map(mx).fillna(0.0).values
    prot["bin"] = ident_bins(prot.best_identity.values).values
    prot["best_hit"] = prot.accession.map(top.s).fillna("")
    prot["best_hit_organism"] = prot.best_hit.map(src)
    prot["best_hit_same_organism"] = (prot.best_hit_organism == prot.organism_full) & (prot.best_identity > 0)
    log(f"query organism strings found among GO-GPT organisms: {prot.organism_full.isin(set(src)).mean():.1%}")
    cols = ["accession", "taxon", "organism", "length", "best_identity", "bin", "best_hit", "best_hit_same_organism",
            "nk_aspects", "lk_aspects", "reviewed", "hpa_only", "assigned_by", "evidence"]
    prot[cols].to_csv(OUT / "proteins.tsv", sep="\t", index=False)
    rep = json.loads((OUT / "build_report.json").read_text())

    order = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
    rows = []
    for split in ("nk", "lk", "all", "nk excl. HPA-only", "lk excl. HPA-only"):
        base = prot[~prot.hpa_only] if "excl" in split else prot
        s = split.split()[0]
        for asp in ("MFO", "BPO", "CCO", "any"):
            if s == "all":
                m = base.nk_aspects.ne("") | base.lk_aspects.ne("") if asp == "any" else \
                    base.nk_aspects.str.contains(asp) | base.lk_aspects.str.contains(asp)
            else:
                c = base[f"{s}_aspects"]
                m = c.ne("") if asp == "any" else c.str.contains(asp)
            d = base[m]
            vc = d.bin.value_counts()
            rows.append({"split": split, "aspect": asp, "n": len(d), **{b: int(vc.get(b, 0)) for b in order},
                         "same_org_best_hit": int(d.best_hit_same_organism.sum())})
    tab = pd.DataFrame(rows)
    org = prot.groupby(["taxon", "organism"]).size().sort_values(ascending=False).head(15)
    org_rows = []
    for (tx, name), n in org.items():
        d = prot[prot.taxon == tx]
        vc = d.bin.value_counts()
        org_rows.append({"taxon": tx, "organism": name, "n": int(n), "nk": int(d.nk_aspects.ne("").sum()),
                         "lk": int(d.lk_aspects.ne("").sum()), "hpa_only": int(d.hpa_only.sum()),
                         "swissprot": int(d.reviewed.sum()), "same_org_best_hit": int(d.best_hit_same_organism.sum()),
                         **{b: int(vc.get(b, 0)) for b in order}})
    org_tab = pd.DataFrame(org_rows)
    rep["identity_bins"] = tab.to_dict("records")
    rep["top_organisms"] = org_tab.to_dict("records")
    full100 = set(h.q[(h.pident >= 100) & (h.qcov >= 95)])
    b100 = prot[prot.bin == "100"]
    rep["bin100"] = {"n": len(b100), "qcov_ge95": int(b100.accession.isin(full100).sum()),
                     "hpa_only": int(b100.hpa_only.sum())}
    rep["same_org_share"] = float(prot.best_hit_same_organism.mean())
    rep["same_org_share_among_hits"] = float(prot.best_hit_same_organism[prot.best_identity > 0].mean())
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=1, default=str))

    def md(df):
        cols = list(df.columns)
        out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
        return "\n".join(out)

    c = rep["counts"]
    lines = ["# fresh24 benchmark: label-free statistics", "",
             "Built by `python -m cafa6.build_fresh24` (construction in `data/fresh24/README.md`). No model was "
             "scored on this set; the numbers below use sequences and DIAMOND hits only.", "",
             f"Proteins: {len(prot):,}. New experimental annotations dated {rep['dates_new']['min']} to "
             f"{rep['dates_new']['max']} (cutoff {CUTOFF}).", "",
             "## Proteins and terms per aspect and split", "",
             md(pd.DataFrame([{"split": s, "aspect": x, "proteins": c[s][x]["proteins"], "terms (unpropagated)":
                               c[s][x]["terms"]} for s in ("nk", "lk") for x in ("MFO", "BPO", "CCO", "any")])), "",
             "nk = no experimental annotation in any aspect at the cutoff; lk = experimental annotation in another "
             "aspect but not in this one. A protein can be lk in one aspect only if it had prior annotation elsewhere, "
             "so nk and lk proteins are disjoint.", "",
             "## Composition", "",
             f"Swiss-Prot {rep['composition']['swissprot']:,}, TrEMBL {rep['composition']['trembl']:,}. "
             f"HPA-only proteins (every new term assigned by HPA, all CCO, almost all rows from one load dated "
             f"2025-11): "
             f"{rep['composition']['hpa_only']:,} ({rep['composition']['hpa_only_nk']:,} nk). Proteins sharing an "
             f"identical sequence with another fresh24 protein: {rep['composition']['intra_identical_proteins']:,}.",
             "", "New-term rows by assigning group (top 15): " + ", ".join(
                 f"{k} {v:,}" for k, v in rep["composition"]["assigned_by_rows"].items()), "",
             "## Best DIAMOND identity to GO-GPT train+val (per aspect and split)", "",
             "DIAMOND --more-sensitive -e 1e-3 -k 51, --dbsize = GO-GPT train letters; best identity = max pident "
             "over hits (0 = no hit); same_org_best_hit = the top-bitscore hit has the query's UniProt organism "
             "string (GO-GPT stores organisms as strings).", "",
             md(tab), "",
             f"100% bin: {rep['bin100']['n']:,} proteins, of which {rep['bin100']['qcov_ge95']:,} have a 100% hit "
             f"covering >= 95% of the query (HSP coverage) and {rep['bin100']['hpa_only']:,} are HPA-only; the rest "
             "are fragments or isoforms matching part of a GO-GPT protein.", "",
             f"Same-organism share (top hit from the same organism): {rep['same_org_share']:.1%} of all proteins, "
             f"{rep['same_org_share_among_hits']:.1%} of proteins with a hit.", "",
             "## Top 15 organisms", "", md(org_tab), "",
             "## Dates of the new annotations (year of GAF column 14, GO-GPT-release GT rows)", "",
             md(pd.DataFrame([rep["dates_new"]["by_year"]])), "",
             "## Construction counts", "", "```", json.dumps({k: rep[k] for k in
                                                              ("prior", "late_rows", "disjoint", "go_mapping")},
                                                             indent=1, default=str), "```", ""]
    STATS_MD.write_text("\n".join(lines))
    log(f"wrote {OUT}/proteins.tsv and {STATS_MD}")
    print(md(tab))
    print(md(org_tab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["fetch", "build", "diamond", "stats"])
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--conns", type=int, default=20, help="parallel range requests per source")
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    for s in a.steps:
        globals()[f"cmd_{s}"](a)


if __name__ == "__main__":
    main()
