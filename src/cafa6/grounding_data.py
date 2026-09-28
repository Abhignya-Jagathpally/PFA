"""Build grounding resources (GO <-> ligands, UniProt features, reference-proteome GOA) under data/grounding/.

Steps (idempotent: every output that already exists is skipped; delete it to rebuild):

* ``static``  - external2go mappings (interpro2go, ec2go, rhea2go), Rhea TSVs + reaction->ChEBI participants
  (Rhea REST export), GO (go.obo for namespaces/names, go-plus.json.gz for the CHEBI logical definitions that
  plain go.obo does not carry), ChEBI flat files (names, is_a relations, SMILES streamed from structures.tsv.gz),
  ExPASy ENZYME (EC names + transferred entries). Products: interpro2go.tsv, ec2go.tsv, rhea2go.tsv,
  rhea_participants.tsv, go2chebi.tsv, chebi.tsv, ec.tsv and the final go2ligand.tsv
  [go, go_name, chebi, name, smiles, source(go_logical_def|rhea|ec), relation, via, ubiquitous, smiles_from].
  BRENDA needs an account; it is only stubbed (env BRENDA_EMAIL/BRENDA_PASSWORD) and Rhea + ExPASy ENZYME are
  used as the open substitute.
* ``uniprot`` - UniProt REST features for every accession in data/proc/train.parquet, data/proc_gogpt/train.parquet
  and data/raw/Test/testsuperset.fasta -> uniprot_features.parquet
  [id, interpro, pfam, ec, binding_chebi, cofactor_chebi, taxon, found, primary_acc, feature_source]. Resumable via
  uniprot_partial.tsv / uniprot_done.txt (+ uniprot_secondary.tsv, uniprot_uniparc.tsv). Merged (secondary)
  accessions are resolved with a per-accession ``sec_acc:`` search; deleted entries (found=False) get InterPro/Pfam
  from UniParc by sequence MD5 (feature_source=uniparc) and empty curated lists; otherwise empty lists.

  LEAKAGE NOTE: binding_chebi, cofactor_chebi and ec are *curated* UniProt annotations and must ONLY be used for
  training/reference proteins downstream (never as inputs for val/test/query proteins). interpro and pfam are
  computational (InterProScan) matches and may be used for all proteins.
* ``goa``     - GOA GAF files for 10 reference proteomes -> goa_reference.parquet
  [id, go, evidence, date(YYYYMMDD int), taxon, aspect]. All evidence codes kept (filter downstream);
  NOT-qualified annotations are dropped; only DB == UniProtKB rows.

Usage: PYTHONPATH=src python -m cafa6.grounding_data [--step all|static|uniprot|goa|readme]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from cafa6.data import ROOT

OUT = ROOT / "data" / "grounding"
RAW = OUT / "raw"
LOGS = ROOT / "logs"
MANIFEST = OUT / "sources.json"
MIN_FREE = 5 * 1024**3
BIG = 50 * 1024**2

GO_EXT = "http://current.geneontology.org/ontology/external2go/"
RHEA_TSV = "https://ftp.expasy.org/databases/rhea/tsv/"
CHEBI_FLAT = "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/"
GOA = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/"
UNIPROT = "https://rest.uniprot.org/uniprotkb/"
UNIPROT_FIELDS = "accession,xref_interpro,xref_pfam,ec,ft_binding,cc_cofactor,organism_id"
ACC_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")

RO_LABELS = {
    "RO_0004009": "has_primary_input", "RO_0004008": "has_primary_output",
    "RO_0004007": "has_primary_input_or_output", "RO_0000057": "has_participant",
    "RO_0002332": "regulates_levels_of", "RO_0012001": "has_small_molecule_activator",
    "RO_0012000": "has_small_molecule_regulator", "RO_0002608": "process_has_causal_agent",
    "RO_0002505": "has_intermediate", "RO_0002473": "composed_primarily_of",
    "RO_0002588": "results_in_assembly_of", "BFO_0000051": "has_part",
}

GOA_SOURCES = {
    9606: ("HUMAN/goa_human.gaf.gz", "human"),
    10090: ("MOUSE/goa_mouse.gaf.gz", "mouse"),
    3702: ("ARABIDOPSIS/goa_arabidopsis.gaf.gz", "arabidopsis"),
    559292: ("YEAST/goa_yeast.gaf.gz", "yeast"),
    10116: ("RAT/goa_rat.gaf.gz", "rat"),
    284812: ("proteomes/78.S_pombe.goa", "s_pombe"),
    83333: ("proteomes/18.E_coli_MG1655.goa", "e_coli_k12"),
    7227: ("FLY/goa_fly.gaf.gz", "fly"),
    6239: ("WORM/goa_worm.gaf.gz", "worm"),
    83332: ("proteomes/30.M_tuberculosis_ATCC_25618.goa", "m_tuberculosis_h37rv"),
}

log = logging.getLogger("grounding")


def setup_logging(step: str):
    LOGS.mkdir(exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not log.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    name = "grounding_uniprot.log" if step == "uniprot" else "grounding_data.log"
    fh = logging.FileHandler(LOGS / name)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return fh


# ----------------------------------------------------------------------------------------------- utilities

def free_bytes() -> int:
    return shutil.disk_usage(Path.home()).free


def ensure_space(nbytes: int, what: str):
    free = free_bytes()
    log.info("disk check for %s: need %.1f MB, free %.2f GB", what, nbytes / 1e6, free / 1024**3)
    if free - nbytes < MIN_FREE:
        raise SystemExit(f"STOP: downloading {what} would leave < 5 GB free ({free / 1024**3:.2f} GB free now)")


def manifest_update(key: str, **info):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m[key] = {**m.get(key, {}), **info}
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True))


def http_get(url: str, *, session: requests.Session | None = None, stream: bool = False, params=None,
             timeout: int = 300, tries: int = 8) -> requests.Response:
    """GET with retries/backoff; DNS failures on this host are intermittent, so connection errors are retried too."""
    s = session or requests
    for k in range(tries):
        try:
            r = s.get(url, params=params, stream=stream, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = int(r.headers.get("Retry-After", 0) or 0) or min(2 ** k * 2, 90)
                log.warning("HTTP %s for %s; retry in %ss", r.status_code, url[:120], wait)
                r.close()
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            wait = min(2 ** k * 2, 90)
            log.warning("%s for %s; retry in %ss", type(e).__name__, url[:120], wait)
            time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def remote_size(url: str) -> int | None:
    for k in range(4):
        try:
            r = requests.head(url, timeout=60, allow_redirects=True)
            n = r.headers.get("content-length")
            return int(n) if n else None
        except requests.RequestException:
            time.sleep(2 ** k)
    return None


def download(urls: str | list[str], dest: Path, gzip_on_save: bool = False) -> Path:
    """Download the first working URL to dest (atomic via .part). gzip_on_save compresses plain-text sources."""
    if dest.exists():
        return dest
    urls = [urls] if isinstance(urls, str) else urls
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in urls:
        n = remote_size(url)
        if n and n > BIG:
            ensure_space(n, url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            r = http_get(url, stream=True, tries=5)
            opener = (lambda p: gzip.open(p, "wb", compresslevel=6)) if gzip_on_save else (lambda p: open(p, "wb"))
            with opener(tmp) as fh:
                for ch in r.iter_content(1 << 20):
                    fh.write(ch)
            tmp.rename(dest)
            manifest_update(dest.name, url=url, downloaded=dt.date.today().isoformat(),
                            remote_bytes=n, stored_bytes=dest.stat().st_size)
            log.info("downloaded %s -> %s (%.1f MB stored)", url, dest.relative_to(ROOT), dest.stat().st_size / 1e6)
            return dest
        except Exception as e:  # try next mirror
            last = e
            log.warning("download failed from %s: %s", url, e)
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"all mirrors failed for {dest.name}: {last}")


def open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.suffix == ".gz" else open(p, encoding="utf-8")


def write_tsv(df: pd.DataFrame, p: Path):
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.rename(p)
    log.info("wrote %s (%d rows)", p.relative_to(ROOT), len(df))


def write_parquet(df: pd.DataFrame, p: Path):
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(p)
    log.info("wrote %s (%d rows)", p.relative_to(ROOT), len(df))


# ----------------------------------------------------------------------------------------------- static

def parse_external2go(p: Path, prefix: str, col: str) -> pd.DataFrame:
    rows = []
    with open_text(p) as fh:
        for line in fh:
            if line.startswith("!") or " > " not in line:
                continue
            left, right = line.rstrip("\n").split(" > ", 1)
            key = left.split()[0]
            if not key.startswith(prefix):
                continue
            go = right.rsplit(";", 1)[-1].strip()
            if go.startswith("GO:"):
                rows.append((key[len(prefix):], go))
    return pd.DataFrame(rows, columns=[col, "go"]).drop_duplicates()


def build_external2go():
    for name, prefix, col, fmt in (("interpro2go", "InterPro:", "interpro", "{}"), ("ec2go", "EC:", "ec", "{}"),
                                   ("rhea2go", "RHEA:", "rhea", "RHEA:{}")):
        out = OUT / f"{name}.tsv"
        if out.exists():
            continue
        src = download(GO_EXT + name, RAW / f"{name}.gz", gzip_on_save=True)
        df = parse_external2go(src, prefix, col)
        df[col] = df[col].map(fmt.format)
        write_tsv(df, out)


def build_rhea():
    for f in ("rhea-chebi-smiles.tsv", "rhea-directions.tsv", "rhea2ec.tsv", "rhea-ec-iubmb.tsv",
              "chebiId_name.tsv", "rhea2go.tsv", "rhea-relationships.tsv"):
        download(RHEA_TSV + f, RAW / f"{f}.gz", gzip_on_save=True)
    out = OUT / "rhea_participants.tsv"
    if out.exists():
        return
    ensure_space(10 * 1024**2, "Rhea REST export")
    r = http_get("https://www.rhea-db.org/rhea/", timeout=600,
                 params={"query": "", "columns": "rhea-id,chebi-id,equation", "format": "tsv", "limit": 1000000})
    raw = RAW / "rhea_reactions_chebi.tsv.gz"
    with gzip.open(raw, "wt") as fh:
        fh.write(r.text)
    manifest_update(raw.name, url=r.url, downloaded=dt.date.today().isoformat(), stored_bytes=raw.stat().st_size)
    rows = []
    for line in r.text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1]:
            continue
        for c in parts[1].split(";"):
            if c.startswith("CHEBI:"):
                rows.append((parts[0], c))
    write_tsv(pd.DataFrame(rows, columns=["rhea", "chebi"]).drop_duplicates(), out)


def rhea_master_map() -> dict[str, str]:
    """Directed/bidirectional Rhea ids -> undirected master id (participants are exported per master)."""
    d = pd.read_csv(RAW / "rhea-directions.tsv.gz", sep="\t", dtype=str)
    m = {}
    for row in d.itertuples(index=False):
        for x in row:
            m[f"RHEA:{x}"] = f"RHEA:{row[0]}"
    return m


def build_enzyme():
    out = OUT / "ec.tsv"
    if out.exists():
        return
    src = download("https://ftp.expasy.org/databases/enzyme/enzyme.dat", RAW / "enzyme.dat.gz", gzip_on_save=True)
    rows, cur = [], None
    with open_text(src) as fh:
        for line in fh:
            tag, val = line[:2], line[5:].strip()
            if tag == "ID":
                cur = {"ec": val, "name": "", "transferred_to": ""}
            elif tag == "DE" and cur is not None:
                cur["name"] += (" " if cur["name"] else "") + val
            elif tag == "//" and cur is not None:
                m = re.match(r"Transferred entry: (.*)\.$", cur["name"])
                if m:
                    cur["transferred_to"] = ";".join(re.findall(r"\d+\.\d+\.\d+\.n?\d+", m.group(1)))
                cur["name"] = cur["name"].rstrip(".")
                rows.append(cur)
                cur = None
    write_tsv(pd.DataFrame(rows), out)


def go_namespaces() -> tuple[dict[str, str], dict[str, str]]:
    src = download(["http://purl.obolibrary.org/obo/go.obo", "https://current.geneontology.org/ontology/go.obo"],
                   RAW / "go.obo.gz", gzip_on_save=True)
    ns, names, cur = {}, {}, None
    with open_text(src) as fh:
        for line in fh:
            if line.startswith("[Term]"):
                cur = None
            elif line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif cur and line.startswith("name: "):
                names[cur] = line[6:].strip()
            elif cur and line.startswith("namespace: "):
                ns[cur] = line[11:].strip()
            elif cur and line.startswith("alt_id: "):
                pass
    return ns, names


def build_go2chebi():
    out = OUT / "go2chebi.tsv"
    if out.exists():
        return
    src = download(["http://purl.obolibrary.org/obo/go/extensions/go-plus.json.gz",
                    "https://current.geneontology.org/ontology/extensions/go-plus.json.gz"], RAW / "go-plus.json.gz")
    with gzip.open(src, "rt") as fh:
        g = json.load(fh)["graphs"][0]
    short = lambda u: u.rsplit("/", 1)[-1]
    labels = {short(n["id"]): n.get("lbl") for n in g["nodes"] if n.get("lbl")}
    rel = lambda p: RO_LABELS.get(short(p)) or (labels.get(short(p)) or short(p)).replace(" ", "_")
    rows = []
    for a in g.get("logicalDefinitionAxioms", []):
        go = short(a["definedClassId"])
        if not go.startswith("GO_"):
            continue
        for r in a.get("restrictions", []):
            f = short(r["fillerId"])
            if f.startswith("CHEBI_"):
                rows.append((go.replace("_", ":"), f.replace("_", ":"), rel(r["propertyId"]), "logical_def"))
    for e in g["edges"]:
        s, o = short(e["sub"]), short(e["obj"])
        if s.startswith("GO_") and o.startswith("CHEBI_") and e["pred"] != "is_a":
            rows.append((s.replace("_", ":"), o.replace("_", ":"), rel(e["pred"]), "subclass_restriction"))
    df = pd.DataFrame(rows, columns=["go", "chebi", "relation", "axiom"])
    df = df.groupby(["go", "chebi", "relation"], as_index=False).axiom.agg(lambda x: ";".join(sorted(set(x))))
    write_tsv(df, out)


def chebi_int(c: str) -> int:
    return int(c.split(":")[1])


def build_chebi():
    out = OUT / "chebi.tsv"
    if out.exists():
        return
    comp = download(CHEBI_FLAT + "compounds.tsv.gz", RAW / "chebi_compounds.tsv.gz")
    relf = download(CHEBI_FLAT + "relation.tsv.gz", RAW / "chebi_relation.tsv.gz")
    reltype = download(CHEBI_FLAT + "relation_type.tsv.gz", RAW / "chebi_relation_type.tsv.gz")

    need = set(pd.read_csv(OUT / "go2chebi.tsv", sep="\t").chebi) | set(pd.read_csv(OUT / "rhea_participants.tsv", sep="\t").chebi)
    need = {chebi_int(c) for c in need}

    cp = pd.read_csv(comp, sep="\t", usecols=["id", "name", "status_id", "parent_id", "chebi_accession"],
                     dtype={"parent_id": "Int64"}, keep_default_na=False, na_values=[""])
    names = dict(zip(cp.id, cp.name))
    parent = {i: int(p) for i, p in zip(cp.id, cp.parent_id) if pd.notna(p)}
    resolve = lambda i: parent.get(i, i)

    rt = pd.read_csv(reltype, sep="\t")
    isa_id = int(rt.loc[rt.iloc[:, 1].astype(str).str.lower() == "is_a"].iloc[0, 0])
    rel = pd.read_csv(relf, sep="\t", usecols=["relation_type_id", "init_id", "final_id"])
    rel = rel[rel.relation_type_id == isa_id]
    children = collections.defaultdict(list)
    for c, p in zip(rel.init_id, rel.final_id):
        children[p].append(c)

    # candidate ids whose SMILES we may need: needed ids, their parents (secondary ids) and is_a descendants (depth<=3)
    cand, frontier = set(), {resolve(i) for i in need} | need
    cand |= frontier
    for _ in range(3):
        frontier = {c for p in frontier for c in children.get(p, [])} - cand
        cand |= frontier

    smiles = {}
    with open_text(RAW / "rhea-chebi-smiles.tsv.gz") as fh:
        for line in fh:
            k, _, s = line.rstrip("\n").partition("\t")
            if k.startswith("CHEBI:") and s:
                smiles[chebi_int(k)] = s
    rhea_ids = set(smiles)
    n = remote_size(CHEBI_FLAT + "structures.tsv.gz") or 0
    ensure_space(n, "ChEBI structures.tsv.gz (stream-parsed, not stored)")
    csv.field_size_limit(1 << 30)
    r = http_get(CHEBI_FLAT + "structures.tsv.gz", stream=True, timeout=900)
    r.raw.decode_content = False
    got = 0
    with gzip.GzipFile(fileobj=r.raw) as gz:
        reader = csv.reader(io.TextIOWrapper(gz, encoding="utf-8", errors="replace"), delimiter="\t")
        hdr = next(reader)
        ic, isml, idef = hdr.index("compound_id"), hdr.index("smiles"), hdr.index("default_structure")
        for row in reader:
            if len(row) <= idef:
                continue
            cid = int(row[ic])
            if cid in cand and row[isml] and (row[idef] == "true" or cid not in smiles) and cid not in rhea_ids:
                smiles[cid] = row[isml]
                got += 1
    manifest_update("chebi_structures.tsv.gz (streamed)", url=CHEBI_FLAT + "structures.tsv.gz",
                    downloaded=dt.date.today().isoformat(), remote_bytes=n, stored_bytes=0)
    log.info("ChEBI structures: %d SMILES taken from structures.tsv.gz", got)

    rhea_names = {}
    with open_text(RAW / "chebiId_name.tsv.gz") as fh:
        for line in fh:
            k, _, v = line.rstrip("\n").partition("\t")
            rhea_names[k] = v.strip()

    rp = pd.read_csv(OUT / "rhea_participants.tsv", sep="\t")
    rxn_count = collections.Counter(chebi_int(c) for c in rp.chebi)

    def fallback(i: int):
        """Class terms (e.g. CHEBI:30413 heme) often lack a structure: borrow the is_a descendant (depth <= 3) with
        SMILES that is used in the most Rhea reactions (heme -> heme b), else the shallowest one."""
        layer, hits = [i], []
        for depth in range(1, 4):
            layer = [c for p in layer for c in children.get(p, [])]
            hits += [(c, depth) for c in layer if c in smiles]
        if not hits:
            return "", ""
        best = sorted(hits, key=lambda h: (-rxn_count[h[0]], h[1], "*" in smiles[h[0]], len(smiles[h[0]]), h[0]))[0][0]
        return smiles[best], f"descendant:CHEBI:{best}"

    rows = []
    for i in sorted(need):
        j = resolve(i)
        s, src = (smiles[i], "self") if i in smiles else ((smiles[j], f"parent:CHEBI:{j}") if j in smiles else fallback(j))
        nm = rhea_names.get(f"CHEBI:{i}") or names.get(i) or names.get(j) or ""
        rows.append((f"CHEBI:{i}", nm, s, src))
    write_tsv(pd.DataFrame(rows, columns=["chebi", "name", "smiles", "smiles_from"]), out)


def build_go2ligand():
    out = OUT / "go2ligand.tsv"
    if out.exists():
        return
    ns, go_names = go_namespaces()
    mf = {g for g, n in ns.items() if n == "molecular_function"}
    master = rhea_master_map()
    part = pd.read_csv(OUT / "rhea_participants.tsv", sep="\t")
    rxn2chebi = part.groupby("rhea").chebi.agg(list).to_dict()
    n_rxn = part.groupby("chebi").rhea.nunique()
    ubiq = set(n_rxn[n_rxn >= 300].index)
    log.info("ubiquitous ChEBI (in >=300 Rhea reactions): %d", len(ubiq))

    rows = []
    g2c = pd.read_csv(OUT / "go2chebi.tsv", sep="\t")
    for go, c, rel in g2c[g2c.go.isin(mf)][["go", "chebi", "relation"]].itertuples(index=False):
        rows.append((go, c, "go_logical_def", rel, ""))

    r2g = pd.read_csv(OUT / "rhea2go.tsv", sep="\t")
    for rh, go in r2g.itertuples(index=False):
        m = master.get(rh, rh)
        for c in rxn2chebi.get(m, []):
            rows.append((go, c, "rhea", "reaction_participant", m))

    ec = pd.read_csv(OUT / "ec.tsv", sep="\t", dtype=str, keep_default_na=False)
    moved = {e: t.split(";") for e, t in zip(ec.ec, ec.transferred_to) if t}
    ec2rhea = collections.defaultdict(set)
    d = pd.read_csv(RAW / "rhea2ec.tsv.gz", sep="\t", dtype=str)
    for rid, e in zip(d.RHEA_ID, d.ID):
        ec2rhea[e].add(master.get(f"RHEA:{rid}", f"RHEA:{rid}"))
    d = pd.read_csv(RAW / "rhea-ec-iubmb.tsv.gz", sep="\t", dtype=str)
    for rid, e in zip(d.REACTION_ID, d.EC):
        ec2rhea[e].add(master.get(f"RHEA:{rid}", f"RHEA:{rid}"))
    e2g = pd.read_csv(OUT / "ec2go.tsv", sep="\t", dtype=str)
    for e, go in e2g.itertuples(index=False):
        for e2 in moved.get(e, [e]):
            for m in sorted(ec2rhea.get(e2, ())):
                for c in rxn2chebi.get(m, []):
                    rows.append((go, c, "ec", "reaction_participant", f"EC:{e2}|{m}"))

    df = pd.DataFrame(rows, columns=["go", "chebi", "source", "relation", "via"])
    df = df[df.go.isin(mf)]
    df = (df.groupby(["go", "chebi", "source", "relation"], as_index=False)
          .via.agg(lambda x: ";".join(sorted(set(v for v in x if v))[:20])))
    ch = pd.read_csv(OUT / "chebi.tsv", sep="\t", keep_default_na=False)
    df = df.merge(ch, on="chebi", how="left")
    df["go_name"] = df.go.map(go_names)
    df["ubiquitous"] = df.chebi.isin(ubiq)
    df = df[["go", "go_name", "chebi", "name", "smiles", "source", "relation", "via", "ubiquitous", "smiles_from"]]
    write_tsv(df.sort_values(["go", "source", "chebi"]).reset_index(drop=True), out)


def brenda_stub():
    if os.environ.get("BRENDA_EMAIL") and os.environ.get("BRENDA_PASSWORD"):
        log.warning("BRENDA credentials found but the BRENDA SOAP client is not implemented (stub); "
                    "using Rhea + ExPASy ENZYME as the open substitute.")
    else:
        log.info("BRENDA skipped: BRENDA_EMAIL/BRENDA_PASSWORD not set (account required). "
                 "Using Rhea (rhea2ec, rhea-ec-iubmb, participants) + ExPASy ENZYME as the open substitute.")


def step_static():
    OUT.mkdir(parents=True, exist_ok=True)
    build_external2go()
    build_rhea()
    build_enzyme()
    build_go2chebi()
    build_chebi()
    brenda_stub()
    build_go2ligand()


# ----------------------------------------------------------------------------------------------- uniprot

def accession_sets() -> dict[str, list[str]]:
    tr = pd.read_parquet(ROOT / "data/proc/train.parquet", columns=["id", "split"])
    gg = pd.read_parquet(ROOT / "data/proc_gogpt/train.parquet", columns=["id", "split"])
    sets = {f"cafa_{s}": sorted(tr.id[tr.split == s]) for s in ("train", "val", "test")}
    sets.update({f"gogpt_{s}": sorted(gg.id[gg.split == s]) for s in ("train", "val", "test")})
    kag = []
    with open(ROOT / "data/raw/Test/testsuperset.fasta") as fh:
        for line in fh:
            if line.startswith(">"):
                kag.append(line[1:].split()[0])
    sets["kaggle_superset"] = kag
    return sets


BIND_RE = re.compile(r'/ligand_id="ChEBI:(CHEBI:\d+)"')
COF_RE = re.compile(r"Xref=ChEBI:(CHEBI:\d+)")


def parse_uniprot_tsv(text: str) -> list[list[str]]:
    out = []
    lines = text.splitlines()
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) < 7:
            continue
        acc, ipr, pf, ec, fb, cof, tax = p[:7]
        split = lambda s: ";".join(x for x in (t.strip() for t in s.split(";")) if x)
        uniq = lambda xs: ";".join(dict.fromkeys(xs))
        out.append([acc, split(ipr), split(pf), uniq(x.strip() for x in ec.split(";") if x.strip()),
                    uniq(BIND_RE.findall(fb)), uniq(COF_RE.findall(cof)), tax.strip()])
    return out


class UniprotFetcher:
    def __init__(self):
        self.partial = OUT / "uniprot_partial.tsv"
        self.done_f = OUT / "uniprot_done.txt"
        self.sec_f = OUT / "uniprot_secondary.tsv"
        self.lock = threading.Lock()
        self.local = threading.local()

    def session(self) -> requests.Session:
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s

    def fetch_batch(self, accs: list[str]) -> list[list[str]]:
        try:
            r = http_get(UNIPROT + "accessions", session=self.session(), timeout=180,
                         params={"accessions": ",".join(accs), "fields": UNIPROT_FIELDS, "format": "tsv"})
            return parse_uniprot_tsv(r.text)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and len(accs) > 1:
                h = len(accs) // 2
                return self.fetch_batch(accs[:h]) + self.fetch_batch(accs[h:])
            if e.response is not None and e.response.status_code == 400:
                log.warning("UniProt rejected accession %s", accs[0])
                return []
            raise

    def run_batches(self, accs: list[str], bs: int = 500, workers: int = 4):
        batches = [accs[i:i + bs] for i in range(0, len(accs), bs)]
        done = set(self.done_f.read_text().split()) if self.done_f.exists() else set()
        todo = [(i, b) for i, b in enumerate(batches) if str(i) not in done]
        log.info("UniProt: %d accessions, %d batches, %d already done, %d to fetch", len(accs), len(batches),
                 len(batches) - len(todo), len(todo))
        if todo and free_bytes() < MIN_FREE:
            raise SystemExit("STOP: < 5 GB free before UniProt fetch")
        t0, n = time.time(), 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(self.fetch_batch, b): i for i, b in todo}
            for f in as_completed(futs):
                i = futs[f]
                rows = f.result()
                with self.lock:
                    with open(self.partial, "a") as fh:
                        for r in rows:
                            fh.write("\t".join([str(i)] + r) + "\n")
                    with open(self.done_f, "a") as fh:
                        fh.write(f"{i}\n")
                n += 1
                if n % 20 == 0 or n == len(todo):
                    el = time.time() - t0
                    log.info("UniProt batches %d/%d (%.1f/s, eta %.0fs), free %.2f GB", n, len(todo), n / el,
                             (len(todo) - n) / max(n / el, 1e-9), free_bytes() / 1024**3)
                if n % 100 == 0 and free_bytes() < MIN_FREE:
                    ex.shutdown(cancel_futures=True)
                    raise SystemExit("STOP: free space dropped below 5 GB during UniProt fetch")

    def fetch_secondary(self, acc: str):
        r = http_get(UNIPROT + "search", session=self.session(), timeout=120,
                     params={"query": f"sec_acc:{acc}", "fields": UNIPROT_FIELDS, "format": "tsv", "size": 5})
        rows = parse_uniprot_tsv(r.text)
        return acc, rows[0] if rows else None

    def run_secondary(self, missing: list[str], workers: int = 4):
        done = {}
        if self.sec_f.exists():
            for line in self.sec_f.read_text().splitlines():
                p = line.split("\t")
                done[p[0]] = p
        todo = [a for a in missing if a not in done]
        log.info("UniProt secondary-accession pass: %d missing, %d already resolved/checked, %d to query",
                 len(missing), len(missing) - len(todo), len(todo))
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            for n, f in enumerate(as_completed([ex.submit(self.fetch_secondary, a) for a in todo]), 1):
                acc, row = f.result()
                with self.lock, open(self.sec_f, "a") as fh:
                    fh.write("\t".join([acc] + (row if row else [""] * 7)) + "\n")
                if n % 500 == 0 or n == len(todo):
                    log.info("secondary %d/%d (%.1f/s)", n, len(todo), n / (time.time() - t0))

    def fetch_uniparc(self, md5s: list[str]) -> list[tuple[str, str, str, str]]:
        r = http_get(UNIPROT.replace("uniprotkb/", "uniparc/") + "search", session=self.session(), timeout=300,
                     params={"query": " OR ".join(f"checksum:{m}" for m in md5s), "format": "json", "size": 500})
        out = []
        for e in r.json().get("results", []):
            md5 = hashlib.md5(e.get("sequence", {}).get("value", "").encode()).hexdigest().upper()
            ipr, pf = {}, {}
            for f in e.get("sequenceFeatures", []):
                if f.get("interproGroup", {}).get("id"):
                    ipr[f["interproGroup"]["id"]] = 1
                if f.get("database") == "Pfam":
                    pf[f["databaseId"]] = 1
            out.append((md5, e.get("uniParcId", ""), ";".join(ipr), ";".join(pf)))
        return out

    def run_uniparc(self, missing: list[str], seqs: dict[str, str], bs: int = 50, workers: int = 4):
        """Deleted/obsolete UniProtKB entries: computational InterPro/Pfam matches from UniParc by sequence MD5."""
        f = OUT / "uniprot_uniparc.tsv"
        done = set()
        if f.exists():
            done = {line.split("\t")[0] for line in f.read_text().splitlines()}
        todo = [a for a in missing if a not in done and a in seqs]
        log.info("UniParc pass: %d missing, %d already done, %d with sequence to query", len(missing),
                 len(done & set(missing)), len(todo))
        by_md5 = collections.defaultdict(list)
        for a in todo:
            by_md5[hashlib.md5(seqs[a].upper().encode()).hexdigest().upper()].append(a)
        keys = sorted(by_md5)
        batches = [keys[i:i + bs] for i in range(0, len(keys), bs)]
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(self.fetch_uniparc, b): b for b in batches}
            for n, fu in enumerate(as_completed(futs), 1):
                res = {m: (u, i, p) for m, u, i, p in fu.result()}
                with self.lock, open(f, "a") as fh:
                    for m in futs[fu]:
                        u, i, p = res.get(m, ("", "", ""))
                        for a in by_md5[m]:
                            fh.write(f"{a}\t{u}\t{i}\t{p}\n")
                if n % 20 == 0 or n == len(batches):
                    log.info("UniParc batches %d/%d", n, len(batches))


def load_sequences() -> dict[str, str]:
    seqs = {}
    for p in ("data/proc/test.parquet", "data/proc_gogpt/train.parquet", "data/proc/train.parquet"):
        d = pd.read_parquet(ROOT / p, columns=["id", "seq"])
        seqs.update(zip(d.id, d.seq))
    return seqs


def step_uniprot():
    ulog = str(LOGS / "grounding_uniprot.log")
    if not any(getattr(h, "baseFilename", None) == ulog for h in log.handlers):
        fh = logging.FileHandler(ulog)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    out = OUT / "uniprot_features.parquet"
    if out.exists():
        log.info("uniprot_features.parquet exists; skipping fetch")
    else:
        sets = accession_sets()
        allacc = sorted(set().union(*sets.values()))
        valid = [a for a in allacc if ACC_RE.match(a)]
        log.info("accessions: %s; union %d, valid format %d", {k: len(v) for k, v in sets.items()}, len(allacc), len(valid))
        uf = UniprotFetcher()
        uf.run_batches(valid)
        cols = ["batch", "primary_acc", "interpro", "pfam", "ec", "binding_chebi", "cofactor_chebi", "taxon"]
        part = pd.read_csv(uf.partial, sep="\t", names=cols, dtype=str, keep_default_na=False)
        part = part.drop(columns="batch").drop_duplicates("primary_acc")
        part["id"] = part.primary_acc
        part["feature_source"] = "uniprotkb"
        missing = sorted(set(valid) - set(part.id))
        uf.run_secondary(missing)
        if uf.sec_f.exists():
            sec = pd.read_csv(uf.sec_f, sep="\t", names=["id"] + cols[1:], dtype=str, keep_default_na=False)
            sec = sec[sec.primary_acc != ""].drop_duplicates("id")
            sec["feature_source"] = "uniprotkb_secondary"
            part = pd.concat([part, sec], ignore_index=True)
        df = pd.DataFrame({"id": allacc}).merge(part, on="id", how="left")
        df["found"] = df.primary_acc.notna() & (df.primary_acc != "")
        uf.run_uniparc(sorted(df.id[~df.found]), load_sequences())
        upf = OUT / "uniprot_uniparc.tsv"
        if upf.exists():
            up = pd.read_csv(upf, sep="\t", names=["id", "upi", "interpro", "pfam"], dtype=str,
                             keep_default_na=False).drop_duplicates("id").set_index("id")
            up = up[up.upi != ""]
            m = ~df.found & df.id.isin(up.index)
            df.loc[m, "interpro"] = df.id[m].map(up.interpro)
            df.loc[m, "pfam"] = df.id[m].map(up.pfam)
            df.loc[m, "feature_source"] = "uniparc"
            log.info("UniParc filled InterPro/Pfam for %d accessions missing from UniProtKB", int(m.sum()))
        df["feature_source"] = df.feature_source.fillna("none")
        lst = lambda s: [x for x in s.split(";") if x] if isinstance(s, str) and s else []
        for c in ("interpro", "pfam", "ec", "binding_chebi", "cofactor_chebi"):
            df[c] = df[c].map(lst)
        df["taxon"] = pd.to_numeric(df.taxon, errors="coerce").astype("Int64")
        df["primary_acc"] = df.primary_acc.fillna("")
        df = df[["id", "interpro", "pfam", "ec", "binding_chebi", "cofactor_chebi", "taxon", "found", "primary_acc",
                 "feature_source"]]
        write_parquet(df, out)
    uniprot_coverage()


def uniprot_coverage():
    df = pd.read_parquet(OUT / "uniprot_features.parquet")
    df = df.set_index("id")
    has_ipr = df.interpro.map(len) > 0
    cov = {}
    for k, ids in accession_sets().items():
        ids = pd.Index(sorted(set(ids)))
        cov[k] = {"n": len(ids), "found": round(float(df.found.reindex(ids).fillna(False).mean()), 4),
                  "interpro": round(float(has_ipr.reindex(ids).fillna(False).mean()), 4),
                  "interpro_uniprotkb_only": round(float((has_ipr & df.found).reindex(ids).fillna(False).mean()), 4),
                  "pfam": round(float((df.pfam.map(len) > 0).reindex(ids).fillna(False).mean()), 4),
                  "ec": round(float((df.ec.map(len) > 0).reindex(ids).fillna(False).mean()), 4),
                  "binding_chebi": round(float((df.binding_chebi.map(len) > 0).reindex(ids).fillna(False).mean()), 4),
                  "cofactor_chebi": round(float((df.cofactor_chebi.map(len) > 0).reindex(ids).fillna(False).mean()), 4)}
    (OUT / "uniprot_coverage.json").write_text(json.dumps(cov, indent=1))
    for k, v in cov.items():
        log.info("coverage %-16s %s", k, v)
    return cov


# ----------------------------------------------------------------------------------------------- goa

def parse_gaf(fh, taxon: int, stats: collections.Counter) -> pd.DataFrame:
    ids, gos, evs, dates, asp = [], [], [], [], []
    for line in fh:
        if line.startswith("!"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 14:
            continue
        stats["rows"] += 1
        if p[0] != "UniProtKB":
            stats["non_uniprot"] += 1
            continue
        if "NOT" in p[3].split("|"):
            stats["not_qualified"] += 1
            continue
        ids.append(p[1]); gos.append(p[4]); evs.append(p[6]); asp.append(p[8])
        dates.append(int(p[13]) if p[13].isdigit() else 0)
    df = pd.DataFrame({"id": ids, "go": gos, "evidence": evs, "date": dates, "aspect": asp})
    df["taxon"] = taxon
    return df


def step_goa():
    out = OUT / "goa_reference.parquet"
    if out.exists():
        log.info("goa_reference.parquet exists; skipping")
        return
    parts, summary = [], {}
    for taxon, (path, name) in GOA_SOURCES.items():
        url = GOA + path
        n = remote_size(url) or 0
        stats = collections.Counter()
        if n > 300 * 1024**2:
            log.info("%s is %.0f MB: stream-parsing without storing", url, n / 1e6)
            r = http_get(url, stream=True, timeout=1800)
            r.raw.decode_content = False
            src = gzip.GzipFile(fileobj=r.raw) if url.endswith(".gz") else r.raw
            df = parse_gaf(io.TextIOWrapper(src, encoding="utf-8", errors="replace"), taxon, stats)
        else:
            dest = RAW / "goa" / (Path(path).name if path.endswith(".gz") else f"{Path(path).name}.gaf.gz")
            download(url, dest, gzip_on_save=not path.endswith(".gz"))
            with open_text(dest) as fh:
                df = parse_gaf(fh, taxon, stats)
        df = df.drop_duplicates(["id", "go", "evidence", "date"])
        summary[name] = {"taxon": taxon, "url": url, **stats, "kept": len(df), "proteins": int(df.id.nunique())}
        log.info("GOA %s (%d): %s", name, taxon, summary[name])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)[["id", "go", "evidence", "date", "taxon", "aspect"]]
    df["date"] = df.date.astype("int32")
    df["taxon"] = df.taxon.astype("int32")
    for c in ("evidence", "aspect"):
        df[c] = df[c].astype("category")
    (OUT / "goa_summary.json").write_text(json.dumps(summary, indent=1))
    write_parquet(df, out)


# ----------------------------------------------------------------------------------------------- readme

def count_rows(p: Path):
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    if p.suffix == ".tsv":
        with open(p) as fh:
            return sum(1 for _ in fh) - 1
    return None


def write_readme():
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    lines = ["# data/grounding", "",
             f"Built by `PYTHONPATH=src python -m cafa6.grounding_data --step all` (README generated "
             f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}). Re-running skips existing outputs.", "",
             "**Leakage rule:** in `uniprot_features.parquet`, `binding_chebi`, `cofactor_chebi` and `ec` are curated "
             "UniProt annotations — use them ONLY for training/reference proteins. `interpro`/`pfam` are computational "
             "matches and may be used for any protein.", "",
             "## Products", "", "| file | size | rows | description |", "|---|---|---|---|"]
    desc = {
        "interpro2go.tsv": "[interpro, go] from GO external2go/interpro2go",
        "ec2go.tsv": "[ec, go] from GO external2go/ec2go",
        "rhea2go.tsv": "[rhea, go] from GO external2go/rhea2go",
        "rhea_participants.tsv": "[rhea (master id), chebi] Rhea REST export (all reaction participants)",
        "go2chebi.tsv": "[go, chebi, relation, axiom] CHEBI references in go-plus logical definitions / restrictions (all aspects)",
        "chebi.tsv": "[chebi, name, smiles, smiles_from] ChEBI ids in go2chebi or Rhea participants; SMILES from Rhea, "
                     "else ChEBI structures; class terms without structure borrow the is_a descendant (depth<=3) used in most Rhea reactions "
                     "(smiles_from=descendant:...; parent:... for secondary ids)",
        "ec.tsv": "[ec, name, transferred_to] ExPASy ENZYME",
        "go2ligand.tsv": "[go, go_name, chebi, name, smiles, source, relation, via, ubiquitous, smiles_from] MF terms only; "
                         "source go_logical_def|rhea|ec; ubiquitous = ChEBI in >=300 Rhea reactions (H2O, H+, ATP, NAD+, ...)",
        "uniprot_features.parquet": "[id, interpro, pfam, ec, binding_chebi, cofactor_chebi, taxon, found, primary_acc, "
                                    "feature_source(uniprotkb|uniprotkb_secondary|uniparc|none)] UniProt REST for CAFA "
                                    "train, GO-GPT and Kaggle-superset accessions; deleted entries get InterPro/Pfam from UniParc (sequence MD5)",
        "uniprot_coverage.json": "fraction of accessions found / with >=1 InterPro etc. per set",
        "goa_reference.parquet": "[id, go, evidence, date, taxon, aspect] GOA for 10 reference proteomes, all evidence "
                                 "codes, NOT-qualified rows dropped, DB=UniProtKB only",
        "goa_summary.json": "per-proteome GOA row counts",
    }
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.name in desc:
            rc = count_rows(f)
            lines.append(f"| `{f.name}` | {f.stat().st_size / 1e6:.2f} MB | {rc if rc is not None else ''} | {desc[f.name]} |")
    lines += ["", "## Raw sources (`raw/`)", "", "| file | url | downloaded | stored size |", "|---|---|---|---|"]
    for k in sorted(m):
        v = m[k]
        lines.append(f"| `{k}` | {v.get('url', '')} | {v.get('downloaded', '')} | {v.get('stored_bytes', 0) / 1e6:.2f} MB |")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    lines += ["", f"Total size of data/grounding: {total / 1e6:.1f} MB.", "",
              "## Notes / substitutions", "",
              "- GO CHEBI links come from `go-plus.json.gz` (plain `go.obo` carries no CHEBI logical definitions); "
              "`go.obo` is used only for names/namespaces. purl.obolibrary.org is tried first, current.geneontology.org is the mirror.",
              "- BRENDA requires an account: only a stub (env BRENDA_EMAIL/BRENDA_PASSWORD); the EC route uses Rhea "
              "(rhea2ec + rhea-ec-iubmb) and ExPASy ENZYME transferred-entry resolution instead. Partial ECs (x.x.x.-) are not expanded.",
              "- Rhea participants include currency metabolites; filter with `ubiquitous` downstream, but only for "
              "source rhea/ec (a go_logical_def row such as ATP binding -> ATP is also flagged ubiquitous and is the actual ligand).",
              "- `chebi.tsv` is restricted to go2chebi + Rhea ChEBI ids; some UniProt binding/cofactor ChEBI ids are not in it.",
              "- UniProt merged accessions are resolved via `sec_acc:` search (`primary_acc` column); deleted ones have found=False, "
              "empty curated lists, and InterPro/Pfam from UniParc (matched by sequence MD5; UniParc lists only InterPro entries "
              "attached to member-database signatures, so it can be slightly sparser than UniProtKB xrefs).",
              "- GOA: human/mouse/rat/arabidopsis/yeast(559292)/fly/worm from species GAFs; S. pombe, E. coli MG1655 (K-12) and "
              "M. tuberculosis ATCC 25618 (H37Rv) from goa/proteomes. `taxon` is the requested reference taxon id.", ""]
    if (OUT / "uniprot_coverage.json").exists():
        cov = json.loads((OUT / "uniprot_coverage.json").read_text())
        lines += ["## UniProt coverage", "",
                  "`found` = in current UniProtKB (incl. merged accessions); `>=1 InterPro` includes UniParc fill-in for "
                  "deleted entries; `InterPro (UniProtKB only)` excludes it.", "",
                  "| set | n | found | >=1 InterPro | InterPro (UniProtKB only) | >=1 Pfam | EC | binding ChEBI | cofactor ChEBI |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k, v in cov.items():
            lines.append(f"| {k} | {v['n']} | {v['found']:.3f} | {v['interpro']:.3f} | {v['interpro_uniprotkb_only']:.3f} | "
                         f"{v['pfam']:.3f} | {v['ec']:.3f} | {v['binding_chebi']:.3f} | {v['cofactor_chebi']:.3f} |")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines))
    log.info("wrote README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "static", "uniprot", "goa", "readme"])
    a = ap.parse_args()
    setup_logging(a.step)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    log.info("step=%s free=%.2f GB", a.step, free_bytes() / 1024**3)
    if a.step in ("all", "static"):
        step_static()
    if a.step in ("all", "goa"):
        step_goa()
    if a.step in ("all", "uniprot"):
        step_uniprot()
    write_readme()
    log.info("done step=%s free=%.2f GB", a.step, free_bytes() / 1024**3)


if __name__ == "__main__":
    main()
