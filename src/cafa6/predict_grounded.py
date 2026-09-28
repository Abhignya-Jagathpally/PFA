"""Grounded, hierarchy-consistent GO output from HiGO-v2 (see cafa6.grounded).

Per protein: the predicted GO set (every term with score >= the aspect's F-max threshold; MCM makes the set
closed under is_a/part_of within the vocabulary), each term with its parent path from the ontology, score,
confidence band and evidence: InterPro domains, cross-species reference orthologs (with identity), ligands
(ChEBI name + SMILES) and, for single sequences, HiGO's top evidence residues.

Modes:
  python -m cafa6.predict_grounded --split test                 # clustered test split -> JSONL + TSV
  python -m cafa6.predict_grounded --fasta demo.fa [--taxon N]  # new sequences -> JSON + markdown report
  python -m cafa6.predict_grounded --submission                 # Kaggle testsuperset -> submission.tsv.gz
New sequences are searched with DIAMOND against all labelled proteins and scanned with InterProScan through
the EBI REST service (needs network; EBI_EMAIL can be set, otherwise the git user email is sent).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import obonet
import pandas as pd
import requests
import scipy.sparse as sp
import torch

from cafa6.comparative import REFERENCE_TAXA
from cafa6.data import PROC, ROOT
from cafa6.evaluate import OBO
from cafa6.grounded import COMPONENTS, Closure, Grounding, calibrate
from cafa6.homology import diamond_score, diamond_search, max_identity
from cafa6.stack import features, mcm_numpy

RUN_DIR = ROOT / "runs" / "grounded"
BANDS = ((0.7, "high"), (0.4, "medium"), (0.0, "low"))
ASPECT_NAME = {"F": "Molecular Function", "P": "Biological Process", "C": "Cellular Component"}
ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}


class Explainer:
    def __init__(self, vocab, anc, goa_exclude=(), goa_date_max=None):
        self.vocab, self.anc = vocab, anc
        self.terms, self.aspect = vocab.term.values, vocab.aspect.values
        self.g = obonet.read_obo(str(OBO))
        self.closure = Closure(OBO, self.terms)
        self.gr = Grounding(vocab, self.closure, goa_date_max=goa_date_max, exclude_ids=goa_exclude)
        cal = json.loads((RUN_DIR / "calibrator.json").read_text())["grounded"]
        self.params = {a: {"w": np.array([v["w"][k] for k in COMPONENTS], np.float32), "b": v["b"]}
                       for a, v in cal.items()}
        m = pd.read_csv(RUN_DIR / "metrics.tsv", sep="\t")
        m = m[m.method == "grounded"].set_index("aspect").tau_w
        self.tau = np.array([m[{"F": "MF", "P": "BP", "C": "CC"}[a]] for a in self.aspect], np.float32)
        self.children = {}
        for c, a in anc:
            self.children.setdefault(a, []).append(c)
        self.lig = self.gr.lig.drop_duplicates("chebi").set_index("chebi")
        self.ip_idx = {ip: i for i, ip in enumerate(self.gr.ips)}
        self.species = {t: n for t, n in REFERENCE_TAXA.items()}

    # ------------------------------------------------------------------ scoring
    def score(self, pm, pk, X, ev, lfreq):
        p, _ = calibrate(None, None, None, None, None, pm, pk, X, ev, self.vocab, lfreq, self.gr.has_lig,
                         params=self.params)
        return mcm_numpy(p, self.anc)

    def evidence(self, ids, taxa, hits, db_df, Y_db, interpro_lists=None):
        q_tax = pd.Series(taxa, index=ids)
        D = self.gr.domains(ids, interpro_lists)
        O, G, h = self.gr.orthologs(hits, ids, q_tax, db_df, Y_db)
        L_dom, L_hit, PL_dom, PL_hit = self.gr.ligands(D, h, ids, return_protein_ligands=True)
        return {"D": D, "O": O, "G": G, "L_dom": L_dom, "L_hit": L_hit}, h, (PL_dom + PL_hit).tocsr()

    # ------------------------------------------------------------------ description
    def path(self, t):
        out, cur = [], t
        while cur not in ROOTS:
            ps = sorted(v for _, v, k in self.g.out_edges(cur, keys=True) if k == "is_a")
            if not ps:
                break
            cur = ps[0]
            out.append(f"{cur} {self.g.nodes[cur].get('name', '')}")
        return out

    def describe(self, pid, p_row, pm_row, dom_ips, hits_q, PL_row, db_index, Y_db, residues=None, seq=None):
        chosen = np.where(p_row >= self.tau)[0]
        chosen_set = set(chosen.tolist())
        out = []
        pl = set(np.asarray(PL_row.indices).tolist()) if PL_row is not None else set()
        for t in chosen:
            term = self.terms[t]
            specific = not any(c in chosen_set for c in self.children.get(t, []))
            d = {"go": term, "name": self.g.nodes[term].get("name", "") if term in self.g else "",
                 "aspect": self.aspect[t], "score": round(float(p_row[t]), 3), "higo_score": round(float(pm_row[t]), 3),
                 "band": next(b for thr, b in BANDS if p_row[t] >= thr), "specific": bool(specific)}
            if specific:
                d["parents"] = self.path(term)
                d["domains"] = [ip for ip in dom_ips if ip in self.ip_idx and self.gr.IP[self.ip_idx[ip], t] > 0]
                orth = []
                for s, st, pident in hits_q:
                    row = db_index.get(s)
                    lab = row is not None and Y_db[row, t] > 0
                    goa = s in self.gr.goa_row and self.gr.GOA[self.gr.goa_row[s], t] > 0
                    if lab or goa:
                        orth.append({"protein": s, "species": self.species.get(int(st), str(st)), "identity": float(pident),
                                     "source": "train label" if lab else "GOA"})
                d["orthologs"] = orth[:3]
                ligs = [self.gr.chebis[c] for c in self.gr.LigT[t].indices if c in pl]
                d["ligands"] = [{"chebi": c, "name": str(self.lig.at[c, "name"]) if c in self.lig.index else "",
                                 "smiles": str(self.lig.at[c, "smiles"]) if c in self.lig.index else ""} for c in ligs[:3]]
                if residues is not None and seq is not None:
                    top = np.argsort(-residues[t])[:5]
                    d["top_residues"] = [f"{seq[i]}{i + 1}" for i in sorted(top) if residues[t][i] > 0]
            out.append(d)
        return sorted(out, key=lambda d: (d["aspect"], -d["score"]))


def report_md(pid, seq, terms, domains_spans):
    L = [f"# Grounded GO prediction for {pid} ({len(seq)} aa)", "", f"Sequence: `{seq}`", "",
         "Scores come from HiGO-v2 (fine-tuned ESM2-35M with hierarchy-aware GO queries, recalibrated with domain, "
         "ortholog and ligand evidence). The set is closed under the ontology: every listed term's ancestors are "
         "also predicted. Only the most specific terms are expanded below.", ""]
    if domains_spans:
        L += ["Domains found by InterProScan:", ""] + [f"- {ip} {name} ({s}-{e})" for ip, name, s, e in domains_spans] + [""]
    for a in ("F", "P", "C"):
        sel = [t for t in terms if t["aspect"] == a]
        L += [f"## {ASPECT_NAME[a]}", ""]
        if not sel:
            L += ["No term above the threshold.", ""]
            continue
        n_all = len(sel)
        for t in [t for t in sel if t["specific"]]:
            L += [f"### {t['go']} {t['name']}", "",
                  f"- Score {t['score']:.2f} ({t['band']} confidence); HiGO before grounding {t['higo_score']:.2f}",
                  f"- Parent path: {' -> '.join(t['parents']) if t['parents'] else 'root'}"]
            if t.get("domains"):
                L.append(f"- Domain support: {', '.join(t['domains'])}")
            if t.get("orthologs"):
                L.append("- Orthologs: " + "; ".join(f"{o['protein']} ({o['species']}, {o['identity']:.0f}% id, {o['source']})"
                                                     for o in t["orthologs"]))
            if t.get("ligands"):
                L.append("- Ligands: " + "; ".join(f"{g['name']} ({g['chebi']}) `{g['smiles']}`" for g in t["ligands"]))
            if t.get("top_residues"):
                L.append(f"- HiGO evidence residues: {', '.join(t['top_residues'])}")
            if not any(t.get(k) for k in ("domains", "orthologs", "ligands")):
                L.append("- No external evidence; the score rests on the sequence model alone.")
            L.append("")
        L += [f"({n_all} {ASPECT_NAME[a]} terms in the full set, including ancestors.)", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ InterPro matches
def uniparc_interpro(seq):
    """Precomputed InterPro matches for a sequence already in UniParc (exact match by MD5)."""
    import hashlib
    md5 = hashlib.md5(seq.upper().encode()).hexdigest().upper()
    r = requests.get("https://rest.uniprot.org/uniparc/search", timeout=120,
                     params={"query": f"checksum:{md5}", "format": "json", "size": 1})
    ips, spans = set(), []
    for e in r.json().get("results", []):
        for f in e.get("sequenceFeatures", []):
            g = f.get("interproGroup") or {}
            if g.get("id"):
                ips.add(g["id"])
                for loc in f.get("locations", []):
                    spans.append((g["id"], g.get("name", ""), loc.get("start"), loc.get("end")))
    return sorted(ips), sorted(set(spans), key=lambda x: (x[2] or 0))


def interproscan(seq, email):
    base = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
    job = requests.post(f"{base}/run", data={"email": email, "sequence": seq, "goterms": "false",
                                             "pathways": "false"}, timeout=60).text.strip()
    for _ in range(60):
        st = requests.get(f"{base}/status/{job}", timeout=60).text.strip()
        if st == "FINISHED":
            break
        if st in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError(f"InterProScan job {job}: {st}")
        time.sleep(10)
    res = requests.get(f"{base}/result/{job}/json", timeout=120).json()
    ips, spans = [], []
    for m in res["results"][0]["matches"]:
        e = m["signature"].get("entry")
        if e and e.get("accession"):
            ips.append(e["accession"])
            for loc in m.get("locations", []):
                spans.append((e["accession"], e.get("name") or e.get("description") or "", loc["start"], loc["end"]))
    return sorted(set(ips)), sorted(set(spans), key=lambda x: x[2])


def read_fasta(path):
    recs, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name:
        recs.append((name, "".join(buf)))
    return recs


# ------------------------------------------------------------------ modes
def load_common():
    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    anc = np.load(PROC / "go_ancestors.npy")
    tr = np.where(df.split == "train")[0]
    lfreq = np.log(np.asarray(Y[tr].mean(0)).ravel() + 1e-5).astype(np.float32)
    return df, vocab, Y, anc, tr, lfreq


def mode_split(args):
    df, vocab, Y, anc, tr, lfreq = load_common()
    te = np.where(df.split == "test")[0]
    ids = df.id.values[te]
    ex = Explainer(vocab, anc, goa_exclude=set(df.id.values[df.split != "train"]))
    hits = pd.read_parquet(PROC / "diamond_hits_valtest.parquet")
    ev, h, PL = ex.evidence(ids, df.taxon.values[te], hits, df.iloc[tr], Y[tr])
    p = np.load(RUN_DIR / "pred_test.npy").astype(np.float32)
    pm = np.load(ROOT / "runs" / "dev_35m_lora" / "pred_test.npy").astype(np.float32)
    ips = ex.gr._lists(ids, "interpro")
    db_index = {s: i for i, s in enumerate(df.id.values[tr])}
    hq = {q: list(zip(g.s, g.st, g.pident)) for q, g in h.sort_values("pident", ascending=False).groupby("q")}
    out = RUN_DIR / "output"
    out.mkdir(exist_ok=True)
    rows = []
    with open(out / "test_grounded.jsonl", "w") as f:
        for i, pid in enumerate(ids):
            terms = ex.describe(pid, p[i], pm[i], ips[i], hq.get(pid, []), PL[i], db_index, Y[tr])
            f.write(json.dumps({"id": pid, "terms": terms}) + "\n")
            rows += [(pid, t["go"], t["aspect"], t["score"], t["band"], t["specific"], len(t.get("domains", [])),
                      len(t.get("orthologs", [])), ";".join(g["chebi"] for g in t.get("ligands", []))) for t in terms]
    tsv = pd.DataFrame(rows, columns=["id", "go", "aspect", "score", "band", "specific", "n_domains", "n_orthologs", "ligands"])
    tsv.to_csv(out / "test_grounded.tsv", sep="\t", index=False)
    in_set = p >= ex.tau[None, :]
    viol = int((in_set[:, anc[:, 0]] & ~in_set[:, anc[:, 1]]).sum())   # child predicted, ancestor not
    info = {"proteins": len(ids), "terms": len(tsv), "terms_per_protein": len(tsv) / len(ids),
            "specific_with_domain": float((tsv[tsv.specific].n_domains > 0).mean()),
            "specific_with_ortholog": float((tsv[tsv.specific].n_orthologs > 0).mean()),
            "specific_with_ligand": float((tsv[tsv.specific].ligands != "").mean()),
            "set_closure_violations": viol}
    (out / "test_grounded_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def mode_fasta(args):
    from cafa6.explain import load_run
    df, vocab, Y, anc, tr, lfreq = load_common()
    recs = read_fasta(args.fasta)
    ids = [r[0] for r in recs]
    q = pd.DataFrame({"id": ids, "seq": [r[1] for r in recs]})
    ex = Explainer(vocab, anc)
    hits = diamond_search(df, q, max_hits=50, exclude_self=False)
    pk = diamond_score(hits, ids, df.id.values, Y).astype(np.float32)
    mi = max_identity(hits, ids)
    X = features(pd.DataFrame({"id": ids, "max_pident": mi.values}), ids)
    email = os.environ.get("EBI_EMAIL") or subprocess.run(["git", "config", "user.email"], capture_output=True,
                                                          text=True, cwd=ROOT).stdout.strip()
    ipr = []
    for pid, seq in recs:
        try:
            found = uniparc_interpro(seq)
            ipr.append(found if found[0] else interproscan(seq, email))
        except Exception as e:                                 # network or service failure: no domain evidence
            print(f"InterPro lookup failed for {pid}: {e}")
            ipr.append(([], []))
    ev, h, PL = ex.evidence(ids, [args.taxon] * len(ids), hits, df, Y, interpro_lists=[x[0] for x in ipr])
    model, cfg, _, tmap, tok = load_run("dev_35m_lora")
    pm, attn = [], []
    for pid, seq in recs:
        s = seq[:cfg["max_len"]]
        enc = tok([s], return_tensors="pt")
        ids_t, att = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
        res = att.clone(); res[:, 0] = 0; res[:, -1] = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z, a = model(ids_t, att, res, torch.tensor([tmap.get(args.taxon, 0)], device="cuda"), return_attn=True)
        pm.append(model.predict_proba(z.float()).cpu().numpy()[0])
        attn.append(a.float().cpu().numpy()[0][:, 1:len(s) + 1])
    pm = np.stack(pm).astype(np.float32)
    p = ex.score(pm, pk, X, ev, lfreq)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db_index = {s: i for i, s in enumerate(df.id.values)}
    tax = pd.Series(df.taxon.values, index=df.id.values)
    tax = tax[~tax.index.duplicated()]
    for i, (pid, seq) in enumerate(recs):
        hq = h[h.q == pid].sort_values("pident", ascending=False)
        terms = ex.describe(pid, p[i], pm[i], ipr[i][0], list(zip(hq.s, hq.s.map(tax), hq.pident)), PL[i],
                            db_index, Y, residues=attn[i], seq=seq)
        (out / f"{pid}.json").write_text(json.dumps({"id": pid, "length": len(seq), "domains": ipr[i][1],
                                                      "terms": terms}, indent=2))
        (out / f"{pid}.md").write_text(report_md(pid, seq, terms, ipr[i][1]))
        print(f"wrote {out / (pid + '.md')}: {len(terms)} terms, {sum(t['specific'] for t in terms)} specific")


def mode_submission(args):
    from cafa6.explain import load_run
    from cafa6.train import Batcher, predict
    df, vocab, Y, anc, tr, lfreq = load_common()
    test = pd.read_parquet(PROC / "test.parquet")
    ex = Explainer(vocab, anc, goa_exclude=set(test.id.values))
    terms = vocab.term.values
    hits = pd.read_parquet(PROC / "diamond_hits_kaggle_test.parquet")
    mi = max_identity(hits, test.id.values)
    ident = pd.DataFrame({"id": test.id.values, "max_pident": mi.values})
    model, cfg, _, tmap, tok = load_run("dev_35m_lora")
    batcher = Batcher(test.seq.tolist(), tok, cfg["max_len"], cfg["tokens_per_batch"] * 2, cfg["max_batch"] * 2)
    taxa = test.taxon.map(lambda t: tmap.get(int(t), 0)).values
    out = RUN_DIR / "submission"
    out.mkdir(exist_ok=True)
    n_rows = 0
    with gzip.open(out / "submission.tsv.gz", "wt") as f:
        for s in range(0, len(test), args.chunk):
            idx = np.arange(s, min(s + args.chunk, len(test)))
            ids = test.id.values[idx]
            pm = predict(model, batcher, batcher.seqs, taxa, idx, "cuda").astype(np.float32)
            pk = diamond_score(hits, ids, df.id.values, Y).astype(np.float32)
            ev, _, _ = ex.evidence(ids, test.taxon.values[idx], hits, df, Y)
            p = ex.score(pm, pk, features(ident, ids), ev, lfreq)
            k = min(args.top_k, p.shape[1])
            top = np.argpartition(-p, k - 1, axis=1)[:, :k]
            for r in range(len(idx)):
                cols = top[r][p[r, top[r]] >= args.min_score]
                cols = cols[np.argsort(-p[r, cols])]
                f.writelines(f"{ids[r]}\t{terms[c]}\t{p[r, c]:.3f}\n" for c in cols)
                n_rows += len(cols)
            print(f"{idx[-1] + 1}/{len(test)} proteins, {n_rows} rows", flush=True)
    info = {"model": "HiGO-v2 grounded (dev_35m_lora + calibrator)", "n_proteins": int(len(test)), "n_rows": int(n_rows),
            "top_k": args.top_k, "min_score": args.min_score}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test"])
    ap.add_argument("--fasta")
    ap.add_argument("--submission", action="store_true")
    ap.add_argument("--taxon", type=int, default=0)
    ap.add_argument("--out", default=str(RUN_DIR / "output"))
    ap.add_argument("--top_k", type=int, default=500)
    ap.add_argument("--min_score", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()
    if args.split:
        mode_split(args)
    elif args.fasta:
        mode_fasta(args)
    elif args.submission:
        mode_submission(args)
    else:
        ap.error("one of --split, --fasta, --submission is required")
    (ROOT / "logs" / "predict_grounded.done").touch()


if __name__ == "__main__":
    main()
