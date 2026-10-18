"""Local web app for the HiGO-R2 GO-term predictor (cafa6.predict_r2.Predictor, loaded once at start-up, CPU only).

Pages and endpoints
  GET  /                 form: sequence or FASTA (at most 20 records, 5,000 residues each), organism (NCBI taxon id;
                         'OX=<id>' in a FASTA header overrides it per record)
  POST /predict          HTML results: per protein, one table per aspect (terms at or above MF 0.28 / BP 0.42 /
                         CC 0.45, GO ids linked to QuickGO, score bars), lower-scoring terms folded, the nearest
                         reference hits, and a TSV download link
  GET  /download/{key}   predictions.tsv of a recent request (the last 50 are kept in memory)
  POST /api/predict      JSON in: {"fasta": "...", "taxon": 9606} or {"sequences": [{"id": "...", "seq": "...",
                         "taxon": 9606}, ...]}, optional "min_score" (default 0.01); JSON out: {"results": [...],
                         "seconds": ...} with the fields of predict_r2's predictions.json
  GET  /health           {"status": "ok"}

Run
  python -m cafa6.webapp --host 127.0.0.1 --port 7860       (PYTHONPATH=src)
The default host 127.0.0.1 accepts connections from this machine only. To serve the local network, start it with
--host 0.0.0.0 (anyone who can reach the port can submit jobs; there is no authentication). To use it from another
computer without exposing it, keep 127.0.0.1 and tunnel over SSH from that computer:
  ssh -N -L 7860:127.0.0.1:7860 user@server      then open http://127.0.0.1:7860 locally.
Requests run one at a time (a lock around the predictor); one protein takes about 4-5 s on CPU, mostly DIAMOND.
"""
from __future__ import annotations

import argparse
import html
import secrets
import threading
import time
from collections import OrderedDict

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from cafa6.predict_r2 import (ASPECTS, THRESH, Predictor, low_identity_note, parse_fasta, resolve_device, to_tsv,
                              validate)

MAX_SEQS, MAX_LEN = 20, 5000
ORGANISMS = [("9606", "Human (9606)"), ("10090", "Mouse (10090)"), ("10116", "Rat (10116)"),
             ("7955", "Zebrafish (7955)"), ("7227", "Fruit fly (7227)"), ("559292", "Yeast S288c (559292)"),
             ("83333", "E. coli K-12 (83333)"), ("3702", "Arabidopsis (3702)"), ("other", "Other (enter id)")]
ASPECT_NAMES = {"MF": "Molecular function", "BP": "Biological process", "CC": "Cellular component"}
EXAMPLE = (">example OX=9606\nMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAI"
           "NNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH")

app = FastAPI(title="HiGO-R2 GO-term predictor")
STATE: dict = {"predictor": None}
LOCK = threading.Lock()
TSV_CACHE: "OrderedDict[str, str]" = OrderedDict()

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2330}
header{background:#1f3a5f;color:#fff;padding:14px 28px}header h1{margin:0;font-size:20px}
header p{margin:4px 0 0;font-size:13px;opacity:.85}main{max-width:1100px;margin:0 auto;padding:20px 28px}
.card{background:#fff;border:1px solid #dde1e7;border-radius:6px;padding:18px 20px;margin-bottom:18px}
textarea{width:100%;box-sizing:border-box;height:190px;font-family:ui-monospace,Menlo,Consolas,monospace;
font-size:12px;padding:8px;border:1px solid #c4cad3;border-radius:4px}
label{font-weight:600;font-size:14px}select,input[type=text]{padding:6px;font-size:14px;border:1px solid #c4cad3;
border-radius:4px}.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:12px}
button{background:#1f6feb;color:#fff;border:0;border-radius:4px;padding:9px 20px;font-size:15px;cursor:pointer}
button:hover{background:#185abc}.muted{color:#5b6472;font-size:13px}.err{background:#fdecea;border-color:#f5c2bd}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 12px}th,td{text-align:left;padding:4px 8px;
border-bottom:1px solid #eceff3;vertical-align:middle}th{background:#f0f2f5;font-weight:600}
td.go{white-space:nowrap;font-family:ui-monospace,Menlo,Consolas,monospace}a{color:#1f6feb;text-decoration:none}
a:hover{text-decoration:underline}.bar{background:#e6ebf2;border-radius:3px;width:160px;height:12px;display:inline-block;
vertical-align:middle;margin-right:6px}.bar span{display:block;height:12px;border-radius:3px;background:#2e7d32}
.MF .bar span{background:#1565c0}.BP .bar span{background:#2e7d32}.CC .bar span{background:#ef6c00}
h2{font-size:18px;margin:0 0 6px}h3{font-size:15px;margin:14px 0 4px}.note{background:#fff8e1;border:1px solid #f3d27a;
padding:8px 10px;border-radius:4px;font-size:13px;margin:8px 0}details summary{cursor:pointer;font-size:13px;color:#1f6feb}
.dl{display:inline-block;margin:4px 0 12px}
"""


def predictor() -> Predictor:
    if STATE["predictor"] is None:
        STATE["predictor"] = Predictor()
    return STATE["predictor"]


def page(body: str, title: str = "HiGO-R2 GO-term predictor") -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>
<header><h1>HiGO-R2 GO-term predictor</h1><p>ESM2-35M HiGO network + sharpened kNN over GO-GPT train+val references,
combined by the twin stacker (look-3 pipeline). Research model; check scores against the nearest references.</p></header>
<main>{body}</main></body></html>""")


def form_html(fasta: str = "", taxon_choice: str = "9606", taxon_other: str = "", errors: list[str] | None = None) -> str:
    opts = "".join(f'<option value="{v}"{" selected" if v == taxon_choice else ""}>{html.escape(n)}</option>'
                   for v, n in ORGANISMS)
    err = ""
    if errors:
        err = '<div class="card err"><b>Input problems</b><ul>' + "".join(
            f"<li>{html.escape(e)}</li>" for e in errors) + "</ul></div>"
    return f"""{err}<div class="card"><form method="post" action="/predict">
<label for="fasta">Protein sequence or FASTA</label>
<p class="muted">Up to {MAX_SEQS} sequences, {MAX_LEN:,} residues each; amino-acid letters only
(ACDEFGHIKLMNPQRSTVWY plus X, B, Z, U, O). A header with OX=&lt;taxon id&gt; sets that record's organism.</p>
<textarea id="fasta" name="fasta" placeholder="{html.escape(EXAMPLE)}">{html.escape(fasta)}</textarea>
<div class="row"><label for="taxon_choice">Organism</label><select id="taxon_choice" name="taxon_choice">{opts}</select>
<label for="taxon_other" class="muted">other NCBI taxon id</label>
<input type="text" id="taxon_other" name="taxon_other" size="10" value="{html.escape(taxon_other)}" placeholder="e.g. 9913">
<button type="submit">Predict GO terms</button></div>
<p class="muted">One protein takes about 5 s on CPU, mostly the DIAMOND search over 133,054 references.
Without a reference above 30% identity accuracy is much lower (IA-weighted Fmax 0.45 below 30%, 0.34 with no hit,
against 0.72 overall on the GO-GPT temporal test).</p></form></div>"""


def resolve_taxon(choice: str | None, other: str | None) -> tuple[int | None, list[str]]:
    if choice and choice != "other":
        return int(choice), []
    other = (other or "").strip()
    if not other:
        return None, []
    if not other.isdigit():
        return None, [f"taxon id must be a number, got {other!r}"]
    return int(other), []


def run_prediction(recs: list[dict]) -> tuple[list[dict], float]:
    t = time.time()
    with LOCK:
        res = predictor().predict(recs)
    return res, time.time() - t


def score_rows(terms: list[dict]) -> str:
    rows = []
    for s in terms:
        w = max(1.0, min(100.0, s["score"] * 100))
        rows.append(f'<tr><td class="go"><a href="https://www.ebi.ac.uk/QuickGO/term/{s["term"]}" target="_blank" '
                    f'rel="noopener">{s["term"]}</a></td><td>{html.escape(s["name"])}</td>'
                    f'<td><span class="bar"><span style="width:{w:.0f}%"></span></span>{s["score"]:.3f}</td></tr>')
    return "".join(rows)


def results_html(results: list[dict], seconds: float, key: str) -> str:
    out = [f'<div class="card"><b>{len(results)} protein(s)</b> predicted in {seconds:.1f} s. '
           f'<a class="dl" href="/download/{key}">Download all scores (TSV)</a> &nbsp; <a href="/">New query</a></div>']
    for r in results:
        org = html.escape(r["organism"] or "organism not in the GO-GPT data")
        tx = r["taxon"] if r["taxon"] is not None else "not given"
        parts = [f'<div class="card"><h2>{html.escape(r["id"])}</h2><p class="muted">{r["length"]} residues; taxon '
                 f'{tx} ({org}); best reference identity {r["max_identity"]:.1f}% (bin {r["identity_bin"]}); '
                 f'{r["n_hits"]} hits used</p>']
        note = low_identity_note(r)
        if note:
            parts.append(f'<div class="note">{html.escape(note)}</div>')
        for asp in ASPECTS:
            above = r["above_threshold"][asp]
            rest = [s for s in r["scores"] if s["aspect"] == asp and s["score"] < THRESH[asp]][:30]
            parts.append(f'<h3>{ASPECT_NAMES[asp]} ({asp}): {len(above)} terms at or above {THRESH[asp]}</h3>')
            if above:
                parts.append(f'<table class="{asp}"><tr><th>GO term</th><th>name</th><th>score</th></tr>'
                             f'{score_rows(above)}</table>')
            else:
                parts.append('<p class="muted">none above the threshold</p>')
            if rest:
                parts.append(f'<details><summary>next {len(rest)} terms below the threshold</summary>'
                             f'<table class="{asp}"><tr><th>GO term</th><th>name</th><th>score</th></tr>'
                             f'{score_rows(rest)}</table></details>')
        parts.append("<h3>Nearest references (DIAMOND, GO-GPT train + val)</h3>")
        if r["nearest"]:
            parts.append("<table><tr><th>reference</th><th>organism</th><th>set</th><th>identity %</th>"
                         "<th>query coverage %</th><th>bitscore</th><th>e-value</th></tr>")
            for h in r["nearest"]:
                parts.append(f'<tr><td class="go"><a href="https://www.uniprot.org/uniprotkb/{html.escape(h["id"])}" '
                             f'target="_blank" rel="noopener">{html.escape(h["id"])}</a></td>'
                             f'<td>{html.escape(str(h["organism"]))}</td><td>{h["split"]}</td><td>{h["pident"]:.1f}</td>'
                             f'<td>{h["qcov"]:.0f}</td><td>{h["bits"]:.0f}</td><td>{h["evalue"]:.1e}</td></tr>')
            parts.append("</table>")
        else:
            parts.append('<p class="muted">no hit at e &le; 1e-3</p>')
        parts.append("</div>")
        out.append("".join(parts))
    return "".join(out)


def remember_tsv(results: list[dict]) -> str:
    key = secrets.token_urlsafe(9)
    TSV_CACHE[key] = to_tsv(results)
    while len(TSV_CACHE) > 50:
        TSV_CACHE.popitem(last=False)
    return key


@app.get("/", response_class=HTMLResponse)
def index():
    return page(form_html())


@app.get("/health")
def health():
    return {"status": "ok", "loaded": STATE["predictor"] is not None}


@app.post("/predict", response_class=HTMLResponse)
def predict_form(fasta: str = Form(""), taxon_choice: str = Form("9606"), taxon_other: str = Form("")):
    taxon, errs = resolve_taxon(taxon_choice, taxon_other)
    recs = parse_fasta(fasta, taxon)
    errs += validate(recs, MAX_SEQS, MAX_LEN)
    if errs:
        return page(form_html(fasta, taxon_choice, taxon_other, errs))
    results, dt = run_prediction(recs)
    return page(results_html(results, dt, remember_tsv(results)), "HiGO-R2 results")


@app.get("/download/{key}")
def download(key: str):
    if key not in TSV_CACHE:
        raise HTTPException(404, "unknown or expired result; run the prediction again")
    return PlainTextResponse(TSV_CACHE[key], media_type="text/tab-separated-values",
                             headers={"Content-Disposition": 'attachment; filename="higo_r2_predictions.tsv"'})


class SeqIn(BaseModel):
    id: str
    seq: str
    taxon: int | None = None


class PredictIn(BaseModel):
    fasta: str | None = None
    sequences: list[SeqIn] | None = None
    taxon: int | None = None
    min_score: float = 0.01


@app.post("/api/predict")
def api_predict(req: PredictIn):
    if req.sequences:
        recs = [{"id": s.id, "seq": "".join(s.seq.split()).upper().rstrip("*"),
                 "taxon": s.taxon if s.taxon is not None else req.taxon} for s in req.sequences]
    else:
        recs = parse_fasta(req.fasta or "", req.taxon)
    errs = validate(recs, MAX_SEQS, MAX_LEN)
    if errs:
        return JSONResponse({"errors": errs}, status_code=422)
    results, dt = run_prediction(recs)
    for r in results:
        r["scores"] = [s for s in r["scores"] if s["score"] >= req.min_score]
    return {"results": results, "seconds": round(dt, 2)}


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this machine only; 0.0.0.0 = local network")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default min(16, cores))")
    ap.add_argument("--diamond-threads", type=int, default=None, help="default min(32, cores)")
    ap.add_argument("--device", default="auto", help="HiGO device: auto (GPU bf16 if present) | cpu | cuda[:N]")
    a = ap.parse_args()
    STATE["predictor"] = Predictor(threads=a.threads, diamond_threads=a.diamond_threads,
                                   higo_device=resolve_device(a.device))
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
