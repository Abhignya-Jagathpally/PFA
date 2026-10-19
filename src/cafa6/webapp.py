"""Ontologue: local web app for the HiGO-R2 GO-term predictor (cafa6.predict_r2.Predictor, loaded once at start-up).

Pages and endpoints
  GET  /                 landing page (what the model does, pipeline diagram, measured results and limits) with the
                         prediction form: sequence or FASTA (at most 20 records, 5,000 residues each), organism (NCBI taxon id;
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

BRAND = "Ontologue"
app = FastAPI(title=f"{BRAND} GO-term predictor")
STATE: dict = {"predictor": None}
LOCK = threading.Lock()
TSV_CACHE: "OrderedDict[str, str]" = OrderedDict()

CSS = """
:root{--ink:#1f2a2e;--muted:#5f6b70;--bg:#fbfaf7;--teal:#2a7f72;--teal-d:#1f6156;--mint:#e4f2ee;--lav:#ecebf8;
--peach:#fcefe5;--line:#e7e4dd;--mf:#5b7fd6;--bp:#3aa68a;--cc:#e08a4c;
--serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;margin:0;background:var(--bg);
color:var(--ink);-webkit-font-smoothing:antialiased;line-height:1.55}
a{color:inherit}
nav{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center;padding:16px 6vw;
background:rgba(251,250,247,.86);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
nav .logo{font-family:var(--serif);font-weight:600;font-size:22px;text-decoration:none;color:var(--ink)}
nav .logo b{color:var(--teal)}nav .links a{margin-left:26px;font-size:14px;text-decoration:none;color:var(--muted)}
nav .links a:hover{color:var(--teal)}
.hero{padding:clamp(60px,10vh,120px) 6vw clamp(50px,8vh,90px);position:relative;overflow:hidden;
background:radial-gradient(1100px 520px at 85% 10%,#e2f1ec 0,rgba(226,241,236,0) 60%),
radial-gradient(800px 480px at 5% 100%,#efedf9 0,rgba(239,237,249,0) 60%),var(--bg)}
.hero .kicker{display:inline-block;background:var(--mint);color:var(--teal-d);font-weight:600;font-size:13px;
padding:6px 14px;border-radius:999px}
.hero h1{font-family:var(--serif);font-size:clamp(40px,5.8vw,80px);line-height:1.04;font-weight:600;
letter-spacing:-.015em;margin:22px 0 22px;max-width:14ch}.hero h1 em{color:var(--teal);font-style:italic}
.hero p.lede{font-size:clamp(17px,1.45vw,20px);color:var(--muted);max-width:46ch;margin:0 0 32px}
.cta{display:inline-block;background:var(--teal);color:#fff;font-weight:600;padding:14px 28px;border-radius:999px;
text-decoration:none;font-size:15px;border:0;cursor:pointer;box-shadow:0 10px 24px -12px rgba(42,127,114,.6);
transition:transform .15s,background .15s}.cta:hover{transform:translateY(-1px);background:var(--teal-d)}
.cta.ghost{background:#fff;color:var(--teal-d);border:1px solid #cfe3dd;box-shadow:none;margin-left:10px}
.hero .helix{position:absolute;right:2vw;top:clamp(30px,6vh,70px);width:40vw;max-width:520px;pointer-events:none}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;padding:0 6vw;margin-top:-10px;position:relative}
.stats div{background:#fff;border:1px solid var(--line);border-radius:20px;padding:24px 24px 26px;
box-shadow:0 18px 40px -30px rgba(31,42,46,.35)}
.stats b{display:block;font-family:var(--serif);font-size:clamp(30px,3.4vw,46px);font-weight:600;color:var(--teal)}
.stats span{font-size:14px;color:var(--muted);display:block;margin-top:4px}
section.block{padding:clamp(70px,10vh,120px) 6vw}
section.block h2{font-family:var(--serif);font-size:clamp(30px,3.8vw,50px);font-weight:600;line-height:1.1;
letter-spacing:-.01em;margin:8px 0 14px;max-width:22ch}
section.block .sub{font-size:18px;max-width:62ch;color:var(--muted);margin:0 0 40px}
.eyebrow{font-weight:600;font-size:13px;color:var(--teal);letter-spacing:.04em}
.diagram{background:#fff;border:1px solid var(--line);border-radius:26px;padding:30px;
box-shadow:0 30px 60px -40px rgba(31,42,46,.35)}.diagram svg{width:100%;height:auto;display:block}
.flow{stroke-dasharray:5 7;animation:flow 1.6s linear infinite}@keyframes flow{to{stroke-dashoffset:-24}}
.pulse{animation:pulse 3s ease-in-out infinite}@keyframes pulse{50%{opacity:.4}}
@media (prefers-reduced-motion:reduce){.flow,.pulse{animation:none}}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}
.cards article{background:#fff;border:1px solid var(--line);border-radius:22px;padding:30px 28px 32px}
.cards article .n{width:44px;height:44px;border-radius:14px;display:flex;align-items:center;justify-content:center;
font-weight:700;color:var(--teal-d);background:var(--mint)}
.cards article:nth-child(2) .n{background:var(--lav);color:#4b47a3}
.cards article:nth-child(3) .n{background:var(--peach);color:#a2582a}
.cards article h3{font-family:var(--serif);font-size:23px;line-height:1.2;font-weight:600;margin:18px 0 10px}
.cards article p{color:var(--muted);margin:0;font-size:15px}
.soft{background:linear-gradient(180deg,#f4f8f6 0,#fbfaf7 100%)}
.versus{display:grid;grid-template-columns:1.15fr 1fr;gap:34px;align-items:start}
.vtable{width:100%;border-collapse:separate;border-spacing:0;font-size:15px;background:#fff;border:1px solid var(--line);
border-radius:18px;overflow:hidden}.vtable th,.vtable td{padding:13px 16px;border-bottom:1px solid var(--line);text-align:left}
.vtable tr:last-child td{border-bottom:0}.vtable th{font-size:12px;color:var(--muted);font-weight:600;background:#faf9f6}
.vtable td.us{font-weight:700;color:var(--teal-d)}.vtable tr.hl td{background:#f0f8f5}
.limits{background:#fff;border:1px solid var(--line);border-radius:22px;padding:28px 28px}
.limits h3{margin:0 0 10px;font-family:var(--serif);font-weight:600;font-size:22px}
.limits p{color:var(--muted);margin:0 0 10px}.limits b{color:var(--teal-d)}
.run{background:linear-gradient(180deg,#eef6f3 0,#f6f4fb 100%)}
.panel{background:#fff;border:1px solid var(--line);border-radius:24px;padding:28px;
box-shadow:0 30px 60px -40px rgba(31,42,46,.35)}
.panel label{font-weight:600;font-size:14px;color:var(--ink)}
textarea{width:100%;height:200px;margin-top:10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;
padding:14px 16px;border:1px solid #dcd8cf;border-radius:16px;background:#fafaf8;color:var(--ink);resize:vertical}
textarea:focus,select:focus,input:focus{outline:2px solid #9fd0c4;outline-offset:1px}
select,input[type=text]{padding:11px 14px;font-size:14px;border:1px solid #dcd8cf;border-radius:999px;background:#fff;
color:var(--ink)}.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:16px}
.hint{color:var(--muted);font-size:13px}.linkbtn{background:none;border:0;color:var(--teal);cursor:pointer;font-size:13px;
text-decoration:underline;padding:0}
footer{padding:40px 6vw;font-size:13px;color:var(--muted);border-top:1px solid var(--line);background:#f6f5f1}
footer b{color:var(--ink)}
main.res{max-width:1120px;margin:0 auto;padding:40px 28px 80px}
.card{background:#fff;border:1px solid var(--line);border-radius:20px;padding:24px 26px;margin-bottom:20px}
.card h2{font-family:var(--serif);font-size:26px;font-weight:600;margin:0 0 4px}
.card h3{font-size:15px;font-weight:600;margin:22px 0 6px;color:var(--teal-d)}
.muted{color:var(--muted);font-size:13px}.err{background:#fff3f1;border-color:#f3c9c1;color:#7a2e22}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 12px}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid #efece6;vertical-align:middle}th{font-size:12px;color:var(--muted);font-weight:600}
td.go{white-space:nowrap;font-family:ui-monospace,Menlo,Consolas,monospace}
.card a{color:var(--teal-d);text-decoration:none}.card a:hover{text-decoration:underline}
.bar{background:#f0eee9;border-radius:99px;width:170px;height:9px;display:inline-block;vertical-align:middle;
margin-right:8px}.bar span{display:block;height:9px;border-radius:99px;background:var(--bp)}
.MF .bar span{background:var(--mf)}.BP .bar span{background:var(--bp)}.CC .bar span{background:var(--cc)}
.note{background:#fff8e8;border:1px solid #f1dfb2;padding:10px 12px;border-radius:12px;font-size:13px;margin:10px 0}
details summary{cursor:pointer;font-size:13px;color:var(--teal-d)}.dl{font-weight:600}
@media (max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}.cards,.versus{grid-template-columns:1fr}
.hero .helix{opacity:.25;width:90vw}.cta.ghost{margin:10px 0 0}}
"""


def predictor() -> Predictor:
    if STATE["predictor"] is None:
        STATE["predictor"] = Predictor()
    return STATE["predictor"]


NAV = f"""<nav><a class="logo" href="/">{BRAND}<b>.</b></a><div class="links"><a href="/#how">How it works</a>
<a href="/#results">Results</a><a href="/#limits">Limits</a><a href="/#run">Run it</a></div></nav>"""

FOOTER = f"""<footer><b>{BRAND}</b> is the serving name of the HiGO-R2 research pipeline: frozen ESM2-35M with
per-term attention (HiGO head), identity-sharpened kNN label transfer over 133,054 GO-GPT train + val references,
a per-term refiner fitted on a temporal validation set, and GO true-path closure. Benchmark: GO-GPT temporal test,
8,630 proteins, IA-weighted Fmax, CAFA evaluator (prop=max). Research software; check predictions against the nearest
references shown with every result. Built on ESM2 (MIT), UniProt and Gene Ontology data (CC BY 4.0), DIAMOND (GPL-3.0).
</footer>"""


def page(body: str, title: str = f"{BRAND} | protein function, read from its relatives") -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>
{NAV}<main class="res">{body}</main>{FOOTER}</body></html>""")


def form_html(fasta: str = "", taxon_choice: str = "9606", taxon_other: str = "", errors: list[str] | None = None) -> str:
    opts = "".join(f'<option value="{v}"{" selected" if v == taxon_choice else ""}>{html.escape(n)}</option>'
                   for v, n in ORGANISMS)
    err = ""
    if errors:
        err = '<div class="card err"><b>Fix these and run again</b><ul>' + "".join(
            f"<li>{html.escape(e)}</li>" for e in errors) + "</ul></div>"
    example_js = html.escape(EXAMPLE.replace("\n", "\\n"), quote=True)
    return f"""{err}<div class="panel"><form method="post" action="/predict">
<label for="fasta">Protein sequence or FASTA</label>
<textarea id="fasta" name="fasta" spellcheck="false" placeholder="{html.escape(EXAMPLE)}">{html.escape(fasta)}</textarea>
<p class="hint">Up to {MAX_SEQS} sequences, {MAX_LEN:,} residues each. Amino-acid letters only. A header with
OX=&lt;taxon id&gt; sets that record's organism. <button type="button" class="linkbtn"
onclick="document.getElementById('fasta').value='{example_js}'">Load an example (human HRAS G-domain)</button></p>
<div class="row"><label for="taxon_choice">Organism</label><select id="taxon_choice" name="taxon_choice">{opts}</select>
<input type="text" id="taxon_other" name="taxon_other" size="12" value="{html.escape(taxon_other)}"
placeholder="other taxon id" aria-label="other NCBI taxon id">
<button class="cta" type="submit">Predict functions</button></div>
<p class="hint">About 5 seconds per protein, most of it the search over 133,054 annotated references. Every result
lists the relatives it was read from.</p></form></div>"""


HERO_ART = """<svg class="helix" viewBox="0 0 600 600" aria-hidden="true">
<g stroke="#cfdcd8" stroke-width="1.5" fill="none">
<path d="M300 60 L180 190 M300 60 L420 190 M180 190 L110 330 M180 190 L260 330 M420 190 L340 330 M420 190 L500 330
M260 330 L200 470 M260 330 L330 470 M340 330 L330 470 M500 330 L470 470 M110 330 L200 470"/></g>
<g stroke="#2a7f72" stroke-width="2.5" fill="none" class="flow"><path d="M300 60 L180 190 L260 330 L330 470"/></g>
<g fill="#ffffff" stroke="#b9cbc5" stroke-width="2">
<circle cx="300" cy="60" r="14"/><circle cx="180" cy="190" r="14"/><circle cx="420" cy="190" r="12"/>
<circle cx="110" cy="330" r="10"/><circle cx="260" cy="330" r="14"/><circle cx="340" cy="330" r="10"/>
<circle cx="500" cy="330" r="10"/><circle cx="200" cy="470" r="9"/><circle cx="330" cy="470" r="16"/>
<circle cx="470" cy="470" r="9"/></g>
<g fill="#2a7f72"><circle cx="300" cy="60" r="7"/><circle cx="180" cy="190" r="7"/><circle cx="260" cy="330" r="7"/>
<circle cx="330" cy="470" r="9" class="pulse"/></g>
<text x="352" y="476" fill="#1f6156" font-size="15" font-family="ui-monospace,Menlo,monospace">GO:0003924</text>
<text x="322" y="66" fill="#7d8a8e" font-size="13" font-family="ui-monospace,Menlo,monospace">molecular_function</text>
</svg>"""

PIPELINE_SVG = """<svg viewBox="0 0 1260 380" role="img" aria-labelledby="dgT dgD">
<title id="dgT">Ontologue pipeline</title>
<desc id="dgD">A protein sequence goes two ways: a DIAMOND search reads the GO terms of its closest annotated relatives
(sharpened kNN), and a protein language model with one attention query per GO term reads the residues. A per-term
refiner fitted on future-dated annotations combines both, the Gene Ontology closure makes parents score at least as
high as children, and the output is scored GO terms for molecular function, biological process and cellular
component.</desc>
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0 0 L10 5 L0 10z" fill="#7d8a8e"/></marker></defs>
<g fill="none" stroke="#9db3ad" stroke-width="2" marker-end="url(#ar)">
<path class="flow" d="M170 190 C198 190 198 95 223 95"/><path class="flow" d="M170 190 C198 190 198 285 223 285"/>
<path class="flow" d="M515 95 C542 95 542 190 568 190"/><path class="flow" d="M515 285 C542 285 542 190 568 190"/>
<path class="flow" d="M770 190 L818 190"/><path class="flow" d="M1030 190 L1078 190"/></g>
<g font-family="system-ui,Segoe UI,Roboto,Arial,sans-serif">
<rect x="20" y="140" width="150" height="100" rx="20" fill="#ffffff" stroke="#dcd8cf" stroke-width="1.5"/>
<text x="40" y="176" fill="#2a7f72" font-size="12" font-weight="700" letter-spacing="1.5">INPUT</text>
<text x="40" y="202" fill="#1f2a2e" font-size="17" font-weight="800">Sequence</text>
<text x="40" y="224" fill="#5f6b70" font-size="13">+ organism</text>
<rect x="225" y="40" width="290" height="110" rx="20" fill="#e4f2ee"/>
<text x="245" y="70" fill="#1f6156" font-size="12" font-weight="700" letter-spacing="1.5">01 RELATIVES</text>
<text x="245" y="98" fill="#1f2a2e" font-size="17" font-weight="800">Sharpened kNN</text>
<text x="245" y="120" fill="#5f6b70" font-size="13">DIAMOND over 133k annotated proteins</text>
<text x="245" y="138" fill="#5f6b70" font-size="13">closest relatives weigh the most</text>
<rect x="225" y="230" width="290" height="110" rx="20" fill="#ecebf8"/>
<text x="245" y="260" fill="#4b47a3" font-size="12" font-weight="700" letter-spacing="1.5">02 LANGUAGE</text>
<text x="245" y="288" fill="#1f2a2e" font-size="17" font-weight="800">ESM2 + HiGO head</text>
<text x="245" y="310" fill="#5f6b70" font-size="13">7,901 GO-term queries attend</text>
<text x="245" y="328" fill="#5f6b70" font-size="13">over every residue</text>
<rect x="570" y="135" width="200" height="110" rx="20" fill="#fcefe5"/>
<text x="590" y="165" fill="#a2582a" font-size="12" font-weight="700" letter-spacing="1.5">03 REFINER</text>
<text x="590" y="193" fill="#1f2a2e" font-size="17" font-weight="800">Per-term fusion</text>
<text x="590" y="215" fill="#5f6b70" font-size="13">fitted only on</text>
<text x="590" y="233" fill="#5f6b70" font-size="13">future-dated labels</text>
<rect x="820" y="135" width="210" height="110" rx="20" fill="#eef4f8"/>
<text x="840" y="165" fill="#3b6a8a" font-size="12" font-weight="700" letter-spacing="1.5">04 ONTOLOGY</text>
<text x="840" y="193" fill="#1f2a2e" font-size="17" font-weight="800">True-path closure</text>
<text x="840" y="215" fill="#5f6b70" font-size="13">a parent never scores</text>
<text x="840" y="233" fill="#5f6b70" font-size="13">below its child</text>
<rect x="1080" y="120" width="170" height="140" rx="20" fill="#2a7f72"/>
<text x="1098" y="152" fill="#cfeee6" font-size="12" font-weight="700" letter-spacing="1.5">OUTPUT</text>
<text x="1098" y="180" fill="#ffffff" font-size="17" font-weight="900">GO terms</text>
<circle cx="1104" cy="203" r="6" fill="#a9bdf3"/><text x="1116" y="208" fill="#ffffff" font-size="13">Molecular function</text>
<circle cx="1104" cy="225" r="6" fill="#a8e6d3"/><text x="1116" y="230" fill="#ffffff" font-size="13">Biological process</text>
<circle cx="1104" cy="247" r="6" fill="#f7c6a2"/><text x="1116" y="252" fill="#ffffff" font-size="13">Cellular component</text>
</g></svg>"""


def landing_html() -> str:
    return f"""
<header class="hero">{HERO_ART}
<div class="kicker">Protein function prediction</div>
<h1>Protein function, read from its <em>relatives</em>.</h1>
<p class="lede">Paste a sequence and {BRAND} suggests its Gene Ontology functions. It learns from the closest
annotated relatives and a protein language model, then keeps every answer consistent with the ontology.</p>
<a class="cta" href="#run">Try it with a protein</a><a class="cta ghost" href="#how">How it works</a>
</header>
<div class="stats">
<div><b>0.718</b><span>IA-weighted Fmax on GO-GPT's own temporal test, 8,630 proteins</span></div>
<div><b>+0.019</b><span>over GO-GPT, the GO engine inside BioReason-Pro (95% CI +0.014 to +0.025)</span></div>
<div><b>0</b><span>hierarchy violations in 2.75 million parent-child pairs. GO-GPT: 2.9%</span></div>
<div><b>~5 s</b><span>per protein, with the evidence it used</span></div>
</div>
<section class="block" id="how"><div class="eyebrow">How it works</div>
<h2>Two ways of reading a protein, one consistent answer</h2>
<p class="sub">Most proteins have relatives that someone has already studied. {BRAND} finds them, trusts the closest
ones most, and lets a language model fill the gaps, then keeps the answer consistent with the Gene Ontology.</p>
<div class="diagram">{PIPELINE_SVG}</div></section>
<section class="block" style="padding-top:0"><div class="cards">
<article><div><div class="n">1</div><h3>Closer relatives count for more</h3>
<p>A 99% identical relative is worth far more than a 60% one. Identity-sharpened label transfer weights each
neighbour by how close it is, instead of counting them all alike.</p></div></article>
<article><div><div class="n">2</div><h3>Each function looks at the sequence</h3>
<p>Each of 7,901 GO terms has its own learned query that attends over the residues of a frozen ESM2 model, so
every prediction comes with the residues behind it.</p></div></article>
<article><div><div class="n">3</div><h3>Always consistent with the ontology</h3>
<p>If a protein binds GTP, it binds a nucleotide. Every output is closed under the true-path rule, so a parent
term never scores below its child.</p></div></article>
</div></section>
<section class="block soft" id="results"><div class="eyebrow">Results</div>
<h2>Tested on annotations that came later</h2>
<p class="sub">Every choice was fixed on a temporal validation set before the test set was opened. The test set holds
annotations that appeared after the training cutoff, so the model is graded on functions nobody had recorded yet.</p>
<div class="versus"><table class="vtable">
<tr><th>GO-GPT temporal test</th><th>IA-weighted Fmax</th><th>S-min (lower is better)</th></tr>
<tr class="hl"><td class="us">{BRAND}</td><td class="us">0.718</td><td class="us">7.56</td></tr>
<tr><td>GO-GPT, ten samples</td><td>0.698</td><td>8.17</td></tr>
<tr><td>GO-GPT, released beam</td><td>0.687</td><td>9.20</td></tr></table>
<div class="limits"><h3>Checked carefully</h3>
<p>51 of 56 paired comparisons against GO-GPT stay significant after Holm-Bonferroni correction, across Fmax,
S-min and AUPR in all three ontologies. None of the significant differences favours GO-GPT.</p>
<p>Biggest gains: <b>biological process +0.028</b>, molecular function +0.018, cellular component +0.012
(IA-weighted Fmax).</p></div></div></section>
<section class="block" id="limits"><div class="eyebrow">Know its limits</div>
<h2>Most reliable when relatives exist</h2>
<div class="versus"><p class="sub" style="margin:0">{BRAND}'s edge comes from reading relatives well. For a protein
with no detectable relative, the language-model branch carries the prediction alone, and there GO-GPT is still ahead.
Each result shows the identity of the closest reference, so you know which case you are in.</p>
<table class="vtable"><tr><th>closest relative</th><th>{BRAND}</th><th>GO-GPT</th></tr>
<tr><td>95% identity and above</td><td class="us">0.86 to 0.88</td><td>0.84</td></tr>
<tr><td>30 to 50%</td><td class="us">0.48</td><td>0.45</td></tr>
<tr><td>below 30%</td><td>0.45</td><td class="us">0.46</td></tr>
<tr><td>no hit</td><td>0.34</td><td class="us">0.38</td></tr></table></div></section>
<section class="block run" id="run"><div class="eyebrow">Run it</div>
<h2>Try it with your protein</h2>
<p class="sub">Up to 20 sequences at a time. Results show scored GO terms by ontology, links to QuickGO, the nearest
annotated relatives, and a TSV download.</p>{form_html()}</section>"""


def landing_page() -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{BRAND} | protein function, read from its relatives</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>
{NAV}{landing_html()}{FOOTER}</body></html>""")


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
           f'<a class="dl" href="/download/{key}">Download all scores (TSV)</a> &nbsp; <a href="/#run">New query</a></div>']
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
    return landing_page()


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
    return page(results_html(results, dt, remember_tsv(results)), f"{BRAND} | results")


@app.get("/download/{key}")
def download(key: str):
    if key not in TSV_CACHE:
        raise HTTPException(404, "unknown or expired result; run the prediction again")
    return PlainTextResponse(TSV_CACHE[key], media_type="text/tab-separated-values",
                             headers={"Content-Disposition": 'attachment; filename="ontologue_predictions.tsv"'})


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
