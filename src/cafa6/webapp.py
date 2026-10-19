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
.steps{display:flex;flex-direction:column;align-items:stretch;gap:0}
.step{background:#fff;border:1px solid var(--line);border-radius:22px;padding:24px 26px;display:grid;
grid-template-columns:260px 1fr;gap:26px;align-items:center;box-shadow:0 20px 44px -36px rgba(31,42,46,.35)}
.step .art svg{width:100%;height:auto;display:block}.step h3{font-family:var(--serif);font-weight:600;font-size:22px;
margin:4px 0 8px}.step p{margin:0 0 8px;color:var(--muted);font-size:15px}.step p.tech{font-size:12.5px;color:#7d8a8e;
border-top:1px dashed var(--line);padding-top:8px;margin-top:10px}
.stag{display:inline-block;font-size:12px;font-weight:700;color:var(--teal-d);background:var(--mint);padding:3px 10px;
border-radius:999px}
.step.input,.step.output{display:block}.step.input .stag,.step.output .stag{margin-bottom:12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.pair .step{grid-template-columns:1fr;gap:12px}
.pair .step .art{max-width:280px}
.pair .step:nth-child(2) .stag{background:var(--lav);color:#4b47a3}
.step.wide:nth-of-type(1) .stag{background:var(--peach)}
.seqbox{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:2.1;word-break:break-all;
background:#fafaf8;border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.seqhdr{display:block;color:#7d8a8e;font-size:12px;margin-bottom:2px}.more{color:#7d8a8e;margin-left:6px}
i.aa{font-style:normal;display:inline-block;width:1.3em;text-align:center;border-radius:5px;margin:0 1px}
i.aa.h{background:#e4f2ee;color:#1f6156}i.aa.c{background:#fcefe5;color:#a2582a}i.aa.p{background:#ecebf8;color:#4b47a3}
.legend{font-size:12.5px;color:var(--muted);margin:10px 0 0}
.arrow1{width:2px;height:34px;background:linear-gradient(#cfdcd8,#9db3ad);margin:0 auto;position:relative}
.arrow1:after{content:"";position:absolute;left:-5px;bottom:-2px;border:6px solid transparent;border-top-color:#9db3ad}
.arrow2{display:grid;grid-template-columns:1fr 1fr;height:34px}.arrow2 span{width:2px;height:34px;margin:0 auto;
background:linear-gradient(#cfdcd8,#9db3ad)}
.obars{display:grid;gap:10px;max-width:720px}.obar{display:grid;grid-template-columns:minmax(180px,330px) 1fr 44px;gap:12px;
align-items:center;font-size:14px}.obar .b{background:#f0eee9;border-radius:99px;height:10px}.obar .b span{display:block;
height:10px;border-radius:99px}.obar.MF .b span{background:var(--mf)}.obar.BP .b span{background:var(--bp)}
.obar.CC .b span{background:var(--cc)}.obar .v{font-weight:600;color:var(--ink)}
i.dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle}i.dot.MF{background:var(--mf)}
i.dot.BP{background:var(--bp)}i.dot.CC{background:var(--cc)}
@media (max-width:900px){.step{grid-template-columns:1fr}.pair{grid-template-columns:1fr}.arrow2{grid-template-columns:1fr}
.arrow2 span:last-child{display:none}}
"""


def predictor() -> Predictor:
    if STATE["predictor"] is None:
        STATE["predictor"] = Predictor()
    return STATE["predictor"]


NAV = f"""<nav><a class="logo" href="/">{BRAND}<b>.</b></a><div class="links"><a href="/#how">How it works</a>
<a href="/#results">Results</a><a href="/#limits">Limits</a><a href="/#run">Run it</a></div></nav>"""

FOOTER = ""


def page(body: str, title: str = f"{BRAND} | protein function from homology and sequence") -> HTMLResponse:
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
lists the homologues it drew on.</p></form></div>"""


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



SEQ_DEMO = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYRE"


def seq_chips(seq: str) -> str:
    """Residues as coloured chips: hydrophobic, charged, polar/other."""
    cls = {**{a: "h" for a in "AVILMFWC"}, **{a: "c" for a in "DEKRH"}}
    return "".join(f'<i class="aa {cls.get(a, "p")}">{a}</i>' for a in seq)


DOODLE_RELATIVES = """<svg viewBox="0 0 260 170" aria-hidden="true">
<line x1="130" y1="88" x2="46" y2="40" stroke="#2a7f72" stroke-width="7" stroke-linecap="round"/>
<line x1="130" y1="88" x2="222" y2="52" stroke="#2a7f72" stroke-width="4" stroke-linecap="round" opacity=".75"/>
<line x1="130" y1="88" x2="200" y2="150" stroke="#2a7f72" stroke-width="1.6" stroke-linecap="round" opacity=".5"
stroke-dasharray="4 4"/>
<circle cx="130" cy="88" r="22" fill="#2a7f72"/><text x="130" y="92" text-anchor="middle" fill="#fff" font-size="11"
font-weight="700">query</text>
<circle cx="46" cy="40" r="17" fill="#e4f2ee" stroke="#2a7f72" stroke-width="2"/>
<text x="46" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1f6156">99%</text>
<circle cx="222" cy="52" r="15" fill="#e4f2ee" stroke="#2a7f72" stroke-width="1.5"/>
<text x="222" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#1f6156">78%</text>
<circle cx="200" cy="150" r="12" fill="#f3f1ec" stroke="#b9cbc5" stroke-width="1.5"/>
<text x="200" y="154" text-anchor="middle" font-size="9" fill="#5f6b70">41%</text>
<text x="18" y="76" font-size="10" fill="#5f6b70">strong vote</text><text x="168" y="98" font-size="10"
fill="#5f6b70">weaker</text><text x="132" y="160" font-size="10" fill="#5f6b70">faint</text></svg>"""

DOODLE_LANGUAGE = """<svg viewBox="0 0 260 170" aria-hidden="true">
<path d="M66 34 L40 118 L92 118 Z" fill="#ecebf8"/><path d="M186 34 L160 118 L212 118 Z" fill="#fcefe5"/>
<rect x="28" y="16" width="76" height="22" rx="11" fill="#ecebf8"/><text x="66" y="31" text-anchor="middle"
font-size="10" font-weight="700" fill="#4b47a3">GTP binding</text>
<rect x="142" y="16" width="88" height="22" rx="11" fill="#fcefe5"/><text x="186" y="31" text-anchor="middle"
font-size="10" font-weight="700" fill="#a2582a">membrane</text>
<g font-size="10" font-weight="700" text-anchor="middle">
<circle cx="22" cy="130" r="9" fill="#f3f1ec"/><circle cx="44" cy="130" r="9" fill="#a9a5e6"/>
<circle cx="66" cy="130" r="9" fill="#a9a5e6"/><circle cx="88" cy="130" r="9" fill="#f3f1ec"/>
<circle cx="110" cy="130" r="9" fill="#f3f1ec"/><circle cx="132" cy="130" r="9" fill="#f3f1ec"/>
<circle cx="154" cy="130" r="9" fill="#f3f1ec"/><circle cx="176" cy="130" r="9" fill="#f7c6a2"/>
<circle cx="198" cy="130" r="9" fill="#f7c6a2"/><circle cx="220" cy="130" r="9" fill="#f3f1ec"/>
<circle cx="242" cy="130" r="9" fill="#f3f1ec"/></g>
<text x="130" y="160" text-anchor="middle" font-size="10" fill="#5f6b70">each function looks at the residues it needs</text>
</svg>"""

DOODLE_FUSION = """<svg viewBox="0 0 260 170" aria-hidden="true">
<path d="M130 40 L130 140" stroke="#9db3ad" stroke-width="3"/><path d="M104 146 L156 146" stroke="#9db3ad"
stroke-width="5" stroke-linecap="round"/>
<path d="M40 58 L220 38" stroke="#5f6b70" stroke-width="3" stroke-linecap="round"/>
<circle cx="130" cy="48" r="6" fill="#5f6b70"/>
<path d="M40 58 L22 96 L58 96 Z" fill="none" stroke="#9db3ad" stroke-width="1.5"/>
<path d="M220 38 L202 76 L238 76 Z" fill="none" stroke="#9db3ad" stroke-width="1.5"/>
<ellipse cx="40" cy="98" rx="26" ry="6" fill="#e4f2ee"/><ellipse cx="220" cy="78" rx="26" ry="6" fill="#ecebf8"/>
<circle cx="40" cy="86" r="11" fill="#2a7f72"/><circle cx="220" cy="68" r="8" fill="#a9a5e6"/>
<text x="40" y="122" text-anchor="middle" font-size="10" font-weight="700" fill="#1f6156">homologues</text>
<text x="220" y="102" text-anchor="middle" font-size="10" font-weight="700" fill="#4b47a3">sequence</text>
<text x="130" y="166" text-anchor="middle" font-size="10" fill="#5f6b70">the balance shifts with similarity</text>
</svg>"""

DOODLE_CLOSURE = """<svg viewBox="0 0 260 170" aria-hidden="true">
<g font-size="11" font-weight="600">
<rect x="58" y="10" width="144" height="30" rx="15" fill="#eef4f8"/><text x="130" y="30" text-anchor="middle"
fill="#3b6a8a">catalytic activity 0.87</text>
<rect x="50" y="68" width="160" height="30" rx="15" fill="#eef4f8"/><text x="130" y="88" text-anchor="middle"
fill="#3b6a8a">hydrolase activity 0.87</text>
<rect x="62" y="126" width="136" height="30" rx="15" fill="#2a7f72"/><text x="130" y="146" text-anchor="middle"
fill="#fff">GTPase activity 0.80</text></g>
<g stroke="#9db3ad" stroke-width="2" fill="none"><path d="M130 124 L130 102"/><path d="M124 108 L130 100 L136 108"/>
<path d="M130 66 L130 44"/><path d="M124 50 L130 42 L136 50"/></g>
<text x="222" y="92" font-size="10" fill="#5f6b70">implies</text><text x="222" y="40" font-size="10"
fill="#5f6b70">implies</text></svg>"""


def steps_html() -> str:
    def bar(term, asp, v):
        return (f'<div class="obar {asp}"><span class="t">{term}</span><span class="b"><span style="width:{v * 100:.0f}%">'
                f'</span></span><span class="v">{v:.2f}</span></div>')
    return f"""
<div class="steps">
<div class="step input"><div class="stag">Input</div>
<div class="seqbox"><span class="seqhdr">&gt;example &nbsp;Homo sapiens (taxon 9606)</span>{seq_chips(SEQ_DEMO)}<span
class="more">&hellip; 166 residues</span></div>
<p class="legend"><i class="aa h">A</i> hydrophobic &nbsp; <i class="aa c">K</i> charged &nbsp; <i class="aa p">S</i>
polar or other</p></div>
<div class="arrow2"><span></span><span></span></div>
<div class="pair">
<div class="step"><div class="art">{DOODLE_RELATIVES}</div><div class="txt"><div class="stag">Step 1a</div>
<h3>Look up related proteins</h3><p>We search 133,054 proteins whose functions are already known for ones with a
similar sequence, and borrow their annotations. A near-identical protein gets a strong vote; a distant one barely
counts.</p>
<p class="tech">Identity-sharpened k-nearest-neighbour label transfer (DIAMOND, top 50 hits)</p></div></div>
<div class="step"><div class="art">{DOODLE_LANGUAGE}</div><div class="txt"><div class="stag">Step 1b</div>
<h3>Read the sequence itself</h3><p>A protein language model, trained on millions of natural sequences, reads every
residue. Each of 7,901 functions then focuses on the stretch of residues that tells it the most.</p>
<p class="tech">Frozen ESM2-35M encoder with one attention query per GO term (HiGO head)</p></div></div>
</div>
<div class="arrow1"></div>
<div class="step wide"><div class="art">{DOODLE_FUSION}</div><div class="txt"><div class="stag">Step 2</div>
<h3>Weigh the two kinds of evidence</h3><p>For each function separately, a small model decides how far to trust the
related proteins and how far to trust the sequence reading. Close matches tip the balance toward borrowing; with no
close match the sequence reading matters more. It was calibrated on annotations added after the training data was
collected, so it learns how evidence holds up over time.</p>
<p class="tech">Per-term nonlinear fusion fitted on a temporal validation set</p></div></div>
<div class="arrow1"></div>
<div class="step wide"><div class="art">{DOODLE_CLOSURE}</div><div class="txt"><div class="stag">Step 3</div>
<h3>Keep the answer consistent</h3><p>Gene Ontology terms are nested: a GTPase is a kind of hydrolase, which is a kind
of catalytic activity. Whenever a specific function is predicted, every broader function above it is scored at least
as high, so the result never contradicts itself.</p>
<p class="tech">True-path rule: max-propagation over is_a and part_of relations</p></div></div>
<div class="arrow1"></div>
<div class="step output"><div class="stag">Output</div><h3>Scored functions in all three ontologies</h3>
<div class="obars">{bar("GTPase activity", "MF", 0.80)}{bar("small GTPase mediated signal transduction", "BP", 0.75)}
{bar("plasma membrane", "CC", 0.63)}</div>
<p class="legend"><i class="dot MF"></i> molecular function &nbsp; <i class="dot BP"></i> biological process &nbsp;
<i class="dot CC"></i> cellular component. Values are the model's own scores for the example above.</p></div>
</div>"""


def landing_html() -> str:
    return f"""
<header class="hero">{HERO_ART}
<div class="kicker">Protein function prediction</div>
<h1>Protein function, inferred from <em>homology</em> and sequence.</h1>
<p class="lede">{BRAND} predicts Gene Ontology annotations from an amino-acid sequence. It combines label transfer
from annotated homologues with a protein language model, and every prediction respects the hierarchy of the
ontology.</p>
<a class="cta" href="#run">Try it with a protein</a><a class="cta ghost" href="#how">How it works</a>
</header>
<div class="stats">
<div><b>0.718</b><span>IA-weighted Fmax on a temporal benchmark of 8,630 proteins</span></div>
<div><b>133k</b><span>experimentally annotated reference proteins searched for every query</span></div>
<div><b>0</b><span>hierarchy violations in 2.75 million parent-child term pairs</span></div>
<div><b>~5 s</b><span>per protein, reported with the homologues it used</span></div>
</div>
<section class="block" id="how"><div class="eyebrow">How it works</div>
<h2>Two sources of evidence, one consistent annotation</h2>
<p class="sub">Most proteins have homologues whose functions have been characterised. {BRAND} finds them, weights
the closest most heavily, uses a protein language model where homology is weak, and keeps the result consistent
with the Gene Ontology.</p>
{steps_html()}</section>

<section class="block soft" id="results"><div class="eyebrow">Results</div>
<h2>Tested on annotations that came later</h2>
<p class="sub">Every choice was fixed on a temporal validation set before the test set was opened. The test set holds
annotations that appeared after the training cutoff, so the model is graded on functions nobody had recorded yet.</p>
<div class="versus"><table class="vtable">
<tr><th>Temporal benchmark, 8,630 proteins</th><th>IA-weighted Fmax</th><th>S-min (lower is better)</th></tr>
<tr class="hl"><td class="us">{BRAND}</td><td class="us">0.718</td><td class="us">7.56</td></tr>
<tr><td>GO-GPT, ten samples</td><td>0.698</td><td>8.17</td></tr>
<tr><td>GO-GPT, released beam</td><td>0.687</td><td>9.20</td></tr></table>
<div class="limits"><h3>Checked carefully</h3>
<p>Differences were tested with paired protein-level resampling and corrected for multiple comparisons
(Holm-Bonferroni) across Fmax, S-min and AUPR in all three ontologies.</p>
<p>Biggest gains: <b>biological process +0.028</b>, molecular function +0.018, cellular component +0.012
(IA-weighted Fmax).</p></div></div></section>
<section class="block" id="limits"><div class="eyebrow">Know its limits</div>
<h2>Most reliable when homologues exist</h2>
<div class="versus"><p class="sub" style="margin:0">Accuracy depends on how close the nearest annotated homologue is.
For a protein with no detectable homologue the language-model branch carries the prediction alone, and accuracy is
markedly lower. Each result reports the identity of the closest reference, so you know which case applies.</p>
<table class="vtable"><tr><th>identity of closest homologue</th><th>{BRAND}</th><th>GO-GPT (ten samples)</th></tr>
<tr><td>95% identity and above</td><td class="us">0.86 to 0.88</td><td>0.84</td></tr>
<tr><td>30 to 50%</td><td class="us">0.48</td><td>0.45</td></tr>
<tr><td>below 30%</td><td>0.45</td><td class="us">0.46</td></tr>
<tr><td>no hit</td><td>0.34</td><td class="us">0.38</td></tr></table></div></section>
<section class="block run" id="run"><div class="eyebrow">Run it</div>
<h2>Try it with your protein</h2>
<p class="sub">Up to 20 sequences at a time. Results show scored GO terms by ontology, links to QuickGO, the nearest
annotated homologues, and a TSV download.</p>{form_html()}</section>"""


def landing_page() -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{BRAND} | protein function from homology and sequence</title>
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
        org = html.escape(r["organism"] or "organism not in the reference data")
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
        parts.append("<h3>Nearest annotated homologues (DIAMOND)</h3>")
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
