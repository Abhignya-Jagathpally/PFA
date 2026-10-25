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
import json
import html
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from cafa6.eval_page import EVAL_CSS, evaluation_body
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
.tldr{background:#fff;border:1px solid var(--line);border-left:5px solid var(--teal);border-radius:16px;padding:20px 24px;font-size:16.5px;line-height:1.6;color:var(--ink);max-width:1000px}
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
.pair .step .art{max-width:280px}.pair .step .art.wide-art{max-width:none}
.pair .step:nth-child(2) .stag{background:var(--lav);color:#4b47a3}
.step.wide:nth-of-type(1) .stag{background:var(--peach)}
.seqbox{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.7;word-break:break-all;
background:#fafaf8;border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.seqhdr{display:block;color:#7d8a8e;font-size:12px;margin-bottom:2px}.more{color:#7d8a8e;margin-left:6px}
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


NAV = f"""<nav><a class="logo" href="/">{BRAND}<b>.</b></a><div class="links"><a href="/#what">Background</a><a href="/#how">How it works</a>
<a href="/#scores">Scores</a><a href="/#results">Results</a><a href="/#fresh">Fresh data</a><a href="/evaluation">Full evaluation</a><a href="/#limits">Reliability</a><a href="/#run">Run it</a></div></nav>"""

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



SEQ_DEMO = ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYRE"
            "QIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH")
ATTN = json.loads((Path(__file__).parent / "web_assets" / "hras_attention.json").read_text())
# G1-G5 nucleotide-binding boxes of Ras GTPases, positions in human HRAS (Bourne et al., Nature 1991;
# Vetter & Wittinghofer, Science 2001). Taken from the literature, not from the model.
MOTIFS = [("G1", 10, 17), ("G2", 30, 38), ("G3", 57, 60), ("G4", 116, 119), ("G5", 145, 147)]


def attention_svg() -> str:
    """Measured attention of the HiGO 'GTPase activity' query over the 166 residues of the example."""
    a = ATTN["GO:0003924"]
    step, x0, y0, h = 3.0, 14, 30, 92
    peak = max(a)
    bars = "".join(f'<rect x="{x0 + i * step:.1f}" y="{y0 + h - v / peak * h:.1f}" width="2.4" height="{v / peak * h:.1f}" '
                   f'rx="1" fill="#6f6bd0"/>' for i, v in enumerate(a) if v > 0)
    marks = "".join(f'<rect x="{x0 + (s0 - 1) * step:.1f}" y="{y0 + h + 6}" width="{(e - s0 + 1) * step:.1f}" height="8" '
                    f'rx="3" fill="#2a7f72"/>' for _, s0, e in MOTIFS)
    labels = "".join(f'<text x="{x0 + (s0 - 1) * step + dx:.1f}" y="{y0 + h + 28 + dy}" font-size="11" fill="#1f6156" '
                     f'font-weight="600">{n}</text>'
                     for (n, s0, e), dx, dy in zip(MOTIFS, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))
    ticks = "".join(f'<text x="{x0 + (r - 1) * step:.1f}" y="{y0 + h + 52}" font-size="10" fill="#7d8a8e">{r}</text>'
                    for r in (1, 50, 100, 150))
    head = (f'<text x="{x0}" y="16" font-size="12" fill="#4b47a3" font-weight="700">Measured attention of the '
            f'"GTPase activity" query</text>')
    axis = f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + 166 * step}" y2="{y0 + h}" stroke="#cfdcd8"/>'
    return (f'<svg viewBox="0 0 {x0 * 2 + 166 * step:.0f} {y0 + h + 60}" aria-hidden="true">{head}{axis}{bars}{marks}'
            f'{labels}{ticks}</svg>')

DOODLE_RELATIVES = """<svg viewBox="0 0 260 170" aria-hidden="true">
<line x1="130" y1="92" x2="46" y2="40" stroke="#2a7f72" stroke-width="8" stroke-linecap="round"/>
<line x1="130" y1="92" x2="222" y2="50" stroke="#2a7f72" stroke-width="3.5" stroke-linecap="round" opacity=".8"/>
<line x1="130" y1="92" x2="204" y2="150" stroke="#2a7f72" stroke-width="1" stroke-linecap="round" opacity=".55"
stroke-dasharray="3 4"/>
<circle cx="130" cy="92" r="22" fill="#2a7f72"/><text x="130" y="96" text-anchor="middle" fill="#fff" font-size="11"
font-weight="700">query</text>
<circle cx="46" cy="40" r="18" fill="#e4f2ee" stroke="#2a7f72" stroke-width="2"/>
<text x="46" y="44" text-anchor="middle" font-size="10" font-weight="700" fill="#1f6156">99%</text>
<circle cx="222" cy="50" r="16" fill="#e4f2ee" stroke="#2a7f72" stroke-width="1.5"/>
<text x="222" y="54" text-anchor="middle" font-size="10" font-weight="700" fill="#1f6156">94%</text>
<circle cx="204" cy="150" r="14" fill="#f3f1ec" stroke="#b9cbc5" stroke-width="1.5"/>
<text x="204" y="154" text-anchor="middle" font-size="10" fill="#5f6b70">80%</text>
<text x="14" y="80" font-size="10" fill="#5f6b70">weight 1</text><text x="176" y="88" font-size="10"
fill="#5f6b70">about 0.37</text><text x="120" y="160" font-size="10" fill="#5f6b70">about 0.02</text></svg>"""

DOODLE_FUSION = """<svg viewBox="0 0 260 170" aria-hidden="true">
<g font-size="10.5">
<rect x="8" y="14" width="112" height="26" rx="13" fill="#e4f2ee"/><text x="64" y="31" text-anchor="middle"
fill="#1f6156" font-weight="700">homology score</text>
<rect x="8" y="48" width="112" height="26" rx="13" fill="#ecebf8"/><text x="64" y="65" text-anchor="middle"
fill="#4b47a3" font-weight="700">network score</text>
<rect x="8" y="82" width="112" height="26" rx="13" fill="#f3f1ec"/><text x="64" y="99" text-anchor="middle"
fill="#5f6b70">best-hit identity</text>
<rect x="8" y="116" width="112" height="26" rx="13" fill="#f3f1ec"/><text x="64" y="133" text-anchor="middle"
fill="#5f6b70">organism, term depth</text></g>
<g stroke="#9db3ad" stroke-width="1.6" fill="none"><path d="M120 27 C150 27 150 78 172 78"/>
<path d="M120 61 C146 61 150 78 172 78"/><path d="M120 95 C146 95 150 78 172 78"/><path d="M120 129 C150 129 150 78 172 78"/></g>
<rect x="172" y="56" width="80" height="44" rx="14" fill="#fcefe5"/><text x="212" y="76" text-anchor="middle"
font-size="11" font-weight="700" fill="#a2582a">one score</text><text x="212" y="91" text-anchor="middle" font-size="10"
fill="#a2582a">per GO term</text></svg>"""

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
<text x="216" y="88" font-size="10" fill="#5f6b70">is a</text><text x="216" y="30" font-size="10"
fill="#5f6b70">is a</text></svg>"""


def steps_html() -> str:
    def bar(term, asp, v):
        return (f'<div class="obar {asp}"><span class="t">{term}</span><span class="b"><span style="width:{v * 100:.0f}%">'
                f'</span></span><span class="v">{v:.2f}</span></div>')
    m = ATTN["mass_on_motifs"]
    return f"""
<div class="steps">
<div class="step input"><div class="stag">Input</div>
<div class="seqbox"><span class="seqhdr">&gt;example &nbsp;OX=9606 (Homo sapiens) &nbsp;human HRAS, residues 1-166</span>
{SEQ_DEMO}</div>
<p class="legend">All you need is an amino-acid sequence and, if you know it, the organism (an NCBI taxon id). No
multiple sequence alignment, no structure and no prior annotation are required.</p></div>
<div class="arrow2"><span></span><span></span></div>
<div class="pair">
<div class="step"><div class="art">{DOODLE_RELATIVES}</div><div class="txt"><div class="stag">Step 1a</div>
<h3>First, we look for similar proteins</h3><p>We search your sequence with DIAMOND (a fast BLAST-like aligner)
against 133,054 proteins whose GO annotations are already known, and keep up to 50 matches. Each match then votes
for its own GO terms. Votes fall off quickly with distance: for every 5 percentage points of identity below the best
match, a vote counts about a third as much (it is also scaled by alignment score). This is why we call it
<em>sharpened</em>: the closest matches dominate, but a few close ones can still pool their evidence.</p>
<p class="tech">Identity-sharpened k-nearest-neighbour label transfer, temperature 5, top 50 hits</p></div></div>
<div class="step"><div class="art wide-art">{attention_svg()}</div><div class="txt"><div class="stag">Step 1b</div>
<h3>Then, we read the sequence with ESM-2</h3><p>Protein language models like ESM-2 are transformers, like their
large language model counterparts, but trained on protein sequences instead of text. Each amino acid is a token, and
the model learns by predicting masked residues. We use the 35-million-parameter ESM-2 as is (frozen) to describe
every residue, and on top of it a small head with one learned query per GO term (7,901 of them) that weighs the
residues and gives the term a score. In the plot, {m["GO:0003924"]:.0%} of the "GTPase activity" query lands on the
five G boxes that bind GTP in Ras proteins (G1 is the P-loop, G3 the DxxG motif, G4 NKxD and G5 SAK; positions
from the literature), which make up only 17% of the sequence.
Keep in mind that these weights show where the head looks; they are not proof that those residues cause the
function.</p>
<p class="tech">Frozen ESM2-35M with per-term sparse attention pooling (HiGO head)</p></div></div>
</div>
<div class="arrow1"></div>
<div class="step wide"><div class="art">{DOODLE_FUSION}</div><div class="txt"><div class="stag">Step 2</div>
<h3>Next, we combine the two scores</h3><p>For every candidate GO term, a small neural network looks at the homology
score and the ESM-2 score together with a few facts about the evidence: how similar the best match is, whether it
comes from the same organism, and how specific the term is. We fitted it on proteins whose annotations grew after
the reference snapshot, so that it learns how well each kind of evidence holds up for annotations that come later.
In practice, step 1a does most of the work; the ESM-2 score matters most when there is no close match.</p>
<p class="tech">Per-term nonlinear fusion, cross-fitted on a temporal validation set</p></div></div>
<div class="arrow1"></div>
<div class="step wide"><div class="art">{DOODLE_CLOSURE}</div><div class="txt"><div class="stag">Step 3</div>
<h3>Finally, we make it consistent with the ontology</h3><p>GO terms are nested: GTPase activity is a kind of
hydrolase activity, which is a kind of catalytic activity. So if a protein is a GTPase, it is also a hydrolase. We
give every broader term at least the score of its most specific predicted descendant, which means the output never
contradicts itself. The scores in the drawing are the model's real output for the example.</p>
<p class="tech">True-path rule: max-propagation along is_a and part_of relations</p></div></div>
<div class="arrow1"></div>
<div class="step output"><div class="stag">Output</div><h3>Scored GO terms in all three ontologies</h3>
<div class="obars">{bar("GTPase activity", "MF", 0.80)}{bar("small GTPase mediated signal transduction", "BP", 0.75)}
{bar("plasma membrane", "CC", 0.63)}</div>
<p class="legend"><i class="dot MF"></i> molecular function &nbsp; <i class="dot BP"></i> biological process &nbsp;
<i class="dot CC"></i> cellular component. Three of the 138 terms the example scores above the cut-offs.</p></div>
</div>"""


def landing_html() -> str:
    return f"""
<header class="hero">{HERO_ART}
<div class="kicker">Gene Ontology prediction from sequence</div>
<h1>Protein function, inferred from <em>homology</em> and sequence.</h1>
<p class="lede">{BRAND} predicts Gene Ontology terms for your protein from its amino-acid sequence alone. It
transfers annotations from similar, already annotated proteins, adds a score from the protein language model ESM-2,
and makes sure the result is consistent with the hierarchy of the ontology.</p>
<a class="cta" href="#run">Try it with a protein</a><a class="cta ghost" href="#how">How it works</a>
</header>
<section class="block" id="tldr" style="padding-bottom:0"><div class="tldr"><b>TLDR:</b> <em>{BRAND} predicts Gene
Ontology (GO) terms from a single protein sequence by combining annotation transfer from similar proteins with
scores from a frozen ESM-2 protein language model, and its output never contradicts the GO hierarchy. On a temporal
benchmark of 8,630 proteins (GO-GPT's own test) it scores higher than every published method we could compare
fairly, including GO-GPT and BioReason-Pro. On a fresher benchmark of 11,231 proteins annotated after 2024, which
has far fewer close relatives, it still beats GO-GPT but SPROF-GO scores higher. However, most of that accuracy comes from the homology step, and for proteins without a close match
in the reference set the predictions are much weaker, so caution is recommended there. Read on to see how it works
and what the scores mean, and then try it on your favourite protein!</em></div></section>
<section class="block" id="what"><div class="eyebrow">Background</div>
<h2>What is... a GO term?</h2>
<div class="versus"><div class="sub" style="margin:0"><p style="margin-top:0">The Gene Ontology (GO) is a shared
vocabulary for describing what proteins do. It has three parts: <b>molecular function</b> (what a protein does at the
molecular level, for example GTPase activity), <b>biological process</b> (the larger programmes it takes part in,
for example signal transduction) and <b>cellular component</b> (where in the cell it acts, for example the plasma
membrane).</p>
<p>The terms form a hierarchy, from broad ones like "catalytic activity" down to very specific ones. A protein
usually has many terms at once, so predicting them is a multi-label problem, and specific terms are much more
informative than broad ones. That is why the benchmark weights each term by its information content (IA).</p></div>
<div class="limits"><h3>Why predict it at all?</h3><p>Only a small fraction of known protein sequences have
experimentally supported GO annotations, and lab work cannot keep up with sequencing. Computational predictions help
prioritise which functions to test.</p><p>They are hypotheses, not measurements, so always check them against the
evidence that comes with each result.</p></div></div></section>
<section class="block soft" id="get"><div class="eyebrow">What you get</div>
<h2>What does {BRAND} give you?</h2>
<div class="cards">
<article><div class="n">1</div><h3>Terms in three ontologies</h3><p>Molecular function, biological process and
cellular component, each as a table of GO terms with names, scores and links to QuickGO. Terms above the cut-off
come first, and the lower-scoring ones are folded underneath.</p></article>
<article><div class="n">2</div><h3>The closest matches</h3><p>The five nearest annotated proteins, with identity,
coverage, alignment score and e-value, linked to UniProt. That way you can see where the annotations came from and
judge them yourself.</p></article>
<article><div class="n">3</div><h3>A warning when evidence is thin</h3><p>If no reference reaches 30% identity,
the result tells you so. You can also download every score as a TSV, or call the JSON API for batches.</p></article>
</div></section>
<section class="block" id="how"><div class="eyebrow">How it works</div>
<h2>Let's walk through a prediction</h2>
<p class="sub">Every sequence goes through the same four steps. As an example we'll use the GTP-binding domain of
human HRAS, a small GTPase that acts as an on/off switch in growth signalling. Every number shown is the model's own
output for this sequence.</p>
{steps_html()}</section>
<section class="block soft" id="scores"><div class="eyebrow">Reading the scores</div>
<h2>How should you read a score?</h2>
<div class="versus"><div class="sub" style="margin:0"><p style="margin-top:0">A score between 0 and 1 tells you how
strongly the combined evidence supports a term. It is a ranking score, not a calibrated probability, so 0.6 does not
mean a 60% chance.</p>
<p>The results page marks the terms that clear a cut-off for each ontology. We chose these cut-offs because they gave
the best F-measure on the benchmark. Since broader terms always score at least as high as narrower ones, the most
useful predictions are usually the most specific terms that still clear the cut-off.</p></div>
<table class="vtable"><tr><th>ontology</th><th>cut-off</th></tr>
<tr><td>Molecular function</td><td class="us">0.28</td></tr>
<tr><td>Biological process</td><td class="us">0.42</td></tr>
<tr><td>Cellular component</td><td class="us">0.45</td></tr></table></div></section>
<section class="block" id="results"><div class="eyebrow">Results</div>
<h2>How well does it work?</h2>
<p class="sub">We tested {BRAND} on 8,630 proteins whose GO annotations were added after the reference data was
collected, so the model had to predict functions that had not yet been recorded for those proteins. We picked the
final pipeline on a separate validation set before scoring it on this benchmark.</p>
<div class="versus"><div><table class="vtable">
<tr><th>Temporal benchmark, 8,630 proteins</th><th>IA-weighted Fmax</th><th>S-min (lower is better)</th></tr>
<tr class="hl"><td class="us">{BRAND}</td><td class="us">0.718</td><td class="us">7.56</td></tr>
<tr><td>Step 1a alone (homology transfer)</td><td>0.721</td><td>8.48</td></tr>
<tr><td>BLAST best hit, same references</td><td>0.709</td><td>8.42</td></tr>
<tr><td>GO-GPT, ten samples</td><td>0.698</td><td>8.17</td></tr>
<tr><td>BLAST best hit, training references</td><td>0.695</td><td>8.72</td></tr>
<tr><td>GO-GPT, released beam</td><td>0.687</td><td>9.20</td></tr>
<tr><td>BioReason-Pro (RL, released)</td><td>0.649</td><td>10.62</td></tr>
<tr><td>DIAMONDScore</td><td>0.580</td><td>12.50</td></tr>
<tr><td>SPROF-GO (2023)</td><td>0.541</td><td>13.06</td></tr>
<tr><td>DeepGOPlus (2019)</td><td>0.428</td><td>15.80</td></tr>
<tr><td>ProteInfer, 5-model ensemble (2023)</td><td>0.417</td><td>17.20</td></tr>
<tr><td>DeepGO-SE (2024)<sup>&dagger;</sup></td><td>0.302</td><td>16.39</td></tr>
<tr><td>DeepFRI, sequence model (2021)</td><td>0.274</td><td>17.52</td></tr>
<tr><td>Naive (term frequency)</td><td>0.243</td><td>16.98</td></tr>
<tr><td style="color:#a2582a">InterLabelGO+ (2024)<sup>&Dagger;</sup></td><td style="color:#a2582a">0.728</td>
<td style="color:#a2582a">7.80</td></tr>
<tr><td style="color:#7d8a8e">ProtBoost (2024, published, other scorer)</td><td style="color:#7d8a8e">0.606</td>
<td style="color:#7d8a8e">n/a</td></tr></table>
<p class="hint" style="margin-top:10px">We ran every row ourselves on the same proteins with the same scoring
code, from the authors' released code and weights, except ProtBoost (grey), whose models and predictions are no longer
available anywhere, so only its published number from a different scorer is shown. BioReason-Pro reproduces its
authors' published number. "Same references" means the 133,054 proteins {BRAND} searches; "training references"
means the smaller set the published baselines used.<br><sup>&dagger;</sup> DeepGO-SE's released cellular-component
models are faulty (they also fail on the authors' own test split), so its CC score, and therefore its mean, is not
meaningful; its MF 0.522 and BP 0.253 are.<br><sup>&Dagger;</sup> InterLabelGO+ is not a valid comparison here: the
training labels shipped with it come from a later GO snapshot and contain 96% of this benchmark's answers.<br>
Prot2Text-v2 (2025) is not listed because it writes free-text descriptions rather than GO terms and defines no GO
evaluation.</p></div>
<div class="limits"><h3>A word of caution</h3>
<p>Differences were tested with paired protein-level resampling and corrected for multiple comparisons
(Holm-Bonferroni). Every published method that can be compared fairly scores below {BRAND}, and all the
intervals exclude zero. The one method that scores higher, InterLabelGO+, was shipped with training labels that
already contain 96% of this benchmark's answers, so its number is not a comparison.</p>
<p>However, most of the accuracy comes from step 1a. Homology transfer on its own scores 0.721, slightly above the
full pipeline on Fmax, while the full pipeline makes fewer missed or wrong specific terms (S-min 7.56 against 8.48).
A plain BLAST best hit with the same references is also close (+0.009 for {BRAND}, 95% interval +0.004 to +0.015).
So the margin over GO-GPT comes mainly from how the homology evidence is used, not from the language-model branch.</p>
<p>IA-weighted Fmax by ontology: <b>molecular function 0.738</b>, <b>biological process 0.658</b>,
<b>cellular component 0.758</b>.</p><p><a class="cta ghost" style="margin-left:0" href="/evaluation">See the full evaluation</a></p></div></div></section>
<section class="block soft" id="fresh"><div class="eyebrow">Results on fresh data</div>
<h2>What happens on proteins annotated after 2024?</h2>
<p class="sub">The benchmark above is dominated by proteins with a near-identical annotated relative. To check how
the ranking holds up, we built a second benchmark nobody had tuned on: 11,231 proteins that received their first
experimental annotations in an ontology after February 2024 (checked against the archived 2024 annotation release),
none of them in any training or validation set. Only 12% have a relative at 95% identity or more, and 20% have
none at all. Every method below was run once, frozen, with the scoring fixed before any result existed.</p>
<div class="versus"><div><table class="vtable">
<tr><th>Fresh benchmark, 11,231 proteins</th><th>IA-weighted Fmax</th><th>S-min</th></tr>
<tr><td class="us">SPROF-GO (2023)</td><td class="us">0.515</td><td class="us">11.95</td></tr>
<tr><td style="color:#7d8a8e">Forecasting model, ESM2-650M (research, not served)<sup>&sect;</sup></td><td style="color:#7d8a8e">0.514</td><td style="color:#7d8a8e">12.34</td></tr>
<tr><td style="color:#7d8a8e">Forecasting model, ESM2-35M (research)<sup>&sect;</sup></td><td style="color:#7d8a8e">0.487</td><td style="color:#7d8a8e">12.64</td></tr>
<tr class="hl"><td class="us">{BRAND}</td><td>0.479</td><td>12.65</td></tr>
<tr><td>GO-GPT, ten samples</td><td>0.468</td><td>12.53</td></tr>
<tr><td>DeepGOPlus (2019)</td><td>0.466</td><td>12.95</td></tr>
<tr><td>GO-GPT, released beam</td><td>0.459</td><td>14.46</td></tr>
<tr><td>Step 1a alone (homology transfer)</td><td>0.458</td><td>12.55</td></tr>
<tr><td>DIAMONDScore</td><td>0.456</td><td>12.45</td></tr>
<tr><td>ProteInfer, 5-model ensemble (2023)</td><td>0.453</td><td>13.64</td></tr>
<tr><td>BLAST best hit, same references</td><td>0.375</td><td>13.58</td></tr>
<tr><td>DeepFRI, sequence model (2021)</td><td>0.270</td><td>15.16</td></tr>
<tr><td>Naive (term frequency)</td><td>0.230</td><td>14.65</td></tr></table>
<p class="hint" style="margin-top:10px"><sup>&sect;</sup> Versions of {BRAND} trained to forecast which annotations a
protein will gain next, rather than to reproduce its current ones. With the larger ESM2-650M encoder it ties SPROF-GO
(0.514 against 0.515, difference not significant) and beats {BRAND} by 0.034 and GO-GPT by 0.046; it is ahead of
SPROF-GO from 50% identity upward and behind it below. It is not yet the model behind this website.</p></div>
<div class="limits"><h3>What this tells you</h3>
<p>{BRAND} still beats GO-GPT on fresh data (+0.012, 95% interval +0.007 to +0.017), but every method built on GO-GPT's
training data drops sharply (from about 0.70 to about 0.47), because much of the first benchmark rewarded copying
near-identical relatives.</p><p><b>SPROF-GO is ahead in every identity group below 95%</b> (0.454 against 0.334 with
no relative at all). For proteins without close annotated relatives, it is currently the better choice; {BRAND} is
the better choice at 95% identity and above.</p></div></div></section>
<section class="block soft" id="limits"><div class="eyebrow">When to trust a prediction</div>
<h2>When should you trust a prediction?</h2>
<div class="versus"><div class="sub" style="margin:0"><p style="margin-top:0">Most of the accuracy comes from
annotated proteins that resemble yours, so the first thing to check on any result is the identity of its closest
match.</p><p><b>95% identity or more:</b> most predicted terms are correct.<br><b>50 to 95%:</b> useful, but have a
look at the matches before relying on specific terms.<br><b>Below 50%, or no match:</b> treat the output as
hypotheses. Without a close match the ESM-2 score carries the prediction, and accuracy is much lower. Here
sequence-based models do better: SPROF-GO is ahead below 50% identity (0.40 against 0.34 with no match) and
level at 50 to 80%, and GO-GPT is ahead below 30%.</p></div>
<table class="vtable"><tr><th>identity of closest match</th><th>{BRAND}</th><th>GO-GPT (ten samples)</th><th>SPROF-GO</th></tr>
<tr><td>95% and above</td><td class="us">0.86 to 0.88</td><td>0.84</td><td>0.55 to 0.58</td></tr>
<tr><td>80 to 95%</td><td class="us">0.68</td><td>0.67</td><td>0.57</td></tr>
<tr><td>50 to 80%</td><td class="us">0.51</td><td>0.49</td><td>0.51</td></tr>
<tr><td>30 to 50%</td><td>0.48</td><td>0.45</td><td class="us">0.51</td></tr>
<tr><td>below 30%</td><td>0.45</td><td>0.46</td><td class="us">0.53</td></tr>
<tr><td>no match</td><td>0.34</td><td>0.38</td><td class="us">0.40</td></tr></table></div>
<p class="hint" style="margin-top:14px">IA-weighted Fmax on the benchmark by identity of the closest reference
protein.</p></section>
<section class="block" id="scope"><div class="eyebrow">Scope</div>
<h2>No MSA or structure required, but some limits</h2>
<div class="cards">
<article><div class="n">&ndash;</div><h3>Sequence only</h3><p>{BRAND} does not need multiple sequence alignments,
3D structures, predicted structures or domain databases such as InterPro. That makes it fast and easy to use, but it
also means it cannot use any of that information when it would help.</p></article>
<article><div class="n">&ndash;</div><h3>A fixed reference snapshot</h3><p>The 133,054 reference proteins and
their annotations are a fixed snapshot (annotations up to November 2022). Anything annotated since then is not used,
even for well-studied proteins.</p></article>
<article><div class="n">&ndash;</div><h3>No text or interactions</h3><p>It does not read the literature,
protein-protein interaction networks or expression data, and it does not write explanations. Its evidence is the
matches it lists and the scores it returns.</p></article>
</div></section>
<section class="block run" id="run"><div class="eyebrow">Run it</div>
<h2>Try it on your favourite protein!</h2>
<p class="sub">Paste up to 20 sequences at a time, 5,000 residues each. If you know a thing or two about your
protein, compare the predictions with what you know, and tell us where they go wrong.</p>{form_html()}</section>"""

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


@app.get("/evaluation", response_class=HTMLResponse)
def evaluation():
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{BRAND} | full evaluation</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}{EVAL_CSS}</style></head><body>
{NAV}{evaluation_body()}{FOOTER}</body></html>""")


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
