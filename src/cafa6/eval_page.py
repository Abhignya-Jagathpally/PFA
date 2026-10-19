"""Full-evaluation page for the Ontologue web app, rendered from the eval_suite output files.

Reads runs/eval_suite_v2/ when it exists (all compared methods), else runs/eval_suite/ (Ontologue vs GO-GPT). Tables
are grouped by what they measure: primary accuracy, generalisation to novel proteins, ontology consistency and
statistical reliability. Nothing is computed here; every number comes from the TSV files.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ASPECTS = ["MF", "BP", "CC", "mean"]
ASPECT_NAMES = {"MF": "Molecular function", "BP": "Biological process", "CC": "Cellular component", "mean": "Mean"}
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
BIN_NAMES = {"no_hit": "no hit", "<30": "&lt;30%", "30-50": "30&ndash;50%", "50-80": "50&ndash;80%",
             "80-95": "80&ndash;95%", "95-99.9": "95&ndash;99.9%", "100": "100%"}
DISPLAY = {"HiGO-R2": "Ontologue", "[component] kNN prior (train+val refs)": "Step 1a alone (homology transfer)",
           "[component] HiGO head alone": "Step 1b alone (ESM-2 head)"}
ORDER = ["HiGO-R2", "[component] kNN prior (train+val refs)", "[component] HiGO head alone",
         "BLAST-KNN (train+val refs)", "GO-GPT ten-sample", "BLAST-KNN (training refs)", "GO-GPT beam", "GO-GPT greedy",
         "BioReason-Pro RL", "DIAMONDScore", "Naive"]
BINARY = {"GO-GPT beam", "GO-GPT greedy", "BioReason-Pro RL"}
METRIC_NAMES = {"Fmax": "F-max", "Fmax_w": "IA-weighted F-max", "Smin_w": "S-min (IA-based, lower is better)",
                "Smin": "S-min (count-based)", "AUPR_micro": "AUPRC (pooled pairs)",
                "AUPR_micro_w": "AUPRC (pooled pairs, IA-weighted)", "AUPR_prot": "AUPRC (per protein)",
                "AUPR_prot_w": "AUPRC (per protein, IA-weighted)"}


def source_dir() -> Path:
    v2 = ROOT / "runs" / "eval_suite_v2"
    return v2 if (v2 / "summary.tsv").exists() else ROOT / "runs" / "eval_suite"


def name(m: str) -> str:
    return html.escape(DISPLAY.get(m, m.replace("HiGO-R2", "Ontologue")))


def order(models) -> list[str]:
    ms = list(dict.fromkeys(models))
    return sorted(ms, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))


def fmt(v, lo=None, hi=None, d=3) -> str:
    if v is None or pd.isna(v):
        return "&ndash;"
    ci = f'<span class="ci">{lo:.{d}f}&ndash;{hi:.{d}f}</span>' if lo is not None and not pd.isna(lo) else ""
    return f"{v:.{d}f}{ci}"


def metric_table(df: pd.DataFrame, metric: str, lower_better=False, d=3, flag_binary=False) -> str:
    t = df[df.metric == metric]
    models = order(t.model)
    best = {}
    for a in ASPECTS:
        col = t[(t.aspect == a) & t.model.isin([m for m in models if not m.startswith("[component]")])]
        if len(col):
            best[a] = (col.value.min() if lower_better else col.value.max())
    rows = []
    for m in models:
        cells = []
        for a in ASPECTS:
            r = t[(t.model == m) & (t.aspect == a)]
            if not len(r):
                cells.append("<td>&ndash;</td>")
                continue
            r = r.iloc[0]
            strong = (a in best and abs(r.value - best[a]) < 1e-12 and not m.startswith("[component]"))
            cells.append(f'<td class="{"b" if strong else ""}">{fmt(r.value, r.lo, r.hi, d)}</td>')
        cls = "hl" if m == "HiGO-R2" else ("comp" if m.startswith("[component]") else "")
        mark = ' <span class="flag" title="binary scores">*</span>' if flag_binary and m in BINARY else ""
        rows.append(f'<tr class="{cls}"><td>{name(m)}{mark}</td>{"".join(cells)}</tr>')
    head = "".join(f"<th>{ASPECT_NAMES[a]}</th>" for a in ASPECTS)
    return f'<table class="etable"><tr><th>method</th>{head}</tr>{"".join(rows)}</table>'


def bins_table(ib: pd.DataFrame, f1: str) -> str:
    t = ib[(ib.identity == "train") & (ib.f1 == f1) & (ib.aspect == "mean")]
    models = order(t.model)
    n = {b: int(t[t.bin == b].n.max()) for b in BINS if len(t[t.bin == b])}
    best = {b: t[(t.bin == b) & ~t.model.str.startswith("[component]")].mean_f1.max() for b in BINS}
    rows = [f'<tr class="nrow"><td>proteins in bin</td>{"".join(f"<td>{n.get(b, 0):,}</td>" for b in BINS)}</tr>']
    for m in models:
        cells = []
        for b in BINS:
            r = t[(t.model == m) & (t.bin == b)]
            if not len(r):
                cells.append("<td>&ndash;</td>")
                continue
            r = r.iloc[0]
            strong = abs(r.mean_f1 - best[b]) < 1e-12 and not m.startswith("[component]")
            cells.append(f'<td class="{"b" if strong else ""}">{fmt(r.mean_f1, r.lo, r.hi)}</td>')
        cls = "hl" if m == "HiGO-R2" else ("comp" if m.startswith("[component]") else "")
        rows.append(f'<tr class="{cls}"><td>{name(m)}</td>{"".join(cells)}</tr>')
    head = "".join(f"<th>{BIN_NAMES[b]}</th>" for b in BINS)
    return f'<table class="etable"><tr><th>identity of nearest training protein</th>{head}</tr>{"".join(rows)}</table>'


def spearman_table(sp: pd.DataFrame) -> str:
    t = sp[(sp.identity == "train") & (sp.f1 == "F1_w")]
    rows = []
    for m in order(t.model):
        cells = "".join(f"<td>{fmt(*t[(t.model == m) & (t.aspect == a)][['spearman', 'lo', 'hi']].iloc[0])}</td>"
                        if len(t[(t.model == m) & (t.aspect == a)]) else "<td>&ndash;</td>" for a in ASPECTS)
        cls = "hl" if m == "HiGO-R2" else ("comp" if m.startswith("[component]") else "")
        rows.append(f'<tr class="{cls}"><td>{name(m)}</td>{cells}</tr>')
    head = "".join(f"<th>{ASPECT_NAMES[a]}</th>" for a in ASPECTS)
    return f'<table class="etable"><tr><th>method</th>{head}</tr>{"".join(rows)}</table>'


def consistency_table(c: pd.DataFrame) -> str:
    rows = []
    for m in order(c.model):
        t = c[c.model == m]
        pairs, viol = int(t.pairs.sum()), int(t.violations.sum())
        rate = viol / pairs if pairs else 0.0
        per = {a: t[t.aspect == a].rate.iloc[0] if len(t[t.aspect == a]) else float("nan") for a in ("MF", "BP", "CC")}
        prot = t.proteins_with_violation.sum() / max(t.proteins_scored.max(), 1)
        cls = "hl" if m == "HiGO-R2" else ("comp" if m.startswith("[component]") else "")
        rows.append(f'<tr class="{cls}"><td>{name(m)}</td><td>{pairs:,}</td><td class="{"b" if viol == 0 else ""}">'
                    f'{rate:.2%}</td>' + "".join(f"<td>{per[a]:.2%}</td>" for a in ("MF", "BP", "CC")) + "</tr>")
    return ('<table class="etable"><tr><th>method</th><th>parent-child pairs checked</th><th>violation rate</th>'
            '<th>MF</th><th>BP</th><th>CC</th></tr>' + "".join(rows) + "</table>")


def retrieval_table(r: pd.DataFrame) -> str:
    r = r[r.vocabulary.str.startswith("common")] if (r.vocabulary.str.startswith("common")).any() else r
    r = r[~r.vectors.str.contains("seeds, sd")]
    label = {"HiGO effective query q = e + mean_anc(e)": "Ontologue: learned term queries",
             "HiGO raw term_emb e": "Ontologue: raw term embeddings",
             "GO-GPT token embedding (wte)": "GO-GPT: token embeddings"}
    rows = []
    for _, x in r.iterrows():
        lab = label.get(x.vectors, x.vectors.replace("HiGO", "Ontologue"))
        if x.vectors.startswith("untrained"):
            lab = "Untrained vectors, same ancestor-mean construction"
        cls = "hl" if "effective" in x.vectors else ""
        rows.append(f'<tr class="{cls}"><td>{html.escape(lab)}</td>' + "".join(
            f"<td>{x[f'direct_{k}']:.3f}</td>" for k in ("P@1", "P@5", "P@10", "R@1", "R@5", "R@10")) + "</tr>")
    rnd = r.iloc[0]
    rows.append('<tr class="comp"><td>Random ranking (expected)</td>' + "".join(
        f"<td>{rnd[f'direct_random_{k}']:.4f}</td>" for k in ("P@1", "P@5", "P@10", "R@1", "R@5", "R@10")) + "</tr>")
    head = "".join(f"<th>{k}</th>" for k in ("P@1", "P@5", "P@10", "R@1", "R@5", "R@10"))
    return f'<table class="etable"><tr><th>term vectors</th>{head}</tr>{"".join(rows)}</table>'


def tests_html(p: pd.DataFrame) -> str:
    fam = []
    for f, t in p.groupby("family", sort=False):
        fam.append(f"<tr><td>{html.escape(f)}</td><td>{len(t)}</td><td>{int((t.p_holm < 0.05).sum())}</td>"
                   f"<td>{int((t.q_bh < 0.05).sum())}</td><td>{int(((t.p_holm < 0.05) & (t['diff'] > 0)).sum())}</td>"
                   f"<td>{int(((t.p_holm < 0.05) & (t['diff'] < 0)).sum())}</td></tr>")
    summary = ('<table class="etable"><tr><th>family</th><th>tests</th><th>significant (Holm)</th>'
               '<th>significant (BH)</th><th>Ontologue better</th><th>Ontologue worse</th></tr>' + "".join(fam) + "</table>")
    t = p[(p.family == "primary") & (p.aspect == "mean")]
    comps = list(dict.fromkeys(t.comparison))
    mets = list(dict.fromkeys(t.metric))
    lower = {"Smin_w", "Smin"}
    head = "".join(f"<th>vs {html.escape(c.replace('HiGO-R2 - ', ''))}</th>" for c in comps)
    rows = []
    for met in mets:
        cells = []
        for c in comps:
            r = t[(t.metric == met) & (t.comparison == c)]
            if not len(r):
                cells.append("<td>&ndash;</td>")
                continue
            r = r.iloc[0]
            good = (r["diff"] < 0) if met in lower else (r["diff"] > 0)
            sig = r.p_holm < 0.05
            mark = "sig-good" if sig and good else ("sig-bad" if sig else "ns")
            cells.append(f'<td class="{mark}">{r["diff"]:+.3f}<span class="ci">{r.lo:+.3f}&ndash;{r.hi:+.3f}</span></td>')
        rows.append(f"<tr><td>{METRIC_NAMES.get(met, html.escape(met))}</td>{''.join(cells)}</tr>")
    grid = f'<table class="etable"><tr><th>metric (mean over ontologies)</th>{head}</tr>{"".join(rows)}</table>'
    full = p[["family", "comparison", "metric", "aspect", "diff", "lo", "hi", "p_raw", "p_holm", "q_bh"]].copy()
    full["comparison"] = full.comparison.str.replace("HiGO-R2", "Ontologue")
    body = "".join("<tr>" + "".join(f"<td>{v:.4g}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>"
                                    for v in row) + "</tr>" for row in full.itertuples(index=False))
    fullt = ('<details><summary>Show all ' + str(len(full)) + ' tests</summary><table class="etable small"><tr>' +
             "".join(f"<th>{c}</th>" for c in full.columns) + "</tr>" + body + "</table></details>")
    return summary, grid, fullt


def evaluation_body() -> str:
    d = source_dir()
    s = pd.read_csv(d / "summary.tsv", sep="\t")
    s = s[s.kind != "difference"] if "kind" in s else s
    au = pd.read_csv(d / "auprc.tsv", sep="\t")
    sm = pd.read_csv(d / "smin.tsv", sep="\t")
    ib = pd.read_csv(d / "identity_bins.tsv", sep="\t")
    sp = pd.read_csv(d / "spearman.tsv", sep="\t")
    co = pd.read_csv(d / "consistency.tsv", sep="\t")
    rt = pd.read_csv(d / "retrieval.tsv", sep="\t")
    pv = pd.read_csv(d / "pvalues.tsv", sep="\t")
    n_methods = len(set(s.model) - {m for m in s.model if m.startswith("[component]")})
    summary, grid, fullt = tests_html(pv)
    star = '<p class="hint"><b>*</b> These methods output term sets without graded scores (every predicted term scores 1), so their precision-recall curve has a single operating point. Their AUPRC is reported but is not a like-for-like comparison.</p>'
    comp_note = ('<p class="hint">Rows in grey are parts of Ontologue scored on their own (step 1a: homology transfer; '
                 'step 1b: the ESM-2 head), shown so you can see where the accuracy comes from. Bold marks the best '
                 'method in each column (parts excluded). Small numbers are 95% bootstrap intervals.</p>')
    return f"""
<section class="ehero"><div class="eyebrow">Full evaluation</div>
<h1>How we evaluated {html.escape('Ontologue')}</h1>
<p class="lede">Here is everything we measured on the temporal benchmark: 8,630 proteins whose GO annotations were
added after the reference data was collected. All methods were scored with the same code on the same proteins
({n_methods} methods, plus two of Ontologue's own parts scored separately). We grouped the results by the question they answer, so you can jump to the part you care
about.</p>
<div class="toc"><a href="#primary">1. Primary accuracy</a><a href="#novel">2. Novel proteins</a>
<a href="#consistency">3. Ontology consistency</a><a href="#stats">4. Statistical reliability</a></div></section>

<section class="eblock" id="primary"><div class="eyebrow">1. Primary accuracy</div>
<h2>How accurate are the predictions overall?</h2>
<p class="sub">These are the standard CAFA metrics, computed for each ontology and averaged. Fmax is the best
F-measure over all score thresholds. The IA-weighted version gives specific, informative terms more weight than
broad ones, which is closer to what a biologist cares about.</p>
<h3>F-max</h3><p class="what">Best balance of precision and recall over thresholds, every term counted equally.</p>
{metric_table(s, "Fmax")}
<h3>IA-weighted F-max</h3><p class="what">The same, but each term is weighted by its information accretion, so
getting a rare, specific term right counts for more. This is the benchmark's headline metric.</p>
{metric_table(s, "Fmax_w")}
<h3>AUPRC</h3><p class="what">Area under the precision-recall curve, IA-weighted, pooled over all protein-term pairs.
It rewards a good ranking of scores across every threshold, not just the best one.</p>
{metric_table(au, "AUPR_micro_w", flag_binary=True)}{star}
<h3>S-min</h3><p class="what">The smallest combined amount of missed and wrongly predicted information (in bits of
information content). Lower is better.</p>
{metric_table(sm, "Smin_w", lower_better=True)}
{comp_note}</section>

<section class="eblock soft" id="novel"><div class="eyebrow">2. Novel-protein generalisation</div>
<h2>How does accuracy change for proteins unlike anything known?</h2>
<p class="sub">We split the benchmark by how similar each protein is to its closest training protein and report
the average per-protein F1 in each group, at each method's own best threshold. This is the most important view if
you work with proteins that have no close relatives.</p>
<h3>Per-protein F1 by identity</h3>{bins_table(ib, "F1")}
<h3>IA-weighted per-protein F1 by identity</h3>{bins_table(ib, "F1_w")}
<h3>Dependence on homology</h3><p class="what">Spearman correlation between a protein's IA-weighted F1 and its
identity to the nearest training protein. A higher value means accuracy depends more on having a close relative.</p>
{spearman_table(sp)}
<p class="hint">Ontologue is ahead from about 95% identity upward and behind for proteins with no hit or low
identity, where it relies on the ESM-2 head alone. The groups below 30% hold fewer than 100 proteins, so their
intervals are wide.</p></section>

<section class="eblock" id="consistency"><div class="eyebrow">3. Ontology consistency</div>
<h2>Do the predictions respect the structure of GO?</h2>
<h3>GO hierarchy violation rate</h3><p class="what">Share of predicted (child, parent) term pairs where the child scores
higher than its parent, which would mean "it is a GTPase but less likely a hydrolase". Checked on the raw output
files before any evaluator post-processing.</p>
{consistency_table(co)}
<h3>GO relationship retrieval from learned term embeddings</h3><p class="what">For models that learn a vector per GO
term: rank all other terms by similarity and check whether a term's direct parents and children come first.
Precision@k and recall@k over the shared vocabulary. Only Ontologue and GO-GPT have such vectors.</p>
{retrieval_table(rt)}
<p class="hint">Part of Ontologue's score comes from how its term queries are built (each term shares the mean of its
ancestors), which is why the untrained row is already well above random (its spread over 5 random seeds is
below 0.004). Learning adds to it. This table measures how
GO structure is encoded, not prediction accuracy.</p></section>

<section class="eblock soft" id="stats"><div class="eyebrow">4. Statistical reliability</div>
<h2>Are the differences real or noise?</h2>
<p class="sub">Every interval on this page is a 95% bootstrap interval from resampling proteins (1,000 times).
Differences between Ontologue and each other method were tested with a paired permutation test (10,000 times,
swapping the two methods' predictions protein by protein), and the p-values were corrected for the number of tests
with Holm-Bonferroni and Benjamini-Hochberg.</p>
<h3>Summary of all tests</h3>{summary}
<h3>Ontologue minus each method (mean over ontologies)</h3><p class="what">Green: Ontologue significantly better after
Holm correction. Red: significantly worse. Grey: not significant. For S-min a negative difference is better.</p>
{grid}{fullt}
<p class="hint">Source files: <code>{html.escape(str(d.relative_to(ROOT)))}/</code>, produced by
<code>src/cafa6/eval_suite.py</code>.</p></section>"""


EVAL_CSS = """
.ehero{padding:clamp(50px,8vh,90px) 6vw 30px}.ehero h1{font-family:var(--serif);font-size:clamp(34px,4.6vw,58px);
font-weight:600;margin:10px 0 14px;line-height:1.08}.ehero .lede{font-size:18px;color:var(--muted);max-width:70ch}
.toc{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.toc a{background:#fff;border:1px solid var(--line);
border-radius:999px;padding:8px 16px;text-decoration:none;font-size:14px;color:var(--teal-d)}
.eblock{padding:clamp(50px,8vh,90px) 6vw}.eblock h2{font-family:var(--serif);font-size:clamp(26px,3.2vw,40px);
font-weight:600;margin:6px 0 12px}.eblock h3{font-family:var(--serif);font-weight:600;font-size:21px;margin:34px 0 4px}
.eblock .sub{font-size:17px;color:var(--muted);max-width:75ch}.what{color:var(--muted);font-size:14.5px;
margin:0 0 10px;max-width:90ch}
.etable{width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid var(--line);
border-radius:14px;overflow:hidden;font-size:13.5px;margin:6px 0 8px}.etable th,.etable td{padding:9px 12px;
text-align:left;border-bottom:1px solid #efece6;vertical-align:top}.etable th{font-size:12px;color:var(--muted);
background:#faf9f6;font-weight:600}.etable tr:last-child td{border-bottom:0}.etable td.b{font-weight:700;
color:var(--teal-d)}.etable tr.hl td{background:#f0f8f5}.etable tr.comp td{color:#7d8a8e;font-style:italic}
.etable tr.nrow td{color:#7d8a8e;font-size:12px;background:#fcfbf8}.etable .ci{display:block;font-size:11px;
color:#9aa5a8;font-weight:400}.etable td.sig-good{background:#e7f5ef;color:#1f6156;font-weight:600}
.etable td.sig-bad{background:#fdecea;color:#8a2d22;font-weight:600}.etable td.ns{color:#7d8a8e}
.etable.small{font-size:12px}.flag{color:#a2582a;font-weight:700}
"""
