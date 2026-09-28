"""Step 7: interpretability of GO-conditioned evidence.

For confident, specific predictions (p >= 0.5) on held-out test proteins:
- evidence map alpha_t over residues (entmax -> exact zeros), sparsity = share of residues with alpha > 0;
- deletion faithfulness: mask the top-k evidence residues with <mask> vs k random residues, compare drop in p_t;
- InterPro enrichment: attention mass inside InterPro-annotated regions / fraction of residues covered.

Usage: python -m cafa6.explain --run dev_35m_lora [--n_proteins 300] [--n_interpro 60]
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import torch
import yaml
from transformers import AutoTokenizer

from cafa6.data import PROC, ROOT
from cafa6.model import HiGO


def load_run(name, device="cuda"):
    run = ROOT / "runs" / name
    cfg = yaml.safe_load(open(run / "config.yaml"))
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    tmap = {int(k): v for k, v in json.load(open(run / "taxa.json")).items()}
    model = HiGO(cfg["plm"], len(vocab), np.load(PROC / "go_ancestors.npy"), vocab.ia.values.astype(np.float32),
                 n_taxa=len(tmap) + 1, adapter=cfg["adapter"], lora_r=cfg.get("lora_r", 8), d=cfg.get("d", 256),
                 hier_query=cfg.get("hier_query", True), pooling=cfg.get("pooling", "entmax"),
                 use_mcm=cfg.get("use_mcm", True), use_taxon=cfg.get("use_taxon", True),
                 lora_targets=cfg.get("lora_targets", ["query", "value"]))
    model.load_state_dict(torch.load(run / "trainable_weights.pt"), strict=False)
    return model.to(device).eval(), cfg, vocab, tmap, AutoTokenizer.from_pretrained(cfg["plm"])


@torch.no_grad()
def forward_one(model, tok, seq, tax, terms_idx, device="cuda"):
    enc = tok([seq], return_tensors="pt")
    ids, att = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    res = att.clone(); res[:, 0] = 0; res[:, -1] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z, a = model(ids, att, res, torch.tensor([tax], device=device), return_attn=True)
    p = model.predict_proba(z.float())[0]
    own = torch.sigmoid(z.float())[0]
    A = a[0][:, 1:-1].float() if a is not None else None      # T x L (residue positions)
    return own[terms_idx].cpu().numpy(), (A[terms_idx].cpu().numpy() if A is not None else None), p


@torch.no_grad()
def masked_prob(model, tok, seq, tax, positions, t, mode="input", device="cuda"):
    """Own-term probability sigmoid(z_t) (pre-MCM, so it depends only on alpha_t) after deleting `positions`.
    mode='input': replace residues by <mask> in the backbone input (end-to-end, includes contextual leakage);
    mode='pool' : keep the input, drop the residues from the evidence pooling only (faithfulness of the head)."""
    enc = tok([seq], return_tensors="pt")
    ids, att = enc["input_ids"].clone(), enc["attention_mask"]
    res = att.clone(); res[:, 0] = 0; res[:, -1] = 0
    pos = np.asarray(positions) + 1
    if mode == "input":
        ids[0, pos] = tok.mask_token_id
    else:
        res[0, pos] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z, _ = model(ids.to(device), att.to(device), res.to(device), torch.tensor([tax], device=device))
    return float(torch.sigmoid(z.float())[0, t])


LOCAL_TYPES = {"domain", "repeat", "active_site", "binding_site", "conserved_site", "ptm"}


def interpro_regions(acc):
    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{acc}/?page_size=100"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or not r.text:
            return []
        out = []
        for e in r.json().get("results", []):
            for prot in e.get("proteins", []):
                for loc in prot.get("entry_protein_locations", []) or []:
                    for fr in loc.get("fragments", []):
                        out.append((e["metadata"]["accession"], e["metadata"]["type"], int(fr["start"]), int(fr["end"])))
        return out
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n_proteins", type=int, default=300)
    ap.add_argument("--n_interpro", type=int, default=60)
    ap.add_argument("--k_frac", type=float, default=0.1)
    args = ap.parse_args()
    random.seed(0); np.random.seed(0)
    out = ROOT / "runs" / args.run / "explain"
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, vocab, tmap, tok = load_run(args.run)
    df = pd.read_parquet(PROC / "train.parquet")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    te = np.where((df.split == "test") & (df.length <= cfg["max_len"]) & (df.length >= 50))[0]
    te = np.random.default_rng(0).choice(te, min(args.n_proteins, len(te)), replace=False)
    ia = vocab.ia.values
    rows = []
    for j in te:
        seq, tax = df.seq.iloc[j], tmap.get(int(df.taxon.iloc[j]), 0)
        y = Y[j].toarray()[0]
        own, A, _ = forward_one(model, tok, seq, tax, np.arange(len(vocab)))
        # most specific confident correct prediction per aspect (highest IA among own p>=0.5 & true)
        for asp in ("F", "P", "C"):
            cand = np.where((vocab.aspect.values == asp) & (own >= 0.5) & (y > 0))[0]
            if len(cand) == 0:
                continue
            t = int(cand[np.argmax(ia[cand])])
            p_t, a = float(own[t]), A[t]
            L = len(seq)
            k = max(1, int(args.k_frac * L))
            top = np.argsort(-a)[:k]
            rnd = np.random.default_rng(j).choice(L, k, replace=False)
            row = {"id": df.id.iloc[j], "term": vocab.term.iloc[t], "aspect": asp, "ia": float(ia[t]),
                   "L": L, "p": p_t, "sparsity_nonzero_frac": float((a > 1e-6).mean()),
                   "attn_mass_top": float(a[top].sum() / max(a.sum(), 1e-9))}
            for mode in ("input", "pool"):
                row[f"drop_top_{mode}"] = p_t - masked_prob(model, tok, seq, tax, top, t, mode)
                row[f"drop_random_{mode}"] = p_t - masked_prob(model, tok, seq, tax, rnd, t, mode)
            row["attn"] = a.astype(np.float16).tolist()
            rows.append(row)
    res = pd.DataFrame(rows)
    res.drop(columns=["attn"]).to_csv(out / "faithfulness.tsv", sep="\t", index=False)
    from scipy.stats import wilcoxon
    summary = {"n_cases": len(res), "k_frac": args.k_frac,
               "mean_nonzero_frac": float(res.sparsity_nonzero_frac.mean()),
               "mean_attn_mass_top": float(res.attn_mass_top.mean())}
    cols = ["sparsity_nonzero_frac", "attn_mass_top"]
    for mode in ("input", "pool"):
        dt, dr = res[f"drop_top_{mode}"], res[f"drop_random_{mode}"]
        summary[mode] = {"mean_drop_top": float(dt.mean()), "mean_drop_random": float(dr.mean()),
                         "frac_top_gt_random": float((dt > dr).mean()),
                         "wilcoxon_p": float(wilcoxon(dt, dr, alternative="greater").pvalue)}
        cols += [f"drop_top_{mode}", f"drop_random_{mode}"]
    summary["by_aspect"] = res.groupby("aspect")[cols].mean().round(4).to_dict()

    # InterPro enrichment on a subset (rate-limited public API)
    enr = []
    for _, r in res.drop_duplicates("id").head(args.n_interpro).iterrows():
        regs = interpro_regions(r["id"])
        time.sleep(0.3)
        if not regs:
            continue
        cover = np.zeros(r["L"], bool)
        for _, typ, s, e in regs:
            if typ in LOCAL_TYPES:
                cover[max(s - 1, 0):min(e, r["L"])] = True
        if not cover.any() or cover.mean() > 0.8:
            continue
        for _, rr in res[res.id == r["id"]].iterrows():
            a = np.asarray(rr["attn"], dtype=np.float32)
            a = a / max(a.sum(), 1e-9)
            enr.append({"id": rr["id"], "term": rr["term"], "aspect": rr["aspect"], "coverage": float(cover.mean()),
                        "attn_mass_in_interpro": float(a[cover].sum())})
    enr = pd.DataFrame(enr)
    if len(enr):
        enr["enrichment"] = enr.attn_mass_in_interpro / enr.coverage.clip(lower=1e-6)
        enr.to_csv(out / "interpro_overlap.tsv", sep="\t", index=False)
        summary["interpro"] = {"n_cases": len(enr), "mean_coverage": float(enr.coverage.mean()),
                               "mean_attn_mass_in": float(enr.attn_mass_in_interpro.mean()),
                               "median_enrichment": float(enr.enrichment.median()),
                               "frac_enrichment_gt1": float((enr.enrichment > 1).mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # figure: evidence maps of the same protein under different GO terms (function-conditioned evidence)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    multi = res.groupby("id").filter(lambda g: len(g) >= 2)
    pid = multi.id.iloc[0] if len(multi) else res.id.iloc[0]
    sub = res[res.id == pid]
    fig, axes = plt.subplots(len(sub), 1, figsize=(12, 1.6 * len(sub)), sharex=True)
    axes = np.atleast_1d(axes)
    regs = interpro_regions(pid)
    for ax, (_, rr) in zip(axes, sub.iterrows()):
        ax.bar(np.arange(rr["L"]) + 1, np.asarray(rr["attn"], dtype=np.float32), width=1.0, color="tab:blue")
        for _, typ, s, e in regs:
            if typ in LOCAL_TYPES:
                ax.axvspan(s, e, color="tab:orange", alpha=0.12)
        ax.set_ylabel(rr["aspect"])
        ax.set_title(f"{pid}  {rr['term']}  p={rr['p']:.2f}  IA={rr['ia']:.2f}  "
                     f"drop(top10%, pool)={rr['drop_top_pool']:.2f}", fontsize=8)
    axes[-1].set_xlabel("residue (orange = local InterPro regions: domains/sites/repeats)")
    plt.tight_layout()
    plt.savefig(out / f"evidence_{pid}.png", dpi=130)
    (ROOT / "logs" / f"explain_{args.run}.done").touch()


if __name__ == "__main__":
    main()
