"""Pilot: a sparse autoencoder (SAE) on HiGO's LoRA-adapted residue activations.

Motivation (see REPORT.md section 4.3 and `explain.py`): HiGO's evidence maps are faithful to the pooling
head, but masking the top-evidence residues in the input has no effect, presumably because the transformer
spreads that information to neighbouring residues. InterPLM (bioRxiv 2024) and later work train SAEs on a
PLM's own activations to split superposed directions into more interpretable latents. This script trains
one on H (the `H = self.proj(self.plm(...).last_hidden_state)` tensor from `HiGO.forward`, i.e. after the
LoRA adapter) and asks whether single latents concentrate on specific GO terms and on contiguous residue
stretches. It is a correlational probe and does not replace the masking test.

Pipeline:
1. Reconstruct HiGO with the *exact* hyperparameters used for `runs/dev_35m_lora` (config.yaml + taxa.json,
   the same construction as `train.py`/`explain.py`) and load `trainable_weights.pt`.
2. Extract H at residue positions (via `res_mask`, as in `HiGO.forward`) for a sample of VALIDATION
   proteins, batched via `train.Batcher` (same tokenization/batching convention as training). Kept only in
   RAM (a few hundred MB-1.5GB for ~3000 proteins), never written to disk in full.
3. Train a small SAE (256 -> 4096 -> 256, ReLU + L1) on the pooled residue activations.
4. Validate interpretability: per-latent GO-term purity (enrichment vs. background) and per-latent
   positional contiguity of firing residues. Writes `latent_go_purity.tsv`, `latent_contiguity.tsv`,
   `sae_weights.pt`, and `SUMMARY.md` under `runs/pilot_sae/`.

Usage: python -m cafa6.sae [--n_proteins 3000] [--d_hidden 4096] [--l1 3e-3] [--steps 4000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from cafa6.data import PROC, ROOT
from cafa6.model import HiGO
from cafa6.train import Batcher

RUN = "dev_35m_lora"


# ----------------------------------------------------------------------------- setup
def pick_device(min_free_gb: float = 4.0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_i, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_i = free, i
    if best_free / 1e9 >= min_free_gb:
        return torch.device(f"cuda:{best_i}")
    return torch.device("cpu")


def load_higo(device):
    run = ROOT / "runs" / RUN
    cfg = yaml.safe_load(open(run / "config.yaml"))
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    tmap = {int(k): v for k, v in json.load(open(run / "taxa.json")).items()}
    model = HiGO(cfg["plm"], len(vocab), np.load(PROC / "go_ancestors.npy"), vocab.ia.values.astype(np.float32),
                 n_taxa=len(tmap) + 1, adapter=cfg["adapter"], lora_r=cfg.get("lora_r", 8), d=cfg.get("d", 256),
                 hier_query=cfg.get("hier_query", True), pooling=cfg.get("pooling", "entmax"),
                 use_mcm=cfg.get("use_mcm", True), use_taxon=cfg.get("use_taxon", True),
                 lora_targets=cfg.get("lora_targets", ["query", "value"]),
                 go_edges=np.load(PROC / "go_edges.npy") if cfg.get("use_go_gat") else None,
                 use_go_gat=cfg.get("use_go_gat", False), gat_layers=cfg.get("gat_layers", 2),
                 gat_heads=cfg.get("gat_heads", 4), n_slots=cfg.get("n_slots", 8))
    state = torch.load(run / "trainable_weights.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint: {len(state)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(cfg["plm"])
    return model, cfg, vocab, tmap, tok


# ----------------------------------------------------------------------------- activation extraction
@torch.no_grad()
def extract_activations(model, batcher, seqs, taxa, sel_rows, device):
    """Runs H = proj(plm(...).last_hidden_state) (exactly HiGO.forward's H) over `sel_rows`, batched, and
    returns residue-level activations for residue positions only (res_mask), plus, for each residue,
    which local protein index (0..len(sel_rows)-1, in `sel_rows` order) and position within that protein
    it belongs to. Kept in RAM only; never persisted to disk in full."""
    row_to_local = {int(j): i for i, j in enumerate(sel_rows)}
    batches = batcher.batches(list(sel_rows), shuffle=False)
    X_chunks, pid_chunks, pos_chunks = [], [], []
    bf16 = device.type == "cuda"
    n_done = 0
    for bi, b in enumerate(batches):
        pieces = [batcher.crop(seqs[j]) for j in b]
        ids, att, res = batcher.encode(pieces)
        tx = torch.as_tensor([taxa[j] for j in b], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            H = model.plm(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            H = model.proj(H.float())                       # B x L x d, exactly HiGO.forward's H
        mask = res.bool()
        for r, j in enumerate(b):
            m = mask[r]
            h = H[r][m].detach().to("cpu", torch.float32)   # Lj x d
            Lj = h.shape[0]
            X_chunks.append(h)
            pid_chunks.append(torch.full((Lj,), row_to_local[int(j)], dtype=torch.int32))
            pos_chunks.append(torch.arange(Lj, dtype=torch.int32))
        n_done += len(b)
        if bi % 20 == 0:
            print(f"  extract batch {bi}/{len(batches)}  proteins {n_done}/{len(sel_rows)}", flush=True)
    X = torch.cat(X_chunks, 0)
    prot_idx = torch.cat(pid_chunks, 0).numpy()
    pos = torch.cat(pos_chunks, 0).numpy()
    return X, prot_idx, pos


# ----------------------------------------------------------------------------- SAE
class SAE(nn.Module):
    """Standard InterPLM/Anthropic-style SAE: pre-encoder bias subtraction, tied-init decoder,
    unit-norm decoder columns (re-applied after every optimizer step) so L1 cannot be gamed by
    shrinking latents while inflating decoder norm."""

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) / d_in ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(self.W_enc.detach().t().clone())
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, latent):
        return latent @ self.W_dec + self.b_dec

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent), latent


def train_sae(X: torch.Tensor, d_hidden: int, l1_coef: float, steps: int, batch_size: int, device):
    sae = SAE(X.shape[1], d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    n = X.shape[0]
    log = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        xb = X[idx].to(device)
        recon, latent = sae(xb)
        mse = F.mse_loss(recon, xb)
        l1 = latent.abs().mean()
        loss = mse + l1_coef * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % 250 == 0 or step == steps - 1:
            frac_active = (latent > 0).float().mean().item()
            rec = {"step": step, "mse": mse.item(), "l1": l1.item(), "frac_active": frac_active}
            log.append(rec)
            print(f"  sae step {step}: mse={mse.item():.4f} l1={l1.item():.4f} frac_active={frac_active:.4f}",
                  flush=True)
    with torch.no_grad():
        xb = X[torch.randint(0, n, (min(n, 65536),))].to(device)
        log.append({"var_explained": float(1 - F.mse_loss(sae(xb)[0], xb) / xb.var(0).mean())})
    return sae, log


@torch.no_grad()
def latent_firing_rates(sae: SAE, X: torch.Tensor, device, chunk: int = 8192) -> np.ndarray:
    n, d_hidden = X.shape[0], sae.W_enc.shape[1]
    counts = torch.zeros(d_hidden)
    for i in range(0, n, chunk):
        xb = X[i:i + chunk].to(device)
        latent = sae.encode(xb)
        counts += (latent > 0).float().sum(0).cpu()
    return (counts / n).numpy()


@torch.no_grad()
def latent_subset_activations(sae: SAE, X: torch.Tensor, selected: np.ndarray, device, chunk: int = 8192):
    n = X.shape[0]
    out = np.zeros((n, len(selected)), dtype=np.float16)
    sel_t = torch.as_tensor(selected, dtype=torch.long)
    for i in range(0, n, chunk):
        xb = X[i:i + chunk].to(device)
        latent = sae.encode(xb)[:, sel_t.to(device)]
        out[i:i + chunk] = latent.detach().cpu().numpy().astype(np.float16)
    return out


# ----------------------------------------------------------------------------- interpretability analyses
def go_purity(latent_sub: np.ndarray, prot_idx: np.ndarray, n_proteins: int, selected: np.ndarray,
              Y_sub: np.ndarray, vocab: pd.DataFrame, min_support: int = 30, n_perm: int = 50,
              seed: int = 0) -> pd.DataFrame:
    """Per selected latent: does-it-fire-anywhere-in-protein indicator, then most-enriched GO term
    (P(term | fires) / P(term)) among terms with >= min_support positive proteins in this sample.
    The best of several hundred terms is inflated even for a random protein set, so each latent also
    gets a null: the best-term enrichment of n_perm random protein sets of the same size."""
    rng = np.random.default_rng(seed)
    protein_max = np.zeros((n_proteins, len(selected)), dtype=np.float32)
    np.maximum.at(protein_max, prot_idx, latent_sub.astype(np.float32))
    fires = protein_max > 0                                    # n_proteins x n_selected

    term_support = Y_sub.sum(0)
    valid_terms = np.where(term_support >= min_support)[0]
    baseline_p = term_support[valid_terms] / n_proteins

    rows = []
    for li, latent_id in enumerate(selected):
        f = fires[:, li]
        n_fire = int(f.sum())
        if n_fire < 5:
            rows.append({"latent": int(latent_id), "n_fire_proteins": n_fire, "best_term": None,
                         "aspect": None, "enrichment": 0.0, "support": 0, "p_given_fire": 0.0,
                         "p_baseline": 0.0, "null_mean": np.nan, "null_p95": np.nan, "perm_p": np.nan})
            continue
        Yv = Y_sub[:, valid_terms]
        p_given_fire = Yv[f].mean(0)
        enrichment = p_given_fire / np.clip(baseline_p, 1e-9, None)
        best = int(np.argmax(enrichment))
        t = valid_terms[best]
        null = np.array([(Yv[rng.choice(n_proteins, n_fire, replace=False)].mean(0) /
                          np.clip(baseline_p, 1e-9, None)).max() for _ in range(n_perm)])
        rows.append({"latent": int(latent_id), "n_fire_proteins": n_fire,
                     "best_term": vocab.term.iloc[t], "aspect": vocab.aspect.iloc[t],
                     "enrichment": float(enrichment[best]), "support": int(term_support[t]),
                     "p_given_fire": float(p_given_fire[best]), "p_baseline": float(baseline_p[best]),
                     "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
                     "perm_p": float((1 + (null >= enrichment[best]).sum()) / (1 + n_perm))})
    df = pd.DataFrame(rows).sort_values("enrichment", ascending=False).reset_index(drop=True)
    return df


def contiguity(latent_sub: np.ndarray, prot_idx: np.ndarray, pos: np.ndarray, selected: np.ndarray,
               n_proteins: int, max_proteins_per_latent: int = 25, seed: int = 0) -> pd.DataFrame:
    """Per selected latent: for a sample of proteins where it fires, is the firing pattern a contiguous
    stretch (domain-like) or scattered? Reports mean neighbor-fraction (share of fired residues with a
    fired neighbor) and mean relative run-length (mean consecutive-run length / #fired residues)."""
    boundaries = np.nonzero(np.diff(prot_idx))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(prot_idx)]))
    block_of_pid = {int(prot_idx[s]): (s, e) for s, e in zip(starts, ends)}

    rng = np.random.default_rng(seed)
    rows = []
    for li, latent_id in enumerate(selected):
        col = latent_sub[:, li]
        # which proteins fire this latent
        fire_pids = np.unique(prot_idx[col > 0])
        if len(fire_pids) == 0:
            rows.append({"latent": int(latent_id), "n_proteins_sampled": 0, "mean_fired_residues": 0.0,
                        "mean_neighbor_frac": 0.0, "mean_rel_runlen": 0.0})
            continue
        samp = rng.choice(fire_pids, min(max_proteins_per_latent, len(fire_pids)), replace=False)
        nbr_fracs, rel_runs, n_fired_list = [], [], []
        for p in samp:
            s, e = block_of_pid[int(p)]
            acts = col[s:e]
            idxs = np.nonzero(acts > 0)[0]
            if len(idxs) == 0:
                continue
            idx_set = set(idxs.tolist())
            nbr = np.mean([(i - 1 in idx_set) or (i + 1 in idx_set) for i in idxs])
            runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
            mean_run = np.mean([len(r) for r in runs])
            nbr_fracs.append(nbr); rel_runs.append(mean_run / len(idxs)); n_fired_list.append(len(idxs))
        rows.append({"latent": int(latent_id), "n_proteins_sampled": len(nbr_fracs),
                     "mean_fired_residues": float(np.mean(n_fired_list)) if n_fired_list else 0.0,
                     "mean_neighbor_frac": float(np.mean(nbr_fracs)) if nbr_fracs else 0.0,
                     "mean_rel_runlen": float(np.mean(rel_runs)) if rel_runs else 0.0})
    df = pd.DataFrame(rows).sort_values("mean_neighbor_frac", ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_proteins", type=int, default=3000)
    ap.add_argument("--d_hidden", type=int, default=4096)
    ap.add_argument("--l1", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--n_latents_analyze", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / "runs" / "pilot_sae"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}")

    model, cfg, vocab, tmap, tok = load_higo(device)

    df = pd.read_parquet(PROC / "train.parquet")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    seqs = df.seq.tolist()
    taxa = df.taxon.map(lambda t: tmap.get(t, 0)).values

    val_pool = np.where((df.split == "val") & (df.length <= cfg["max_len"]) & (df.length >= 20))[0]
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(val_pool, min(args.n_proteins, len(val_pool)), replace=False)
    sel = np.asarray(sorted(sel.tolist()))
    print(f"val pool (len<=max_len) = {len(val_pool)}, sampled = {len(sel)}")

    batcher = Batcher(seqs, tok, cfg["max_len"], cfg.get("tokens_per_batch", 16384), cfg.get("max_batch", 32))

    print("extracting LoRA-adapted residue activations H = proj(plm(...))...")
    X, prot_idx, pos = extract_activations(model, batcher, seqs, taxa, sel, device)
    n_proteins = len(sel)
    print(f"activations: X={tuple(X.shape)} ({X.element_size() * X.nelement() / 1e6:.1f} MB, RAM only), "
          f"residues/protein mean={np.bincount(prot_idx).mean():.1f}")

    print(f"training SAE: d_in={X.shape[1]} d_hidden={args.d_hidden} l1={args.l1} steps={args.steps}")
    sae, sae_log = train_sae(X, args.d_hidden, args.l1, args.steps, args.batch_size, device)

    firing_rate = latent_firing_rates(sae, X, device)
    lo, hi = 0.005, 0.20
    dist = np.where((firing_rate >= lo) & (firing_rate <= hi), 0.0,
                     np.minimum(np.abs(firing_rate - lo), np.abs(firing_rate - hi)))
    n_dead = int((firing_rate == 0).sum())
    n_healthy = int(((firing_rate >= lo) & (firing_rate <= hi)).sum())
    print(f"latent firing rates: {n_dead}/{args.d_hidden} dead, {n_healthy} in healthy range [{lo},{hi}]")
    selected = np.argsort(dist)[:args.n_latents_analyze]

    torch.save({"state_dict": sae.state_dict(), "d_in": X.shape[1], "d_hidden": args.d_hidden,
               "l1_coef": args.l1, "steps": args.steps, "seed": args.seed, "n_proteins": n_proteins,
               "firing_rate": firing_rate, "selected_latents": selected, "sae_train_log": sae_log,
               "source_run": RUN}, out / "sae_weights.pt")

    print(f"analyzing {len(selected)} latents (closest to healthy firing-rate range)...")
    latent_sub = latent_subset_activations(sae, X, selected, device)

    Y_sub = np.asarray(Y[sel].todense())
    purity = go_purity(latent_sub, prot_idx, n_proteins, selected, Y_sub, vocab)
    purity.to_csv(out / "latent_go_purity.tsv", sep="\t", index=False)

    contig = contiguity(latent_sub, prot_idx, pos, selected, n_proteins)
    contig.to_csv(out / "latent_contiguity.tsv", sep="\t", index=False)

    merged = purity.merge(contig, on="latent", how="left")
    merged = merged.sort_values("enrichment", ascending=False)
    top5 = merged.head(5)

    scored = purity[purity.best_term.notna()]
    mean_enrich_all = scored.enrichment.mean()
    mean_null = scored.null_mean.mean()
    frac_above_null = float((scored.enrichment > scored.null_p95).mean())
    frac_gt5 = float((scored.enrichment > 5).mean())
    n_terms_distinct = scored.best_term.nunique()
    top_terms = scored.best_term.value_counts().head(3)
    var_expl = sae_log[-1]["var_explained"]

    lines = []
    lines.append("# Pilot SAE interpretability probe (runs/pilot_sae/)\n")
    lines.append(f"Source checkpoint: `runs/dev_35m_lora` (HiGO-35M, LoRA r=8, d=256). "
                f"Activations: `H = model.proj(model.plm(...).last_hidden_state)` "
                f"(the LoRA-adapted, projected residue representation from `HiGO.forward`), "
                f"residue positions only (`res_mask`), from {n_proteins} validation proteins "
                f"({X.shape[0]} residues total, kept in RAM only).\n")
    lines.append(f"SAE: {X.shape[1]} -> {args.d_hidden} -> {X.shape[1]}, ReLU + L1 ({args.l1}), "
                f"{args.steps} Adam steps, unit-norm decoder columns, variance explained {var_expl:.2f}. "
                f"{n_dead}/{args.d_hidden} latents never fire, {n_healthy} fire on "
                f"{lo:.1%}-{hi:.1%} of residues; the {len(selected)} latents closest to that range are "
                f"analysed.\n")
    lines.append("## Result\n")
    lines.append(
        f"Best-term enrichment (P(term | latent fires in protein) / P(term), terms with >=30 proteins): "
        f"mean {mean_enrich_all:.2f}x, against {mean_null:.2f}x for random protein sets of the same size "
        f"(label permutation). {frac_above_null:.0%} of latents exceed the 95th percentile of their null; "
        f"{frac_gt5:.0%} are above 5x. The {len(scored)} latents map to only {n_terms_distinct} distinct "
        f"terms (most common: " + ", ".join(f"{t} x{c}" for t, c in top_terms.items()) + "), so many "
        "latents are redundant detectors of a few large protein families.\n")
    lines.append(
        "This is protein-level co-occurrence. It shows that some directions in the adapted representation "
        "track specific functions; it does not show that particular residues cause a prediction, which is "
        "what the masking test in `explain.py` measured, so the two results are not directly comparable.\n")
    lines.append("## Top 5 latents by GO-term enrichment\n")
    lines.append("| latent | best GO term | aspect | enrichment | support (proteins) | "
                 "P(term\\|fires) | P(term) | n_fire_proteins | neighbor_frac | rel_runlen |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in top5.iterrows():
        lines.append(f"| {int(r.latent)} | {r.best_term} | {r.aspect} | {r.enrichment:.2f}x | "
                     f"{int(r.support)} | {r.p_given_fire:.3f} | {r.p_baseline:.3f} | "
                     f"{int(r.n_fire_proteins)} | {r.mean_neighbor_frac:.2f} | {r.mean_rel_runlen:.2f} |")
    lines.append("")
    lines.append("## Contiguity (top 5 latents by GO enrichment, positional pattern)\n")
    lines.append(
        "`neighbor_frac` = fraction of fired residues (within a protein) that have an adjacent fired "
        "residue; `rel_runlen` = mean consecutive-run length / number of fired residues in that protein. "
        "Both close to 1 => one contiguous block (domain-like); both low => scattered single-residue hits.\n")
    mean_fired_top5 = merged.head(5)["mean_fired_residues"].mean() if "mean_fired_residues" in merged else float("nan")
    if pd.notna(mean_fired_top5) and mean_fired_top5 < 5:
        lines.append(
            f"The top latents fire on very few residues per protein (mean {mean_fired_top5:.1f} among the "
            "top 5) and rarely on adjacent residues, so they look like point features (a site or short "
            "motif) rather than whole domains. With this few residues the contiguity numbers are noisy.\n")
    lines.append(
        f"Caveats: pilot scale ({n_proteins} val proteins, {args.steps} SAE steps, {args.d_hidden} "
        "latents); latents are selected and scored on the same sample; GO labels are propagated, so a "
        "latent's best term can be a broad ancestor; the representation was trained to predict these GO "
        "labels, so some enrichment is expected by construction; contiguity is a coarse proxy, not a "
        "domain-boundary check against InterPro.\n")
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote: {out / 'sae_weights.pt'}, {out / 'latent_go_purity.tsv'}, "
          f"{out / 'latent_contiguity.tsv'}, {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
