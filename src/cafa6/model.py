"""HiGO: hierarchy-aware, GO-conditioned evidence pooling on a parameter-efficiently tuned ESM2.

Components
- Backbone: ESM2, frozen, with LoRA or DoRA adapters on attention projections (or fully frozen).
- Hierarchical GO tokenization: q_t = e_t + mean_{a in anc(t)} e_a, so a term's query shares
  parameters with every ancestor; rare deep terms borrow statistical strength from their parents.
- Evidence pooling: per-term sparse attention (entmax-1.5) over residues -> v_t. The attention row
  alpha_t is the term's evidence map (exactly zero on most residues, so it is directly readable).
- Hierarchy constraint: C-HMCNN max-constraint module (MCM): p_hat(t) = max_{d in desc(t) U {t}} p(d),
  which guarantees p_hat(parent) >= p_hat(child) (true-path rule), trained with MCLoss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect
from transformers import AutoModel


def build_backbone(name: str, adapter: str = "lora", r: int = 8, alpha: int = 16, dropout: float = 0.05,
                   targets=("query", "value")):
    plm = AutoModel.from_pretrained(name, add_pooling_layer=False)
    for p in plm.parameters():
        p.requires_grad = False
    if adapter in ("lora", "dora"):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                         bias="none", use_dora=(adapter == "dora"))
        plm = get_peft_model(plm, cfg)
    return plm


class HiGO(nn.Module):
    def __init__(self, plm_name: str, n_terms: int, anc_pairs: np.ndarray, ia: np.ndarray, n_taxa: int,
                 adapter: str = "lora", lora_r: int = 8, d: int = 256, hier_query: bool = True,
                 pooling: str = "entmax", use_mcm: bool = True, use_taxon: bool = True, entmax_alpha: float = 1.5,
                 lora_targets=("query", "value"), go_edges: np.ndarray | None = None, use_go_gat: bool = False,
                 gat_layers: int = 2, gat_heads: int = 4, n_slots: int = 8):
        super().__init__()
        self.plm = build_backbone(plm_name, adapter, r=lora_r, alpha=2 * lora_r, targets=tuple(lora_targets))
        h = self.plm.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, d), nn.LayerNorm(d))
        self.n_terms, self.d = n_terms, d
        self.term_emb = nn.Parameter(torch.randn(n_terms, d) * 0.02)
        self.term_bias = nn.Parameter(torch.zeros(n_terms))
        self.key = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)
        self.hier_query, self.pooling, self.use_mcm, self.use_taxon = hier_query, pooling, use_mcm, use_taxon
        self.entmax_alpha = entmax_alpha
        self.taxon_emb = nn.Embedding(n_taxa, d) if use_taxon else None
        # --- pilot: GO-DAG graph conditioning (opt-in, additive refinement of the ancestor-mean query) ---
        self.use_go_gat = use_go_gat
        if use_go_gat:
            from cafa6.go_graph import GOGraphEncoder
            assert go_edges is not None, "use_go_gat=True requires go_edges"
            self.go_gat = GOGraphEncoder(n_terms, d, go_edges, n_layers=gat_layers, heads=gat_heads)
        # --- pilot: slot-based domain-aware pooling (opt-in, selected via pooling="slots") ---
        self.n_slots = n_slots
        if pooling == "slots":
            self.slot_query = nn.Parameter(torch.randn(n_slots, d) * 0.02)
            self.slot_key = nn.Linear(d, d, bias=False)
        # ancestor-mean operator A (T x T, row-normalised) as sparse buffer
        c, a = anc_pairs[:, 0], anc_pairs[:, 1]
        deg = np.bincount(c, minlength=n_terms).astype(np.float32)
        vals = 1.0 / np.maximum(deg[c], 1)
        self.register_buffer("A", torch.sparse_coo_tensor(np.stack([c, a]), vals, (n_terms, n_terms)).coalesce())
        # descendant gather for MCM: for each term t, include itself and all descendants
        self_pairs = np.stack([np.arange(n_terms), np.arange(n_terms)], 1)
        pairs = np.concatenate([self_pairs, anc_pairs], 0)  # (desc, anc): anc takes max over desc
        self.register_buffer("mcm_src", torch.as_tensor(pairs[:, 0], dtype=torch.long))
        self.register_buffer("mcm_dst", torch.as_tensor(pairs[:, 1], dtype=torch.long))
        self.register_buffer("ia", torch.as_tensor(ia, dtype=torch.float32))

    def queries(self):
        q = self.term_emb
        if self.hier_query:
            with torch.autocast("cuda", enabled=False):
                q = q + torch.sparse.mm(self.A, self.term_emb.float())
        if self.use_go_gat:
            q = self.go_gat(q)
        return q

    def mcm(self, p: torch.Tensor) -> torch.Tensor:
        """max over descendants (incl. self) -> hierarchy-consistent scores."""
        src = p[:, self.mcm_src]
        out = torch.full_like(p, float("-inf"))
        return out.scatter_reduce(1, self.mcm_dst.expand(p.size(0), -1), src, reduce="amax", include_self=True)

    def forward(self, input_ids, attention_mask, res_mask, taxon=None, return_attn: bool = False,
                term_chunk: int = 2048):
        """res_mask marks sequence residues (no <cls>/<eos>/<pad>), so padding never receives evidence."""
        H = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        H = self.proj(H.float())                                   # B x L x d
        mask = res_mask.bool()
        K = self.key(H)                                           # B x L x d
        q = self.queries()                                        # T x d
        ctx = self.taxon_emb(taxon) if (self.use_taxon and taxon is not None) else None
        scale = self.d ** -0.5
        if self.pooling == "slots":
            # BioBlobs-inspired: learn n_slots residue-region summaries (unsupervised, from the same
            # masked entmax-1.5 mechanism as per-term pooling), then let term queries attend over the
            # compact slots instead of raw residue positions.
            sc_s = torch.einsum("nd,bld->bnl", self.slot_query, K) * scale
            sc_s = sc_s.masked_fill(~mask[:, None, :], -1e4)
            a_s = entmax_bisect(sc_s, alpha=self.entmax_alpha, dim=-1, n_iter=30)
            S = torch.einsum("bnl,bld->bnd", a_s, H)              # B x n_slots x d
            Kt = self.slot_key(S)                                 # B x n_slots x d
        logits, attns = [], []
        for s in range(0, self.n_terms, term_chunk):
            qc = q[s:s + term_chunk]
            if self.pooling == "mean":
                m = mask.float()
                v = (H * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)   # B x d
                v = v[:, None, :].expand(-1, qc.size(0), -1)
                a = None
            elif self.pooling == "slots":
                sc = torch.einsum("td,bnd->btn", qc, Kt) * scale
                a = entmax_bisect(sc, alpha=self.entmax_alpha, dim=-1, n_iter=30)
                v = torch.einsum("btn,bnd->btd", a, S)            # B x t x d
            else:
                sc = torch.einsum("td,bld->btl", qc, K) * scale
                sc = sc.masked_fill(~mask[:, None, :], -1e4)
                a = entmax_bisect(sc, alpha=self.entmax_alpha, dim=-1, n_iter=30) if self.pooling == "entmax" \
                    else torch.softmax(sc, -1)
                v = torch.einsum("btl,bld->btd", a, H)            # B x t x d
            v = self.out(v)
            if ctx is not None:
                v = v + ctx[:, None, :]
            logits.append((F.gelu(v) * qc[None]).sum(-1) + self.term_bias[s:s + term_chunk])
            if return_attn:
                attns.append(a)
        z = torch.cat(logits, 1)
        return (z, torch.cat(attns, 1) if return_attn and attns[0] is not None else None)

    def predict_proba(self, z):
        p = torch.sigmoid(z)
        return self.mcm(p) if self.use_mcm else p

    def loss(self, z, y, ia_weight: bool = True, ia_clip: float = 5.0):
        """MCLoss (C-HMCNN) with optional IA-based per-term weights, averaged per protein."""
        p = torch.sigmoid(z)
        if self.use_mcm:
            pos = self.mcm(y * p)
            neg = self.mcm(p)
            out = (1 - y) * neg + y * pos
        else:
            out = p
        out = out.clamp(1e-6, 1 - 1e-6)
        bce = -(y * torch.log(out) + (1 - y) * torch.log(1 - out))
        if ia_weight:
            w = 1.0 + self.ia.clamp(max=ia_clip)
            bce = bce * (w / w.mean())
        return bce.mean()
