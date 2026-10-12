"""HiGO: hierarchy-aware, GO-conditioned evidence pooling on a parameter-efficiently tuned ESM2.

Components
- Backbone: ESM2, frozen, with LoRA or DoRA adapters on attention projections (or fully frozen).
- Hierarchical GO tokenization: q_t = e_t + mean_{a in anc(t)} e_a, so a term's query shares
  parameters with every ancestor; rare deep terms borrow statistical strength from their parents.
- Evidence pooling: per-term sparse attention (entmax-1.5) over residues -> v_t. The attention row
  alpha_t is the term's evidence map (exactly zero on most residues, so it is directly readable).
- Hierarchy constraint: C-HMCNN max-constraint module (MCM): p_hat(t) = max_{d in desc(t) U {t}} p(d),
  which guarantees p_hat(parent) >= p_hat(child) (true-path rule), trained with MCLoss.

HiGO-R additions (all off by default, so existing runs build and score exactly as before)
- Scalar-mix readout (`readout="mix"`): H = LN(sum_l softmax(w)_l LN_l(h_l)) over the layers in `readout_layers`,
  optionally on an encoder truncated after `plm_layers` blocks (ESM2-3B at 30, as GO-GPT reads it).
- Retrieval-conditioned logits (`knn_fusion`): z_t = z_t^HiGO + lambda_t logit(clip(p_knn_t)) + g_t . phi_t with
  lambda_t = softplus(a_aspect + delta_t + beta . psi(max identity)); proteins without a hit keep z^HiGO.
- CRLA (`crla`): one conditioned residual latent adapter at the readout, gated by tanh(alpha) with alpha = 0, so the
  block is an exact identity at initialisation.
A frozen encoder (no trainable backbone parameter) always runs under no_grad.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect
from transformers import AutoModel


ASPECT_INDEX = {"F": 0, "P": 1, "C": 2}
KNN_FEATURES = ("s_star", "frac", "count", "max_id", "has_hit", "ia", "depth")


def build_backbone(name: str, adapter: str = "lora", r: int = 8, alpha: int = 16, dropout: float = 0.05,
                   targets=("query", "value"), n_layers: int | None = None):
    plm = AutoModel.from_pretrained(name, add_pooling_layer=False)
    if n_layers is not None and n_layers < plm.config.num_hidden_layers:
        plm.encoder.layer = nn.ModuleList(list(plm.encoder.layer)[:n_layers])
        plm.config.num_hidden_layers = n_layers
    for p in plm.parameters():
        p.requires_grad = False
    if adapter in ("lora", "dora"):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                         bias="none", use_dora=(adapter == "dora"))
        plm = get_peft_model(plm, cfg)
    return plm


def esm_core(plm):
    return plm.get_base_model() if hasattr(plm, "get_base_model") else plm


class ScalarMix(nn.Module):
    """LN(sum_l softmax(w)_l LN_l(h_l)). The per-layer LayerNorms carry no affine parameters; they are needed because
    Hugging Face ESM2 normalises only the final output, so intermediate states are the raw residual stream."""

    def __init__(self, n_layers: int, hidden: int, top_init: float = 5.0):
        super().__init__()
        w = torch.zeros(n_layers)
        w[-1] = top_init
        self.w = nn.Parameter(w)
        self.out_ln = nn.LayerNorm(hidden)

    def forward(self, hs):
        a = torch.softmax(self.w.float(), 0)
        x = sum(a[i] * F.layer_norm(h.float(), h.shape[-1:]) for i, h in enumerate(hs))
        return self.out_ln(x)


class CRLA(nn.Module):
    """Conditioned residual latent adapter (TRLA-inspired), applied to every residue state independently:
        a = W_down LN(h) + b_down;  a <- (1 + Gamma c) * a + B c;  h' = h + tanh(alpha) (W_up GELU(a) + b_up).
    alpha starts at 0 so the block is an exact identity; W_up keeps its default init, otherwise no gradient would
    reach the block. Gamma and B start at 0 so the conditioning is silent at first."""

    def __init__(self, hidden: int, m: int, c_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, m)
        self.up = nn.Linear(m, hidden)
        self.gamma = nn.Linear(c_dim, m, bias=False)
        self.beta = nn.Linear(c_dim, m, bias=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, h, c=None):
        a = self.down(self.ln(h))
        if c is not None:
            a = (1 + self.gamma(c))[:, None, :] * a + self.beta(c)[:, None, :]
        return h + torch.tanh(self.alpha) * self.up(F.gelu(a))


def _entmax15_fwd(X, n_iter: int):
    """entmax_bisect(alpha=1.5) forward, same bisection and final normalisation, computed in fp32."""
    Xs = X.float() * 0.5
    max_val = Xs.amax(-1, keepdim=True)
    tau_lo = max_val - 1.0
    dm = (max_val - (1.0 / X.shape[-1]) ** 0.5) - tau_lo
    p = Xs
    for _ in range(n_iter):
        dm = dm / 2
        tau_m = tau_lo + dm
        p = torch.clamp(Xs - tau_m, min=0) ** 2
        tau_lo = torch.where(p.sum(-1, keepdim=True) - 1 >= 0, tau_m, tau_lo)
    return (p / p.sum(-1, keepdim=True)).to(X.dtype)


def _entmax15_bwd(Y, dY):
    gppr = torch.where(Y > 0, Y.float().sqrt(), 0.0)
    dX = dY.float() * gppr
    q = dX.sum(-1, keepdim=True) / gppr.sum(-1, keepdim=True)
    return (dX - q * gppr).to(dY.dtype)


_ENTMAX15_COMPILED = {}


def _entmax15_kernels():
    if not _ENTMAX15_COMPILED:
        _ENTMAX15_COMPILED["fwd"] = torch.compile(_entmax15_fwd, dynamic=True)
        _ENTMAX15_COMPILED["bwd"] = torch.compile(_entmax15_bwd, dynamic=True)
    return _ENTMAX15_COMPILED["fwd"], _ENTMAX15_COMPILED["bwd"]


class _Entmax15Bisect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, n_iter):
        Y = _entmax15_kernels()[0](X, n_iter)
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        return _entmax15_kernels()[1](Y, dY.contiguous()), None


def entmax15_bisect_fused(X: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """entmax_bisect(X, alpha=1.5, dim=-1, n_iter) with each bisection step fused into one kernel by torch.compile.
    Intermediates stay in fp32, so bf16 inputs give results within bf16 rounding of entmax_bisect, not bit-equal."""
    return _Entmax15Bisect.apply(X, n_iter)


def rbf(x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """Radial-basis encoding of a value in [0, 1] (identity as a fraction)."""
    c = torch.linspace(0, 1, n, device=x.device)
    return torch.exp(-(((x.float()[:, None] - c) * (n - 1)) ** 2))


class HiGO(nn.Module):
    def __init__(self, plm_name: str, n_terms: int, anc_pairs: np.ndarray, ia: np.ndarray, n_taxa: int,
                 adapter: str = "lora", lora_r: int = 8, d: int = 256, hier_query: bool = True,
                 pooling: str = "entmax", use_mcm: bool = True, use_taxon: bool = True, entmax_alpha: float = 1.5,
                 lora_targets=("query", "value"), go_edges: np.ndarray | None = None, use_go_gat: bool = False,
                 gat_layers: int = 2, gat_heads: int = 4, n_slots: int = 8,
                 plm_layers: int | None = None, plm_bf16: bool = False, readout: str = "last",
                 readout_layers=None, mix_top_init: float = 5.0, crla: bool = False, crla_dim: int = 64,
                 knn_fusion: bool = False, knn_phi=KNN_FEATURES, knn_eps: float = 1e-3, knn_lambda_init: float = 0.5,
                 n_rbf: int = 8, aspect=None, fast_entmax: bool = False):
        super().__init__()
        self.fast_entmax = fast_entmax and entmax_alpha == 1.5
        self.plm = build_backbone(plm_name, adapter, r=lora_r, alpha=2 * lora_r, targets=tuple(lora_targets),
                                  n_layers=plm_layers)
        self.plm_frozen = not any(p.requires_grad for p in self.plm.parameters())
        if plm_bf16:
            assert self.plm_frozen, "plm_bf16 is only for a frozen encoder"
            self.plm.to(torch.bfloat16)
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

        # --- HiGO-R: scalar-mix readout over intermediate encoder layers ---
        self.readout, self.mix, self._taps = readout, None, {}
        if readout == "mix":
            n_layers = self.plm.config.num_hidden_layers
            self.readout_layers = sorted(int(l) for l in (readout_layers or [n_layers]))
            assert 1 <= self.readout_layers[0] and self.readout_layers[-1] <= n_layers, \
                f"readout_layers {self.readout_layers} outside 1..{n_layers}"
            self.mix = ScalarMix(len(self.readout_layers), h, mix_top_init)
            layers = esm_core(self.plm).encoder.layer
            for l in self.readout_layers:
                layers[l - 1].register_forward_hook(self._tap(l))
        elif readout != "last":
            raise ValueError(f"unknown readout {readout!r}")
        # --- HiGO-R: retrieval-conditioned per-term logits ---
        self.n_rbf, self.knn_fusion = n_rbf, knn_fusion
        if knn_fusion or crla:
            assert aspect is not None, "knn_fusion / crla need the per-term aspect array"
            self.register_buffer("term_aspect", torch.as_tensor([ASPECT_INDEX[a] for a in aspect], dtype=torch.long))
        if knn_fusion:
            unknown = set(knn_phi) - set(KNN_FEATURES)
            assert not unknown, f"unknown knn_phi features {unknown}"
            self.knn_phi, self.knn_eps = tuple(knn_phi), knn_eps
            inv_softplus = math.log(math.expm1(knn_lambda_init))
            self.knn_a = nn.Parameter(torch.full((3,), inv_softplus))
            self.knn_beta = nn.Parameter(torch.zeros(n_rbf))
            self.knn_g = nn.Parameter(torch.zeros(3, len(self.knn_phi)))
            self.knn_delta = nn.Parameter(torch.zeros(n_terms))                  # strong weight decay (train.py)
            self.knn_g_delta = nn.Parameter(torch.zeros(n_terms, len(self.knn_phi)))
            depth = np.bincount(anc_pairs[:, 0], minlength=n_terms).astype(np.float32)
            const = np.stack([np.minimum(ia, 5.0) / 5.0, depth / max(depth.max(), 1.0)], 1)
            self.register_buffer("knn_const", torch.as_tensor(const, dtype=torch.float32))
        # --- HiGO-R: conditioned residual latent adapter at the readout (variant A) ---
        self.crla = CRLA(h, crla_dim, d + n_rbf + 1) if crla else None

    def _tap(self, layer: int):
        def hook(module, args, output):
            self._taps[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    def encode(self, input_ids, attention_mask):
        """Backbone residue states: last_hidden_state, or the scalar mix of the tapped layers."""
        with torch.no_grad() if self.plm_frozen else contextlib.nullcontext():
            out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
            if self.mix is None:
                return out.last_hidden_state
            hs = [self._taps[l] for l in self.readout_layers]
            self._taps.clear()
        return self.mix(hs)

    def protein_context(self, taxon, knn, batch: int, device):
        """CRLA conditioning vector c = [taxon embedding ; psi(max identity) ; has_hit] (zeros where unavailable)."""
        tx = self.taxon_emb(taxon) if (self.use_taxon and taxon is not None) else torch.zeros(batch, self.d, device=device)
        if knn is None:
            r = torch.zeros(batch, self.n_rbf + 1, device=device)
        else:
            hit = knn["has_hit"].float()
            r = torch.cat([rbf(knn["max_id"], self.n_rbf) * hit[:, None], hit[:, None]], 1)
        return torch.cat([tx.float(), r], 1)

    def retrieval_logit(self, knn: dict) -> torch.Tensor:
        """lambda_t * logit(clip(p_knn_t)) + g_t . phi_t, zeroed for proteins without a usable hit."""
        p = knn["p"].float().clamp(self.knn_eps, 1 - self.knn_eps)
        lp = torch.log(p) - torch.log1p(-p)
        hit = knn["has_hit"].float()
        max_id = knn["max_id"].float()
        psi = rbf(max_id, self.n_rbf)                                             # B x R
        lam = F.softplus(self.knn_a[self.term_aspect][None] + self.knn_delta[None] + (psi @ self.knn_beta)[:, None])
        B, T = p.shape
        feats = []
        for f in self.knn_phi:
            if f in ("s_star", "frac", "count"):
                if knn.get(f) is None:
                    raise KeyError(f"knn feature {f!r} requested by knn_phi but not provided by the kNN source")
                v = knn[f].float()
                feats.append(torch.log1p(v) / math.log(51.0) if f == "count" else v)
            elif f == "max_id":
                feats.append(max_id[:, None].expand(B, T))
            elif f == "has_hit":
                feats.append(hit[:, None].expand(B, T))
            else:
                feats.append(self.knn_const[:, 0 if f == "ia" else 1][None].expand(B, T))
        g = self.knn_g[self.term_aspect] + self.knn_g_delta                     # T x F
        r = lam * lp + (torch.stack(feats, -1) * g[None]).sum(-1)
        return r * hit[:, None]

    def _entmax(self, sc):
        if self.fast_entmax:
            return entmax15_bisect_fused(sc, 30)
        return entmax_bisect(sc, alpha=self.entmax_alpha, dim=-1, n_iter=30)

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
                term_chunk: int = 2048, knn: dict | None = None):
        """res_mask marks sequence residues (no <cls>/<eos>/<pad>), so padding never receives evidence.
        knn (only with knn_fusion): dict of tensors p, max_id, has_hit and optionally s_star, frac, count; without
        it every protein is treated as having no hit and gets the plain HiGO logits."""
        H = self.encode(input_ids, attention_mask)
        if self.crla is not None:
            c = self.protein_context(taxon, knn, H.size(0), H.device)
            H = self.crla(H.float(), c)
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
            a_s = self._entmax(sc_s)
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
                a = self._entmax(sc)
                v = torch.einsum("btn,bnd->btd", a, S)            # B x t x d
            else:
                sc = torch.einsum("td,bld->btl", qc, K) * scale
                sc = sc.masked_fill(~mask[:, None, :], -1e4)
                a = self._entmax(sc) if self.pooling == "entmax" else torch.softmax(sc, -1)
                v = torch.einsum("btl,bld->btd", a, H)            # B x t x d
            v = self.out(v)
            if ctx is not None:
                v = v + ctx[:, None, :]
            logits.append((F.gelu(v) * qc[None]).sum(-1) + self.term_bias[s:s + term_chunk])
            if return_attn:
                attns.append(a)
        z = torch.cat(logits, 1)
        if self.knn_fusion and knn is not None:
            z = z.float() + self.retrieval_logit(knn)
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


def build_higo(cfg: dict, vocab, anc_pairs: np.ndarray, n_taxa: int, go_edges: np.ndarray | None = None) -> HiGO:
    """The one place a run config becomes a model; train.py and every run loader go through it."""
    return HiGO(cfg["plm"], len(vocab), anc_pairs, vocab.ia.values.astype(np.float32), n_taxa=n_taxa,
                adapter=cfg["adapter"], lora_r=cfg.get("lora_r", 8), d=cfg.get("d", 256),
                hier_query=cfg.get("hier_query", True), pooling=cfg.get("pooling", "entmax"),
                use_mcm=cfg.get("use_mcm", True), use_taxon=cfg.get("use_taxon", True),
                lora_targets=cfg.get("lora_targets", ["query", "value"]), go_edges=go_edges,
                use_go_gat=cfg.get("use_go_gat", False), gat_layers=cfg.get("gat_layers", 2),
                gat_heads=cfg.get("gat_heads", 4), n_slots=cfg.get("n_slots", 8),
                plm_layers=cfg.get("plm_layers"), plm_bf16=cfg.get("plm_bf16", False),
                readout=cfg.get("readout", "last"), readout_layers=cfg.get("readout_layers"),
                mix_top_init=cfg.get("mix_top_init", 5.0), crla=cfg.get("crla", False),
                crla_dim=cfg.get("crla_dim", 64), knn_fusion=cfg.get("knn_fusion", False),
                knn_phi=cfg.get("knn_phi", KNN_FEATURES), knn_eps=cfg.get("knn_eps", 1e-3),
                knn_lambda_init=cfg.get("knn_lambda_init", 0.5), aspect=vocab.aspect.values,
                fast_entmax=cfg.get("fast_entmax", False))


# Trainable tensors that may legitimately be absent from a checkpoint and then keep their initial value.
OPTIONAL_KEYS: tuple[str, ...] = ()


def load_trainable(model: nn.Module, state: dict, optional=OPTIONAL_KEYS) -> list[str]:
    """Loads a trainable-only checkpoint (as saved by train.py) and raises on any mismatch, instead of letting
    load_state_dict(strict=False) silently drop or skip weights. Returns the optional keys left at init."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    known = set(model.state_dict())
    missing = sorted(trainable - set(state))
    unexpected = sorted(set(state) - known)
    fatal_missing = [k for k in missing if not k.startswith(tuple(optional))] if optional else missing
    if fatal_missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model built from its config: "
                           f"{len(fatal_missing)} missing {fatal_missing[:8]}, "
                           f"{len(unexpected)} unexpected {unexpected[:8]}")
    model.load_state_dict(state, strict=False)
    return [k for k in missing if k not in fatal_missing]
