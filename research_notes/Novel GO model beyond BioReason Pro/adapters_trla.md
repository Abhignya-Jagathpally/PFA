# Parameter-efficient adapters and representation steering for protein language models: TRLA and an in-house GO adapter for HiGO

Scope: (1) technical read of TRLA (Wu et al., 2026); (2) survey of adapter / steering methods relevant to PLMs; (3) design of a conditioned token-wise residual latent adapter for HiGO (frozen ESM2 + LoRA q/v r8 + per-term entmax evidence pooling + MCM/MCLoss, 6,276 terms).

Access note: the OpenReview forum and PDF block scripted access behind a browser challenge. Both were read through an in-IDE browser session; the full 11-page PDF text (main paper + supplementary tables S1 to S3, figures S1 to S4) was extracted, so all TRLA numbers below are from the paper itself.

---

## Q1. Is TRLA's mechanism transferable from fitness regression / seq2seq to multi-label GO classification? What must change?

### Takeaway
TRLA is a small (4.7M) token-wise residual MLP placed at the single interface between a frozen ProtT5-3B encoder and its frozen decoder, trained by MSE to map embeddings of a lower-fitness sequence onto those of a paired higher-fitness sequence. The architectural part (a nonlinear, per-residue residual update of frozen hidden states, zero-cost to the backbone) transfers directly to HiGO; the training signal (paired latent regression, substitution-only, same length) does not, and must be replaced by the end-to-end MCLoss signal, which makes the transferred module a supervised representation-steering adapter in the ReFT family. Its main value for GO is as a place to inject conditioning (taxon, domain tokens, homology evidence, aspect) at the residue level.

### Cited Findings

**Venue, status, code**
- Venue: ICML 2026 Workshop "Generative and Agentic AI for Biology" (GenBio); forum shows "GenBio 2026 Poster", Submission Number 51, published 28 May 2026, last modified 27 Jun 2026, CC BY 4.0. It is a workshop paper, not the ICML main track. — [OpenReview forum](https://openreview.net/forum?id=ZUBNaqRSDL)

> Continuation note (added 2026-10-12, second session): the Q1 section above was cut off partway through the venue bullet by the earlier session and is kept unchanged; only the link above was completed. The TRLA PDF was not re-read. The sections below use only the TRLA facts already recorded above: (a) one token-wise residual MLP of about 4.7M parameters; (b) placed at the single interface between a frozen ProtT5-3B encoder and its frozen decoder, so no gradient has to enter the backbone; (c) trained with MSE to map latents of a lower-fitness sequence onto those of a paired higher-fitness sequence (substitution-only, equal length). Hidden width, depth, normalisation and initialisation are not recorded above, so "TRLA-style block" below means the generic per-residue form h_i + f(h_i), not a reproduction of the paper's exact block (inference).

---

## Q2. Survey of parameter-efficient adaptation methods for a frozen-PLM GO classifier

Conventions. [fetched] = abstract or paper checked in this session; [known] = from prior knowledge, venue and numbers not re-checked in this session. Anything outside the ICLR / ICML / NeurIPS main tracks is labelled. Parameter counts "on 35M" are my arithmetic for ESM2-35M (12 layers, d = 480, FFN 1,920), not figures reported by the papers. As a check, our LoRA q/v r8 is 12 × 2 × (2 · 480 · 8) = 184,320 parameters = 0.55% of the encoder, which matches the run log.

### Takeaway
1. Among weight-space methods, LoRA is a strong default; DoRA, LoRA+, VeRA and AdaLoRA mostly trade parameter count or convergence speed rather than accuracy. This agrees with our own results (DoRA ≈ LoRA; all-linear r16 at 4.7% no better than q/v r8).
2. Representation-space methods (bottleneck adapters, (IA)^3, ReFT) edit hidden states, which is the natural place to inject per-protein conditioning; conditional and hyper-network adapters exist for exactly this purpose.
3. Only side-tuning / ladder side networks remove backprop through the backbone, which makes them the relevant family for ESM2-3B.
4. Protein PEFT studies agree that fine-tuning beats frozen embeddings on average and that LoRA ≈ full fine-tuning for mid-size and large ESM2. None of them evaluated CAFA-style GO prediction with a structured head like HiGO, so the size of the "fine-tuning gain" over a strong head is an open question for us (see Q4, control C0).

### Weight-space low-rank methods
- **LoRA (Hu et al.; ICLR 2022) [known].** Mechanism: W x + (α/r) · B A x with A ∈ R^{r×d}, B ∈ R^{d×r}, B zero-initialised; can be merged at inference. Params: 2dr per adapted matrix: 184k for q/v r8 on 35M, 1.35M on 650M, 2.95M on 3B. Memory: optimizer state is tiny, but activations must be stored or recomputed through every layer above the lowest adapted one, so training memory drops only modestly compared with full fine-tuning (LST measures about 26% saving for LoRA/Adapter, see below). Classification evidence: GLUE with RoBERTa/DeBERTa on par with full fine-tuning. Relevance to GO: our current adapter; its own contribution on top of the HiGO head has not been measured.
- **DoRA (Liu et al.; ICML 2024) [known].** Splits W into a magnitude vector and a unit-norm direction, applies LoRA to the direction and trains the magnitude. Params ≈ LoRA + d per matrix. Evidence: small gains over LoRA on commonsense reasoning and visual instruction tuning; on ProtT5 subcellular location, DoRA ≈ LoRA and about 30% slower (Schmirler et al., below) [fetched]. Relevance: we already saw DoRA ≈ LoRA; no further work suggested.
- **LoRA+ (Hayou, Ghosh, Yu; ICML 2024) [known].** Same parameterisation as LoRA, but B gets a larger learning rate than A (a ratio around 16 is suggested), derived from a width-scaling argument. No extra params. Evidence: faster convergence and small accuracy gains on GLUE and LLaMA fine-tuning. Relevance: a free change (two optimizer groups) if LoRA is kept; not expected to move Fmax much (inference).
- **VeRA (Kopiczko, Blankevoort, Asano; ICLR 2024) [known].** One pair of frozen random matrices shared across layers; only per-layer scaling vectors are trained: Λ_b B Λ_d A x. Params: d_out + r per matrix, about 17.7k for q/v on 35M at r = 256. Evidence: GLUE close to LoRA with roughly 10× fewer params. Relevance: low. Parameter count is not our constraint; the HiGO head alone has about 1.9M params.
- **AdaLoRA (Zhang et al.; ICLR 2023) [known].** LoRA in SVD form P Λ Q, with singular values pruned by an importance score so the rank budget is spread adaptively across matrices. Evidence: beats LoRA at small budgets on GLUE and SQuAD. Relevance: could show which ESM2 layers want rank, but our r16 vs r8 result suggests rank is not the bottleneck (inference).

### Activation scaling, prompt and prefix methods
- **(IA)^3 (Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"; NeurIPS 2022) [known].** Learned vectors rescale keys, values and FFN intermediate activations: k ⊙ l_k, v ⊙ l_v, (W_1 x) ⊙ l_ff. Params on 35M: 12 × (480 + 480 + 1,920) = 34,560. Evidence: strong few-shot results (T-Few). On ProtT5 subcellular location it was worse than LoRA and DoRA (Schmirler et al.) [fetched]. Relevance: weak on its own, but the rescaling vectors are a cheap carrier for conditioning if they are predicted from taxon or retrieval features (a hyper-(IA)^3; inference).
- **Prefix-tuning (Li & Liang; ACL 2021, not ICLR/ICML/NeurIPS), prompt tuning (Lester, Al-Rfou, Constant; EMNLP 2021, not ICLR/ICML/NeurIPS), P-tuning v2 (Liu et al.; ACL 2022, not ICLR/ICML/NeurIPS) [known].** Trainable key/value prefixes at every layer (prefix, P-tuning v2) or trainable soft input tokens (prompt tuning). Params: P × 2 × d × L for prefixes (P = 16 on 35M: 184k). Memory: P extra tokens in every attention; full backprop. Evidence: prompt tuning matches full fine-tuning only at very large scale; P-tuning v2 closes the gap on NLU classification at moderate scale. On ProtT5, prefix tuning was the weakest of the four PEFT methods tested by Schmirler et al. [fetched]. Relevance: a single soft "context token" carrying the taxon/retrieval summary is a plausible alternative to FiLM, since every residue can attend to it.

### Bottleneck adapters and representation interventions
- **Adapters (Houlsby et al.; ICML 2019) [known].** Bottleneck MLP h + W_up σ(W_down h) after the attention and FFN sublayers, initialised near identity. Params: 2dm + d + m per adapter; with m = 64 and two per layer on 35M, 1.49M. Evidence: within 0.4% of full fine-tuning on GLUE with about 3.6% params per task. Relevance: structurally the same object as a TRLA token-wise residual MLP. TRLA differs in placement (one block at an interface instead of one per layer) and in training signal.
- **ReFT / LoReFT (Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts; NeurIPS 2024) [known].** Edits hidden representations at chosen layers and token positions instead of weights. LoReFT: Φ(h) = h + Rᵀ(W h + b − R h), where R ∈ R^{r×d} has orthonormal rows, so the edit acts only inside an r-dimensional subspace. Params: 2rd + r per intervened layer (r = 4 on 35M: 3,844 per layer, 46k for all 12). Evidence: competitive with or better than LoRA on commonsense reasoning, arithmetic, instruction following and GLUE, with 15–65× fewer parameters (abstract figure as remembered; not re-checked). Memory: like LoRA, backprop through layers above the lowest intervention. Relevance: the closest published family to TRLA, and the evidence that representation edits at a few layers can match weight edits on classification. ReFT already intervenes at selected positions; using InterPro domain residues as those positions is our extension (inference).

### Conditional and hyper-network adapters
- **FiLM (Perez et al.; AAAI 2018, not ICLR/ICML/NeurIPS) [known].** γ(c) ⊙ h + β(c). The simplest way to make a token-wise MLP depend on a per-protein vector.
- **CA-MTL (Pilault, Elhattami, Pal; ICLR 2021) [known].** Task-conditioned attention, conditional layer norm and conditional bottleneck adapters in a mostly frozen transformer; multi-task GLUE gains with few params.
- **HyperFormer (Karimi Mahabadi, Ruder, Dehghani, Henderson; ACL 2021, not ICLR/ICML/NeurIPS) [known].** A shared hyper-network generates adapter weights from task and layer embeddings. **HyperTuning (Phang et al.; ICML 2023) [known]** generates PEFT parameters from few-shot examples without backprop into the LM. **CoDA, Conditional Adapters (Lei et al.; NeurIPS 2023) [known]** routes only a subset of tokens through the heavy frozen layers, with light adapters on the rest, for faster inference.
- Relevance to GO: here the "task" is a per-protein context (lineage, neighbours' labels, domains). Generating whole adapter matrices per protein is unnecessary; FiLM on an adapter bottleneck gives a per-protein modulation for O(d_c · m) params (inference). CoDA-style routing is a possible later speed-up for long proteins: send domain-covered residues through more capacity (inference).

### Mixture-of-LoRA experts
- **MoV / MoLoRA (Zadouri et al., "Pushing Mixture of Experts to the Limit"; ICLR 2024) [known].** Mixture of (IA)^3 vectors or LoRA experts with a soft token-level router; near full fine-tuning on held-out instruction tasks with under 1% params. **MoLE (Wu, Huang, Wei; ICLR 2024) [known].** Learned gating that composes several trained LoRAs per layer.
- Relevance to GO: experts could specialise by aspect (MF/BP/CC), kingdom or domain family. Since more LoRA rank did not help us, the case for experts rests on specialisation, not capacity. A FiLM-conditioned single adapter tests the same hypothesis more cheaply (inference).

### Side-tuning and ladder side networks
- **Side-tuning (Zhang et al.; ECCV 2020, not ICLR/ICML/NeurIPS) [known].** Output is a learned blend of a frozen base network and a small trainable side network.
- **LST, Ladder Side-Tuning (Sung, Cho, Bansal; NeurIPS 2022) [fetched abstract].** A small separate transformer (width reduced by a factor such as 8, initialised from a pruned copy of the backbone) receives downsampled intermediate activations from the backbone through gated "ladder" connections and makes the prediction. No gradient enters the backbone, which therefore runs in no-grad mode. Reported: saves 69% of full fine-tuning GPU memory on GLUE versus 26% for Adapter and LoRA at similar parameter counts (2.7× more saving); more accurate than Adapter and LoRA in a low-memory regime; on T5-large and T5-3B, better GLUE than full fine-tuning and other PETL methods — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html).
- Relevance to GO: high for ESM2-3B. With a ladder, the 3B encoder only runs forward (as in GO-GPT, which reads a frozen layer 30), and all trainable computation lives in a side network of about 6–8M params plus the HiGO head. It cannot change the encoder's attention patterns, so on 35M it is likely weaker than LoRA (inference).

### Zero-initialised gating
- **Flamingo (Alayrac et al.; NeurIPS 2022) [known].** New gated cross-attention layers inside a frozen LM, each output multiplied by tanh(α) with α = 0 at initialisation, so the frozen model is recovered exactly at step 0. This is the gate used in Q3.

### Protein language model PEFT and readout studies
- **Schmirler, Heinzinger, Rost; Nature Communications 2024 (journal) [fetched].** ESM2 (8M–3B), ProtT5 and Ankh on eight per-protein and per-residue tasks. Fine-tuning almost always improved over frozen embeddings with a head. LoRA matched full fine-tuning in most cases with up to 4.5× faster training. On ProtT5 subcellular location, LoRA (0.25% params) and DoRA (0.28%) beat IA3 (0.12%) and prefix tuning (0.5%), and DoRA was about 30% slower. Full fine-tuning of ESM2-3B hit exploding gradients. They recommend LoRA for ESM2-650M and larger — [Nat Commun](https://www.nature.com/articles/s41467-024-51844-2). GO prediction was not among the tasks. Relevance: supports keeping LoRA for 650M/3B. Their frozen baseline used a simple head, so their fine-tuning gains say little about the gain on top of HiGO's per-term entmax pooling.
- **Sledzieski, Kshirsagar, Baek, Dodhia, Lavista Ferres, Berger; PNAS 2024 (journal) [known].** LoRA on ESM2 for protein–protein interaction and homooligomer symmetry: PEFT matched or beat full fine-tuning with far fewer params, and low ranks or adapting a subset of matrices sufficed. Consistent with our r16 all-linear ≈ r8 q/v.
- **SES-Adapter (Tan et al.; J. Chem. Inf. Model. 2024, 64(16):6338–6349, journal) [fetched].** Frozen PLM; FoldSeek 3Di and DSSP structure sequences are embedded and fused with PLM embeddings by RoPE cross-modal attention (structure tokens as queries, PLM embeddings as keys/values), then pooled. Nine PLMs (ESM2, ProtBert, ProtT5, Ankh) and nine datasets, including function and annotation tasks; average +3% (max +11%) over vanilla PLMs and faster convergence — [JCIM](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c00689), [code](https://github.com/tyang816/SES-Adapter). Relevance: the closest precedent for fusing token-aligned side information into a frozen PLM; InterPro spans would play the role of the 3Di tokens.
- **Feature reuse and scaling (Li, Amini, Yue, Yang, Lu; ICML 2024) [known].** Across many protein tasks, transfer gains come mostly from early and intermediate layers and often do not grow with pretraining scale. **Valeriani et al. (NeurIPS 2023) [known]**: homology-level information in protein LMs peaks in intermediate layers. Both argue for reading out intermediate ESM2 layers instead of only the last; GO-GPT's choice of layer 30 of 36 fits this.
- **LP-FT (Kumar, Raghunathan, Jones, Ma, Liang; ICLR 2022) [known].** Fine-tuning can distort pretrained features and underperform out of distribution. Relevant to our remote-homolog advantage: any adapter should start as an exact identity and be kept from erasing features that transfer to distant sequences (inference).

### Cost summary on ESM2-35M (my arithmetic)

| Method | Trainable params on 35M | Backprop through encoder |
|---|---|---|
| LoRA q/v r8 (current) | 184k (0.55%) | yes, all 12 layers |
| LoRA all-linear r16 (tested) | ~1.6M (4.7%) | yes |
| (IA)^3 | 35k | yes |
| Houlsby, m = 64, 2 per layer | 1.49M | yes |
| LoReFT r = 4, all layers | 46k | yes, above lowest intervention |
| Prefix, P = 16 | 184k | yes |
| Ladder side network (width 256, 5 rungs) | ~4.6M | no |

---

## Q3. Design: a conditioned token-wise residual latent adapter for HiGO (working name CRLA)

### Takeaway
One block, three placements, in build order:
- **Variant A (TRLA placement):** a single conditioned block at the encoder–head interface, applied to a learned mix of intermediate layers. This is the direct TRLA analogue, and with LoRA off there is no backprop into the encoder.
- **Variant B (interior):** the same block after every second (35M) or third (650M, 3B) layer in the upper two-thirds of ESM2, each zero-gated.
- **Variant C (ladder, for 3B):** a ladder side network built from these blocks, so the 3B encoder runs forward only.

Conditioning comes from two sources. A per-protein vector c (taxon lineage, a summary of retrieved neighbours' labels, and top identity) modulates the bottleneck through FiLM. A per-residue vector u_i from InterPro spans is added to the bottleneck. Every residual output passes through a tanh(α) gate with α = 0, so at step 0 the model is exactly the current HiGO.

### Inputs and shared conditioning encoders
Per protein: lineage nodes at 8 ranks (superkingdom to species), InterPro entry spans (98% coverage), and up to k = 16 retrieved training neighbours j with identity s_j ∈ [0, 1], alignment coverage cov_j and propagated labels y_j ∈ {0,1}^6276.

```
Taxon lineage
  t   = mean_{rho in available ranks} ( E_tax[node_rho] + E_rank[rho] )        E_tax: 8,192 x 64 (nodes with >= 20 train proteins, plus an OOV row); E_rank: 8 x 64
Retrieval summary
  phi(s) = RBF(s; 16 centres on [0,1], width 0.07)                              16-d
  w_IA   = 1 + min(IA, 5)                                                        same weights as the MCLoss
  y~_j   = (y_j * w_IA) / ||y_j * w_IA||_1
  pi_j   = softmax_j( w_pi . phi(s_j) + w_cov * cov_j )
  r      = sum_j pi_j ( W_y y~_j + W_s phi(s_j) )        if k > 0              W_y: 6,276 x 128 (sparse matmul); W_s: 16 x 128
  r      = r_none                                        if k = 0 or dropped
Protein context
  c   = LN( W_c [ t ; r ; phi(s_max) ] + b_c )   in R^128                       W_c: 208 x 128
Residue domain vector (residue i of the current crop/window)
  u_i = mean_{e : i in span(e)} E_dom[e] + 1[i within 3 aa of a span boundary] * e_edge
  u_i = 0 for uncovered residues and for <cls>/<eos>/<pad>                      E_dom: 16,384 x 64 (entries seen >= 5 times in train)
```

Shared conditioning params: 2,406,224 (taxon table 0.52M, domain table 1.05M, label projection 0.80M, everything else < 0.04M). These are mostly lookup tables with sparse gradients, and the count does not depend on the backbone. Option: tie W_y to the HiGO term embeddings, r_label = W_r (Σ_j π_j y~_j)ᵀ E_term with W_r: 256 × 128 (33k params). This saves 0.77M and puts neighbour labels in the same space as the hierarchy-aware term queries. It also lets retrieval gradients move the queries, so test it both ways (inference).

### The block (applied to each residue independently, as in TRLA)
For hidden state h_i ∈ R^d at insertion point ℓ, with bottleneck width m:

```
a_i   = W_down^l LN^l(h_i) + b_down^l                                 (m)
a_i   = (1 + Gamma^l c) * a_i + B^l c + D^l u_i                       FiLM from protein context, additive domain term
h_i'  = h_i + tanh(alpha_l) * ( W_up^l GELU(a_i) + b_up^l )
```

Initialisation:
- α_ℓ = 0, so the block is an exact identity at step 0.
- W_down and W_up get standard (Xavier) initialisation, so the gradient with respect to α_ℓ is nonzero from the first step.
- Γ = B = 0, so conditioning starts silent; D has small normal values (std 0.02).
- Do not zero both α and W_up. If both are zero, every gradient into the block is zero and it never trains.

Params per block = (d·m + m) + (m·d + d) + 2d + 2(128·m + m) + 64·m + 1.

### Intermediate-layer readout
All variants replace `last_hidden_state` (the current input to `self.proj`) with a scalar mix:

```
H_read = LN( sum_{l in R} softmax(w)_l * LN_l(h^(l)) )      LN_l without affine; w init: 5.0 on the top layer of R, 0 elsewhere (~97% weight on top)
```

The per-layer LN is needed because HF ESM2 applies `emb_layer_norm_after` only to the final output; intermediate hidden states are the un-normalised residual stream.

| Backbone | Layers, d | Readout set R | Interior insertion points (variant B) | m |
|---|---|---|---|---|
| ESM2-35M | 12, 480 | {6, 8, 10, 12} | after blocks 4, 6, 8, 10, 12 (5 blocks) | 64 |
| ESM2-650M | 33, 1280 | {18, 22, 26, 30, 33} | after 12, 15, ..., 33 (8 blocks) | 128 |
| ESM2-3B | 36, 2560, truncated at 30 | {18, 22, 26, 30} | after 12, 15, ..., 30 (7 blocks) | 256 |

For 3B, the encoder is truncated after layer 30, following GO-GPT's readout layer. Layers 31–36 are dropped, which saves 17% of encoder compute. That layer 30 is also best for HiGO is an assumption; the learned mix weights will show where the signal sits (inference).

### Parameter counts

| Backbone | Per block | Variant A (1 block) | Variant B (interior blocks) | B as % of encoder | LoRA q/v r8, for comparison | Shared conditioning |
|---|---|---|---|---|---|---|
| ESM2-35M | 83,553 | 83.6k | 417,765 | 1.25% | 184,320 (0.55%) | 2.41M |
| ESM2-650M | 372,865 | 373k | 2,982,920 | 0.46% | 1,351,680 (0.21%) | 2.41M |
| ESM2-3B | 1,401,089 | 1.40M | 9,807,623 | 0.35% | 2,949,120 (0.10%) | 2.41M |

Add to these the scalar-mix weights (|R| + 1 scalars) and the existing head: term embeddings 1.61M, key/out 0.13M, proj d·256 + LN (124k / 328k / 656k), and the taxon embedding. B can be combined with a readout block (+1 block). For scale, the TRLA module was about 4.7M on a 3B encoder; variant B on 3B is about twice that, and variant A is about a third.

### Variant C: ladder side network for ESM2-3B

```
s = P_0 LN(h^(12))                                              P_k: 2560 -> 256; h detached (no gradient into the encoder)
for k = 1..7, layers l_k = 12, 15, ..., 30:
    s = (1 - sigmoid(mu_k)) * s + sigmoid(mu_k) * P_k LN(h^(l_k))      LST-style ladder gate
    s = s + SelfAttn_k(LN(s))                                   4 heads, width 256 (optional; gives the side stream residue context)
    s = CRLA_k(s; c, u)                                          block above with d = 256, m = 64 (54,209 params)
H_read = LN(h^(30)) + tanh(beta) * W_o s                         W_o: 256 -> 2560; beta = 0
```

Params: 8.13M with side attention, 6.28M without (most of it is the 2560→256 ladder projections). With β = 0, the start point is "frozen 3B layer 30 + full HiGO head", which is control C0-3B in Q4.

### Memory with gradient checkpointing (bf16, tokens_per_batch = 16,384 as in the current configs; all figures are estimates)
- **35M:** under 2 GB for the encoder side in every variant; the head's per-term entmax over 6,276 terms dominates.
- **650M, variant B:** weights 1.3 GB; checkpoint boundaries 16,384 × 1,280 × 2 B × 33 = 1.4 GB; one recomputed layer with attention under about 3 GB; head 1–2 GB. Total about 6–8 GB.
- **3B, variant B:** weights 5.7 GB; boundaries for 30 layers 2.5 GB. Recomputing one layer costs 0.34 GB for the FFN intermediate (16,384 × 10,240 × 2 B) plus up to 1.3 GB for eager attention probabilities at L = 1,024 with 40 heads (×2–3 for scores, softmax and dropout copies). Peak about 12–20 GB per GPU, so the batch could grow 3–4× on a 96 GB H100.
- **3B, variant A or C:** the encoder runs under `torch.no_grad()`, keeping only the 4–7 needed hidden states (0.3–0.6 GB); side network activations are under 2 GB. Total about 8–10 GB, with no checkpointing needed.
- **Optimizer state:** fp32 AdamW at about 16 bytes per param. 3B variant B (9.8M + 2.4M + about 2.5M head) is about 0.24 GB, which is negligible.

Compute per step, counting one encoder-layer forward as 1 unit (estimate):
- Current LoRA-3B with checkpointing: 36 forward + 36 recompute + 72 backward = 144 units.
- Variant B, with layers 1–11 under no_grad and LoRA off: 30 + 19 + 38 = 87 units (0.6× the current cost).
- Variants A and C: about 30 units plus the side network (about 0.22× the current cost), so roughly 4–5× more optimisation steps per GPU-hour on 3B.

The head's term-by-residue entmax is a fixed cost not counted here.

**Does a ladder avoid backprop through the 3B encoder?** Yes. Ladder inputs are detached, so the encoder runs in inference mode and keeps no activations beyond the rung layers. The price is that the encoder's own computation cannot adapt (no LoRA, no change to attention). How much that costs is unknown until control C0 is run: the 0.4885 frozen baseline also lacked the HiGO head. GO-GPT itself uses a frozen 3B. Inference: freezing should cost less at 3B than at 35M because the 3B representations are richer.

**Disk (about 4 GB free):** no hidden-state caching. For about 10^5 proteins × about 500 residues, 3B layer-30 states would take roughly 256 GB per layer, far above the free space. Everything else is small:
- Retrieval table: top-16 per protein (int32 id + fp16 identity + fp16 coverage) is about 13 MB.
- Domain spans: under 50 MB.
- Checkpoints: train.py already saves only trainable parameters, at most about 100 MB per run.
- Predictions: the largest outputs are the float16 `pred_*.npy` files, roughly 0.4 GB per 30k proteins (the exact sizes depend on split sizes).

### Training recipe
1. **Retrieval.** Run MMseqs2 or DIAMOND for every protein against the training set.
   - For training proteins, exclude self and same-cluster hits (`df.cluster`), mirroring TabR's leave-self-out (see `tier1_design_ideas.md`).
   - For each batch, sample an identity cap: no retrieval 0.25, ≤ 30% 0.25, ≤ 50% 0.2, ≤ 70% 0.15, uncapped 0.15. These probabilities are a starting guess. The caps make the model see remote-homolog-like contexts and keep it from learning to copy the top hit.
   - Validation and test use uncapped retrieval against the training set.
2. **Dropout of side inputs.** Lineage falls back to root-only with p = 0.1 (matches the existing `taxon_dropout`); domain dropout is p = 0.1 per protein.
3. **Cropping.** `Batcher.crop` picks a random start; it must also return that offset so domain spans can be shifted and clipped to the crop. Evaluation windows need the same treatment. Special tokens get u = 0.
4. **Implementation.**
   - Register forward hooks on `encoder.layer[l]` (under PEFT: `plm.base_model.model.encoder.layer[l]`) that replace `output[0]` with the block output. Set c and u on the module before calling the encoder. Hooks run inside the checkpointed function, so they are recomputed correctly.
   - Read intermediate layers via `output_hidden_states=True`; index l is the output after block l.
5. **Optimisation.**
   - AdamW. Adapter blocks use lr 3e-4 (the current LoRA lr); conditioning tables, gates and the scalar mix use `head_lr` 1e-3.
   - No weight decay on gates, LN or embedding tables.
   - Cosine schedule with 6% warmup, grad clip 1.0, bf16 autocast with fp32 master weights.
   - Loss: the unchanged IA-weighted MCLoss. Epochs and patience as in `dev.yaml` for 35M (15, patience 3); 3–5 epochs for 3B.
6. **Staging.** Variant A first, encoder frozen and LoRA off. Then variant B with LoRA off and with LoRA on. For 3B, variant A, then C.
7. **Optional regulariser** to protect remote-homolog performance: λ · mean ||tanh(α) W_up GELU(a)||² with λ ≈ 1e-3. This keeps edits small unless they reduce the loss (inference, in the spirit of LP-FT).
8. **Monitoring.** Log tanh(α_ℓ) per layer, scalar-mix weights and the entropy of π. Evaluate val Fw stratified by maximum identity to the training set.

**Complementary head-level retrieval prior** (from `tier1_design_ideas.md`, listed here because the ablations need it): z_t ← z_t + λ_t · logit(clamp(Σ_j π_j y_jt, 0.01, 0.99)), with λ_t = softplus(per-aspect scalar + small per-term offset), applied before the MCM. Most of the close-homolog gain is expected from this term. Adapter conditioning matters for a different reason: it lets retrieval shape which residues each term attends to (inference).

---

## Q4. Failure modes targeted, and the minimal ablation set

### Which failure mode each component targets

| Component | Failure mode it targets | Where a gain should show up if it works |
|---|---|---|
| Frozen encoder + full HiGO head (control C0) | We cannot tell how much of val 0.5270 comes from LoRA and how much from the head | (reference point) |
| Unconditioned residual block at readout (variant A) | Last-layer ESM2 features are shaped for masked-token prediction; a nonlinear per-residue map may be what LoRA approximates | All identity bins; should recover most of LoRA's gain if LoRA's role is reshaping features |
| Scalar-mix intermediate readout | Function and homology signal peaks in middle layers (Valeriani 2023; Li 2024); GO-GPT reads layer 30 of 36 | Remote homologs; larger backbones |
| Retrieval conditioning c, plus the head prior | GO-GPT wins on proteins with close training homologs; HiGO has no direct access to neighbours' labels | Identity bins > 50% |
| Identity-capped retrieval training | The model collapses onto copying neighbours and loses its remote-homolog strength | The < 30% bin should not drop |
| Domain vector u_i | Entmax pooling must find domain boundaries unaided; evidence gets mixed in multidomain proteins | Multidomain proteins; MF |
| Lineage t through FiLM | Taxon enters only as an additive logit-level vector over taxa with ≥ 50 training proteins; rare species get the shared "other" row | CC and BP for under-represented taxa |
| Zero-initialised tanh gates | An adapter that starts off-identity damages pretrained features (LP-FT) | Stability; no early val drop |
| Ladder side network (3B) | Backprop through 3B is the cost bottleneck; the current 3B run reached 1 epoch | Throughput, hence more epochs on 3B |

### Minimal ablation set
Unless stated otherwise, every run uses ESM2-35M, the same split, and the epochs and patience from `dev.yaml`. Report val Fw mean and per-aspect scores, stratified by maximum identity to the training set (< 30, 30–50, 50–70, > 70%).

1. **C0: frozen encoder + full HiGO head** (entmax, hierarchical queries, MCM, taxon), last-layer readout. This is the missing control. Use `adapter: frozen`, and first check that `train.py` accepts the then-empty adapter parameter group in AdamW (I have not verified this).
2. **C1: current LoRA q/v r8 + head.** The 0.5270 run exists; rerun for seeds.
3. **A0:** C0 + one unconditioned readout block + scalar mix.
4. **A0-last:** A0 with last-layer readout, to isolate the scalar mix. Can be dropped if A0 ≈ C0.
5. **B0:** interior blocks, unconditioned, LoRA off.
6. **B-full:** B0 + conditioning (c and u) + head retrieval prior.
7. **B-full minus retrieval, minus domains, minus lineage:** three leave-one-out runs.
8. **Retrieval in the head only:** the per-term prior with no adapter conditioning. This shows whether retrieval needs to enter the representation or only the logits.
9. **B-full + LoRA q/v r8:** tests whether the two are complementary.

A parameter-matched LoRA (q/v r18 = 414,720 params ≈ B0) is optional. The existing all-linear r16 result already shows that more LoRA parameters did not help.

Seeds: 3 for C0, C1 and the best variant; 1 for the rest initially. Inference: the 0.0017 gap between all-linear r16 and q/v r8 is probably within seed noise, so single-seed differences below about 0.005 should not be read as wins.

Decision rules (thresholds are judgement calls):
- **If C0 ≈ C1 (within about 0.005):** LoRA is not contributing and the head carries the gain. Work on 35M should then focus on conditioning, not feature adaptation, and frozen 3B + head (+ ladder) becomes the main path.
- **If C1 − C0 > 0.02 and A0 recovers most of it:** the gain is a readout-level feature map, and the TRLA placement suffices. Use variant A or C on 3B.
- **If A0 does not recover the gain but B0 does:** the adaptation must happen inside the encoder, and 3B needs variant B (backprop through layers 12–30).
- **Retrieval:** accept it only if the > 50% identity bins improve and the < 30% bin drops by no more than about 0.005.

3B minimum, after the 35M set:
- C0-3B: frozen, layer-30 readout, full head.
- C-3B: ladder + conditioning.
- The existing LoRA-3B run, compared at equal wall-clock time.

---

## Q5. Ranked recommendation (all gains are estimates on the internal val IA-weighted Fmax, not measured)

1. **Run C0 (frozen 35M + full HiGO head) and add seeds for C1.**
   - Cost: 2–3 runs of the 35M config.
   - Expected gain: none directly. This decides whether the adapter or the head is doing the work and whether a frozen 3B is viable. It is the highest-value use of GPU time.
2. **Add retrieval conditioning on 35M.** Build the head-level per-term prior first, then add c through one conditioned readout block (variant A with conditioning). Train with identity caps.
   - Estimate: +0.01 to +0.03 val Fw overall, concentrated in the > 50% identity bins where GO-GPT wins.
   - Risk: a small drop on remote homologs if the caps are too loose.
3. **Frozen ESM2-3B, truncated at layer 30, with scalar-mix readout, a variant A block and the HiGO head.** No backprop through the encoder.
   - Estimate: +0.01 to +0.03 over the best 35M model. This is based on the 1-epoch 3B lead (0.485 vs 0.445) and on roughly 4–5× more steps per GPU-hour than the current LoRA-3B setup, which allows more epochs.
4. **Ladder side network on 3B (variant C).** Build it only if item 3 leaves gain on the table, for example if the LoRA-3B run, once complete, beats frozen 3B by more than 0.01.
   - Estimate: 0 to +0.01 over item 3.
5. **Interior conditioned blocks with domain vectors (variant B) on 35M or 650M.**
   - Estimate: 0 to +0.01. The domain term should mostly help MF on multidomain proteins (inference).
6. **Low priority:** mixture-of-LoRA or per-aspect experts, hyper-network adapters, LoRA+, VeRA, AdaLoRA.
   - Estimate: ≤ 0.005 each. Our rank and DoRA results suggest the low-rank weight update is not the bottleneck.

Items 1–3 together might plausibly add +0.02 to +0.05 over val 0.5270. Gains from separate components are rarely additive, so treat this as a rough range (estimate).