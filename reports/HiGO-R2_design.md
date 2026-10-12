# HiGO-R2: per-niche design choices

Date: 2026-10-12. Scope: one technique per niche of the HiGO pipeline, each justified only by a main-track ICLR,
ICML or NeurIPS paper with public code. Target: beat the released GO-GPT checkpoint on GO-GPT's temporal test
(0.687 IA-weighted Fmax, beam; the ten-sample bar is 0.698, and a win the bootstrap can see needs about 0.708).

## 0. Starting point and constraints

What the code does now (read in `src/cafa6/model.py`, `train.py`, `data.py`, `knn_sharp.py`, `stacker.py`,
`stacker_tv.py`, `meanflow_refiner.py`):

- `model.build_backbone`: ESM2 from Hugging Face, frozen, with optional PEFT LoRA/DoRA on `query,value`, optional
  truncation after `plm_layers` blocks; `plm_bf16` casts a frozen encoder to bf16. A frozen encoder runs under
  `no_grad`.
- `model.ScalarMix`: `LN(sum_l softmax(w)_l LN_l(h_l))` over forward-hooked layers (`readout: mix`).
- `HiGO.forward`: projection to d = 256, ancestor-mean term queries (`queries`), per-term entmax-1.5 bisection
  (30 iterations) over residues in chunks of 2,048 terms, taxon vector added to the pooled value,
  logit = `gelu(v_t) . q_t + b_t`.
- `HiGO.retrieval_logit`: `lambda_t logit(clip p_knn) + g_t . phi_t`, `lambda_t = softplus(a_aspect + delta_t +
  beta . RBF(max id))`, zeroed without a hit; training samples identity caps per batch (`data.KNN_CAPS`).
- `HiGO.mcm` / `HiGO.loss`: C-HMCNN max-constraint and MCLoss with IA weights `1 + min(IA, 5)`.
- `CRLA`: one zero-gated, FiLM-conditioned token-wise bottleneck at the readout.
- `train.py`: AdamW, cosine schedule with warmup, bf16 autocast, early stopping on random-val fast Fmax.
- `knn_sharp.py`: the L0 sharpened kNN (`T5_k50_g0_na` registered on TV v2) and the in-memory cafaeval scorer.
- `stacker_tv.py`: cross-fitted stacker on TV v2 with HiGO + kNN features and test-like importance weights.

Measured facts used below (from `NOTES.md` and `research_notes/HiGO-R loop/observations.md`):

| fact | number |
|---|---|
| sharpened kNN on test, T=1-5, k=50 (all-norm) | 0.699-0.701 |
| kNN vs GO-GPT beam by identity bin (test) | 100%: 0.898 vs 0.835; 95-99.9%: 0.851 vs 0.829; 80-95%: 0.640 vs 0.657; 50-80%: 0.409 vs 0.469; 30-50%: 0.427 vs 0.438; <30%: 0.435 vs 0.402; no hit: 0 vs 0.374 |
| test bin shares | 100% 25.4%, 95-99.9% 28.2%, 80-95% 15.8%, 50-80% 18.9%, 30-50% 8.0%, <30% 0.9%, no hit 2.9% |
| TV v2 bin shares (7,566 proteins) | 100% 33.9%, 95-99.9% 35.2%, 80-95% 21.6%, 50-80% 4.0%, 30-50% 3.2%, <30% 0.9%, no hit 1.2% |
| HiGO-35M alone on test | 0.455; HiGO+gate beats GO-GPT at 30-50% (0.480 vs 0.440) and <30% (0.461 vs 0.395) |
| out-of-vocabulary terms in the kNN branch | worth 0.014 on test, 0.010 test-like on TV v2 |
| homology-only L1 stacker on TV v2 | +0.0008 overall, +0.0057 [+0.0028, +0.0084] test-like |
| MeanFlow refiner vs per-term fusion | +0.0014 (bar 0.005); BCE twin beat it |
| C0 (frozen 35M + head) vs C1 (LoRA q,v r8), TV v2 data, random val, epoch 1 | 0.5211 vs 0.5296 (+0.0085); 990 s vs 1,250 s per epoch on one H100 NVL |
| softmax vs entmax pooling (CAFA-6 val) | -0.0005; 0.000 after the homology gate |
| ESM2-3B LoRA, 1 epoch | 0.485 vs 0.445 for 35M (CAFA-6 val); 2.4-2.5 GPU-h per epoch |

The headroom is in the 50-95% bins (34.6% of test), where GO-GPT leads kNN by 0.017 (80-95%) and 0.060 (50-80%),
and in 30-50% (8%), where HiGO already leads. At >=95% (53.6% of test) kNN leads GO-GPT, and the job is to not
lose that lead. If the stacked system reached GO-GPT's level at 50-80% and a few points above it at 80-95% and
30-50% while keeping kNN at >=95%, the overall gain over kNN would be about +0.015 to +0.02 (a bin-weighted
approximation; Fmax is not exactly additive over bins).

Environment checks made for this report: transformers 5.17 has the attention-interface path in
`modeling_esm` (SDPA and flash variants), torch 2.13 ships the fused SDPA kernels, the `flash_attn` and
`schedulefree` packages are not installed, `peft` and `entmax` are. Two H100 NVL (95.8 GB each), 15 GB free disk.

### Evaluation protocol for every kill test

- Data: TV v2 (`data/proc_gogpt_tv2`), and only HiGO runs trained on `proc_gogpt_tv2` (post-T0 labels stripped)
  that write `pred_tv`. The kNN label database for TV is the T0 snapshot.
- Primary metric: TV v2 IA-weighted Fmax (mean of MF, BP, CC) under the test-like cell weights
  (`knn_sharp.cell_weights`), with the unweighted TV v2 mean as a guardrail. This settles open question 3 of the
  ledger; register it before the first test below runs.
- Per-bin numbers use `knn_sharp.BINS`. "Mid bins" means 50-95% pooled (about 1,940 TV v2 proteins); "remote bins"
  means <50% plus no hit (about 400 proteins; CIs there are wide, so remote-bin rules use the lower CI bound).
- Paired protein-level bootstrap, 1,000 replicates (`Scorer.bootstrap`). A component that changes the HiGO model
  is scored through the stacker of niche 8 (HiGO as a feature), because that is how it reaches the test.
- No test look is spent on any single niche; the two remaining looks go to the full system (section 11).

---

## 1. Protein backbone and readout layers

**Choice.** Frozen ESM2-3B truncated after block 30, bf16 weights, scalar-mix readout over blocks {18, 22, 26, 30};
keep the ESM2-35M LoRA model as a second, cheap ensemble member.

**Papers.**
- Valeriani et al., "The geometry of hidden representations of large transformer models", NeurIPS 2023
  (proceedings verified: proceedings.neurips.cc, 2023, a0e66093). On ESM2, intrinsic dimension rises, then falls
  in intermediate layers; the layers at the end of the first ID peak / at the relative ID minimum carry the most
  semantic (homology-level) neighbour structure, and the last layers are not the best readout.
  Code: https://github.com/diegodoimo/geometry_representations (exists).
- Li, Amini, Yue, Yang, Lu, "Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language
  Models", ICML 2024 (PMLR v235, li24a; OpenReview wdTiuvd0fR, verified). For most downstream tasks the transferred
  signal comes from features learned early in pretraining and does not scale with model size. This is the reason
  the 3B choice carries a kill test instead of being assumed.
  Code: https://github.com/microsoft/protein-transfer (exists).
- The ESM2 weights themselves come from the ESM code base, https://github.com/facebookresearch/esm (exists);
  ESM2's own paper is in Science (outside the allowed venues), so the backbone choice rests on the measurements
  above (3B 1-epoch lead, GO-GPT reading layer 30) and on the two readout papers.

**Code changes.** None in `model.py` beyond niche 9's `attn_implementation`; the config is
`plm: facebook/esm2_t36_3B_UR50D`, `adapter: frozen` (any value other than lora/dora gives a frozen encoder),
`plm_layers: 30`, `plm_bf16: true`, `readout: mix`, `readout_layers: [18, 22, 26, 30]`, `mix_top_init: 5.0`.
New script (small): cache the per-protein mean of the layer-30 states for niche 5 (0.6 GB, see niche 9).

**Cost.** Encoder forward O(30 (L^2 d + L d^2)) per protein with d = 2,560; weights about 4.8 GB in bf16 for 30
blocks; no activations stored except the four tapped layers (B L 2560 x 2 bytes each). Estimate: 0.3-0.5 GPU-h per
epoch over about 112k-119k proteins with SDPA (no backward through the encoder), against 2.4-2.5 GPU-h measured for
LoRA-3B. The head cost is unchanged (niche 3).

**Expected gain.** HiGO-3B alone perhaps +0.03 to +0.05 over HiGO-35M alone; after stacking with kNN, +0.005 to
+0.015 overall, almost all in the mid bins (50-95%) and 30-50%. Nothing expected at >=95%.

**Kill test.** Train frozen-3B-mix and reuse C0 (frozen 35M, same head, same data, same epochs budget), feed each
as the HiGO feature to the niche-8 stacker. Keep 3B if the test-like TV v2 Fmax gain is >= +0.005 and the mid-bin
gain has a lower CI bound > 0. Separately, keep the scalar mix only if mix minus layer-30-only is >= +0.003
test-like; otherwise read layer 30 alone.

**Rejected.**
- SaProt (ICLR 2024; https://github.com/westlake-repl/SaProt, exists): needs Foldseek 3Di tokens from predicted
  structures for every protein; the structure files do not fit in 15 GB, and coverage for TV/test isoforms is
  uncertain.
- ProtST (ICML 2023; https://github.com/DeepGraphLearning/ProtST, exists): pretrained against Swiss-Prot function
  text, which describes the function of many test proteins after T0 for the GO-GPT split; leakage risk on a
  temporal benchmark.
- ESM2-650M: sits between the two; spending the GPU budget on 3B (GO-GPT's own backbone) tests the larger
  question first.
- Reading only the last layer: contradicted by Valeriani et al. and by GO-GPT's own layer-30 choice.

---

## 2. Parameter-efficient adapter

**TRLA, assessed.** "Token-Wise Residual Latent Adapters: Steering Seq2Seq Models for Protein Fitness
Extrapolation" (Wu, Karimi, Banerjee et al., Amazon), OpenReview ZUBNaqRSDL. Venue: ICML 2026 workshop
"Generative and Agentic AI for Biology" (listed on icml.cc/virtual/2026/workshop/54073; the earlier project notes
record the forum as "GenBio 2026 Poster"). It is a workshop paper, not ICML main track, and no public code was
found (Amazon Science page and search, 2026-10-12). It therefore cannot serve as a justification under the venue
rule. What it is: a 5M-parameter token-wise residual MLP placed between a frozen ProtT5-3B encoder and its frozen
decoder, trained with MSE to map the encoder latents of a lower-fitness sequence onto those of a paired
higher-fitness sequence (same length, substitutions only); one forward pass generates the improved sequence.

Does the token-wise residual latent steering transfer to multi-label GO classification?
- The architecture transfers: a per-residue residual MLP on frozen states is a bottleneck adapter (Houlsby et al.,
  ICML 2019) at one location, or a ReFT-style representation intervention (NeurIPS 2024). Our `CRLA` is this
  block plus FiLM conditioning and a zero-init `tanh(alpha)` gate.
- The steering objective does not transfer. TRLA's signal is a paired target latent (the better variant's
  encoding). GO prediction has no paired target representation; the only supervision is the end-to-end BCE /
  MCLoss on labels, so "steering" reduces to ordinary supervised adaptation of a readout block.
- What remains distinctive for us is a place to inject per-protein context (taxon, retrieval summary) at the
  residue level, so that pooling can depend on it. That is a hypothesis about conditioning, not about TRLA's
  mechanism, and on a frozen encoder the head's `proj`, `key` and per-term attention already form a learned
  nonlinear map of residue states. Expected value: small (0 to +0.005), low priority.

**Comparison.**

| method | venue | code (checked) | where it acts | backprop through encoder | trainable params on 3B (q,v or equivalent) | measured here |
|---|---|---|---|---|---|---|
| LoRA | ICLR 2022 | https://github.com/microsoft/LoRA | weights, low rank | yes | 2.95M (q,v r8) | C1-C0 +0.0085 at epoch 1 (35M, random val) |
| DoRA | ICML 2024 (oral, per repo) | https://github.com/NVlabs/DoRA | weight magnitude + direction | yes | about LoRA + d per matrix | equal to LoRA (0.5189 vs 0.5194), slower |
| (IA)^3 | NeurIPS 2022 | https://github.com/r-three/t-few | activation scaling | yes | about 0.2M | not run |
| prefix tuning | ACL 2021 (outside allowed venues); analysed in He et al., ICLR 2022, https://github.com/jxhe/unify-parameter-efficient-tuning | attention key/value prefixes | yes | P x 2 x d x layers | not run |
| ReFT / LoReFT | NeurIPS 2024 | https://github.com/stanfordnlp/pyreft | hidden states in a rank-r subspace | above the lowest intervention | 2rd + r per layer | not run |
| LST ladder side-tuning | NeurIPS 2022 | https://github.com/ylsung/Ladder-Side-Tuning | separate side network fed by detached backbone states | no | 6-8M (side width 256) | not run |
| TRLA | ICML 2026 workshop | none found | one token-wise residual MLP at the interface | no | about 5M | `CRLA` exists, untested |

He et al. (ICLR 2022) show prefix tuning is an attention-placed adapter and that, at moderate budgets, changing
the FFN path is more effective than changing attention; with LoRA already near DoRA and r16-all-linear no better
than r8-q,v here, rank and placement are not the bottleneck.

**Choice.** No adapter inside the 3B encoder. Trainable capacity sits in the HiGO head; if niche-1's test shows a
gap that adaptation could close (C1 - C0 >= 0.01 on TV v2 at 35M), add an LST ladder side network (NeurIPS 2022)
reading the scalar-mix layers, with detached inputs and a zero-init gated output. LoRA q,v r8 stays only for the
35M member and only if C1 - C0 >= 0.01 test-like on TV v2 (the rule already registered for L3).

**Code changes.** `model.py`: new `LadderSide` module (gated per-rung mixing of `LN(h_l)` projected 2560 -> 256,
2-4 light transformer blocks of width 256, output `LN(h_30) + tanh(beta) W_o s`, beta = 0), hooked where `encode`
returns. Taps must be detached; the encoder stays under `no_grad`.

**Cost.** LST forward/backward O(R L d_s^2) with d_s = 256 and R rungs, against O(30 L d^2) for LoRA-3B backward;
about 0.3x LoRA-3B step time (estimate from the counting in `adapters_trla.md`), memory under 10 GB per GPU.

**Expected gain.** 0 to +0.01 on the HiGO feature over frozen 3B, mid bins; about a third of that after stacking.

**Kill test.** Frozen-3B-mix + LST vs frozen-3B-mix, both through the niche-8 stacker: keep LST only if
>= +0.004 test-like and the remote-bin change has a lower CI bound above -0.01.

**Rejected.** DoRA (measured tie, slower); LoRA on 3B (2.4-2.5 GPU-h per epoch, 5-8x the frozen cost, for a gain
the 35M runs put at about 0.01); (IA)^3 and prefix tuning (weaker than LoRA in the protein PEFT literature, and
prefix tuning fails the venue rule on its own); TRLA as a justification (venue, no code).

---

## 3. Residue-to-term pooling

**Choice.** Keep per-term query cross-attention over residues (one learned query per GO term, shared across
proteins), but compute it with softmax through the fused `torch.nn.functional.scaled_dot_product_attention`
kernel for training and test-time scoring; keep entmax-1.5 only for `explain.py` runs.

**Papers.**
- Set Transformer (Lee et al., ICML 2019; https://github.com/juho-lee/set_transformer, exists): pooling by
  multihead attention with learned seed vectors; our per-term queries are 7,901 seeds.
- Perceiver IO (Jaegle et al., ICLR 2022; https://github.com/google-deepmind/deepmind-research/tree/master/perceiver,
  exists): decoding a large structured output space with one output query per output element by cross-attention.
- Sparsemax (Martins and Astudillo, ICML 2016; https://github.com/deep-spin/entmax, exists, implements sparsemax
  and entmax): the sparse alternative, kept for readable evidence maps.
- FlashAttention-2 (Dao, ICLR 2024; https://github.com/Dao-AILab/flash-attention, exists) for the kernel (niche 9).

**Code changes.** `HiGO.forward`: new `pooling: "sdpa"` branch:
`v = F.scaled_dot_product_attention(qc.expand(B, 1, T, d), K[:, None], H[:, None], attn_mask=mask[:, None, None, :])`
then `self.out(v[:, 0])`; the default 1/sqrt(d) scale matches the current `scale`. The `term_chunk` loop can grow
to all 7,901 terms at once. `explain.py` keeps `pooling: entmax`.

**Cost.** Current entmax: time O(n_iter B T L) on top of O(B T L d) for scores and values, with n_iter = 30, and
O(B T L) floats stored for the backward pass (for B = 16, T = 7,901, L = 1,000 that is 126M floats, 0.5 GB per
copy in fp32, several copies through the bisection). SDPA: time O(B T L d), memory O(B (T + L) d), no B x T x L
tensor. The frozen-35M run spends most of its 990 s per epoch in the head, so this is the largest single
throughput lever for every later experiment.

**Expected gain.** About 0 Fmax (softmax tied entmax: -0.0005 raw, 0.000 gated); 2-4x head throughput (estimate),
which buys epochs and seeds for niches 1 and 6.

**Kill test.** Frozen-35M C0 config with `pooling: sdpa` vs `entmax`, same epochs: keep SDPA if the TV v2
test-like difference is >= -0.003 and measured epoch time drops by >= 1.5x. Otherwise keep entmax and use the
lost time budget as the cost of the other niches.

**Rejected.** Slot pooling (measured -0.0096, loses residue maps); mean pooling (-0.008 at 4 epochs);
label-wise attention with per-term key matrices (T x d x d parameters, 0.5G for d = 256, no support for the gain).

---

## 4. Label-structure modelling and hierarchy consistency

**Choice.** Keep C-HMCNN's max-constraint module and MCLoss with ancestor-shared term queries; apply the
max-constraint after every fusion step (HiGO-R logits, stacker output, calibration) so the final scores obey the
true-path rule over the full ontology including out-of-vocabulary terms (`Ontology.propagate_max`).

**Paper.** Giunchiglia and Lukasiewicz, "Coherent Hierarchical Multi-Label Classification Networks", NeurIPS 2020.
Code: https://github.com/EGiunchiglia/C-HMCNN (exists).

**Code changes.** None in `model.mcm` / `model.loss`. In the stacker / calibrator (niches 7-8) the order is:
stacker -> calibration -> `propagate_max`. The per-(aspect, bin) temperature is monotone within one protein and
one aspect, so on its own it would keep parent >= child; the stacker output is not hierarchy-consistent, though,
so `propagate_max` runs last.

**Cost.** O(B P) per batch with P = number of (descendant, ancestor) pairs plus self, one scatter-max forward and
two in MCLoss. Small next to pooling.

**Expected gain.** No new gain; the measured contribution is about +0.012 at convergence on val and +0.002 to
+0.004 on test after the homology gate. Out-of-vocabulary terms carried by the kNN channel are worth 0.010-0.014.

**Kill test.** On the frozen-3B run (niche 1): MCLoss vs plain BCE with MCM only at inference. Keep MCLoss if it is
>= +0.002 test-like through the stacker; otherwise switch to BCE + inference MCM (one scatter pass less per step).

**Rejected.**
- Multi-label box embeddings, MBM (Patel et al., ICLR 2022; https://github.com/iesl/box-mlc-iclr-2022, exists):
  learns label containment from data, but GO containment is given, and box intersection volumes over 7,901 terms
  add O(T d) per pair without improving on the exact constraint.
- Semantic loss (Xu et al., ICML 2018; https://github.com/UCLA-StarAI/Semantic-Loss, exists): penalises
  violations softly; MCM enforces them exactly at no extra parameter cost.
- GO-graph GAT on queries: measured -0.018 in the pilot.

---

## 5. Retrieval augmentation

**Choice.** Three channels, in this order of priority:
1. kNN-LM-style interpolation in logit space with per-term, identity-conditioned trust (the existing
   `retrieval_logit`, z_t = z_HiGO + lambda_t logit p_knn + g_t . phi_t), trained with per-batch identity caps
   so the model sees remote-homologue retrieval during training.
2. A TabR-style label-embedding channel: the neighbours' sharpened label vector is projected through a learned
   term-embedding matrix and added to the protein context, so neighbour labels can change every term's logit
   (co-occurrence), not only the term's own.
3. Embedding retrieval for proteins without a DIAMOND hit: cosine kNN over cached ESM2-3B layer-30 mean
   embeddings, used only when `has_hit` is false.

**Papers.**
- kNN-LM: Khandelwal et al., "Generalization through Memorization: Nearest Neighbor Language Models", ICLR 2020.
  Code: https://github.com/urvashik/knnlm (exists). It interpolates the model's distribution with a
  neighbour-label distribution; our version generalises the single lambda to a per-term, identity-dependent one,
  which the measured facts demand (kNN is best at >=95%, HiGO below 50%).
- TabR: Gorishniy et al., "TabR: Tabular Deep Learning Meets Nearest Neighbors", ICLR 2024 (OpenReview
  rhgIgTSSxW, verified). Code: https://github.com/yandex-research/tabular-dl-tabr (exists). Its retrieval module
  adds a label embedding of each neighbour to the value vector and trains with the query excluded from its own
  context; we keep DIAMOND's similarity (bitscore, identity sharpening) and borrow the label-embedding value path
  and the leave-self-out training (already done: self hits dropped in `KnnPrior.from_hits`).

**Code changes.** Channel 1: none (config `knn_fusion: true`, `knn_T: 5`, `knn_topk: 50`, `knn_norm: aspect`,
`knn_caps` as in `data.KNN_CAPS`; the TV v2 knn contract files exist). Channel 2: `HiGO.__init__` adds
`W_r` (T x d, initialised from `term_emb`), gate `alpha_r = 0`; in `forward`, `ctx = ctx + tanh(alpha_r) *
LN(p_knn @ W_r) * has_hit`. Channel 3: a script writes `emb_mean_l30.npy` (fp16) for train, TV and test; a
`KnnPrior.from_embeddings` fills `p` for rows without a DIAMOND hit, with `max_id` left at 0 and a separate flag
feature.

**Cost.** Channel 1: O(B T) per batch (dense B x T prior). Channel 2: O(B nnz(p) d) sparse-dense product plus
T x d parameters (2.0M). Channel 3: one O(N_q N_db d) matrix product at d = 2,560 (119k x 8k x 2,560 x 2 FLOP,
under a second on one GPU), 0.6 GB of disk.

**Expected gain.** Channel 1 vs post-hoc stacking alone: 0 to +0.008 overall, mid bins (the in-network model can
let retrieval shape per-term logits during training; the stacker can only reweight). Channel 2: 0 to +0.005,
30-95% bins. Channel 3: at most +0.002 overall (2.9% of test has no hit; GO-GPT scores 0.374 there, HiGO 0.360).

**Kill tests.**
- Channel 1: HiGO-3B with `knn_fusion` vs HiGO-3B without it, each through the niche-8 stacker. Keep if
  >= +0.003 test-like and the remote-bin lower CI bound is above -0.01 (the identity caps must protect remote
  homologues).
- Channel 2: on top of channel 1; keep if >= +0.003 test-like with mid-bin lower CI bound > 0.
- Channel 3: keep if the no-hit plus <30% bins improve by >= +0.02 with lower CI bound > 0 (about 160 TV v2
  proteins; a weak test, which is why it is last).

**Rejected.**
- RETRO (ICML 2022): no official code; chunked cross-attention over retrieved text is built for generation.
- REALM (ICML 2020): learned dense retrieval trained end to end; for proteins above 30% identity DIAMOND is the
  stronger retriever, and re-indexing during training does not fit the disk or GPU budget.
- PoET (NeurIPS 2023; https://github.com/OpenProteinAI/PoET, exists, inference only): conditions a generative
  model on homologue sets; generative scoring and sequence-of-sequences context lengths cost far more than the
  logit-space fusion, and the MeanFlow result already showed generative label modelling losing to BCE.
- Retrieval-conditioned residue attention through `CRLA`: kept behind niche 2's rule; no paper in the allowed
  venues shows a gain for it on classification.

---

## 6. Loss and optimisation

**Choice.**
- Loss: keep IA-weighted MCLoss (a per-term weighted BCE). A per-term weight w_t that does not depend on y leaves
  the minimiser of each term's loss at P(y_t = 1 | x), so the outputs stay calibrated marginals, which is what
  Fmax rewards; IA weighting only reallocates capacity toward rare, informative terms. Its use is still open
  (ledger question 6) and gets the matched-epoch L6 ablation.
- Optimiser: Schedule-Free AdamW in place of AdamW + cosine. Our runs are routinely stopped before the cosine
  horizon (e.g. 4 of 10 epochs); Schedule-Free evaluates an averaged iterate that is usable at any stopping time,
  which is the weight-averaging route to flatter minima with no extra pass.
- SAM as a second, conditional step for the trainable head on the frozen encoder.

**Papers.**
- Defazio et al., "The Road Less Scheduled", NeurIPS 2024 oral (proceedings verified, 136b9a13).
  Code: https://github.com/facebookresearch/schedule_free (exists).
- Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
  Code: https://github.com/google-research/sam (exists, JAX).
- ASAM (Kwon et al., ICML 2021, PMLR v139 kwon21b, verified): the official repository
  https://github.com/SamsungLabs/ASAM returned 404 on 2026-10-12, so it fails the code rule; scale-invariant
  sharpness would otherwise be the better fit for LayerNorm-heavy heads.

**Code changes.** `train.py`: `pip install schedulefree` (a few kB); replace `torch.optim.AdamW` +
`get_cosine_schedule_with_warmup` by `schedulefree.AdamWScheduleFree(groups, lr, warmup_steps)`; call
`opt.train()` before the training loop and `opt.eval()` before `predict` and before saving
`trainable_weights.pt`; keep the parameter groups from `param_groups`. SAM: in the step, run the encoder once under
`no_grad`, keep H, then do the two head passes (perturb trainable params by rho g/||g||, recompute loss, restore,
step), so the second pass never re-runs the encoder.

**Cost.** Schedule-Free: one extra copy of the trainable parameters (a few MB), no extra compute. SAM: about 2x the
head's forward/backward; with a frozen 3B encoder that is about +30-50% step time (the encoder forward is shared);
on frozen 35M it is close to 2x because the head dominates.

**Expected gain.** Schedule-Free: +0.002 to +0.005 on the HiGO feature, all bins, mostly by removing the
stopped-early schedule mismatch. SAM: 0 to +0.005, mostly where the test differs most from training (remote
bins, temporal shift).

**Kill tests.**
- Schedule-Free vs AdamW+cosine on frozen 35M (cheapest arm), same epoch budget, 2 seeds each: keep if the mean
  test-like gain is >= +0.003 and both seeds agree in sign.
- SAM (rho 0.05) vs no SAM on the frozen-3B head, equal wall-clock: keep if >= +0.003 test-like; drop otherwise.

**Rejected.**
- LDAM-DRW (Cao et al., NeurIPS 2019; https://github.com/kaidic/LDAM-DRW, exists) and logit adjustment (Menon et
  al., ICLR 2021; https://github.com/google-research/google-research/tree/master/logit_adjustment, exists): both
  move scores toward balanced error, which shifts per-term outputs away from calibrated marginals; IA weighting
  already addresses rare terms without that shift.
- Model soups (Wortsman et al., ICML 2022; https://github.com/mlfoundations/model-soups, exists): averaging works
  for models fine-tuned from one shared initialisation; our head and term embeddings start from different random
  inits per seed, so seeds would have to be averaged in output space (niche 8) instead.
- EMA of weights justified by "How to Scale Your EMA" (Busbridge et al., NeurIPS 2023): no official code found,
  and Schedule-Free already provides an averaged iterate.

---

## 7. Calibration and Fmax-optimal thresholding

**Choice.** Calibrated marginals from the stacker (weighted log-loss under the test-like weights), followed by a
per-(aspect x identity bin) temperature on the stacker's out-of-fold logits, fit on TV v2, then MCM. Thresholding
stays CAFA's single global tau per aspect (that is what Fmax scans); no per-term thresholds.

**Papers.**
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017. Code:
  https://github.com/gpleiss/temperature_scaling (exists). One scalar per group, fit by NLL, monotone, so it
  cannot hurt within-group ranking and only fixes cross-group comparability, which is the part a single global
  threshold depends on.
- Ovadia et al., "Can You Trust Your Model's Uncertainty?", NeurIPS 2019. Code:
  https://github.com/google-research/google-research/tree/master/uq_benchmark_2019 (exists). Calibration fit on
  in-distribution validation degrades under shift; this is why the temperature is fit on TV v2 and not on random
  val (the same failure the project measured for stackers).
- The F-measure plug-in results of Koyejo et al. (NeurIPS 2014; NIPS 2015, proceedings verified) explain why a
  threshold on calibrated marginals is the right target, but neither paper has public code (checked), so they are
  background here and not the justification.

**Code changes.** `stacker_tv.py`: after cross-fitting, fit 3 x 7 temperatures on the pooled out-of-fold logits
with the test-like weights (L-BFGS on the weighted BCE), apply to TV (out-of-fold) and test, then
`Ontology.propagate_max`. Report weighted ECE per bin next to Fmax.

**Cost.** CPU seconds; 21 parameters.

**Expected gain.** +0.001 to +0.004 overall; the identity bins where the stacker is least calibrated are the mid
bins (few TV v2 proteins at 50-80%).

**Kill test.** Out-of-fold TV v2 with vs without the per-bin temperatures: keep if test-like Fmax is >= +0.001
and weighted ECE drops in at least 5 of 7 bins; drop if Fmax falls at all.

**Rejected.**
- Mix-n-Match (Zhang, Kailkhura, Han, ICML 2020; https://github.com/zhang64-llnl/Mix-n-Match-Calibration,
  exists): ensemble and composition calibrators need more held-out data per group than the 50-80% bin has.
- Per-term thresholds or per-term isotonic maps: 7,901 x 3 parameters from 7,566 proteins; overfits, and Fmax uses
  one tau per aspect anyway.
- Dirichlet calibration (NeurIPS 2019): multi-class, not multi-label.

---

## 8. Stacking and ensembling under temporal shift

**Choice.** Cross-fitted stacker on TV v2 only (the existing `stacker_tv.py` design): candidates = HiGO top-k
union kNN terms over the full ontology; features = HiGO logit, kNN prior and its support statistics, identity
(raw + RBF), same-organism best hit, organism group, IA, depth; folds grouped by connected components of
near-duplicates; importance weights = test-like cell weights (organism group x identity bin x same-organism hit),
used as sqrt(w) sampling x sqrt(w) loss weight. Model: the "gated twin" (per-bin, per-aspect logit gates on
kNN and HiGO plus a zero-initialised nonlinear residual), with the per-term logistic fusion as control.
Seeds of HiGO are averaged in logit space before they enter as a feature.

**Papers.**
- Fang, Lu, Niu, Sugiyama, "Rethinking Importance Weighting for Deep Learning under Distribution Shift", NeurIPS
  2020 spotlight (proceedings verified). Code: https://github.com/TongtongFANG/DIW (exists). The paper's point is
  that importance weights estimated from learned features become circular; here the covariates of the shift are
  discrete and observed (cells), so the density ratio is a ratio of cell frequencies with no estimation loop,
  and the weighted loss is the plain importance-weighted ERM that DIW starts from.
- Gorishniy et al. (TabR, ICLR 2024) for the leave-self-out discipline that cross-fitting enforces here.

**Code changes.** `stacker_tv.py` already implements this; the missing inputs are `pred_tv` / `pred_test` from a
HiGO trained on `proc_gogpt_tv2` (L3 runs in progress, then the 3B run). Register the test-like criterion before
running it.

**Cost.** Candidate rows: about (HiGO top-300 union kNN terms) per protein, tens of millions of rows for TV v2;
GBM/twin training O(rows x features x iterations), CPU or one GPU-hour at most.

**Expected gain.** +0.005 to +0.015 overall over L0 kNN, almost all in 30-95% bins; the homology-only stacker
already gave +0.0057 test-like, and the HiGO feature carries the remote-homologue signal it lacked.

**Kill test.** Gated twin with the TV-clean HiGO feature vs L0 kNN on TV v2 (lofo out-of-fold estimate): keep if
>= +0.005 test-like with a lower CI bound > 0, and the >=95% bins do not drop by more than 0.003. Drop the HiGO
feature (fall back to the homology-only stacker) if the HiGO-in minus HiGO-out difference is < +0.003.

**Rejected.**
- Group DRO (Sagawa et al., ICLR 2020; https://github.com/kohpangwei/group_DRO, exists): optimises the worst
  group, while the target is the test-mix average; the worst group is no-hit/remote, with about 1% of test weight.
- Learned density-ratio weights (DIW proper): unnecessary when shift covariates are discrete and observed.
- Fitting any stacker on random val: measured to lose to kNN at >=95% on test (O30, O42).
- Ensembling with GO-GPT outputs as the headline: excluded by the plan.

---

## 9. Efficiency

**Choices.**
- SDPA attention inside ESM2: `AutoModel.from_pretrained(name, add_pooling_layer=False, attn_implementation="sdpa")`
  in `model.build_backbone` (transformers 5.17 supports it for ESM). FlashAttention-2 (ICLR 2024) and
  FlashAttention (Dao et al., NeurIPS 2022) give O(L) attention memory instead of O(L^2) and fewer HBM passes;
  code https://github.com/Dao-AILab/flash-attention (exists). With a padding mask PyTorch picks the
  memory-efficient kernel instead of the flash kernel; both avoid the L x L matrix.
- SDPA for the per-term pooling (niche 3).
- bf16 for the frozen encoder (`plm_bf16: true`) and bf16 autocast for the head, fp32 master weights for trainable
  parameters: Micikevicius et al., "Mixed Precision Training", ICLR 2018; reference implementation in
  https://github.com/NVIDIA/apex (exists); torch autocast is the path used.
- Length bucketing with a token budget (already in `Batcher.batches`) instead of token packing. Packing via
  `flash_attn_varlen_func` needs the `flash_attn` package (not installed; building it competes for the 15 GB) and
  would also need the per-term pooling to respect sequence boundaries. Revisit only if measured padding exceeds
  20% of tokens.
- Gradient checkpointing: off for every frozen-encoder run (nothing to checkpoint); on only for LoRA-3B, which this
  design does not run. Its standard reference (Chen et al. 2016) is an arXiv paper outside the allowed venues; it
  is an engineering switch here, not a design claim.
- Caching under 15 GB: no residue-level cache. Per-protein mean of 3B layer 30 for all proteins is
  about 119k x 2,560 x 2 B = 0.6 GB (+0.1 GB for TV and test): cache it. Residue-level layer-30 states would be
  about 119k x 450 x 2,560 x 2 B = 270 GB per layer; even PCA-256 in fp16 is 27 GB and PCA-128 in int8 is 6.9 GB,
  which leaves too little margin for checkpoints and predictions and drops the scalar mix. Predictions stay in
  `pred_topk` sparse format (about 15x smaller than dense).

**Complexity summary (per protein of length L; B proteins per batch; T = 7,901 terms; d = 256 head width; D =
2,560 encoder width; P = ancestor pairs; k = 50 neighbours).**

| component | time | memory (training) |
|---|---|---|
| ESM2-3B truncated at 30, frozen, SDPA | O(30 (L^2 D + L D^2)) | weights 4.8 GB bf16; 4 tapped layers B L D; no activations for backward |
| ScalarMix | O(4 L D) | 4 B L D |
| LST side network (conditional) | O(R L d_s^2 + R L D d_s) | O(R B L d_s) |
| per-term pooling, entmax (now) | O(B T L d + 30 B T L) | O(B T L) per chunk, kept for backward |
| per-term pooling, SDPA (proposed) | O(B T L d) | O(B (T + L) d) |
| MCM / MCLoss | O(B P) | O(B P) |
| kNN logit fusion | O(B T) | O(B T) |
| TabR label channel | O(B nnz(p) d) | T d parameters |
| embedding retrieval (no-hit) | O(N_q N_db D) once | 0.6 GB disk |
| Schedule-Free | O(#trainable) | one extra parameter copy |
| SAM on frozen encoder | about 2x head cost | one gradient copy |
| stacker (GBM / twin) | O(rows x features x iterations) | rows x features |

**Kill test (efficiency, no Fmax claim).** Measure epoch time and peak memory of frozen-3B-mix with SDPA vs eager
attention on 2,000 training proteins: keep SDPA if peak memory falls and predictions agree to within 1e-2 max
absolute probability difference; otherwise fall back to eager (correctness first).

---

## 10. Considered and rejected (summary)

| technique | venue | reason |
|---|---|---|
| TRLA as the adapter justification | ICML 2026 workshop | not main track; no public code; its paired-latent MSE objective has no counterpart in multi-label classification |
| DoRA | ICML 2024 | measured tie with LoRA, slower |
| LoRA on ESM2-3B | ICLR 2022 | 2.4-2.5 GPU-h per epoch vs about 0.4 frozen; expected gain about 0.01 |
| (IA)^3, prefix tuning | NeurIPS 2022; ACL 2021 | weaker than LoRA in protein PEFT comparisons; prefix tuning outside allowed venues |
| ReFT / LoReFT | NeurIPS 2024 | needs backprop through the encoder above the intervention; LST gives the same placement benefit without it |
| MBM box embeddings | ICLR 2022 | GO containment is known; MCM is exact |
| Semantic loss | ICML 2018 | soft version of what MCM enforces exactly |
| RETRO, REALM | ICML 2022; ICML 2020 | no official RETRO code; learned retrievers weaker than DIAMOND at >=30% identity |
| PoET | NeurIPS 2023 | generative, expensive; generative label modelling already lost to BCE here |
| ASAM | ICML 2021 | official repository 404 on 2026-10-12 |
| LDAM-DRW, logit adjustment | NeurIPS 2019; ICLR 2021 | move outputs away from calibrated marginals |
| Model soups | ICML 2022 | needs a shared initialisation; our heads do not have one |
| Group DRO | ICLR 2020 | worst-group objective, target is the test-mix average |
| Mix-n-Match, Dirichlet calibration | ICML 2020; NeurIPS 2019 | too little held-out data per bin; multi-class |
| SaProt, ProtST | ICLR 2024; ICML 2023 | structure files exceed disk; text pretraining risks temporal leakage |
| QLoRA 4-bit frozen 3B | NeurIPS 2023 (https://github.com/artidoro/qlora, exists) | memory is not binding (95 GB per GPU); dequantisation slows the forward |
| MeanFlow refiner | NeurIPS 2025 | measured +0.0014 vs bar 0.005; BCE twin better |
| residue-level embedding cache | - | 27 GB at PCA-256 fp16; does not fit |

---

## 11. Ranked implementation order (expected test Fmax per GPU-hour)

Gains are estimates on the test-like TV v2 metric, overall, for the full stacked system; GPU-hours are on one
H100 NVL. Items with near-zero GPU cost rank first when they have any expected gain.

| rank | step | GPU-h | expected gain | bins | gate |
|---|---|---|---|---|---|
| 1 | Finish L3 (C0, C1 on `proc_gogpt_tv2`, `pred_tv`), run `stacker_tv.py` with the TV-clean HiGO feature and the test-like criterion | about 2 (remaining L3 epochs) | +0.005 to +0.015 | 30-95% | niche 8 test |
| 2 | SDPA in ESM2 and in the per-term pooling (`pooling: sdpa`), bf16 frozen encoder | about 0.5 (verification runs) | 0 directly; 1.5-4x throughput for every later step | - | niches 3 and 9 tests |
| 3 | Schedule-Free AdamW in `train.py`, 2 seeds on frozen 35M | about 4 (two GPUs, in parallel) | +0.002 to +0.005 | all | niche 6 test |
| 4 | Frozen ESM2-3B at layer 30 with scalar mix {18, 22, 26, 30} + HiGO head (Schedule-Free if step 3 passes), trained on `proc_gogpt_tv2`, then restacked | about 4-8 (8 epochs, 2 seeds) | +0.005 to +0.015 | 50-95%, 30-50% | niche 1 test |
| 5 | In-network kNN fusion (`knn_fusion`, identity caps) on the 3B head, then restacked | about 4 | 0 to +0.008 | 50-95% | niche 5 channel 1 test |
| 6 | Per-(aspect x bin) temperature on stacker output | 0 | +0.001 to +0.004 | mid bins | niche 7 test |
| 7 | TabR label-embedding channel | about 4 | 0 to +0.005 | 30-95% | niche 5 channel 2 test |
| 8 | SAM on the 3B head | about 6 | 0 to +0.005 | remote bins | niche 6 test |
| 9 | LST ladder on 3B (only if C1 - C0 >= 0.01) | about 6 | 0 to +0.004 | mid bins | niche 2 test |
| 10 | Embedding retrieval for no-hit proteins | under 0.5 | 0 to +0.002 | no hit | niche 5 channel 3 test |
| 11 | CRLA conditioning | about 4 | 0 to +0.005 | unknown | niche 2 rule |

Step 6 costs no GPU time and could run any time after step 1; it is placed after the model steps because its
gain is the smallest and it must be refit whenever the stacker changes. The two remaining test looks: one after
step 5 (full system: kNN row, stacker, HiGO-3B, calibration, all pre-registered on TV v2) and one final look
after any of steps 7-11 that passed. Expected total over L0 kNN if steps 1-6 pass: about +0.01 to +0.03, i.e.
0.71-0.73 on test against 0.687 (beam) and 0.698 (ten-sample); these ranges are not additive guarantees.

---

## 12. What is new in the combined model

GO-GPT reads frozen ESM2-3B layer 30 and decodes GO terms autoregressively, with no explicit access to annotated
neighbours, so close-homologue transfer has to be memorised in its weights. DeepGOPlus combines DIAMOND scores and
a CNN with one global mixing weight; SPROF-GO pools PLM embeddings with attention and diffuses labels over a
homology graph; ProtBoost, as described in its CAFA-5 write-up, stacks gradient-boosted models over several PLM
embeddings and kNN features fitted on random folds; for GOAlpha we have the CAFA-6 leaderboard result (0.524
private) but no verified method description, so no claim is made against its design. HiGO-R2 differs in three
linked ways. First, retrieval trust is per term and a learned function of identity inside the network (z_t =
z_HiGO + lambda_t(identity) logit p_knn + ...), trained with identity-capped retrieval so the same model copies
near-duplicates and falls back to residue evidence for remote homologues, instead of one global gate. Second,
every term has its own query that attends over residues of a frozen, truncated 3B encoder read through a learned
mix of intermediate layers, and the outputs are made consistent with the GO hierarchy over the full ontology
(including out-of-vocabulary terms carried only by neighbours), so each prediction comes with a residue evidence
map and respects the true-path rule. Third, every learned combination step is fitted and selected only on a
temporal validation set that contains the duplicate-accession mechanism of the test and is reweighted to the
test's organism x identity x same-organism cells, with calibrated marginals as the output target; the project's
measurements show that this selection protocol, not model capacity, decided whether learned fusion beat a
zero-training kNN on the temporal test.

---

## Appendix: verification status of venues and code (2026-10-12)

| paper | venue | how the venue was checked | code URL | code check |
|---|---|---|---|---|
| Valeriani et al. | NeurIPS 2023 | proceedings PDF page | github.com/diegodoimo/geometry_representations | exists |
| Li et al. (feature reuse) | ICML 2024 | PMLR v235 + OpenReview | github.com/microsoft/protein-transfer | exists |
| TRLA (Wu et al.) | ICML 2026 workshop (GenBio) | icml.cc workshop page, Amazon Science | none found | - |
| LoRA | ICLR 2022 | not re-checked this session | github.com/microsoft/LoRA | exists |
| DoRA | ICML 2024 oral | repository description | github.com/NVlabs/DoRA | exists |
| (IA)^3 / T-Few | NeurIPS 2022 | not re-checked this session | github.com/r-three/t-few | exists |
| He et al. unified PET | ICLR 2022 | repository description | github.com/jxhe/unify-parameter-efficient-tuning | exists |
| ReFT | NeurIPS 2024 | not re-checked this session (search interrupted) | github.com/stanfordnlp/pyreft | exists |
| LST | NeurIPS 2022 | proceedings link in `adapters_trla.md` | github.com/ylsung/Ladder-Side-Tuning | exists |
| Set Transformer | ICML 2019 | not re-checked this session | github.com/juho-lee/set_transformer | exists |
| Perceiver IO | ICLR 2022 | not re-checked this session | github.com/google-deepmind/deepmind-research/tree/master/perceiver | exists |
| Sparsemax | ICML 2016 | not re-checked this session | github.com/deep-spin/entmax | exists |
| C-HMCNN | NeurIPS 2020 | not re-checked this session | github.com/EGiunchiglia/C-HMCNN | exists |
| MBM box embeddings | ICLR 2022 | iclr.cc poster page | github.com/iesl/box-mlc-iclr-2022 | exists |
| Semantic loss | ICML 2018 | not re-checked this session | github.com/UCLA-StarAI/Semantic-Loss | exists |
| kNN-LM | ICLR 2020 | not re-checked this session | github.com/urvashik/knnlm | exists |
| TabR | ICLR 2024 | OpenReview + ICLR proceedings | github.com/yandex-research/tabular-dl-tabr | exists |
| PoET | NeurIPS 2023 | not re-checked this session | github.com/OpenProteinAI/PoET | exists |
| Schedule-Free | NeurIPS 2024 oral | NeurIPS proceedings + oral page | github.com/facebookresearch/schedule_free | exists |
| SAM | ICLR 2021 | listed with venue in a unified SAM library; not re-checked on OpenReview | github.com/google-research/sam | exists |
| ASAM | ICML 2021 | PMLR v139 | github.com/SamsungLabs/ASAM | 404 |
| LDAM-DRW | NeurIPS 2019 | repository description | github.com/kaidic/LDAM-DRW | exists |
| Logit adjustment | ICLR 2021 | not re-checked this session | google-research/google-research/tree/master/logit_adjustment | exists |
| Model soups | ICML 2022 | not re-checked this session | github.com/mlfoundations/model-soups | exists |
| How to Scale Your EMA | NeurIPS 2023 | proceedings + poster page | none found | - |
| Temperature scaling (Guo et al.) | ICML 2017 | not re-checked this session | github.com/gpleiss/temperature_scaling | exists |
| Ovadia et al. | NeurIPS 2019 | not re-checked this session | google-research/google-research/tree/master/uq_benchmark_2019 | exists |
| Koyejo et al. | NeurIPS 2014, NIPS 2015 | NeurIPS proceedings + poster page | none (checked) | - |
| Mix-n-Match | ICML 2020 | not re-checked this session | github.com/zhang64-llnl/Mix-n-Match-Calibration | exists |
| DIW (Fang et al.) | NeurIPS 2020 spotlight | NeurIPS proceedings | github.com/TongtongFANG/DIW | exists |
| Group DRO | ICLR 2020 | not re-checked this session | github.com/kohpangwei/group_DRO | exists |
| FlashAttention / -2 | NeurIPS 2022 / ICLR 2024 | not re-checked this session | github.com/Dao-AILab/flash-attention | exists |
| Mixed Precision Training | ICLR 2018 | not re-checked this session | github.com/NVIDIA/apex | exists |
| QLoRA | NeurIPS 2023 | not re-checked this session | github.com/artidoro/qlora | exists |
| SaProt | ICLR 2024 | not re-checked this session | github.com/westlake-repl/SaProt | exists |
| ProtST | ICML 2023 oral | repository description | github.com/DeepGraphLearning/ProtST | exists |

Rows marked "not re-checked this session" rely on the standard citation; the web search tool stopped responding
partway through verification, and those venues should be confirmed on OpenReview or the proceedings site before
the report is cited externally.
