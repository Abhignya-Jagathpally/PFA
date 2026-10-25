# HiGO-R observations ledger: what was measured, and KEEP / DROP / UNTESTED per component

Date: 2026-10-12, about 07:10. Read-only pass. No training was run and no test look was spent; the only computation
done for this ledger was re-reading `train_log.jsonl` files, summary TSV/JSON files and evaluating the fitted gate
weights in `runs/*/gates.json`. Every number below is copied from (or, where marked "computed here", directly derived
from) the file named in the evidence column.

Target: GO-GPT temporal test (8,630 proteins, S3_all), IA-weighted Fmax, mean of MF/BP/CC, cafaeval prop=max,
norm=cafa, CAFA-5 IA, GO 2023-01-01.

## 0. Conventions

**Splits** (the split column in every table uses these names):

| name | what it is | n |
|---|---|---|
| C6-val | CAFA-6 30%-identity clustered validation split; fast GPU metric during training | 8,377 |
| C6-test | CAFA-6 30%-identity clustered test split; cafaeval | 8,058 |
| G-val | GO-GPT's random validation split (`data/proc_gogpt`); fast GPU metric | 13,305 |
| G-valB | cluster half B of G-val used for selection in the MeanFlow pilot | 6,653 |
| TV1 | temporal validation v1 (`data/proc_gogpt_tv`): dated post-T0 experimental labels, 10 reference proteomes, no zebrafish | 1,154 |
| TV2 | temporal validation v2 (`data/proc_gogpt_tv2`): dated stratum 1,238 + duplicate-accession (DA) stratum 6,328 | 7,566 |
| TV2-tl | TV2 reweighted to the test's organism x identity-bin x same-organism-best-hit cells (`chosen_v2.json` rule) | eff. n 959.9 |
| G-test | GO-GPT temporal test, S3_all | 8,630 |

**Verdict rule used here.**
- KEEP: a measured gain larger than the noise yardstick (or a registered bar) on a split that is relevant to G-test,
  and no measured loss on another split.
- DROP: a measured loss, or a gain below the registered bar on the registered selection split.
- UNTESTED: no measurement, a measurement confounded with another change, or a measurement only on a split known to
  rank things differently from G-test.
- "KEEP (neutral)" marks a zero-cost default whose effect is inside the noise on every split. It stays only because
  nothing argues against it.

**Noise yardstick.** HiGO-35M 5-fold CV on C6-val: 0.5135 +- 0.0038 (`runs/cv_summary/stats.tsv`). Six HiGO models
on C6-test: std 0.0063 raw, 0.0048 gated (`REPORT.md` §8.3). Paired protein bootstrap 95% half-width on G-test is
about 0.007 (`research_notes/Novel GO model beyond BioReason Pro/red_team.md` §3.2). Single-seed deltas under
about 0.005 are not interpretable.

**Reference rows on G-test** (`runs/bench_gogpt/summary.tsv`, `runs/gogpt_sample10/summary.tsv`):

| row | MF | BP | CC | mean |
|---|---|---|---|---|
| GO-GPT paper (Table S3) | 0.701 | 0.539 | 0.713 | 0.651 |
| GO-GPT shipped predictions | 0.689 | 0.542 | 0.717 | 0.649 |
| GO-GPT released, beam | 0.7143 | 0.6171 | 0.7284 | 0.6866 |
| GO-GPT released, greedy | 0.7155 | 0.6156 | 0.7279 | 0.6864 |
| **GO-GPT released, ten-sample frequency (L2 bar)** | 0.7203 | 0.6296 | 0.7454 | **0.6984** |
| sharpened kNN, red-team grid best (T1/T5, k50, all-norm) | | | | 0.701 |
| HiGO-35M + homology gate | 0.6469 | 0.4337 | 0.6472 | 0.5759 |
| DIAMOND + naive | 0.6495 | 0.4160 | 0.6378 | 0.5678 |
| HiGO-35M alone | 0.5022 | 0.3394 | 0.5225 | 0.4547 |

Scorer check: `knn_sharp.Scorer` reproduces cafaeval to about 1e-10 for GO-GPT beam (0.68659775) and HiGO-35M
(0.45468809) (`runs/knn_sharp/scorer_check.json`).

Test looks used: 2 of 4 (`runs/test_looks.md`): L0 v1 at 06:02:26 and the MeanFlow pilot at 06:41:45. The GO-GPT
ten-sample run is a reference measurement, not a look at our models.

---

## 1. Ledger by niche

### 1.1 Backbone / PLM size

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| ESM2-35M (current backbone) | `gogptbench_35m_lora`, 4 epochs of a 10-epoch cosine schedule | G-test | raw 0.4547; + gate 0.5759; at >=50% identity 0.5986 vs GO-GPT beam 0.7304 | KEEP as the reference row only (under-trained, not a candidate) | `runs/bench_gogpt/summary.tsv`, `runs/gogptbench_35m_lora/train_log.jsonl` (G-val 0.4784 -> 0.5447 over 4 epochs) |
| ESM2-3B + LoRA | `final_3b`, stopped partway into epoch 2 | C6-val | epoch 0: 0.4848 vs 35M epoch 0: 0.4454 (+0.039, same epoch, one seed) | UNTESTED (one epoch; converged gap unknown) | `runs/final_3b/train_log.jsonl`, `runs/dev_35m_lora/train_log.jsonl` |
| Frozen 3B truncated at layer 30 + scalar mix of layers 18/22/26/30 (L5) | `l5_smoke` (1,000 train / 300 val) | G-val subset | 0.3495 after 1 epoch; mix weights 0.245/0.250/0.253/0.252 (still uniform) | UNTESTED (smoke only; full run `l5_3b_mix_tv2` started, empty log at 07:05) | `runs/l5_smoke/info.json`, `runs/l5_smoke/train_log.jsonl`, `runs/l5_3b_mix_tv2/config.yaml` |
| ESM2-650M | weights downloaded only | - | none | UNTESTED | `logs/dl_esm2_650m.log` |

### 1.2 Adapter (LoRA, DoRA, CRLA)

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| LoRA q,v r8 vs frozen + plain mean-pool head | `dev_35m_lora` vs `base_35m_frozen_mlp` | C6-val | 0.5270 vs 0.4885 (+0.0385) | UNTESTED for the adapter alone (baseline lacks adapter and every HiGO head component) | `runs/dev_35m_lora/train_log.jsonl`, `runs/base_35m_frozen_mlp/train_log.jsonl` |
| same | same | C6-test | raw 0.5077 vs 0.4766 (+0.0311); gated 0.5277 vs 0.5060 (+0.0217) | same as above | `runs/eval_main/summary_all.tsv` |
| L3: frozen 35M + full HiGO head (C0) vs LoRA (C1), GO-GPT data with post-T0 labels stripped | `l3_c0_tv2`, `l3_c1_tv2` (running) | G-val | epoch 0: C0 0.4931, C1 0.5031 (+0.0100); C0 epoch 1: 0.5211 | UNTESTED (the decision rule needs `pred_tv` scored on TV2; one seed per arm; G-val is the split that mis-ranked retrieval settings) | `runs/l3_c0_tv2/train_log.jsonl`, `runs/l3_c1_tv2/train_log.jsonl`, `logs/l3_c*_tv2.log` |
| L3 on TV1 data (first attempt) | `l3_c0_frozen_full`, `l3_c1_lora` | - | no epoch completed; `l3_c1` crashed on a taxon-embedding size mismatch at resume (70 vs 13) | superseded by the TV2 runs | `logs/l3_c1.log`, empty `runs/l3_c*_{frozen_full,lora}/train_log.jsonl` |
| DoRA q,v r8 | `dev_35m_dora`, 6 epochs | C6-val | 0.5189 at epoch 5 vs LoRA 0.5194 at epoch 5 (-0.0005) | DROP | `runs/dev_35m_dora/train_log.jsonl`, `NOTES.md` Step 4 |
| LoRA all linear layers r16 (4.7% trainable) | `dev_35m_lora_all_r16` | C6-val | best 0.5253 vs 0.5270 (-0.0017); val loss rises from epoch 6 | DROP | `runs/dev_35m_lora_all_r16/train_log.jsonl` |
| same | same | C6-test | raw 0.5087 vs 0.5077 (+0.0010); gated 0.5291 vs 0.5277 (+0.0014), inside the 0.0048 std | DROP (no gain beyond noise for 8.5x parameters) | `runs/eval_main/summary_all.tsv` |
| CRLA (conditioned residual latent adapter, L9) | smoke tests only (`smoke_crla_mix_knn_frozen`, `smoke_crla_mix_lora_ckpt`) | - | no metric | UNTESTED (plan gates it on L3 showing that adaptation matters) | `logs/smoke_crla_*.done`, `runs/smoke_bc/smoke_crla_*.log` |

### 1.3 Pooling

All rows: ESM2-35M LoRA, one seed. "@4" = best C6-val within the first 4 epochs (full model 0.5086); "best" = best
over the whole run (full model 0.5270 at epoch 8). C6-test full model: raw 0.5077, gated 0.5277.

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| GO-conditioned per-term evidence pooling vs mean pooling | `abl_mean_pool` (15 epochs) | C6-val | @4 0.5006 (-0.0080); best 0.5218 (-0.0052) | KEEP | `runs/abl_mean_pool/train_log.jsonl` |
| same | same | C6-test | raw 0.5047 (-0.0030); gated 0.5274 (-0.0002) | (gate absorbs it; still the source of evidence maps) | `runs/eval_main/summary_all.tsv` |
| entmax-1.5 vs softmax | `abl_softmax_pool` (7 epochs) | C6-val | @4 0.5081 (-0.0005); best 0.5224 at epoch 6 (-0.0046, shorter run) | KEEP (neutral on accuracy; sparse maps: 39% of residues non-zero) | `runs/abl_softmax_pool/train_log.jsonl`, `NOTES.md` Step 6 |
| same | same | C6-test | raw 0.5034 (-0.0043); gated 0.5276 (-0.0001) | same | `runs/eval_main/summary_all.tsv` |
| slot pooling, 8 slots | `pilot_slots_35m_lora` vs `pilot_baseline_35m_lora` (20k/4k, 4 epochs) | C6-val subset | 0.4665 vs 0.4761 (-0.0096); behind at every epoch | DROP (no gain, loses residue evidence maps) | `runs/pilot_module_report.md` |
| evidence maps as explanations | `dev_35m_lora/explain` | C6-test (300 proteins) | pooling deletion of top-10% residues: -0.067 vs -0.0005 random (p = 0.007); input masking: -0.0003 vs 0.010 (no effect); InterPro enrichment median 1.13 | KEEP the claim "faithful to the head, not causal in sequence"; residue-1 artefact fix UNTESTED | `NOTES.md` Step 6, `runs/dev_35m_lora/explain/` |
| any pooling component on G-test | - | G-test | none | UNTESTED | - |

### 1.4 Hierarchical queries and GO-graph modules

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| ancestor-shared queries vs flat GO queries | `abl_no_hier_query` (12 epochs) | C6-val | @4 0.5063 (-0.0023); best 0.5231 (-0.0039), about 1 CV std | KEEP (neutral) | `runs/abl_no_hier_query/train_log.jsonl` |
| same | same | C6-test | raw 0.5068 (-0.0008); gated 0.5277 (+0.0001) | same | `runs/eval_main/summary_all.tsv` |
| GO-graph GAT on queries (`gelu(LayerNorm(conv(x)+x))`) | `pilot_gat_35m_lora` (20k/4k, 4 epochs) | C6-val subset | 0.4582 vs 0.4761 (-0.0180); gap by epoch 0.040, 0.022, 0.022, 0.018; epoch-0 train loss 0.49 vs 0.06 | DROP (current form) | `runs/pilot_module_report.md`, `REPORT.md` §7.1 |
| gated, zero-initialised GAT residual `q + g*GAT(q)` | never run | - | none | UNTESTED (not on the ladder) | `REPORT.md` §7.1 |
| label-query self-attention over top-256 candidates (L10) | never run | - | none | UNTESTED | plan, ladder L10 |

### 1.5 Max-constraint module (MCM) and MCLoss

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| MCM + MCLoss vs plain BCE without MCM | `abl_no_mcm` (11 epochs) | C6-val | @4 0.5021 (-0.0065); best 0.5152 (-0.0118, about 3 CV std) | KEEP | `runs/abl_no_mcm/train_log.jsonl` |
| same | same | C6-test | raw 0.5052 (-0.0024); gated 0.5239 (-0.0037); the only head ablation whose loss survives the gate | KEEP | `runs/eval_main/summary_all.tsv` |
| MCM after the grounded calibrator | `grounded`, `grounded_gogpt` | C6-test, G-test | 0 ontology violations for every variant | KEEP (output property) | `runs/grounded/violations.json`, `runs/grounded_gogpt/violations.json` |
| MCM on G-test models | - | G-test | no ablation | UNTESTED | - |

### 1.6 IA weighting

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| IA-weighted loss (w = 1 + min(IA, 5)) vs unweighted | `abl_no_ia_weight` (4 epochs) vs full (12 epochs) | C6-val | @4: removing it gives 0.5095 (+0.0009) | UNTESTED (confounded) | `runs/abl_no_ia_weight/train_log.jsonl` |
| same | same | C6-test | removing it: raw 0.4929 (-0.0147); gated 0.5191 (-0.0086); 4 vs 12 epochs, so the loss mixes component and training length | UNTESTED (confounded; val and test disagree in sign) | `runs/eval_main/summary_all.tsv` |
| IA-weighted rows in the L1 stacker | `stacker_l1`, `stacker_l1_v2` | TV1 | logreg_iaw 0.5117 vs logreg 0.5105 (+0.0011); gbm_iaw 0.5055 vs gbm 0.5068 (-0.0013) | DROP (no effect in the stacker) | `runs/stacker_l1/summary.tsv` |
| same | same | TV2 | logreg_iaw 0.7291 vs logreg 0.7275 (+0.0015); gbm_iaw 0.73443 vs gbm 0.73442 (0.0000) | same | `runs/stacker_l1_v2/summary.tsv` |
| soft-F1 x ZLPR, parent masking, logit adjustment (L6) | never run | - | none | UNTESTED | plan, ladder L6 |

### 1.7 Taxon / organism

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| species embedding (41 species, 10% dropout) | `abl_no_taxon` (4 epochs) | C6-val | @4 0.4956 (-0.0130), largest head effect, mostly BP/CC | KEEP | `runs/abl_no_taxon/train_log.jsonl` |
| same | same | C6-test | raw 0.4760 (-0.0316); gated 0.5076 (-0.0201); 4-epoch run, so an upper bound | KEEP | `runs/eval_main/summary_all.tsv` |
| species embedding on GO-GPT data (70 taxa) | in every G-data run; no ablation | G-test | none. Test is 45.2% zebrafish; TV2 is 8.6% zebrafish | UNTESTED | `logs/tv_knn.log` (TV2 build block) |
| organism of the best kNN hit as a feature (same-organism flag) | used only as a TV2 reweighting cell, not as a model feature | TV2, G-test | test same-organism best hit: 100% bin 0.909, 95-99.9% 0.878, 80-95% 0.526 | UNTESTED as a feature | `runs/knn_sharp/chosen_v2.json`, `logs/tv_knn.log` |

### 1.8 Retrieval / kNN

kNN weights: `w_j = bits_j * exp((pident_j - max pident)/T) * cov_j^gamma` over the top-k hits by bits; "na" =
weights normalised per aspect, otherwise over all hits.

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| bits-weighted kNN + 0.8/0.2 naive blend (old baseline) | `baselines`, `bench_gogpt` | C6-test, G-test | C6-test 0.4614; G-test 0.5678 | DROP as the retrieval baseline (superseded by sharpening) | `runs/baselines/metrics.tsv`, `runs/bench_gogpt/summary.tsv` |
| identity sharpening (finite T) vs flat (T = inf) | red-team grid; L0 v1 look #1 | G-test | sharpened grid 0.685-0.701 (12 settings, all-norm, full label sets); flat `Tinf_k50_g1_na` 0.5961 | KEEP | `red_team.md` §3.2; `runs/knn_sharp/final.json` |
| same | `knn_sharp` v2 grid | TV2-tl | T2_k50 - Tinf_k50: +0.0958 [+0.0887, +0.1026]; T2_k10 - Tinf_k10: +0.0781 [+0.0718, +0.0835] (500 reps) | KEEP | `runs/knn_sharp/chosen_v2.json` |
| temperature value | red-team grid; v2 grid | G-test; TV2-tl | G-test at k50: T1 0.701, T2 0.700, T5 0.701, T10 0.691. TV2-tl at k50 aspect-norm: T2 0.7149, T5 0.7168, T10 0.7155 (spread 0.002) | KEEP T in 1-5 (registered T=5); finer choice UNTESTED (below resolution) | `red_team.md` §3.2, `runs/knn_sharp/v2/summary.tsv` |
| neighbours k | same | G-test; TV2-tl | G-test: k50 >= k10 >= k3 at every T. TV2-tl: T5 k50 0.7168 vs T5 k10 0.7129 | KEEP k=50 | same |
| coverage exponent gamma | v1 and v2 grids | TV1, TV2-tl, G-val | gamma 1 vs 0 on matched pairs: TV2-tl T5_k50 0.7161 vs 0.7168; TV1 test-like Tinf_k50 0.5550 vs 0.5509 | DROP (no consistent effect; registered rule prefers fewer knobs) | `runs/knn_sharp/summary.tsv`, `runs/knn_sharp/v2/summary.tsv` |
| per-aspect weight normalisation | v1 and v2 grids | TV2-tl | T5_k50_g0_na 0.7168 vs T5_k50_g0 0.7031 (+0.0137) | KEEP on validation; UNTESTED on G-test (red-team grid is all-norm only) | `runs/knn_sharp/v2/summary.tsv` |
| same | same | TV1 test-like | Tinf_k50_g1_na 0.5550 vs Tinf_k50_g1 0.5316 (+0.0234) | same | `runs/knn_sharp/summary.tsv` |
| **registered L0 v2: `T5_k50_g0_na`** | pre-registered 06:17:54 on TV2-tl | TV2 / TV2-tl | 0.7345 / 0.7168; vs v1 choice +0.0975 [+0.0917, +0.1031]; vs T2_k10_g0 (all-norm) +0.0151 [+0.0119, +0.0183] (TV2-tl, 500 reps) | KEEP as the L0 baseline; **G-test score UNTESTED** | `runs/knn_sharp/chosen_v2.json`, `runs/test_looks.md` |
| "sharpened kNN ties GO-GPT" | red-team val-selected T10_k50 (all-norm) | G-test | 0.691; minus GO-GPT beam +0.005 [-0.0015, +0.012] (300 reps); S6 0.709 vs 0.702. Grid best 0.701 is +0.003 over the ten-sample bar 0.6984 (no CI) | KEEP (tie, not a significant win) | `red_team.md` §0, §3.2; `runs/gogpt_sample10/summary.tsv` |
| per-bin profile of sharpened kNN T2 k10 | red-team | G-test | 100% 0.898, 95-99.9% 0.851, 80-95% 0.640, 50-80% 0.409, 30-50% 0.427, <30% 0.435, no hit 0 (GO-GPT beam: 0.835 / 0.829 / 0.657 / 0.469 / 0.438 / 0.402 / 0.374) | KEEP at >=95%; loses at 50-95% | `red_team.md` §3.2 |
| full propagated label sets vs vocabulary-only (7,901 terms) | MeanFlow pilot kNN vs red-team kNN, both T2 k10 | G-test | 0.6843 (vocab-only) vs 0.698-0.699 (full); vocabulary holds 93.8% of test IA mass (MF 88.7%) | KEEP full label sets (about 0.014 at stake; cross-script comparison, no CI) | `runs/meanflow_pilot/summary.tsv`, `red_team.md` §1 claim 16 |
| best-hit copy | red-team | G-test | binary 0.679; scored by identity 0.684; vs sharpened T2 k10 0.699 | DROP (sharpened kNN is better) | `red_team.md` §3.2 |
| ortholog channel (10 model-organism panels) | `pilot_comparative_genomics` | C6-val, C6-test | DIAMOND + orthologs + naive 0.4644 / 0.4569 vs DIAMOND + naive 0.4685 / 0.4614 (-0.004 on both) | DROP | `runs/pilot_comparative_genomics/metrics.tsv`, `REPORT.md` §7.2 |
| kNN database including GO-GPT val proteins (13,305) for the final model | never measured | - | none | UNTESTED | plan, "Date cutoffs" |
| retrieval inside the network (L4a per-term kNN prior; L4b TabR attention) | config `configs/l4a_knn_prior.yaml`, smoke `smoke_knn_lora` | - | no metric | UNTESTED | `configs/l4a_knn_prior.yaml`, `logs/smoke_knn_lora.done` |
| abstention on no-hit proteins | red-team bin formula | G-test | kNN 0.684 by count-weighted bins vs 0.699 measured (norm=cafa ignores proteins without predictions) | metric property; naive fallback for no-hit with sharpened kNN UNTESTED | `red_team.md` §3.1 |

### 1.9 Fusion / stacking

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| per-aspect homology gate g = sigmoid(w . [id, has_hit, 1]) | `eval_main`, `bench_gogpt` | C6-test | 0.5077 -> 0.5277 (+0.0200) | DROP as the fusion mechanism; KEEP as a reference row | `runs/eval_main/summary_all.tsv` |
| same | same | G-test | 0.4547 -> 0.5759 (+0.1213), but 0.11-0.13 below sharpened kNN (0.685-0.701). Gate at 100% identity (computed here from fitted weights): G-data MF 0.659, BP 0.577, CC 0.613; C6 MF 0.417, BP 0.293, CC 0.332 | same | `runs/bench_gogpt/summary.tsv`, `runs/bench_gogpt/gates.json`, `runs/eval_main/gates.json` |
| per-bin kNN/HiGO+gate blend, weights fit on G-val | red-team, kNN T10 k50 | G-test | 0.687 vs kNN alone 0.691 (-0.004); blend - GO-GPT +0.001 [-0.005, +0.008] | DROP (random-val fit) | `red_team.md` §3.2 |
| same with kNN T2 k10 fixed a priori | red-team | G-test | 0.700 vs kNN alone 0.699 (+0.001, tie); test-fit oracle 0.711 (not reportable) | DROP as fitted (no gain); idea re-fit on TV2 UNTESTED | `red_team.md` §3.2 |
| per-term logistic fusion (HiGO logit, kNN logit, p_knn, id, has_hit, interaction), fit on G-val A | MeanFlow pilot control, lambda 1e-3 | G-valB | 0.6169 vs kNN 0.5476 (+0.069) and HiGO 0.5294 | DROP (random-val fit) | `runs/meanflow_pilot/summary.tsv` |
| same | same | G-test | 0.6388 vs kNN (vocab-only) 0.6843 (-0.0455); >=95% bins 0.7527 vs 0.8557; 50-95% 0.5411 vs 0.5092; <50%/no hit 0.4360 vs 0.3552 | same | same |
| nonlinear per-term combiner ("discriminative twin", 6.75M params, BCE), fit on G-val A | MeanFlow pilot | G-valB | 0.6246 vs fusion 0.6169 (+0.0077) | UNTESTED on TV2 (direction supported, random-val fit lost on test) | same |
| same | same | G-test | 0.6719 vs fusion 0.6388 (+0.0331), vs kNN 0.6843 (-0.0124); >=95% 0.8062 vs 0.8557; 50-95% 0.5485 vs 0.5092; <50% 0.4443 vs 0.3552 | same | same |
| L1 stacker, homology-only features, TV1 | `stacker_l1` (5-fold CV) | TV1 | best logreg_iaw 0.5117 vs L0 0.5185: -0.0068 [-0.0142, -0.00004]; gbm -0.0117 [-0.0194, -0.0026] | DROP | `runs/stacker_l1/chosen.json`, `summary.tsv` |
| same, fit on TV1, scored on G-val | same | G-val | 0.5853 vs L0 0.5938 (-0.0085) | DROP | `runs/stacker_l1/per_bin.tsv` |
| L1 stacker, homology-only features + sharpened score, TV2 | `stacker_l1_v2` gbm+sharp | TV2 | 0.7353 vs L0 0.7345: +0.0008 [-0.0016, +0.0032]; remote bins 30-50% +0.0227, <30% -0.0019 | DROP by the registered rule (>= +0.005 on TV2 mean) | `runs/stacker_l1_v2/chosen.json` |
| same | same | TV2-tl | +0.0057 [+0.0028, +0.0084] (would pass the bar on the selection criterion used for L0 v2) | open: criterion mismatch (see §2) | `runs/stacker_l1_v2/summary.tsv` |
| same, logistic variants | same | TV2 | logreg -0.0069 [-0.0091, -0.0046]; logreg_iaw -0.0054 [-0.0075, -0.0031] | DROP | same |
| L1 stacker with a TV-clean HiGO probability feature | never run (existing HiGO saw TV labels) | - | none | UNTESTED (needs L3 `pred_tv`) | `runs/stacker_l1_v2/chosen.json` (`higo_feature`) |
| ensemble with GO-GPT outputs | never run | - | none | UNTESTED (plan allows it only as a separate labelled row) | plan |

### 1.10 Calibration and grounding

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| grounded calibrator (13 features, per aspect, fit on random val) | `grounded` | C6-test | 0.5408 vs gate 0.5277 (+0.0131) | DROP for the G-test headline | `runs/grounded/summary.tsv` |
| same | `grounded_gogpt` | G-test | 0.5503 vs gate 0.5759 (-0.0256) | DROP | `runs/grounded_gogpt/summary.tsv` |
| calibrator with gate features only | same | C6-test; G-test | 0.5180 (-0.0097 vs gate); 0.5661 (-0.0098 vs gate) | DROP | same |
| domain evidence (interpro2go, 2026 release, not date-filtered) | `grounded` minus `grounded_no_domain` | C6-test | 0.5408 vs 0.5187 (+0.0221; BP +0.033) | DROP from headline numbers (leak-exposed) | `runs/grounded/summary.tsv` |
| same | same | G-test | 0.5503 vs 0.5483 (+0.0019) | same | `runs/grounded_gogpt/summary.tsv` |
| InterPro signature IDs as learned tokens (L7, no interpro2go) | never run | - | none | UNTESTED | plan, ladder L7 |
| ortholog labels + ortholog GOA in the calibrator | `grounded` minus `grounded_no_ortholog` | C6-test | 0.5408 vs 0.5413 (-0.0005) | DROP | `runs/grounded/summary.tsv` |
| same | same | G-test | 0.5503 vs 0.5649 (-0.0147); fitted ortholog-GOA weight 0.7-0.8 for MF/BP | DROP | `runs/grounded_gogpt/summary.tsv`, `runs/grounded_gogpt/calibrator.json` |
| ligand support | `grounded` minus `grounded_no_ligand` | C6-test; G-test | +0.0006; +0.0004 | DROP for accuracy (kept only for checkable MF output) | same files |
| clipped-logit kNN feature `logit(clip(p_knn, 1e-4))` | calibrator weights | C6-test, G-test | fitted weight 0.023-0.035 (C6), 0.025-0.131 (G); most values collapse to -9.21 | DROP; use raw p_knn and per-term support features | `runs/grounded/calibrator.json`, `runs/grounded_gogpt/calibrator.json` |
| output stability across 6 models | dev + 5 CV folds | C6-test | Jaccard raw 0.581, gated 0.646, grounded 0.786 | KEEP as a CAFA-6 output property (does not bear on G-test) | `runs/grounded/stability.tsv` |
| `term_bias_init: prior`, `wd_exclude: true` | enabled in L3/L5 configs, never ablated | - | none | UNTESTED | `runs/l3_c0_tv2/config.yaml` |

### 1.11 Generative refiners

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| MeanFlow one-step label refiner, best of 27 settings (prior source sigma 0.3, 500 steps, 32 samples) | `meanflow_pilot` | G-valB | 0.6183 vs per-term fusion 0.6169: +0.0014 [-0.0001, +0.0029] (300 reps); bar +0.005 | DROP | `runs/meanflow_pilot/result_valB.json`, `notes.md` |
| same, frozen config, test look #2 | same | G-test | 0.6544 vs fusion 0.6388: +0.0155 [+0.0137, +0.0175]; vs twin 0.6719 (-0.0175); vs kNN 0.6843 (-0.0299) | DROP (verdict fixed on G-valB; twin beats it on both splits) | `runs/meanflow_pilot/result_test.json`, `runs/test_looks.md` |
| MeanFlow noise source | same | G-valB | best 0.6166 (1,000 steps, S=32) | DROP | `runs/meanflow_pilot/summary.tsv` |
| number of samples S | same | G-valB | chosen config S=1 0.5921, S=8 0.6162, S=32 0.6183 | DROP sampling-based scores (a single sample is near-binary) | same |
| longer MeanFlow training | same | G-valB | sigma 0.3: 0.6183 (500) -> 0.6143 (1,000) -> 0.6125 (1,500) | over-fits 6.6k proteins in 7.9k dims | same |
| LLM reasoning layer, label-set diffusion, AR decoder of our own | not run (excluded by the plan) | - | none | UNTESTED (excluded) | plan, "What HiGO-R leaves out" |

### 1.12 Decoding / output scoring

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| GO-GPT beam vs greedy | `gogpt_cafa5` rescored | G-test | 0.6866 vs 0.6864 | reference only | `runs/bench_gogpt/summary.tsv` |
| GO-GPT ten-sample frequency (T 0.7, top-k 20) | `gogpt_sample10` | G-test | 0.6984 (+0.0118 over beam); S6 0.7141; >=50% 0.7418; 30-50% 0.4509; <30% 0.4531; no hit 0.3821. Propagating samples first: 0.6980 | KEEP as the bar to beat (measured value is below the plan's estimate of about 0.705) | `runs/gogpt_sample10/summary.tsv`, `settings.json` |
| our scores: per-term marginals after MCM, one Fmax threshold | all our rows | all | no alternative decoder of ours was run except MeanFlow sampling (§1.11) | KEEP (default) | - |
| sliding windows (1,022 residues, stride 511, max-pooled logits) vs truncation | never ablated | - | none | UNTESTED | `NOTES.md` Step 2 |

### 1.13 Selection splits (the dominant source of error so far)

| component | experiment / run | split | metric, delta (CI if available) | verdict | evidence |
|---|---|---|---|---|---|
| random validation for retrieval settings | v1 grid vs red-team test grid | G-val vs G-test | Spearman over 8 shared settings -0.542; G-val picks T10 (best G-val row `T10_k50_g1_na` 0.6010), which is the worst T on G-test (0.691 vs 0.701) | DROP | `runs/knn_sharp/chosen.json`, `runs/knn_sharp/summary.tsv` |
| random validation for fitting combiners | MeanFlow pilot, red-team blend, grounded calibrator | G-val vs G-test | every combiner fit on random val scores below sharpened kNN on G-test, except the red-team T2 blend which ties (0.700 vs 0.699) | DROP | §1.9, §1.10 |
| TV1 (1,154 proteins, no zebrafish) | L0 v1 selection, look #1 | TV1 vs G-test | TV1 picked `Tinf_k50_g1_na` (TV1 0.5185, test-like 0.5550); G-test 0.5961, minus GO-GPT -0.0905 [-0.0976, -0.0826] (1,000 reps); S6 0.6069 vs 0.7021. Spearman vs test grid -0.542 for TV1 and TV1 test-like | DROP | `runs/knn_sharp/final.json`, `final_summary.tsv`, `runs/test_looks.md` |
| TV2 (dated + duplicate-accession strata) | v2 grid | TV2 vs G-test | Spearman vs red-team test grid: TV2 -0.133, TV2-tl -0.084, TV2 >=95% bins 0.916, DA stratum 0.253, dated stratum -0.542. Gets the large effects (sharp vs flat, aspect norm) right | KEEP for effects >= 0.01; UNTESTED for smaller effects and for the 50-80% bin (4.0% of TV2 vs 18.9% of test; the merged `*|50-80` cell has 25 proteins carrying 13.4% of the TV2-tl weight) | `runs/knn_sharp/chosen_v2.json`, `logs/tv_knn.log` |

---

## 2. Open questions that one cheap experiment would settle

Ranked by expected effect on the G-test score. "Expected effect" is an estimate from the per-bin numbers above; the
bin weights come from `red_team.md` §3.1 (each +0.01 is worth 0.0025-0.0028 overall in each >=95% bin, 0.0016-0.0019
in each 50-95% bin, under 0.001 in each remote bin).

1. **Does a TV2-fit stacker that sees HiGO beat L0 in the 50-95% bins without losing the >=95% bins?** The only
   measured signal of a learned model beating kNN on G-test is in the 50-95% bins (fusion 0.5411, twin 0.5485 vs
   kNN 0.5092; <50%: 0.436-0.444 vs 0.355), and all of those models lost at >=95% because they were fit on random
   val. Experiment: when L3 writes `pred_tv`, rerun `stacker_l1_v2` (gbm+sharp and a twin-style MLP) with the HiGO
   probability as a feature, 5-fold CV on TV2, report TV2, TV2-tl and per-bin deltas. CPU, under 10 minutes. If the
   50-95% gain of about +0.03 to +0.04 holds and >=95% is not below L0, that is about +0.012 to +0.014 overall.
2. **Should the combiner be routed by identity bin?** Experiment: on TV2, compare "kNN only at >=95%, stacker below"
   against "stacker everywhere", using the same TV2 CV folds. It costs nothing extra once (1) runs and directly
   addresses the failure seen on test (fusion 0.7527 and twin 0.8062 vs kNN 0.8557 at >=95%).
3. **How much do out-of-vocabulary kNN terms add on top of any learned model?** Vocab-only kNN 0.6843 vs full-label
   kNN 0.698-0.699 on G-test (cross-script, about 0.014). Experiment: on TV2, score L0 v2 restricted to the 7,901-term
   vocabulary vs full label sets, then score "stacker on vocab terms + kNN pass-through for OOV terms". CPU only.
4. **Which TV2 criterion governs keep/drop: plain mean or test-like mean?** L0 v2 was chosen on TV2-tl; L1 v2 was
   judged on the plain mean (+0.0008, fail) while TV2-tl gives +0.0057 [+0.0028, +0.0084] (pass). Experiment: none
   beyond registering one criterion in `runs/test_looks.md` and recomputing per-bin bootstrap CIs on TV2 for the
   80-95% and 50-80% bins. If the 50-80% CI is wider than 0.02, the 50-95% decisions in L4/L5 cannot be made on TV2.
5. **Does the registered L0 v2 (`T5_k50_g0_na`, aspect norm) hold on G-test?** Aspect norm is +0.0137 on TV2-tl and
   +0.0234 on TV1 test-like, but the only G-test grid is all-norm (best 0.701). This decides whether the kNN row is
   about 0.70 or about 0.71 and therefore how much room a learned model has above it. Experiment: score it in the next
   scheduled look (after L3), not in a look of its own.
6. **Does adding the 13,305 GO-GPT val proteins to the kNN database change the >=95% bins?** Experiment: count G-test
   proteins whose best hit (by bits) or max identity rises when val proteins are added; this uses DIAMOND hits only,
   no test labels. If a few hundred proteins move into >=95%, the expected effect is in the same range as item 1.
7. **Does the adapter matter (L3)?** C1 leads C0 by +0.0100 at epoch 0 on G-val. Experiment: score `pred_tv` of both
   finished runs on TV2 (overall, TV2-tl, per bin); keep LoRA only if C1 - C0 >= 0.01. Expected G-test effect is
   small because it acts through the HiGO feature in the 50-95% bins, but it decides whether CRLA (L9) is built.
8. **Naive fallback on no-hit proteins with sharpened kNN.** kNN abstains on 250 no-hit proteins (2.9%); abstention
   raises measured Fmax (0.684 by bin formula vs 0.699). Experiment: on TV2, score L0 v2 with and without a naive or
   HiGO fallback for no-hit proteins. Effect under 0.003 but it must be reported for the claim to stand.
9. **Is the IA-weighted loss useful?** C6 val +0.0009 vs C6 test -0.0086 gated under a 4 vs 12 epoch confound.
   Experiment: matched-epoch ablation on TV2 data, 2 seeds (L6). GPU, so not cheap; listed for completeness.

---

## 3. Things that looked good on one split but not another

1. **Grounded calibrator.** C6-test +0.0131 over the gate (0.5408 vs 0.5277); G-test -0.0256 (0.5503 vs 0.5759).
   The ortholog-GOA weight (0.7-0.8 on MF/BP) learned on a random split, where homologs' GOA and the target's labels
   share an annotation period, does not hold on a temporal test (`runs/grounded*/summary.tsv`).
2. **Domain evidence.** C6-test +0.0221 (BP +0.033); G-test +0.0019. `interpro2go` is a 2026 release, so part of the
   C6 gain may be leakage (`runs/grounded*/summary.tsv`, `REPORT.md` §8.4).
3. **kNN temperature chosen on random val.** G-val ranks the 8 red-team settings with Spearman -0.542 against G-test;
   G-val prefers T10 (0.6010 for `T10_k50_g1_na`), G-test prefers T1-T5 (0.701 vs 0.691 for T10)
   (`runs/knn_sharp/chosen.json`, `red_team.md` §3.2).
4. **TV1 choice of flat kNN.** `Tinf_k50_g1_na` was the best TV1 setting (0.5185; test-like 0.5550) and scored 0.5961
   on G-test, 0.105 below the sharpened grid best (0.701) and -0.0905 [-0.0976, -0.0826] against GO-GPT. TV1's close
   hits are cross-species orthologs (same-organism best hit at 95-99.9%: TV1 13.3% vs test 87.8%), while the test's
   >=95% bins are mostly same-organism duplicate accessions (94% of the 100% bin per the `temporal_val.py`
   docstring; best hit in the same organism 0.909 per `logs/tv_knn.log`). On TV2, the dated stratum still prefers flat
   averaging (`Tinf_k50_g0` 0.489 vs `T5_k50_g0_na` 0.474) while the DA stratum prefers sharpening (0.669 vs 0.769)
   (`runs/knn_sharp/v2/summary.tsv`).
5. **Every combiner fit on random val vs zero-training kNN.** On G-valB, per-term fusion beats sharpened kNN by
   +0.069 (0.6169 vs 0.5476) and the twin by +0.077; on G-test both lose to kNN (0.6388 and 0.6719 vs 0.6843).
   The loss is at >=95% identity: kNN in those bins scores 0.6997 on G-valB and 0.8557 on G-test, so a model fit on
   G-val learns to distrust close neighbours. Red-team per-bin blend fit on G-val: 0.687 vs kNN alone 0.691; with
   kNN T2 fixed a priori the val-fit blend ties (0.700 vs 0.699) (`runs/meanflow_pilot/summary.tsv`,
   `red_team.md` §3.2).
6. **MeanFlow vs fusion, reversed direction.** G-valB +0.0014 [-0.0001, +0.0029] (drop); G-test +0.0155
   [+0.0137, +0.0175]. The verdict stays DROP: it was fixed on G-valB, and the BCE twin with the same inputs and
   network beats MeanFlow on both splits (G-valB 0.6246 vs 0.6183; G-test 0.6719 vs 0.6544)
   (`runs/meanflow_pilot/result_*.json`).
7. **L1 v2 stacker: plain TV2 mean vs TV2 test-like.** +0.0008 [-0.0016, +0.0032] on the TV2 mean (fails the +0.005
   rule) vs +0.0057 [+0.0028, +0.0084] on TV2-tl (passes). Same data, two criteria (`runs/stacker_l1_v2/summary.tsv`).
8. **L1 v1 stacker transfer.** Fit on TV1, it is -0.0068 on TV1 CV and -0.0085 on G-val (0.5853 vs 0.5938)
   (`runs/stacker_l1/per_bin.tsv`).
9. **IA-weighted loss.** Removing it: C6-val +0.0009 (looks useless); C6-test -0.0147 raw, -0.0086 gated (looks
   useful). Confounded by 4 vs 12 epochs (`runs/abl_no_ia_weight/train_log.jsonl`, `runs/eval_main/summary_all.tsv`).
10. **Pooling and query ablations.** On C6-val, mean pooling (-0.0080 @4), flat queries (-0.0023 @4) and softmax
    (-0.0005 @4) cost accuracy; on C6-test after the gate all three are within 0.0002 of the full model. The gate
    absorbs head-level effects (`runs/eval_main/summary_all.tsv`).
11. **LoRA on all layers.** C6-val -0.0017 vs LoRA q,v; C6-test gated +0.0014. Both inside the noise; neither split
    supports the 8.5x parameters (`runs/dev_35m_lora_all_r16/train_log.jsonl`, `runs/eval_main/summary_all.tsv`).
12. **HiGO's advantage depends on how much homology the split has.** C6-test (19% of proteins at >=50% identity):
    HiGO + gate 0.528 vs DIAMOND + naive 0.461 (+0.067), +0.16 on no-hit proteins. G-test (88% at >=50%): HiGO + gate
    0.5759 vs DIAMOND + naive 0.5678 (+0.008) and 0.11-0.13 below sharpened kNN; HiGO alone 0.4547
    (`REPORT.md` §4.2, `runs/bench_gogpt/summary.tsv`).
13. **HiGO's remote-homolog lead depends on the GO-GPT decoding mode.** Against beam, HiGO + gate leads at 30-50%
    (0.4802 vs 0.4398), <30% (0.4614 vs 0.3950) and is close with no hit (0.3594 vs 0.3742). Against ten-sample
    scoring the leads shrink to +0.0293 and +0.0083 and the no-hit bin is lost by 0.0227 (0.3594 vs 0.3821)
    (`runs/bench_gogpt/summary.tsv`, `runs/gogpt_sample10/summary.tsv`).
14. **GO-GPT itself.** 0.897 zero-shot on 1,500 CAFA-6 clustered-test proteins (97% of them in its training data with
    labels) vs 0.6866 on its own temporal test (`NOTES.md`, GO-GPT benchmark section).
15. **Ortholog channel: wrong baseline rather than wrong split.** +0.028 against unblended DIAMOND kNN, -0.004 against
    DIAMOND + naive on both C6-val and C6-test (`REPORT.md` §7.2).

---

## 4. Corrections to the headline facts given for this ledger

- "Zero-training sharpened kNN (T5_k50_g0_na, aspect-norm) ties/beats GO-GPT": `T5_k50_g0_na` has no G-test score.
  The G-test evidence is the red-team grid, which is all-norm: T5_k50_g0 0.701, val-selected T10_k50 0.691
  (+0.005 [-0.0015, +0.012] vs beam). Against the ten-sample bar 0.6984 the grid best is +0.003 with no CI. So
  "ties" is measured; "beats" is not significant on any measured setting.
- "Every learned model fit on random val lost to zero-training kNN on test, mostly at >=95% identity": holds for
  the MeanFlow-pilot fusion, twin and MeanFlow, the red-team T10 blend, the gate and the grounded calibrator. The
  red-team blend with kNN T2 fixed a priori ties (0.700 vs 0.699). The MeanFlow-pilot kNN was vocabulary-only
  (0.6843); against full-label kNN (0.698-0.699) the gap would be larger by about 0.014.
- The other headline facts check out against the files: L0 v1 `Tinf_k50_g1_na` 0.5961 on G-test (look #1); MeanFlow
  +0.0014 on G-valB vs bar 0.005, BCE twin better; TV1 disagreement explained by duplicate accessions; HiGO alone
  0.4547 on G-test.

## 5. Update 2026-10-12 22:50: TV v2 decisions and test look #3

All numbers are cross-fitted TV v2 test-like mean Fmax unless marked. The reference is raw sharpened kNN over train +
val references, at 0.7259. Rules are in `runs/test_looks.md`.

| component | evidence | verdict | files |
|---|---|---|---|
| kNN references = train + val (GO-GPT val proteins added) | +0.0091 [+0.0070, +0.0110]; +0.0036 [+0.0020, +0.0051] with weights from the new identities; 599 test proteins (6.9%) get a closer hit | KEEP | `runs/knn_trainval/` |
| Twin stacker on frozen-35M HiGO (l3_c0_tv2) | 0.7342, +0.0083 [+0.0040, +0.0120] | KEEP (chosen pipeline) | `runs/stacker_tv/l3_c0/result.json` |
| Identity-gated twin on l3_c0 | 0.7326, +0.0067 [+0.0027, +0.0103] | kept by rule, not chosen (lower) | same |
| Per-term logistic fusion (any input) | +0.0000 (lofo picks step 0) | DROP | `runs/stacker_tv/*/result.json` |
| LoRA 35M (l3_c1_tv2) through the stacker | best gated twin 0.7287, +0.0028 [-0.0013, +0.0064] | DROP (adapter) | `runs/stacker_tv/l3_c1/result.json` |
| In-network kNN fusion (l4a_tv2) through the stacker | best 0.7275, +0.0016 [-0.0021, +0.0050] (random-val lead of +0.06 did not transfer) | DROP | `runs/stacker_tv/l4a/result.json` |
| Frozen ESM2-3B, layer mix 18/22/26/30 (l5_3b_tv2) | best gated twin 0.7306, +0.0047 [+0.0008, +0.0084], under the +0.005 bar | DROP | `runs/stacker_tv/l5_3b/result.json` |
| Isotonic calibration on the l3_c0 twin | 0.7287 vs 0.7342 uncalibrated | DROP | `runs/calibrate_tv/l3_c0_iso/result.json` |
| kNN + isotonic (train + val) | 0.7290, +0.0031 [+0.0016, +0.0042] | kept by rule, not chosen (more than 0.003 below the twin) | `runs/calibrate_tv/knn_trainval_iso/result.json` |
| **Test look #3**: l3_c0 twin pipeline on G-test | 0.7178 vs GO-GPT ten-sample 0.6984 (+0.0193 [+0.0138, +0.0250]) and beam 0.6866 (+0.0312 [+0.0254, +0.0374]); loses <30% (0.451 vs 0.461) and no hit (0.342 vs 0.382) | **beats the bar** | `runs/test_look/look3_l3c0_twin/` |
| Optimiser: Schedule-Free vs WSD + tail averaging | 2 seeds each, running; rule 22:55:00 | OPEN | `logs/opt_*.log` |

**What the split mismatch looked like this time:** on random val, in-network fusion led the plain head by
+0.06 and 3B led 35M by +0.01 to +0.04. On TV v2, neither improved the stacked pipeline over the 35M HiGO feature.
The random-split lesson from section 3 held again: random val rewards retrieval-like gains that the stacker, given the
kNN prior directly, already captures.

### 5.1 Update 2026-10-18: optimiser head-to-head

| component | evidence (TV v2 test-like, twin stacker) | verdict | files |
|---|---|---|---|
| AdamW Schedule-Free (NeurIPS 2024), 2 seeds | 0.7271, 0.7233 | DROP (no gain; seeds disagree) | `runs/stacker_tv/opt_sf_s*/result.json` |
| AdamW WSD + tail averaging (NeurIPS 2024 / SWAD NeurIPS 2021), 2 seeds | 0.7283, 0.7182 | DROP | `runs/stacker_tv/opt_wsd_s*/result.json` |
| Seed spread of the stacked 35M pipeline (5 runs) | twin 0.7182-0.7342, SD about 0.006; only l3_c0 passed its rule | the l3_c0 TV gain is the top of the spread | same, plus `runs/stacker_tv/l3_c0/` |

Look #4 was not used. The look-3 pipeline is final.

### 5.2 Update 2026-10-18: full evaluation suite (`runs/eval_suite/`)

| observation | evidence (G-test, descriptive) | consequence |
|---|---|---|
| The stacker does not beat its own kNN prior on test | kNN prior (train+val) Fmax_w 0.7206 vs pipeline 0.7178; MF 0.758 vs 0.738, BP 0.649 vs 0.658, CC 0.755 vs 0.758 | the TV v2 gain of the twin (+0.0083) did not transfer; the GO-GPT margin comes from the retrieval prior |
| Novel proteins | per-protein F1_w no hit 0.133 vs GO-GPT 0.286; <30% 0.335 vs 0.352; 30-50% 0.341 vs 0.375 | HiGO-R2 is the weaker model for proteins without close homologues |
| Homology dependence | Spearman(F1_w, identity) 0.523 vs 0.431, difference +0.092 [+0.075, +0.108] | same dependence as the kNN prior (0.528) |
| Hierarchy consistency | 0 violations vs GO-GPT ten-sample 2.87% of pairs | KEEP the max-propagation step |
| GO term embeddings | direct-neighbour P@1 0.678 (HiGO queries) vs 0.384 untrained ancestor-mean vs 0.154 GO-GPT tokens | the learned queries encode GO structure beyond the ancestor-mean construction |
| Multiple testing | 51/56 primary comparisons pass Holm; none significant in favour of GO-GPT | the headline holds after correction |

### 5.3 Update 2026-10-19: Day 1 analyses on TV v2 (`runs/day1_analyses/`, exploratory, no test scoring)

| question | evidence | verdict |
|---|---|---|
| Are partial-coverage hits a major error source at 50-95% identity (premise of residue-level alignment-conditioned retrieval)? | Low query coverage costs recall more than precision (kNN precision -0.08, recall -0.12 at <50% coverage); those proteins hold 19.4% of the band's false-positive IA mass for 24.2% of its weight; removing all their false positives gains at most +0.011 (Ontologue) | minor; DROP idea 2 as a precision fix; if revisited, test region-wise retrieval for recall with an oracle first |
| Do abstention fallbacks help? | Abstention 1.6% MF, 3.2% BP, 0.2% CC of protein-aspect pairs; fallbacks +0.0008 and +0.0010 test-like (CIs exclude 0) | below the 0.003 bar; DROP as a work item |

### 5.4 Update 2026-10-25: fresh24 looks #1 and #2 (`runs/fresh24/`, `runs/fresh24_look2/`)

| observation | evidence (FR24-P, 11,231 no-knowledge proteins, annotations added after 2024-02) | consequence |
|---|---|---|
| SPROF-GO is the best method on fresh data | IA-weighted Fmax mean SPROF-GO 0.515, Ontologue 0.479 (diff -0.035 [-0.040, -0.030], Holm p 0.015); SPROF-GO ahead in every bin below 95% identity | the GO-GPT-test ranking does not hold on a benchmark with less close homology; Ontologue's claim is limited to homology-rich settings |
| Methods built on GO-GPT's data drop most | Ontologue 0.718 -> 0.479, GO-GPT ten-sample 0.698 -> 0.468, BLAST-KNN 0.695 -> 0.372; external tools stable (SPROF-GO 0.541 -> 0.515, DeepGOPlus 0.428 -> 0.466) | the GO-GPT test overstates retrieval-based methods (54% of it at >=95% identity vs 12% of FR24-P) |
| Ontologue still above GO-GPT | +0.012 [+0.007, +0.017] over ten-sample, +0.020 over beam (Holm p 0.015) | the comparison with GO-GPT holds on fresh data, with a smaller margin |
| Forecasting training helps where retrieval is weak | F1_ens 0.487 vs Ontologue 0.479 (+0.008 [+0.004, +0.012], p_Holm 0.008); gains at no hit +0.022, <30 +0.024, 30-50 +0.013, 50-80 +0.017; losses at 95-99.9 -0.024, 100 -0.066 | KEEP idea 1 as the research direction; a model that uses F1 below ~95% identity and Ontologue above is the obvious next candidate, but it must be selected on forecast-valid / TV data, not on fresh24 |
| Forecasting model still below SPROF-GO | -0.027 [-0.032, -0.022] | the language-model branch (35M) is the bottleneck for distant proteins; SPROF-GO uses ProtT5-XL; a larger encoder is the next lever |

### 5.5 Update 2026-10-25: forecasting round 2 and fresh24 look #3 (`runs/forecast/round2/`, `runs/fresh24_look3/`)

| observation | evidence | consequence |
|---|---|---|
| Larger frozen encoder helps the forecasting model | round-2 selection (A+B valid, all rows): F1-650M 0.2983 vs F1-35M 0.2947; FR24-P: 0.514 vs 0.487 (+0.026 [+0.023, +0.030]) | KEEP the 650M encoder; ESM2-3B is the next step |
| Routing to the kNN prior at high identity does not help | every routed variant below its unrouted model on A+B valid (e.g. b+route98 0.2896 vs b 0.2983) | DROP identity routing; the forecasting model already handles close homologues |
| Forecasting model ties SPROF-GO on fresh data | FR24-P IA-weighted Fmax 0.514 vs 0.515 (-0.001 [-0.006, +0.005], ns); SPROF-GO better on Fmax (0.015) and S-min (0.39); F1-650M ahead at >= 50% identity, behind below 50%; on all no-knowledge proteins 0.529 vs 0.524 | the learned model now adds what retrieval misses; the remaining gap is distant proteins and calibration (S-min) |
| Forecasting model beats Ontologue and GO-GPT on fresh data | +0.034 over Ontologue, +0.046 over GO-GPT ten-sample (Holm p 0.005) | candidate to replace the served model once the last fresh24 look is spent on a final confirmation |
