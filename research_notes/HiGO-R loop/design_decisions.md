# HiGO-R design decisions, one per niche, with venue and code checks

Date: 2026-10-12, about 08:10. Scope: the eleven niches of the HiGO-R code (`src/cafa6/model.py`, `train.py`, `data.py`,
`knn_sharp.py`, `stacker.py`, `temporal_val.py`, `evaluate.py`). No code was edited. Companion to `observations.md`
in this folder (O-numbers below refer to it).

Target. Test (GO-GPT) S3_all, IA-weighted Fmax, mean of MF/BP/CC. The bar is GO-GPT ten-sample frequency, **0.6984**
(`runs/gogpt_sample10/summary.tsv`), not beam 0.6866. Red-team bootstrap half-width is about 0.007, so a claim needs
about 0.708 or more and must beat the sharpened kNN row (0.690-0.701 on test) with a paired bootstrap.

Conventions.
- *Venue checked* = venue confirmed in this session on openreview.net, proceedings.mlr.press or proceedings.neurips.cc
  (or the conference's own virtual site / proceedings.iclr.cc when OpenReview showed a browser check). *Known* =
  venue from prior notes, not re-fetched today. Only checked papers carry a decision; known papers appear only as
  secondary support.
- *Repo checked* = repository and file path confirmed today through the GitHub API or the installed package.
  License is the repository license.
- All TV numbers are TV v2 (`data/proc_gogpt_tv2`, 7,566 proteins). **Registered criterion for every kill rule
  below:** primary = TV v2 test-like mean (`knn_sharp.test_like_weights`), guardrail = TV v2 plain mean must not drop
  by more than 0.002, and per-bin rows are reported. This settles open question 3 of `observations.md` and must be
  logged before the next TV decision. Deltas are paired bootstrap, 500 reps, over proteins.
- Noise yardsticks: seed std 0.004-0.006 on a single model (O13); TV v2 test-like paired CI half-width about 0.003
  (L1 v2: +0.0057 [+0.0028, +0.0084]). A single-seed effect below 0.005 is not interpretable.

---

## 0. State of the GPUs and runs at the time of writing

| run | GPU | status (val random GO-GPT, fast metric `val_Fw_mean`) |
|---|---|---|
| `l3_c0_tv2` frozen 35M, full head | 0 | epochs 0-4: 0.4931, 0.5211, 0.5344, 0.5415, 0.5460; about 20 min/epoch |
| `l3_c1_tv2` LoRA q,v r8 35M | 0 | epochs 0-3: 0.5031, 0.5296, 0.5394, 0.5484; about 25 min/epoch |
| `l5_3b_tv2` frozen 3B, layers 30, bf16, mix 18/22/26/30 (uniform init) | 1 | epoch 0: **0.5379** (C0 epoch 0: 0.4931); about 44 min/epoch while sharing GPU 1 |
| `l4a_tv2` frozen 35M + in-network kNN fusion | 1 | epochs 0-1: 0.6106, 0.6227; about 20 min/epoch |

Both GPUs are busy (GPU 0: 18.4 GB, GPU 1: 16.7 GB, 100% util). Random-val numbers are only used to watch
convergence; random val ranks homology-related choices in reverse (O30, O42) and decides nothing below.
Free disk: 15 GB. `/dev/shm`: 126 GB; RAM available about 196 GB.

---

## 1. Verified sources

### 1.1 Papers (venue checked today)

| short name | title | venue | link |
|---|---|---|---|
| Tenney19 | What do you learn from context? Probing for sentence structure in contextualized word representations | ICLR 2019 | [openreview HkAEIaFCX refs](https://openreview.net/references/pdf?id=HkAEIaFCX) |
| Skean25 | Layer by Layer: Uncovering Hidden Representations in Language Models | ICML 2025 (oral) | [PMLR v267 skean25a](https://proceedings.mlr.press/v267/skean25a.html) |
| Li24 | Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models | ICML 2024 | [PMLR v235 li24a](https://proceedings.mlr.press/v235/li24a.html) |
| SaProt | SaProt: Protein Language Modeling with Structure-aware Vocabulary | ICLR 2024 | [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/file/1c42513b8895ab11fbbb5b7e8e6b6b02-Paper-Conference.pdf) (fetched in `tier1_design_ideas.md`) |
| LoRA | LoRA: Low-Rank Adaptation of Large Language Models | ICLR 2022 | [openreview nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9), [ICLR virtual](https://iclr.cc/virtual/2022/poster/6319) |
| DoRA | DoRA: Weight-Decomposed Low-Rank Adaptation | ICML 2024 | [PMLR v235 liu24bn](https://proceedings.mlr.press/v235/liu24bn.html) |
| LP-FT | Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution | ICLR 2022 (oral) | [openreview UYneFzXSJWh](https://openreview.net/pdf?id=UYneFzXSJWh) |
| LST | LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning | NeurIPS 2022 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html) (fetched in `adapters_trla.md`) |
| Sparsemax | From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification | ICML 2016 | [PMLR v48 martins16](https://proceedings.mlr.press/v48/martins16.html) |
| SetTF | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | ICML 2019 | [PMLR v97 lee19d](https://proceedings.mlr.press/v97/lee19d.html) |
| C-HMCNN | Coherent Hierarchical Multi-Label Classification Networks | NeurIPS 2020 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6dd4e10e3296fa63738371ec0d5df818-Abstract.html) |
| kNN-LM | Generalization through Memorization: Nearest Neighbor Language Models | ICLR 2020 | [openreview HklBjCEKvH](https://openreview.net/attachment?id=HklBjCEKvH&name=original_pdf) |
| RETRO | Improving Language Models by Retrieving from Trillions of Tokens | ICML 2022 | [PMLR v162 borgeaud22a](https://proceedings.mlr.press/v162/borgeaud22a.html) |
| MemTF | Memorizing Transformers | ICLR 2022 (spotlight) | [openreview TrjbxzRcnf-](https://openreview.net/pdf?id=TrjbxzRcnf-) |
| TabR | TabR: Tabular Deep Learning Meets Nearest Neighbors | ICLR 2024 | [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/file/4ef594af0d9a519db8fb292452c461fa-Paper-Conference.pdf) (fetched in `tier1_design_ideas.md`) |
| LogitAdj | Long-tail learning via logit adjustment | ICLR 2021 | [openreview 37nvvqkCo5](https://openreview.net/references/pdf?id=jTKabwkHS), [ICLR virtual](https://iclr.cc/virtual/2021/poster/2675) |
| Koyejo15 | Consistent Multilabel Classification | NeurIPS 2015 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2015/hash/85f007f8c50dd25f5a45fca73cad64bd-Abstract.html) |
| Koyejo14 | Consistent Binary Classification with Generalized Performance Metrics | NeurIPS 2014 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2014/hash/98053046e0dce5c7d946c67b96a85e18-Abstract.html) |
| AdamW | Decoupled Weight Decay Regularization | ICLR 2019 | [openreview Bkg6RiCqY7](https://openreview.net/pdf/5963886abef941684ffc0cf670297e47fb1e5155.pdf) |
| SAM | Sharpness-Aware Minimization for Efficiently Improving Generalization | ICLR 2021 (spotlight) | [openreview 6Tm1mposlrM](https://openreview.net/forum?id=6Tm1mposlrM) |
| ASAM | ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks | ICML 2021 | [PMLR v139 kwon21b](https://proceedings.mlr.press/v139/kwon21b.html) |
| FlatMin | When Do Flat Minima Optimizers Work? | NeurIPS 2022 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html) |
| SWAD | SWAD: Domain Generalization by Seeking Flat Minima | NeurIPS 2021 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html) |
| WSD | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations | NeurIPS 2024 (spotlight) | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8b970e15a89bf5d12542810df8eae8fc-Abstract.html) |
| Soups | Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time | ICML 2022 | [PMLR v162 wortsman22a](https://proceedings.mlr.press/v162/wortsman22a.html) |
| LMC | Linear Mode Connectivity and the Lottery Ticket Hypothesis | ICML 2020 | [PMLR v119 frankle20a](https://proceedings.mlr.press/v119/frankle20a.html) |
| GenMeas | Fantastic Generalization Measures and Where to Find Them | ICLR 2020 | [openreview SJgIPJBFvH](https://openreview.net/pdf?id=SJgIPJBFvH) |
| TempScale | On Calibration of Modern Neural Networks | ICML 2017 | [PMLR v70 guo17a](https://proceedings.mlr.press/v70/guo17a.html) |
| EnsSel | Ensemble Selection from Libraries of Models | ICML 2004 | [ACM DL (ICML '04)](https://dl.acm.org/doi/10.1145/1015330.1015432); ICML 2004 predates PMLR, the ACM proceedings record is the official one |
| DeepEns | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | NeurIPS 2017 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) |
| LightGBM | LightGBM: A Highly Efficient Gradient Boosting Decision Tree | NeurIPS 2017 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) |
| FA1 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS 2022 | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) |
| FA2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | ICLR 2024 | [openreview mZn2Xyh9Ec](https://openreview.net/forum?id=mZn2Xyh9Ec), [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/file/98ed250b203d1ac6b24bbcf263e3d4a7-Paper-Conference.pdf) |
| MixedPrec | Mixed Precision Training | ICLR 2018 | [openreview r1gs9JgRZ](https://openreview.net/pdf?id=r1gs9JgRZ) |
| ProSST | ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention | NeurIPS 2024 | [proceedings.neurips.cc PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf) (fetched in `tier1_design_ideas.md`) |

Known, not re-fetched (secondary support only): SGDR (ICLR 2017), LDAM (NeurIPS 2019), Transformer bucketing
(Vaswani et al., NeurIPS 2017), Edge of Stability (Cohen et al., ICLR 2021), loss-landscape visualisation (Li et al.,
NeurIPS 2018), selective classification (Geifman & El-Yaniv, NeurIPS 2017), auto-sklearn (NeurIPS 2015),
sparse and continuous attention with alpha-entmax (Martins et al., NeurIPS 2020).

Correction to the brief: "calibrated per-term marginals are Fmax-optimal" is Koyejo et al., *Consistent Multilabel
Classification*, NeurIPS 2015 (confirmed). The binary-metric result that it builds on is NeurIPS 2014.

Not admissible under the venue rule (used nowhere as justification): entmax-1.5 itself (Peters, Niculae, Martins,
ACL 2019), ASL (ICCV 2021), distribution-balanced loss (ECCV 2020), SWA (Izmailov et al., UAI 2018), torch.compile /
PyTorch 2 (ASPLOS 2024), Platt scaling (1999 book chapter), isotonic regression calibration (KDD 2002).

### 1.2 Code (repo checked today)

| repo / package | license | file and function used |
|---|---|---|
| huggingface/transformers (installed 5.17) | Apache-2.0 | `models/esm/modeling_esm.py` `EsmModel` (`_supports_sdpa = True`), `output_hidden_states`; in-repo `model.build_backbone`, `model.esm_core` |
| allenai/allennlp | Apache-2.0 | `allennlp/modules/scalar_mix.py` `ScalarMix`; in-repo `model.ScalarMix` |
| huggingface/peft (installed 0.21) | Apache-2.0 | `peft/tuners/lora/` `LoraConfig(r, target_modules, use_dora)`; `use_dora` field confirmed in the installed package |
| NVlabs/DoRA | NVIDIA Source Code License (non-commercial) | reference only; use peft `use_dora` |
| ylsung/Ladder-Side-Tuning | MIT | side-network and downsampler reference |
| deep-spin/entmax (installed 1.3) | MIT | `entmax/activations.py` `sparsemax`, `entmax15`; `entmax/root_finding.py` `entmax_bisect` (both paths confirmed locally); in-repo `model.entmax15_bisect_fused` |
| juho-lee/set_transformer | MIT | `modules.py` `PMA`, `MAB` |
| EGiunchiglia/C-HMCNN | **GPL-3.0** | `main.py` `get_constr_out`, `ConstrainedFFNNModel`, MCLoss inline in the train loop. Do not copy; the in-repo re-implementation `HiGO.mcm` / `HiGO.loss` stays |
| urvashik/knnlm | MIT | `fairseq/knnlm.py` `KNN_Dstore.get_knn_log_prob` (log-space interpolation with a validation-tuned lambda) |
| yandex-research/tabular-dl-tabr | MIT | `bin/tabr.py` (TabR model; 19 KB file confirmed), `bin/neighbors.py` |
| lucidrains/RETRO-pytorch | Apache-2.0 | reference only (rejected on cost) |
| lucidrains/memorizing-transformers-pytorch | MIT | reference only (rejected on cost) |
| google-research/google-research | Apache-2.0 | `logit_adjustment/main.py`, `logit_adjustment/utils.py` (logit-adjusted loss and post-hoc adjustment) |
| kaidic/LDAM-DRW | MIT | repo confirmed; `losses.py` `LDAMLoss` per README, file not opened |
| davda54/sam | MIT | `sam.py` `SAM(params, base_optimizer, rho, adaptive=True)`, `first_step`, `second_step`. `adaptive=True` is ASAM; SamsungLabs/ASAM returned 404 |
| khanrc/swad | MIT | `domainbed/swad.py` `LossValley`, `IIDMax`; `domainbed/lib/swa_utils.py` `AveragedModel` |
| epfml/schedules-and-scaling | MIT | `src/optim/weight_averaging.py`; schedule in `src/optim/utils.py` |
| mlfoundations/model-soups | MIT | `main.py` (uniform and greedy soup), `utils.py` (root listing confirmed) |
| pytorch (installed 2.13) | BSD-3 | `torch.optim.swa_utils.AveragedModel`, `get_ema_multi_avg_fn`, `update_bn` (confirmed locally); `torch.nn.functional.scaled_dot_product_attention` |
| gpleiss/temperature_scaling | MIT | `temperature_scaling.py` `ModelWithTemperature.set_temperature` |
| scikit-learn (installed 1.9.1) | BSD-3 | `sklearn.ensemble.HistGradientBoostingClassifier` (LightGBM-style histogram GBDT); in-repo `stacker.AspectModel`, `stacker.features` |
| automl/auto-sklearn | BSD-3 | `autosklearn/ensembles/ensemble_selection.py` `EnsembleSelection` (file confirmed) |
| Dao-AILab/flash-attention | BSD-3 | installed `flash_attn` 2.8.3 fails to import (ABI mismatch); not used |
| tomgoldstein/loss-landscape | MIT | repo confirmed; plotting entry point not opened |
| westlake-repl/SaProt | MIT | reference for niche 11 |

---

## 2. Decisions by niche

Each niche follows the same template: **Decision**, **Fallback**, **Paper**, **Code**, **Expected effect and local
evidence**, **Cost**, **TV v2 experiment**, **Kill rule**.

### Niche 1. PLM backbone and layer readout

- **Decision.** Frozen ESM2-3B, truncated after layer 30, bf16 weights, learned scalar mix over layers
  18/22/26/30 with uniform initialisation (`l5_3b_tv2.yaml`, `readout: mix`, `mix_top_init: 0.0`).
- **Fallback.** Frozen ESM2-650M (cached, 2.5 GB, 33 layers) with mix over layers 20/24/28/32 (about 60-95% depth).
  Last resort: 35M, the current L3 C0 head.
- **Paper.** Tenney19 (task-specific scalar mix of all layers beats concatenating the top layer by 1-3 F1 on frozen
  encoders). Skean25 (intermediate layers beat the final layer by up to 16% on 32 embedding tasks). Li24 (protein
  transfer mostly uses early and intermediate features). SaProt (ESM2 35M to 650M gives +0.054/+0.057/+0.066
  MF/BP/CC Fmax on DeepFRI GO).
- **Code.** HF `EsmModel` with hidden states truncated in `model.build_backbone(plm_layers=30)`; mix in
  `model.ScalarMix`, which is the allennlp `ScalarMix` pattern (softmax-normalised scalars times a global gamma).
- **Expected effect and local evidence.** For HiGO alone, epoch 0 on random val is 0.5379 against 0.4931 for 35M
  frozen, +0.045 (O19 and the table in section 0). Converged, expect +0.03 to +0.05 for HiGO alone. After fusion with
  kNN, the gain is confined to the bins where HiGO carries weight (50-95% identity, 34.6% of test; remote bins 11.9%).
  Using the red-team bin values (+0.01 in a bin is worth 0.0016-0.0019 overall in 50-95% and below 0.001 remotely),
  this gives +0.005 to +0.015 overall (inference). GO-GPT reports 650M about equal to 3B, so 650M is a credible
  fallback.
- **Cost.** 6.9 GiB and 1.5 s/step measured; observed 44 min/epoch while sharing GPU 1, so 8 epochs (patience 2) take
  about 4-6 GPU-h. No extra disk.
- **TV v2 experiment.** `l5_3b_tv2` against `l3_c0_tv2` (same head, frozen). Score `pred_tv_all_top500.npz` on TV v2:
  HiGO alone and HiGO through the niche-9 stacker, overall, test-like and per bin. Log the learned mix weights from
  `train.diagnostics`.
- **Kill rule.** Keep 3B if HiGO-alone test-like is at least +0.02 over C0 *and* the stacked pipeline gains at least
  +0.005 test-like. If HiGO gains but the stack does not, switch to 650M (cheaper, same readout). If the mix collapses
  to more than 90% on one layer, freeze the readout to that single layer for later runs and for the feature cache
  (niche 10).

### Niche 2. Parameter-efficient adapter

- **Decision.** No adapter (frozen encoder) as the default for 3B. On 35M, keep the L3 arm that wins under the rule
  below.
- **Fallback.** LoRA on q,v with r=8 (peft) if L3 passes the rule; for 3B only as LoRA on the top 6 kept layers, or a
  Ladder Side-Tuning side network if the top-layer LoRA does not fit the budget.
- **Paper.** LoRA (ICLR 2022); LP-FT (ICLR 2022: fine-tuning can distort pretrained features and lose OOD accuracy);
  LST (NeurIPS 2022: the backbone stays in no-grad mode and a small side network reads its intermediate states);
  DoRA (ICML 2024) as the rejected variant.
- **Code.** peft `LoraConfig(target_modules=["query","value"], r=8)` through `model.build_backbone`; DoRA via
  `use_dora=True`; LST reference in ylsung/Ladder-Side-Tuning.
- **Expected effect and local evidence.** On 35M random val, C1 leads C0 by +0.0100 at epoch 0 and by +0.0069 at
  epoch 3 (0.5484 vs 0.5415). The lead is shrinking. DoRA ties LoRA (O16); LoRA r16 on all layers does not help
  (O17). Expected on TV after fusion: 0 to +0.005. The LP-FT argument predicts the adapter helps most on in-distribution
  close homologs, where kNN already dominates.
- **Cost.** C1 takes 25 min/epoch against 20 for C0 on 35M. For 3B, LoRA through all layers costs about 2.4-2.5
  GPU-h/epoch (O18) against about 0.7 for frozen, roughly 3.5x.
- **TV v2 experiment.** L3 as running. Score both `pred_tv` files alone and stacked. Add a second seed per arm only if
  the first-seed difference is between 0.005 and 0.015 test-like.
- **Kill rule.** Keep LoRA only if C1 − C0 ≥ 0.01 test-like on the stacked pipeline (registered L3 rule) with both
  seeds agreeing in sign. Otherwise the default is frozen everywhere, and CRLA (L9) is not built.

### Niche 3. Per-term pooling over residues

- **Decision.** Keep per-term query attention pooling: GO-term queries attend over residues, with a sparse normaliser.
  Treat the alpha of the alpha-entmax family as a hyperparameter. The technique is justified by sparsemax, which is
  alpha=2 in the same family. Keep the current alpha=1.5 as the default, because O9 shows a tie with softmax and the
  evidence maps depend on sparsity.
- **Fallback.** Softmax pooling (alpha=1). If a venue-clean single setting is required, use alpha=2 (sparsemax).
- **Paper.** SetTF, ICML 2019 (pooling by multihead attention with learned seed queries, one query per output, here
  one per GO term). Sparsemax, ICML 2016 (sparse attention normaliser, introduced together with a multi-label
  loss). Entmax-1.5 itself is ACL 2019 and is not used as justification.
- **Code.** set_transformer `modules.py` `PMA`; entmax package `entmax_bisect(alpha=...)` and `sparsemax`; in-repo
  `HiGO._entmax` and `entmax15_bisect_fused`.
- **Expected effect and local evidence.** O8: GO-conditioned pooling beats mean pooling on val (−0.005 to −0.008
  without it), with no gain left after the gate on test. O9: entmax and softmax tie (−0.0005 val, 0.000 gated test).
  Expect alpha=2 against alpha=1.5 to land at 0 ± 0.003.
- **Cost.** Zero if left as is. One 35M arm is about 3-4 GPU-h; on the 3B feature cache (niche 10) about 0.5 GPU-h.
- **TV v2 experiment.** Only on the 3B cache: alpha in {1.5, 2.0} with the same seed, stacked pipeline.
- **Kill rule.** Change the default only if the difference is at least 0.005 test-like. Otherwise keep alpha=1.5 for
  the residue maps (39% non-zero, O14).

### Niche 4. Label hierarchy and consistency

- **Decision.** Keep the max-constraint module (MCM) on the output and MCLoss in training, applied after the retrieval
  logit is added, so the fused output is closed under true-path propagation. Keep the ancestor-mean queries.
- **Fallback.** Post-hoc max-propagation over the GO DAG on the final scores (the cafaeval `prop=max` path), which
  gives the same closure without MCLoss.
- **Paper.** C-HMCNN, NeurIPS 2020 (constraint layer plus MCLoss; coherent outputs without post-processing).
- **Code.** EGiunchiglia/C-HMCNN `main.py` `get_constr_out` (GPL-3.0, reference only); in-repo `HiGO.mcm` and
  `HiGO.loss` (MCLoss, IA-weighted).
- **Expected effect and local evidence.** O10 is the only head component whose gain survives the gate on test (−0.004
  gated and −0.012 on val at convergence when removed), with zero ontology violations. It also keeps the kNN branch
  and the HiGO branch on the same closed label sets, which the stacker needs.
- **Cost.** Already in the model; no GPU-h.
- **TV v2 experiment.** None new. Report ontology violations of the final stacked output, which must be 0, since
  stacking and calibration can break closure.
- **Kill rule.** If the stacker output has violations, apply the fallback max-propagation after calibration. Never drop
  MCM from the network.

### Niche 5. Retrieval conditioning inside the network

- **Decision.** Use kNN-LM-style logit-space interpolation with per-term learned trust, as in the current
  `knn_fusion`:
  \(z_t = z^{HiGO}_t + \lambda_t\,\mathrm{logit}(\mathrm{clip}(p^{kNN}_t)) + g_t\phi_t\), with
  \(\lambda_t = \mathrm{softplus}(a_{asp} + \delta_t + \beta\cdot\mathrm{rbf}(\mathrm{max\_id}))\), set to zero when
  there is no hit. The prior is the registered sharpened kNN (`T5_k50_g0_na`) on full propagated label sets.
  Training follows TabR's leakage rule: self excluded, train-vs-train DIAMOND hits, identity caps sampled per batch
  (`knn_caps`) so that the network sees the full identity range. Then, as in kNN-LM, where lambda is tuned on held-out
  data, **refit only the trust parameters** \((a_{asp}, \beta)\) on TV v2 with everything else frozen.
- **Fallback.** (b) A TabR-style attention block over the top-k labelled neighbours (value = label embedding plus
  key-difference term) feeding a per-term prior. (c) If both fail, no in-network retrieval: kNN enters only through
  the niche-9 stacker. RETRO and Memorizing Transformers are rejected: chunked cross-attention or a per-layer kNN
  memory needs grad through a 3B encoder or a new datastore of residue states, which the disk budget cannot hold.
- **Paper.** kNN-LM, ICLR 2020 (interpolate a parametric model with a kNN distribution; lambda tuned on validation).
  TabR, ICLR 2024 (retrieval attention with label values; the query is excluded from its own context during training).
  RETRO, ICML 2022 and MemTF, ICLR 2022 (considered, rejected on cost).
- **Code.** urvashik/knnlm `fairseq/knnlm.py` `KNN_Dstore.get_knn_log_prob`; yandex-research/tabular-dl-tabr
  `bin/tabr.py`; in-repo `HiGO.retrieval_logit`, `train.knn_bank`, `Batcher.retrieval`, `param_groups`
  (`knn_delta_wd: 1.0`).
- **Expected effect and local evidence.**
  - Zero-training sharpened kNN is 0.690-0.701 on test and 0.856 in the ≥95% bins, against 0.806 for the best
    learned combiner fit on random val (O29, O42). The network must therefore not override kNN at high identity.
  - The per-term logistic fusion lost to kNN, and the nonlinear twin beat that fusion by +0.033 on test (O43). This
    supports learned per-term trust.
  - L4a random-val curve: 0.6106 → 0.6227 against 0.4931 → 0.5211 for C0. That gap mostly measures kNN on random
    val and is not evidence for test.
  - Expected gain over the best of {kNN, stacker without HiGO}: +0.005 to +0.015 overall, concentrated in 50-95%,
    where GO-GPT beam leads kNN (0.469 vs 0.409 at 50-80%, 0.657 vs 0.640 at 80-95%).
  - Without the TV refit, the risk is the same as in O42: trust learned on train-vs-train hits may be too low at
    duplicate-accession identity.
- **Cost.** 35M: about 20 min/epoch, 4-5 GPU-h to convergence (running). On 3B: same as niche 1 plus about 10%.
  The TV refit of \((a, \beta)\) is a CPU job of minutes, but needs `predict` to save \(z^{HiGO}\) and the kNN logit
  separately (code change, not made here).
- **TV v2 experiment.** Compare `l4a_tv2` with {kNN `T5_k50_g0_na`, L1 stacker v2, C0 + stacker}, on TV v2 test-like,
  overall and per bin. Arm L4a-refit: trust parameters refit on TV with 5-fold CV over proteins.
- **Kill rule.** Keep L4a only if it beats the best non-network baseline by at least 0.005 test-like, and in neither
  ≥95% bin is it more than 0.005 below kNN. If the ≥95% bins lose and refit does not repair them, switch to fallback
  (c). TabR attention (b) is tried only if L4a passes the 50-95% bins but fails overall.

### Niche 6. Loss for an extreme, long-tailed multi-label output and IA weighting

- **Decision.** Use per-term binary cross-entropy through MCLoss, with constant per-term IA weights \(1 + \min(IA, 5)\)
  normalised to mean 1, and the bias initialised to the log-odds of the term prior (`term_bias_init: prior`). The
  argument is that a proper per-label loss with constant per-label weights keeps the same population minimiser (the
  calibrated marginal). It stays Fmax-consistent, which is the plug-in condition in Koyejo15, and only moves capacity
  toward informative terms.
- **Fallback.** Unweighted MCLoss (`ia_weight: false`) if the L6 ablation shows harm.
- **Paper.** Koyejo15, NeurIPS 2015 (for instance-averaged multilabel metrics, the optimal classifier thresholds
  per-label marginals). LogitAdj, ICLR 2021, and LDAM (NeurIPS 2019, known) are tested only as ablations. They are
  consistent for *balanced* error, so they shift marginals away from calibration, which Fmax with one shared
  threshold does not reward.
- **Code.** In-repo `HiGO.loss`; google-research `logit_adjustment/utils.py` for the adjusted loss (tau·log prior added
  to logits in training, removed at test); LDAM-DRW `losses.py` for reference.
- **Expected effect and local evidence.** O12: IA weighting is +0.001 on val and −0.009 gated on test, but confounded
  by 4 against 12 epochs. Expected from logit adjustment: −0.005 to +0.003 (inference: it would raise rare-term scores,
  and IA-weighted Fmax rewards rare terms only when they are right).
- **Cost.** Each arm is 0.5 GPU-h on the 3B cache, or 3-4 GPU-h on live 35M.
- **TV v2 experiment.** L6 on the 3B cache: {IA weight on, off} × 2 seeds, then logit adjustment with tau ∈ {0.5, 1}
  on the winner, scored after the niche-8 calibration so that pure calibration shifts do not count as gains.
- **Kill rule.** Keep IA weighting unless "off" wins by at least 0.005 test-like with both seeds. Adopt logit
  adjustment only at +0.005 test-like after calibration. Otherwise drop it for good.

### Niche 7. Optimisation and the loss landscape

- **Decision.**
  - Optimiser: AdamW with weight decay excluded from biases, norms and embeddings (current `wd_exclude`).
  - Schedule: replace cosine-to-zero with **warmup–stable–decay** (WSD): 6% warmup, constant LR, then a 20% linear
    cooldown.
  - **Tail averaging:** an EMA of head weights, or a uniform average of checkpoints, evaluated against the last
    iterate.
  - WSD makes resume and extension free: a run that has not converged can be extended from the stable phase without
    restarting the schedule.
  - SAM, adaptive form (ASAM), only on the head, as one arm.
- **Fallback.**
  - If the EMA fails the rule: cosine with the last iterate (current).
  - If a run plateaus early: SGDR-style warm restart from the stable phase (SGDR is ICLR 2017, known).
  - Model soups only across runs that pass a linear-mode-connectivity check.
- **Paper.**
  - AdamW, ICLR 2019.
  - WSD, NeurIPS 2024: constant LR plus cooldown matches cosine, and weight averaging improves checkpoints along the
    path at no extra training cost.
  - SWAD, NeurIPS 2021: dense averaging that finds flatter minima; overfit-aware start and end.
  - SAM, ICLR 2021 and ASAM, ICML 2021.
  - FlatMin, NeurIPS 2022: SWA often fails with Transformers, SAM tends to help NLP, and averaged SAM iterates are
    the most robust. This is why both are measured and neither is assumed.
  - Soups, ICML 2022 and LMC, ICML 2020: average only within one linearly connected basin.
  - GenMeas, ICLR 2020: sharpness-based measures are the best predictors of the generalisation gap, used for
    monitoring.
- **Code.** torch `optim.AdamW`; epfml/schedules-and-scaling `src/optim/utils.py` (WSD) and
  `src/optim/weight_averaging.py`; torch `swa_utils.AveragedModel` with `get_ema_multi_avg_fn`; khanrc/swad
  `domainbed/swad.py` `LossValley`; davda54/sam `SAM(adaptive=True)`; model-soups `main.py` greedy soup.
- **Monitoring and detecting a poor minimum.**
  - Log per epoch (cheap, from `pred_tv`): TV v2 test-like Fmax, train loss, val loss, and the gap between
    random-val and TV Fmax.
  - Every 2 epochs on a fixed 512-protein train subset: a sharpness proxy, the loss increase under one ASAM ascent
    step at rho=0.05 (GenMeas measure family), and the top Hessian eigenvalue by 20 power iterations on the head only.
  - Diagnostics already logged: kNN lambda per aspect, mix weights, CRLA gate.
  - Poor-minimum flags:
    - TV test-like falls for 2 epochs while val loss still falls (overfitting to the random-val structure).
    - Sharpness rises by more than 2× between epochs, together with a train-loss plateau (edge-of-stability type,
      known).
    - The EMA-minus-last gap goes negative. This signals that averaging crosses a barrier, i.e. no shared basin.
- **Expected effect and local evidence.** No local measurement exists. With a frozen encoder only the head
  (about 2-30M parameters) is optimised; its loss-landscape effects are expected to be small. EMA: +0.001 to +0.004 for
  the HiGO component. ASAM: 0 to +0.004. Under fusion, both are diluted roughly by half (inference).
- **Cost.**
  - EMA: no extra steps, plus one extra prediction pass per evaluation (about 0.1 GPU-h on 3B).
  - ASAM on a frozen encoder: the second forward/backward is on the head only, if the encoder states of the batch
    are kept. That adds about 30% on 3B live (encoder dominated) and about 90% on the cache.
  - Sharpness monitoring: about 2 min per check.
- **TV v2 experiment.** On the 3B cache, same seed: {cosine + last, WSD + last, WSD + EMA, WSD + ASAM + EMA}, then the
  winner with a second seed.
- **Kill rule.**
  - EMA is kept by default if it is at least 0.000 on TV test-like (no cost).
  - ASAM is kept only at +0.004 or more with both seeds.
  - A soup is used only if the LMC barrier along the interpolation path is below 0.002 TV Fmax; otherwise ensemble
    the predictions instead (niche 9).

### Niche 8. Calibration and thresholding for Fmax

- **Decision.**
  - Fit temperature scaling per cell on TV v2: (aspect × identity bin {≥95, 80-95, 50-80, <50, no hit} ×
    IA tercile), one temperature and one bias per cell, applied to the final fused logit. Then use one global
    threshold sweep (the CAFA protocol).
  - Global monotone transforms are useless, because the threshold is swept on test. Only *relative* calibration
    across proteins and terms changes Fmax; that is exactly what per-cell temperatures change.
  - Abstention is reported, not exploited: the kNN row is reported with and without a naive fallback on no-hit
    proteins (O29, open question 9).
- **Fallback.** Per-aspect × identity-bin only (15 cells), if the 45-cell fit overfits under CV.
- **Paper.** TempScale, ICML 2017 (one-parameter temperature fit on held-out data); Koyejo15, NeurIPS 2015 (calibrated
  marginals plus one threshold are optimal for instance-averaged F). Selective classification (NeurIPS 2017, known)
  for the abstention disclosure.
- **Code.** gpleiss/temperature_scaling `ModelWithTemperature.set_temperature`, adapted to a sigmoid per cell; in-repo
  `knn_sharp.Scorer` for the sweep.
- **Expected effect and local evidence.** The O42 failure is a cross-bin calibration failure: models trust close
  neighbours too little. The L1 GBM on TV v2 gains +0.0057 test-like, which shows that relative re-weighting by
  identity is worth about 0.005 even without a new signal. Expected: +0.003 to +0.008 on top of any fused model.
- **Cost.** CPU only, minutes (128 cores). No GPU-h.
- **TV v2 experiment.** 5-fold CV over TV proteins, stratified by cell. Fit on 4 folds and score the fifth.
  Compare uncalibrated, 15-cell and 45-cell versions.
- **Kill rule.** Keep the cell scheme with the best CV test-like score if it beats uncalibrated by at least 0.002 and
  never loses more than 0.003 in any identity bin. Otherwise no calibration.

### Niche 9. Ensembling and stacking on TV

- **Decision.**
  - A per-aspect histogram-GBDT stacker on (protein, term) pairs, fit on TV v2 with 5-fold CV. Features: the
    current L1 set (p_knn, sharpened score, s_best, gap, support fractions, log_n, max_id, has_hit, IA, depth)
    **plus the HiGO-R logit**, taken from a model that never saw TV labels. Candidates are `pred_tv` of C0/C1,
    L4a and L5.
  - Then niche-8 calibration on the stacker output.
  - Across HiGO-R variants: Caruana ensemble selection with replacement on TV test-like Fmax, over the library
    {kNN, L1, C0, C1, L4a, L5, seeds}.
- **Fallback.** Per-aspect logistic stacker (it lost on v2, O40), or plain L4a output with calibration only.
- **Paper.** LightGBM, NeurIPS 2017 (histogram GBDT, which `HistGradientBoostingClassifier` implements); EnsSel,
  ICML 2004 (forward selection optimised directly for the target metric); DeepEns, NeurIPS 2017 (independent seeds,
  averaged probabilities).
- **Code.** sklearn `HistGradientBoostingClassifier`; in-repo `stacker.features`, `stacker.AspectModel`;
  auto-sklearn `autosklearn/ensembles/ensemble_selection.py` `EnsembleSelection`.
- **Expected effect and local evidence.** L1 v2 without HiGO: +0.0008 overall, +0.0057 test-like, +0.023 at 30-50%.
  The GBM gains sit in 50-80% (0.543-0.546 vs 0.518). HiGO helps below 50% identity (0.480 vs 0.457 for DIAMOND at
  30-50%, O6), the place where the homology-only stacker has the least signal. Expected with the HiGO feature:
  +0.005 to +0.012 test-like over L0. A seed ensemble adds +0.002 to +0.006 to the HiGO component (seed std
  0.004-0.006).
- **Cost.** Stacker: CPU, under 1 h on 128 cores (7,566 proteins × candidate terms). Seeds: one extra run each
  (0.5 GPU-h on the 3B cache, 4-6 GPU-h live).
- **TV v2 experiment.** Rerun `stacker_l1_v2` with the HiGO feature from each candidate `pred_tv`. Same folds, same
  registered criterion. Report the delta against L0 and L1-v2.
- **Kill rule.** Keep the stacker if it is at least +0.005 test-like over L0 with plain mean ≥ −0.002 and no ≥95% bin
  below kNN by more than 0.003. Keep a seed or variant in the ensemble only if forward selection picks it with a gain
  of at least 0.001.

### Niche 10. Efficiency

- **Decision.**
  - Attention: PyTorch SDPA, which dispatches FlashAttention-2 and memory-efficient kernels; it is the HF `EsmModel`
    default when supported. The standalone `flash_attn` package is broken (ABI mismatch).
  - Batching: keep length bucketing (sorted chunks of 4,096, 16,384-token budget, `max_batch` 32).
  - Precision: bf16 frozen weights and bf16 autocast for the head, fp32 master weights for the trainable head.
  - **Recompute** the encoder for live runs. After L5 fixes the mix weights, build one **3B feature cache in
    `/dev/shm`**: the fixed-mix layer-30-region states, projected to 512 dimensions by a fixed linear map (PCA fit on
    a 2M-residue sample), stored in bf16. Size: 54.0M train tokens + about 13M for val/test/TV, × 512 × 2 B ≈ 69 GB,
    which fits in the 126 GB `/dev/shm`. Nothing goes to disk.
  - No torch.compile as a design decision (venue rule). The existing fused-entmax compile in `model.py` stays an
    implementation detail.
- **Fallback.** If 69 GB of RAM is not available to the shared node: PCA-384 (52 GB). If the cache hurts accuracy (rule
  below): no cache; run 3B arms live, one at a time per GPU.
- **Paper.** FA1, NeurIPS 2022; FA2, ICLR 2024 (the SDPA flash backend); MixedPrec, ICLR 2018 (half-precision
  storage, fp32 master weights); LST, NeurIPS 2022 (the backbone in no-grad mode feeds a side network through learned
  *down-sampling* projections, so a fixed low-rank projection of frozen states is the cached version of that
  interface). Bucketing follows the token-budget batching of Vaswani et al., NeurIPS 2017 (known).
- **Code.** `torch.nn.functional.scaled_dot_product_attention`; HF `EsmModel(attn_implementation="sdpa")`; in-repo
  `train.Batcher.batches`, `predict` (window max-pooling), `plm_bf16`; sklearn `PCA` or `torch.pca_lowrank` for the
  projection.
- **Expected effect and local evidence.** Live 3B: 1.5 s/step, about 3,900 steps/epoch. With the cache, a head-only
  epoch is bounded by the head (estimated 5-10 min) instead of about 45 min, a 5-8x speed-up (inference). That turns
  niches 3, 6, 7 and 9 from 4-6 GPU-h arms into 0.5-1 GPU-h arms. No accuracy effect is intended.
- **Cost.** Building the cache is one forward pass over about 142k proteins, about 1 GPU-h, plus PCA on CPU. RAM is
  69 GB while it is held.
- **TV v2 experiment.** Train the L5 head for 2 epochs on the cache and compare it with the same 2 epochs of the live
  L5 log, on TV v2 test-like.
- **Kill rule.** Use the cache for ablations only if the cached head is within 0.005 test-like of the live head at
  matched epochs. Final models are always retrained live (no projection).

### Niche 11. Structure and multimodal signals

- **Decision.** None on the GO-GPT benchmark.
- **Fallback.** For CAFA-6 work only: ProstT5 3Di tokens (a SaProt-style vocabulary) on the 35M head. That needs a
  weight download of about 3 GB, which is outside the "no large weights" rule today.
- **Paper.** SaProt, ICLR 2024 (+0.015 to +0.026 Fmax at 35M on DeepFRI GO); ProSST, NeurIPS 2024.
- **Code.** westlake-repl/SaProt (MIT).
- **Expected effect and local evidence.** On the GO-GPT test, 88% of proteins are at ≥50% identity, where kNN dominates.
  The remote bins (<50% plus no hit) are 11.9% of test, and a +0.02 gain there is worth at most about +0.002 overall
  (red-team bin values). On the CAFA-6 clustered split, 81% of proteins are below 50% identity (O45), where structure
  matters more.
- **Cost.** About 1 GPU-h for 3Di inference plus a 35M arm; 150 MB of tokens.
- **TV v2 experiment.** None scheduled.
- **Kill rule.** Not started unless the GO-GPT headline is met and CAFA-6 work resumes.

---

## 3. Ranking by expected gain per GPU-hour

Expected gain is the overall test (GO-GPT) effect after fusion and stacking, as a range. GPU-h is the extra GPU time
the decision needs beyond what is already running. CPU-only items rank first by construction; their order follows the
size of the expected gain.

| rank | decision | niche | expected gain (overall, after fusion) | extra GPU-h | gain per GPU-h |
|---|---|---|---|---|---|
| 1 | GBDT stacker on TV with a TV-clean HiGO feature | 9 | +0.005 to +0.012 | ~0 (CPU) | highest |
| 2 | Per-cell temperature calibration on TV, plus TV refit of the kNN trust parameters | 8, 5 | +0.003 to +0.008 | ~0 (CPU) | highest |
| 3 | Keep MCM after fusion and check closure after stacking | 4 | protects O10's +0.004 | 0 | n/a (keep) |
| 4 | Tail EMA with a WSD schedule | 7 | +0.001 to +0.004 | ~0.1 | ≈ 0.02 per GPU-h |
| 5 | In-network kNN fusion with trust refit (L4a, running) | 5 | +0.005 to +0.015 | ~4 (running) | ≈ 0.0025 |
| 6 | Frozen 3B with scalar mix (L5, running) | 1 | +0.005 to +0.015 | ~5 (running) | ≈ 0.002 |
| 7 | 3B feature cache in `/dev/shm` (enabler for ranks 9-12) | 10 | 0 directly; 5-8x cheaper arms | ~1 | enabler |
| 8 | L4a on 3B features (L4 × L5) | 1, 5 | +0.003 to +0.010 over L4a-35M | ~5 live, ~1 cached | ≈ 0.001 live, ≈ 0.006 cached |
| 9 | Seed ensemble (3 seeds) of the final head, Caruana selection | 9 | +0.002 to +0.006 | ~1.5 cached | ≈ 0.003 |
| 10 | ASAM on the head | 7 | 0 to +0.004 | ~1 cached | ≈ 0.002 |
| 11 | LoRA adapter (L3 decides; 3B LoRA ~3.5x frozen cost) | 2 | 0 to +0.005 | 0 for 35M (running); ~15 for 3B | < 0.0005 on 3B |
| 12 | IA-weight and logit-adjustment ablations | 6 | −0.005 to +0.003 | ~2 cached | ≈ 0 |
| 13 | Sparsemax alpha=2 against entmax alpha=1.5 | 3 | 0 ± 0.003 | ~0.5 cached | ≈ 0 |
| 14 | Structure tokens (ProstT5 3Di) | 11 | ≤ +0.002 on GO-GPT | ~2 plus download | lowest (skip) |

One-line reasons:
1. The stacker fixes the trust problem of O42 directly and is the first time HiGO's remote-bin signal meets the
   kNN features on TV.
2. Relative calibration across identity bins is where the measured random-val failures came from; it costs CPU
   minutes.
3. MCM is the only head component with a gain that survives on test.
4. Weight averaging is free once checkpoints exist.
5. The learned per-term trust is the planned mechanism to win at 50-95% identity.
6. 3B gave +0.045 at epoch 0 for HiGO alone.
7. The cache makes every head-only ablation cheap.
8. L4a on 3B combines the two largest single effects.
9. The seed spread (0.004-0.006) is large enough for an ensemble to pay off.
10. ASAM is only cheap on a frozen encoder.
11. The adapter lead on 35M is shrinking (+0.010 to +0.007) and LoRA on 3B is expensive.
12. Balanced-error losses do not target Fmax.
13. Softmax and entmax already tie.
14. Remote homologs are a small share of the test.

---

## 4. Execution order (both GPUs and CPU)

Times are from 08:10 today and are estimates from the observed epoch times.

**Now to about 11:00**
- GPU 0: L3 C0 and C1 continue (15 epochs max, patience 3). Expected stop between 10:00 and 11:30.
- GPU 1: L5 (8 epochs max, patience 2, about 44 min/epoch, ends about 12:30-13:30) and L4a-35M (ends about 10:30-11:30).
- CPU, before any of these finish:
  1. Log the registered TV criterion (test-like primary, plain mean as guardrail) in `runs/test_looks.md` or the
     ladder notes.
  2. Compute per-bin bootstrap CIs for the 80-95% and 50-80% bins on TV v2 using the kNN prior (open question 4). If
     the 50-80% CI half-width is above 0.03, the niche 5 and 9 kill rules use the pooled 50-95% bin instead.
  3. Fit the niche-8 calibration on kNN alone and on the L1 v2 stacker as a dry run of the protocol (no new
     predictions needed).
  4. Code changes needed later, listed for whoever edits the code:
     - `stacker.py` gains a `--higo pred_tv_all_top500.npz` feature.
     - `train.predict` saves \(z^{HiGO}\) and the kNN logit separately for the trust refit.
     - `train.py` gains WSD, EMA and an ASAM option.
     - A cache writer and reader for niche 10.

**When L3 ends (about 11:00), GPU 0**
- CPU: score C0 and C1 `pred_tv` alone and through the stacker. Apply the niche 2 rule.
- GPU 0: start **L4b = L4a on frozen 3B with mix** (niche 1 × niche 5), using the L5 epoch-0 evidence. Early kill: if
  L4b's TV test-like at epoch 1 is not above L4a-35M at epoch 1, stop it and wait for L5.
- If LoRA wins L3 by 0.01 or more, also queue a second C1 seed after L4b (35M, GPU 0, about 5 h). Otherwise none.

**When L4a-35M ends (about 11:00), CPU, then GPU 1 is shared only by L5**
- Stacker and calibration on L4a `pred_tv`: first full HiGO-R pipeline number on TV v2. Apply the niche 5 rule,
  including the ≥95% guard.
- Run the trust refit on TV (CPU).

**When L5 ends (about 13:00), GPU 1**
- Apply the niche 1 rule on TV. If 3B passes, build the 3B feature cache (about 1 GPU-h) with the learned mix
  weights. Validate it with the niche 10 rule (2 cached epochs against the live log).
- Cached arms on GPU 1, about 0.5-1 GPU-h each, run back to back:
  1. WSD+EMA against cosine (niche 7).
  2. ASAM (niche 7).
  3. IA weight on/off × 2 seeds (niche 6).
  4. Logit adjustment if IA weight passes (niche 6).
  5. Alpha=2 against alpha=1.5 (niche 3).
  6. Two extra seeds of the winner (niche 9).
- If 3B fails the niche 1 rule: launch frozen 650M with mix (fallback) live on GPU 1 instead of the cache.

**About 17:00-19:00 (both GPUs)**
- Retrain the chosen configuration live (no projection), with niche 7 winners, 2-3 seeds on both GPUs in parallel. The
  base is L4b, or L4a-35M if L4b fails.
- CPU: final stacker (niche 9) plus calibration (niche 8), Caruana selection across seeds and variants, closure check
  (niche 4). Freeze everything on TV.

**Test looks (2 left of 4)**
- Look 3: the frozen pipeline from the step above, with rows: GO-GPT 0.651 / 0.6866 / 0.6984, kNN `T5_k50_g0_na` with
  and without no-hit fallback, L1, HiGO+gate, HiGO-R. This merges the scheduled L3 and L5 looks, as `observations.md`
  open question 10 asks.
- Look 4: reserved for the final model after any change made strictly on TV.

---

## 5. Points that change the brief

- The bar is 0.6984 (ten-sample frequency), not 0.687. A win needs about 0.708 or more.
- The 650M checkpoint is cached locally (2.5 GB), so the niche 1 fallback needs no download.
- Koyejo et al. for multilabel F is NeurIPS 2015; its binary predecessor is NeurIPS 2014.
- Entmax-1.5 (ACL 2019), SWA (UAI 2018), ASL (ICCV 2021) and torch.compile (ASPLOS 2024) fall outside the venue rule.
  The doc keeps entmax only as a hyperparameter of the sparsemax family, uses SWAD/WSD for averaging, and treats
  compile as an implementation detail.
- C-HMCNN's code is GPL-3.0; the in-repo re-implementation must not import from it.
- L3, L4a and L5 are all running already, with GPU 1 shared by L5 and L4a. "GPU 1 idle" no longer holds.
