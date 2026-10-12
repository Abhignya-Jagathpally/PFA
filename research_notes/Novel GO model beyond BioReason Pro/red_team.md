# Red team: attacking the candidate designs for a de novo GO model that beats GO-GPT on its temporal benchmark

Date: 2026-10-12. Read-only review of `sota_critique.md`, `tier1_design_ideas.md`, `adapters_trla.md`,
`structure_multimodal_decoding.md`, `codebase_inventory.md`, `data_training_evaluation.md`,
`../Beating BioReason Pro de novo/code_audit.md` and `cafa6/REPORT.md` §4 and §8. Nothing was modified except this file;
no GPU was used. All new numbers below come from CPU checks run against existing files (`data/proc_gogpt`,
`runs/gogpt_cafa5`, `runs/gogptbench_35m_lora`, `runs/bench_gogpt`, `/tmp/gogpt_train`). The scripts were piped to
python through stdin, so nothing was written to disk.

Labels: **[measured]** = computed during this review; **[note]** = taken from one of the notes; **[inference]** = my
reasoning, not tested.

---

## 0. The finding that changes the plan

**A zero-training, identity-sharpened kNN already ties GO-GPT on its own benchmark [measured].** It copies the
labels of DIAMOND hits against GO-GPT's training split, weighted by `bits * exp((pident - max_pident)/T)`, and scores
0.685–0.701 IA-weighted Fmax on the 8,630-protein test across a 12-setting grid (T ∈ {1, 2, 5, 10}, top-k ∈ {3, 10, 50}).
GO-GPT's released checkpoint scores 0.687. The setting picked on GO-GPT's random validation split (T=10, k=50) scores
0.691 on test. Paired protein bootstrap (300 reps) of that setting minus GO-GPT: +0.005, 95% CI [−0.0015, +0.012]. So
the kNN baseline ties GO-GPT and is not significantly better.

To check the evaluator, I re-implemented the cafaeval computation (prop=max, norm=cafa, 0.01 thresholds, GO
2023-01-01, CAFA-5 IA) in memory. It reproduces `runs/bench_gogpt/summary.tsv` to four decimals: GO-GPT beam
0.6866 (MF 0.7143 / BP 0.6171 / CC 0.7284), HiGO-35M raw 0.4547, HiGO+gate 0.5759.

Consequences:
1. The headline "beats GO-GPT" can be reached without any new architecture. A paper or report that presents a new
   model as beating GO-GPT without this kNN row on the same table would be misleading. The novel model has to beat
   **the sharpened kNN**, not just GO-GPT.
2. GO-GPT's benchmark is mostly a nearest-neighbour label-transfer test. GO-GPT scores 0.835 at 100% identity, where
   copying the best hit scores 0.872 and the sharpened kNN 0.90. GO-GPT's only remaining edge over kNN is in the
   80–95% bin (0.657 vs 0.61–0.64) and, for bare kNN, the 50–80% bin (0.469 vs 0.41–0.44).
3. The random validation split is a poor guide for tuning. Validation picked T=10, which is among the worst settings
   on test; T=1–2 is best on test. Per-bin blend weights between kNN and HiGO+gate fit on validation brought the blend
   down to 0.687, versus 0.711 with test-tuned weights (that 0.711 is an oracle and must not be reported). A temporal
   validation set is therefore a prerequisite, not a refinement.

---

## 1. Claim check

| # | Claim (source) | Status | Reason |
|---|---|---|---|
| 1 | Released GO-GPT checkpoint = 0.687 wFmax on the 8,630-protein test (inventory, REPORT §4.1) | **verified** | Recomputed 0.6866 from `runs/gogpt_cafa5/preds_shard*.jsonl` with an independent evaluator. |
| 2 | 88% of test proteins have a ≥50%-identity training hit; per-bin GO-GPT 0.730 / 0.440 / 0.395 / 0.374 (REPORT §4.1) | **verified, small binning drift** | From `identity.parquet`: 7,607 / 694 / 79 / 250 proteins; the bench file has 7,585 / 713 / 82 / 250. With my bins GO-GPT scores 0.729 / 0.434 / 0.402 / 0.374. The drift probably comes from a different identity source in `bench_gogpt.py`; it is not material. |
| 3 | 51% of test proteins have a ≥95% hit and 27% a 100% hit (code_audit) | **51% verified; 27% doubtful** | Best-by-bits hit ≥95%: 51.1% (4,410). Max pident ≥95%: 53.5%. At 100%: 25.4% by max pident (2,190) and 20.7% by best-by-bits hit. The audit's own count (n=1,922) is 22%, so its "27%" is inconsistent with itself. |
| 4 | Best-hit copy reaches IA recall 0.87–0.92 at precision 0.71–0.74 on ≥95% proteins (code_audit) | **plausible, consistent** | I did not recompute pooled P/R. Per-bin Fmax of best-hit copy: 0.872 at 100%, 0.830 at 95–99.9%. Best-hit coverage is high (median qcov 100%, p25 98.5%), so short local hits are not driving this. |
| 5 | HiGO-35M+gate = 0.576; still under-trained (val 0.478→0.545) (inventory) | **verified** | Recomputed 0.5759. The training log shows 4 epochs inside a 10-epoch cosine schedule, killed by `finish_fast.sh`. |
| 6 | Paper: GO-GPT 0.65 greedy / 0.67 ten-sample / 0.70 best-of-10; shipped predictions score 0.649 (sota_critique) | **plausible** | Preprint self-report, not re-fetched. Our 0.649 rescoring of the shipped predictions agrees with the 0.65. Missing piece: nobody has measured ten-sample scoring on the **public** checkpoint. If it adds the paper's +0.02, the bar is about 0.70–0.71, not 0.687 [inference]. |
| 7 | The public checkpoint was "trained on more/later data" (code_audit, sota_critique) | **doubtful (speculation)** | The only evidence is BP +0.08 over the paper. A different checkpoint-selection epoch would also explain it. Do not repeat this as fact. |
| 8 | GO-GPT ablations: ESM2-650M ≈ 3B, text-initialised GO tokens give no gain, IEA as targets gives no gain, SwissProt pretraining gives no gain (sota_critique, data note) | **plausible, weak** | Appendix prose in a non-peer-reviewed preprint, with no numbers transcribed. The direction is credible because it matches GO-GPT v2's switch to 650M. |
| 9 | ProtBoost: IEA as **features** +0.02 (data note) | **plausible, not transferable as is** | Kaggle leaderboard ablation in an arXiv preprint. They used dated GOA archives. Locally, the only GOA we have is a 2026 file for 10 reference proteomes with **no zebrafish**, and zebrafish is 45% of the GO-GPT test set [measured]. |
| 10 | SaProt table: ESM2 35M→650M gives +0.05–0.07 Fmax (tier1) | **verified as reported; transfer doubtful** | The DeepFRI PDB split is unweighted, not temporal, and has low homology. On this benchmark 51% of proteins sit at ≥95% identity, where kNN already scores 0.85–0.90, so encoder size can only matter for roughly the remaining 40% [inference]. |
| 11 | 3B leads 35M after one epoch (0.485 vs 0.445) (inventory, adapters) | **verified but weak evidence** | Same-epoch comparison on CAFA-6. Converged 35M reached 0.527, and 3B never finished, so this says nothing about the converged gap. |
| 12 | TRLA is an ICML 2026 GenBio workshop poster, a 4.7M token-wise residual MLP between a frozen ProtT5 encoder and decoder (adapters) | **plausible, partly lost** | The note's Q1 section was truncated and the PDF was not re-read. The mechanism details used later (width, init) are reconstructions. It is a workshop paper, so it should not be cited as "ICML". |
| 13 | Koyejo et al. (NeurIPS 2015): thresholding calibrated marginals with a shared threshold is Bayes-optimal for instance-averaged F (tier1) | **verified theorem, imperfect fit** | CAFA's `norm=cafa` averages precision only over proteins with predictions, and IA weighting applies. The theorem does not cover that variant (the note says so). See §3.4: abstention measurably changes the score. |
| 14 | IA-weighted loss is a lever (tier1, data note) | **doubtful on our evidence** | REPORT §4.2 ablation: +0.001 on val, −0.009 on test, under a 4-epoch confound. No clean measurement exists. |
| 15 | Domain (InterPro) evidence gives the only post-hoc gain (+0.013, BP +0.033 on CAFA-6) (REPORT §8.2) | **verified locally, leak-exposed** | It reverses on the temporal test (grounded 0.550 < gate 0.576), and `interpro2go` is a 2026 release that was never date-filtered. |
| 16 | 93.8% of test GT IA mass lies inside our 7,901-term GO-GPT-bench vocab (inventory, audit) | **plausible** | Two notes computed it independently and agree. This caps the gain from vocabulary expansion at about 6% of IA mass, mostly MF (88.7% covered). |
| 17 | "BioReason-Pro reports 73.6% Fmax" (tier1) | **misleading** | That figure is **unweighted** Fmax. It cannot be compared with the 0.687 weighted figure. |

### Contradictions between notes

- **GO-GPT's vocabulary size.** `data_training_evaluation.md` §4 says "GO-GPT's ~7,901-term vocabulary (≥20 proteins)".
  That is wrong: 7,901 is our own GO-GPT-bench vocabulary (≥30). GO-GPT's is 9,977 terms (≥20); its tokenizer has
  10,303 entries.
- **IEA labels.** GO-GPT reports that IEA as targets did not help; ProtBoost reports +0.02 from IEA as features. These
  are not contradictory because they use IEA in different roles. The leakage side is the real conflict:
  - The data note correctly asks for a GOA snapshot from on or before Nov 2022.
  - The inventory notes that our local GOA is a 2026 release, filtered by annotation date (`--goa_date_max 20221130`).
    A date filter applied to a 2026 file is not equivalent to a 2022 snapshot. Annotations removed or obsoleted since
    2022 are missing. IEA rows carry the date of the last pipeline run, so a date filter drops most IEA rather than
    restoring the 2022 IEA set [inference; check the date distribution of IEA rows].
  - `interpro2go` was never filtered at all.
- **GO-text label encoder.**
  - `tier1_design_ideas.md` ranks it #3.
  - GO-GPT reports no gain from text initialisation, and `structure_multimodal_decoding.md` rates it low.
  - Resolution: text can only matter for out-of-vocabulary and rare terms, which carry at most about 6% of test IA
    mass here.
- **Does 3B help?**
  - Against: GO-GPT says 650M ≈ 3B and 150M is clearly worse; PoET-2 says scaling is neutral or negative for function
    prediction.
  - For: SaProt shows +0.05 from 35M to 650M, and our 3B run led 35M after one epoch.
  - Synthesis [inference]: 35M→650M probably matters for the 50–95% identity bins; 650M→3B probably does not; neither
    matters much at ≥95%, where retrieval dominates.
- **Sampling settings.** `tier1_design_ideas.md` says GO-GPT samples with top-k 50; `sota_critique.md` (citing the
  paper) says top-k 20. Use the paper's setting when measuring the ten-sample baseline.
- **Validation used for decisions.** The decision rules in `adapters_trla.md` are written against the random-split
  val number (0.5270), while the data note, the audit and this review all say tuning must use temporal validation.
  The adapter rules need to be re-stated on temporal validation.
- **What "SOTA" means.**
  - `structure_multimodal_decoding.md` points to an independent BioReason-Pro figure of 0.685 as the "nearest
    published figure to 0.687". That figure comes from 91 post-2026 proteins and is irrelevant as a comparator.
  - BioReason-Pro was evaluated on no-knowledge (NK) proteins only and with GO-GPT outputs as input. Our only fair
    comparator is GO-GPT on the full NK+LK (no-knowledge plus limited-knowledge) set, plus the S6 NK-only subset.

---

## 2. Attacks on each candidate component

Every component lists: the failure mode it targets; the strongest reason it might not help on this benchmark; its
leakage risk; its cost on our hardware (2× H100, about 4 GB free disk); and a kill criterion. All kill criteria are
evaluated on the **temporal validation set (TV)** defined in §4 step P2, reported per identity bin, with ≥2 seeds for
any learned component.

### 2.1 Identity-sharpened kNN and best-hit copying (zero training)
- **Targets:** the ≥95% identity bins, which hold 51% of proteins and contribute about 0.0026 overall per +0.01 in
  bin score. The current `diamond_score` is a bitscore-weighted mean over up to 50 hits, which dilutes the closest
  neighbour.
- **Why it might not help:** it already does, on test (§0). The risk sits elsewhere:
  - It gives nothing on no-hit proteins.
  - It is worse than HiGO+gate in the 50–80% bin (0.41–0.44 vs 0.475).
  - Its best setting does not transfer from random validation.
  - On CAFA-6's clustered split only 19% of test proteins have a ≥50% hit, so the gain there will be small
    [inference].
- **Leakage:** low, provided the database holds only GO-GPT training-split proteins with pre-cutoff labels, which is
  true of `diamond_hits_valtest.parquet`. Self-hits are removed by accession, while identical sequences under other
  accessions are kept. That is legitimate homology transfer, and GO-GPT benefits from the same thing by memorisation.
  Adding the 13,305 validation proteins to the database for the final model is allowed.
- **Cost:** CPU minutes; the hits are already cached.
- **Kill criterion:** none for the baseline, which must be reported regardless. Drop a particular sharpening variant
  if it is not better than the bits-weighted mean by ≥0.03 on TV for the ≥80% bins.

### 2.2 In-network retrieval over labelled neighbours (TabR-style attention and a per-term retrieval prior)
- **Targets:** the post-hoc gate cannot express per-term trust, and the 80–95% and 50–80% bins are where GO-GPT still
  beats kNN.
- **Why it might not help:**
  - The model can learn "copy the neighbour" and add nothing beyond the post-hoc kNN stack.
  - The identity caps needed to prevent that make the training distribution unlike test, where 51% of proteins sit at
    ≥95%.
  - Training needs train-vs-train hits, and same-protein or near-duplicate neighbours in training will teach copying
    unless neighbours above 95% identity are excluded for a fraction of batches.
  - ProtEx, the only GO-specific evidence, is a preprint.
- **Leakage:** medium. The retrieval database at test time must be train(+val) only. During training, exclude self and
  any neighbour whose labels include the query's held-out TV labels.
- **Cost:**
  - One train-vs-train DIAMOND run (CPU, about 30–60 min with 64 threads); store top-20 hits only, about 100 MB
    [inference].
  - Training overhead +10–20%.
  - Implementation M–L: `Batcher`, `train.py`, `model.py`, and every run loader.
- **Kill criterion:** drop it if the learned model does not beat the **post-hoc kNN+HiGO stacker of §2.1/§4 step L1**
  by ≥0.01 in the 50–95% bins on TV, with both seeds agreeing. Beating HiGO+gate is not enough.

### 2.3 Larger encoder (ESM2-650M / 3B, intermediate layer)
- **Targets:** representation quality for moderately close homologs; GO-GPT reads layer 30 of 36.
- **Why it might not help:**
  - Its gains land in the bins worth the least per point once kNN owns ≥95%: the 80–95% and 50–80% bins together
    give 0.0035 overall per +0.01.
  - The GO-GPT and PoET-2 evidence says 650M→3B is flat.
  - A 3B LoRA run costs about 2.4 GPU-h per epoch on CAFA-6 and about 4.3 GPU-h per epoch on GO-GPT data (1.8× the
    proteins), and the earlier run never finished.
- **Leakage:** none beyond pretraining. ESM2's UR50 2021 snapshot predates the test annotations; the test sequences
  themselves may be in UR50, which affects GO-GPT equally.
- **Cost:**
  - 3B is cached (11.3 GB). 650M needs a 2.6 GB download, which does not fit until disk is freed.
  - Frozen 3B truncated at layer 30 runs forward only, at roughly 0.25× the per-step cost of LoRA+checkpointing
    (adapters note estimate).
  - No residue caching is possible on disk.
- **Kill criterion:** drop 3B, and stay on 650M or 35M, if frozen 3B + head + retrieval is not ≥0.01 better than the
  same 35M model in the 50–95% bins on TV after equal wall-clock time.

### 2.4 Conditioned residual latent adapter (CRLA, TRLA-inspired) and ladder side network
- **Targets:** backprop cost through 3B, and a place to inject conditioning (taxon, neighbours, domains) at the
  residue level.
- **Why it might not help:**
  - The basic control is missing: frozen 35M with the full HiGO head was never run, so we do not know whether the
    adapter does anything at all. LoRA r16 ≈ r8 and DoRA ≈ LoRA suggest the adapter is not the bottleneck.
  - TRLA is a workshop paper on fitness regression with a paired latent-MSE objective, so the transferable part is just
    "a residual MLP", which is a Houlsby adapter.
  - Neighbour labels probably act through the logits, not the residues (the adapters note says the same).
- **Leakage:** none extra.
- **Cost:** large engineering cost (hooks, crop offsets, run loader in four places), with gains estimated in the note
  at 0 to +0.01.
- **Kill criterion:**
  - Do not build it unless control C0 (frozen + full head) vs C1 (LoRA) shows C1 − C0 ≥ 0.01 on TV.
  - After building, drop it if the "retrieval in the head only" variant (ablation 8 in the adapters note) is within
    0.005 of the full CRLA.

### 2.5 GO-text label encoder
- **Targets:** rare and out-of-vocabulary terms.
- **Why it might not help:**
  - GO-GPT reports no gain.
  - The in-vocabulary IA mass is already 93.8%.
  - Rare-term predictions rarely survive the single global Fmax threshold.
- **Leakage:**
  - Low for term names and definitions from GO 2023-01-01.
  - Avoid the BioReason-Pro Qwen3 GO embeddings (`go_embedding.pt`, 177 MB). They are part of the competitor's
    release, and their provenance (which GO release, whether trained on task data) is unknown. That is not strictly
    "GO-GPT weights", but it muddies the de novo claim [inference].
- **Cost:** low; all-mpnet-base-v2 is cached and the embeddings are a few MB.
- **Kill criterion:** drop it if TV gain is <0.003 overall and <0.01 on the high-IA tercile.

### 2.6 Label-query decoder vs autoregressive decoding
- **Targets:** GO-GPT's binary outputs and missing label-label dependencies in a factorised head.
- **Why it might not help:**
  - Under Fmax, calibrated marginals with one threshold are close to optimal (Koyejo et al.), so dependency modelling
    can only help through better marginals.
  - Query self-attention over about 8k terms is expensive.
  - Rebuilding our own autoregressive decoder would copy GO-GPT's design, so it is not novel, and kNN already beats
    GO-GPT where GO-GPT is strongest.
- **Leakage:** none.
- **Cost:** M–L. Restricting queries to the top-256 candidates from kNN plus the head keeps it cheap.
- **Kill criterion:** drop it if two query self-attention layers over the top-256 candidates give <0.005 on TV.

### 2.7 Structure tokens via ProstT5 3Di
- **Targets:** remote homologs.
- **Why it might not help:**
  - Remote homologs are about 12% of this test: 30–50% + <30% + no-hit = 1,023 proteins, worth about 0.0012
    overall per +0.01.
  - The SaProt gains come from AF2 3Di; predicted 3Di is a third distribution.
  - PoET-2 found little benefit from structure on supervised function.
  - Neither ProstT5 nor SaProt is cached, and we do not have the disk.
- **Leakage:** low (sequence-derived), but AFDB v6 models may be built with templates published after 2023
  [inference].
- **Cost:** a 2–3 GB download plus about 1 GPU-h to generate the 3Di; S–M integration.
- **Kill criterion:** park it until everything above is done. If run, drop it if remote-bin gain on TV is <0.02. Even
  a +0.02 there is worth only about +0.002 overall, so it can only be justified by the CAFA-6 clustered test.

### 2.8 IEA / InterPro as input features
- **Targets:** information GO-GPT does not use.
- **Why it might not help:**
  - **The leakage problem is likely larger than the signal.**
    - Our `interpro2go` is a 2026 release.
    - Current IEA (as opposed to a dated snapshot) includes transfers derived from the test-window experimental
      annotations themselves; ProtBoost says about 30% of IEA later becomes experimental.
    - The grounded calibrator already reversed sign on the temporal test (0.550 vs 0.576).
  - Coverage is uneven: the dated GOA we hold has no zebrafish (45% of test).
- **Leakage:** high unless:
  - raw InterPro **IDs** (computed signatures) are used as learned tokens, without `interpro2go`; or
  - dated GOA/`interpro2go` releases from on or before Nov 2022 are downloaded. Per-species files for human, mouse,
    rat and zebrafish are probably in EBI's `goa/old/` archives; verify.
- **Cost:** IDs are already on disk (`uniprot_features.parquet`; GO-GPT HF data has spans). Dated GOA needs network
  and a few hundred MB.
- **Kill criterion:**
  - Use IDs only, never current mappings, for the headline.
  - Drop the feature if the TV gain is <0.005.
  - Separately report any run using current-release mappings as "possibly leaky".

### 2.9 Temporal validation
- **Targets:** every tuning decision. §0 shows that the random split picks wrong hyperparameters and wrong blend
  weights.
- **Why it might not help:**
  - The only dated labels we hold cover 10 reference proteomes: rat, human and mouse (about 41% of test organisms),
    plus fly and yeasts. Zebrafish (45% of test) is missing, so the TV distribution still differs from test.
  - GO-GPT cannot be scored on TV, because its training set includes the TV proteins' post-T0 labels.
- **Leakage:** none, if T0 ≤ 2021-11 and those labels are removed from training.
- **Cost:** CPU hours; a small ZFIN GAF download (with dates) would close the zebrafish gap.
- **Kill criterion:** not optional. If TV and test disagree on the ranking of the zero-training variants by more than
  the bootstrap CI, fall back to choosing parameters on the most test-like TV subset: matching organism mix and
  identity distribution.

### 2.10 IA-weighted soft-F1 × ZLPR loss, parent masking (PU), logit adjustment
- **Targets:** mismatch between the training loss and IA-weighted Fmax; incomplete negatives.
- **Why it might not help:**
  - Our own IA-weighting ablation is null to negative.
  - On a test dominated by copy-the-neighbour, the loss mostly shapes the 50–95% bins.
  - Logit adjustment targets balanced error, not Fmax.
  - PU corrections raise scores for unannotated-but-true terms, which the benchmark scores as false positives.
- **Leakage:** none.
- **Cost:** free (35M runs).
- **Kill criterion:** keep only changes that give ≥0.005 on TV with 2 seeds pointing the same way. Run them as 35M
  ablations in parallel with heavier steps, never as the headline change.

### 2.11 Vocabulary expansion
- **Targets:** the 6.2% of test IA mass outside our vocabulary (MF 11.3%).
- **Why it might not help:**
  - These terms have fewer than 30 training proteins and rarely clear the global threshold.
  - kNN copying already reaches out-of-vocabulary terms for free, since the copy uses full propagated labels.
  - Cost grows linearly in the term-wise attention and in prediction storage, and storage does not fit dense at 30k
    terms.
- **Leakage:** none.
- **Cost:** M (sparse prediction storage is required first).
- **Kill criterion:** first measure the kNN+head stack with and without out-of-vocabulary kNN terms. If
  out-of-vocabulary kNN terms already capture most of the reachable IA mass, drop vocabulary expansion.

### 2.12 Ensembling
- **Targets:** variance (seed std about 0.004–0.006) and complementary errors.
- **Why it might not help:** gains are small (+0.005–0.015) and do not make a scientific claim. An ensemble with
  GO-GPT cannot be the headline.
- **Leakage:** none if weights are fit on TV.
- **Cost:** inference only.
- **Kill criterion:** use only as a final row. The GO-GPT ensemble must be a separate row labelled "uses GO-GPT
  outputs".

---

## 3. Quantifying the target

### 3.1 Bin sizes and value of a point
Bins come from `identity.parquet` max pident against GO-GPT's training split [measured]:

| bin | n | share | overall gain per +0.01 in bin |
|---|---|---|---|
| 100% | 2,190 | 25.4% | 0.0025 |
| 95–99.9% | 2,430 | 28.2% | 0.0028 |
| 80–95% | 1,360 | 15.8% | 0.0016 |
| 50–80% | 1,627 | 18.9% | 0.0019 |
| 30–50% | 694 | 8.0% | 0.0008 |
| <30% | 79 | 0.9% | 0.0001 |
| no hit | 250 | 2.9% | 0.0003 |

The count-weighted average of per-bin Fmax reproduces the overall score well for GO-GPT (0.687 vs 0.6866). For kNN it
fails: 0.684 by the formula vs 0.699 measured. kNN abstains on no-hit proteins, and `norm=cafa` averages precision
only over proteins with predictions. **Abstention on hard proteins raises the measured Fmax** [measured]. This is a
property of the metric, not a modelling gain, and it should be reported if exploited.

### 3.2 Per-bin scores of existing methods [measured; mean of MF/BP/CC Fmax inside each bin]

| method | all | 100 | 95–99.9 | 80–95 | 50–80 | 30–50 | <30 | no hit | S6 (NK only) |
|---|---|---|---|---|---|---|---|---|---|
| GO-GPT released (beam) | 0.687 | 0.835 | 0.829 | **0.657** | 0.469 | 0.438 | 0.402 | **0.374** | 0.702 |
| HiGO-35M raw | 0.455 | 0.459 | 0.492 | 0.472 | 0.428 | 0.417 | 0.396 | 0.360 | — |
| HiGO-35M + gate | 0.576 | 0.658 | 0.636 | 0.579 | 0.475 | 0.478 | 0.466 | 0.361 | 0.585 |
| best-hit copy (binary) | 0.679 | 0.872 | 0.830 | 0.609 | 0.394 | 0.363 | 0.345 | 0 | — |
| best-hit copy, scored by identity | 0.684 | 0.872 | 0.830 | 0.611 | 0.394 | 0.363 | 0.345 | 0 | — |
| sharpened kNN T=2, top-10 (fixed a priori) | **0.699** | **0.898** | **0.851** | 0.640 | 0.409 | 0.427 | 0.435 | 0 | — |
| sharpened kNN T=10, top-50 (val-selected) | 0.691 | 0.879 | 0.836 | 0.618 | 0.444 | 0.463 | 0.451 | 0 | 0.709 |
| that kNN + HiGO+gate, per-bin weights fit on val | 0.687 | 0.861 | 0.811 | 0.618 | 0.486 | 0.490 | 0.497 | 0.362 | 0.702 |
| kNN T=2 + HiGO+gate, weights fit on val | 0.700 | 0.887 | 0.844 | 0.632 | 0.489 | 0.489 | 0.475 | 0.362 | — |
| *oracle: kNN T=2 + gate, weights fit on test* | *0.711* | *0.899* | *0.852* | *0.651* | *0.489* | *0.490* | *0.493* | *0.362* | — |

Grid of kNN alone on test, (T, k) → Fmax:
- T=1: (1,3) 0.694, (1,10) 0.698, (1,50) 0.701
- T=2: (2,3) 0.694, (2,10) 0.698, (2,50) 0.700
- T=5: (5,3) 0.690, (5,10) 0.699, (5,50) 0.701
- T=10: (10,3) 0.685, (10,10) 0.689, (10,50) 0.691

Validation ranks these settings nearly in reverse, from 0.554 to 0.579.

Bootstrap, paired over proteins (300 reps):
- val-selected kNN − GO-GPT = +0.005 [−0.002, +0.012];
- val-tuned blend − GO-GPT = +0.001 [−0.005, +0.008].

The 95% half-width is about 0.007, so a claim of a significant win needs roughly +0.01 or more.

### 3.3 What each bin must score
- If the remote bins stay at HiGO+gate level (30–50% 0.478, <30% 0.466, no hit 0.361), the ≥50% group (7,607
  proteins) must average about **0.700 for overall 0.67**, **0.719 for 0.687**, and **0.734 for 0.70**. GO-GPT
  averages 0.729 there. The sharpened kNN T=2 averages 0.745.
- Holding the other bins at the val-tuned blend's values, the 80–95% bin must reach **0.505 for 0.67**, **0.613 for
  0.687**, **0.695 for 0.70** and **0.759 for 0.71**. GO-GPT scores 0.657 there.
- A plausible target scenario [inference] combines:
  - sharpened kNN at ≥95% (0.898 / 0.851);
  - a learned model reaching GO-GPT's level in the 80–95% bin (0.657);
  - about 0.48 in the 50–80%, 30–50% and <30% bins, and 0.37 with no hit.
  
  That gives about **0.715**. It is close to the test-oracle ceiling of the zero-training blends (0.711–0.717). The
  realistic band for a well-built model is 0.70–0.72.
- The fair probabilistic target (0.67, from the paper) is already exceeded by the zero-training kNN. That target is
  stale: it was set by the paper's checkpoint. The bar that matters is the public checkpoint with ten-sample scoring,
  which has not been measured. If it gains the paper's +0.02, that bar is about 0.705 [inference]. Measuring it is
  step L2 below.

### 3.4 Is the target plausible with best-hit copying plus a 3B encoder?
For 0.687: yes, already reached, since copy plus sharpening alone ties. For a significant win at ≥0.70: plausible but
not shown. Three points:
1. The ≥95% bins are close to saturation at about 0.85–0.90 with kNN.
2. The remaining headroom is in the 80–95% and 50–80% bins, 35% of proteins. There, a 3B encoder is what GO-GPT uses,
   and GO-GPT still only reaches 0.657 / 0.469.
3. So the encoder helps at most to GO-GPT's level in those bins, and the extra points must come from combining the
   encoder with neighbour labels better than GO-GPT does (it has no retrieval) [inference].

---

## 4. Experiment ladder (autoresearch style)

Rules for every step:
- Fixed budget; one change per step.
- Metric = IA-weighted Fmax on **TV**, reported overall and by identity bin, plus the CAFA-6 clustered val as a
  guardrail.
- Keep or drop is decided before looking at the GO-GPT test.
- The GO-GPT test is touched at most at P3, after L3, after L5 and at the end, and every look is logged in a
  `test_looks.md`.
- No step may lower CAFA-6 clustered val by more than 0.005 relative to the current best.

### Prerequisites (no GPU)
- **P0. Rescue /tmp.**
  - `benchmarks/gogpt_eval.py` imports GO-GPT code from `/tmp/BioReason-Pro-main` (31 MB).
  - The GO-GPT training parquet (the only copy of the full labels, InterPro spans, AFDB ids and STRING ids) lives in
    `/tmp/gogpt_train` (148 MB).
  - Both disappear on reboot. Copy them into the repo or the HF cache (about 180 MB) before anything else.
- **P1. Free disk (user decision; about 12 GB available).**
  - `runs/dev_35m_lora/submission/submission.tsv`: 1.1 GB; a `.gz` copy exists.
  - `~/.cache/pip`: 2.9 GB.
  - Unrelated HF models: Bio_ClinicalBERT 3.0 GB, MOMENT 1.3 GB, Geneformer 1.2 GB.
  - `~/.cache/torch/hub`: 2.6 GB.
  - Also switch prediction outputs to sparse top-k storage. A dense 8k-term pair is 136–210 MB per run, and `evaluate`
    writes TSVs into `PROC`.
  - `grounding_data.py` refuses to run below 5 GiB free.
- **P2. Build TV (CPU, about 4 h).**
  - Pick T0 = 2021-11-30.
  - From `goa_reference.parquet` (dated, 10 proteomes), take GO-GPT training proteins that gained experimental
    annotations after T0 in an aspect with none before T0 (NK/LK, as in CAFA).
  - Remove those post-T0 labels from their training label sets, and from the kNN database labels.
  - Report the TV organism mix and identity distribution next to the test's.
  - Add ZFIN dated annotations if a small download is possible, because zebrafish is 45% of test.
  - Keep the random val for early stopping only.
- **P3. Freeze the protocol.** Evaluator = `bench_gogpt.py` settings (S3_all and S6_NK, GO 2023-01-01, CAFA-5 IA,
  prop=max, norm=cafa). Paired bootstrap with 1,000+ reps. The comparison rows are GO-GPT paper 0.651, shipped
  predictions 0.649, public checkpoint 0.687, and the public checkpoint with ten samples (L2).

### Ladder (ordered by expected gain per GPU-hour)

| step | change | budget | keep rule (on TV) | drop rule |
|---|---|---|---|---|
| **L0** | Sharpened kNN family: T, k, coverage weight, per-term support count; database = train(+val for final) | 2 CPU-h | Report the best variant as the mandatory baseline row | Variants within 0.003 of each other: pick the simplest |
| **L1** | Post-hoc stacker: per-term logistic over [kNN score, best identity among hits carrying the term, fraction of hits carrying it, term IA/depth, HiGO prob, has-hit], per aspect, fit on TV | 3 CPU-h | ≥ +0.005 over L0 and remote bins ≥ HiGO+gate − 0.005 | Otherwise keep L0 plus the per-bin blend |
| **L2** | Measure the bar: public GO-GPT with ten-sample frequency scoring (T=0.7, top-k 20) on test; inference only | about 6–10 GPU-h | — | — (defines the target) |
| **L3** | HiGO-35M trained to convergence on GO-GPT data minus TV labels: **C0 frozen + full head** vs **C1 LoRA**; cosine horizon equal to the epochs actually run; 2 seeds | 4 runs × about 7 GPU-h (2 GPUs in parallel) | Keep LoRA if C1 − C0 ≥ 0.01 | If C0 ≈ C1, all scaling goes frozen-encoder (no CRLA, no ladder) |
| **L4** | Retrieval inside the network on the L3 winner: (a) per-term retrieval prior on the logits only; (b) + TabR neighbour attention; identity-capped neighbour sampling; train-vs-train hits (CPU) | 2 × about 8 GPU-h | ≥ +0.01 over **L1** in the 50–95% bins, ≥95% bins not below L0 | Drop (b) if within 0.005 of (a); drop both if (a) does not beat L1 |
| **L5** | Scale: frozen ESM2-3B truncated at layer 30, scalar-mix readout, L4 head; DDP on 2 GPUs (650M instead if disk is freed and 3B is too slow) | about 10–15 GPU-h | ≥ +0.01 over the same head on 35M in the 50–95% bins | Otherwise stay on 35M or 650M |
| **L6** | Loss variants on 35M in parallel with L4/L5: soft-F1 × ZLPR, parent masking, logit adjustment, positive re-weighting | 4 × about 7 GPU-h | ≥ +0.005, 2 seeds agree | Otherwise drop |
| **L7** | InterPro **IDs** as learned tokens / pooling priors (no `interpro2go`) | about 8 GPU-h | ≥ +0.005 | Otherwise drop; never use current mappings in the headline |
| **L8** | Vocabulary: add OOV terms from kNN; then min_count 10 + text-initialised queries for new terms | CPU first, then about 8 GPU-h | ≥ +0.003 overall or ≥ +0.01 high-IA tercile | Otherwise drop |
| **L9** | Adapters (CRLA variant A/C, ladder) on 3B | about 15 GPU-h | Only if L3 showed LoRA matters and L5 used frozen; ≥ +0.005 | Otherwise drop |
| **L10** | Label-query self-attention over top-256 candidates | about 10 GPU-h | ≥ +0.005 | Otherwise drop |
| **L11** | Structure (ProstT5 3Di) | about 3 GB disk + about 10 GPU-h | Remote bins ≥ +0.02 (justified by CAFA-6, not GO-GPT) | Park by default |
| **L12** | Final: retrain on train+val (TV labels restored) with frozen settings; 3 seeds; seed ensemble; one test look; separate GO-GPT-ensemble row | about 20 GPU-h | — | — |

Risks to track:
- The `/tmp` dependency in P0.
- Disk filling during evaluation (`evaluate.py` writes TSVs into `PROC`).
- `train_log.jsonl` append-mode reruns being silently ignored by `cv_summary.py`.
- `strict=False` reloads silently dropping new weights.
- GO-GPT's test overlaps its training set for 760 limited-knowledge proteins. Self-hits are excluded by accession, so
  our kNN does not use their own labels from other aspects. CAFA rules would allow that (inference); if it is used,
  report it separately on the LK subset.

---

## 5. Verdict

The most defensible model design is a **retrieval-conditioned per-term classifier**. It has three parts:
1. A sharpened, coverage-aware kNN label prior, computed against the training set only, enters each GO term's logit
   with a learned per-term trust.
2. A frozen ESM2 encoder (650M or 3B, read at an intermediate layer) supplies the residual evidence through the
   existing HiGO per-term pooling and max-constraint head.
3. Training uses identity-capped neighbour sampling, and every hyperparameter and blend is tuned on a temporal
   validation split built from dated annotations.

It can plausibly beat GO-GPT for a reason we have measured. The zero-training retrieval component alone already ties
GO-GPT: 0.691 val-selected, and 0.685–0.701 across settings, against 0.687. GO-GPT's remaining edge is confined to
the 80–95% and 50–80% identity bins (35% of proteins). There, a model that has GO-GPT's kind of encoder **and**
direct access to neighbour labels should at least match it [inference]. That would put the overall score at about
0.70–0.72 if the retrieval prior is kept at its ≥95% level. This design is also cheap to falsify: steps L0–L4 cost
under 40 GPU-hours.

What we must **not** claim:
- That a new architecture beats GO-GPT, unless the learned model beats the sharpened kNN baseline (and the L1
  stacker) on the same test with a paired bootstrap, and that kNN row is shown in the same table.
- That we beat "BioReason-Pro" in general. That holds only on GO Fmax, only against its reported 0.66, and on a
  different subset.
- Any comparison against the paper's 0.651 alone.
- Any tuning or blend selected on the test set: the 0.711 oracle in §3.2 is off limits.
- Any number that uses current-release InterPro2GO or GOA as if it were leak-free.

---

## Appendix: how the measurements were made
- **Evaluator.**
  - Ground truth: `data/proc_gogpt/gt_test.tsv`, which is already propagated; roots are excluded.
  - Predictions were propagated over is_a and part_of in GO 2023-01-01, or were already ancestor-closed (HiGO with
    MCM, kNN of propagated label sets, and convex blends of the two).
  - IA weights come from the CAFA-5 file. Thresholds run from 0.01 to 1.00 in steps of 0.01. Precision is averaged
    over proteins with ≥1 prediction; recall over all proteins with GT in the aspect.
  - It matches the cafaeval outputs in `runs/bench_gogpt/summary.tsv` to four decimals for three methods.
- **kNN inputs.**
  - Hits: `data/proc_gogpt/diamond_hits_valtest.parquet` (DIAMOND `--more-sensitive`, top 50, e ≤ 1e-3, against the
    119,749-protein train split, self removed).
  - Labels: full propagated `go_bp ∪ go_mf ∪ go_cc` from `/tmp/gogpt_train/.../train-*.parquet`, so they are not
    limited to the vocabulary.
  - Weights: w_j = bits_j · exp((pident_j − max pident)/T) over the top-k hits by bits, normalised; scores clipped at 1.
- **Blends.** s = w_bin · kNN + (1 − w_bin) · (HiGO+gate). Weights come from a grid {0, 0.2, …, 1}, per aspect and
  identity bin, chosen on GO-GPT's random val split (`pred_val.npy`, `diamond_val` hits); the "test-tuned" rows are
  oracles.
- **Bootstrap.** Proteins resampled over the union of aspects, 300 reps, seed 0; Fmax recomputed from cached
  per-protein curves.
- **Organism mix of the test set:**
  - zebrafish 45.2%, rat 19.4%, human 19.1%, mouse 3.2%, fly 2.7%, fission yeast 1.6%;
  - zebrafish proteins have a ≥95% hit 46% of the time vs 60% for the rest;
  - `goa_reference.parquet` covers human, mouse, rat, Arabidopsis, fly, two yeasts, worm, *E. coli* and *M.
    tuberculosis*, with no zebrafish.
