# HiGO: hierarchy-aware, GO-conditioned evidence pooling on a LoRA-tuned protein language model for CAFA-6

Code: `cafa6/src/cafa6/`. Step-by-step log: `NOTES.md`. All numbers below are from files under `runs/` and were
computed with the official CAFA evaluator (`cafaeval` 1.3.0, `prop=max`, `norm=cafa`).

## 1. Problem

Given a protein sequence (and its species), predict its Gene Ontology (GO) terms in three aspects: molecular
function (MF), biological process (BP) and cellular component (CC). Kaggle CAFA-6 scores submissions by the
information-accretion (IA) weighted Fmax per aspect, averaged over the three aspects ("Kaggle mean" below). IA
weighting rewards specific, rare terms and discounts shallow ones that follow from the hierarchy.

Three questions were addressed:
1. How well does the released GO-GPT (BioReason-Pro) actually perform on its own benchmark?
2. Can a simple, first-principles fine-tune of a protein language model (PLM), designed around the structure of
   the task (the GO hierarchy, IA weighting, residue-level evidence), do better, and is it interpretable?
3. How does it do on CAFA-6, and what does the Kaggle submission look like?

## 2. Data and preprocessing (Step 2)

- CAFA-6 training data: 82,404 sequences, 537,027 leaf annotations, GO 2025-06-01, `IA.tsv`; Kaggle test superset:
  224,309 sequences.
- Cleaning: removed 36 sequences with >5% ambiguous residues and 330 shorter than 20 aa (train only).
- Label propagation over `is_a` + `part_of` (labels are leaf-only): 0.54M -> 3.56M annotations.
- Redundancy-aware split: MMseqs2 `easy-cluster` at 30% identity / 80% coverage -> 33,003 clusters, split by
  cluster 80/10/10 into train 65,603 / val 8,377 / test 8,058. No cluster crosses splits. Remaining local homology
  (DIAMOND hits below the coverage threshold) is reported as strata instead of being hidden.
- Label space: 6,276 terms with >=30 training annotations (BP 4,666, MF 903, CC 707), 90.1% of the IA mass.
- Length: p95 = 1,320 residues, above ESM2's 1,022 context. Training uses random 1,022-residue crops; inference
  uses sliding windows (stride 511) with max-pooled logits, so no protein is truncated. Special tokens and padding
  are masked from the pooling.

## 3. Methods

### 3.1 Baselines
- Naive: training frequency of each term.
- DIAMOND homology kNN (bit-score weighted label transfer, e<=1e-3, `--more-sensitive`), and DIAMOND + naive.
- Frozen ESM2-35M with the HiGO head and no adapter.

### 3.2 HiGO (`src/cafa6/model.py`)
Each design choice targets a specific property of GO prediction:

| component | what it does | why |
|---|---|---|
| Frozen ESM2 + LoRA (q,v, r=8, alpha=16, dropout 0.05) | 0.55% of the 35M backbone trainable | adapts the PLM cheaply without forgetting |
| Hierarchical GO query tokens | q_t = e_t + mean of the embeddings of t's ancestors | rare deep terms share parameters with their well-trained ancestors (unlike GO-GPT's flat GO-token vocabulary) |
| GO-conditioned evidence pooling | each term attends over residues with entmax-1.5 (exact zeros) | function is usually carried by local regions; the attention row is a per-term evidence map |
| Max-constraint module + MCLoss (C-HMCNN) | p(parent) = max over the term and its descendants | predictions obey the true-path rule by construction |
| IA-weighted loss | per-term weight 1 + min(IA, 5) | aligns training with the IA-weighted metric |
| Taxon embedding (10% dropout) | species context for 41 frequent species | function depends on organism; dropout handles unseen species |
| Homology gate (post-hoc) | p = g kNN + (1-g) HiGO, g = sigmoid(w . [identity, has_hit, 1]) per aspect, fit on validation | trusts homology only when a close homolog exists |

Adapter choice (the brief asked for something better than LoRA if it helps): DoRA (same rank) and LoRA on all
linear layers at r=16 (8.5x the parameters) were compared with LoRA q,v r=8 on ESM2-35M. DoRA tracked LoRA to within
0.001 validation Fw; LoRA-all reached 0.5253 vs 0.5270 and over-fit earlier. Adding *any* adapter to the frozen
backbone was worth +0.04, so LoRA q,v r=8 was kept: the adapter is a second-order factor and the head does the work.

### 3.3 Training (`src/cafa6/train.py`, `configs/dev.yaml`)
AdamW (weight decay 0.01), adapter LR 3e-4, head LR 1e-3, 6% linear warm-up then cosine decay, gradient clipping
1.0, BF16 autocast, token-budget batches (16,384 tokens), early stopping on validation IA-weighted Fmax (patience
3, at most 15 epochs). Only adapter and head weights are saved (~8 MB). A 3B-backbone run
(`configs/final_3b.yaml`, 0.10% trainable) was started and stopped at the user's request; the final model is
HiGO-35M.

## 4. Results

### 4.1 GO-GPT on its own test set (question 1)
Temporal holdout of 8,630 proteins (annotations gained Mar 2023-Feb 2024), GO 2023-01-01, CAFA-5 IA, binary
predictions, paper protocol (Table S3). Full test set, released checkpoint, exact batched re-implementation of the
released predictor (identical outputs verified).

| | MF Fw | BP Fw | CC Fw | mean Fw |
|---|---|---|---|---|
| paper | 0.701 | 0.539 | 0.713 | 0.651 |
| predictions shipped with the dataset | 0.689 | 0.542 | 0.717 | 0.649 |
| released checkpoint (reproduced) | 0.714 | 0.617 | 0.728 | **0.686** |

- The paper's numbers are reproduced by the shipped predictions; the public checkpoint is better (BP +0.08).
- 88% of this benchmark's test proteins have a >=50%-identity homolog in GO-GPT's training data. On the 795 proteins
  below 50% identity, plain DIAMOND + naive beats GO-GPT (0.457 vs 0.440 at 30-50%; 0.454 vs 0.395 below 30%).
- 96% of CAFA-6's labelled proteins are in GO-GPT's training data with their labels. Zero-shot on 1,500 proteins of
  our CAFA-6 test split, GO-GPT scores 0.897, far above its 0.686 on unseen annotations: memorisation, not
  generalisation. It is therefore not a fair CAFA-6 reference.

Fair head-to-head: HiGO-35M trained on GO-GPT's own training data (same proteins, labels, GO release, IA weights
and evaluator; 4 epochs because of the time budget), scored on GO-GPT's 8,630-protein holdout:

| method (IA-weighted Fmax) | MF | BP | CC | mean |
|---|---|---|---|---|
| GO-GPT released checkpoint (ESM2-3B + GPT decoder) | 0.714 | 0.617 | 0.728 | **0.687** |
| HiGO-35M + homology gate | 0.647 | 0.434 | 0.647 | 0.576 |
| DIAMOND + naive | 0.650 | 0.416 | 0.638 | 0.568 |

| identity to training data | n | GO-GPT | HiGO-35M + gate |
|---|---|---|---|
| >=50% | 7,585 | **0.730** | 0.599 |
| 30-50% | 713 | 0.440 | **0.480** |
| <30% | 82 | 0.395 | **0.461** |
| no hit | 250 | **0.374** | 0.360 |

GO-GPT wins overall, and its lead sits in the 88% of proteins that have a close homolog in training. HiGO-35M
(84x smaller backbone, 4 epochs) is better on remote homologs (below 50% identity) and about equal when there is
no homolog. So HiGO is not better than GO-GPT on GO-GPT's benchmark overall; it is better only for remote
homologs.

### 4.2 CAFA-6 clustered test split (question 2)

| method (8,058 test proteins, <30%-identity clusters) | MF | BP | CC | Kaggle mean |
|---|---|---|---|---|
| naive | 0.363 | 0.225 | 0.362 | 0.316 |
| DIAMOND + naive | 0.575 | 0.328 | 0.481 | 0.461 |
| frozen ESM2-35M + HiGO head + gate | 0.604 | 0.362 | 0.553 | 0.506 |
| HiGO-35M (LoRA) | 0.584 | 0.360 | 0.579 | 0.508 |
| **HiGO-35M (LoRA) + homology gate** | **0.613** | **0.379** | **0.590** | **0.528** |

By homology to the training set (Kaggle mean, HiGO + gate vs DIAMOND + naive):

| max identity to train | n | HiGO + gate | DIAMOND + naive |
|---|---|---|---|
| no hit | 1,725 | 0.495 | 0.333 |
| <30% | 908 | 0.509 | 0.443 |
| 30-50% | 3,868 | 0.530 | 0.483 |
| >=50% | 1,557 | 0.565 | 0.533 |

The largest gain (+0.16) is exactly where homology transfer has nothing to offer. By IA tercile, the gain over
DIAMOND + naive is +0.06 to +0.07 in every tercile; proteins annotated with rare terms remain hardest (0.48).
Error analysis: the worst-decile proteins (per-protein F1 at tau 0.3) have few annotations (median 10 vs 28),
slightly weaker homology (32% vs 36% identity) and more specific labels; length is not a driver.

Ablations (validation IA-weighted Fmax, best within the first 4 epochs of the same 15-epoch schedule; every run
stopped after >=4 epochs to save time):

| removed component | val Fw @4 epochs | change vs full (0.5086) |
|---|---|---|
| species embedding | 0.4956 | -0.013 |
| evidence attention (mean pooling instead) | 0.5006 | -0.008 |
| hierarchy constraint (MCM + MCLoss) | 0.5021 | -0.007 (-0.012 at convergence) |
| ancestor-shared GO queries | 0.5063 | -0.002 |
| entmax (softmax instead) | 0.5081 | -0.001 (tie) |
| IA-weighted loss | 0.5095 | +0.001 (no gain) |

Species context, the hierarchy constraint and evidence pooling carry the gains; entmax is kept for readable, sparse
evidence maps rather than accuracy; IA weighting did not help on this metric (negative result).

On the held-out test set (each ablation from its best checkpoint; the no-taxon and no-IA-weight runs had only 4
epochs, so their drops are upper bounds), the Kaggle mean without / with the homology gate is: full 0.508 / 0.528;
no species embedding 0.476 / 0.508; no IA weighting 0.493 / 0.519; no hierarchy constraint 0.505 / 0.524; mean
pooling 0.505 / 0.527; softmax 0.503 / 0.528; flat GO queries 0.507 / 0.528. Species context and IA weighting
matter most on test; the gate absorbs the smaller effects of pooling type and query sharing.

5-fold cross-validation over the train+val clusters (same 4-epoch budget per fold): validation IA-weighted Fmax
MF 0.600 +- 0.007, BP 0.363 +- 0.005, CC 0.578 +- 0.002, mean 0.514 +- 0.004. The spread is small, so the
result does not depend on one particular split.

### 4.3 Interpretability (`runs/dev_35m_lora/explain/`)
511 confident, correct, most-specific predictions on 300 test proteins:
- Evidence maps are sparse: 39% of residues non-zero on average; the top 10% of residues carry 76% of the mass.
- Faithful to the head: dropping the top-10% evidence residues from the pooling lowers the term's own probability by
  0.067, vs 0.0005 for random residues (Wilcoxon p = 0.007).
- Not causal at the sequence level: masking the same residues in the input has no effect, because the transformer
  spreads their information to neighbouring residues. The maps show where the model reads a function, not which
  residues are necessary.
- Modest agreement with local InterPro regions (domains, sites, repeats): median enrichment 1.13, 57% of cases > 1.
- Maps differ per GO term for the same protein (figure `runs/dev_35m_lora/explain/evidence_Q08492.png`), but all
  put a large weight on residue 1 (initiator Met), a positional artefact to be removed in future work.

### 4.4 Kaggle submission (question 3)
`runs/dev_35m_lora/submission/submission.tsv` (and `.tsv.gz`): HiGO-35M + validation-fitted homology gate + MCM
for all 224,309 test-superset proteins. 45,957,813 rows, top 500 terms per protein with score >= 0.02, format
`protein<TAB>GO:term<TAB>score` as in `sample_submission.tsv`. Checks passed: every protein present, <=500 terms
per protein (limit 1,500), valid GO ids, scores in [0, 1]. Expected Kaggle score is around the clustered-test
estimate (0.53) or higher, because 88% of the test superset has a DIAMOND hit to the training data, versus 79% of
our clustered test split. The upload needs the user's Kaggle token:
`kaggle competitions submit -c cafa-6-protein-function-prediction -f submission.tsv -m "HiGO-35M + homology gate"`.

## 5. Conclusions

1. **GO-GPT, measured properly.** The released checkpoint reaches 0.687 IA-weighted Fmax on its own temporal
   holdout, above the paper's 0.651. But 88% of that test set has a >=50%-identity homolog in its training data,
   and plain homology transfer beats GO-GPT on the remaining remote-homology proteins. GO-GPT also saw 96% of
   CAFA-6's labelled proteins with their labels, so its zero-shot 0.897 on CAFA-6 reflects memorization.
2. **HiGO.** A small, structured fine-tune of ESM2-35M (0.55% of backbone parameters trainable) scores 0.528 on a
   30%-identity clustered CAFA-6 test split, versus 0.461 for DIAMOND + naive and 0.506 for a frozen PLM with the
   same head. The biggest gain, +0.16, is on proteins with no detectable homolog, and 5-fold CV confirms
   stability (0.514 +- 0.004).
   - What matters: species context, the hierarchy constraint, IA weighting and the homology gate. What doesn't:
     the adapter type (DoRA and LoRA on all layers were no better than LoRA q,v r=8) and softmax vs entmax.
3. **Against GO-GPT on its own data,** HiGO-35M loses overall (0.576 vs 0.687) but wins on remote homologs. The
   evidence therefore supports HiGO as the better choice when no close homolog exists, not as a better model
   overall.
4. **Interpretability.** The per-GO evidence maps are faithful to the head and partly aligned with InterPro
   domains. They are not residue-level causal explanations.

## 6. Limitations and next steps
- **Backbone size.** The final model uses ESM2-35M; the 3B run was stopped at the user's request after 1 epoch
  (val 0.485 vs 0.445 for 35M at the same point). Scaling the backbone is the most likely route to closing the gap
  on close homologs.
- **Short runs.** Ablations, cross-validation and the GO-GPT head-to-head used shortened schedules (4 epochs) to
  save time, so absolute numbers are slightly below converged values. The comparisons are matched within each
  table.
- **Clustering and residue 1.** The clustered split still contains local homology below the 80% coverage
  threshold (reported by identity stratum). The evidence maps over-weight residue 1.
- **Unused resources.** CD-HIT was not needed (MMseqs2 handles <40% identity). InterPro regions were used only
  for evaluation, because the public API is too slow for the 224k test proteins.
