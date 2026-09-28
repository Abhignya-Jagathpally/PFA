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
- **Unused resources.** CD-HIT was not needed (MMseqs2 handles <40% identity). The InterProScan API is too slow
  for the 224k test proteins; section 8 takes precomputed InterPro matches from UniProt/UniParc instead.

## 7. Pilots for a second model version

Four ideas were tried at small scale to decide what goes into the next full run. None of them is a full retrain.

### 7.1 GO-graph attention and slot pooling

HiGO shares information along the ontology with a fixed ancestor-mean operator and the max constraint.
BioReason-Pro (a GAT over the GO graph) and POSA-GO / TRGOA (attention over the partial order) suggest learning
this instead. Separately, pooling over all residues can wash out small functional regions in long multi-domain
proteins; BioBlobs (arXiv:2510.01632) pools residues into a few learned slots without needing a domain database.

Both are opt-in `HiGO` options that are off by default: `use_go_gat` (two `GATConv` layers over the 6,276-term
graph applied to the term queries) and `pooling="slots"` (8 slot queries pool residues with entmax; term
queries then attend over the slots). Pilot: 20k training / 4k validation proteins, 4 epochs, seed 0, validation
IA-weighted Fmax from the training loop (comparable across the three runs, not with the test numbers above).

| run | MF | BP | CC | mean | vs baseline |
|---|---|---|---|---|---|
| baseline | 0.5354 | 0.3389 | 0.5541 | **0.4761** | - |
| + GO-graph GAT | 0.5044 | 0.3283 | 0.5418 | 0.4582 | -0.018 |
| + slot pooling | 0.5186 | 0.3355 | 0.5455 | 0.4665 | -0.010 |

Neither helps at this budget. The GAT result says more about the implementation than the idea: each layer
computes `gelu(LayerNorm(conv(x) + x))`, which rescales the small initial queries to unit variance and clips
their negative part, so the queries are replaced rather than refined (epoch-0 loss 0.49 vs 0.06). The gap
closes every epoch (0.040 to 0.018). A gated residual `q + g * GAT(q)` with `g` starting at 0 is the next thing
to try. Slot pooling is within what one seed can tell apart, and all runs were still improving at epoch 4. A
slots model also loses the residue-level evidence maps from section 4.3.

### 7.2 Ortholog channel

`comparative.py` runs DIAMOND against the training proteins of 10 model organisms (human, mouse, rat, fly,
worm, Arabidopsis, budding and fission yeast, E. coli K-12, M. tuberculosis) and drops same-species hits.
70% of validation and 71% of test proteins get at least one hit.

| method (Kaggle mean) | val | test |
|---|---|---|
| naive | 0.3059 | 0.3164 |
| ortholog kNN alone | 0.3815 | 0.3777 |
| DIAMOND kNN, no naive blend | 0.4365 | 0.4292 |
| DIAMOND + naive | **0.4685** | **0.4614** |
| DIAMOND + orthologs + naive | 0.4644 | 0.4569 |

With the same naive blend on both arms, the ortholog channel lowers the score by 0.004 on both splits. An
earlier version of this analysis compared against the unblended DIAMOND row and reported +0.028; that gain came
from the naive prior. The result is expected in hindsight: the panel is a subset of the training set that
DIAMOND already searches, so the channel re-weights existing neighbours instead of finding new ones, and 82%
of the validation/test proteins come from these same organisms. It will not be added to the model.

### 7.3 Sparse autoencoder on the adapted residue embeddings

Section 4.3 found that masking the top-evidence residues in the input does not change predictions.
SAEs trained on PLM activations (InterPLM, 2024) can split superposed directions into more interpretable
latents, so `sae.py` trains a 256 -> 4096 -> 256 ReLU SAE (L1 3e-3, 4,000 steps) on HiGO's adapted residue
embeddings for 3,000 validation proteins (1.26M residues).

The dictionary did not come out sparse. 371 latents fire on more than 20% of residues and the remaining 3,725
on less than 0.5%, so about 371 latents are active per residue for a 256-dimensional input, and the
reconstruction is essentially lossless (99.9% variance explained). The 50 analysed latents are all from the
rare group:

| latent | most enriched GO term | enrichment | null (random sets) | proteins with term | proteins where latent fires |
|---|---|---|---|---|---|
| 1063 | GO:0004497 monooxygenase activity | 10.15x | 2.7x | 50 | 260 |
| 1021 | GO:0015370 solute:sodium symporter activity | 8.04x | 2.4x | 32 | 338 |
| 2334 | GO:0004497 monooxygenase activity | 7.75x | 2.7x | 50 | 271 |
| 3235 | GO:0015370 solute:sodium symporter activity | 7.68x | 2.7x | 32 | 244 |
| 3191 | GO:0004672 protein kinase activity | 7.51x | 2.7x | 38 | 284 |

Across the 50 latents, best-term enrichment averages 5.52x against 2.59x for random protein sets of the same
size (the best of several hundred terms is inflated even by chance), and every latent is above the 95th
percentile of its permutation null. But the 50 latents cover only 15 terms, 17 of them protein kinase activity,
and fire on about 2 residues per protein, rarely adjacent. So a few directions track large protein families.
This is protein-level correlation in a representation trained on these labels. It does not tell us which
residues drive a prediction, so it does not settle the masking result. Worth repeating with a much stronger
sparsity penalty (or a top-k SAE) and with latents selected on one set and scored on another.

### 7.4 Summary

| pilot | result | next step |
|---|---|---|
| GO-graph GAT | -0.018 val Fmax | gated residual, zero-initialised |
| Slot pooling | -0.010 val Fmax, one seed | more seeds / longer schedule before deciding |
| Ortholog channel | -0.004 Kaggle mean on val and test vs DIAMOND + naive | drop |
| SAE | family-level latents, 2x above permutation null; dictionary not sparse | stronger sparsity, held-out scoring |

None of the four is ready to go into the main model yet. The GAT is the only one with a concrete fix to try.


## 8. HiGO-v2: grounded, hierarchy-consistent GO output

The model output is now a set of GO terms per protein. Every term comes with its ancestors, a calibrated score
and confidence band, and the evidence behind it: InterPro domains, orthologs in reference genomes, and the
ligands (ChEBI with SMILES) that the domains or orthologs imply.

### 8.1 Evidence sources (`grounding_data.py`, `data/grounding/README.md`)

| source | what it gives | built from |
|---|---|---|
| Domains (D) | GO terms implied by the protein's InterPro entries, closed under is_a/part_of | UniProt/UniParc InterPro matches + interpro2go |
| Orthologs (O) | training labels of DIAMOND hits in 10 reference proteomes, same species removed, weighted by identity | `comparative.py` panel |
| Ortholog GOA (G) | GOA annotations of those hits (evaluation proteins excluded, date-filtered for the GO-GPT benchmark) | GOA reference proteomes, 2.35M rows |
| Ligands (L) | whether a term's ligand (ChEBI + SMILES) is supported by the protein's domains or its orthologs' curated binding/cofactor sites | GO logical definitions, Rhea, ExPASy ENZYME, ChEBI; 73k MF term-ligand pairs |

Test proteins only get computed evidence (InterPro matches, DIAMOND hits). Curated UniProt features (binding
sites, cofactors) are only read for training and reference proteins, so no test annotation reaches the
features. Currency metabolites (water, ATP, NAD and the like) are dropped from the Rhea/EC ligand links because they
match almost every enzyme. BRENDA needs a registered account, so Rhea and ExPASy ENZYME stand in for it
(`grounding_data.py` has a BRENDA hook that reads `BRENDA_EMAIL`/`BRENDA_PASSWORD`).

### 8.2 Calibrator (`grounded.py`)

A per-aspect logistic model over 13 features per protein-term pair: the HiGO and kNN logits, max identity, has
hit, D, O, G, L, has ligand, term frequency and three interactions. It is fit on the validation split and
followed by MCM, so every output set is closed under the ontology (0 violations for all variants below).

| method (CAFA-6 test, Kaggle mean) | MF | BP | CC | mean |
|---|---|---|---|---|
| HiGO | 0.5835 | 0.3601 | 0.5793 | 0.5077 |
| HiGO + homology gate | 0.6134 | 0.3793 | 0.5903 | 0.5277 |
| calibrator, gate features only | 0.5878 | 0.3798 | 0.5863 | 0.5180 |
| grounded, no domains | 0.5911 | 0.3792 | 0.5858 | 0.5187 |
| grounded, no ligands | 0.6137 | 0.4121 | 0.5947 | 0.5402 |
| grounded, no orthologs | 0.6169 | 0.4128 | 0.5943 | 0.5413 |
| **HiGO-v2 grounded** | 0.6157 | 0.4121 | 0.5947 | **0.5408** |

Domains carry the gain: +0.013 over the gated model, almost all of it in BP (+0.033). Removing domains drops
the calibrator back to the gate-only level. Orthologs and ligands add nothing measurable on top. The ortholog panel is
a subset of the training set that DIAMOND already searches (section 7.2), and ligand support mostly
re-states domain evidence for MF. Ligands are kept for the output because they make MF predictions checkable
(a heme-binding call with no heme-binding domain or ortholog is easy to spot), not because they raise the
score. Evidence coverage on test: domains 78%, orthologs 71%, ligand support 35-36%.

Fitted weights (standardized features): D 1.3-1.5, O about 0.8, HiGO logit 0.3-0.5, kNN logit only 0.02-0.04
(plus 0.1-0.18 through the identity interaction). The validation split is mostly remote homologs, so the
calibrator learns to trust the kNN score less than it should for close homologs. This shows up in the demo
below and matters for the Kaggle superset, where 88% of proteins have a hit.

### 8.3 Stability

Same pipeline applied to six HiGO models (the dev run and the five CV folds), CAFA-6 test:

| variant | mean Kaggle Fmax | std | predicted-set Jaccard at tau | violations |
|---|---|---|---|---|
| HiGO | 0.4948 | 0.0063 | 0.581 | 0 |
| + homology gate | 0.5180 | 0.0048 | 0.646 | 0 |
| grounded | 0.5323 | 0.0043 | **0.786** | 0 |

Grounding makes the output much more reproducible across training runs: the six models agree on 79% of their
predicted terms, against 58% for the raw model, because shared evidence (domains, orthologs) pulls them toward
the same terms.

### 8.4 GO-GPT benchmark: grounding does not transfer

| method (GO-GPT temporal test, Kaggle mean) | MF | BP | CC | mean |
|---|---|---|---|---|
| HiGO + homology gate | 0.6469 | 0.4337 | 0.6472 | **0.5759** |
| DIAMOND + naive | 0.6495 | 0.4160 | 0.6378 | 0.5678 |
| calibrator, gate features only | 0.6341 | 0.4308 | 0.6336 | 0.5661 |
| grounded, no orthologs | 0.6301 | 0.4287 | 0.6360 | 0.5649 |
| HiGO-v2 grounded | 0.6183 | 0.4058 | 0.6268 | 0.5503 |

On the temporal benchmark the grounded calibrator is 0.026 below the gate. The calibrator puts weight 0.7-0.8
on ortholog GOA (G) for MF/BP, which is fitted on a random validation split where homologs' GOA and the
target's labels come from the same annotation period. On a temporal holdout, new annotations are exactly the
ones homologs do not have yet, so that weight is too high. The variant choice was made on validation. Picking a
variant from these test numbers would be selection on test, so the table is reported as is. Fitting the
calibrator on a time-split validation set is the obvious fix.

A second caveat applies to both benchmarks: the interpro2go and GOA files are current releases, which postdate
the CAFA-6 split and the GO-GPT cutoff (GOA is date-filtered for GO-GPT, interpro2go is not). Part of the
domain gain may come from mappings curated after the test annotations.

### 8.5 Precision, recall and F1 per aspect (`prf_table.py`, `runs/prf/tables.md`)

IA-weighted precision, recall and F1 at each model's own Fmax threshold (cafaeval, prop=max, norm=cafa);
General is the mean over MF, BP and CC.

CAFA-6 clustered test (8,058 proteins):

| Model | MF P | MF R | MF F1 | BP P | BP R | BP F1 | CC P | CC R | CC F1 | General P | General R | General F1 | Kaggle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.727 | 0.242 | 0.363 | 0.222 | 0.227 | 0.224 | 0.320 | 0.415 | 0.362 | 0.423 | 0.295 | 0.316 | 0.316 |
| DIAMOND + naive | 0.671 | 0.503 | 0.575 | 0.383 | 0.287 | 0.328 | 0.565 | 0.419 | 0.481 | 0.540 | 0.403 | 0.461 | 0.461 |
| Frozen ESM2 + MLP | 0.659 | 0.494 | 0.565 | 0.337 | 0.329 | 0.333 | 0.559 | 0.507 | 0.532 | 0.518 | 0.444 | 0.477 | 0.477 |
| Frozen ESM2 + MLP + homology gate | 0.693 | 0.535 | 0.604 | 0.391 | 0.337 | 0.362 | 0.565 | 0.540 | 0.552 | 0.550 | 0.471 | 0.506 | 0.506 |
| HiGO (LoRA) | 0.677 | 0.513 | 0.584 | 0.384 | 0.339 | 0.360 | 0.621 | 0.543 | 0.579 | 0.561 | 0.465 | 0.508 | 0.508 |
| HiGO + homology gate | 0.697 | 0.548 | 0.613 | 0.412 | 0.352 | 0.379 | 0.617 | 0.566 | 0.590 | 0.575 | 0.488 | 0.528 | 0.528 |
| **HiGO-v2 grounded (proposed)** | 0.663 | 0.575 | 0.616 | 0.441 | 0.387 | 0.412 | 0.625 | 0.567 | 0.595 | 0.576 | 0.510 | 0.541 | 0.541 |

GO-GPT temporal test (8,630 proteins):

| Model | MF P | MF R | MF F1 | BP P | BP R | BP F1 | CC P | CC R | CC F1 | General P | General R | General F1 | Kaggle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.182 | 0.187 | 0.184 | 0.287 | 0.165 | 0.210 | 0.301 | 0.383 | 0.337 | 0.257 | 0.245 | 0.244 | 0.244 |
| DIAMOND + naive | 0.649 | 0.650 | 0.649 | 0.375 | 0.468 | 0.416 | 0.614 | 0.663 | 0.638 | 0.546 | 0.594 | 0.568 | 0.568 |
| GO-GPT (beam) | 0.757 | 0.676 | 0.714 | 0.631 | 0.604 | 0.617 | 0.684 | 0.779 | 0.728 | 0.690 | 0.687 | 0.687 | 0.687 |
| GO-GPT (greedy) | 0.756 | 0.679 | 0.716 | 0.627 | 0.605 | 0.616 | 0.683 | 0.779 | 0.728 | 0.689 | 0.688 | 0.686 | 0.686 |
| HiGO (LoRA) | 0.529 | 0.478 | 0.502 | 0.354 | 0.326 | 0.339 | 0.516 | 0.529 | 0.522 | 0.467 | 0.444 | 0.455 | 0.455 |
| HiGO + homology gate | 0.639 | 0.655 | 0.647 | 0.431 | 0.437 | 0.434 | 0.647 | 0.648 | 0.647 | 0.572 | 0.580 | 0.576 | 0.576 |
| **HiGO-v2 grounded (proposed)** | 0.637 | 0.601 | 0.618 | 0.382 | 0.433 | 0.406 | 0.592 | 0.666 | 0.627 | 0.537 | 0.566 | 0.550 | 0.550 |

The grounded model trades a little MF precision for recall (MF threshold 0.17 vs 0.23) and gains in BP on both precision and recall. On the GO-GPT set it loses MF recall and BP precision, which fits the ortholog-GOA overweighting in section 8.4. Unweighted versions are in `runs/prf/tables.md`.

### 8.6 Output format (`predict_grounded.py`)

- `--split test`: `runs/grounded/output/test_grounded.{jsonl,tsv}`, one record per protein with every
  predicted term, its ancestors, score, confidence band (high >= 0.7, medium >= 0.4, low) and evidence.
- `--fasta FILE`: runs InterPro lookup (UniParc by sequence MD5, InterProScan as fallback), DIAMOND against
  all labelled proteins, and HiGO with attention; writes a JSON and a Markdown report per sequence.
- `--submission`: grounded Kaggle submission for the 224,309-protein superset
  (`runs/grounded/submission/submission.tsv.gz`, 382 MB, 84,991,864 rows, top 500 terms with score >= 0.02;
  every row has three fields, a valid GO id and a score in [0, 1]; all proteins present). Not uploaded (needs
  the Kaggle token).

On the CAFA-6 test split the grounded output averages 50 terms per protein, ancestors included, with 0 closure
violations. Of the specific terms, 82% have ortholog support, 12% domain support and 2% ligand support.

Demo (`benchmarks/globin_demo.fasta`, the 147-aa sequence from `GOGPT_output.docx`, identical to human
hemoglobin epsilon P02100): 27 terms, 13 of them specific, 0 violations. Oxygen binding (GO:0019825) goes
from 0.16 (HiGO) to 0.42 after grounding, supported by the globin domain IPR012292, three orthologs at 80-100%
identity, and ligand O2 (`O=O`). The CC and BP calls (cytosol 0.68, transport 0.69, protein-containing complex 0.58)
match hemoglobin. Heme binding stays under the MF threshold. P02100's experimental training labels (14 terms)
do not include it, and the calibrator underweights the 100%-identity neighbour (section 8.2). Oxygen carrier
activity (GO:0005344) is not in HiGO's 6,276-term vocabulary. For comparison, the report in `GOGPT_output.docx`
for the same sequence calls it a 106-aa adaptor (it has 147 residues). It proposes protein binding, DNA binding and
signal transduction with low confidence and misses the globin fold. Several of its GO IDs are wrong:
GO:0023613 ("signal transduction") does not exist (the term is GO:0007165), and GO:0048519, given as the
biological process root, is negative regulation of biological process.

### 8.7 Does fine-tuning a PLM with the GO hierarchy pay off?

- **Fine-tuning vs a frozen PLM.** With the same head, LoRA on ESM2-35M beats the frozen backbone by 0.031
  (0.508 vs 0.477) without the gate and 0.022 (0.528 vs 0.506) with it. Most of that gain is on proteins
  without a close homolog.
- **The hierarchy inside the model.** At convergence the MCM constraint is worth 0.012 on validation. On test,
  after the homology gate, the hierarchy constraint keeps 0.004 and ancestor-shared queries give 0.000.
  The main value of the hierarchy is that the output is always consistent (0 violations everywhere), which is
  what makes the grounded output readable: every specific term comes with its path to the root.
- **The hierarchy outside the model.** Domain evidence closed under the ontology adds another 0.013 on the
  clustered test split (0.541) and raises cross-run agreement from 0.65 to 0.79.
- **Against GO-GPT.** On GO-GPT's temporal test set, where 88% of proteins have a close homolog, neither
  fine-tuning nor grounding closes the gap to the 3B generative model (0.576 / 0.550 vs 0.687).

So yes, but with limits. A small hierarchy-aware fine-tune plus grounding gives the best results on remote
homologs and the most stable, checkable output. It does not beat a much larger model on close homologs, where
homology transfer and backbone size matter most. The next things to try are a time-split validation set for the
calibrator, letting it trust near-identical neighbours, and a larger backbone.
