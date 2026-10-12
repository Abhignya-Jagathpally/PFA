# Data, training objectives and evaluation for a GO model that beats GO-GPT / BioReason-Pro

Scope: what data to train on, which losses and objectives to use, and how to evaluate so that a claim of "beats GO-GPT (0.687 IA-weighted Fmax on the BioReason-Pro temporal holdout)" holds up. Source labels: **[preprint]** = not peer reviewed (arXiv/bioRxiv); **[Kaggle]** = competition write-up, not peer reviewed; **[peer-reviewed]** = journal/conference. Items under "Inferences" are my reasoning, not source claims.

---

## 1. Training data: IEA, GOA at scale, InterPro2GO pseudo-labels, self-training, and leakage control

### Takeaway
The strongest CAFA-5 teams did not add electronic (IEA) labels as training targets; they used them as **input features** to a model that predicts experimental labels, and this was worth about +0.02 wFmax for the 2nd-place team. GO-GPT's authors report that both adding electronic annotations as targets and SwissProt pretraining failed to help on the same temporal holdout we target. Leakage control should be done by rebuilding every feature source (GOA, IEA, InterPro, homology DB) from snapshots dated no later than Nov 2022 / Jan 2023.

### Cited Findings
- GO-GPT / BioReason-Pro training labels came from the **GOA November 2022 release**, using evidence codes IDA, IPI, EXP, IGI, IMP, IEP, IC, TAS, propagated over is_a and part_of, with GO IDs standardized to the January 2023 ontology and **CAFA5 IA weights used unchanged** — [BioReason-Pro paper, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- The temporal test set was built from GOA/UniProt at two time points (Nov 2022 and Feb 2024), retaining proteins that gained experimental annotations Mar 2023–Feb 2024 and lacked annotation in the target aspect before; it includes both no-knowledge and limited-knowledge proteins (8,630 proteins, 230,824 propagated annotations, mean 26.75 terms per protein) — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- GO-GPT's validation set was a **random 90/10 split** of pre-Nov-2022 proteins, not a temporal split; the best checkpoint was chosen by validation performance — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- "Design choices that did not help" for GO-GPT: (a) pretraining on all reviewed SwissProt proteins and GO annotations, then fine-tuning on CAFA5 training data, did not improve the temporal holdout, possibly because SwissProt contains electronic plus experimental annotations with different distribution/specificity; (b) adding electronic annotations as training targets for CAFA5 training proteins did not improve the temporal holdout ("may introduce noise and distribution shifts"); (c) initializing GO token embeddings from term name/definition text gave no measurable gain; (d) beam search did not beat sampling with probability aggregation — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- GO-GPT used a vocabulary of terms with ≥20 training proteins (~10,000 tokens in their description), stated this was not systematically ablated; intermediate ESM2 layers (~80–90% depth, e.g., layer 30/36 of ESM2-3B) worked better than the final layer; an organism embedding helped — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- CAFA-5 2nd place (ProtBoost/U900): split GOA evidence codes into experimental vs electronic and used **electronic labels as features** for predicting experimental labels; "about 30% of electronic labels becomes experimental, so using it as model feature performs better than just adding it as is" — [U900 write-up, Kaggle](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/writeups/u900-private-2nd-public-5th-solution-py-boost-and-)
- ProtBoost ablation (leaderboard scores): Py-Boost on ProtT5 + conditional probability modelling 0.51 → +taxon one-hot 0.53 → conditional probability modelling for Py-Boost and LogReg 0.55 → **+GOA electronic annotations 0.57** → +NN ensemble 0.59; GCN stacking over the GO graph took the result from 0.59 to 0.62; final private score 0.58240 — [ProtBoost, arXiv preprint](https://arxiv.org/html/2412.04529)
- ProtBoost parsed GOA from the dated archive `ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/` — [ProtBoost, preprint](https://arxiv.org/html/2412.04529)
- CAFA-5 3rd place (tito): non-experimental GOA labels (11 non-experimental evidence codes) one-hot encoded into a (batch, 11, n_labels) tensor and processed by a kernel-size-1 1D-CNN as **features**, not used as training targets; also used T5/ESM2 embeddings and taxonomy one-hot; validation was **time-based** (experimental annotations added after the Nov 2022 training snapshot, excluding sub-ontologies already annotated) — [tito write-up, Kaggle](https://www.kaggle.com/its7171/writeups)
- CAFA-5 1st place (GOCurator, Fudan) was built on NetGO 3.0, combining structure-, sequence-, text- and literature-based components with learning-to-rank; it won by 5.8% over 2nd place — [ML4PE seminar abstract](https://www.ml4proteinengineering.com/mar-5). Its key new component GORetriever (literature retrieval + reranking) raised NetGO3.0 wFmax from 0.587 to 0.604 — [GORetriever, Bioinformatics, peer-reviewed](https://pmc.ncbi.nlm.nih.gov/articles/PMC11520413/)
- NetGO 3.0 = NetGO 2.0 components (Naïve, BLAST-KNN, LR-3mer, LR-InterPro, Net-KNN, LR-Text) + LR-ESM (logistic regression on mean-pooled ESM-1b), combined via learning-to-rank; its protocol used training = experimental annotations before Jan 2020, **validation = NK and LK proteins annotated Jan–Dec 2020**, test = NK proteins Jan–Dec 2021 — [NetGO 3.0, GPB, peer-reviewed](https://pmc.ncbi.nlm.nih.gov/articles/PMC10626176/)
- NetGO 3.0 reports 95% CIs from bootstrapping with only 100 iterations — [NetGO 3.0](https://pubmed.ncbi.nlm.nih.gov/37075830/)
- DeepGOZero (InterPro binary vector input + ELEmbeddings) predicts terms with zero training annotations via ontology axioms — [DeepGOZero, Bioinformatics 2022](https://europepmc.org/article/med/35758802)

### Inferences
- **Recommended data recipe (inference):** keep experimental-only targets (matching GO-GPT and the test labels), and feed IEA/InterPro2GO/homology-transferred labels as *input channels* (per-term prior vector or the tito-style evidence-code tensor). This replicates what worked for CAFA-5 #2 and #3 and avoids the failure GO-GPT reported.
- **Leakage rule (inference):** every feature source must be frozen at or before the GO-GPT training cutoff. Concretely: GOA from `goa/old/UNIPROT/` at the Nov 2022 (or earlier) release; InterPro/InterPro2GO release from ≤ Jan 2023; DIAMOND database built only from proteins/labels present in the Nov 2022 snapshot. IEA labels in a 2024/2025 GOA release are partly *derived from* the very experimental annotations in the test window (ProtBoost's "30% of electronic labels become experimental" shows how strongly they overlap), so using a current GOA as a feature would inflate scores.
- Our 2.35M-row GOA of 10 reference proteomes must be checked for its release date; if it postdates Feb 2023, it cannot be used for the GO-GPT benchmark without re-downloading the dated snapshot.
- The 224k Kaggle superset / UniRef50 can be used for self-training (pseudo-labels from a teacher trained on ≤Nov-2022 labels) without leakage as long as test proteins are removed or their pseudo-labels come from a teacher that never saw post-cutoff labels. Teacher-generated pseudo-labels for test proteins themselves are transductive; report such runs separately.
- Disk budget (~4 GB) argues for precomputed, quantized (fp16) mean-pooled embeddings and sparse label/feature matrices rather than per-residue caches.

### Gaps
- I found no published ablation that measures self-training/distillation on UniRef50 for GO prediction under a temporal split; its value is untested here.
- The GOCurator Kaggle discussion post (#466917) was not fetched; component-level ablations of the 1st-place solution are not in these notes.
- BioReason-Pro lists 8 evidence codes and omits the CAFA-5 high-throughput codes (HTP, HDA, HMP, HGI, HEP). Whether the test set included HTP-coded annotations should be verified in `wanglab/bioreason-pro-test-data` before matching label filters.
- Whether GO-GPT's released checkpoint was trained on more than the published 119,749 proteins could not be confirmed from the paper text.

---

## 2. Incomplete labels: PU learning, ancestor smoothing, NOT annotations, NK vs LK

### Takeaway
GO labels are open-world: absence is not negation. Practical, low-risk tools are (a) masking loss on child terms whose parents are all negative (ProtBoost), (b) weak negatives / label smoothing / expected-positive regularization from the missing-label literature, and (c) using the rare NOT annotations as hard negatives that propagate *down* the ontology. Evaluation must state which knowledge type (NK/LK) is scored, because the GO-GPT test set mixes both.

### Cited Findings
- GO uses an open-world model; absence of an annotation does not imply the protein lacks the function — [GO docs](http://geneontology.org/docs/go-annotations/)
- NOT annotations mean a gene product was experimentally shown not to have a function (or lost an essential active site); NOT is not used for negative/inconclusive results; **NOT statements propagate down** to more specific terms, opposite to positive annotations — [GO docs](http://geneontology.org/docs/go-annotations/)
- In UniProt-GOA, fewer than 2.5% of experimentally annotated proteins have any NOT-qualified annotation — [Warwick Vesztrocy & Dessimoz, "Benchmarking GO predictions using negative annotations"](https://exa.ai/library/publication/3hwscqfjhj9)
- Heuristic negatives produce false negatives that hurt classifiers; NoGO provides curated negative examples for several model organisms — [Youngs et al., PLOS Comput Biol 2014](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003644)
- ProtBoost excludes from the loss any (protein, term) pair where all GO parents are zero (treated as NA), i.e., conditional-probability modelling P(term | parents) — [ProtBoost, preprint](https://arxiv.org/html/2412.04529)
- Missing-label multi-label learning: the "assume negative" loss is biased by false negatives even when they are few; mitigations are weak negatives (down-weighted negatives), label smoothing, expected positive regularization (EPR: positive-only BCE + a batch penalty keeping predicted positive count near k), and ROLE (online estimation of missing labels) — [Cole et al., CVPR 2021, peer-reviewed](https://arxiv.org/html/2106.09708v2)
- CAFA defines No-Knowledge (no prior experimental annotation in any aspect) and Limited-Knowledge (prior experimental annotation only in other aspects) benchmarks; CAFA3's headline figures used NK in full mode, LK in supplement — [CAFA3, Genome Biology 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6864930/)
- InterLabelGO+ uses the same NK/LK definitions for its validation/test — [InterLabelGO+, Bioinformatics 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11568131/)
- GO-GPT's 0.65–0.70 numbers are on the mixed NK+LK set; BioReason-Pro itself was evaluated on NK only because GPT-5 traces could leak cross-aspect information — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)

### Inferences
- For LK proteins, the protein's *pre-cutoff experimental annotations in other aspects* are legitimate inputs (CAFA allows them). A model that conditions on them (e.g., known CC when predicting BP) has an advantage on LK proteins only; report NK and LK separately so a gain is not attributed to architecture when it comes from LK side information.
- Recommended default (inference): BCE/ZLPR with weak negatives (negative weight 0.1–0.5 for terms whose siblings/parents are positive, full weight elsewhere) + ProtBoost-style parent masking. EPR/ROLE are worth an ablation but Cole et al. found EPR alone weak.
- Ancestor label smoothing: put a small soft target on children of observed leaf terms whose prior P(child | parent) is high in training; tune on the temporal validation set, not the random split.
- NOT annotations are too rare (<2.5% of proteins) to move Fmax much; use them as hard negatives mostly to stop homology transfer from propagating known-absent functions.

### Gaps
- No source quantifies PU-learning gains on CAFA temporal benchmarks specifically.
- Fraction of the GO-GPT test set that is NK vs LK was not found in the text read; compute it from the HF test dataset.

---

## 3. Objectives aligned with IA-weighted Fmax

### Takeaway
Loss choices with direct evidence on CAFA-style data: IA-weighted protein-centric soft-F1 *multiplied with* ZLPR or BCE (InterLabelGO+), and hierarchy-aware stacking/post-processing (ProtBoost GCN, CAFA "max" propagation). Soft-F1 alone performs poorly. Hierarchy-coherent outputs (MCM/MCLoss) and logit adjustment are well grounded in the ML literature but lack published CAFA temporal ablations.

### Cited Findings
- InterLabelGO+ loss: IA-weighted F1 loss (protein-centric PTF1 and GO-centric GOF1) combined with ZLPR (zero-bounded log-sum-exp & pairwise rank) loss; adding PTF1 or GOF1 to BCE improved results; ZLPR beat BCE and was further improved by combining with PTF1/GOF1; **PTF1 or GOF1 alone performed poorly** — [InterLabelGO+, Bioinformatics 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11568131/)
- InterLabelGO+ combines DL and DIAMOND AlignmentKNN with weights depending on mean top sequence identity; this dynamic weighting raised CAFA5 late-submission wFmax from 0.5604 to 0.5764 (would have placed 3rd) — [InterLabelGO+](https://academic.oup.com/bioinformatics/article-abstract/40/11/btae655/7876260)
- A Freddolino-lab CAFA-5 team (6th) used a "soft F1 loss" that markedly improved over BCE — [ProtBoost related-work summary, preprint](https://arxiv.org/html/2412.04529)
- sigmoidF1: smooth F1 surrogate computed over a batch (sigmoid-thresholded confusion matrix; minimize 1−F1~); any confusion-matrix metric can be smoothed this way — [Bénédict et al., TMLR 2022](https://arxiv.org/html/2108.10566)
- BioReason-Pro's RL reward is the IA-weighted F1 between predicted and true propagated term sets, "directly optimizing for the same weighted metric used in evaluation" — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- C-HMCNN: max-constraint module makes each parent score ≥ max of its descendants (no hierarchy violations at any threshold); MCLoss lets the parent's positive term be delegated to children — [Giunchiglia & Lukasiewicz, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)
- ProtBoost post-processing: soft root-to-leaf and leaf-to-root propagation with 0.7/0.3 parent/child averaging; GCN stacking on the GO graph with per-model logits, two propagated variants, a prior flag, the electronic-annotation feature and trainable term embeddings (29 features), BCE over all proteins×nodes — [ProtBoost, preprint](https://arxiv.org/html/2412.04529)
- DeepGO-SE: ESM2 embeddings projected into ELEmbeddings "world models"; aggregating truth values over up to 10 models approximates semantic entailment; removing axiom losses hurt MFO but not BPO/CCO; reports Fmax 0.554 MF / 0.432 BP / 0.721 CC on a **similarity-based split** (any DIAMOND hit e≤0.001 grouped) — [DeepGO-SE, Nature MI 2024](https://www.nature.com/articles/s42256-024-00795-w)
- Logit adjustment: subtract τ·log π_y post hoc or add it in the loss; Fisher-consistent for balanced error; improves rare classes on long-tailed benchmarks (e.g., CIFAR-10-LT 0.7136 → 0.7732 post hoc) — [Menon et al., ICLR 2021](https://arxiv.org/html/2007.07314v2); [code](https://github.com/google-research/google-research/tree/master/logit_adjustment)
- cafaeval: computes weighted precision/recall after IA weighting; `-prop max` propagates the max score upward; `-norm cafa` normalizes precision over proteins with ≥1 prediction above threshold and recall over all GT proteins; threshold step default 0.01 — [cafaeval PyPI](https://pypi.org/project/cafaeval/); [CAFA-evaluator, Bioinformatics Advances 2024](https://www.ccs.neu.edu/home/radivojac/papers/piovesan_bioinformadv_2024.pdf)

### Inferences
- Recommended objective (inference): L = ZLPR (or BCE with weak negatives) × (1 + λ·IA-weighted protein-centric soft-F1) per aspect, computed with the **same IA file** used for evaluation. Keep the soft-F1 per aspect because Fmax is reported per aspect with its own threshold.
- Logit adjustment for multi-label sigmoid outputs: add −τ·log(prior_term) to logits at training time; tune τ per aspect on temporal validation. It targets balanced error, not Fmax, so it is an ablation, not a default. IA weighting already up-weights rare specific terms.
- Since `-prop max` makes parents ≥ children at evaluation, a training-time MCM head aligns the training signal with what is scored and removes post-hoc propagation mismatch; cheap to add to any sigmoid head.
- Per-aspect thresholds: Fmax is threshold-free per aspect, but if predictions are calibrated across aspects differently, pick nothing on test. If a single deployed threshold is needed, set it on temporal validation.

### Gaps
- No published head-to-head of MCLoss vs plain BCE+propagation on CAFA temporal data was found.
- No source tests logit adjustment for GO prediction.

---

## 4. Vocabulary: rare and unseen terms

### Takeaway
GO-GPT's ~7,901-term vocabulary (≥20 proteins) versus our 6,276 (≥30) means our model cannot score some test terms at all; that loses IA-weighted recall on precisely the rare, high-IA terms. Remedies: lower the threshold, back off to homology/IEA transfer for out-of-vocabulary terms, or use ontology/text-based term embeddings for zero-shot scoring.

### Cited Findings
- GO-GPT: vocabulary of terms in ≥20 training proteins improved Fmax; ~10,000 tokens per the paper's description; not systematically ablated — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- ProtBoost predicted the top 5,000–13,000 terms depending on the model and filled missing terms with a prior plus a flag in the stacker — [ProtBoost, preprint](https://arxiv.org/html/2412.04529)
- CAFA5 training set had 31,466 distinct terms (BP 21,285; CC 2,957; MF 7,224) — [ProtBoost, preprint](https://arxiv.org/html/2412.04529)
- DeepGOZero and DeepGO-SE make zero-shot predictions for terms with no training annotations via ELEmbeddings of GO axioms — [DeepGOZero](https://www.biorxiv.org/content/10.1101/2022.01.14.476325v1)
- STAR-GO combines definition-text embeddings with ontology structure for zero-shot GO prediction — [STAR-GO, arXiv preprint](https://arxiv.org/pdf/2512.05245)
- GO-GPT found text-initialized term embeddings did not help its closed-vocabulary decoder — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)

### Inferences
- Measure first: compute the IA mass of test GT terms that fall outside our 6,276 vs GO-GPT's 7,901 vocabulary. That bound shows how much recall is unreachable.
- Cheapest fix: lower min-count to 10–20 and use a shared-parameter term head (term embedding = f(GO graph, text)) so rare terms borrow strength; route OOV terms through DIAMOND kNN transfer and IEA features, then propagate.
- Text-based open vocabulary helps most on rare terms; GO-GPT's negative result applies to initialization of a closed vocabulary, not to scoring OOV terms.

### Gaps
- No source reports how much of the GO-GPT test IA mass lies outside a ≥20 or ≥30 vocabulary.

---

## 5. Evaluation protocol to claim "beats GO-GPT"

### Takeaway
Match GO-GPT's exact protocol (test set, GO 2023-01-01 obo, CAFA5 IA file, cafaeval with prop=max norm=cafa, NK+LK as published), use a temporal validation window inside the pre-cutoff period for all tuning, and report paired protein-level bootstrap with Holm correction, per aspect, plus identity-bin stratification and multiple seeds.

### Cited Findings
- GO-GPT significance: **paired bootstrap, 10,000 iterations**, resampling proteins, recomputing Fmax difference; two-sided p from fraction of deltas with opposite sign; **Holm–Bonferroni** across 12 tests (2 comparisons × 3 aspects × 2 metrics) — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- GO-GPT numbers by inference mode: greedy wFmax 0.65 (vs InterLabelGO+ 0.63; MF difference not significant, p=0.073), 10-sample frequency probabilities 0.67, best-of-10 (oracle selection) 0.70; unweighted 0.76 vs 0.70 — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- Identity stratification in BioReason-Pro: bins by best-hit BLAST identity to training; because Fmax is not decomposable, they report **mean per-protein F1 per bin** and regress (model − BLAST) differences on identity with HC3 robust SEs — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- All BioReason-Pro evaluations used the official CAFA5 evaluation toolkit (cafaeval) with IA weighting and propagation of predictions and GT — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- CAFA3 reports 95% CIs from 10,000 bootstrap iterations over benchmark proteins, NK benchmarks in full evaluation mode, with coverage shown — [CAFA3, Genome Biology](https://pmc.ncbi.nlm.nih.gov/articles/PMC6864930/)
- Temporal validation precedents: NetGO 3.0 (validation = NK+LK annotated in the year before test) — [NetGO 3.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10626176/); tito (CAFA-5 #3) built validation from annotations newer than the training snapshot — [Kaggle](https://www.kaggle.com/its7171/writeups); ProtBoost used random GroupKFold on unique sequences plus a small recent-annotation holdout — [ProtBoost](https://arxiv.org/html/2412.04529)
- DeepGO-SE deliberately excluded BLAST/DIAMOND baselines because its split targets novel sequences — [DeepGO-SE](https://repository.kaust.edu.sa/server/api/core/bitstreams/85f63047-5471-468d-9726-2fb9b0ad251d/content)

### Inferences
- **Protocol checklist (inference):**
  1. Same 8,630 proteins, same propagated GT, same GO 2023-01-01 obo, CAFA5 IA file, `cafaeval -prop max -norm cafa`, th_step 0.01; report MF/BP/CC and the mean, weighted and unweighted.
  2. Rescore GO-GPT ourselves from the released test-data predictions with our evaluator run, so both numbers come from one pipeline (confirm we reproduce 0.65/0.67).
  3. Report NK and LK separately in addition to the combined set.
  4. **Temporal validation:** train on labels ≤ ~May 2022 (GOA snapshot), validate on proteins gaining new experimental annotations Jun–Nov 2022 (NK/LK as in CAFA); tune thresholds, ensemble weights, calibrator, τ, λ there; then retrain on everything ≤ Nov 2022 with fixed hyperparameters. This directly fixes the calibrator mis-calibration seen with random/cluster splits.
  5. Freeze the final configuration before touching the test set; log every test evaluation (count of test looks) and report all of them, or at least the number.
  6. ≥3–5 seeds; report mean ± sd and the paired bootstrap on the seed-ensemble vs GO-GPT 10-sample mode (the fair comparison for a probabilistic model is 0.67, not best-of-10 0.70, which uses oracle selection).
  7. Stratify by max identity to the ≤Nov-2022 training set (e.g., <30, 30–50, 50–70, 70–90, ≥90%); given ~88% of test proteins have a ≥50% homolog, a gain that exists only in high-identity bins is mostly a homology-transfer gain; report per-bin mean F1 and per-bin Fmax with bootstrap CIs.
  8. Compute match: state encoder size (GO-GPT uses ESM2-3B frozen), parameters, GPU-hours, and inference samples; include a variant restricted to the 119,749 GO-GPT training proteins to separate data from method.
  9. Baselines on the same set: DIAMOND kNN (≤Nov 2022 DB), naive frequency, InterLabelGO+ (0.63 reported) and ProtBoost if feasible.

### Gaps
- No source specifies a minimum Fmax delta considered meaningful; with 8,630 proteins, bootstrap CIs will define it empirically.

---

## 6. Legitimate ensembling and test-time strategies

### Takeaway
Ensembles of PLM-based heads with DIAMOND kNN, IEA/InterPro features and a GO-graph-aware stacker drove every top CAFA-5 result; GO-GPT itself gains ~0.02 from 10-sample probability averaging. All are legitimate if components only use pre-cutoff data and ensemble weights are fit on temporal validation.

### Cited Findings
- Combining DL with DIAMOND kNN via identity-dependent weights gave +0.016 wFmax for InterLabelGO+ — [InterLabelGO+](https://pmc.ncbi.nlm.nih.gov/articles/PMC11568131/)
- ProtBoost: GCN stacking across 5 base models + electronic annotations, 0.59 → 0.62 on their scale — [ProtBoost](https://arxiv.org/html/2412.04529)
- GOCurator/NetGO: learning-to-rank over many component predictors (BLAST-KNN, LR-InterPro, LR-ESM, text) — [NetGO 3.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC10626176/); [GORetriever](https://pmc.ncbi.nlm.nih.gov/articles/PMC11520413/)
- GO-GPT: 10 samples averaged → 0.67 vs greedy 0.65; best-of-10 0.70 is an oracle upper bound — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)
- DeepGO-SE aggregates up to 10 independently trained models (min/mean/max of truth values), chosen on validation — [DeepGO-SE](https://www.nature.com/articles/s42256-024-00795-w)
- ESM2-650M, ESM-C 600M, ESM3 and Profluent-E1 gave modest differences, ESM2-3B best for GO-GPT — [BioReason-Pro, preprint](https://exa.ai/library/publication/m8jwfm1wygl)

### Inferences
- Allowed: seed/backbone ensembles (ESM2-3B, ESM-C, ProtT5), DIAMOND kNN against the ≤Nov-2022 DB, IEA/InterPro features from dated releases, sliding-window embedding averaging for long sequences (>1,022 residues), GCN or learning-to-rank stacker fit on temporal validation.
- Not allowed for a fair claim: any DB, GOA, InterPro or literature/text source dated after Jan 2023; tuning ensemble weights or thresholds on the test set; best-of-N selection using test labels.
- Text/literature components (GORetriever-style) are powerful but literature published in the test window can leak labels; restrict PubMed to ≤ Nov 2022 publication dates or exclude.

### Gaps
- No source measures sliding-window TTA gains for GO prediction.
