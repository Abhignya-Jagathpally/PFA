# Kaggle CAFA 6 Protein Function Prediction: rules, evaluation, leaderboard, winning approaches, implications

Research date: 2026-10-12. Most Kaggle pages (overview, data, rules, leaderboard, writeups) rendered through the fetch tool and are quoted directly. Discussion threads and individual notebooks were not fetched; claims taken from them come from search snippets and are labelled.

**Headline fact:** the competition is over. Final submissions were due 2026-02-02 and the final evaluation was 2026-06-01. Prizes are settled (GOAlpha won with private score 0.52430). We cannot "win" it any more. What remains possible is (a) a late submission, if Kaggle scores late entries (evidence below suggests it does, but this is unverified without login), which would give a true prospective benchmark against the final ground truth, and (b) designing a model that would have beaten 0.524 under the same information constraints.

---

## 1. Evaluation: metric, propagation, ground truth, knowledge categories

### Takeaway
The score is an IA-weighted Fmax computed per aspect (MF, BP, CC) and per knowledge category (no-knowledge, limited-knowledge, partial-knowledge). The three aspect Fmax values are averaged within each category, and the three category scores are then averaged again, so partial-knowledge proteins count for one third of the score. For partial-knowledge proteins, terms already known before the deadline are excluded from evaluation, so re-predicting training labels earns nothing for them. Training labels still matter as features (cross-aspect, homolog transfer, GOA electronic evidence).

### Cited Findings
- Prospective competition: "Proteins having one or more of their functions published by researchers during the curation phase of the competition will comprise the future test set." — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Test set definition per aspect: proteins that (1) had no experimental terms in that aspect and accumulate some, or (2) had experimental terms in **all three** aspects before the deadline and accumulate new ones in that aspect. — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Metric: "The maximum F1-measure based on the weighted precision and recall will be calculated on each of the three test sets", following Jiang et al. 2016 (Genome Biol 17:184, supplement p.31), full evaluation mode, weights = information accretion `ia(f)` from IA.tsv. Root terms have weight 0. — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Aggregation: "The three maximum F-measures of the three subontologies ... will be combined as an arithmetic mean for each subtype of knowledge gain. Finally, the three F-measures from the three subtypes no-knowledge, limited-knowledge, and partial-knowledge will be combined again as an arithmetic mean to compute the final performance." — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Propagation: if predictions are not propagated, "the predictions will be recursively propagated by assigning each parent term a score that is the maximum score among its children's scores." Invalid GO terms (not in the listed GO version) are excluded. — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Evaluator: the CAFA-evaluator-PK fork adds a `-known` file of "annotations known previous to prediction" that "should be excluded from evaluation ... Evaluation should happen _only_ for these [new] terms"; it has no effect on no-knowledge and limited-knowledge targets. It also supports a terms-of-interest file (e.g. to drop terms added to the ontology after prediction). — [CAFA-evaluator-PK README](https://github.com/claradepaolis/CAFA-evaluator-PK)
- Evaluator details: obsolete terms excluded; only `is_a` and `part_of` edges; cross-aspect edges dropped; alt_ids mapped; terms absent from the IA file are removed; ground truth and predictions are always propagated to the roots. `-max_terms` truncation "is not checked [for score], meaning that terms are not sorted before the check, and the check is performed before propagation." — [CAFA-evaluator README (kaggle branch)](https://raw.githubusercontent.com/BioComputingUP/CAFA-evaluator/kaggle/README.md); [CAFA-evaluator-PK README](https://github.com/claradepaolis/CAFA-evaluator-PK)
- CAFA-5 Kaggle command, for reference: `main.py go-basic.obo pred/ test_terms.tsv -ia IA.txt -prop fill -norm cafa -th_step 0.001 -max_terms 500`. — [CAFA-evaluator kaggle branch](https://raw.githubusercontent.com/BioComputingUP/CAFA-evaluator/kaggle/README.md)
- `-norm cafa`: precision is normalised by the number of predicted targets and recall by the number of ground-truth targets. — [CAFA-evaluator-PK README](https://github.com/claradepaolis/CAFA-evaluator-PK)
- Ground-truth labels: "experimentally determined term–protein assignments". The training set used experimental or high-throughput evidence plus TAS and IC, from UniProtKB release of 18 June 2025 (Swiss-Prot 2025_03). GO graph is the 2025-06-01 release. — [Data page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data)
- 2nd-place team's reading: "we are evaluated only on protein/ontology pairs that are experimentally found. So, if we predict a term from ontology that does not exist, we will not get any penalty", so stackers should be trained per aspect only on proteins that have that aspect. — [2nd place writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/2nd-solution-of-cafa-6-py-boost-gcn-articles-da)
- Top teams dropped known (train) pairs from submissions: 5th place "Drop known labels: any [protein, GO] pair already present in train_terms.tsv is removed"; 11th place "Terms already known for the test proteins were removed where appropriate." — [5th place (AyPy)](https://www.kaggle.com/aypyaypy/writeups); [11th place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/11th-place-solution)
- 2nd place: "In 2 of those test sets ... proteins ... already have some labelling. ... We put some effort on making it work, but the effect was quite limited." Their consistent small gain came from 512-dim SVD of the protein's labels in the other two aspects, used as features for the third aspect. — [2nd place writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/2nd-solution-of-cafa-6-py-boost-gcn-articles-da)
- 6th place built a temporal validation (train on GOA release 226, validate on release 229) that allowed computing "the CAFA6 metrics on no-knowledge, limited-knowledge, and partial-knowledge protein targets." — [6th place writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/6th-place-solution-for-the-cafa-6-protein-function)
- Tooling for building NK/LK/PK splits and PK-known files from two GOA snapshots exists in `democafa` (by co-organiser An Phan). — [democafa_package](https://github.com/anphan0828/democafa_package)

### Inferences
- (Inference) A protein that had experimental terms in only one or two aspects and gains more terms in an aspect it already had is in none of the test sets. Gains in aspects it lacked make it a limited-knowledge target.
- (Inference) For partial-knowledge targets the known terms (and probably their propagated ancestors, if the known file is propagated; not verified) are removed from scoring. Predicting them is neutral rather than harmful, but they use up the per-protein term budget and can shift the Fmax threshold. The right move is to remove known pairs, as the 5th and 11th place teams did, and to spend the budget on new, typically more specific terms.
- (Inference) Because the three categories are weighted equally, partial-knowledge (well-studied proteins that pick up new, often deep terms) carries one third of the score even though it is a small group. A model tuned only on a clustered no-knowledge style split (our setting) optimises one third of the objective.
- (Inference) Excluding self-hits in our current submission is correct for no-knowledge transfer. For the 82,404 overlapping proteins, though, their own labels in other aspects (limited-knowledge) and their own labels in the same aspect (partial-knowledge, as context for "what's new") are legal, useful inputs that we currently throw away.
- (Inference) `-prop max` vs `fill` is unconfirmed for CAFA 6, but submitting already max-propagated scores makes the difference irrelevant.

### Gaps
- Exact CAFA 6 evaluator command (prop mode, th_step, whether `-max_terms` is used, whether the known file is propagated) was not found. The Kaggle metric code page was not fetched.
- Exact evidence-code list for the CAFA 6 ground truth (whether HTP/HDA/HMP/HGI/HEP and TAS/IC are included in the test set as in training) is stated only for the training set.
- Size and composition of the final test set per category and aspect was not found (likely in a future CAFA 6 paper; the CAFA 5 paper is at [bioRxiv 2026.04.27.716980](https://www.biorxiv.org/content/10.64898/2026.04.27.716980v1)).

---

## 2. Submission format, limits, rules, timeline

### Takeaway
The file has three tab-separated columns, no header, at most 1500 GO terms per protein across all aspects, and scores in (0, 1] with up to 3 significant figures. Optional "Text" rows are not scored for the leaderboard. Teams got 5 submissions per day and selected 2 final ones. External data and pretrained models were allowed if publicly and freely accessible. Winners must open-source their code under an OSI licence.

### Cited Findings
- Format: `target<TAB>GO:id<TAB>score`; scores in (0, 1.000], up to 3 significant figures; score 0 not allowed (omit the pair); unlisted proteins are scored as all zeros; "one target cannot be associated with more than 1500 terms for MF, BP, and CC subontologies combined." — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Free text: optional, up to 5 lines per protein, `target<TAB>Text<TAB>score<TAB>text`, printable ASCII, at most 3,000 characters per protein in total, and PubMed IDs are encouraged. It is "not ... evaluated during the time of competition ... not ... included in the leaderboard calculation, and ... not ... considered for winning prizes". It will be evaluated 9–12 months after the deadline with LLM judging (phase 1) followed by human evaluation (phase 2). — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Timeline: start 2025-10-15; entry and team-merger deadline 2026-01-26; final submission deadline 2026-02-02 (the overview also says "following the final submission deadline (February 1, 2026)", an inconsistency on the page); 2–3 interim leaderboard updates; final evaluation 2026-06-01. — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Prizes: $15k / $10k / $8k / $7k / $5k / $5k (total $50,000). 9,108 entrants, 2,259 teams, 3,747 submissions. — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Rules: maximum 5 submissions per day; up to 2 final submissions selected; team size at most 5; winner licence is open source (MIT named). — [Rules](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/rules)
- External data: "You may use data other than the Competition Data ... [if] publicly available and equally accessible to use by all Participants ... at no cost", with a "Reasonableness" standard for paid tools. Pretrained models with incompatible licences are allowed (and exempted from the open-source obligation). — [Rules](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/rules)
- The data page says: "The participants are not required to use these data and are also welcome to use any other data available to them." — [Data page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data)
- Public leaderboard during the competition: "a relatively small selection of proteins ... provided to us by the UniProtKB team, but not available in UniProtKB or other public databases"; distribution shift versus the final set is expected. — [Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview)
- Final leaderboard: "The private leaderboard is calculated with approximately 50% of the test data." — [Leaderboard](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/leaderboard)
- Organisers' guidance (1st place): "it is important to ensure that all evidence used by a method is restricted to information available before the test protein experimental annotations were obtained." — [1st place writeup](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/1st-place-solution-for-the-cafa-6-protein-function)

### Inferences
- (Inference) There was no explicit data cutoff date in the rules. The 2nd place team used GOA release 228 (available at the end of the competition) and copied a few newly experimental test-protein labels directly into the submission. The 11th place team used a December 2025 GOA snapshot. Any public data available **before 2026-02-02** was fair game.
- (Inference) For a late submission today, any data dated after 2026-02-02 (GOA, UniProt curated fields, literature) would leak the ground truth. A late score obtained with such data would not be comparable to the leaderboard.
- (Inference) The evaluator truncates per protein and namespace before sorting. Rows should therefore be written sorted by descending score, and a top-500 per protein cap (ours) is safely under 1500.

### Gaps
- Whether late submissions are currently accepted could not be verified (needs login). Indirect evidence: the 11th place writeup says "After the private leaderboard was revealed, several nearby LTR variants scored slightly higher. The best reached 0.42522", which implies post-deadline scoring — [11th place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/11th-place-solution). A search summary also claimed late submissions receive private scores, but the cited notebook (private 0.33878) may predate the deadline — [notebook](https://www.kaggle.com/code/seddiktrk/cafa-6-predictions).
- Maximum file size was not stated. The 1500-term cap is the only size limit found.

---

## 3. Leaderboard state and how our numbers compare

### Takeaway
On the final private leaderboard, 1st GOAlpha scored 0.52430, 2nd 0.48731, 3rd 0.44640, places 4–6 about 0.434–0.439, 10th 0.42531, and 50th about 0.392 (2,177 teams). The 1st place lead of 0.037 is unusually large. Our 0.528–0.541 clustered-split Fmax measures something else (recovery of existing annotations, no PK or LK categories), so it does not map to leaderboard scores. Public notebook-level methods scored about 0.34 privately.

### Cited Findings
- Final private top 10: GOAlpha 0.52430; protein_team 0.48731; yuanllo153 0.44640; FrOzen777 0.43883; AyPy 0.43468; Stefan Stefanov 0.43424; alan chen + john hsu 0.42902; Xiong Peng 0.42796; evansrita 0.42688; Bowen Jin 0.42531. 20th 0.40498; 30th 0.39878; 49th 0.39181; 2,177 teams in total. — [Leaderboard](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/leaderboard)
- Public leaderboard snapshot (third-party mirror, Nov 2025): top about 0.38, 10th about 0.36 (rounded). — [CLIST standings](https://clist.by/standings/cafa-6-protein-function-prediction-healthcare-biology-tabular-custom-metric-62574283/)
- 1st place: "most component methods achieved higher scores on the private leaderboard than on the public leaderboard". Literature methods were best on public but dropped on private. — [1st place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/1st-place-solution-for-the-cafa-6-protein-function)
- A public notebook combining ESM2-650M embeddings with a "GAF Submission" (GOA annotation transfer) shows Private Score 0.33878. — [CAFA 6: Predictions notebook](https://www.kaggle.com/code/seddiktrk/cafa-6-predictions) (snippet only)
- 11th place used ordinary out-of-fold validation and noted it "evaluates recovery of existing annotations rather than prediction of future experimental annotations". — [11th place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/11th-place-solution)
- CAFA 5 reference: ProtBoost scored 0.58240 on the CAFA 5 final leaderboard (2nd), so CAFA 6 scores are lower overall (the CAFA 6 metric adds PK and a category average). — [ProtBoost arXiv 2412.04529](https://arxiv.org/pdf/2412.04529)

### Inferences
- (Inference) Clustered-split Fmax at 30% identity is pessimistic for homology (no close homologs) but optimistic in another way: it scores all of a protein's labels, including shallow, easy terms, whereas the prospective set scores newly added terms and, for PK, only the new ones. Neither direction can be calibrated without a temporal benchmark. The comparison to 0.524 has to come from a late submission or a GOA-release temporal split (e.g. labels at ≤2025-06-18 → targets 2025-06-18..2026-02-02, using our dated `goa_reference.parquet`).
- (Inference) Being within the top 10 needs about 0.425. Being 1st needs above 0.524, which no sequence-only Kaggle team reached. 2nd place and below relied on PLM + GOA electronic features + boosting/NN stacks, and only the team that added literature-derived and network evidence through learning-to-rank broke 0.49.

### Gaps
- Per-aspect and per-category breakdowns of top scores were not found (the 1st place figures are images).
- Public leaderboard final values for the top teams were not retrieved (only the Nov 2025 mirror).

---

## 4. Winning approaches (CAFA 6) and lessons from CAFA 5

### Takeaway
Every top-6 CAFA 6 solution is an ensemble. Recurring ingredients: homology kNN (BLAST/DIAMOND), structure kNN (Foldseek on AlphaFold), PLM embeddings (ESM2-3B, ProtT5) with linear, boosted or MLP heads, taxonomy features, **GOA non-experimental annotations (IEA/IBA/ISS/ISO...) as features**, PPI kNN (STRING), and literature features (PubMed TF-IDF / MedCPT embeddings, GORetriever). They are combined by learning-to-rank or a GCN stacker. The winner's extra edge came from literature mining plus LTR over many heterogeneous components, trained on a species-matched validation set.

### Cited Findings
- **1st, GOAlpha (Fudan, Zhu lab; 0.52430):** seven components (BLAST-KNN, Net-KNN on PPI, FoldSeek-KNN, SVM-ESM2, LR-InterPro with 118,677 binary InterPro features, GOXML literature XML-classification, GORetriever literature reranking) plus GO term frequency and a 21-dim species encoding, combined with XGBoost learning-to-rank (GOLabeler style). Labels came from SwissProt, GOA and GO "according to the CAFA6 evidence codes". The LTR model was trained on a 1,000-protein validation set sampled to match the test species distribution. "Sequence-based methods achieved the strongest standalone performance"; "a combination of homology search and protein language models can recover a substantial fraction of the final performance". Literature used "high-quality manually curated literature associated with SwissProt proteins". — [1st place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/1st-place-solution-for-the-cafa-6-protein-function)
- **2nd, protein_team (Py-Boost + GCN; 0.48731):** update of their CAFA 5 ProtBoost. T5/ESM embeddings with taxon one-hot (~30 taxa) feed Py-Boost GBDTs (4.5k outputs), logistic regression (13.5k outputs) and MLPs, in "raw" and "conditional" variants (P(term) = P_cond × (1 − ∏ over parents of (1 − P(parent)))). Cross-aspect SVD features, PubMed TF-IDF article embeddings ("the most important new update"), a GCN stacker over GO with 38 binary GOA electronic-evidence features, and OBO consistency post-processing. GOA release 228 was used for the final prediction, and newly experimental test labels found there were added directly. They also trained on "Train + Old train" (CAFA 5 data). Code: [btbpanda/CAFA6-updates](https://github.com/btbpanda/CAFA6-updates). — [2nd place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/2nd-solution-of-cafa-6-py-boost-gcn-articles-da)
- From the same team's CAFA 5 analysis: "about 30% of electronic labels become experimental, so using it as a model feature performs better than just adding it as is." — [CAFA 5 2nd place](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/writeups/u900-private-2nd-public-5th-solution-py-boost-and-)
- **4th, FrOzen777 (0.43883):** code released as "CAFA6-4th-from-CAFA5-2th" with training, TTA inference and preprocessing; no writeup details were retrieved. — [Zenodo record via Exa](https://exa.ai/library/publication/dyrd4dldvlr)
- **5th, AyPy (0.43468):** weighted average of 3 supervised sets (each NN 0.4 + logreg 0.3 + PyBoost 0.3 on ESM2-3B + ProtT5-XL, taxon at phylum or order level, NEA features from the top 1,000 non-experimental GO terms, plus MedCPT PubMed embeddings in one set) and 3 kNN sources (Foldseek on AlphaFold2, DIAMOND, STRING PPI for BP only). The NN uses the SPROF-GO Max Constraint Module with loss weight 1+IA. Post-processing: max propagation, drop known train pairs, taxon constraints, top-200 per protein and aspect, score ≥ 0.01. — [AyPy writeups](https://www.kaggle.com/aypyaypy/writeups)
- **6th, Stefan Stefanov (0.43424):** an ensemble of 4 NNs over ESM embeddings, Foldseek 3Di / ProstT5 embeddings, PubMed embeddings, a transformer over the taxon lineage, and non-experimental GO terms with evidence codes (IBA, IBD, IEA, IGC, IKR, IRD, ISA, ISM, ISO, ISS, NAS, ND, RCA) via a TermEvidenceEncoder (from tito's CAFA 5 idea). Input dropout and taxon masking augmentations. Temporal validation from GOA release 226 to 229. Code: [stefanistefanov/CAFA6-Solution](https://github.com/stefanistefanov/CAFA6-Solution). — [6th place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/6th-place-solution-for-the-cafa-6-protein-function)
- **11th (0.42114):** used a December 2025 GOA snapshot. Experimental annotations were added to training, and non-experimental ones were used as an LTR component and evidence-code features. XGBoost LTR per aspect over a candidate union. After the reveal, a variant without evidence-code features but with richer BLAST pair features (identity, coverage, multi-homolog support) scored 0.42522. — [11th place](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/11th-place-solution)
- **CAFA 5 lessons:** 1st GOCurator (Zhu lab; NetGO3.0-style ensemble of BLAST-KNN, LR-ESM, LR-InterPro, structure and text, with GORetriever as the key new piece; adding GORetriever raised wFmax 0.587→0.604 on the CAFA 5 blind set). 2nd U900/ProtBoost (0.58240; Py-Boost + GCN + GOA electronic features). 3rd tito (PLM embeddings + taxonomy + non-experimental GOA labels as an (11 × n_terms) tensor through a 1D CNN, with time-aware validation). — [GORetriever paper](https://aaltodoc.aalto.fi/server/api/core/bitstreams/5c772562-1734-4dda-a3d2-ddca14e5df68/content); [ProtBoost arXiv](https://arxiv.org/pdf/2412.04529); [tito writeups](https://www.kaggle.com/its7171/writeups)
- Zhu lab also has DualGOFiller (RECOMB 2025) for "enhancing function prediction in partially annotated proteins", cited in the 1st place references; whether it was used in GOAlpha is not stated. — [1st place references](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/1st-place-solution-for-the-cafa-6-protein-function)

### Inferences
- (Inference) The two consistent winners' levers we lack are (1) GOA non-experimental evidence as input features for test proteins, and (2) literature evidence. Our InterPro/Pfam features partly cover the first (IEA via InterPro2GO is a large share of IEA), but IBA/ISO/ISS (phylogenetic and ortholog transfer) are not covered.
- (Inference) LTR or stacking over heterogeneous scorers, trained on a held-out set that mimics the test species mix, was the decisive design for 1st place. Our homology gate is a limited version of this.
- (Inference) Literature features helped on the public LB but hurt some components on private (1st place). They should enter through a learned stacker, not as a dominant component.

### Gaps
- 3rd (yuanllo153) and 4th place method details were not found.
- Discussion threads (e.g. taxonomy post [discussion/613750](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/discussion/613750)) were not fetched.

---

## 5. Data facts from local files (computed 2026-10-12 with small CPU scripts)

### Takeaway
There are 82,404 training proteins (537,027 non-propagated label rows; BP 16,858 distinct terms, MF 6,616, CC 2,651). The test superset has 224,309 proteins in 8,453 taxa, and all 82,404 training proteins are in it (36.7%). 36,775 training proteins have all three aspects, so they are partial-knowledge candidates; the other 45,629 are limited-knowledge candidates in their missing aspects. Our local GOA reference contains post-deadline experimental annotations for test proteins, which must be date-filtered.

### Cited Findings (local computation; file paths under `cafa6/data/`)
- `train_terms.tsv`: 537,027 rows; 82,404 proteins. CC: 60,292 proteins, 2,651 terms, 157,770 rows (2.62 per protein). MF: 58,001 proteins, 6,616 terms, 128,452 rows (2.21). BP: 59,958 proteins, 16,858 terms, 250,805 rows (4.18).
- Aspect coverage of training proteins: CFP 36,775; C only 9,934; FP 8,714; CP 7,341; P only 7,128; F only 6,270; CF 6,242.
- `train_taxonomy.tsv`: 1,381 taxa. Top: human 17,162; mouse 12,508; Arabidopsis 11,863; yeast S288C 5,520; rat 4,909; S. pombe 4,636; E. coli K-12 3,466; fly 3,201; worm 2,540; M. tuberculosis 1,530; zebrafish 1,384; Dictyostelium 1,277.
- Test superset: 224,309 unique IDs, 8,453 taxa (matches the 8,453 rows of `testsuperset-taxon-list.tsv`). Top: human 20,420; mouse 17,240; Arabidopsis 16,397; rat 8,219; yeast 6,733; cattle (9913) 6,052; S. pombe 5,123; E. coli 4,531; worm 4,493; rice (39947) 4,195; B. subtilis (224308) 4,191; Dictyostelium 4,160. 7,072 test taxa have no training protein.
- Train FASTA headers are `sp|ACC|NAME ...` (Swiss-Prot); test headers are `ACC taxid`. All 82,404 training accessions are in the test superset (82,404 / 224,309 = 36.7%).
- `sample_submission.tsv`: 20,004 lines; 10,000 proteins × 2 dummy GO rows (GO:0000001, GO:0000002) plus 4 `Text` rows for A0A0C5B5G6 (format `ID Text score sentence`).
- `IA.tsv`: 40,122 terms; 11,828 with IA = 0; mean IA 2.65.
- `go-basic.obo`: `data-version: releases/2025-06-01`.
- `grounding/goa_reference.parquet` (10 reference proteomes, downloaded 2026-09-28, dates up to 2026-07-27): 86,265 test-superset proteins appear in it. 85,475 have non-experimental GOA (19,171 of them are not in train), and 67,862 have experimental GOA (1,219 not in train). There are 36,921 experimental (protein, term) pairs not in `train_terms` on 19,713 proteins; 16,736 of these are dated after 2025-06-18 (8,681 proteins), and 5,158 after 2026-02-02.
- `grounding/uniprot_features.parquet`: 224,285 of 224,309 test proteins were found; 213,636 have InterPro matches.

### Inferences
- (Inference) Experimental GOA pairs dated after 2026-02-02 on test-superset proteins are essentially final ground truth. Using them in a late submission would inflate the score; they must be excluded from any feature or label used for test inference. Pairs dated 2025-06-18..2026-02-02 were legal at the time (2nd and 11th place used such releases) and form a natural **temporal validation set** with NK/LK/PK categories.
- (Inference) Non-experimental GOA covers about 38% of the test superset even with only 10 proteomes. Full UniProt-GOA (all taxa) at a pre-deadline release would cover far more, but needs a dated snapshot (e.g. GOA release around 226–228; file dates must be checked).
- (Inference) Some of the "not in train_terms" pre-June-2025 experimental pairs probably reflect evidence-code or filtering differences (e.g. IPI on protein binding, ND) rather than missing labels; not checked.

### Gaps
- Whether `train_terms.tsv` is propagated was not checked.
- Test proteins outside the 10 proteomes have no local GOA; full GOA coverage is unknown.

---

## 6. Implications for our model plan

### Takeaway
Optimise for the actual objective, which is the mean over NK, LK and PK of the mean over MF, BP and CC IA-Fmax for newly added terms. Build a temporal NK/LK/PK validation from dated GOA, add pre-deadline GOA non-experimental evidence and known-label features, combine components with a species-matched LTR/stacker, and remove known pairs at output. A late submission (if accepted) is the only way to compare against 0.524.

### Cited Findings
- See sections 1–4: category-averaged metric and known-term exclusion ([Overview](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview), [CAFA-evaluator-PK](https://github.com/claradepaolis/CAFA-evaluator-PK)); NEA features in the 2nd, 5th, 6th and 11th place solutions; LTR on a species-matched set in 1st ([1st](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/1st-place-solution-for-the-cafa-6-protein-function)); per-aspect stacker restricted to proteins having that aspect ([2nd](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/writeups/2nd-solution-of-cafa-6-py-boost-gcn-articles-da)).

### Inferences (all recommendations are inferences)
1. **Validation:** replace or augment the 30%-identity clustered split with a temporal split. Use labels as of 2025-06-18 (the `train_terms` set) and targets = experimental GOA pairs dated 2025-06-18..2026-02-02, classified NK/LK/PK per aspect. Score with CAFA-evaluator-PK (`-known` = train pairs, IA.tsv, max propagation) and average the categories as Kaggle does. Keep the clustered split as a secondary NK-hardness check.
2. **Known proteins (36.7% overlap):** keep self-hit exclusion for homology transfer, but (a) feed each protein's own labels in other aspects as features (LK, as in 2nd place's cross-aspect SVD), (b) feed its own same-aspect labels as context for PK and predict only new terms, and (c) drop all known (protein, term) pairs from the output. Ancestors of known terms have zero new information under PK scoring.
3. **External evidence (legal if pre-deadline):** GOA non-experimental codes (IEA, IBA, ISO, ISS, ...) as an evidence-code × term tensor; InterPro (already have); Foldseek on AlphaFold DB; STRING PPI kNN for BP; optionally literature (PubMed TF-IDF / MedCPT) through the stacker only. Any source must be restricted to snapshots dated ≤ 2026-02-02 for a fair late score. Our `goa_reference.parquet` has a `date` column that allows this filter.
4. **Ensembling:** per-aspect LTR (XGBoost/LightGBM ranker) or stacker over HiGO-v2, homology kNN, PLM linear heads, NEA and priors. Train it on a held-out set sampled to the test-superset species distribution (human, mouse, Arabidopsis, rat, yeast, cattle, pombe, E. coli, worm, rice, B. subtilis, Dicty), restricted per aspect to proteins that have that aspect.
5. **Output:** max-propagate, sort rows by descending score, apply a taxon constraint, keep top-k per protein and aspect (top 200 per aspect with score ≥ 0.01 worked for 5th place; our 500 per protein ≥ 0.02 is within the 1500 limit). Write scores with 3 significant figures and no zeros.
6. **Free text:** optional and unscored for the leaderboard; low priority unless the later CAFA 6 text assessment is a goal.

### Gaps
- Whether Kaggle still accepts late submissions (and whether they are scored on the private 50% only) is unverified; a Kaggle account or token is needed to check.
- No public per-category numbers exist to tell which category (NK, LK or PK) is our weakest; the temporal split above would answer this locally.
