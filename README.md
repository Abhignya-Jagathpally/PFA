# Ontologue (HiGO-R2): GO-term prediction from protein sequence

Ontologue predicts Gene Ontology (GO) terms (molecular function MF, biological process BP, cellular component CC)
for a protein sequence. Its code name in the repository is **HiGO-R2**. The pipeline has four parts:

1. **Identity-sharpened kNN label transfer.** DIAMOND blastp (`--more-sensitive`, e <= 1e-3, top 50 by bitscore)
   against the GO-GPT training **and** validation proteins (133,054 references). Each hit gets the weight
   bits * exp((pident - max pident) / 5). Every reference contributes its full propagated label set, and each
   aspect is normalised over the hits annotated in that aspect (setting `T5_k50_g0_na`).
2. **Frozen ESM2-35M HiGO head.** One learned query per GO term with entmax-1.5 attention over residues,
   ancestor-mean term queries, a taxon embedding and the C-HMCNN max-constraint module (MCM), trained with
   IA-weighted MCLoss (`runs/l3_c0_tv2`, `configs/l3_c0_tv2.yaml`). The encoder is not fine-tuned.
3. **Per-term twin stacker** (`stacker_tv` model b) that combines the HiGO logits with the kNN prior and its
   support and identity features. It is fit only on the temporal validation set **TV v2**, with 5-fold
   cross-fitting.
4. **GO true-path propagation**: max-propagation over the full ontology, so a parent never scores below its child.

The specification is `research_notes/HiGO-R loop/HiGO-R2_model.md` (section 0 is the result, section 2 the
per-component decisions). The pipeline diagram is `docs/higo_r2_pipeline.png`.

The repository started as a CAFA-6 Kaggle project (HiGO with a LoRA-tuned ESM2, `NOTES.md`, `REPORT.md`). That
earlier phase is documented in those two files. This README covers HiGO-R2, which is evaluated on the GO-GPT /
BioReason-Pro temporal benchmark.

## Results

All numbers are IA-weighted Fmax (mean of MF, BP, CC) on the GO-GPT temporal test set: 8,630 proteins with new
experimental annotations from March 2023 to February 2024, GO 2023-01-01 and CAFA-5 IA. Scoring is cafaeval with
prop=max and norm=cafa, through `knn_sharp.Scorer`, which reproduces the cafaeval CLI. Brackets are paired protein
bootstrap 95% CIs (1,000 reps). The HiGO-R2 row is the single pre-registered test look #3
(`runs/test_looks.md`, 2026-10-12 22:48:05). Every other row was scored afterwards as a description of frozen
files.

| method (all 8,630 proteins) | MF | BP | CC | mean | HiGO-R2 minus method |
|---|---|---|---|---|---|
| **HiGO-R2 (look #3)** | 0.738 | 0.658 | 0.758 | **0.718** | — |
| GO-GPT ten-sample frequency (released checkpoint, our 10 samples) | 0.720 | 0.630 | 0.745 | 0.698 | +0.019 [+0.014, +0.025] |
| BLAST-KNN, paper rule, training references | 0.728 | 0.647 | 0.710 | 0.695 | +0.023 [+0.017, +0.029] |
| GO-GPT beam 5 (released checkpoint) | 0.714 | 0.617 | 0.728 | 0.687 | +0.031 [+0.025, +0.037] |
| GO-GPT greedy (released checkpoint) | 0.716 | 0.616 | 0.728 | 0.686 | +0.031 [+0.026, +0.038] |
| GO-GPT, authors' predictions (the file behind the paper) | 0.689 | 0.542 | 0.717 | 0.649 | +0.068 [+0.062, +0.075] |
| BioReason-Pro RL (released checkpoint, authors' `eval.py`) | 0.692 | 0.536 | 0.719 | 0.649 | +0.069 [+0.062, +0.076] |
| DIAMONDScore (DeepGOPlus rule) | 0.679 | 0.423 | 0.637 | 0.580 | +0.138 [+0.132, +0.144] |
| SPROF-GO | 0.615 | 0.400 | 0.609 | 0.541 | +0.177 [+0.169, +0.185] |
| DeepGOPlus 1.0.10 | 0.539 | 0.264 | 0.481 | 0.428 | +0.290 [+0.283, +0.298] |
| ProteInfer GO, 5-model ensemble | 0.514 | 0.253 | 0.484 | 0.417 | +0.301 [+0.292, +0.308] |
| DeepGO-SE (CC checkpoints faulty as released) | 0.522 | 0.253 | 0.131 | 0.302 | +0.416 [+0.408, +0.424] |
| DeepFRI, sequence-only CNN | 0.282 | 0.179 | 0.360 | 0.274 | +0.444 [+0.436, +0.453] |
| Naive (training label frequency) | 0.184 | 0.209 | 0.337 | 0.243 | +0.474 [+0.466, +0.483] |

Sources: `runs/test_look/look3_l3c0_twin/`, `runs/compare_paper/report.md` and `summary.tsv`,
`runs/eval_suite/report.md`. Every difference to a method in the Holm family excludes 0 (p_Holm 0.013, the
smallest attainable with 1,000 reps). On the S6 subset used by the paper's BioReason-Pro tables (8,159 proteins),
HiGO-R2 scores 0.735 against 0.714 for GO-GPT ten-sample and 0.663 for BioReason-Pro RL.

### Where the margin comes from

- **The kNN component alone scores slightly higher than the full pipeline.** The train+val sharpened kNN prior
  alone scores 0.7206 on the test set, against 0.7178 for the pipeline (`runs/eval_suite`, logged on 2026-10-18 as
  descriptive scoring). The HiGO head alone scores 0.4815. On TV v2 the stacker added +0.0083 [+0.0040, +0.0120]
  over the kNN prior, but that gain did not carry over to the test set. The pre-registered choice was not changed
  after this was seen, because that would be selection on test.
- **The reference set accounts for much of the margin.** With the same train+val references, the paper's
  BLAST-KNN rule reaches 0.709, and HiGO-R2's lead shrinks to +0.009 [+0.004, +0.015].
- **Unweighted Fmax is a tie with GO-GPT ten-sample**: 0.765 against 0.761, with a difference of +0.0037
  [-0.0003, +0.0084]. The IA-weighted margin is therefore mostly on specific, high-IA terms.

### Novel proteins

HiGO-R2 depends on homology more than GO-GPT does. The Spearman correlation of per-protein weighted F1 with
best-hit identity is 0.523 for HiGO-R2 and 0.431 for GO-GPT ten-sample, a difference of +0.092 [+0.075, +0.108].
The table gives IA-weighted Fmax by identity to the nearest GO-GPT training protein, with the threshold chosen
within each bin (`runs/compare_paper/per_bin.tsv`):

| nearest training identity | n | HiGO-R2 | GO-GPT ten-sample | SPROF-GO |
|---|---|---|---|---|
| no hit | 250 | 0.342 | **0.382** | **0.403** |
| <30% | 79 | 0.451 | 0.461 | **0.529** |
| 30-50% | 694 | 0.480 | 0.450 | **0.514** |
| 50-80% | 1,627 | 0.506 | 0.493 | **0.512** |
| 80-95% | 1,360 | **0.675** | 0.667 | 0.572 |
| 95-99.9% | 2,430 | **0.858** | 0.836 | 0.582 |
| 100% | 2,190 | 0.879 | 0.842 | 0.549 |

- **GO-GPT ten-sample is better without a close homologue.** It scores higher for proteins with no hit and with
  under 30% identity (329 proteins, 3.8% of the test set).
- **SPROF-GO is better for distant proteins.** It is clearly above HiGO-R2 below 50% identity (0.40 vs 0.34 with no
  hit, 0.53 vs 0.45 under 30%) and level at 50-80% (0.512 vs 0.506).
- **On per-protein weighted F1 at the global threshold**, GO-GPT ten-sample leads in every bin below 80%. HiGO-R2
  leads only at 80% identity and above (`runs/eval_suite/identity_bins.tsv`).
- **BLAST-KNN is best at 100% identity** (0.894).
- **Recall of terms that best-hit transfer misses** (`runs/novel_terms`, IA-weighted, at each model's Fmax
  threshold): HiGO-R2 recovers 0.318 / 0.183 / 0.291 (MF / BP / CC). GO-GPT ten-sample recovers 0.329 / 0.167 /
  0.350 and GO-GPT beam 0.309 / 0.222 / 0.410. For proteins with no hit, the mean over aspects is 0.110 for
  HiGO-R2, against 0.237 and 0.284 for the two GO-GPT rows.

### Other measurements (`runs/eval_suite/report.md`)

- **Hierarchy violations.** The HiGO-R2 output has none in 2.75M child-parent pairs. GO-GPT ten-sample violates
  2.87% of pairs, in 27.3% of proteins.
- **Ranking quality.** Term-centric AUPR is higher for HiGO-R2: 0.785 against 0.623 for GO-GPT ten-sample. Its
  S-min is lower: 7.56 against 8.17.
- **Paired comparisons.** 51 of the 56 primary paired comparisons against the two GO-GPT rows pass
  Holm-Bonferroni, and none of the significant differences favours GO-GPT.

### Rows that are not like-for-like comparisons

- **InterLabelGO+** (released package, Zenodo 15508797) scores 0.728, 0.010 above HiGO-R2. Its training labels
  contain 8,629 of the 8,630 test proteins and 96.1% of the test (protein, term) pairs, so the row is an upper
  bound with test-label leakage, not a comparison. It is kept out of the test family
  (`runs/compare_paper/extra_models.md`).
- **The released GO-GPT checkpoint is stronger than the paper's numbers.** It is about 0.035 above the GO-GPT
  predictions behind the paper, almost all of it in BP. The GO-GPT rows above use the stronger released
  checkpoint.
- **BioReason-Pro RL reproduces the paper.** Its S6 scores are within 0.002 of Table S8 in every aspect, and it
  scores the same as its own input, the authors' GO-GPT predictions.

## Repository map

Run everything from `cafa6/` with `PYTHONPATH=src` and the project interpreter `../bin/python`.

### `src/cafa6/`, HiGO-R2 pipeline and evaluation

| module | purpose |
|---|---|
| `data.py` | data loading, GO parsing and propagation, label space; `CAFA6_PROC` selects the processed-data directory |
| `model.py` | HiGO network: ESM2 backbone (frozen, LoRA or DoRA), per-term entmax pooling, hierarchical queries, taxon embedding, MCM, in-network kNN fusion |
| `train.py` | trainer (`--config configs/*.yaml [key=value ...]`); writes `pred_{val,test,tv_all}_top500.npz` |
| `prepare_gogpt.py` | builds `data/proc_gogpt` from `wanglab/gogpt-training-data` and the GO-GPT test set |
| `temporal_val.py` | temporal validation splits TV v1 / TV v2 (`--v2`) from dated GOA annotations |
| `knn_sharp.py` | identity-sharpened kNN prior, the in-memory cafaeval scorer (`Scorer`), grid search and pre-registration on TV |
| `knn_trainval.py` | adds the GO-GPT validation proteins as kNN references; keep rule on TV v2 |
| `stacker_tv.py` | cross-fitted stackers on TV v2 (logistic fusion, twin, gated twin); writes `pred_test_twin.npz`, optional `--save-model` |
| `calibrate_tv.py` | per-(aspect, identity bin) temperature + bias or isotonic calibration on TV v2 (not used in the final pipeline) |
| `test_look.py` | one budgeted test look: scores a prediction file next to the two GO-GPT rows with a paired bootstrap; `--no-log` and `--out` for re-scoring |
| `eval_suite.py` | descriptive metrics of the frozen look-3 file (Fmax, AUPR, S-min, identity bins, consistency, GO-embedding retrieval, Holm tests); `--extended` adds the compare_paper methods |
| `eval_page.py` | renders the eval_suite output as the web app's evaluation page |
| `compare_paper.py` | the BioReason-Pro paper's comparison under one scorer: BioReason-Pro generation, baselines, external predictors (`import-tsv`), `score`, `tables` |
| `novel_terms.py` | recall and precision of test terms that best-hit transfer would miss, by identity bin |
| `predict_r2.py` | the look-3 pipeline on new sequences, on CPU (`--build`, `--fasta`, `--self-test`, `--verify-stacker`) |
| `webapp.py` | Ontologue web app (FastAPI) on top of `predict_r2.Predictor` |
| `bench_gogpt.py` | HiGO head-to-head on GO-GPT's benchmark (earlier phase, homology gate) |
| `meanflow_refiner.py` | MeanFlow label-refiner pilot (test look #2) |
| `stacker.py` | L1 GBDT stacker over kNN evidence on TV (homology only) |
| `evaluate.py`, `homology.py` | cafaeval wrapper; DIAMOND search, DiamondScore and identity strata |

Earlier CAFA-6 phase (see `NOTES.md` and `REPORT.md`): `prepare.py`, `baselines.py`, `stack.py`, `cv_summary.py`,
`dump_preds.py`, `explain.py`, `submit.py`, `grounding_data.py`, `grounded.py`, `predict_grounded.py`,
`prf_table.py`, and the pilots `go_graph.py`, `comparative.py` and `sae.py`.

### Other directories

- `configs/`: training configs. `l3_c0_tv2.yaml` is the HiGO head of the final pipeline. `l3_c1_tv2`,
  `l4a_tv2` and `l5_3b_tv2` are the LoRA, in-network-fusion and frozen-3B arms that were dropped on TV v2. The rest
  are from the CAFA-6 phase and the pilots.
- `benchmarks/gogpt_eval.py`: generation and scoring of the released GO-GPT checkpoint (beam, greedy and
  ten-sample; GPU).
- `scripts/`: `reproduce_tables.sh` (see below), plus `chain_stack.sh` and `chain_calibrate.sh`, which ran the
  stacker and calibration once a training run finished.
- `docs/`: `serving.md` (CLI, web app, data needs, licences), pipeline diagrams and web app images.
- `research_notes/`, `reports/`: design notes, observation ledgers and the HiGO-R2 specification.
- `runs/`: run outputs. Only small result tables are tracked; they were force-added, since `runs/` is in
  `.gitignore`. Tracked files include `runs/test_looks.md` (pre-registration and test-look log),
  `runs/test_look/look3_l3c0_twin/` (summary, per-bin, result, cafaeval output), `runs/eval_suite/`,
  `runs/compare_paper/` (reports and tables), `runs/novel_terms/`, `runs/knn_sharp/`, `runs/knn_trainval/`,
  `runs/stacker_tv/*/{summary,per_bin}.tsv` and `result.json`, `runs/stacker_tv/l3_c0/model_twin.pt` (the
  serving stacker) and `runs/gogpt_*` metrics. **Prediction files are not tracked**: `pred_test_twin.npz`,
  `runs/compare_paper/preds/`, GO-GPT generations and model weights other than the stacker.
- `data/`: not tracked (regenerated by the steps below), except `data/grounding/README.md`.
- `tools/`: DIAMOND and MMseqs2 binaries (not tracked) and loop helper scripts. `external/`: third-party
  repositories and dataset snapshots (not tracked).
- `tests/test_higo_r.py`: unit tests for the HiGO-R blocks (cached ESM2-8M, synthetic labels;
  `PYTHONPATH=src ../bin/python tests/test_higo_r.py`).

## Setup

- **Python.** Python 3.12 venv at `PFA/` (one level above this directory), used as `../bin/python`. Install
  `pip install -r requirements.txt`. The pinned versions include torch 2.13 (CUDA 13), transformers 5.17,
  peft 0.21, cafaeval 1.3.0, entmax 1.3 and scikit-learn 1.9.1. The web app and reports also need `fastapi`,
  `uvicorn`, `huggingface_hub`, `pyarrow` and `tabulate`.
- **DIAMOND.** DIAMOND 2.1.10, the static Linux binary from the bbuchfink/diamond GitHub releases, at
  `tools/diamond` (`tools/diamond version` should print 2.1.10). MMseqs2 at `tools/mmseqs/` is only needed for
  the CAFA-6 clustering in `prepare.py`.
- **Hugging Face caches** (`~/.cache/huggingface/hub`):
  - `facebook/esm2_t12_35M_UR50D` (130 MB). Needed by `train.py` and `predict_r2.py`; serving loads it offline.
  - Datasets `wanglab/gogpt-training-data` and `wanglab/bioreason-pro-test-data`.
  - Only for regenerating the reference rows, on GPU: `wanglab/gogpt` (GO-GPT), `wanglab/bioreason-pro-rl` and
    ESM3.
- **GO-GPT benchmark reference files** in `runs/gogpt_cafa5/ref/`: `test.parquet` (8,630 test proteins with
  ground truth and the authors' `go_pred`), `go-basic_gogpt.obo` (GO 2023-01-01), `IA.txt` (CAFA-5),
  `common_proteins.txt` and `cafa_evals.py`. All come from the BioReason-Pro repository and its HF test data (see
  `runs/gogpt_cafa5/SUMMARY.md`).

### Data preparation, in order

Several of these commands append pre-registration or test-look lines to `runs/test_looks.md` (`--register`,
`--grid*`, `--final`, `stacker_tv`, `calibrate_tv`, `test_look` without `--no-log`). The log already holds every
decision of this project. When rebuilding data, run only the build steps listed here, and do not re-run
selection or look commands.

```bash
export PYTHONPATH=src
# 1. GO-GPT train/val/test in HiGO layout -> data/proc_gogpt (downloads wanglab/gogpt-training-data)
../bin/python -m cafa6.prepare_gogpt
# 2. UniProt features, reference-proteome GOA, ligands -> data/grounding
../bin/python -m cafa6.grounding_data --step all
#    TV v2 also needs dated GAFs in data/grounding/goa_tv2/: D_rerio.goa (EBI GOA zebrafish proteome),
#    zfin.gaf.gz + zfin_uniprot.txt (ZFIN), rgd.gaf.gz + GENES_RAT.txt (RGD)
# 3. temporal validation set TV v2 -> data/proc_gogpt_tv2 (DIAMOND train-vs-train and TV hits)
../bin/python -m cafa6.temporal_val --v2 --threads 48
# 4. the pre-registered kNN prior (runs/knn_sharp/chosen_v2.json) -> data/proc_gogpt_tv2/knn/{tv,val,test}_knn.npz
../bin/python -m cafa6.knn_sharp --tv v2 --write T5_k50_g0_na data/proc_gogpt_tv2/knn
# 5. train + val references -> data/proc_gogpt_tv2/knn_trainval/{tv,test}_knn.npz
../bin/python -m cafa6.knn_trainval --diamond --stats --validity --threads 24
../bin/python -m cafa6.knn_trainval --write
# 6. HiGO head, frozen ESM2-35M (GPU, about 15 min per epoch) -> runs/l3_c0_tv2/pred_{val,test,tv_all}_top500.npz
CAFA6_PROC=data/proc_gogpt_tv2 ../bin/python -m cafa6.train --config configs/l3_c0_tv2.yaml
# 7. twin stacker on TV v2 -> runs/stacker_tv/l3_c0/pred_test_twin.npz (the look-3 file)
CAFA6_KNN_DIR=$PWD/data/proc_gogpt_tv2/knn_trainval ../bin/python -m cafa6.stacker_tv \
  --higo-run runs/l3_c0_tv2 --out runs/stacker_tv/l3_c0 --threads 16
# 8. serving data -> data/serve (reference DIAMOND db, label matrix, taxon map; about 2 min)
../bin/python -m cafa6.predict_r2 --build
```

The GO-GPT reference predictions (`runs/gogpt_cafa5/preds_shard*.jsonl` and
`runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz`) come from `benchmarks/gogpt_eval.py all` and
`benchmarks/gogpt_eval.py all --decode sample --out runs/gogpt_sample10`, both on GPU. The compare_paper
predictions come from the subcommands listed in `runs/compare_paper/report.md` (Files section) and from
`runs/compare_paper/extra_models.md`.

## Usage

### Predict GO terms for new sequences

```bash
PYTHONPATH=src ../bin/python -m cafa6.predict_r2 --fasta in.fa --taxon 9606 --out out_dir
```

- **Organism.** It is read from `OX=<taxon>` in each FASTA header, with `--taxon` as the fallback.
- **Outputs.**
  - `out_dir/predictions.tsv`: protein, GO term, score, aspect and name, for every propagated score >= 0.01.
  - `out_dir/report.md`: the terms at or above the per-aspect thresholds (MF 0.28, BP 0.42, CC 0.45) and the 5
    nearest references.
  - `out_dir/predictions.json`: everything, including identity bin and per-stage timings.
- **Hardware.** It runs on CPU with about 1.5 GB of RAM. One protein takes about 4 s, most of it in DIAMOND.
- **Checks.** `--self-test` runs 20 test proteins end to end against the benchmark file, and `--verify-stacker`
  checks the saved stacker. Neither reads test labels.

Details are in `docs/serving.md`.

### Web app

```bash
nohup env PYTHONPATH=src ../bin/python -m cafa6.webapp --host 127.0.0.1 --port 7860 > logs/webapp.log 2>&1 &
```

Open http://127.0.0.1:7860. There is a JSON API at `POST /api/predict` and a health check at `GET /health`. The
app has no authentication, so keep it on 127.0.0.1 and reach it from a laptop through an SSH tunnel:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@server     # then open http://127.0.0.1:7860 on the laptop
```

### Reproduce the tables

```bash
bash scripts/reproduce_tables.sh                                # all stages, outputs in /tmp/repro
OUT=/tmp/r2 STAGES="look3 novel" bash scripts/reproduce_tables.sh
CHECK_ONLY=1 bash scripts/reproduce_tables.sh                   # inputs and imports only
```

The script re-scores the frozen prediction files already under `runs/`. It does no training and no GPU work.
Every output goes to the scratch directory `OUT` (default `/tmp/repro`), and the script refuses a directory inside
`runs/` or `data/`. It checks that `runs/test_looks.md` is unchanged.

| stage | command | compared with |
|---|---|---|
| `look3` | `cafa6.test_look ... --out $OUT/test_look/look3_l3c0_twin --no-log` | `runs/test_look/look3_l3c0_twin/{summary,per_bin}.tsv`, `result.json` |
| `novel` | `cafa6.novel_terms --out $OUT/novel_terms` | `runs/novel_terms/{summary,by_bin}.tsv` |
| `compare` | `cafa6.compare_paper score --out ...` (all proteins and catalogue subset), then `tables --out ...` | `runs/compare_paper/{summary,per_bin,summary_sub}.tsv`, `spearman_identity.json`, `report_tables.md` |
| `eval` | `cafa6.eval_suite`, output redirected to `$OUT/eval_suite` | `runs/eval_suite/{headline,identity_bins,consistency}.tsv` |
| `eval_ext` | `cafa6.eval_suite --extended`, output in `$OUT/eval_suite_v2` | `runs/eval_suite_v2/` if present (untracked) |

Each comparison prints MATCH or DIFF with the largest absolute difference. Tolerances are 1e-9, or 1e-4 for the
compare_paper files, which are written with 4 decimals. The script exits non-zero if any comparison differs.
`look3` takes about 30 s and `novel` 1-2 min on CPU. `compare` and `eval` take tens of minutes. The inputs are the
untracked prediction files, so the script runs only on a machine that has them.

## Evaluation protocol

- **Selection data.** Every model choice was made on TV v2 (`data/proc_gogpt_tv2`): 7,566 GO-GPT training
  proteins held out with their post-T0 labels stripped.
  - TV v2 has a dated stratum (1,238 proteins with annotations first dated between 2021-11-30 and 2022-11-30)
    and a duplicate-accession stratum (6,328). The duplicate-accession stratum reproduces how the test set's
    >=95% identity bins arise.
  - The criterion is the TV v2 mean Fmax with "test-like" weights: proteins are weighted so that cells
    (organism group x identity bin x same-organism best hit) carry their test shares. These shares come from
    DIAMOND hits only, with no test labels. A guardrail requires that the plain TV v2 mean not drop by more than
    0.002.
  - GO-GPT's random validation split ranks homology choices almost in reverse of the test set, so it was used
    only to monitor convergence.
- **Pre-registration.** Every keep/drop rule was written to `runs/test_looks.md` before its first TV v2 score.
  Each rule states the threshold to clear (typically +0.005 test-like with a bootstrap CI excluding 0) and the
  simplicity order among candidates within 0.003 of the best. The final choice (22:48:00) picked the twin stacker
  on `l3_c0_tv2` at 0.7342 TV v2 test-like, against 0.7259 for raw train+val kNN.
- **Test-look budget.** The budget was four looks on the GO-GPT test set, and three were used:
  - Look 1: flat kNN, 0.5961.
  - Look 2: the MeanFlow pilot, 0.6544.
  - Look 3: HiGO-R2, 0.7178.

  Look 4 was reserved for an optimiser change that did not clear its registered bar, so it was not used.
- **Descriptive scoring.** After look 3, `eval_suite`, `novel_terms` and `compare_paper` scored the frozen
  look-3 file next to other methods. Each run is logged in `runs/test_looks.md` as descriptive scoring, not a
  selection look. Nothing was tuned or chosen from these numbers. The kNN-component result (0.7206) is recorded,
  so any later decision has to treat the test set as already used for that comparison.
- **Metric conventions.** These follow the CAFA protocol used by the GO-GPT paper:
  - Fmax uses one threshold per aspect, chosen on the test set for every row.
  - Precision is averaged over the proteins with at least one prediction.
  - Scores are cut at 0.01, limited to the top 1,500 per protein and rounded to 3 decimals before scoring.

## Caveats

- **The learned components are not shown to help on test.** On the test set the stacker is 0.0028 behind its own
  kNN prior. Five 35M HiGO seeds gave TV v2 twin scores from 0.718 to 0.734, and only the one at the top of that
  spread passed the keep rule. Showing that the HiGO feature helps would need a new temporal split or a newly
  registered look.
- **Accuracy without a close homologue is lower.** GO-GPT is better for proteins without a hit or with under 30%
  identity, and SPROF-GO below 50% identity (level at 50-80%). The web app and the reports flag proteins in these bins.
- **The kNN references are a fixed November 2022 snapshot.** A query that is itself a reference copies its own
  labels unless `--exclude-self` is set.
- **Comparisons with other tools have limits.**
  - BLAST-KNN uses DIAMOND `--more-sensitive` instead of blastp, which leaves 250 proteins without a hit.
  - The external predictors were trained on older (2019-2022) and smaller data, so their gap mixes method,
    training data and vocabulary.
  - ProtBoost and InterLabelGO+ appear in the paper's table only as paper-reported numbers. InterLabelGO+ was
    also run, but its released labels leak the test set.
- **CPU serving differs slightly from the benchmark file.** The CPU path runs HiGO in fp32, while the benchmark
  file used CUDA bf16. This changes 5 of 901 above-threshold decisions in the self-test.

## Licences

This repository does not include a licence file yet. Third-party components and data:

- **ESM2** weights and code: MIT.
- **UniProt** sequences and taxonomy, and the **Gene Ontology** (go-basic 2023-01-01 and propagated labels):
  CC BY 4.0, with attribution.
- **GO-GPT training data** (`wanglab/gogpt-training-data`): its dataset card says Apache-2.0. The data derives
  from UniProt and GO, and the card asks users to cite BioReason-Pro (Fallahpour et al., bioRxiv
  2026.03.19.712954).
- **CAFA-5 IA weights** (`IA.txt`): from the CAFA-5 evaluation material. Check the terms before redistributing
  the file.
- **DIAMOND**: GPL-3.0. Calling the unmodified binary is fine, but redistributing it means shipping it under the
  GPL.
- **C-HMCNN**: GPL-3.0. The MCM in `model.py` is a re-implementation and does not import that code.
- **Python stack**: PyTorch (BSD-3), transformers (Apache-2.0), FastAPI (MIT), uvicorn (BSD-3), scikit-learn
  (BSD-3).

Licences of the external predictors and of BioReason-Pro are listed in `runs/compare_paper/extra_models.md` and
in the snapshots under `external/`.
