# HiGO-R2 against the BioReason-Pro paper's GO comparison, under one scorer

Produced by `src/cafa6/compare_paper.py` on 2026-10-18.

This is a descriptive evaluation of frozen predictions. HiGO-R2 is the look-3 file
`runs/stacker_tv/l3_c0/pred_test_twin.npz`, unchanged; nothing was fit or selected on test. The paper's
own numbers, with section and table references, are in `paper_table.md`. The full set of generated tables
is in `report_tables.md`.

## Setup

- **Test set.** The GO-GPT temporal test set of the paper: 8,630 proteins, ground truth
  `data/proc_gogpt/gt_test.tsv`.
- **Ontology and weights.** GO 2023-01-01 (`runs/gogpt_cafa5/ref/go-basic_gogpt.obo`) and CAFA-5 IA
  (`runs/gogpt_cafa5/ref/IA.txt`).
- **Scorer.** `knn_sharp.Scorer`, i.e. cafaeval with prop=max and norm=cafa, thresholds 0.01–0.99.
  - Checked against the cafaeval CLI on GO-GPT beam and on the authors' GO-GPT file. Weighted Fmax,
    unweighted Fmax and weighted S-min agree to the third decimal (S-min, for example: BP 16.076, CC 6.015,
    MF 5.507 in both).
- **Subsets.**
  - S3_all: all 8,630 proteins.
  - S6_noknowledge: 8,159 proteins, i.e. without the 471 limited-knowledge proteins of
    `common_proteins.txt`. This is the subset of the paper's BioReason-Pro tables.
- **Identity bins.** From the best-hit percent identity in `data/proc_gogpt_tv2/knn/test_knn.npz`. That
  identity is identical to the best DIAMOND hit against GO-GPT training proteins.
- **Statistics.** Paired protein bootstrap with 1,000 reps and seed 0. The same resamples are used for every
  method, and each difference is HiGO-R2's weighted Fmax mean minus the method's.
  - p-values are two-sided bootstrap p-values, Holm-corrected over the eight methods that make up the
    family.
  - Three rows are variants outside the family: BioReason-Pro final-answer extraction, BioReason-Pro with
    the label-independent prompt, and BLAST-KNN with train+val references. They get raw p-values.
  - With 1,000 reps the smallest attainable Holm p is 0.008, so "p_Holm 0.008" means no resample reversed
    the sign.

## Comparison table

Fw is IA-weighted Fmax. S-min is IA-weighted and averaged over the three aspects. The last two columns give
HiGO-R2 minus the method in Fw mean, with 95% CIs.

| method | source of predictions | S3 Fw MF | S3 Fw BP | S3 Fw CC | S3 Fw mean | S6 Fw mean | S3 S-min | HiGO-R2 − method, S3 | HiGO-R2 − method, S6 |
|---|---|---|---|---|---|---|---|---|---|
| HiGO-R2 (ours) | frozen look-3 file | 0.738 | 0.658 | 0.758 | **0.718** | **0.735** | 7.56 | — | — |
| GO-GPT ten-sample frequency | released checkpoint, our 10 samples (T 0.7, top-k 20) | 0.720 | 0.630 | 0.745 | 0.698 | 0.714 | 8.17 | +0.019 [+0.014, +0.025], p_Holm 0.008 | +0.021 [+0.015, +0.027] |
| BLAST-KNN (paper §4.4.2 rule) | our DIAMOND hits to training proteins | 0.728 | 0.647 | 0.710 | 0.695 | 0.715 | 8.72 | +0.023 [+0.017, +0.029], p_Holm 0.008 | +0.020 [+0.015, +0.027] |
| GO-GPT beam 5 | released checkpoint, our run | 0.714 | 0.617 | 0.728 | 0.687 | 0.702 | 9.20 | +0.031 [+0.025, +0.037], p_Holm 0.008 | +0.033 [+0.027, +0.040] |
| GO-GPT greedy | released checkpoint, our run | 0.716 | 0.616 | 0.728 | 0.686 | 0.702 | 9.46 | +0.031 [+0.026, +0.038], p_Holm 0.008 | +0.033 [+0.027, +0.040] |
| GO-GPT (authors' predictions) | `go_pred` column of the HF test set (the file behind the paper) | 0.689 | 0.542 | 0.717 | 0.649 | 0.664 | 10.98 | +0.068 [+0.062, +0.075], p_Holm 0.008 | +0.072 [+0.065, +0.079] |
| BioReason-Pro RL (greedy) | released RL checkpoint, authors' `eval.py`, our run | 0.692 | 0.536 | 0.719 | 0.649 | 0.663 | 10.62 | +0.069 [+0.062, +0.076], p_Holm 0.008 | +0.072 [+0.065, +0.079] |
| DIAMONDScore (DeepGOPlus) | our DIAMOND hits to training proteins | 0.679 | 0.423 | 0.637 | 0.580 | 0.590 | 12.50 | +0.138 [+0.132, +0.144], p_Holm 0.008 | +0.145 [+0.140, +0.152] |
| Naive | training label frequencies | 0.184 | 0.209 | 0.337 | 0.243 | 0.244 | 16.98 | +0.474 [+0.466, +0.483], p_Holm 0.008 | +0.491 [+0.483, +0.499] |
| *variant:* BioReason-Pro RL, final-answer extraction | same run, GO ids after `</think>` | 0.690 | 0.535 | 0.718 | 0.648 | 0.662 | 10.68 | +0.070 [+0.064, +0.077] | +0.073 [+0.067, +0.080] |
| *variant:* BioReason-Pro RL, label-independent prompt | `ask_all_go_aspects=force_uniprot_summary=True` | 0.688 | 0.537 | 0.717 | 0.647 | 0.661 | 10.91 | +0.071 [+0.064, +0.078] | +0.074 [+0.067, +0.081] |
| *variant:* BLAST-KNN, train+val references | same rule, GO-GPT train + validation proteins | 0.733 | 0.669 | 0.724 | 0.709 | 0.729 | 8.42 | +0.009 [+0.004, +0.015], p 0.001 | +0.006 [+0.001, +0.012], p 0.015 |
| InterLabelGO+ | **paper-reported, different scorer** (Table S3) | 0.692 | 0.525 | 0.696 | 0.638 | — | — | no predictions, no paired test | — |
| ProtBoost | **paper-reported, different scorer** (Table S3) | 0.658 | 0.490 | 0.669 | 0.606 | — | — | no predictions, no paired test | — |
| BioReason-Pro SFT pass@1 | **paper-reported, different scorer** (Table S6, S6 only) | — | — | — | — | 0.641 | — | no predictions, no paired test | — |

Unweighted Fmax, per-aspect S6 values and per-aspect S-min are in `report_tables.md` and `summary.tsv`.
For reference, the S3 unweighted means are: HiGO-R2 0.765, GO-GPT ten-sample 0.761, GO-GPT greedy 0.749,
BLAST-KNN 0.728, authors' GO-GPT 0.716, BioReason-Pro RL 0.715.

## Offset between our scorer and the paper's numbers

| paper row | our row | relation | paper Fw mean | ours | offset (ours − paper), mean | offset MF / BP / CC |
|---|---|---|---|---|---|---|
| GO-GPT greedy, S3 (Table S3) | authors' GO-GPT predictions | same predictions | 0.651 | 0.649 | −0.002 | −0.012 / +0.003 / +0.004 |
| GO-GPT pass@1, S6 (Tables S6/S8) | authors' GO-GPT predictions | same predictions | 0.664 | 0.664 | −0.000 | −0.012 / +0.004 / +0.007 |
| BioReason-Pro RL pass@1, S6 (Table S8) | our BioReason-Pro RL run | same checkpoint and eval script | 0.663 | 0.663 | +0.000 | +0.002 / −0.001 / −0.000 |
| GO-GPT greedy, S3 (Table S3) | GO-GPT greedy, released checkpoint | different checkpoint | 0.651 | 0.686 | +0.035 | +0.015 / +0.077 / +0.015 |
| GO-GPT probability k=10, S3 (Table S5) | GO-GPT ten-sample, released checkpoint | different checkpoint | 0.661 | 0.698 | +0.037 | +0.014 / +0.083 / +0.015 |

On identical predictions our scorer reproduces the paper to within 0.002 on the weighted mean. Unweighted
means differ by at most 0.002.
- The per-aspect MF value is 0.012 lower. This is consistent with a small difference in MF ground truth or
  ontology handling, which this check cannot resolve.
- InterLabelGO+ and ProtBoost were scored by the paper's own pipeline on the same proteins, so their
  paper numbers are approximately on our scale (within about ±0.01 per aspect). They cannot be put into a
  paired test.
- The released GO-GPT checkpoint is about 0.035 stronger than the GO-GPT predictions behind the paper, and
  almost all of the gap is in BP. This repeats the finding of `runs/gogpt_cafa5/SUMMARY.md`.

## Main observations

1. **Ranking.** HiGO-R2 has the highest weighted Fmax on both subsets: 0.718 on S3 and 0.735 on S6. Every
   difference to a method we could run excludes 0 after Holm correction.
   - The nearest competitors are GO-GPT ten-sample (+0.019) and the paper-rule BLAST-KNN (+0.023).
   - HiGO-R2 also has the lowest S-min (7.56 against 8.17 for GO-GPT ten-sample).
2. **BioReason-Pro reproduces.** The released RL checkpoint, run with the authors' `eval.py` under the
   published-benchmark settings, gives S6 Fw 0.720 / 0.543 / 0.727. Table S8 reports 0.718 / 0.544 / 0.727.
   - Its score equals that of its input, the authors' GO-GPT predictions: 0.649 vs 0.649 on S3 and 0.663 vs
     0.664 on S6.
   - The label-independent prompt is lower by 0.002, so telling the model which aspects are annotated
     barely matters. Extracting only the final-answer block is lower by 0.001.
   - BioReason-Pro was not tested with the stronger GO-GPT checkpoint's predictions as input, because that
     is not the authors' protocol.
3. **BLAST-KNN is a strong baseline that the paper did not score with Fmax.** The paper's own
   best-hit rule gives 0.695 on S3. That is above GO-GPT greedy from the released checkpoint (0.686) and well
   above the paper's GO-GPT (0.651) and BioReason-Pro (0.663 on S6).
   - It is strong because 88% of test proteins have a training hit at 50% identity or more.
   - It falls to 0.36–0.39 below 80% identity and gives nothing for the 250 proteins without a hit.
4. **Reference set matters.** HiGO-R2's kNN uses GO-GPT's training and validation proteins as references.
   With the same reference set, BLAST-KNN reaches 0.709, and HiGO-R2's lead shrinks to +0.009
   [+0.004, +0.015] on S3 and +0.006 [+0.001, +0.012] on S6.
   - `runs/test_looks.md` (2026-10-18 17:45) already records that the train+val kNN prior alone scores 0.7206
     on this test set, slightly above HiGO-R2.
5. **By identity bin** (Fw mean, threshold chosen within each bin; full table in `report_tables.md`):
   - HiGO-R2 is best in the 30–50, 50–80, 80–95 and 95–99.9% bins: 0.480 / 0.506 / 0.675 / 0.858, against
     0.450 / 0.493 / 0.667 / 0.836 for GO-GPT ten-sample.
   - GO-GPT ten-sample is better for no-hit proteins (0.382 vs 0.342) and below 30% (0.461 vs 0.451).
   - BLAST-KNN is best at 100% identity (0.894 vs 0.878).
   - BioReason-Pro RL tracks the authors' GO-GPT predictions in every bin; the largest bin difference is
     0.008.
6. **Homology dependence.** These are Spearman correlations between per-protein weighted F1 at the global
   threshold (paper eq. 4.29) and best-hit identity:

   | method | ρ (ours) | ρ (paper, Table S12) |
   |---|---|---|
   | BLAST-KNN | 0.654 | 0.63 |
   | authors' GO-GPT | 0.441 | 0.46 |
   | BioReason-Pro RL | 0.439 | 0.46 |
   | released GO-GPT greedy | 0.470 | — |
   | HiGO-R2 | 0.562 | — |

   HiGO-R2's score depends more on homology than GO-GPT's does; its per-protein F1 is low for no-hit
   proteins at the global threshold (0.13 vs 0.30 for GO-GPT greedy).
7. **Supplementary subset.** The authors' SFT catalogue (`wanglab/protein_catalogue`) covers 1,213 test
   proteins, which are far from training: 470 of them are below 50% identity or have no hit.
   - On these proteins, DIAMONDScore 0.471, HiGO-R2 0.469, BioReason-Pro SFT 0.461 and GO-GPT ten-sample
     0.457 are within each other's CIs.
   - BioReason-Pro RL scores 0.439 (HiGO-R2 − RL +0.030 [+0.018, +0.047]) and BLAST-KNN 0.357.
   - The catalogue's prompt settings are not documented, so the SFT row is only indicative. Table at the
     end of `report_tables.md`.

## What could not be run, and why

- **InterLabelGO+.**
  - The authors' repo and HF datasets ship no InterLabelGO+ predictions on this test set. Notebook
    `data/021-Benchmarking.ipynb` only shows their retraining commands and validation scores.
  - The official pretrained package is hosted only at seq2fun.dcmb.med.umich.edu. That host did not respond
    (two attempts, 20 s and 80 s timeouts).
  - There is no GitHub release or HF mirror; the only HF hit is an unofficial MF-only model.
  - Even if reachable, it would need the 5.7 GB fair-esm ESM2-3B checkpoint, and its training data are the
    CAFA-5 Kaggle set, not the paper's retraining.
  - Reported here as paper numbers only.
- **ProtBoost.**
  - No predictions are shipped. The 2nd-place CAFA-5 repo provides a training pipeline that needs 300 GB of
    disk and a V100-days budget.
  - Its trained artifacts were in cloud storage "for 6 months only" from January 2024.
  - Reported here as paper numbers only.
- **BioReason-Pro SFT on the full test set.** The SFT checkpoint (11.3 GB) was not run; the task asked for
  the RL model. It appears only through the paper's number and the authors' 1,213-protein catalogue.
- **pass@10 / best-of-10 rows** (GO-GPT and BioReason-Pro). These are oracles that choose a sample with the
  ground truth, not predictors, so they are listed only in `paper_table.md`.
- **BLAST-KNN with blastp.** We used DIAMOND `--more-sensitive` (e ≤ 1e-3, top 50 by bitscore) instead of the
  paper's blastp with default parameters.
  - Our run leaves 250 proteins without a hit. The paper's Table S12 implies about 10 on S6 (8,149 of 8,159).
  - The paper reports no BLAST-KNN Fmax to compare against.
- **Reproducing the authors' BioReason-Pro outputs on a sample.** The test-data dataset contains no
  BioReason-Pro outputs (only the GO-GPT `go_pred` text), so no per-protein comparison was possible. The
  aggregate match with Table S8 (all three aspects within 0.002) is the check used instead.

## BioReason-Pro run details

- **Inputs.** As in the authors' `scripts/sh_eval.sh`:
  - ESM3 sm-open v1, layer 37, sequence only. The released `eval.py` never passes structure coordinates,
    so the 25 GB structure set was not needed.
  - The cached GO-graph embedding from the checkpoint.
  - Text context from the HF test set: organism, InterPro, STRING PPI, and the GO-GPT `go_pred`.
- **Decoding.** Greedy, `max_new_tokens` 3000, `max_model_len` 8192, vLLM 0.30, batch 128.
- **Prompt.** The published benchmark used `ask_all_go_aspects=False` and `force_uniprot_summary=False`
  (the prompt names the aspects present in the labels). The label-independent variant used True for both.
- **Output quality.** No empty responses. 4 of 8,630 responses (5 in the variant) never closed `</think>`
  within 3,000 tokens. On average 5.9 GO ids per response.
- **Compatibility shims.** Applied at launch in `compare_paper.py`; the authors' code was not edited:
  - `transformers.processing_utils.CommonKwargs`, which transformers 5.17 removed.
  - The ESM3 block hook now unpacks the `(x, attn)` tuple that esm 3.4 returns.
  - `unsloth` is stubbed. It is imported at module level but unused at inference.
  - `flash_attn` is marked absent, because the installed wheel is ABI-broken.
  - The vLLM engine runs in-process, with the torch sampler.
- **Throughput.** About 0.44 s per protein per GPU at batch 128, including ESM3. The 20-protein timing test
  at batch 20 took 17 s.
- **GPU time.**
  - Published-protocol run: 1,894 s + 1,918 s.
  - Label-independent run: 2,019 s + 2,068 s.
  - Timing and start-up attempts: about 8 min.
  - Total about 2.3 GPU-hours on 2 H100 NVL. GPU 1 kept about 12 GB free throughout, and the web app on
    port 7860 answered HTTP 200 during the runs.
- **RAM-staged files.** The weights (16.6 GB), GO embeddings (0.5 GB), the HF dataset cache and the
  catalogue (0.7 GB) were staged in RAM-backed `/dev/shm/cmp_paper_*`, not on disk. They remain there and
  use RAM until removed or until reboot.

## Files

- `src/cafa6/compare_paper.py` holds the subcommands `bioreason-gen`, `collect-bioreason`, `baselines`,
  `catalogue`, `score` and `tables`.
- `summary.tsv` (S3/S6, all metrics, bootstrap) and `per_bin.tsv` (Fw per bin, per-protein F1 per bin).
- `spearman_identity.json`.
- `summary_sub.tsv`, `per_bin_sub.tsv` and `spearman_identity_sub.json` for the catalogue subset.
- `report_tables.md` and `paper_table.md`.
- `preds/` (33 MB) holds:
  - Score files: `naive`, `blast_knn`, `blast_knn_trainval`, `diamondscore`, `bioreason_rl[_final]`,
    `bioreason_rl_askall[_final]`, `bioreason_sft_catalogue`.
  - The raw BioReason-Pro responses, as `*_responses.jsonl.gz`.
- `bioreason_rl/evals`, `bioreason_rl_askall/evals`: per-protein JSON from `eval.py` (about 105 MB each).
- `bioreason_rl_timing/`: the 20-protein timing test.
- `*_gen_stats.tsv` and `logs/`.
- `paper_src/`: the fetched paper HTML, PDF and figures.
- The authors' repo snapshot is in `external/BioReason-Pro-github`.

To reproduce, from `cafa6/`:

```bash
PYTHONPATH=src ../bin/python -m cafa6.compare_paper baselines
PYTHONPATH=src ../bin/python -m cafa6.compare_paper bioreason-gen --gpu 0 --num_chunks 2 --chunk_id 0   # and chunk 1 on GPU 1
PYTHONPATH=src ../bin/python -m cafa6.compare_paper collect-bioreason
PYTHONPATH=src ../bin/python -m cafa6.compare_paper score
PYTHONPATH=src ../bin/python -m cafa6.compare_paper tables
```
