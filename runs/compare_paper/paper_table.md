# Methods compared on GO prediction in the BioReason-Pro paper, with the paper's own numbers

Source: Fallahpour et al., "BioReason-Pro: Advancing Protein Function Prediction with Multimodal Biological
Reasoning", bioRxiv 10.64898/2026.03.19.712954 **v1** (posted 2026-03-20; preprint, not peer reviewed).
The full-text HTML and the 110-page PDF were fetched on 2026-10-18 and are kept in `paper_src/`.
Text extracted from them is in `paper_src/paper_full.txt` (HTML) and `paper_src/paper_pdf.txt` (PDF).
Tables S3–S9 are images in the HTML but text in the PDF (pages 18–20), so every per-aspect number below
comes from the PDF text. The bar labels of Fig. 2B and Fig. 3E were read from `paper_src/F2.large.jpg` and
`paper_src/F3.large.jpg`.

## What the paper compares, and how it scores

- **GO methods compared** (§4.4.2 "Baselines"):
  - GO-GPT, in three inference modes (§2.2, §4.2.3): greedy single run; "probability" = term frequency
    over 10 samples; and pass@10 = best of 10 samples, an oracle that uses the ground truth.
  - BioReason-Pro SFT and RL, at pass@1 and pass@10 (§2.4).
  - InterLabelGO+ and ProtBoost, "retrained on the identical pre-November 2022 dataset" (§2.2, §4.4.2).
  - BLAST-KNN (§4.4.2): blastp with standard parameters against training sequences. Each test protein takes
    the best hit by percent identity, "all GO terms from the best-matching training protein are assigned
    ... with a score equal to the percent identity".
- **Not compared on GO.** Prot2Text-v2 and BLAST free-text transfer appear only in the LLM-judge and
  human evaluations (§2.3, §2.5). NetGO 3.0, DPFunc, PhiGnet, ProtGO and ProtNote are only cited (§1).
  The repo notebook `data/021-Benchmarking.ipynb` lists NetGO4 and SUPERMAGOv2 as planned, but they are not
  in the paper.
- **Scorer** (§4.4.1): official CAFA-5 evaluator (cafaeval), with IA weighting and propagation of both
  predictions and ground truth.
  - The paper does not state the prop/norm options. The repo's `evals/cafa_evals.py` calls `cafa_eval`
    with defaults (prop=max, norm=cafa) and `th_step=0.5`.
  - GO-GPT greedy and BioReason-Pro outputs are binary (score 1.0).
  - BioReason-Pro GO IDs are taken by regex. §4.3 says from "the final answer block", but the released
    `evals/run_cafa_eval.sh` uses `--final_answer_only False`, i.e. the whole response.
- **Subsets:**
  - **S3_all**: all 8,630 test proteins (MF/BP/CC n = 2,080 / 5,819 / 3,440). Used by Tables S3–S5,
    GO-GPT vs baselines.
  - **S6_noknowledge**: 8,159 proteins, i.e. without the 471 limited-knowledge proteins of
    `data/common_proteins.txt` (n = 1,878 / 5,623 / 3,293). Used by Tables S6–S9 (BioReason-Pro vs GO-GPT)
    and by Fig. 3E (§4.1.3: limited-knowledge proteins are excluded from the BioReason-Pro evaluation).
- **Statistics.** Paired bootstrap with 10,000 iterations; Holm–Bonferroni over 12 tests (Tables S3–S5) and
  90 tests (Tables S6–S9) (App. B.5.1).

## Table A. S3_all: all 8,630 proteins (Tables S3, S4, S5)

Fw = IA-weighted Fmax, F = unweighted Fmax. Means are our arithmetic means of the three aspects (the paper
quotes only rounded headline means).

| Method (mode) | Fw MF | Fw BP | Fw CC | Fw mean | F MF | F BP | F CC | F mean | Source |
|---|---|---|---|---|---|---|---|---|---|
| GO-GPT (greedy) | 0.701 | 0.539 | 0.713 | 0.651 | 0.766 | 0.587 | 0.801 | 0.718 | Table S3 |
| GO-GPT (probability, k=10) | 0.706 | 0.547 | 0.730 | 0.661 | 0.766 | 0.600 | 0.816 | 0.727 | Table S5 |
| GO-GPT (pass@10, oracle) | 0.743 | 0.595 | 0.764 | 0.701 | 0.802 | 0.644 | 0.840 | 0.762 | Table S4 |
| InterLabelGO+ | 0.692 | 0.525 | 0.696 | 0.638 | 0.727 | 0.566 | 0.790 | 0.694 | Tables S3–S5 (baseline column) |
| ProtBoost | 0.658 | 0.490 | 0.669 | 0.606 | 0.702 | 0.533 | 0.777 | 0.671 | Tables S3–S5 (baseline column) |
| BLAST-KNN | — | — | — | — | — | — | — | — | no Fmax reported (Fig. 3F per-bin F1 only) |

The text of §2.2 gives these headlines:
- Weighted Fmax: GO-GPT greedy 0.65, InterLabelGO+ 0.63, GO-GPT ten-sample 0.67, best-of-10 0.70.
- Unweighted Fmax: GO-GPT 0.76 vs InterLabelGO+ 0.70.
- The MF difference between GO-GPT greedy and InterLabelGO+ is not significant (p = 0.073).

Two of these headlines do not match the tables:
- The ten-sample mean from Table S5 is 0.661, not 0.67.
- The unweighted 0.76 equals the pass@10 mean (0.762), not the greedy mean (0.718).

## Table B. S6_noknowledge: 8,159 proteins (Tables S6–S9)

| Method (mode) | Fw MF | Fw BP | Fw CC | Fw mean | F MF | F BP | F CC | F mean | Source |
|---|---|---|---|---|---|---|---|---|---|
| GO-GPT (pass@1) | 0.729 | 0.545 | 0.718 | 0.664 | 0.792 | 0.593 | 0.806 | 0.730 | Tables S6/S8 baseline column |
| GO-GPT (pass@10, oracle) | 0.769 | 0.601 | 0.770 | 0.713 | 0.826 | 0.649 | 0.844 | 0.773 | Tables S6/S8 |
| BioReason-Pro SFT (pass@1) | 0.717 | 0.516 | 0.689 | 0.641 | 0.773 | 0.557 | 0.781 | 0.704 | Table S6 |
| BioReason-Pro SFT (pass@10, oracle) | 0.757 | 0.544 | 0.721 | 0.674 | 0.801 | 0.582 | 0.800 | 0.728 | Table S7 |
| BioReason-Pro RL (pass@1, greedy) | 0.718 | 0.544 | 0.727 | 0.663 | 0.783 | 0.587 | 0.812 | 0.727 | Table S8 |
| BioReason-Pro RL (pass@10, oracle) | 0.730 | 0.549 | 0.743 | 0.674 | 0.792 | 0.592 | 0.825 | 0.736 | Table S9 |

The text of §2.4 gives these headlines:
- Weighted Fmax: SFT 0.64 greedy and 0.67 best-of-10; RL 0.66 greedy and 0.67 best-of-10; "best
  unweighted Fmax sits at 0.74".
- "Both BioReason-Pro variants surpassed InterLabelGO+ at 0.63 and single-run GO-GPT at 0.65".

Table S8 shows that RL pass@1 is not significantly different from GO-GPT pass@1 in weighted MF, BP or CC
(p_corr 0.37, 1.00 and 0.30).

## Figure bar labels (rounded to 2 decimals in the figures)

Read from the figure images. Where a bar shows a dark best-of-10 extension, the label is the top of the bar.

| Figure | Method | F MF | F CC | F BP | F overall | Fw MF | Fw CC | Fw BP | Fw overall |
|---|---|---|---|---|---|---|---|---|---|
| 2B (S3) | ProtBoost | 0.67 | 0.76 | 0.52 | 0.65 | 0.66 | 0.67 | 0.50 | 0.61 |
| 2B (S3) | InterLabelGO+ | 0.74 | 0.79 | 0.57 | 0.70 | 0.67 | 0.70 | 0.52 | 0.63 |
| 2B (S3) | GO-GPT (top = best-of-10) | 0.80 | 0.84 | 0.64 | 0.76 | 0.74 | 0.76 | 0.59 | 0.70 |
| 3E (S6) | InterLabelGO+ | 0.74 | 0.79 | 0.57 | 0.70 | 0.67 | 0.70 | 0.52 | 0.63 |
| 3E (S6) | GO-GPT | 0.77 | 0.82 | 0.61 | 0.73 | 0.71 | 0.73 | 0.55 | 0.66 |
| 3E (S6) | BioReason-Pro SFT (top = best-of-10) | 0.80 | 0.80 | 0.58 | 0.73 | 0.76 | 0.72 | 0.54 | 0.67 |
| 3E (S6) | BioReason-Pro RL (top = best-of-10) | 0.79 | 0.82 | 0.59 | 0.73 | 0.73 | 0.74 | 0.55 | 0.67 |

The figure labels for ProtBoost and InterLabelGO+ do not all agree with Tables S3–S5. For example, the
InterLabelGO+ MF unweighted label is 0.74, against 0.727 in the table; ProtBoost's is 0.67, against 0.702.
Fig. 3E repeats the Fig. 2B InterLabelGO+ values although its subset is S6. The tables are used as the
paper numbers everywhere else in this directory.

## Homology dependence (§2.4, App. B.5.3, Table S12)

These are Spearman ρ values between per-protein F1 at the global Fmax threshold and best-hit BLAST identity
(n = 8,149 paired proteins):

| Method | ρ |
|---|---|
| BLAST | 0.63 |
| GO-GPT | 0.46 |
| GO-GPT (max@10) | 0.45 |
| BioReason-Pro SFT | 0.41 |
| BioReason-Pro SFT (max@10) | 0.41 |
| BioReason-Pro RL | 0.46 |
| BioReason-Pro RL (max@10) | 0.46 |

Fig. 3F shows the per-bin mean F1 (bins [0,30], (30,40], …, (80,90]) only as a plot.
