# Beyond-homology GO terms on the GO-GPT temporal test

Do the models recover test annotations that conventional homology transfer would have missed?

For every GO-GPT test protein and aspect, the "conventional" prediction is DIAMOND best-hit transfer: all GO terms
(closed under the true-path rule) of the top-bitscore training protein, the classic BLAST baseline of CAFA. Terms in
the test ground truth that this baseline does not contain are "beyond-homology" terms. For each model, at its own
per-aspect IA-weighted Fmax threshold (runs/test_look/look3_l3c0_twin/cafaeval/best_f_w.tsv), the script reports:
  recall_beyond     IA-weighted share of beyond-homology true terms the model predicts (pooled over proteins)
  prec_beyond       IA-weighted share of the model's predictions outside the best-hit set that are correct
  n_prot_beyond     proteins with at least one beyond-homology true term, and how many of them get >= 1 recovered
by identity bin of the nearest training protein. Descriptive only: frozen prediction files, no selection.

  python -m cafa6.novel_terms      -> runs/novel_terms/{summary.tsv, by_bin.tsv, report.md}

## Pooled, per aspect

| model             | aspect   | bin   |   tau |   recall_beyond |   prec_beyond |   n_prot_beyond |   n_prot_recovered |
|:------------------|:---------|:------|------:|----------------:|--------------:|----------------:|-------------------:|
| HiGO-R2           | MF       | all   |  0.25 |          0.3178 |        0.0529 |             822 |                402 |
| HiGO-R2           | BP       | all   |  0.35 |          0.183  |        0.1544 |            2965 |               1469 |
| HiGO-R2           | CC       | all   |  0.45 |          0.2914 |        0.0767 |            1027 |                698 |
| GO-GPT ten-sample | MF       | all   |  0.11 |          0.3291 |        0.0623 |             822 |                415 |
| GO-GPT ten-sample | BP       | all   |  0.91 |          0.1666 |        0.1906 |            2965 |               1416 |
| GO-GPT ten-sample | CC       | all   |  0.91 |          0.3498 |        0.1412 |            1027 |                673 |
| GO-GPT beam       | MF       | all   |  0.01 |          0.309  |        0.0744 |             822 |                401 |
| GO-GPT beam       | BP       | all   |  0.01 |          0.2222 |        0.1297 |            2965 |               1640 |
| GO-GPT beam       | CC       | all   |  0.01 |          0.4096 |        0.1114 |            1027 |                715 |

## Recall of beyond-homology terms by identity bin (mean over aspects)

| bin     |   GO-GPT beam |   GO-GPT ten-sample |   HiGO-R2 |
|:--------|--------------:|--------------------:|----------:|
| 100     |         0.47  |               0.457 |     0.519 |
| 95-99.9 |         0.381 |               0.333 |     0.335 |
| 80-95   |         0.353 |               0.312 |     0.299 |
| 50-80   |         0.276 |               0.251 |     0.249 |
| 30-50   |         0.235 |               0.212 |     0.195 |
| <30     |         0.245 |               0.222 |     0.172 |
| no_hit  |         0.284 |               0.237 |     0.11  |

## Precision outside the best-hit set by identity bin

| bin     |   GO-GPT beam |   GO-GPT ten-sample |   HiGO-R2 |
|:--------|--------------:|--------------------:|----------:|
| 100     |         0.062 |               0.084 |     0.061 |
| 95-99.9 |         0.073 |               0.087 |     0.058 |
| 80-95   |         0.115 |               0.139 |     0.102 |
| 50-80   |         0.133 |               0.176 |     0.2   |
| 30-50   |         0.154 |               0.174 |     0.179 |
| <30     |         0.167 |               0.203 |     0.225 |
| no_hit  |         0.231 |               0.261 |     0.228 |

(protein-aspect counts with beyond-homology terms per bin: 100 444, 95-99.9 657, 80-95 842, 50-80 1587, 30-50 800, <30 95, no_hit 389)
