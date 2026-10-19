### Main table (IA-weighted Fmax; S-min in IA units)

| method | predictions | S3 Fw MF | S3 Fw BP | S3 Fw CC | S3 Fw mean | S6 Fw mean | S3 S-min mean | HiGO-R2 minus method, S3 [95% CI] | HiGO-R2 minus method, S6 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| HiGO-R2 (ours) | frozen look-3 file runs/stacker_tv/l3_c0/pred_test_twin.npz | 0.738 | 0.658 | 0.758 | 0.718 | 0.735 | 7.562 | — | — |
| BLAST-KNN (train+val references) | same rule, GO-GPT training + validation proteins as references (sensitivity row) | 0.733 | 0.669 | 0.724 | 0.709 | 0.729 | 8.421 | +0.0093 [+0.0043, +0.0150], p_boot 0.001 (variant, not in Holm family) | +0.0064 [+0.0014, +0.0124], p_boot 0.015 (variant, not in Holm family) |
| GO-GPT ten-sample frequency | released checkpoint, our generation (T 0.7, top-k 20) | 0.720 | 0.630 | 0.745 | 0.698 | 0.714 | 8.168 | +0.0193 [+0.0138, +0.0250], p_Holm 0.008 | +0.0211 [+0.0154, +0.0271], p_Holm 0.008 |
| BLAST-KNN | our DIAMOND hits to GO-GPT training proteins (paper 4.4.2 rule) | 0.728 | 0.647 | 0.710 | 0.695 | 0.715 | 8.723 | +0.0227 [+0.0171, +0.0285], p_Holm 0.008 | +0.0204 [+0.0152, +0.0266], p_Holm 0.008 |
| GO-GPT beam 5 | released checkpoint, our generation (predictor default) | 0.714 | 0.617 | 0.728 | 0.687 | 0.702 | 9.200 | +0.0312 [+0.0254, +0.0374], p_Holm 0.008 | +0.0332 [+0.0272, +0.0397], p_Holm 0.008 |
| GO-GPT greedy | released checkpoint, our generation | 0.716 | 0.616 | 0.728 | 0.686 | 0.702 | 9.462 | +0.0314 [+0.0257, +0.0376], p_Holm 0.008 | +0.0333 [+0.0274, +0.0399], p_Holm 0.008 |
| GO-GPT (authors' predictions) | go_pred column of wanglab/bioreason-pro-test-data | 0.689 | 0.542 | 0.717 | 0.649 | 0.664 | 10.984 | +0.0684 [+0.0624, +0.0750], p_Holm 0.008 | +0.0716 [+0.0650, +0.0785], p_Holm 0.008 |
| BioReason-Pro RL (greedy) | released wanglab/bioreason-pro-rl, authors' eval.py, our run | 0.692 | 0.536 | 0.719 | 0.649 | 0.663 | 10.621 | +0.0687 [+0.0623, +0.0756], p_Holm 0.008 | +0.0718 [+0.0653, +0.0789], p_Holm 0.008 |
| BioReason-Pro RL (greedy, final answer only) | same run, GO ids after </think> | 0.690 | 0.535 | 0.718 | 0.648 | 0.662 | 10.683 | +0.0700 [+0.0638, +0.0770], p_boot 0.001 (variant, not in Holm family) | +0.0732 [+0.0667, +0.0803], p_boot 0.001 (variant, not in Holm family) |
| BioReason-Pro RL (greedy, label-independent prompt) | same checkpoint, eval.py with ask_all_go_aspects=force_uniprot_summary=True (repo default, predict.py) | 0.688 | 0.537 | 0.717 | 0.647 | 0.661 | 10.909 | +0.0707 [+0.0643, +0.0775], p_boot 0.001 (variant, not in Holm family) | +0.0737 [+0.0671, +0.0810], p_boot 0.001 (variant, not in Holm family) |
| DIAMONDScore | our DIAMOND hits to GO-GPT training proteins | 0.679 | 0.423 | 0.637 | 0.580 | 0.590 | 12.502 | +0.1379 [+0.1320, +0.1442], p_Holm 0.008 | +0.1453 [+0.1397, +0.1522], p_Holm 0.008 |
| Naive | GO-GPT training label frequencies | 0.184 | 0.209 | 0.337 | 0.243 | 0.244 | 16.976 | +0.4743 [+0.4662, +0.4826], p_Holm 0.008 | +0.4910 [+0.4831, +0.4991], p_Holm 0.008 |
| BioReason-Pro SFT pass@1 (paper-reported, different scorer) | Table S6 | — | — | — | — | 0.641 | — | not computable (no predictions) | — |
| InterLabelGO+ (paper-reported, different scorer) | Tables S3-S5 | 0.692 | 0.525 | 0.696 | 0.638 | — | — | not computable (no predictions) | — |
| ProtBoost (paper-reported, different scorer) | Tables S3-S5 | 0.658 | 0.490 | 0.669 | 0.606 | — | — | not computable (no predictions) | — |

### Unweighted Fmax, S6 per aspect, S-min per aspect

| method | S3 F MF | S3 F BP | S3 F CC | S3 F mean | S6 Fw MF | S6 Fw BP | S6 Fw CC | S6 F mean | S3 S-min MF | S3 S-min BP | S3 S-min CC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HiGO-R2 (ours) | 0.778 | 0.691 | 0.827 | 0.765 | 0.771 | 0.668 | 0.767 | 0.780 | 5.093 | 13.206 | 4.387 |
| BLAST-KNN (train+val references) | 0.763 | 0.678 | 0.782 | 0.741 | 0.769 | 0.682 | 0.735 | 0.761 | 5.672 | 13.378 | 6.215 |
| GO-GPT ten-sample frequency | 0.780 | 0.678 | 0.826 | 0.761 | 0.750 | 0.639 | 0.753 | 0.775 | 5.466 | 14.121 | 4.917 |
| BLAST-KNN | 0.758 | 0.658 | 0.769 | 0.728 | 0.763 | 0.660 | 0.721 | 0.748 | 5.774 | 14.054 | 6.340 |
| GO-GPT beam 5 | 0.777 | 0.659 | 0.812 | 0.749 | 0.743 | 0.626 | 0.737 | 0.763 | 5.507 | 16.076 | 6.015 |
| GO-GPT greedy | 0.778 | 0.658 | 0.812 | 0.749 | 0.745 | 0.625 | 0.736 | 0.763 | 5.583 | 16.756 | 6.047 |
| GO-GPT (authors' predictions) | 0.757 | 0.586 | 0.805 | 0.716 | 0.717 | 0.549 | 0.725 | 0.729 | 5.907 | 20.437 | 6.609 |
| BioReason-Pro RL (greedy) | 0.758 | 0.581 | 0.806 | 0.715 | 0.720 | 0.543 | 0.727 | 0.727 | 6.031 | 20.183 | 5.649 |
| BioReason-Pro RL (greedy, final answer only) | 0.756 | 0.579 | 0.805 | 0.714 | 0.718 | 0.542 | 0.726 | 0.726 | 6.095 | 20.311 | 5.642 |
| BioReason-Pro RL (greedy, label-independent prompt) | 0.756 | 0.581 | 0.803 | 0.714 | 0.716 | 0.544 | 0.724 | 0.726 | 6.054 | 20.141 | 6.533 |
| DIAMONDScore | 0.721 | 0.468 | 0.740 | 0.643 | 0.703 | 0.425 | 0.642 | 0.653 | 6.616 | 24.247 | 6.643 |
| Naive | 0.375 | 0.279 | 0.593 | 0.416 | 0.186 | 0.209 | 0.338 | 0.419 | 13.407 | 28.212 | 9.308 |

### Our scorer vs the paper's numbers

| paper row | subset | ours | relation | paper Fw mean | ours Fw mean | offset Fw (ours - paper) | offset Fw MF/BP/CC | paper F mean | ours F mean | offset F (ours - paper) |
|---|---|---|---|---|---|---|---|---|---|---|
| GO-GPT greedy (Table S3) | S3_all | GO-GPT (authors' predictions) | same predictions (authors' go_pred file) | 0.651 | 0.649 | -0.002 | -0.012/+0.003/+0.004 | 0.718 | 0.716 | -0.002 |
| GO-GPT pass@1 (S6) (Tables S6/S8) | S6_noknowledge | GO-GPT (authors' predictions) | same predictions (authors' go_pred file) | 0.664 | 0.664 | -0.000 | -0.012/+0.004/+0.007 | 0.730 | 0.729 | -0.002 |
| GO-GPT greedy (Table S3) | S3_all | GO-GPT greedy | released checkpoint, our greedy run | 0.651 | 0.686 | 0.035 | +0.015/+0.077/+0.015 | 0.718 | 0.749 | 0.031 |
| GO-GPT probability k=10 (Table S5) | S3_all | GO-GPT ten-sample frequency | released checkpoint, our ten samples | 0.661 | 0.698 | 0.037 | +0.014/+0.083/+0.015 | 0.727 | 0.761 | 0.034 |
| BioReason-Pro RL pass@1 (Table S8) | S6_noknowledge | BioReason-Pro RL (greedy) | released RL checkpoint, our run of eval.py | 0.663 | 0.663 | 0.000 | +0.002/-0.001/-0.000 | 0.727 | 0.727 | -0.000 |
| BioReason-Pro RL pass@1 (Table S8) | S6_noknowledge | BioReason-Pro RL (greedy, final answer only) | same run, final-answer extraction | 0.663 | 0.662 | -0.001 | +0.000/-0.002/-0.001 | 0.727 | 0.726 | -0.001 |

### Weighted Fmax mean per identity bin (threshold chosen within the bin)

| method | no_hit (n=250) | <30 (n=79) | 30-50 (n=694) | 50-80 (n=1627) | 80-95 (n=1360) | 95-99.9 (n=2430) | 100 (n=2190) | all (n=8630) |
|---|---|---|---|---|---|---|---|---|
| HiGO-R2 (ours) | 0.342 | 0.451 | 0.480 | 0.506 | 0.675 | 0.858 | 0.878 | 0.718 |
| BLAST-KNN (train+val references) | 0.055 | 0.357 | 0.372 | 0.427 | 0.649 | 0.847 | 0.894 | 0.709 |
| GO-GPT ten-sample frequency | 0.382 | 0.461 | 0.450 | 0.493 | 0.667 | 0.836 | 0.842 | 0.698 |
| BLAST-KNN | 0.000 | 0.356 | 0.357 | 0.393 | 0.618 | 0.838 | 0.894 | 0.695 |
| GO-GPT beam 5 | 0.374 | 0.402 | 0.438 | 0.469 | 0.657 | 0.829 | 0.835 | 0.687 |
| GO-GPT greedy | 0.369 | 0.414 | 0.439 | 0.472 | 0.655 | 0.829 | 0.834 | 0.686 |
| GO-GPT (authors' predictions) | 0.312 | 0.466 | 0.424 | 0.442 | 0.620 | 0.760 | 0.812 | 0.649 |
| BioReason-Pro RL (greedy) | 0.320 | 0.465 | 0.424 | 0.444 | 0.622 | 0.759 | 0.809 | 0.649 |
| BioReason-Pro RL (greedy, final answer only) | 0.319 | 0.465 | 0.424 | 0.444 | 0.622 | 0.757 | 0.807 | 0.648 |
| BioReason-Pro RL (greedy, label-independent prompt) | 0.312 | 0.459 | 0.423 | 0.441 | 0.618 | 0.757 | 0.808 | 0.647 |
| DIAMONDScore | 0.000 | 0.459 | 0.460 | 0.452 | 0.564 | 0.651 | 0.691 | 0.580 |
| Naive | 0.203 | 0.248 | 0.251 | 0.245 | 0.257 | 0.248 | 0.252 | 0.243 |

### Mean per-protein weighted F1 at the global Fmax threshold per bin (paper eq. 4.29, Fig. 3F) and its Spearman correlation with best-hit identity (paper Table S12)

| method | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 | all | Spearman rho vs identity |
|---|---|---|---|---|---|---|---|---|---|
| HiGO-R2 (ours) | 0.127 | 0.345 | 0.316 | 0.332 | 0.575 | 0.821 | 0.828 | 0.627 | 0.562 |
| BLAST-KNN (train+val references) | 0.025 | 0.106 | 0.150 | 0.162 | 0.521 | 0.799 | 0.818 | 0.559 | 0.619 |
| GO-GPT ten-sample frequency | 0.284 | 0.341 | 0.333 | 0.337 | 0.568 | 0.780 | 0.777 | 0.608 | 0.473 |
| BLAST-KNN | 0.000 | 0.115 | 0.129 | 0.110 | 0.481 | 0.789 | 0.818 | 0.538 | 0.654 |
| GO-GPT beam 5 | 0.308 | 0.353 | 0.339 | 0.340 | 0.585 | 0.789 | 0.779 | 0.616 | 0.469 |
| GO-GPT greedy | 0.303 | 0.359 | 0.339 | 0.338 | 0.581 | 0.787 | 0.779 | 0.614 | 0.470 |
| GO-GPT (authors' predictions) | 0.259 | 0.388 | 0.322 | 0.311 | 0.528 | 0.689 | 0.750 | 0.563 | 0.441 |
| BioReason-Pro RL (greedy) | 0.270 | 0.388 | 0.322 | 0.312 | 0.531 | 0.689 | 0.748 | 0.564 | 0.439 |
| BioReason-Pro RL (greedy, final answer only) | 0.269 | 0.388 | 0.322 | 0.312 | 0.531 | 0.687 | 0.746 | 0.562 | 0.437 |
| BioReason-Pro RL (greedy, label-independent prompt) | 0.264 | 0.382 | 0.321 | 0.310 | 0.525 | 0.686 | 0.745 | 0.560 | 0.440 |
| DIAMONDScore | 0.000 | 0.343 | 0.327 | 0.270 | 0.411 | 0.517 | 0.594 | 0.442 | 0.424 |
| Naive | 0.169 | 0.216 | 0.194 | 0.195 | 0.236 | 0.228 | 0.246 | 0.223 | 0.109 |

### Supplementary: the 1,213 test proteins in the authors' SFT catalogue

| method | Fw MF | Fw BP | Fw CC | Fw mean | S6 Fw mean | HiGO-R2 minus method [95% CI] | n MF/BP/CC |
|---|---|---|---|---|---|---|---|
| DIAMONDScore | 0.521 | 0.342 | 0.551 | 0.471 | 0.480 | -0.0024 [-0.0142, +0.0114] | 536/691/544 |
| HiGO-R2 (ours) | 0.479 | 0.362 | 0.565 | 0.469 | 0.472 | — | 536/691/544 |
| BioReason-Pro SFT (authors' catalogue) | 0.490 | 0.326 | 0.568 | 0.461 | 0.470 | +0.0076 [-0.0065, +0.0240] | 536/691/544 |
| GO-GPT ten-sample frequency | 0.471 | 0.344 | 0.555 | 0.457 | 0.463 | +0.0121 [-0.0021, +0.0262] | 536/691/544 |
| GO-GPT beam 5 | 0.459 | 0.335 | 0.549 | 0.448 | 0.454 | +0.0212 [+0.0075, +0.0386] | 536/691/544 |
| GO-GPT greedy | 0.459 | 0.335 | 0.546 | 0.447 | 0.452 | +0.0221 [+0.0086, +0.0389] | 536/691/544 |
| BioReason-Pro RL (greedy) | 0.461 | 0.326 | 0.529 | 0.439 | 0.444 | +0.0302 [+0.0178, +0.0473] | 536/691/544 |
| BioReason-Pro RL (greedy, final answer only) | 0.458 | 0.326 | 0.528 | 0.437 | 0.443 | +0.0314 [+0.0189, +0.0486] | 536/691/544 |
| BioReason-Pro RL (greedy, label-independent prompt) | 0.451 | 0.325 | 0.529 | 0.435 | 0.440 | +0.0339 [+0.0208, +0.0507] | 536/691/544 |
| GO-GPT (authors' predictions) | 0.438 | 0.326 | 0.525 | 0.430 | 0.433 | +0.0389 [+0.0265, +0.0557] | 536/691/544 |
| BLAST-KNN | 0.379 | 0.258 | 0.432 | 0.357 | 0.370 | +0.1122 [+0.0963, +0.1315] | 536/691/544 |
| BLAST-KNN (train+val references) | 0.375 | 0.259 | 0.435 | 0.356 | 0.370 | +0.1124 [+0.0967, +0.1320] | 536/691/544 |
| Naive | 0.158 | 0.193 | 0.333 | 0.228 | 0.225 | +0.2412 [+0.2275, +0.2573] | 536/691/544 |
