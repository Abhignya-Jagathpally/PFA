# Annotation-forecast data: statistics

Built by `python -m cafa6.build_forecast` (rules in `data/forecast/README.md`). Rows are (protein, aspect) examples; a gain example has at least one gained term. Terms are propagated, aspect roots excluded.

## Period A

Proteins 24,511 (with a gain 8,461, without 16,050, ratio without/with 1.90); rows 73,533 (gain rows 10,614). Split: train 22,038 / valid 2,473 proteins.

### Rows per source, split, aspect

| source | split | aspect | rows | gain_rows | gained_terms | mean_input |
|---|---|---|---|---|---|---|
| A_extra | train | BPO | 405 | 204 | 3,595 | 4.336 |
| A_extra | train | CCO | 405 | 222 | 2,373 | 0.849 |
| A_extra | train | MFO | 405 | 87 | 584 | 0.558 |
| A_extra | valid | BPO | 31 | 20 | 407 | 3.968 |
| A_extra | valid | CCO | 31 | 19 | 232 | 1.581 |
| A_extra | valid | MFO | 31 | 9 | 50 | 0.710 |
| A_gogpt | train | BPO | 21,633 | 5,021 | 64,764 | 35.664 |
| A_gogpt | train | CCO | 21,633 | 1,668 | 9,048 | 9.657 |
| A_gogpt | train | MFO | 21,633 | 2,363 | 9,581 | 5.726 |
| A_gogpt | valid | BPO | 2,442 | 578 | 6,818 | 35.114 |
| A_gogpt | valid | CCO | 2,442 | 173 | 896 | 9.492 |
| A_gogpt | valid | MFO | 2,442 | 250 | 1,108 | 5.656 |

### Gains by aspect status (nk: no input term in any aspect; lk: none in this aspect; known: input terms in this aspect)

| aspect | aspect_status | rows | gain_rows | gained_terms |
|---|---|---|---|---|
| BPO | known | 17,991 | 5,306 | 65,630 |
| BPO | lk | 6,140 | 334 | 6,534 |
| BPO | nk | 380 | 183 | 3,420 |
| CCO | known | 17,714 | 1,645 | 8,506 |
| CCO | lk | 6,417 | 210 | 1,513 |
| CCO | nk | 380 | 227 | 2,530 |
| MFO | known | 15,016 | 2,351 | 9,257 |
| MFO | lk | 9,115 | 274 | 1,470 |
| MFO | nk | 380 | 84 | 596 |

### Identity bins (best DIAMOND identity to references at the input time)

| bin | proteins | with_gain |
|---|---|---|
| no_hit | 1,674 | 417 |
| <30 | 645 | 194 |
| 30-50 | 4,059 | 1,314 |
| 50-80 | 5,660 | 1,879 |
| 80-95 | 4,502 | 1,774 |
| 95-99.9 | 4,560 | 1,770 |
| 100 | 3,411 | 1,113 |

### Top organisms

| organism | proteins | with_gain |
|---|---|---|
| Homo sapiens | 5,187 | 2,522 |
| Mus musculus | 2,934 | 1,241 |
| Arabidopsis thaliana | 2,092 | 164 |
| Drosophila melanogaster | 2,065 | 571 |
| Rattus norvegicus | 1,574 | 609 |
| Danio rerio | 1,446 | 784 |
| Escherichia coli | 960 | 579 |
| Schizosaccharomyces pombe | 944 | 421 |
| Saccharomyces cerevisiae | 943 | 218 |
| Caenorhabditis elegans | 863 | 278 |
| Trypanosoma brucei brucei | 696 | 1 |
| Dictyostelium discoideum | 409 | 199 |

### Gained terms per gain row (propagated)

mean 9.4; quantiles 0.1: 1, 0.25: 3, 0.5: 6, 0.75: 11, 0.9: 20, 0.99: 58

| gained terms | rows |
|---|---|
| 1 | 1,307 |
| 2 | 1,281 |
| 3-5 | 2,703 |
| 6-10 | 2,384 |
| 11-20 | 1,883 |
| 21-50 | 919 |
| 51-100 | 109 |
| >100 | 28 |

Per protein (all aspects): mean 11.8, median 7.

A_gogpt + A_extra gained terms supported by a release 219 row dated after 20221130: 72.0% (the rest are supported only by rows dated before, i.e. absent from S0 for filtering or late-import reasons).

Assigning groups of the rows behind gained terms (rows counted once per example): UniProt 4,362, ZFIN 814, MGI 771, FlyBase 716, EcoCyc 654, RGD 592, PomBase 498, HPA 424, BHF-UCL 339, Reactome 312

### Retrieval-recoverable share of gained terms (kNN prior at the input time)

Share of gained (protein, term) pairs whose prior score is > 0 / >= 0.1 / >= 0.3 / >= 0.5, and share of gain rows whose gained terms all have p > 0 / p >= 0.5.

| all | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|
| all | 10,614 | 99,456 | 0.546 | 0.303 | 0.240 | 0.186 | 0.333 | 0.096 |

| source | aspect | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| A_extra | BPO | 224 | 4,002 | 0.550 | 0.348 | 0.289 | 0.248 | 0.196 | 0.054 |
| A_extra | CCO | 241 | 2,605 | 0.815 | 0.726 | 0.702 | 0.640 | 0.631 | 0.311 |
| A_extra | MFO | 96 | 634 | 0.713 | 0.639 | 0.621 | 0.591 | 0.510 | 0.281 |
| A_gogpt | BPO | 5,599 | 71,582 | 0.517 | 0.261 | 0.199 | 0.147 | 0.231 | 0.049 |
| A_gogpt | CCO | 1,841 | 9,944 | 0.653 | 0.405 | 0.332 | 0.263 | 0.513 | 0.174 |
| A_gogpt | MFO | 2,613 | 10,689 | 0.563 | 0.347 | 0.273 | 0.218 | 0.400 | 0.117 |

| aspect | status | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| BPO | known | 5,306 | 65,630 | 0.510 | 0.243 | 0.180 | 0.130 | 0.227 | 0.045 |
| BPO | lk | 334 | 6,534 | 0.601 | 0.448 | 0.396 | 0.325 | 0.305 | 0.117 |
| BPO | nk | 183 | 3,420 | 0.546 | 0.341 | 0.290 | 0.253 | 0.180 | 0.060 |
| CCO | known | 1,645 | 8,506 | 0.654 | 0.380 | 0.303 | 0.229 | 0.509 | 0.155 |
| CCO | lk | 210 | 1,513 | 0.656 | 0.562 | 0.513 | 0.471 | 0.552 | 0.343 |
| CCO | nk | 227 | 2,530 | 0.815 | 0.727 | 0.704 | 0.641 | 0.630 | 0.304 |
| MFO | known | 2,351 | 9,257 | 0.546 | 0.305 | 0.228 | 0.174 | 0.398 | 0.096 |
| MFO | lk | 274 | 1,470 | 0.678 | 0.615 | 0.553 | 0.494 | 0.434 | 0.296 |
| MFO | nk | 84 | 596 | 0.701 | 0.648 | 0.643 | 0.611 | 0.476 | 0.321 |

| bin | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|
| 100 | 1,412 | 13,187 | 0.588 | 0.296 | 0.241 | 0.172 | 0.358 | 0.096 |
| 30-50 | 1,606 | 12,497 | 0.553 | 0.407 | 0.294 | 0.215 | 0.323 | 0.089 |
| 50-80 | 2,335 | 21,214 | 0.577 | 0.352 | 0.266 | 0.202 | 0.332 | 0.101 |
| 80-95 | 2,291 | 22,443 | 0.576 | 0.297 | 0.251 | 0.211 | 0.375 | 0.110 |
| 95-99.9 | 2,236 | 24,156 | 0.573 | 0.265 | 0.216 | 0.172 | 0.375 | 0.105 |
| <30 | 233 | 1,888 | 0.335 | 0.319 | 0.259 | 0.219 | 0.133 | 0.069 |
| no_hit | 501 | 4,071 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Period B

Proteins 26,361 (with a gain 8,787, without 17,574, ratio without/with 2.00); rows 79,083 (gain rows 12,410). Split: train 23,736 / valid 2,625 proteins.

### Rows per source, split, aspect

| source | split | aspect | rows | gain_rows | gained_terms | mean_input |
|---|---|---|---|---|---|---|
| B_gogpt | train | BPO | 23,736 | 4,590 | 80,636 | 29.599 |
| B_gogpt | train | CCO | 23,736 | 3,276 | 23,419 | 8.712 |
| B_gogpt | train | MFO | 23,736 | 3,359 | 18,516 | 5.039 |
| B_gogpt | valid | BPO | 2,625 | 502 | 8,575 | 29.177 |
| B_gogpt | valid | CCO | 2,625 | 326 | 2,478 | 8.801 |
| B_gogpt | valid | MFO | 2,625 | 357 | 1,853 | 4.808 |

### Gains by aspect status (nk: no input term in any aspect; lk: none in this aspect; known: input terms in this aspect)

| aspect | aspect_status | rows | gain_rows | gained_terms |
|---|---|---|---|---|
| BPO | known | 17,727 | 3,050 | 44,468 |
| BPO | lk | 6,547 | 698 | 14,737 |
| BPO | nk | 2,087 | 1,344 | 30,006 |
| CCO | known | 17,748 | 2,338 | 13,733 |
| CCO | lk | 6,526 | 459 | 4,181 |
| CCO | nk | 2,087 | 805 | 7,983 |
| MFO | known | 15,122 | 2,067 | 9,408 |
| MFO | lk | 9,152 | 682 | 4,353 |
| MFO | nk | 2,087 | 967 | 6,608 |

### Identity bins (best DIAMOND identity to references at the input time)

| bin | proteins | with_gain |
|---|---|---|
| no_hit | 1,974 | 504 |
| <30 | 733 | 216 |
| 30-50 | 4,444 | 1,199 |
| 50-80 | 5,884 | 1,541 |
| 80-95 | 4,915 | 1,789 |
| 95-99.9 | 5,335 | 2,451 |
| 100 | 3,076 | 1,087 |

### Top organisms

| organism | proteins | with_gain |
|---|---|---|
| Homo sapiens | 3,826 | 1,698 |
| Mus musculus | 3,779 | 1,855 |
| Drosophila melanogaster | 2,482 | 698 |
| Rattus norvegicus | 2,429 | 1,365 |
| Arabidopsis thaliana | 2,424 | 280 |
| Trypanosoma brucei brucei | 1,156 | 373 |
| Saccharomyces cerevisiae | 1,058 | 188 |
| Schizosaccharomyces pombe | 918 | 217 |
| Caenorhabditis elegans | 900 | 228 |
| Danio rerio | 856 | 29 |
| Escherichia coli | 619 | 89 |
| Dictyostelium discoideum | 498 | 266 |

### Gained terms per gain row (propagated)

mean 10.9; quantiles 0.1: 2, 0.25: 3, 0.5: 7, 0.75: 14, 0.9: 26, 0.99: 55.9

| gained terms | rows |
|---|---|
| 1 | 1,150 |
| 2 | 1,283 |
| 3-5 | 3,010 |
| 6-10 | 2,840 |
| 11-20 | 2,314 |
| 21-50 | 1,610 |
| 51-100 | 172 |
| >100 | 31 |

Per protein (all aspects): mean 15.4, median 10.

### Retrieval-recoverable share of gained terms (kNN prior at the input time)

Share of gained (protein, term) pairs whose prior score is > 0 / >= 0.1 / >= 0.3 / >= 0.5, and share of gain rows whose gained terms all have p > 0 / p >= 0.5.

| all | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|
| all | 12,410 | 135,477 | 0.616 | 0.399 | 0.331 | 0.273 | 0.356 | 0.140 |

| source | aspect | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| B_gogpt | BPO | 5,092 | 89,211 | 0.588 | 0.362 | 0.290 | 0.230 | 0.259 | 0.076 |
| B_gogpt | CCO | 3,602 | 25,897 | 0.631 | 0.409 | 0.353 | 0.302 | 0.330 | 0.113 |
| B_gogpt | MFO | 3,716 | 20,369 | 0.718 | 0.546 | 0.486 | 0.427 | 0.516 | 0.253 |

| aspect | status | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| BPO | known | 3,050 | 44,468 | 0.555 | 0.258 | 0.186 | 0.135 | 0.232 | 0.047 |
| BPO | lk | 698 | 14,737 | 0.674 | 0.505 | 0.426 | 0.341 | 0.384 | 0.148 |
| BPO | nk | 1,344 | 30,006 | 0.594 | 0.446 | 0.376 | 0.315 | 0.253 | 0.106 |
| CCO | known | 2,338 | 13,733 | 0.574 | 0.291 | 0.231 | 0.173 | 0.276 | 0.064 |
| CCO | lk | 459 | 4,181 | 0.714 | 0.551 | 0.508 | 0.469 | 0.449 | 0.242 |
| CCO | nk | 805 | 7,983 | 0.685 | 0.538 | 0.483 | 0.438 | 0.416 | 0.181 |
| MFO | known | 2,067 | 9,408 | 0.658 | 0.426 | 0.362 | 0.301 | 0.460 | 0.162 |
| MFO | lk | 682 | 4,353 | 0.806 | 0.668 | 0.612 | 0.562 | 0.630 | 0.391 |
| MFO | nk | 967 | 6,608 | 0.745 | 0.637 | 0.579 | 0.518 | 0.556 | 0.351 |

| bin | examples | gained_terms | p>0 | p>=0.1 | p>=0.3 | p>=0.5 | all_gained_p>0 | all_gained_p>=0.5 |
|---|---|---|---|---|---|---|---|---|
| 100 | 1,550 | 14,279 | 0.605 | 0.334 | 0.255 | 0.174 | 0.339 | 0.083 |
| 30-50 | 1,692 | 20,961 | 0.648 | 0.532 | 0.407 | 0.322 | 0.388 | 0.170 |
| 50-80 | 2,162 | 24,391 | 0.700 | 0.508 | 0.424 | 0.352 | 0.463 | 0.201 |
| 80-95 | 2,499 | 26,930 | 0.673 | 0.401 | 0.359 | 0.316 | 0.412 | 0.177 |
| 95-99.9 | 3,532 | 37,644 | 0.641 | 0.352 | 0.302 | 0.256 | 0.318 | 0.112 |
| <30 | 317 | 3,749 | 0.501 | 0.444 | 0.350 | 0.279 | 0.274 | 0.148 |
| no_hit | 658 | 7,523 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## What retrieval proposes beyond the input (kNN prior >= 0.5, same aspect, not in input_terms)

Rows with and without a gain; precision = share of the proposed new terms that are gained terms. This is the false-alarm side of label transfer that a forecast model has to learn to suppress.

| period | has_gain | rows | rows_with_new_term | mean_new_terms | new_terms | new_terms_gained | precision |
|---|---|---|---|---|---|---|---|
| A | 0 | 62,919 | 0.606 | 8.269 | 520,305 | 0 | 0.000 |
| A | 1 | 10,614 | 0.650 | 14.448 | 153,352 | 18,531 | 0.121 |
| B | 0 | 66,673 | 0.605 | 8.153 | 543,589 | 0 | 0.000 |
| B | 1 | 12,410 | 0.737 | 13.792 | 171,163 | 37,043 | 0.216 |

## Leakage checks

```
{
 "A": {
  "proteins": 24511,
  "fresh24_accession_overlap": 0,
  "fresh24_linked_accession_overlap": 0,
  "fresh24_sequence_overlap": 0,
  "gogpt_test_accession_overlap": 0,
  "gogpt_test_linked_accession_overlap": 0,
  "gogpt_test_sequence_overlap": 0,
  "prior_ids_equal_example_proteins": true,
  "prior_ids_overlap_fresh24": 0,
  "hit_references_in_fresh24": 0,
  "hit_references_in_gogpt_test_only": 0,
  "max_row_date": 20240207,
  "clusters_in_both_splits": 0,
  "valid_with_ge50id_cov80_hit_to_train_example": 0,
  "valid_with_ge50id_any_cov_hit_to_train_example": 0,
  "valid_proteins": 2473
 },
 "B": {
  "proteins": 26361,
  "fresh24_accession_overlap": 0,
  "fresh24_linked_accession_overlap": 0,
  "fresh24_sequence_overlap": 0,
  "gogpt_test_accession_overlap": 0,
  "gogpt_test_linked_accession_overlap": 0,
  "gogpt_test_sequence_overlap": 0,
  "prior_ids_equal_example_proteins": true,
  "prior_ids_overlap_fresh24": 0,
  "hit_references_in_fresh24": 0,
  "hit_references_in_gogpt_test_only": 0,
  "max_row_date": null,
  "clusters_in_both_splits": 0,
  "valid_with_ge50id_cov80_hit_to_train_example": 0,
  "valid_with_ge50id_any_cov_hit_to_train_example": 0,
  "valid_proteins": 2625,
  "reference_terms_dated_le_T0": 4400542,
  "reference_terms_window_only_kept_by_closure": 2314,
  "reference_terms_nonroot": 4705147
 }
}
```

## Build counts

```
{
 "exclusions": {
  "gogpt_trainval_rows": 133054,
  "accession_in_gogpt_test": 760,
  "linked_accession_test_or_fresh24": 34,
  "identical_sequence_test_or_fresh24": 694,
  "pool": 131566
 },
 "A_label_counts": {
  "covered": 123580,
  "pool_not_in_219": 7986,
  "selected": 24075,
  "extra_window_accessions": 499,
  "extra_with_gain": 491,
  "extra_with_pre_rows": 59,
  "lost_terms": {
   "F": 19885,
   "P": 101466,
   "C": 11276
  }
 },
 "B_label_counts": {
  "dated_sources": {
   "rows_by_source": {
    "goa_reference": 673403,
    "rgd": 76566,
    "zfin": 28973,
    "goa_zebrafish_proteome": 15857
   },
   "proteins_by_source": {
    "goa_reference": 72298,
    "goa_zebrafish_proteome": 2452,
    "rgd": 7786,
    "zfin": 4630
   },
   "rows_via_secondary_accession": 280
  },
  "dated_rows_union": 1015815,
  "covered": 117966,
  "selected": 26361,
  "stripped": {
   "tv": 74044,
   "pk": 71132
  },
  "aspect_stats": {
   "tv_aspects": 5040,
   "pk_aspects": 7350,
   "rejected_unexplained_labels": 424,
   "rejected_cover_ge_0.9": 54,
   "win_but_no_gogpt_label": 682
  }
 },
 "B_reference_dating": {
  "reference_terms_nonroot": 4705147,
  "supported_by_row_dated_le_T0": 4400542,
  "window_support_only_kept_by_closure": 2314
 },
 "extra_fetch": {
  "requested": 491,
  "uniprot_release": [
   "2026_03",
   "02-September-2026"
  ],
  "not_primary": 27,
  "seq_modified_after_r219": 21,
  "identical_test_or_fresh24": 7,
  "kept": 436,
  "identical_gogpt_trainval_kept": 11
 },
 "verify": {
  "A": {
   "rows": 73533,
   "target_input_overlap": 0,
   "target_wrong_aspect": 0,
   "input_wrong_aspect": 0,
   "gogpt_target_in_S0": 0,
   "gogpt_input_not_S0": 0,
   "target_without_support": 0,
   "max_support_date": 20240207,
   "extra_input_not_dated_le_S0": 0,
   "extra_target_implied_by_pre": 0
  },
  "B": {
   "rows": 79083,
   "target_input_overlap": 0,
   "target_not_in_S0": 0,
   "input_not_in_S0": 0,
   "target_without_window_row": 0,
   "target_implied_by_row_le_T0": 0,
   "input_window_only": 1712,
   "max_target_support_date": 20221130,
   "min_target_support_date": 20211201
  }
 }
}
```
