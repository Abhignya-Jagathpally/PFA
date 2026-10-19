# fresh24 benchmark: label-free statistics

Built by `python -m cafa6.build_fresh24` (construction in `data/fresh24/README.md`). No model was scored on this set; the numbers below use sequences and DIAMOND hits only.

Proteins: 24,141. New experimental annotations dated 2024-03-01 to 2026-10-09 (cutoff 20240229).

## Proteins and terms per aspect and split

| split | aspect | proteins | terms (unpropagated) |
|---|---|---|---|
| nk | MFO | 3924 | 4865 |
| nk | BPO | 3975 | 8723 |
| nk | CCO | 17976 | 24513 |
| nk | any | 22700 | 38101 |
| lk | MFO | 449 | 518 |
| lk | BPO | 587 | 1263 |
| lk | CCO | 672 | 1191 |
| lk | any | 1441 | 2972 |

nk = no experimental annotation in any aspect at the cutoff; lk = experimental annotation in another aspect but not in this one. A protein can be lk in one aspect only if it had prior annotation elsewhere, so nk and lk proteins are disjoint.

## Composition

Swiss-Prot 6,663, TrEMBL 17,478. HPA-only proteins (every new term assigned by HPA, all CCO, almost all rows from one load dated 2025-11): 11,469 (11,469 nk). Proteins sharing an identical sequence with another fresh24 protein: 359.

New-term rows by assigning group (top 15): HPA 16,004, UniProt 11,576, MGI 6,278, SynGO 2,301, ZFIN 1,132, TAIR 941, FlyBase 871, PHI-base 591, CGD 439, dictyBase 177, ComplexPortal 101, RGD 97, Reactome 89, SigSci 78, PomBase 70

## Best DIAMOND identity to GO-GPT train+val (per aspect and split)

DIAMOND --more-sensitive -e 1e-3 -k 51, --dbsize = GO-GPT train letters; best identity = max pident over hits (0 = no hit); same_org_best_hit = the top-bitscore hit has the query's UniProt organism string (GO-GPT stores organisms as strings).

| split | aspect | n | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 | same_org_best_hit |
|---|---|---|---|---|---|---|---|---|---|---|
| nk | MFO | 3924 | 394 | 156 | 1337 | 1169 | 523 | 239 | 106 | 362 |
| nk | BPO | 3975 | 460 | 83 | 734 | 1304 | 741 | 417 | 236 | 682 |
| nk | CCO | 17976 | 2218 | 264 | 1311 | 2018 | 2389 | 2440 | 7336 | 11021 |
| nk | any | 22700 | 2653 | 419 | 2669 | 3658 | 3140 | 2724 | 7437 | 11448 |
| lk | MFO | 449 | 13 | 3 | 7 | 22 | 92 | 102 | 210 | 335 |
| lk | BPO | 587 | 36 | 2 | 12 | 50 | 62 | 133 | 292 | 409 |
| lk | CCO | 672 | 27 | 1 | 9 | 22 | 101 | 213 | 299 | 453 |
| lk | any | 1441 | 71 | 5 | 23 | 86 | 217 | 403 | 636 | 993 |
| all | MFO | 4373 | 407 | 159 | 1344 | 1191 | 615 | 341 | 316 | 697 |
| all | BPO | 4562 | 496 | 85 | 746 | 1354 | 803 | 550 | 528 | 1091 |
| all | CCO | 18648 | 2245 | 265 | 1320 | 2040 | 2490 | 2653 | 7635 | 11474 |
| all | any | 24141 | 2724 | 424 | 2692 | 3744 | 3357 | 3127 | 8073 | 12441 |
| nk excl. HPA-only | MFO | 3924 | 394 | 156 | 1337 | 1169 | 523 | 239 | 106 | 362 |
| nk excl. HPA-only | BPO | 3975 | 460 | 83 | 734 | 1304 | 741 | 417 | 236 | 682 |
| nk excl. HPA-only | CCO | 6507 | 1835 | 260 | 1275 | 1396 | 776 | 613 | 352 | 963 |
| nk excl. HPA-only | any | 11231 | 2270 | 415 | 2633 | 3036 | 1527 | 897 | 453 | 1390 |
| lk excl. HPA-only | MFO | 449 | 13 | 3 | 7 | 22 | 92 | 102 | 210 | 335 |
| lk excl. HPA-only | BPO | 587 | 36 | 2 | 12 | 50 | 62 | 133 | 292 | 409 |
| lk excl. HPA-only | CCO | 672 | 27 | 1 | 9 | 22 | 101 | 213 | 299 | 453 |
| lk excl. HPA-only | any | 1441 | 71 | 5 | 23 | 86 | 217 | 403 | 636 | 993 |

100% bin: 8,073 proteins, of which 5,325 have a 100% hit covering >= 95% of the query (HSP coverage) and 6,984 are HPA-only; the rest are fragments or isoforms matching part of a GO-GPT protein.

Same-organism share (top hit from the same organism): 51.5% of all proteins, 58.1% of proteins with a hit.

## Top 15 organisms

| taxon | organism | n | nk | lk | hpa_only | swissprot | same_org_best_hit | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9606 | Homo sapiens | 11719 | 11712 | 7 | 11469 | 233 | 10165 | 420 | 8 | 64 | 678 | 1678 | 1853 | 7018 |
| 10090 | Mus musculus | 2826 | 1569 | 1257 | 0 | 538 | 1368 | 127 | 11 | 73 | 327 | 645 | 724 | 919 |
| 185431 | Trypanosoma brucei brucei (strain 927/4 GUTat10.1) | 1553 | 1553 | 0 | 0 | 3 | 148 | 864 | 139 | 403 | 96 | 17 | 32 | 2 |
| 7955 | Danio rerio | 694 | 685 | 9 | 0 | 79 | 39 | 7 | 8 | 64 | 425 | 169 | 19 | 2 |
| 3702 | Arabidopsis thaliana | 584 | 498 | 86 | 0 | 183 | 350 | 96 | 19 | 159 | 210 | 35 | 34 | 31 |
| 7227 | Drosophila melanogaster | 381 | 373 | 8 | 0 | 94 | 119 | 45 | 25 | 154 | 94 | 6 | 20 | 37 |
| 10116 | Rattus norvegicus | 218 | 217 | 1 | 0 | 141 | 8 | 2 | 0 | 1 | 22 | 100 | 87 | 6 |
| 237561 | Candida albicans (strain SC5314 / ATCC MYA-2876) | 145 | 141 | 4 | 0 | 31 | 7 | 21 | 8 | 62 | 48 | 5 | 1 | 0 |
| 224308 | Bacillus subtilis (strain 168) | 139 | 139 | 0 | 0 | 139 | 4 | 29 | 11 | 73 | 24 | 2 | 0 | 0 |
| 9913 | Bos taurus | 127 | 127 | 0 | 0 | 116 | 2 | 0 | 0 | 0 | 16 | 54 | 55 | 2 |
| 229533 | Gibberella zeae (strain ATCC MYA-4620 / CBS 123657 / FGSC 9075 / NRRL 31084 / PH-1) | 90 | 90 | 0 | 0 | 24 | 0 | 13 | 3 | 37 | 29 | 8 | 0 | 0 |
| 6239 | Caenorhabditis elegans | 84 | 81 | 3 | 0 | 49 | 21 | 11 | 8 | 36 | 16 | 1 | 2 | 10 |
| 242507 | Pyricularia oryzae (strain 70-15 / ATCC MYA-4617 / FGSC 8958) | 83 | 83 | 0 | 0 | 33 | 0 | 19 | 1 | 32 | 30 | 1 | 0 | 0 |
| 39947 | Oryza sativa subsp. japonica | 73 | 73 | 0 | 0 | 69 | 7 | 0 | 0 | 12 | 47 | 14 | 0 | 0 |
| 99287 | Salmonella typhimurium (strain LT2 / SGSC1412 / ATCC 700720) | 66 | 66 | 0 | 0 | 66 | 0 | 5 | 2 | 2 | 11 | 28 | 18 | 0 |

## Dates of the new annotations (year of GAF column 14, GO-GPT-release GT rows)

| 2024 | 2025 | 2026 |
|---|---|---|
| 10509 | 26839 | 3725 |

## Construction counts

```
{
 "prior": {
  "pairs_release219": 245532,
  "pairs_release219_via_secondary": 416,
  "pairs_dated": 266550,
  "pairs_union": 302083,
  "proteins": 167535,
  "pairs_dated_not_in_219": 56551,
  "pairs_219_not_dated_in_current": 35533
 },
 "late_rows": {
  "rows_dated_after_cutoff": 248623,
  "rows_triple_already_in_219": 97602,
  "rows_pair_in_219_only": 21073,
  "rows_pair_dated_before": 160524,
  "rows_new_pair": 67026
 },
 "disjoint": {
  "candidates": 29336,
  "removed_accession": 4496,
  "not_current_primary": 0,
  "removed_identical_sequence": 498,
  "kept": 24342
 },
 "go_mapping": {
  "gogpt": {
   "obo": "runs/gogpt_cafa5/ref/go-basic_gogpt.obo",
   "data_version": "releases/2023-01-01",
   "terms_in": 4818,
   "terms_not_in_release": 199,
   "rows_not_in_release": 581,
   "rows_alt_id_mapped": 0,
   "rows_namespace_mismatch": 0,
   "rows_out": 41073,
   "proteins_out": 24141
  },
  "current": {
   "obo": "data/fresh24/ref/go-basic_current.obo",
   "data_version": "releases/2026-07-26",
   "terms_in": 4818,
   "terms_not_in_release": 7,
   "rows_not_in_release": 13,
   "rows_alt_id_mapped": 0,
   "rows_namespace_mismatch": 0,
   "rows_out": 41567,
   "proteins_out": 24336
  }
 }
}
```
