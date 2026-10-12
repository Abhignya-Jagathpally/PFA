# kNN prior with train + val references (knn_trainval)

Question: the sharpened kNN prior T5_k50_g0_na copies labels from GO-GPT training proteins only. GO-GPT's
13,305 validation proteins carry labels from the same pre-cutoff snapshot. Does adding them as references help?

Code: `src/cafa6/knn_trainval.py` (stages `--diamond --stats --validity --register --tv --write`).
Logs: `logs/knn_trainval.log`, plus `logs/knn_trainval_{step1,tv,write}.out`. CPU only, no test labels used.

## DIAMOND
Test (8,630) and TV v2 (7,566) queries against the 13,305 val proteins, with the settings of
`temporal_val.run_diamond`: `--more-sensitive -e 1e-3 -k 51`, self accessions removed, best HSP per pair, top 50
by bitscore. `--dbsize 66,295,805` (letters of the 119,749-protein training database) keeps e-values and the
1e-3 cut on the same scale as the existing train hits. 173,725 hits for 13,253 queries; 87 self-accession hits
removed (the 87 test accessions that are also val proteins). The prior merges these hits with the existing ones
and keeps the top 50 by bitscore over the union. Raw val-reference hits: `runs/knn_trainval/hits_valref.parquet`.

## Label-free test statistics (best-hit identity, train only vs train + val)
| bin | before | after | improved (from this bin) |
|---|---|---|---|
| no_hit | 250 | 233 | 17 |
| <30 | 79 | 72 | 14 |
| 30-50 | 694 | 656 | 105 |
| 50-80 | 1,627 | 1,511 | 203 |
| 80-95 | 1,360 | 1,305 | 163 |
| 95-99.9 | 2,430 | 2,534 | 97 |
| 100 | 2,190 | 2,319 | 0 |

599 of 8,630 test proteins (6.9%) get a higher best identity, and 397 change bin. 233 move into >=95% (213 of them
through a same-organism val hit), and 129 move into 100% (123 same organism). Of the 599 improved, 361 have a
same-organism val protein as their new best hit. Same-organism best hit (top bitscore, as in `cells.parquet`):
5,138 -> 5,387. 7,126 test proteins have at least one val hit. Full transition table:
`stats_test_transition.tsv`; TV v2 equivalent: `stats_tv_*` (465 of 7,566 improved, 89 into >=95%, 105 into
100%). Organism is the `organism` column of `data/proc_gogpt/train.parquet`, as used by `temporal_val.best_hit_cells`.

## Validity
- Label dates: train and val are the train and validation parquet files of `wanglab/gogpt-training-data`, a random
  90/10 split of one snapshot with annotations up to Nov 2022 (`prepare_gogpt.py`; BioReason-Pro describes the
  validation split as random). The test set holds annotations gained Mar 2023 to Feb 2024. Val labels therefore
  predate the test annotation window, like the training labels.
- Accession overlap val vs test: 87 proteins (LK test proteins whose other aspects were known before the cutoff).
  They are excluded as self hits, which is the same convention the existing pipeline uses for the 673 test
  accessions shared with train. No val hit with q == s remains.
- Sequence overlap: 148 test proteins have an identical sequence in val. 86 of them are the shared accessions
  above; the other 62 are different accessions with the same sequence. The train-side figure is 1,242.
- Val vs TV v2: no accession overlap. 60 TV proteins have an identical-sequence val protein under another
  accession, which is the duplicate-accession situation the DA stratum models.
- TV analogue labels: val labels are cut to T0 with the TV v2 dated GOA rule (`dated_goa_v2` + `build_split`, `tv`
  and `pk` pairs removed). 7,065 pairs on 469 val proteins are removed, and 52 val proteins are left without labels.
  As for the training references, organisms outside the dated GOA sources cannot be stripped.

## TV v2 result (pre-registered 09:09:03 in `runs/test_looks.md`, scored 09:12:11)
Rule: keep if test-like TV v2 mean gain >= 0.003, paired bootstrap 95% CI (1000 reps, test-like weights) excludes
0, and plain TV v2 mean not lower by more than 0.002. The rebuilt train-only baseline reproduces
`data/proc_gogpt_tv2/knn/tv_knn.npz` exactly (max abs diff 0).

| | train only | train + val | diff [95% CI] |
|---|---|---|---|
| test-like v2 mean Fmax | 0.7168 | 0.7259 | +0.0091 [+0.0070, +0.0110] |
| plain mean Fmax | 0.7345 | 0.7378 | +0.0033 [+0.0018, +0.0048] |
| MF / BP / CC (plain) | 0.743 / 0.686 / 0.775 | 0.747 / 0.690 / 0.777 | |

Verdict: KEPT. All three conditions hold.

Diagnostics (not part of the rule):
- With test-like weights recomputed from the train + val cells of both TV and test, the gain is +0.0036
  [+0.0020, +0.0051]. The chosen_v2 weights stratify by the train-only cells, which upweights the TV proteins
  whose cell improves. The test gain is therefore more likely in the +0.004 to +0.009 range than at the top of it.
- The gain sits in the 465 TV proteins whose best identity improves (0.658 -> 0.670). The other proteins move
  0.740 -> 0.742, because extra lower-ranked neighbours also shift their priors. Both strata gain: DA
  0.769 -> 0.772 and dated 0.474 -> 0.481.
- Per identity bin (train-only bins), every bin gains, from +0.0016 (80-95) to +0.021 (<30).

## Files
- `data/proc_gogpt_tv2/knn_trainval/tv_knn.npz`, `test_knn.npz`: same format as `knn/`, read with
  `knn_sharp.load_knn(split, Path("data/proc_gogpt_tv2/knn_trainval"))`. The settings key records
  `reference_set: train + val` and the label database. The existing `knn/` files are unchanged.
- `data/proc_gogpt_tv2/knn_trainval/diamond_hits_{tv,test}_trainval.parquet`: the merged hit tables used by
  the priors (existing hits plus val-reference hits).
- `runs/knn_trainval/`: `hits_valref.parquet`, `stats*.{json,tsv,parquet}`, `validity.json`, `tv_result.json`,
  `tv_summary.tsv`.

The test priors were not scored. Swapping them into a pipeline does not by itself trigger a test look; it goes
through the next pre-registered look like any other kept change.
