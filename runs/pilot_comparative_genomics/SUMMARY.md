# Pilot: cross-species orthology signal for CAFA-6 GO prediction

Research question: does DIAMOND search restricted to a curated cross-species reference panel, with same-taxon hits excluded, add incremental signal on top of the existing same-corpus DIAMOND-kNN homology baseline (`diamond_knn` in `baselines.py`, which searches the whole training split regardless of species) -- or is it redundant with it?

Reference panel: 10 model organisms (Homo sapiens, Mus musculus, Arabidopsis thaliana, Saccharomyces cerevisiae S288C, Rattus norvegicus, Schizosaccharomyces pombe, Escherichia coli K-12, Drosophila melanogaster, Caenorhabditis elegans, Mycobacterium tuberculosis H37Rv); 53909 train-split proteins (10 distinct reference taxa actually present).

## Coverage: fraction of proteins with ANY cross-species ortholog hit

| split | n proteins | fraction with >=1 cross-species hit |
|---|---|---|
| val | 8377 | 0.7010 |
| test | 8058 | 0.7130 |

Cross-species hits surviving the same-taxon filter: 198511 rows (raw DIAMOND hits against the reference panel, before filtering: 260691; 76.1% kept). Note 81.6% of val/test proteins are themselves from a reference taxon, so a large share of raw hits against the panel are same-species (excluded).

## Fmax_w per aspect and Kaggle mean

### Validation

| method | MF | BP | CC | kaggle_mean_Fw |
|---|---|---|---|---|
| diamond+naive | 0.5793 | 0.3427 | 0.4837 | 0.4685 |
| diamond+ortho | 0.5747 | 0.3380 | 0.4803 | 0.4644 |
| diamond_knn | 0.5279 | 0.3310 | 0.4507 | 0.4365 |
| ortho_knn | 0.4759 | 0.2782 | 0.3903 | 0.3815 |
| naive | 0.3448 | 0.2184 | 0.3544 | 0.3059 |

### Test

| method | MF | BP | CC | kaggle_mean_Fw |
|---|---|---|---|---|
| diamond+naive | 0.5751 | 0.3281 | 0.4809 | 0.4614 |
| diamond+ortho | 0.5704 | 0.3230 | 0.4773 | 0.4569 |
| diamond_knn | 0.5233 | 0.3141 | 0.4500 | 0.4292 |
| ortho_knn | 0.4739 | 0.2703 | 0.3890 | 0.3777 |
| naive | 0.3631 | 0.2245 | 0.3617 | 0.3164 |

## Does diamond+ortho beat diamond+naive?

Both arms use the same rule (0.8 x kNN score + 0.2 x naive, naive when there is no hit), so the difference is the ortholog channel alone. Raw `diamond_knn` is listed for reference only; comparing against it would credit the naive prior to the ortholog channel.

- **val**: diamond+ortho 0.4644 vs diamond+naive 0.4685 (delta -0.0041), so diamond+ortho loses to diamond+naive.
- **test**: diamond+ortho 0.4569 vs diamond+naive 0.4614 (delta -0.0045), so diamond+ortho loses to diamond+naive.

The reference panel is a subset of the training split that `diamond_knn` already searches, so the ortholog channel cannot find new neighbours; it only re-weights a subset of the existing ones, and taking the elementwise max inflates scores for terms carried by those neighbours.

## Ortholog hit quality by percent-identity bin (`identity_bin`)

Identity of each protein's best cross-species hit (all val/test proteins):

| split | no_hit (all proteins) | <30% | 30-50% | >=50% |
|---|---|---|---|---|
| val | 2505/8377 (29.9%) | 982 (11.7%) | 3516 (42.0%) | 1374 (16.4%) |
| test | 2313/8058 (28.7%) | 1021 (12.7%) | 3614 (44.8%) | 1110 (13.8%) |

Share of proteins with a hit whose best hit is below 30% identity (where homology transfer is least reliable):

- **val**: 16.7%
- **test**: 17.8%

## Scope

This is a pilot, not a trained model. The reference panel (10 model organisms) covers 53909/65603 (82.2%) of the training split, and 81.6% of val/test proteins come from one of these taxa, so the corpus behind `diamond_knn` is already dominated by the same species.
