# Day-1 analyses on TV v2: partial-coverage hits (A) and abstention fallbacks (B)

Script: `src/cafa6/day1_analyses.py` (runtime about 85 s on 8 threads). Tables: `coverage.tsv`, `abstention.tsv`,
`fallbacks.tsv`, `summary.json`. Progress log, with definitions and fallback rules written before scoring:
`logs/day1_analyses.log`. Everything here is scored on TV v2 with `knn_sharp.Scorer` and the chosen_v2 test-like
cell weights. Nothing was scored on the GO-GPT test set.

Inputs and whole-set reference numbers (TV v2 mean Fmax):

| predictor | test-like | plain | test-like tau MF / BP / CC |
|---|---|---|---|
| kNN prior T5_k50_g0_na, train+val refs (`knn_trainval/tv_knn.npz`) | 0.7259 | 0.7378 | 0.50 / 0.39 / 0.59 |
| Ontologue OOF twin (`stacker_tv/l3_c0/pred_tv_oof_twin.npz`) | 0.7342 | 0.7300 | 0.30 / 0.26 / 0.48 |

## A. Do partial-coverage hits explain the 50-95% identity errors?

Setup. 1,878 TV v2 proteins have their best hit (highest pident among the top-50-by-bitscore hits used by the
prior) at 50-95% identity. Together they carry 35.0% of the test-like weight. The hit table has only query
coverage (`qcov`), with no subject coordinates. Target coverage is therefore estimated as
`min(100, qcov * len(query) / len(subject))` (`tcov_est`). This assumes the aligned span is about the same length
on both sequences. Precision, recall and F1 are read per protein at each predictor's whole-set test-like
threshold. Precision is averaged over proteins with a prediction (cafa-style). F1 is the mean per-protein F1.
Values below are IA-weighted and averaged over MF/BP/CC. Unweighted versions are in `coverage.tsv` and show
the same pattern.

kNN prior, grouped by the best hit's query coverage:

| qcov | n | test-like share | precision | recall | per-protein F1 | group Fmax |
|---|---|---|---|---|---|---|
| < 50% | 187 | 0.049 | 0.603 | 0.568 | 0.500 | 0.593 |
| 50-80% | 128 | 0.036 | 0.649 | 0.580 | 0.534 | 0.620 |
| >= 80% | 1563 | 0.265 | 0.680 | 0.687 | 0.615 | 0.684 |

Ontologue OOF, same groups:

| qcov | precision | recall | per-protein F1 | group Fmax |
|---|---|---|---|---|
| < 50% | 0.635 | 0.551 | 0.484 | 0.596 |
| 50-80% | 0.675 | 0.517 | 0.485 | 0.597 |
| >= 80% | 0.707 | 0.632 | 0.584 | 0.671 |

kNN prior, grouped by estimated target coverage: < 50% (n 178, share 0.030) precision 0.592, recall 0.638,
F1 0.519; 50-80% (n 132, share 0.046) 0.615 / 0.654 / 0.562; >= 80% (n 1568) 0.680 / 0.671 / 0.607. The
kNN-weight-averaged qcov over all used hits gives the same picture as the best-hit qcov.

What the numbers say:

- Low query coverage costs more recall than precision. Compared with qcov >= 80%, kNN precision is 0.08 lower
  (<50%) and 0.03 lower (50-80%), while recall is 0.12 and 0.11 lower. For Ontologue the deficits are 0.07 and
  0.03 in precision versus 0.08 and 0.12 in recall. In CC, kNN precision does not drop at all (0.713 for both
  <50% and >=80%). This matches "the query has domains the hit does not cover" (missed terms) more than "the hit
  brings terms from its other domains" (extra terms).
- The low-precision signature the hypothesis predicts does show up for low *target* coverage: precision is
  0.09 lower and recall only 0.03 lower. That group is small, though: 178 proteins, 3.0% of the test-like
  weight, and the coverage itself is an estimate.
- False-positive mass is not concentrated in low-coverage proteins. Best-hit qcov < 80% proteins hold 20.2% of
  the kNN false-positive IA mass in the 50-95% band (plain weighting), against 16.8% of the band's proteins and
  16.8% of its predicted IA mass. Under test-like weighting they hold 19.4% of the false-positive mass against
  24.2% of the weight. For tcov_est < 80% the figures are 17.2% vs 16.5% of proteins (plain) and 15.1% vs 21.6%
  (test-like). The 50-95% band accounts for 28% (plain) or 52% (test-like) of all kNN false-positive IA mass
  on TV v2. Its low-coverage part accounts for 5.7% (plain) or 10.2% (test-like).
- Oracle ceiling. Deleting every false positive of the qcov < 80% band proteins raises whole-set test-like
  Fmax by +0.0152 for the kNN prior and +0.0107 for Ontologue (plain: +0.0076 and +0.0070). The same oracle on
  the qcov >= 80% band proteins gives +0.047 and +0.035. Per unit of test-like weight the two are the same
  (0.18 for kNN in both groups), so partial-coverage proteins are not a disproportionate error source. They
  have about the error rate their weight predicts.

Conclusion for A: partial coverage is a minor error source. A method that removes transferred terms from
partially covering hits works on proteins worth at most +0.011 test-like over Ontologue even with perfect
precision. A practical method would recover a fraction of that, which is in the range of the 0.003 bar. The
larger deficit in this group is recall.

## B. Abstention at Ontologue's per-aspect thresholds

Definition: a (protein, aspect) pair abstains when the protein has ground truth in the aspect but no positive-IA
Ontologue term at score >= tau_a (MF 0.30, BP 0.26, CC 0.48, the whole-set test-like Fmax thresholds).

| aspect | abstaining / with GT | share | test-like share | abstainers' share of GT IA mass | recall shortfall* |
|---|---|---|---|---|---|
| MF | 62 / 3838 | 1.6% | 2.6% | 1.3% | 0.012 |
| BP | 149 / 4637 | 3.2% | 2.3% | 2.2% | 0.022 |
| CC | 11 / 4720 | 0.2% | 0.4% | 0.2% | 0.002 |

*Mean recall the aspect would gain if the abstainers reached the recall of the predicting proteins in their
identity bin (unweighted). This is an upper reference, not an achievable gain.

Abstention share by identity bin, pooled over aspects: no_hit 9.8%, <30 3.2%, 30-50 5.6%, 50-80 8.0%,
80-95 2.1%, 95-99.9 0.9%, 100 1.2%. The highest rates are BP at no_hit (17%) and BP at 50-80% (13%; 3.8%
test-like). Per-aspect and per-bin details are in `abstention.tsv`.

Fallbacks for abstaining pairs only. The rules were written to the log before scoring. The paired protein
bootstrap uses `Scorer.bootstrap` with 1000 reps and seed 0.

| variant | test-like mean Fmax | diff [95% CI] | plain mean Fmax | diff [95% CI] |
|---|---|---|---|---|
| Ontologue base | 0.7342 | | 0.7300 | |
| F1: HiGO top-3 (IA > 0) of the aspect at HiGO score | 0.7349 | +0.0008 [+0.0002, +0.0015] | 0.7308 | +0.0008 [+0.0001, +0.0014] |
| F2: top-1 Ontologue term raised to tau_a | 0.7351 | +0.0010 [+0.0004, +0.0015] | 0.7307 | +0.0007 [+0.0001, +0.0013] |

Both gains are above zero but about a third of the 0.003 bar, so neither fallback is promising. Almost all of
the gain is in MF (test-like MF +0.0021 for F1, +0.0027 for F2). BP and CC are unchanged within ±0.0004.

At the base thresholds:

- F1 makes 41 of 62 MF abstainers predict, with precision 0.76 and recall 0.32. It makes 93 of 149 BP
  abstainers predict, with precision 0.54 and recall 0.05.
- F2 covers 58 of 62 MF abstainers (precision 0.74, recall 0.23) and 139 of 149 BP abstainers (precision 0.66,
  recall 0.07). 4 MF and 10 BP pairs have no positive-IA Ontologue term at all.

So the forced terms are reasonably precise but too generic to recover much recall, especially in BP. The
shortfall ceiling is itself small (about 0.01-0.02 mean recall in MF and BP).

Caveats. This is an exploratory analysis, not a registered selection. The thresholds and abstainer sets come
from the same TV v2 proteins they are scored on, which slightly favours F2. For 18 TV proteins, tv_knn
`max_pident` comes from a hit outside the top 50 by bitscore. Section A bands them by the top-50 best hit.

## What this means for next steps

- Do not build alignment-conditioned retrieval (idea 2) as a precision fix for multi-domain hits. On TV v2,
  low-coverage proteins carry false-positive mass in proportion to their weight. Their perfect-precision
  ceiling is about +0.011 test-like over Ontologue. What sets them apart is lower recall.
- If idea 2 is revisited, the recall side is the version worth a quick check first. That means retrieving per
  query region, so that uncovered query domains get their own hits. It needs its own oracle (e.g., the recall
  of the union of hits covering different query regions) before any build.
- Drop the abstention fallbacks as a standalone item. Abstention affects 0.2-3.2% of (protein, aspect) pairs,
  and both pre-specified fallbacks gain about +0.001 test-like. A top-1 floor (F2) is a harmless default for
  MF if a deployment needs every protein to have a prediction, but it does not move the metric.
- The largest error mass is in the 50-95% band at high coverage (oracle +0.035 for Ontologue). Effort on
  low-identity transfer in general is better spent there than on coverage-specific handling.
