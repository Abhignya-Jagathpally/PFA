# Post-hoc calibration on TV v2 (`src/cafa6/calibrate_tv.py`)

This stage runs on the stacker's outputs. It fits one monotone score map per (aspect, identity bin) and then
re-applies GO max-propagation. The input/output contract and the method are described in the module docstring.

## Why calibrate

IA-weighted Fmax uses one threshold per aspect across all proteins. If the kNN prior or the stacker is more
overconfident in one identity bin than in another, those proteins sit on a different score scale from everyone
else and the shared threshold suits none of them well. A monotone map inside a bin leaves every protein's term
ranking unchanged. All it can do is move a bin's scores relative to the shared threshold.

## Methods

- **Temperature + bias on the logit (the candidate).** Each (aspect, bin) gets \(q = \sigma(\mathrm{logit}(p)/T + b)\).
  This is temperature scaling (Guo et al., ICML 2017) plus a Platt-style bias (Platt 1999), fit separately per
  group. The fit minimises BCE with each row weighted by IA × sqrt(test-like weight), the same row weighting as
  `stacker_tv`'s `ia_bce` loss. A small L2 penalty (1e-3) on (log 1/T, b) keeps the maps for small bins stable.
- **Isotonic regression per (aspect, bin).** Same rows and weights. It is reported as an upper reference only and the
  rule never keeps it.
- **Bins:** `no_hit`, `<30`, `30-50`, `50-80`, `80-95`, `95-99.9`, `100`, on the best DIAMOND identity. Per aspect, a
  bin with fewer than 40 TV v2 proteins that have ground truth in that aspect is merged into its smaller neighbour.
  On TV v2 this merges `no_hit` and `<30` for BP and CC; MF keeps all seven bins.
- **Cross-fitting:** the same five protein folds as `stacker_tv` (`make_folds`, seed 0). They match
  `runs/stacker_tv/knn_only_full/folds.tsv` exactly. With `--stacker-run`, the stacker's own `folds.tsv` is used.
- **Scoring:** `knn_sharp.Scorer`, the chosen_v2 test-like weights from `stacker_tv.test_like_weights`, and the paired
  protein bootstrap `Scorer.bootstrap` (1,000 reps, test-like weights). None of the scoring is reimplemented. Per-bin
  thresholds are read from a curve function that copies the `Scorer.fmax` arithmetic and asserts that its maximum
  equals `Scorer.fmax` to 1e-12.

## Pre-registered rule

The rule was appended to `runs/test_looks.md` at 2026-10-12 08:44:58, before any calibrated score was computed. No
test look was taken. For each input, calibration is kept only if all three conditions hold:

1. The cross-fitted TV v2 test-like mean Fmax beats the uncalibrated input by at least 0.003.
2. The paired bootstrap 95% CI of that difference excludes 0.
3. The plain (unweighted) TV v2 mean Fmax drops by no more than 0.002.

Isotonic is never a candidate. A kept calibration would replace its input in a later pre-registered test look but
would not trigger one on its own.

## Stand-in result: kNN prior T5_k50_g0_na (`knn_standin/`, 58 s on CPU)

| method | test-like TV v2 | Δ vs uncal [95% CI] | plain TV v2 | val (random split, info only) |
|---|---|---|---|---|
| uncalibrated | 0.7168 | | 0.7345 | 0.5948 |
| temperature + bias | 0.7159 | −0.0009 [−0.0028, +0.0013] | 0.7322 (−0.0023) | 0.5910 |
| isotonic (reference) | 0.7232 | +0.0064 [+0.0045, +0.0079] | 0.7339 | not computed |

**Verdict: not kept.** The gain condition and the CI condition both fail, and the plain-mean condition fails
narrowly (−0.0023). The uncalibrated 0.7168 matches the stacker's kNN reference exactly, which confirms the input
path.

Held-out test-like Fmax per fold:

| fold | uncal | T + bias | isotonic |
|---|---|---|---|
| 0 | 0.7132 | 0.7235 | 0.7234 |
| 1 | 0.7053 | 0.7050 | 0.7035 |
| 2 | 0.7029 | 0.7012 | 0.7066 |
| 3 | 0.7495 | 0.7501 | 0.7568 |
| 4 | 0.7303 | 0.7336 | 0.7465 |

Temperature + bias wins on three of the five folds when each fold is scored alone, yet it loses when the folds are
pooled. Pooling puts five slightly different maps under one threshold, the same effect that `runs/stacker_tv/README.md`
describes for mixed-step pooling.

## What the fitted maps and thresholds show

- **The fitted temperatures are large everywhere,** about T ≈ 2–4.5 on all TV with biases from −0.85 to −2.3, and T ≈ 6–9 in the
  merged low-identity groups. The kNN prior is strongly overconfident in the BCE sense, and the maps are stable
  across folds (`maps.tsv`).
- **BCE calibration does not align the Fmax thresholds.** After temperature + bias, the per-bin optimal thresholds
  all drop by about 0.2–0.4. Low-identity bins then want much lower thresholds than the global one. For example, BP
  `<30` wants 0.07 against a global 0.22, and using the global threshold costs that bin 0.28 F where it cost 0.02
  before. This matches the known result that, for calibrated probabilities, the F-optimal threshold is about half
  the attainable F (Lipton, Elkan and Narayanaswamy, ECML-PKDD 2014). Bins with lower attainable F therefore want
  lower probability thresholds. Making the probabilities calibrated per bin pushes the bins' optimal thresholds
  apart rather than together.
- **Isotonic gains for the same structural reason in the other direction.** It brings the worst mid-identity bin in
  line (CC `50-80`: the gap from using the global threshold falls from 0.163 to 0.004) and keeps the high-mass bins
  aligned, so pooled test-like Fmax rises by 0.0064 even though every bin's own Fmax drops slightly.
- **Why a bin's own Fmax moves at all.** A monotone map leaves every bin's achievable precision/recall points
  unchanged. The small per-bin changes come from three places: the scorer's 0.01 threshold grid, rounding to 3
  decimals, and the 0.01 cut. Compressing a bin into a narrow score range (`<30` under T ≈ 8.6 maps [0, 1] into
  roughly [0.25, 0.6]) leaves fewer grid points inside it.

Full tables: `summary.tsv` (method × scope × aspect), `thresholds.tsv` (each bin's own and global optimal
threshold, F at each, and the gap), `folds_scores.tsv`, `maps.tsv`, and `log.txt`.

## Implications for the stacker runs

- The stacker's outputs are already trained under IA-weighted BCE with test-like weights, and model c has its own
  per-bin gate. So the BCE-target temperature map is unlikely to pass the rule there either. It is still the
  registered candidate and should be run as registered.
- The isotonic result suggests that a map aimed at threshold alignment would do better than one aimed at
  probability calibration. Examples are isotonic, or a per-bin shift chosen so that each bin's optimal threshold
  lands on the global one. Neither is covered by the current rule. Using either as a candidate would need its own
  entry in `runs/test_looks.md` before the stacker outputs are scored, and the threshold-shift variant would need
  nested cross-fitting because it picks thresholds on labels.

## Isotonic as the registered candidate (`--method iso`, the default from 08:47)

A second rule was appended to `runs/test_looks.md` at 2026-10-12 08:47:32. No stacker output existed at that time,
and no test look was taken. The candidate is per-(aspect, bin) isotonic regression with the same merged bins, the
same folds, GO max-propagation and the same three keep conditions as the temperature + bias rule. The rule covers
the stacker outputs `l3_c0`, `l3_c1` and `l4a`, and it covers kNN alone as the fallback pipeline. The stand-in
isotonic gain (+0.0064) was seen before this registration, so it does not count as confirmation. A kNN-fallback
verdict under this rule is marked in `result.json` (`knn_fallback_not_confirmation`). The temperature + bias rule
stays as registered (`--method tb`).

Guards added at the same time, before any stacker score:
- Isotonic outputs are clipped to [0.001, 0.999].
- Inputs outside the fitted range are clipped (`out_of_bounds='clip'`), and the map interpolates linearly between
  knots.
- A group with fewer than 40 training proteins in a given fit keeps the identity map. On the kNN stand-in this
  covers:
  - BP and CC `no_hit+<30`: only 16–17 of those proteins have kNN scores.
  - MF `<30` inside the folds: about 32 training proteins. It is fitted only in the final all-TV map (40 proteins).
- A self-check runs on every fold and on the final map, and raises on failure:
  - every map is non-decreasing and stays inside the clip;
  - within each (protein, aspect), no two terms swap order;
  - max-propagation afterwards only raises scores.

**Re-run on the kNN stand-in (`knn_standin_iso/`, 70 s).** This is a pipeline check, not confirmation.

| method | test-like TV v2 | Δ vs uncal [95% CI] | plain TV v2 | val (random split, info only) |
|---|---|---|---|---|
| uncalibrated | 0.7168 | | 0.7345 | 0.5948 |
| temperature + bias | 0.7159 | −0.0009 [−0.0028, +0.0013] | 0.7322 | |
| isotonic (clipped) | 0.7234 | +0.0066 [+0.0047, +0.0082] | 0.7338 (−0.0006) | 0.5951 |

- **Verdict:** the run's verdict reads "KEEP calibration (iso)", with the not-confirmed flag.
- **Self-checks:** all passed, with 0 inversions and 0 entries raised by propagation (the kNN input is already
  ancestor-closed).
- **Ties:** isotonic maps have 14–69 knots per group, so they merge many neighbouring scores. The final map turns
  210k of the 273k ordered adjacent term pairs on TV into ties. The scorer's 0.01 threshold grid is coarser than
  most of these steps anyway.
- **Effect of the clip:** compared with the unclipped stand-in run, the per-bin Fmax for `<30` returns to the
  uncalibrated value (0.5086), because the small groups now keep the identity map.
- **Outputs:** calibrated `pred_val.npz` and `pred_test.npz` were written (test is not scored).

## Commands for the stacker runs (nothing is armed)

Isotonic chain, one per stacker run. Each waits up to 12 h for `runs/stacker_tv/<run>/result.json`, runs
`--method iso`, and appends `AGENT_LOOP_WAKE_stack calib <run> exit <code>` to `logs/stack_chain_<run>.log`:

```bash
cd /home/aj0486@students.ad.unt.edu/PFA/cafa6
for r in l3_c0 l3_c1 l4a; do nohup scripts/chain_calibrate.sh $r > /dev/null 2>&1 & done
```

If the stacker kept no model, `calibrate_tv` exits with code 1 and a message (`chosen = null`). A run with
`--model` is then diagnostic only.

The temperature + bias commands from the first round follow. They now need `--method tb` because the default
changed to `iso`.

`runs/stacker_tv/<run>/` is written by the running stack chain: `l3_c0` and `l3_c1` around 10:00–11:00, then
`l4a`. Run calibration on a run once its `result.json` and `pred_tv_oof_*.npz` exist. Without `--model`, the script
uses `result.json` `chosen` and exits with a message if the stacker kept nothing.

```bash
cd /home/aj0486@students.ad.unt.edu/PFA/cafa6
# the stacker's kept model (decision run under the calibration rule)
PYTHONPATH=src ../bin/python -m cafa6.calibrate_tv --method tb --stacker-run runs/stacker_tv/l3_c0 --name l3_c0 \
  > runs/calibrate_tv/l3_c0.stdout 2>&1
PYTHONPATH=src ../bin/python -m cafa6.calibrate_tv --method tb --stacker-run runs/stacker_tv/l3_c1 --name l3_c1 \
  > runs/calibrate_tv/l3_c1.stdout 2>&1
PYTHONPATH=src ../bin/python -m cafa6.calibrate_tv --method tb --stacker-run runs/stacker_tv/l4a --name l4a \
  > runs/calibrate_tv/l4a.stdout 2>&1
# a specific model (e.g. the gated twin) if the stacker kept nothing: diagnostic only, because the stacker rule
# still decides whether any stacker model goes forward
PYTHONPATH=src ../bin/python -m cafa6.calibrate_tv --method tb --stacker-run runs/stacker_tv/l3_c0 --model c --name l3_c0_c \
  > runs/calibrate_tv/l3_c0_c.stdout 2>&1
```

Each run is CPU only (16 threads at most, CUDA hidden) and takes about 1–2 minutes plus reading the stacker files.
If the stacker's median lofo step was 0, its `pred_{val,test}_{model}.npz` are missing; the script then calibrates
the kNN prior for val/test and logs a warning.
