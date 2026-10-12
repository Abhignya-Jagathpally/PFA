# HiGO-R stacker fit on TV v2 (`src/cafa6/stacker_tv.py`)

This stacker combines HiGO with the pre-registered `T5_k50_g0_na` kNN prior. It is fit and selected only on
TV v2, with grouped 5-fold cross-fitting and test-like weights. It exists because every combiner fit on the random
val split lost to raw kNN on test at ≥95% identity (`runs/meanflow_pilot/notes.md`). The full design is in the module
docstring.

## Launching on L3 output

**Preconditions.** Each L3 run writes these files when it finishes (its config has `pred_id_sets: {tv_all: tv_ids.txt}`
and `pred_topk: 500`):

```
runs/l3_c1_tv2/pred_tv_all_top500.npz   # all 7,566 TV v2 proteins, in tv_ids.txt order (required)
runs/l3_c1_tv2/pred_val_top500.npz      # needed for the refit stage
runs/l3_c1_tv2/pred_test_top500.npz     # written by the refit stage, never scored
```

`pred_tv_top500.npz` (the 6,766-protein `tv` split) is not used.

**Launch.** The rule allows both runs as candidates. Each uses under 1 GB of GPU memory, so they can run side by side:

```bash
cd /home/aj0486@students.ad.unt.edu/PFA/cafa6
ls runs/l3_c1_tv2/pred_{tv_all,val,test}_top500.npz && \
PYTHONPATH=src nohup ../bin/python -m cafa6.stacker_tv --higo-run runs/l3_c1_tv2 --out runs/stacker_tv/l3_c1 \
  --threads 16 > runs/stacker_tv/l3_c1.stdout 2>&1 &
ls runs/l3_c0_tv2/pred_{tv_all,val,test}_top500.npz && \
PYTHONPATH=src nohup ../bin/python -m cafa6.stacker_tv --higo-run runs/l3_c0_tv2 --out runs/stacker_tv/l3_c0 \
  --threads 16 > runs/stacker_tv/l3_c0.stdout 2>&1 &
```

- **Defaults:** models a, b and c; 5 folds; up to 3,000 steps per fold; evaluation every 100 steps; patience 6;
  256 proteins per step; IA-weighted BCE; HiGO top-300; 1,000 bootstrap reps.
- **Device:** the script picks a GPU with more than 10 GB free, checked through nvidia-smi so it never opens a context
  on a busy GPU. Otherwise it falls back to CPU. Use `--device cpu` to force CPU.
- **Expected time:** about 15–25 minutes per run, plus about 3 minutes for the refit. The HiGO-less full run took 11
  minutes on a shared machine.

**Reading the result.**
- `result.json` holds the `verdict` field and per-model `delta` and `ci` values for the lofo estimate.
- `summary.tsv` shows the lofo and es estimates side by side.
- `per_bin.tsv` gives Fmax per identity bin.
- `curves.tsv` has the per-fold loss and Fmax curves.

## Pre-registered selection rule

The rule below is also in the code (`SELECTION_RULE`) and in `runs/test_looks.md` (entry dated 2026-10-12 07:05:39).
It was registered before any stacker score was computed, and no test look has been taken.

- **Candidates:** models a, b and c for each L3 HiGO run with TV v2 predictions (`l3_c0_tv2`, `l3_c1_tv2`).
- **Estimate:** each candidate's cross-fitted, pooled out-of-fold TV v2 test-like mean Fmax. Each fold's step count
  is chosen on the other four folds' held-out curves (lofo).
- **Keep condition:** keep a candidate only if both hold:
  - its estimate exceeds raw `T5_k50_g0_na` kNN by at least 0.005;
  - the paired protein bootstrap 95% CI of that difference (1,000 reps, test-like weights) excludes 0.
- **Choice:** the kept candidate with the highest estimate gets a single test look. If none is kept, the raw kNN
  prior stays.
- **Not selection runs:** smoke runs, val-split runs and HiGO-less runs.

**Why lofo and not es.** The es estimate scores each fold at its own best step, as requested. That makes five step
choices on the same data being scored, so it is reported but optimistic. The lofo estimate never lets a fold's
labels pick its own step.

**How pooling behaves.** Folds sitting at different steps are calibrated differently, and pooled Fmax uses a single
threshold across all of them. Mixed-step pooling is therefore penalised, which makes it a conservative estimate. For
example, the twin on HiGO-less TV scored 0.7128 under es even though every fold was at least as good as its prior.

## Smoke results (2026-10-12)

All numbers are IA-weighted Fmax (mean of MF, BP and CC), computed with `knn_sharp.Scorer`. "Test-like" means
weighted to the test cell mix by the chosen_v2 rule. Every model starts at the kNN prior: step 0 reproduces the
kNN's 0.7168 on TV v2 exactly.

**1. TV v2, HiGO features zeroed** (`smoke_knn_only`: 300 steps; `knn_only_full`: full defaults).

| run | model | lofo test-like | Δ vs kNN [95% CI] | es test-like |
|---|---|---|---|---|
| kNN T5_k50_g0_na | | 0.7168 | | |
| smoke (300 steps) | fusion | 0.7169 | +0.0001 [−0.0000, +0.0002] | 0.7169 |
| smoke | twin | 0.7083 | −0.0085 [−0.0108, −0.0065] | 0.7128 |
| smoke | gated twin | 0.7079 | −0.0089 [−0.0112, −0.0069] | 0.7126 |
| full (3,000 max) | fusion / twin / gated twin | 0.7169 (lofo picks step 0 = kNN) | +0.0001 | 0.7169 / 0.7154 / 0.7173 |

Without HiGO, no learned refiner beats the kNN prior on TV v2; held-out BCE improves while Fmax does not. Any stacker
gain therefore has to come from the HiGO features.

**2. Stand-in HiGO on the random val split** (`runs/gogptbench_35m_lora` predictions on val, kNN `val_knn.npz`,
test-like weights from the val cells). This is a mechanical and design check only, never selection. On val, raw kNN
scores 0.7102 test-like (0.5948 unweighted) and HiGO alone scores 0.5037 (0.5308).

| run | model | lofo test-like | Δ vs kNN [95% CI] | unweighted |
|---|---|---|---|---|
| smoke (300 steps) | fusion | 0.7127 | +0.0024 [+0.0015, +0.0031] | 0.5981 |
| smoke | twin | 0.7099 | −0.0003 [−0.0012, +0.0004] | 0.5950 |
| smoke | gated twin | 0.7152 | +0.0050 [+0.0023, +0.0075] | 0.6330 |
| design, 1,500 steps, IA-weighted BCE | twin | 0.7220 | +0.0117 [+0.0085, +0.0145] | 0.6363 |
| design, 1,500 steps, plain BCE | twin | 0.7222 | +0.0120 [+0.0091, +0.0145] | 0.6340 |

The twin needs about 500–600 steps to pass the prior. The gated twin gets there faster because its per-bin bias lets
HiGO score no-hit proteins at once (no-hit bin: 0.48 against 0 for kNN). IA-weighted BCE tied plain BCE on val and
was kept as the default because the metric weights terms by IA. This loss choice was made on val, not on TV.

**3. Stage-4 mechanical test** (output in `/tmp/stacker_tv_stage4_test`, using a fake HiGO run in `/tmp/fake_higo_tv2`
built from kNN values). It wrote the TV out-of-fold files, refit on all TV, wrote val and test predictions, and
scored val only. Both `/tmp` directories can be removed whenever convenient.

**Sizes.**
- TV v2 with kNN only: 7,566 proteins and 3.03M candidate rows (400 per protein, at most 1,785). 2.02M rows are
  scored, with a 12.2% positive rate, and the candidates cover 87.4% of ground-truth pairs. The effective n of the
  test-like weights is 960.
- Val with HiGO top-300: 6.7M rows (505 per protein), 589 MB of features. Test: 5.5M rows.
- Peak GPU memory is under 1 GB. A TV fold takes 10–60 s, and one evaluation (MCM plus scoring) takes about 1.5 s.

## Files

- `smoke_knn_only/`, `knn_only_full/`: TV v2 runs with HiGO features zeroed.
- `smoke_val_standin/`: smoke of all three models with the stand-in HiGO on val.
- `design_val_bce/`, `design_val_iabce/`: twin loss comparison on val.
- `*.stdout`: console logs (also in each `log.txt`).
