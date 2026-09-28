# Pilot report: GO-DAG graph conditioning and slot-based pooling

Fast, matched-budget pilots comparing two literature-motivated, opt-in extensions to HiGO-35M
against a fast baseline (no new module), **not** the full 15-epoch/65k-protein `dev_35m_lora` run.
All three runs share the same code, data, seed (0), 20,000-protein train subset, 4,000-protein
val subset, 4 epochs (patience 2, but validation Fw improved every epoch in all three runs, so no
run early-stopped), and the project's fast internal validation metric (`fast_fmax_w` in
`src/cafa6/train.py`: IA-weighted Fmax approximation over the 6,276-term vocab, used for early
stopping — not the official `cafaeval` test-set number reported elsewhere in `REPORT.md`). This
metric is computed identically across all three runs on the same val split, so the comparison is
apples-to-apples, but it is a single seed / single run per arm — see the caveat at the end.

Configs: `configs/pilot_baseline.yaml`, `configs/pilot_gat.yaml`, `configs/pilot_slots.yaml`.
Raw logs: `runs/pilot_baseline_35m_lora/`, `runs/pilot_gat_35m_lora/`, `runs/pilot_slots_35m_lora/`.

## Result (best epoch = epoch 3 / last epoch, for all three runs)

| run | MF (val_Fw_F) | BP (val_Fw_P) | CC (val_Fw_C) | **best_val_Fw_mean** | delta vs baseline | train time |
|---|---|---|---|---|---|---|
| **pilot_baseline** (no new module) | 0.5354 | 0.3389 | 0.5541 | **0.4761** | - | 531 s |
| **pilot_gat** (+ GO-DAG GAT, `use_go_gat=true`) | 0.5044 | 0.3283 | 0.5418 | **0.4582** | **-0.0180 (-3.8%)** | 534 s |
| **pilot_slots** (BioBlobs-style slot pooling, `pooling="slots"`, `n_slots=8`) | 0.5186 | 0.3355 | 0.5455 | **0.4665** | **-0.0096 (-2.0%)** | 423 s |

Full per-epoch trajectories (`val_Fw_mean`):

| epoch | baseline | +GO-GAT | +slots |
|---|---|---|---|
| 0 | 0.4042 | 0.3646 | 0.3924 |
| 1 | 0.4531 | 0.4311 | 0.4453 |
| 2 | 0.4711 | 0.4491 | 0.4616 |
| 3 | 0.4761 | 0.4582 | 0.4665 |

## Reading

**Neither module helped at this matched budget; both moved the metric in the wrong direction**,
consistently across all three aspects (MF/BP/CC) and every epoch, not just the final one:

- **+GO-GAT (Task 1) hurt the most**: -0.0180 mean Fw (-3.8% relative), and it was behind the
  baseline from epoch 0 onward, never catching up. The likely mechanism (verified with a separate
  debug run before committing to the pilot): at random initialization, the GAT's per-layer
  `LayerNorm(d)` renormalizes the term queries, which are otherwise deliberately small
  (`term_emb` is initialized at `std=0.02`, and the fixed ancestor-mean queries have norm ~0.35),
  up to `O(sqrt(d))` norm (~11.8 observed for d=256). This inflates the very first training loss
  by ~20x (epoch-0 train loss 0.49 vs baseline's 0.06) and the model spends much of the 4-epoch
  budget recovering scale rather than learning graph structure — this is very plausibly a
  "cold-start" artifact of the specified architecture (GATConv + residual + LayerNorm + GELU, as
  requested) rather than evidence that GO-graph message passing is a bad idea per se. A longer
  schedule, a warm-up-friendly initialization (e.g. a learned zero-init gate on the GAT branch),
  or a lower initial LR on the GAT's parameters would be the natural next things to try before
  concluding the idea itself doesn't work.
- **+slots (Task 2) hurt less**: -0.0096 mean Fw (-2.0% relative), also behind baseline at every
  epoch but by a smaller, fairly stable margin (-0.012 at epoch 0, -0.010 at epoch 3), and it does
  not show the same catastrophic cold start (epoch-0 train loss 0.060 vs baseline's 0.060 — nearly
  identical, since slot pooling uses the same entmax-1.5 scale as the existing per-term pooling,
  just over a compact 8-slot bottleneck instead of full residue length). This looks more like a
  mild capacity/information bottleneck (8 slots per protein, at this scale and this data
  budget, lose more signal than they save from noise reduction) than an optimization artifact,
  though it is not possible to be fully confident from one run.

## Strength of the evidence

This is **one run per arm, one seed, one short (4-epoch, 20k-protein) budget** — exactly the
"fast pilot for directional signal" scope requested, not a claim of a validated negative result.
Given the ablation table in `REPORT.md` (`### Ablations`), other single-component deltas at this
same kind of short budget are in the +-0.01-0.013 range from run-to-run noise alone (e.g.
"no IA weighting" showed +0.0009 at 4 epochs on validation but +0.015 negative on test; "softmax
instead of entmax" was a dead tie). The slots delta (-0.0096) is within that noise band and should
not be treated as a confirmed negative without a repeat run or a longer budget. The GO-GAT delta
(-0.0180) is larger than that noise band and is at least partly explained by the cold-start
mechanism above, which is itself testable (rerun with the GAT branch zero-gated) rather than
inherent to the graph-conditioning idea.

**Bottom line: at this matched fast budget, plain HiGO (no new module) remains the better choice
of the three; +GO-GAT's regression is most likely an optimization/init artifact worth revisiting
with a gated or longer-schedule variant before drawing a conclusion about the underlying idea;
+slots' smaller regression is closer to the run-to-run noise floor and is a weaker, less certain
negative result.**
