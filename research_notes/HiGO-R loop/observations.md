# HiGO-R loop: observation ledger, keep/drop decisions, open questions

Date: 2026-10-12, about 07:00. Read-only pass over `REPORT.md`, `NOTES.md`, `reports/Novel GO model beyond BioReason Pro.md`
(the plan), `research_notes/Novel GO model beyond BioReason Pro/*.md`, `research_notes/Beating BioReason Pro de novo/code_audit.md`,
`runs/test_looks.md` and the metric files under `runs/` and `logs/`. No code or run was touched. Numbers are copied
from the files named in each entry; where a value was derived here (for example a gate value from fitted weights),
the entry says so.

Evaluation-set names used below:
- **val random (CAFA-6)**: the 8,377-protein validation split of the 30%-identity clustered CAFA-6 split; fast GPU metric
  during training, cafaeval afterwards.
- **test (CAFA-6)**: the 8,058-protein clustered CAFA-6 test split.
- **val random (GO-GPT)**: GO-GPT's random 13,305-protein validation split (`data/proc_gogpt`); "val B" is the 6,653-protein
  cluster half used by the MeanFlow pilot.
- **TV v1**: temporal validation, 1,154 proteins, `data/proc_gogpt_tv` (10 reference proteomes, no zebrafish).
- **TV v2**: temporal validation, 7,566 proteins, `data/proc_gogpt_tv2` (dated stratum 1,238 + duplicate-accession "DA"
  stratum 6,328). "test-like" means TV v2 reweighted to the test's organism x identity-bin x same-organism-hit cells.
- **test (GO-GPT)**: GO-GPT temporal holdout, 8,630 proteins (S3_all); S6 = 8,159 no-knowledge proteins.

All Fmax values are IA-weighted, mean of MF/BP/CC, unless stated otherwise.

---

## 1. Ledger

### A. Reference bars and benchmark facts

**O1. GO-GPT reference numbers on its own test.**
- Tested: released `wanglab/gogpt` checkpoint, paper protocol, cafaeval prop=max norm=cafa, CAFA-5 IA.
- Numbers (test GO-GPT, S3_all): paper 0.651; predictions shipped with the HF test set 0.649; beam 0.6866 (MF 0.7143,
  BP 0.6171, CC 0.7284); greedy 0.6864 (MF 0.7155, BP 0.6156, CC 0.7279). S6: beam 0.7021, greedy 0.7019.
- Evidence: `runs/bench_gogpt/summary.tsv`, `runs/knn_sharp/scorer_check.json`, `NOTES.md` (GO-GPT benchmark section).
- Verdict: **KEEP** as fixed comparison rows; the beam value 0.6866 is reproduced to about 1e-10 by two scorers.

**O2. GO-GPT ten-sample frequency scoring (ladder step L2) is the bar.**
- Tested: public checkpoint, 10 samples per protein and aspect, temperature 0.7, top-k 20 (defaults in
  `benchmarks/gogpt_eval.py`), term frequency as score; inference only.
- Numbers (test GO-GPT): `sample10_freq` 0.6984 (MF 0.7203, BP 0.6296, CC 0.7454); `sample10_freq_prop` 0.6980;
  S6 0.7141. By identity: >=50% (n=7,585) 0.7418 vs beam 0.7304; 30-50% (713) 0.4509 vs 0.4398; <30% (82) 0.4531 vs
  0.3950; no hit (250) 0.3821 vs 0.3742. Generation took 2,361 s for the last 7,606 proteins on one GPU.
- The plan estimated about 0.705 for this bar (inference from the paper's +0.02); the measured gain over beam is
  +0.0118.
- Evidence: `runs/gogpt_sample10/summary.tsv`, `logs/gogpt_sample10.log`.
- Verdict: **KEEP** as the headline bar. With the red-team's 95% half-width of about 0.007, a significant win needs
  roughly 0.708 or more (inference). Not counted as one of our test looks, because no setting of ours was selected
  with it.

**O3. Scorer agreement.**
- `knn_sharp.Scorer` vs cafaeval: GO-GPT beam 0.6865977543763847 vs 0.6865977543866331; HiGO-35M (GO-GPT bench)
  0.45468809489606493 vs 0.4546880945684045.
- Evidence: `runs/knn_sharp/scorer_check.json`.
- Verdict: **KEEP** the in-memory scorer for TV and bootstrap work.

**O4. Composition of the GO-GPT test set.**
- Organisms (test GO-GPT): zebrafish 45.18%, rat 19.41%, human 19.10%, mouse 3.17%, fly 2.67%, fission yeast 1.65%.
- Identity bins (max DIAMOND identity to GO-GPT train, `tv_report.json`): 100% 25.38%, 95-99.9% 28.16%, 80-95% 15.76%,
  50-80% 18.85%, 30-50% 8.04%, <30% 0.92%, no hit 2.90%.
- Best hit in the same organism, by bin: 100% 0.909, 95-99.9% 0.878, 80-95% 0.526, 50-80% 0.134, 30-50% 0.110.
  The `temporal_val.py` docstring states that 94% of the 100% bin are same-organism duplicate accessions (further
  UniProt accessions of genes annotated before T0).
- 760 test proteins are limited-knowledge proteins that also occur in GO-GPT's training set with other aspects;
  self-hits are excluded by accession.
- Evidence: `data/proc_gogpt_tv2/tv_report.json` (`test` block), `data/proc_gogpt_tv/tv_report.json`,
  `src/cafa6/temporal_val.py` docstring, `NOTES.md`.
- Verdict: **KEEP** as a design constraint: the >=95% half of the test is a duplicate-accession copy task, and any
  validation set must contain that mechanism (see O33).

**O5. GO-GPT memorises CAFA-6.**
- 96% of CAFA-6 training proteins (and 96% of our val/test split) are in `wanglab/gogpt-training-data` with labels.
  Zero-shot on 1,500 CAFA-6 clustered-test proteins (97% in GO-GPT training): 0.897 (MF 0.887, BP 0.871, CC 0.933);
  HiGO-35M + gate on the same 1,500: 0.536.
- Evidence: `NOTES.md` (GO-GPT benchmark), `runs/gogpt_cafa6/`.
- Verdict: **KEEP** (GO-GPT cannot be a CAFA-6 reference; comparisons go on its own temporal test).

**O6. Remote homologs: HiGO's lead over GO-GPT shrinks against the ten-sample bar.**
- Test GO-GPT, mean Fmax by identity (HiGO+gate / DIAMOND+naive / GO-GPT beam / GO-GPT sample10):
  30-50% (n=713) 0.4802 / 0.4574 / 0.4398 / 0.4509; <30% (82) 0.4614 / 0.4538 / 0.3950 / 0.4531;
  no hit (250) 0.3594 / 0.2033 / 0.3742 / 0.3821.
- Evidence: `runs/bench_gogpt/summary.tsv`, `runs/gogpt_sample10/summary.tsv`.
- Verdict: **OPEN**. Against sample10, HiGO+gate leads by 0.029 (30-50%) and 0.008 (<30%) and trails by 0.023 with no
  hit. The plan's claim "HiGO-R is better below 50% identity" holds against beam, not clearly against sample10.

### B. HiGO head components (CAFA-6, ESM2-35M, LoRA q,v r8, one seed)

Full model: val random (CAFA-6) 0.5086 at <=4 epochs, 0.5270 best (epoch 8 of 12); test (CAFA-6) 0.5077 raw, 0.5277
with homology gate. Ablation deltas below are against these. The no-taxon and no-IA-weight runs stopped at 4 epochs,
the others ran 7-15, so test deltas mix component effect with training length.

**O7. Taxon (species) embedding.**
- Val random (CAFA-6) @4 epochs 0.4956 (-0.0130). Test (CAFA-6) 0.4760 raw (-0.032), 0.5076 gated (-0.020).
- Evidence: `runs/abl_no_taxon/train_log.jsonl`, `runs/eval_main/`, `NOTES.md` ablation tables.
- Verdict: **KEEP**; largest single component effect on both val and test (test delta is an upper bound because of
  the 4-epoch run).

**O8. GO-conditioned evidence pooling vs mean pooling.**
- Val random @4: 0.5006 (-0.0080); best overall 0.5218 at epoch 12 (-0.0052 vs 0.5270). Test: 0.5047 raw (-0.003),
  0.5274 gated (0.000).
- Evidence: `runs/abl_mean_pool/train_log.jsonl`, `NOTES.md`.
- Verdict: **KEEP**; small accuracy gain that the gate absorbs, but it is the source of the per-term evidence maps.

**O9. Entmax-1.5 vs softmax pooling.**
- Val random @4: 0.5081 (-0.0005); best 0.5224 at epoch 6 of 7. Test: 0.5034 raw (-0.004), 0.5276 gated (0.000).
- Evidence: `runs/abl_softmax_pool/train_log.jsonl`, `NOTES.md`.
- Verdict: **KEEP** for sparse, readable evidence maps (39% of residues non-zero); no accuracy reason either way.

**O10. Hierarchy constraint (MCM + MCLoss).**
- Val random @4: 0.5021 (-0.0065); best 0.5152 at epoch 10 of 11 (-0.012 vs 0.5270). Test: 0.5052 raw (-0.002),
  0.5239 gated (-0.004). With MCM every output has 0 ontology violations.
- Evidence: `runs/abl_no_mcm/train_log.jsonl`, `NOTES.md`, `REPORT.md` §8.7.
- Verdict: **KEEP**; the only head component with a gain that survives the gate on test, and it guarantees closure.

**O11. Ancestor-shared (hierarchical) GO queries vs flat queries.**
- Val random @4: 0.5063 (-0.0023); best 0.5231 at epoch 11 (-0.0039). Test: 0.5068 raw (-0.001), 0.5277 gated (0.000).
- Evidence: `runs/abl_no_hier_query/train_log.jsonl`, `NOTES.md`.
- Verdict: **KEEP** as the default (no cost, small val gain), with the note that no test effect is measurable.

**O12. IA-weighted loss.**
- Val random @4: 0.5095 (+0.0009). Test: 0.4929 raw (-0.015), 0.5191 gated (-0.009). The ablation ran 4 epochs, the
  full model 12, so the test delta is confounded with training length.
- Evidence: `runs/abl_no_ia_weight/train_log.jsonl`, `runs/eval_main/`, `NOTES.md`.
- Verdict: **OPEN**. No clean measurement exists; the L3 configs keep `ia_weight: true`. Settled by L6 (matched epochs,
  2 seeds).

**O13. Split and seed noise.**
- 5-fold CV (val random CAFA-6, 4 epochs per fold): 0.5148 / 0.5181 / 0.5156 / 0.5092 / 0.5101, mean 0.5135 +- 0.0038.
  Six models (dev + 5 folds) on test (CAFA-6): std 0.0063 raw, 0.0048 gated, 0.0043 grounded.
- Evidence: `runs/cv_35m_fold*/train_log.jsonl`, `runs/cv_summary/`, `REPORT.md` §8.3.
- Verdict: **KEEP** as the noise yardstick: single-seed deltas below about 0.005 are not interpretable.

**O14. Evidence maps (interpretability).**
- 511 predictions on 300 test (CAFA-6) proteins: 39% of residues non-zero; top 10% carry 76% of mass; removing them
  from pooling lowers the term probability by 0.067 vs 0.0005 for random residues (p = 0.007); masking the same
  residues in the input has no effect (-0.0003 vs 0.010); median InterPro enrichment 1.13 (57% > 1); residue 1
  receives large weight in all maps.
- Evidence: `runs/dev_35m_lora/explain/`, `NOTES.md` Step 6.
- Verdict: **KEEP** the claim "faithful to the head, not causal at sequence level"; residue-1 fix is **OPEN** (not run).

### C. Adapter and backbone

**O15. Frozen plain-head baseline vs LoRA HiGO (confounded comparison).**
- Val random (CAFA-6): frozen ESM2-35M + mean-pool MLP head 0.4885 (epoch 12) vs LoRA HiGO 0.5270. Test (CAFA-6):
  0.4766 vs 0.5077 raw; 0.5060 vs 0.5277 gated.
- The baseline lacks both the adapter and every HiGO head component, so the +0.04 val / +0.022 gated test gap does not
  measure the adapter.
- Evidence: `runs/base_35m_frozen_mlp/train_log.jsonl`, `NOTES.md` Step 4, `REPORT.md` §8.7.
- Verdict: **OPEN**, superseded by L3 (O19).

**O16. DoRA (q,v r8).**
- Val random (CAFA-6): 0.5189 at epoch 5 vs LoRA 0.5194 at epoch 5; stopped after 6 epochs.
- Evidence: `runs/dev_35m_dora/train_log.jsonl`, `NOTES.md`.
- Verdict: **DROP**; tracks LoRA within 0.001.

**O17. LoRA on all linear layers, r16 (4.7% trainable).**
- Val random (CAFA-6): best 0.5253 (epoch 6) vs 0.5270; val loss rises from epoch 6. Test (CAFA-6) gated 0.5291 vs
  0.5277 (single seed, inside the O13 noise band).
- Evidence: `runs/dev_35m_lora_all_r16/train_log.jsonl`, `NOTES.md`.
- Verdict: **DROP**; 8.5x parameters, no gain.

**O18. ESM2-3B with LoRA (CAFA-6).**
- Val random (CAFA-6) after the first epoch: 0.4848 vs 35M 0.4454 after its first epoch. Stopped partway through
  epoch 2 at the user's request; about 2.4-2.5 GPU-h per epoch.
- Evidence: `runs/final_3b/train_log.jsonl`, `NOTES.md` Step 4.
- Verdict: **OPEN**; the plan replaces it by a frozen 3B truncated at layer 30 with a scalar-mix readout (L5). One
  epoch says nothing about the converged gap.

**O19. L3: frozen ESM2-35M + full HiGO head (C0) vs LoRA (C1), GO-GPT data with post-T0 labels stripped (TV v2) -
running.**
- Config: `data/proc_gogpt_tv2`, 119,362 training proteins, 7,901 terms, `term_bias_init: prior`, `wd_exclude: true`,
  `ia_weight: true`, seed 0, up to 15 epochs, patience 3; writes `pred_tv`.
- Numbers so far (val random GO-GPT, 13,305 proteins, fast metric): C0 epoch 0 0.4931, epoch 1 0.5211; C1 epoch 0
  0.5031 (+0.0100 over C0 at epoch 0). For scale, the earlier GO-GPT-bench HiGO (LoRA, different config, full
  labels) reached 0.4784 at epoch 0 and 0.5447 at epoch 3.
- Earlier attempts on TV v1 data (`runs/l3_c0_frozen_full`, `runs/l3_c1_lora`) have empty `train_log.jsonl`; they
  were superseded by the TV v2 runs.
- Evidence: `logs/l3_c0_tv2.log`, `logs/l3_c1_tv2.log`, `runs/l3_c*_tv2/train_log.jsonl`, `runs/gogptbench_35m_lora/
  train_log.jsonl`.
- Verdict: **OPEN**. The decision rule (C1 - C0 >= 0.01 on TV) cannot be applied until both runs finish and
  `pred_tv` is scored; only one seed per arm is running, the plan asks for two.

**O20. HiGO-35M trained on GO-GPT data (head-to-head reference).**
- 4 epochs inside a 10-epoch cosine schedule (stopped early). Val random (GO-GPT) 0.4784 -> 0.5179 -> 0.5341 -> 0.5447.
  Test (GO-GPT): raw 0.4547, + gate 0.5759, DIAMOND + naive 0.5678. >=50% bin: HiGO+gate 0.5986 vs GO-GPT beam
  0.7304.
- Evidence: `runs/gogptbench_35m_lora/train_log.jsonl`, `runs/bench_gogpt/summary.tsv`.
- Verdict: **KEEP** as a reference row only (under-trained; not a candidate).

### D. Architecture pilots (20k train / 4k val CAFA-6 proteins, 4 epochs, seed 0, val random CAFA-6)

**O21. GO-graph GAT on the term queries.**
- 0.4582 vs baseline 0.4761 (-0.0180); gap per epoch 0.040, 0.022, 0.022, 0.018; epoch-0 train loss 0.49 vs 0.06,
  attributed to `gelu(LayerNorm(conv(x)+x))` replacing the small initial queries.
- Evidence: `runs/pilot_gat_35m_lora/train_log.jsonl`, `runs/pilot_module_report.md`, `REPORT.md` §7.1.
- Verdict: **DROP** in its current form. The proposed gated, zero-initialised residual (`q + g*GAT(q)`) was never
  run and is not on the HiGO-R ladder.

**O22. Slot pooling (8 slots, BioBlobs-style).**
- 0.4665 (-0.0096); behind at every epoch; single seed; removes residue-level evidence maps.
- Evidence: `runs/pilot_slots_35m_lora/train_log.jsonl`, `runs/pilot_module_report.md`.
- Verdict: **DROP**; no gain and it costs the interpretability that motivates the head.

**O23. Ortholog channel (DIAMOND vs 10 model-organism panels).**
- Val / test (CAFA-6): DIAMOND + orthologs + naive 0.4644 / 0.4569 vs DIAMOND + naive 0.4685 / 0.4614 (-0.004 on
  both). An earlier +0.028 claim compared against the unblended DIAMOND row.
- Evidence: `REPORT.md` §7.2, `NOTES.md` Step 9.
- Verdict: **DROP**; the panel is a subset of what DIAMOND already searches.

**O24. Sparse autoencoder on adapted residue embeddings.**
- 371 latents fire on >20% of residues, 99.9% variance explained; 50 analysed latents average 5.52x best-term
  enrichment vs 2.59x null and cover 15 terms (17 protein kinase activity).
- Evidence: `runs/pilot_sae/`, `REPORT.md` §7.3.
- Verdict: **DROP** for the model (no predictive role); a stronger-sparsity rerun is a separate interpretability
  question, not on the ladder.

### E. Post-hoc fusion of HiGO and homology

**O25. Per-aspect homology gate (g = sigmoid(w . [identity, has_hit, 1])).**
- Test (CAFA-6): 0.5077 -> 0.5277. Test (GO-GPT): 0.4547 -> 0.5759.
- Gate value at 100% identity with a hit, computed here from the fitted weights: CAFA-6 (`runs/eval_main/gates.json`,
  dev_35m_lora) about 0.42 MF, 0.29 BP, 0.33 CC; GO-GPT bench (`runs/bench_gogpt/gates.json`) about 0.66 MF, 0.58 BP,
  0.61 CC.
- Verdict: **DROP as the fusion mechanism, KEEP as a reference row**. It is one scalar per protein and aspect, it
  caps kNN trust below 0.66 even at 100% identity, and on test (GO-GPT) it is 0.11 or more below a zero-training
  sharpened kNN (O29).

**O26. Grounded calibrator (HiGO-v2: domains, orthologs, ortholog GOA, ligands).**
- Test (CAFA-6): 0.5408 vs gate 0.5277; without domains 0.5187; without orthologs 0.5413; without ligands 0.5402.
  Test (GO-GPT): grounded 0.5503 vs gate 0.5759; gate-features-only calibrator 0.5661; no orthologs 0.5649.
- Fitted ortholog-GOA weight on GO-GPT: 0.686 (MF), 0.809 (BP), 0.317 (CC). `interpro2go` is a 2026 release and was
  never date-filtered; GOA is a 2026 file date-filtered to 2022-11-30 for GO-GPT.
- Evidence: `runs/grounded/summary.tsv`, `runs/grounded_gogpt/summary.tsv`, `runs/grounded_gogpt/calibrator.json`,
  `REPORT.md` §8.2-8.4.
- Verdict: **DROP** for the GO-GPT headline (reverses sign on the temporal test and is leak-exposed). Domain signal
  survives as an idea only through InterPro signature IDs without `interpro2go` (ladder L7, **OPEN**).

**O27. Grounding stabilises outputs across runs.**
- Predicted-set Jaccard across six models: raw 0.581, gated 0.646, grounded 0.786 (test CAFA-6).
- Evidence: `runs/grounded/stability.tsv`, `REPORT.md` §8.3.
- Verdict: **KEEP** as a CAFA-6 output-property statement; it does not bear on the GO-GPT score.

**O28. The calibrator's kNN feature collapses.**
- `lk = logit(clip(p_knn, 1e-4))`: most kNN scores are 0 and map to the constant -9.21. Fitted `lk` weight: CAFA-6
  0.028 / 0.035 / 0.023 (MF/BP/CC); GO-GPT 0.131 / 0.090 / 0.025. `lk_pid` interaction 0.093-0.179 (CAFA-6).
- Evidence: `runs/grounded/calibrator.json`, `runs/grounded_gogpt/calibrator.json`, `codebase_inventory.md` item 27.
- Verdict: **DROP** the clipped-logit kNN encoding; any stacker uses raw `p_knn` and per-term support features.

### F. Retrieval baseline (ladder L0: sharpened kNN)

**O29. Sharpened kNN ties GO-GPT beam on test, with no training.**
- Red-team grid on test (GO-GPT), weights `bits * exp((pident - max)/T)`, normalised over all hits ("norm all"),
  no coverage term: T=1 k3/10/50 0.694/0.698/0.701; T=2 0.694/0.698/0.700; T=5 0.690/0.699/0.701; T=10
  0.685/0.689/0.691. Val-selected T=10 k=50: 0.691, minus GO-GPT +0.005 [-0.0015, +0.012] (300 reps); S6 0.709 vs
  0.702.
- Per bin (T=2, k=10): 100% 0.898, 95-99.9% 0.851, 80-95% 0.640, 50-80% 0.409, 30-50% 0.427, <30% 0.435, no hit 0;
  GO-GPT beam 0.835 / 0.829 / 0.657 / 0.469 / 0.438 / 0.402 / 0.374. Best-hit copy: 0.679 (binary), 0.684 (scored
  by identity).
- Abstention: the count-weighted bin formula gives 0.684 for kNN vs 0.699 measured, because `norm=cafa` averages
  precision only over proteins with predictions.
- Evidence: `research_notes/Novel GO model beyond BioReason Pro/red_team.md` §0, §3.
- Verdict: **KEEP** identity sharpening; the kNN row is mandatory in every comparison table. Against sample10
  (0.6984) the best grid value (0.701) is +0.003, inside the CI.

**O30. Random validation selects the wrong retrieval settings.**
- Spearman of val random (GO-GPT) vs red-team test over the 8 shared settings: -0.542. Val-fit per-bin kNN/HiGO
  blend: 0.687 on test vs 0.711 with test-fit (oracle, not reportable) weights.
- Evidence: `runs/knn_sharp/chosen.json` (`ranking_agreement`), `red_team.md` §3.2.
- Verdict: **DROP** random val as a selection set for anything homology-related.

**O31. TV v1 failed as a selection set (test look #1).**
- TV v1: 1,154 proteins (387 NK, 767 LK), no zebrafish; identity bins 100% 6.4%, 95-99.9% 16.3%, 50-80% 23.3%
  (test 25.4%, 28.2%, 18.9%); same-organism best hit at 95-99.9% 13.3% (test 87.8%).
- Selection: plain TV mean picked `Tinf_k50_g0_na` (registered 05:59:54, superseded before any look); after adding the
  aspect-norm knob and the test-like reweighting, `Tinf_k50_g1_na` (no sharpening, coverage gamma 1, aspect norm):
  TV v1 0.5185, test-like 0.5550. Spearman vs red-team test -0.542 for TV, TV test-like and val alike.
- Test look #1 (test GO-GPT): 0.5961 (MF 0.6925, BP 0.4488, CC 0.6470); minus GO-GPT beam -0.0905 [-0.0976,
  -0.0826], 1,000 reps; S6 0.6069 vs 0.7021. Per bin: 100% 0.7154, 95-99.9% 0.6737, 80-95% 0.5859, 50-80% 0.4552.
- Evidence: `runs/knn_sharp/chosen.json`, `final.json`, `final_per_bin.tsv`, `final_summary.tsv`,
  `data/proc_gogpt_tv/tv_report.json`, `runs/test_looks.md`.
- Verdict: **DROP** TV v1 for selection; it picked a setting about 0.10 below the sharpened grid on test.

**O32. TV v2 recovers the direction of sharpening but not the fine ranking.**
- TV v2: 7,566 proteins (dated NK 438, dated LK 800, DA 6,328); identity bins 100% 33.9%, 95-99.9% 35.2%, 80-95%
  21.6%, 50-80% 4.0%, 30-50% 3.2%, <30% 0.9%, no hit 1.2%. Rule registered at 06:16:58 before scoring.
- Choice: `T5_k50_g0_na` (T=5, k=50, no coverage term, aspect norm), TV v2 0.7345, test-like 0.7168. Versus the v1
  choice +0.0975 [+0.0917, +0.1031]; versus `T2_k10_g0` +0.0151 [+0.0119, +0.0183] (test-like, 500 reps).
- Sharp minus flat (test-like): T2_k10 - Tinf_k10 +0.0781 [+0.0718, +0.0835]; T2_k50 - Tinf_k50 +0.0958 [+0.0887,
  +0.1026]. T2_k10 - T10_k10: +0.0004 [-0.0023, +0.0035] (test: 0.698 vs 0.689).
- Spearman vs red-team test (8 settings): TV v2 -0.133, test-like -0.084, high-identity bins 0.916, DA 0.253, dated
  -0.542. Test-like weights: 44 cells, effective n 959.9; the merged `*|50-80` cell has 25 TV proteins carrying 13.4%
  of the weight.
- Evidence: `runs/knn_sharp/chosen_v2.json`, `runs/knn_sharp/v2/summary.tsv`, `data/proc_gogpt_tv2/tv_report.json`.
- Verdict: **KEEP** TV v2 as the selection set for retrieval knobs (it gets the large effects right), **OPEN** for
  effects under about 0.01 and for the 50-80% bin, which is under-sampled. `T5_k50_g0_na` has no test score yet.

**O33. Why TV v1 and TV v2 disagree: duplicate accessions vs cross-species orthologs.**
- TV v2 strata, `mean_dated` vs `mean_DA`: `Tinf_k50_g0` 0.489 / 0.669; `T5_k50_g0_na` 0.474 / 0.769; `T2_k10_g0`
  0.420 / 0.746. The dated stratum prefers flat averaging; the DA stratum, like the test's >=95% bins, prefers
  sharpening.
- Evidence: `runs/knn_sharp/v2/summary.tsv`, `temporal_val.py` docstring.
- Verdict: **KEEP** as the explanation; any TV used for selection must include a DA-type stratum.

**O34. Aspect normalisation of kNN weights.**
- Aspect-norm minus all-norm, averaged over matched settings: TV v2 test-like +0.0112 (range +0.0059 to +0.0165);
  TV v1 test-like +0.0409; val random (GO-GPT) +0.0226 (range +0.0074 to +0.0310). The red-team test grid used all-norm
  only, so there is no test number for aspect norm.
- Evidence: `runs/knn_sharp/summary.tsv`, `runs/knn_sharp/v2/summary.tsv` (computed here from the tables).
- Verdict: **KEEP** (consistent on every validation set), with no test confirmation yet.

**O35. Coverage exponent gamma.**
- gamma 1 minus gamma 0, matched settings: TV v2 test-like +0.0002 (range -0.0024 to +0.0105); TV v1 +0.0006; val
  random +0.0005.
- Evidence: same files as O34.
- Verdict: **DROP**; no effect, and the registered rule prefers fewer knobs.

**O36. Number of neighbours k.**
- Test (red-team grid): k=50 >= k=10 >= k=3 at every T. TV v2 test-like: the five best settings all use k=50
  (T5 k50 0.7168 vs T5 k10 0.7129, aspect norm).
- Verdict: **KEEP** k=50.

**O37. Temperature T.**
- Flat (T=inf) loses by about 0.08-0.10 on TV v2 test-like (O32) and is the setting that scored 0.596 on test (O31).
  Among finite T at k=50 aspect norm, TV v2 test-like: T5 0.7168, T10 0.7155, T2 0.7149, spread 0.002. On test
  (all-norm) T=1-5 beat T=10 by about 0.01.
- Verdict: **KEEP** a finite T (registered T=5); the exact value is **OPEN** below 0.01.

**O38. kNN restricted to the model vocabulary loses out-of-vocabulary terms.**
- Test (GO-GPT), T=2 k=10 with vocabulary-only labels (7,901 terms): 0.6843 (MeanFlow pilot); red-team with full
  propagated label sets: 0.698-0.699. The 7,901-term vocabulary covers 93.8% of test IA mass (MF 88.7%).
- Evidence: `runs/meanflow_pilot/notes.md` §2, `runs/meanflow_pilot/summary.tsv`, `red_team.md`, `code_audit.md` §0.
- Verdict: **KEEP** full label sets in the kNN branch; the OOV terms are worth about 0.014 on test.

### G. Post-hoc stacker (ladder L1)

**O39. L1 on TV v1 (no HiGO feature).**
- 5-fold CV on TV v1, features [p_knn, s_best, gap, frac_asp, frac_all, log_n, max_id, has_hit, log_hits, IA,
  depth]: logreg_iaw 0.5117 vs L0 (`Tinf_k50_g1_na`) 0.5185, -0.0068 [-0.0142, -0.00004]; logreg -0.0079; gbm
  -0.0117; gbm+sharp -0.0118; gbm_iaw -0.0130.
- Evidence: `runs/stacker_l1/chosen.json`, `summary.tsv`, `per_bin.tsv`.
- Verdict: **DROP**.

**O40. L1 on TV v2 (no HiGO feature).**
- Best gbm+sharp (adds the sharpened score): TV v2 0.7353 vs L0 (`T5_k50_g0_na`) 0.7345, +0.0008 [-0.0016, +0.0032];
  test-like +0.0057 [+0.0028, +0.0084]. Remote bins vs L0: 30-50% +0.0227, <30% -0.0019. Logistic variants are
  below L0 (-0.0054, -0.0069). Registered keep rule (>= +0.005 on TV mean): not met, `keep: false`.
- HiGO probability was left out because the existing HiGO saw the TV labels, so the stacker in the plan (with a HiGO
  feature) has not been tested.
- Evidence: `runs/stacker_l1_v2/chosen.json`, `summary.tsv`, `per_bin.tsv`.
- Verdict: **DROP** for the homology-only feature set (as registered). The L1 hypothesis with a TV-clean HiGO feature
  stays **OPEN** until L3 writes `pred_tv`.

### H. MeanFlow label refiner and the random-val transfer failure

**O41. MeanFlow one-step label refiner.**
- Val B (val random GO-GPT, n=6,653): best MeanFlow (prior source sigma 0.3, 500 steps, 32 samples) 0.6183 vs
  per-term logistic fusion 0.6169, +0.0014 [-0.0001, +0.0029], bar +0.005; best of 27 MeanFlow settings vs 3 fusion
  settings. Discriminative twin (same network, BCE) 0.6246. One sample (S=1) 0.5921 for the chosen config.
- Test look #2 (test GO-GPT): MeanFlow 0.6544, fusion 0.6388 (+0.0155 [+0.0137, +0.0175]), twin 0.6719, sharpened
  kNN (vocab-only) 0.6843, HiGO 0.4547.
- Evidence: `runs/meanflow_pilot/notes.md`, `summary.tsv`, `result_valB.json`, `result_test.json`, `frozen.json`,
  `runs/test_looks.md`.
- Verdict: **DROP**. The decision was fixed on val B; the BCE twin beats MeanFlow at equal capacity on both splits.

**O42. Models fit on random val lose to zero-training kNN at >=95% identity on test.**
- Test (GO-GPT), >=95% bins: per-term fusion 0.7527, MeanFlow 0.7781, twin 0.8062, sharpened kNN 0.8557. kNN in the
  same bins: val B 0.6997 vs test 0.8557. Red-team: val-fit blend 0.687 vs kNN alone 0.691-0.699.
- Evidence: `runs/meanflow_pilot/summary.tsv`, `red_team.md` §3.2.
- Verdict: **DROP** fitting any stacker, gate or calibrator on a random split. Every learned combiner must be fit on
  TV v2 (or a successor) with HiGO predictions from a model that did not see TV labels.

**O43. A small nonlinear per-term combiner beats a per-term logistic fusion.**
- Twin minus fusion: val B +0.0077 (0.6246 vs 0.6169); test +0.0331 (0.6719 vs 0.6388), still below kNN.
- Evidence: `runs/meanflow_pilot/summary.tsv`.
- Verdict: **OPEN**; supports a nonlinear L1/L4 combiner, untested on TV.

### I. Kaggle CAFA-6

**O44. Competition facts.**
- Closed: final submissions 2026-02-02, final evaluation 2026-06-01. Private top 3: GOAlpha 0.52430, protein_team
  0.48731, yuanllo153 0.44640; 10th 0.42531. Score = IA-weighted Fmax per aspect, averaged within NK, LK and PK, then
  across categories; known PK terms are excluded from scoring. Private board is about 50% of the test data.
- Local data: 82,404 training proteins, all in the 224,309 superset (36.7%); 36,775 training proteins have all three
  aspects (PK candidates). Local GOA has 16,736 experimental pairs not in `train_terms` dated after 2025-06-18 (8,681
  proteins), 5,158 of them after 2026-02-02; those must never enter features or labels.
- Our files: HiGO-35M + gate submission (45,957,813 rows) and grounded submission (84,991,864 rows) were built and not
  uploaded (needs the user's token). Clustered-split 0.528 / 0.541 measures recovery of existing annotations and does
  not map to the leaderboard. No local temporal CAFA-6 replica was found under `runs/`.
- Evidence: `kaggle_competition.md` §1-5, `REPORT.md` §4.4, §8.6, plan "Kaggle CAFA-6" section.
- Verdict: **OPEN** (late-submission acceptance unverified; replica not built).

**O45. Homology structure differs between the two benchmarks.**
- CAFA-6 clustered test: >=50% hit 1,557 of 8,058 (19%); no hit 1,725. Kaggle superset: 88.3% have a DIAMOND hit to
  training. GO-GPT test: 88% >=50%.
- Evidence: `NOTES.md` Step 2 and Step 8, `REPORT.md` §4.2.
- Verdict: **OPEN**: retrieval gains measured on GO-GPT's test say little about the CAFA-6 clustered split.

### J. Process

**O46. Test-look budget.**
- 4 looks total; 2 used: L0 v1 choice (06:02:26, 0.5961) and the MeanFlow pilot (06:41:45). The plan schedules looks
  after P3, after L3, after L5 and at the end. The ten-sample GO-GPT run (O2) is not logged as a look.
- Evidence: `runs/test_looks.md`, plan "experiment ladder" rules.
- Verdict: **KEEP** the budget; two looks remain for three scheduled points (L3, L5, final), see Inconsistencies.

**O47. Leakage exposure of current-release resources.**
- `interpro2go` (2026) never date-filtered; GOA is a 2026 file filtered by date, which is not equivalent to a 2022
  snapshot (removed annotations missing, IEA re-dated); no zebrafish in `goa_reference.parquet`.
- Evidence: `red_team.md` §1 contradictions, §2.8; `REPORT.md` §8.4.
- Verdict: **DROP** current-release mappings from any headline number; report such runs as "possibly leaky".

---

## 2. Dropped ideas (do not retry without a new reason)

| idea | evidence (set) | why dropped |
|---|---|---|
| DoRA adapter (O16) | val random CAFA-6 0.5189 vs LoRA 0.5194 at epoch 5 | no gain at equal rank |
| LoRA on all linear layers, r16 (O17) | val random CAFA-6 0.5253 vs 0.5270; over-fits from epoch 6 | 8.5x parameters, no gain |
| GO-graph GAT on queries, current form (O21) | pilot val -0.0180 | replaces instead of refining queries; gated residual never run and not scheduled |
| Slot pooling (O22) | pilot val -0.0096 | no gain, loses residue evidence maps |
| Ortholog channel (O23) | val and test CAFA-6 -0.004 | subset of the DIAMOND database; re-weights existing neighbours |
| SAE as a model component (O24) | not sparse; family-level latents | no predictive role |
| Per-aspect scalar homology gate as the fusion (O25) | test GO-GPT 0.5759 vs kNN 0.685-0.701 | cannot express per-term trust; caps kNN weight below 0.66 at 100% identity |
| Grounded calibrator for the GO-GPT headline (O26) | test GO-GPT 0.5503 vs gate 0.5759 | sign reversal on temporal test; ortholog-GOA overweight; undated `interpro2go` |
| Clipped-logit kNN feature (O28) | fitted weight 0.023-0.131 | collapses to a constant for most terms |
| Random validation for homology selection or stacker fitting (O30, O42) | Spearman -0.542; fusion 0.753 vs kNN 0.856 at >=95% on test | ranks settings in reverse; fitted models distrust close neighbours |
| TV v1 as a selection set (O31) | test look #1: 0.5961, -0.0905 vs GO-GPT | no duplicate-accession mechanism; picked unsharpened kNN |
| Unsharpened kNN (T = inf) (O31, O37) | test 0.5961; TV v2 test-like -0.08 to -0.10 vs sharpened | dilutes the nearest neighbour |
| Coverage exponent gamma (O35) | +0.0002 to +0.0006 mean on every validation set | no effect |
| L1 stacker with homology-only features (O39, O40) | TV v1 -0.0068; TV v2 +0.0008 [-0.0016, +0.0032] | below the registered +0.005 rule |
| MeanFlow one-step label refiner (O41) | val B +0.0014 [-0.0001, +0.0029] vs fusion; BCE twin 0.6246 vs 0.6183 | below the +0.005 bar; generative sampling costs accuracy at equal capacity |
| Current-release `interpro2go` / GOA in headline numbers (O47) | leak exposure | dated archives or signature IDs only |
| Also excluded by the plan before any run | plan "What HiGO-R leaves out" | LLM reasoning layer, autoregressive decoder, diffusion over label sets, full 3D structure, GO-text label encoder as default, BioReason-Pro `go_embedding.pt`, ensemble with GO-GPT as headline |

---

## 3. Open questions that gate the next steps

1. **Does the adapter matter (L3)?** C1 leads C0 by +0.0100 at epoch 0 on val random (GO-GPT); nothing on TV yet.
   Settle: let both runs finish, score `pred_tv` on TV v2 (overall, test-like, per bin), add a second seed per arm;
   keep LoRA only if C1 - C0 >= 0.01 on TV. This decides whether CRLA (L9) is ever built.
2. **Does the L1 stacker help once HiGO is a feature?** The plan's L1 includes HiGO probability; both L1 runs had to
   omit it (O40). Settle: rerun `stacker_l1` on TV v2 with the L3 winner's `pred_tv` as a feature, same registered
   rule, and report the test-like delta next to the overall delta.
3. **Which criterion governs keep/drop on TV v2: overall mean or test-like mean?** L0 v2 was chosen on test-like;
   L1 v2 was judged on overall (+0.0008) while its test-like delta was +0.0057 [+0.0028, +0.0084]. Settle: register
   one criterion (with the other as a guardrail) before the next TV v2 decision.
4. **Is TV v2 informative in the 50-95% bins, where HiGO-R must win (L4, L5 kill rules)?** TV v2 has 4.0% of proteins
   at 50-80% vs 18.9% on test, and a 25-protein cell carries 13.4% of the test-like weight. Settle: per-bin bootstrap
   CIs on TV v2 for the 80-95% and 50-80% bins; if too wide, enlarge that stratum (more dated proteins from the ZFIN,
   RGD and zebrafish GAFs already in `data/grounding/goa_tv2`, or a DA-type holdout at 50-80% identity) before L4.
5. **Does the registered kNN (`T5_k50_g0_na`, aspect norm) hold on test?** Aspect norm is +0.011 to +0.041 on every
   validation set but has never been scored on test. Settle: include it as the kNN row at the next scheduled look
   (after L3); do not spend a look on it alone.
6. **Is IA weighting in the loss useful?** O12 is confounded. Settle: L6 matched-epoch ablation on TV v2 data, 2 seeds,
   keep only if >= +0.005 with both seeds agreeing.
7. **Can HiGO-R keep a remote-homolog lead against sample10?** Sample10 cuts HiGO+gate's lead to +0.029 (30-50%) and
   +0.008 (<30%) and leads with no hit (O6). Settle: report every learned step's remote bins against sample10 as well
   as beam; the L1 remote-bin rule currently compares to L0 because no TV-clean HiGO+gate exists.
8. **How large must the win be?** Bar is sample10 0.6984; the significance margin is about +0.01 (red-team half-width
   0.007) and the model must also beat the kNN row and the L1 stacker on the same test with a paired bootstrap.
   Settle: fix the P3 protocol rows now (0.651, 0.649, 0.6866, 0.6984, kNN, L1, HiGO+gate) before the next look.
9. **Abstention effect.** kNN abstains on no-hit proteins, which raises measured Fmax (0.684 by bin formula vs 0.699).
   HiGO-R will not abstain. Settle: report kNN with and without a naive fallback on no-hit proteins in the P3 table.
10. **Test-look allocation.** Two looks remain for three scheduled points. Settle: decide (and log in
    `runs/test_looks.md`) whether L5 shares a look with the final model or whether L3 is skipped.
11. **Kaggle late submission and local temporal replica.** Acceptance of late entries is unverified; no CAFA-6
    temporal replica exists. Settle: the user checks Kaggle with a token; build the replica (labels <= 2025-06-18,
    targets 2025-06-18..2026-02-02, CAFA-evaluator-PK with `-known`) before any CAFA-6 claim.
12. **Identical Spearman values.** -0.5422 appears for val random, TV v1, TV v1 test-like and the TV v2 dated stratum
    (O31, O32). With 8 settings a tie is possible, but the code path should be checked once before relying on these
    agreement numbers.

---

## 4. Inconsistencies found

1. **GO-GPT greedy vs beam.** `NOTES.md` (GO-GPT table) and `sota_critique.md` (Q1, Q3) give beam 0.686 and greedy
   0.687. `runs/bench_gogpt/summary.tsv` has beam 0.68660 and greedy 0.68636, so greedy rounds to 0.686, as in
   `runs/prf/tables.md`. `REPORT.md` §4.1 shows the same beam value as 0.686 in the first table and 0.687 in the
   head-to-head table.
2. **Share of the GO-GPT test at 100% identity.** `code_audit.md` §0 says 27%, while its own count (n=1,922) is 22%;
   `red_team.md` measures 25.4% by max identity and 20.7% by best-by-bits hit; `tv_report.json` gives 25.38%.
3. **GO-GPT-bench HiGO validation curve.** `code_audit.md` §0 says val Fw 0.4041 -> 0.5447;
   `runs/gogptbench_35m_lora/train_log.jsonl` starts at 0.4784 (0.4042 is the CAFA-6 pilot baseline's epoch 0).
   `red_team.md` claim 5 gives 0.478 -> 0.545, matching the log.
4. **Calibrator kNN weight.** `REPORT.md` §8.2 and `NOTES.md` Step 10: 0.02-0.04; `code_audit.md` §0 and the plan:
   0.02-0.13; `codebase_inventory.md` §7: about 0.03. The calibrator files give 0.023-0.035 for CAFA-6 and
   0.025-0.131 for GO-GPT, so the ranges refer to different benchmarks without saying so.
5. **Gate ceiling.** `code_audit.md` and the plan: g <= about 0.6; `codebase_inventory.md` §7: g <= 0.4; `NOTES.md`:
   about 0.3-0.4 at 100% identity. From the fitted weights, CAFA-6 gives about 0.29-0.42 and GO-GPT about 0.58-0.66;
   again two benchmarks quoted without labels.
6. **IA weighting.** `REPORT.md` §4.2 calls it a negative result, the next paragraph says it matters most on test,
   and §5 lists it under "what matters". The plan summarises it as "+0.001 val, -0.009 test, under a 4-epoch
   confound".
7. **CAFA-6 vocabulary IA coverage.** `REPORT.md` §2 and `NOTES.md` Step 2: 90.1%; `code_audit.md` §0: 89.9%.
8. **GO graph size.** `NOTES.md` Step 2: 10,609 parent edges, 56,371 ancestor pairs; `prepare_report.json` (per
   `codebase_inventory.md` item 42): 10,588 / 56,432; `NOTES.md` Step 9 uses 10,588 edges.
9. **Sharpened kNN T=2, k=10 on test.** `red_team.md` §3.2 table: 0.699; the grid two lines below: 0.698;
   `runs/meanflow_pilot/notes.md` cites "0.698". The plan's table uses 0.699.
10. **Identity-bin counts on the GO-GPT test.** `runs/bench_gogpt/summary.tsv`: 7,585 / 713 / 82 / 250;
    `red_team.md` (from `identity.parquet`): 7,607 / 694 / 79 / 250. Noted by the red team as a different identity
    source.
11. **Same-organism share of the 100% bin.** `temporal_val.py` docstring: 94% are same-organism duplicate accessions;
    `tv_report.json` (test block): same-organism best hit in the 100% bin 0.909. The definitions may differ (any hit
    vs best hit); not resolved here.
12. **Ten-sample bar estimate.** Plan and `red_team.md`: about 0.705 (inference). Measured: 0.6984.
13. **Test-look schedule.** Plan: looks after P3, L3, L5 and final. Actual: one look for L0 v1 and one for the MeanFlow
    pilot, which is not a ladder step; two remain for three scheduled points.
14. **L0 v1 selection used test-derived information indirectly.** `runs/test_looks.md` records that the first
    registration was superseded because "TV disagreed with the red-team test ranking", after which the grid gained a
    knob and the selection criterion changed. No new test score was read, but the red-team test grid informed the
    change. Not a number conflict; recorded so later readers know the v1 choice was not blind to test rankings.
15. **L1 keep-rule criterion vs L0 selection criterion.** L0 v2 was selected on the TV v2 test-like mean; L1 v2 was
    judged on the plain TV v2 mean (`runs/stacker_l1_v2/chosen.json`). Under the test-like mean it would pass
    (+0.0057).
16. **Remote-bin claim.** The plan says "HiGO-R is better below 50% identity, which current numbers already support";
    that holds against GO-GPT beam (0.478 vs 0.438; 0.466 vs 0.402 in the plan's bins), but against sample10 the
    <30% margin is +0.008 and the no-hit bin is lost (O6).
