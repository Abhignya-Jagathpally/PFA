# fresh24: registration and look log

Benchmark: `data/fresh24/` (built by `src/cafa6/build_fresh24.py`, statistics in `runs/fresh24_stats.md`). Experimental GO
annotations added after 2024-02-29, absent from GOA release 219 (2024-02-09), for proteins disjoint from GO-GPT
train/val/test and TV v2. Every entry below is written before the scores it describes exist.

- 2026-10-19 18:40:00 | REGISTRATION before any score on fresh24 (no look yet).
  - **Primary set (FR24-P):** no-knowledge proteins (`gt_nk.tsv`) with `hpa_only == False` in `proteins.tsv`
    (11,231 proteins), all three aspects. The Human Protein Atlas bulk load (11,469 human CCO-only proteins, mostly
    isoforms or fragments of proteins in the GO-GPT training data) is excluded from the primary set because it would
    dominate the CCO score with near-duplicates of training proteins.
  - **Secondary sets:** all no-knowledge proteins (with the HPA load); limited-knowledge proteins (`gt_lk.tsv`);
    FR24-P without the *T. brucei* HTP localisation load and the SynGO load (sensitivity); FR24-P scored against the
    current GO release (`gt_nk_go_current.tsv`); every set broken down by identity bin of the nearest GO-GPT train+val
    protein and by organism group.
  - **Scoring:** IA-weighted Fmax (headline), Fmax, IA-based S-min, per aspect and mean, with GO
    `runs/gogpt_cafa5/ref/go-basic_gogpt.obo`, IA `runs/gogpt_cafa5/ref/IA.txt`, cafaeval protocol (prop=max,
    norm=cafa) via `cafa6.knn_sharp.Scorer`; per-protein F1/F1_w by identity bin as in `eval_suite.py`.
  - **Methods, all frozen, nothing tuned on fresh24:** Ontologue exactly as served (`cafa6.predict_r2`, GPU bf16,
    train+val references of 2022-11, saved twin stacker); its parts (step 1a kNN prior alone, step 1b HiGO head
    alone); BLAST-KNN (training references; train+val references); DIAMONDScore; naive; GO-GPT released checkpoint,
    ten-sample and beam, with the settings used on the GO-GPT test; SPROF-GO; DeepGOPlus 1.0.10; ProteInfer 5-model
    ensemble; DeepFRI sequence CNN; DeepGO-SE (MF and BP only; its released CC models are faulty). BioReason-Pro RL
    if its inputs can be built for these proteins within the time budget. InterLabelGO+ is not scored (its released
    training labels come from a snapshot after the cutoff).
  - **Statistics:** paired protein bootstrap (1,000 reps) for every interval; paired permutation tests (10,000 reps)
    of Ontologue minus each method on {IA-weighted Fmax, Fmax, S-min} x {MF, BP, CC, mean}, Holm-Bonferroni within that
    family on FR24-P; secondary sets are descriptive.
  - **Look budget for fresh24:** look #1 is this single scoring of all frozen methods at once (descriptive; it
    selects nothing). At most three further looks, each registered here beforehand, for new models (for example the
    forecasting-trained model); model selection never uses fresh24.
- 2026-10-24 22:45:00 | REGISTRATION of look #2, before any forecasting-model prediction on fresh24 exists and before look #1 is closed.
  - **Model:** F1_ens, the logit mean of the three forecasting-model seeds runs/forecast/F1_s{0,1,2} (kept by the
    2026-10-19 19:15:00 rule in runs/test_looks.md: A-valid no-knowledge F1 0.4907 vs kNN 0.4668 vs F0 0.4196, all
    three seeds positive; the A-valid set is small, MF/BP/CC 32/55/35 rows, so the seed-ensemble interval against kNN,
    [-0.017, +0.088], includes zero). Inputs on fresh24 exactly as at training: no input terms (no-knowledge rows),
    kNN prior from the served train+val references (labels of 2022-11), same vocabulary pass-through.
  - **Comparisons on FR24-P** (same scorer, sets and statistics as look #1): F1_ens minus Ontologue (served), minus
    Ontologue step 1a (kNN prior), minus GO-GPT ten-sample, minus SPROF-GO; Holm over {IA-weighted Fmax, Fmax, S-min} x
    {MF, BP, CC, mean} x these 4 comparisons. Secondary: identity bins, organism groups, the secondary sets of look #1.
  - **Pre-stated expectation:** F1_ens - Ontologue >= 0 on FR24-P IA-weighted Fmax mean, with the gain concentrated
    below 50% identity. Whatever the outcome, no further model is selected on fresh24.
- 2026-10-25 00:07:42 | LOOK #1 | Ontologue (served), Ontologue step 1a kNN prior, Ontologue step 1b HiGO head, BLAST-KNN (training refs), BLAST-KNN (train+val refs), DIAMONDScore, naive, GO-GPT ten-sample, GO-GPT beam, SPROF-GO, DeepGOPlus 1.0.10, ProteInfer 5-model ensemble, DeepFRI sequence CNN, DeepGO-SE (MF, BP only); BioReason-Pro RL not run (no leakage-free inputs) | FR24-P IA-weighted Fmax mean: Ontologue 0.479, SPROF-GO 0.515, GO-GPT ten-sample 0.468, DeepGOPlus 0.466, GO-GPT beam 0.459, step 1a kNN 0.458, DIAMONDScore 0.456, ProteInfer 0.453, step 1b HiGO 0.432, BLAST-KNN train+val 0.375, BLAST-KNN 0.372, DeepFRI 0.270, naive 0.230, DeepGO-SE n/a (MF+BP mean 0.412) | runs/fresh24/
- 2026-10-25 00:12:42 | LOOK #2 | forecasting model F1_ens vs Ontologue, kNN prior, GO-GPT ten-sample, SPROF-GO (Holm over 48 tests, 35 significant) | FR24-P IA-weighted Fmax mean: F1_ens 0.487, Ontologue 0.479 (F1-Ont +0.008 [+0.004, +0.012], p_Holm 0.008), kNN 0.458, GO-GPT ten-sample 0.468, SPROF-GO 0.515 (F1-SPROF -0.027 [-0.032, -0.022]); gain over Ontologue below 95% identity (no hit +0.022, <30 +0.024, 30-50 +0.013, 50-80 +0.017), loss at >=95% (-0.024, -0.066) | runs/fresh24_look2/
- 2026-10-25 01:16:00 | REGISTRATION of look #3 (before any round-2 candidate exists): the candidate chosen by the 2026-10-25 01:15:00 rule in runs/test_looks.md is scored on FR24-P with the look-#1 scorer, sets and statistics; comparisons: candidate minus Ontologue (served), minus F1-35M ensemble (look #2), minus SPROF-GO, minus GO-GPT ten-sample; Holm over {IA-weighted Fmax, Fmax, S-min} x {MF, BP, CC, mean} x these 4. After look #3 one fresh24 look remains, kept for a final confirmatory number.
