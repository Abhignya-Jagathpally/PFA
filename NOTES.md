# CAFA-6 working notes

Running log of how each step was handled. Newest results are appended per step.
Python: `../bin/python` (venv at `PFA/`). Code: `src/cafa6/`. Outputs: `data/proc/`, `runs/`, `logs/`.

## Step 1 - Environment

- Hardware: 2x NVIDIA H100 NVL (96 GB), driver 580.173.02, CUDA 13.0, 128 CPUs, 251 GB RAM.
- Existing venv reused instead of Conda/Docker (Conda env would not fit the disk budget). Versions pinned in `requirements.txt` (torch 2.13.0+cu130, transformers 5.17, peft 0.21, accelerate 1.15, datasets 4.3, biopython 1.88).
- Added: `cafaeval` 1.3.0 (official CAFA evaluator, IA-weighted Fmax), `entmax` 1.3 (sparse attention).
- Bioinformatics binaries, static, in `tools/`: MMseqs2 (release 8cc5ce3, AVX2) and DIAMOND 2.1.10. CD-HIT was not installed: MMseqs2 `easy-cluster` covers redundancy removal at <40% identity, which CD-HIT cannot do for proteins below 40%.
- Smoke test: ESM2-8M forward pass in BF16 on GPU OK; LoRA mount points confirmed as `encoder.layer.N.attention.self.{query,value}`.
- Disk: only 5.2 GB free on `/`. User approved deleting `~/.cache/kagglehub` only (now 17 GB free). Consequences: no residue-level embedding caches on disk; per-protein embeddings are stored as fp16; backbones limited to cached ESM2-8M/35M/3B (650M not downloaded).
- Loop: `tools/watch_done.sh` emits `AGENT_LOOP_WAKE_cafa6` whenever a job writes `logs/<name>.done`, which wakes the agent to record results.

## Step 2 - Data preprocessing and dataset construction (`python -m cafa6.prepare`, report in `data/proc/prepare_report.json`)

- Input: `Train/train_sequences.fasta` (82,404), `train_terms.tsv` (537,027 leaf annotations, 26,125 terms), `go-basic.obo` (2025-06-01, 40,122 terms), `IA.tsv`, `Test/testsuperset.fasta` (224,309, `>ID taxon`).
- Cleaning (train only; every test protein must still be predicted): dropped 36 sequences with >5% ambiguous residues (X/B/Z/U/O/J) and 330 shorter than 20 aa -> 82,038.
- The provided labels are leaf-only (98.6% of implied ancestors missing), so annotations were propagated over `is_a` + `part_of` (alt_ids remapped): 536,289 -> 3,560,200 (the same rule cafaeval applies).
- Redundancy: MMseqs2 `easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0` -> 33,003 clusters. Split by cluster 80/10/10 (seed 0): train 65,603 / val 8,377 / test 8,058 proteins; verified that no cluster crosses splits.
- Residual homology (DIAMOND, e<=1e-3, test vs train): >=50% id 1,557; 30-50% 3,868; <30% 908; no hit 1,725. Local hits below 80% coverage survive clustering, so results are also reported per identity bin (`data/proc/identity.parquet`).
- Length: train p50/p95/p99/max = 411/1320/2376/35,213; test p95 = 1,095. ESM2's context is 1,022 residues, below p95, so: training uses a random 1,022-residue crop (augmentation); inference uses sliding windows (stride 511) with max-pooled logits. Domain-boundary cropping (Pfam/InterPro) was not used because InterPro annotations are not available for the 224k test proteins without a rate-limited web API; windowing covers the whole sequence instead of truncating it.
- Tokenizer: ESM2 tokenizer; `<cls>`, `<eos>` and padding are excluded from the attention via `res_mask` and padded tokens are masked in the backbone via `attention_mask`.
- Label space: terms with >=30 propagated annotations among train-split proteins, roots excluded: 6,276 terms (BP 4,666, MF 903, CC 707), covering 90.1% of the IA mass. Hierarchy inside the vocab: 10,609 parent edges, 56,371 ancestor pairs.
- Loaders: length-bucketed batches with a 16,384-token budget (max 32 proteins), shuffled per epoch (`cafa6.train.Batcher`).
- Ground truth for cafaeval: `data/proc/gt_{val,test}.tsv` (leaf terms; cafaeval propagates).

## Baselines (`python -m cafa6.baselines`, `runs/baselines/`)

Metric throughout: cafaeval 1.3.0, `prop=max`, `norm=cafa`, CAFA-6 `IA.tsv`; "Kaggle" = mean of IA-weighted Fmax over MF/BP/CC.

| method (clustered test split) | MF | BP | CC | Kaggle mean |
|---|---|---|---|---|
| naive (train term frequency) | 0.3631 | 0.2245 | 0.3617 | 0.3164 |
| DIAMOND kNN (DiamondScore) | 0.5233 | 0.3141 | 0.4500 | 0.4291 |
| DIAMOND + naive (0.8/0.2, naive when no hit) | 0.5751 | 0.3281 | 0.4809 | 0.4614 |

Validation split: 0.3059 / 0.4365 / 0.4685 respectively.

## Step 3 - Pre-trained model and parameter-efficient adapter (`src/cafa6/model.py`)

- Backbone: ESM2 via `transformers.AutoModel` (no pooler), all backbone weights frozen (`requires_grad=False`), then a PEFT adapter is mounted.
- Adapter candidates, compared on the same data/seed (dev runs, ESM2-35M):
  - `lora`: r=8, alpha=16 (=2r), dropout 0.05, targets `query`,`value` -> 184,320 trainable = 0.55% of the backbone.
  - `dora` (weight-decomposed LoRA, Liu et al. 2024, ICML): same targets/rank; adds a learned magnitude per output column -> 195,840 = 0.59%.
  - `lora_all_r16`: r=16 on `query,key,value` and every `dense` (attention output + FFN) -> 1,658,880 = 4.7% (capacity check, deliberately outside the 0.1-1% guideline).
  - `frozen`: no adapter (used by the frozen-ESM2 baseline).
  Motivation for testing beyond LoRA: DoRA separates magnitude and direction updates and consistently matched or beat LoRA at equal rank in its paper; PLM-specific studies (Schmirler et al. 2024, Nat Commun) found LoRA on ESM2 competitive with full fine-tuning, so the adapter itself is not expected to be the main lever; the hierarchy-aware head is.
- Task head ("HiGO"):
  - Residue projection `Linear(h,256)+LayerNorm`.
  - Hierarchical GO tokenization: each of the 6,276 terms has an embedding e_t; its query is q_t = e_t + mean_{a in ancestors(t)} e_a (sparse row-normalised ancestor operator). Unlike GO-GPT's flat `GO:xxxxxxx -> token id` vocabulary, related terms share parameters, so a rare deep term inherits from its well-trained ancestors.
  - GO-conditioned evidence pooling: per-term attention over residues with entmax-1.5 (Peters et al. 2019), which gives exact zeros; alpha_t is the term's evidence map and v_t = sum_l alpha_tl h_l. Logit z_t = <GELU(W v_t + taxon_emb), q_t> + b_t.
  - Taxon embedding for the 41 species with >=50 training proteins (index 0 = other; 10% taxon dropout during training so unseen species are handled).
  - Hierarchy consistency: C-HMCNN max-constraint module (Giunchiglia & Lukasiewicz, NeurIPS 2020): p_hat(t) = max over t and its descendants, which enforces p_hat(parent) >= p_hat(child) by construction; trained with MCLoss.
  - Loss weighting by information accretion: w_t = 1 + min(IA_t, 5), normalised to mean 1, which aligns training with the IA-weighted Fmax used by Kaggle.

## Step 4 - Training configuration (`configs/dev.yaml`, `src/cafa6/train.py`)

- AdamW (weight decay 0.01); adapter LR 3e-4 (LoRA range 1e-4..5e-4); head LR 1e-3; linear warm-up over 6% of steps then cosine decay; gradient clipping 1.0.
- Batch: token budget 16,384 (~30 proteins, max 32), BF16 autocast on H100 (Hopper supports BF16).
- Up to 15 epochs, early stopping with patience 3 on validation IA-weighted Fmax (fast GPU re-implementation over the 6,276-term space; final numbers always from cafaeval). Train loss, val loss and val Fmax per aspect are logged per epoch to `runs/<name>/train_log.jsonl`.
- Only trainable weights (adapter + head) are saved: `runs/<name>/trainable_weights.pt` (35M LoRA: ~8 MB).
- Smoke test (ESM2-8M, 3k proteins, 2 epochs): loss 0.103 -> 0.027, val Fw 0.269 -> 0.277.

### Adapter decision (dev runs, ESM2-35M, best validation IA-weighted Fmax mean, fast GPU metric)

| run | trainable (backbone share) | best val Fw | best epoch / epochs run |
|---|---|---|---|
| `dev_35m_lora` (q,v r8) | 0.55% | **0.5270** | 8 / 12 (early stop) |
| `dev_35m_lora_all_r16` (all linear r16) | 4.7% | 0.5253 | 6 / 10 (early stop; val loss rises from epoch 6: over-fits) |
| `dev_35m_dora` (q,v r8) | 0.59% | 0.5189 at ep 5 (LoRA at ep 5: 0.5194) | stopped after 6 epochs, tracking LoRA to within 0.001 |
| `base_35m_frozen_mlp` (no adapter, frozen ESM2 + HiGO head) | 0% | 0.4885 | 12 / 15 |

Decision: plain LoRA on `query,value`, r=8. The user asked to go beyond LoRA only if it helps: DoRA and 8.5x more LoRA capacity did not help at matched budget, while *any* adapter vs none is worth +0.04. The adapter is a second-order factor; the gains come from the head (see ablations). LoRA is kept for the final 3B model (`configs/final_3b.yaml`: 2.95M trainable = 0.10% of 2.84B).
- 3B run: started (0.10% trainable, 36 GB with gradient checkpointing, ~3 s/step on a shared GPU, ~2.5 h/epoch; val Fw 0.485 after epoch 1 vs 35M's 0.445 after epoch 1) and then stopped at the user's request partway into epoch 2. The final model is therefore HiGO-35M (`dev_35m_lora`); `configs/final_3b.yaml` remains for a future run.

## GO-GPT benchmark on its own test set (`benchmarks/gogpt_eval.py`, `runs/gogpt_cafa5/SUMMARY.md`)

- Test set: BioReason-Pro temporal holdout (8,630 proteins with new experimental annotations Mar 2023-Feb 2024 in an aspect with no prior annotation; GO 2023-01-01; CAFA-5 `IA.txt`); GT statistics match the paper exactly.
- Model: released `wanglab/gogpt` checkpoint (ESM2-3B layer 30 + GPT decoder, organism token), the released predictor's settings (beam 5, length penalty 0.3, <=300 tokens per aspect, fp32 ESM). Batched and with an exact protein-stream cache (verified identical outputs vs the unmodified predictor on 36/36 aspect lists). Full test set, ~1.9 h on one H100 (3 processes).
- Scoring: authors' protocol (binary predictions, score 1; CAFA-evaluator prop=max, norm=cafa). Independently re-scored with `cafa6.bench_gogpt` -> identical to 4 decimals.

| GO-GPT, all 8,630 (Table S3 protocol) | MF Fmax / Fw | BP Fmax / Fw | CC Fmax / Fw | mean Fw |
|---|---|---|---|---|
| paper (single run) | 0.766 / 0.701 | 0.587 / 0.539 | 0.801 / 0.713 | 0.651 |
| predictions shipped in the HF test set (`go_pred`) | 0.757 / 0.689 | 0.586 / 0.542 | 0.805 / 0.717 | 0.649 |
| **released checkpoint, reproduced (beam)** | 0.777 / 0.714 | 0.659 / 0.617 | 0.812 / 0.728 | **0.686** |
| released checkpoint, greedy | 0.778 / 0.716 | 0.658 / 0.616 | 0.812 / 0.728 | 0.687 |

- The paper's numbers are reproduced by the shipped predictions; the public checkpoint scores higher (BP +0.08), so it is probably not the checkpoint behind the paper tables. Reported as "public checkpoint" performance.
- Homology structure of this benchmark: 7,585/8,630 test proteins (88%) have a >=50%-identity DIAMOND hit in GO-GPT's training data; only 332 are below 30% or without hit. GO-GPT's lead over DIAMOND+naive is concentrated in that high-identity bin (0.730 vs 0.595); for the 795 proteins below 50% identity DIAMOND+naive is *better* than GO-GPT (30-50%: 0.457 vs 0.440; <30%: 0.454 vs 0.395).
- Contamination for CAFA-6: 96% of the CAFA-6 training proteins (and 96% of our val/test split) are in `wanglab/gogpt-training-data` *with their GO labels*. A zero-shot GO-GPT score on our CAFA-6 split is therefore an optimistic, leaked reference, not a fair comparison. The fair comparison is the reverse: HiGO trained on GO-GPT's training data and evaluated on GO-GPT's test set (below).
- Zero-shot on CAFA-6 (`runs/gogpt_cafa6/`): 1,500 random proteins of our clustered test split (97% of them in GO-GPT's training set with labels), beam decoding, organism from the CAFA-6 taxon list, CAFA-6 obo + IA: weighted Fmax MF 0.887 / BP 0.871 / CC 0.933 (mean 0.897), unweighted 0.925 / 0.892 / 0.957. HiGO-35M + gate on the same 1,500 proteins: 0.610 / 0.388 / 0.610 (0.536). GO-GPT's 0.897 is far above its own temporal-holdout score (0.686), i.e. it recalls memorised annotations; this number measures leakage, not generalisation.

### Head-to-head on GO-GPT's own benchmark (`python -m cafa6.prepare_gogpt`, `python -m cafa6.bench_gogpt`, `runs/bench_gogpt/`)
HiGO-35M (LoRA q,v r8) trained on `wanglab/gogpt-training-data` (119,749 train / 13,305 val, annotations up to Nov 2022, GO 2023-01-01, 7,901 terms), 4 epochs (time budget), gate fit on GO-GPT's validation split, scored on the 8,630-protein temporal holdout with the paper protocol (CAFA-5 IA, cafaeval prop=max, norm=cafa). 760 test proteins also occur in the training data with other aspects; identity features are therefore computed per split.

| method (IA-weighted Fmax) | MF | BP | CC | mean |
|---|---|---|---|---|
| GO-GPT released checkpoint (ESM2-3B + GPT decoder) | 0.714 | 0.617 | 0.728 | **0.687** |
| HiGO-35M + homology gate | 0.647 | 0.434 | 0.647 | 0.576 |
| DIAMOND + naive | 0.650 | 0.416 | 0.638 | 0.568 |
| HiGO-35M alone | 0.502 | 0.339 | 0.523 | 0.455 |

By identity to the training data (mean): >=50% (n=7,585) GO-GPT 0.730 vs HiGO+gate 0.599; 30-50% (713) HiGO+gate **0.480** vs GO-GPT 0.440; <30% (82) HiGO+gate **0.461** vs 0.395; no hit (250) GO-GPT 0.374 vs HiGO 0.360.
Outcome: on GO-GPT's benchmark GO-GPT is clearly better overall, and the gap sits in the 88% of proteins with a close homolog in training, where the 3B generative model (and a checkpoint that may have seen more data than the published training set, see above) excels. HiGO-35M, trained for 4 epochs, beats GO-GPT on the remote-homology proteins (below 50% identity) and roughly ties it without any homolog. The "better model" claim therefore holds only for remote homologs, not overall. (`python -m cafa6.stack`, `runs/eval_dev/`)

Gate: per-aspect logistic gate g = sigmoid(w . [max identity/100, has_hit, 1]) fit on validation only, p = g p_kNN + (1-g) p_model, then MCM for hierarchy consistency.

| method (test, 8,058 proteins, <30%-identity clusters) | MF | BP | CC | Kaggle mean |
|---|---|---|---|---|
| naive | 0.3631 | 0.2245 | 0.3617 | 0.3164 |
| DIAMOND + naive | 0.5751 | 0.3281 | 0.4809 | 0.4614 |
| frozen ESM2-35M + HiGO head | 0.5647 | 0.3330 | 0.5320 | 0.4766 |
| frozen ESM2-35M + HiGO head + kNN gate | 0.6036 | 0.3620 | 0.5525 | 0.5060 |
| HiGO-35M (LoRA q,v r8) | 0.5835 | 0.3601 | 0.5793 | 0.5077 |
| **HiGO-35M (LoRA) + kNN gate** | 0.6134 | 0.3793 | 0.5903 | **0.5277** |
| HiGO-35M (LoRA all r16) + kNN gate | 0.6134 | 0.3843 | 0.5894 | 0.5291 |

By maximum identity to train (Kaggle mean): no hit (n=1,725) HiGO 0.495 vs DIAMOND+naive 0.333 (+0.16); <30% (908) 0.509 vs 0.443; 30-50% (3,868) gated 0.530 vs 0.483; >=50% (1,557) gated 0.565 vs 0.533. The PLM carries the no-homolog proteins; the gate learns to lean on kNN only when identity is high (fitted gate weight on identity > 0 for all aspects, bias about -2.4, so g is ~0.1 without a hit and ~0.3-0.4 at 100% identity).
By protein IA tercile: gains over DIAMOND+naive grow with specificity: low-IA +0.07, mid +0.07, high-IA (rare terms) +0.06-0.07; the rare-term tercile remains hardest (0.48).
Error analysis (per-protein F1 at tau=0.3, worst decile vs all): the worst proteins have few annotations (median 10 vs 28 terms, i.e. little beyond shallow terms to get right), slightly lower homology (max identity 32% vs 36%) and more specific labels (mean label IA 1.23 vs 1.09); length is not a driver (363 vs 412).

### Ablations (ESM2-35M, LoRA q,v r8, one component removed at a time, same data/seed/schedule)

To save time (user request) every ablation was stopped once it had >=4 epochs; all runs share the same 15-epoch cosine schedule, so the fair comparison is the best validation IA-weighted Fmax within the first 4 epochs. Runs that trained longer also show their best overall.

| run | MF | BP | CC | val Fw mean @4 epochs | delta vs full | best overall (epoch / run) |
|---|---|---|---|---|---|---|
| full HiGO (`dev_35m_lora`) | 0.5867 | 0.3607 | 0.5783 | 0.5086 | - | 0.5270 (8/12) |
| no species embedding | 0.5918 | 0.3459 | 0.5490 | 0.4956 | -0.0130 | - |
| mean pooling instead of evidence attention | 0.5755 | 0.3572 | 0.5692 | 0.5006 | -0.0080 | 0.5218 (12/15) |
| no hierarchy constraint (plain BCE, no MCM) | 0.5796 | 0.3566 | 0.5700 | 0.5021 | -0.0065 | 0.5152 (10/11) |
| flat GO queries (no ancestor sharing) | 0.5840 | 0.3610 | 0.5739 | 0.5063 | -0.0023 | 0.5231 (11/12) |
| softmax instead of entmax pooling | 0.5836 | 0.3616 | 0.5791 | 0.5081 | -0.0005 | 0.5224 (6/7) |
| no IA weighting in the loss | 0.5904 | 0.3622 | 0.5758 | 0.5095 | +0.0009 | - |

Reading: species context (mostly CC/BP) and the hierarchy constraint are the largest contributors, followed by evidence pooling; at convergence the MCM matters most (-0.012). Ancestor-shared queries give a small but consistent gain. Softmax vs entmax is a tie on accuracy, so entmax is kept for its sparse, readable evidence maps rather than for accuracy. IA weighting does not help on validation at this budget.

Held-out test (`python -m cafa6.stack ... --out eval_main`, `runs/eval_main/`; each ablation from its best saved checkpoint, so training lengths differ: no-taxon / no-IA-weight 4 epochs, others 7-15):

| run | test Kaggle mean | + homology gate |
|---|---|---|
| full HiGO | 0.5077 | 0.5277 |
| no species embedding | 0.4760 (-0.032) | 0.5076 (-0.020) |
| no IA weighting | 0.4929 (-0.015) | 0.5191 (-0.009) |
| softmax pooling | 0.5034 (-0.004) | 0.5276 (0.000) |
| mean pooling | 0.5047 (-0.003) | 0.5274 (0.000) |
| no hierarchy constraint | 0.5052 (-0.002) | 0.5239 (-0.004) |
| flat GO queries | 0.5068 (-0.001) | 0.5277 (0.000) |

On test, species context and IA weighting matter most (IA weighting helps on test although it did not on validation; the two short runs are confounded by fewer epochs, so their deltas are upper bounds). The homology gate absorbs most of the smaller effects: after gating, pooling type and query sharing no longer change the score; the hierarchy constraint keeps a small gain.

### 5-fold cross-validation (`python -m cafa6.cv_summary`, `runs/cv_summary/`)
Folds partition the train+val clusters (MMseqs2 30% clusters, test clusters never used); full HiGO-35M, same 4-epoch budget per fold. Validation IA-weighted Fmax: MF 0.600 +- 0.007, BP 0.363 +- 0.005, CC 0.578 +- 0.002, mean **0.5135 +- 0.0038** (folds 0.515 / 0.518 / 0.516 / 0.509 / 0.510). The low spread shows the result is not an artefact of one split.

## Step 6 - Explainability (`python -m cafa6.explain --run dev_35m_lora`, `runs/dev_35m_lora/explain/`)

For 511 confident correct predictions (own-term probability >= 0.5, most specific true term per aspect) on 300 test proteins:
- Sparsity: entmax-1.5 evidence maps put non-zero weight on 39% of residues on average (BP 31%, CC 35%, MF 51%); the top 10% of residues carry 76% of the evidence mass.
- Faithfulness of the evidence map to the head ("pool" deletion: remove the top-10% residues from the pooling, input unchanged): own-term probability drops by 0.067 vs 0.0005 for 10% random residues (Wilcoxon one-sided p = 0.007; MF 0.125, BP 0.060, CC 0.021).
- End-to-end input deletion (replace the same residues by `<mask>` in the sequence): no effect (-0.0003 vs random 0.010, p = 1.0). The transformer re-distributes the information of masked residues to their neighbours, so the maps say *where the head reads* the function, not which residues are causally necessary in the sequence. This is a limitation, reported as such.
- InterPro (EBI API, local entries only: domains, repeats, active/binding/conserved sites, PTMs; proteins where these cover <=80%): 163 cases, mean coverage 48%, attention mass inside 51%, median enrichment 1.13, 57% of cases enriched. Modest agreement.
- Figure: `runs/dev_35m_lora/explain/evidence_Q08492.png` (evidence maps of one protein under different GO terms, local InterPro regions shaded; Q08492 has none). The three terms read different residue sets (the BP map is much sparser than the MF map), but all three put a large weight on residue 1 (the initiator Met), a positional artefact of ESM's N-terminal representation that should be masked or down-weighted in future work.

## Step 8 - Kaggle submission (`python -m cafa6.submit --run dev_35m_lora`, `runs/dev_35m_lora/submission/`)

- Model: HiGO-35M (LoRA q,v r8) + per-aspect homology gate fit on validation, then MCM (hierarchy-consistent). Homology kNN from DIAMOND against all 82,038 labelled training proteins (self hits excluded); 88.3% of the 224,309 test-superset proteins have a hit.
- Inference over the whole sequence with sliding windows (no truncation); top 500 terms per protein with score >= 0.02.
- Output: `submission.tsv` (1.1 GB) and `submission.tsv.gz` (213 MB), format `protein<TAB>GO:term<TAB>score` without header, as in `sample_submission.tsv` (the optional free-text rows are not used).
- Validated: 45,957,813 rows, all 224,309 proteins present, at most 500 terms per protein (limit 1,500), every row has 3 fields, a valid GO id and a score in [0, 1].
- Upload was not done: it needs the user's Kaggle API token (`kaggle competitions submit -c cafa-6-protein-function-prediction -f submission.tsv -m "HiGO-35M + homology gate"`).
- A first version measured the post-MCM probability, which for a term can come from a descendant with a different evidence map, and used whole-protein InterPro families (93% coverage). Both were fixed; the flawed output is kept as `summary_v0_postMCM_flawed.json`.
