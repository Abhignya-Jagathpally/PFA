# Codebase inventory for a new GO-prediction model intended to beat BioReason-Pro / GO-GPT

Scope: everything under `cafa6/` (code, configs, data, runs, logs, docs) plus the Hugging Face cache, inspected on
2026-10-12. Nothing was modified, moved, deleted, trained or run on GPU. Line numbers refer to the files as they are
on disk today (git HEAD `e230974`, with `REPORT.md` modified and `docs/` untracked).

Paths below are relative to `cafa6/` unless stated otherwise.

---

## 0. Summary of the situation in one page

- The working model is **HiGO-35M** (ESM2-35M + LoRA q,v r=8 + per-GO-term entmax attention pooling + ancestor-shared
  GO queries + C-HMCNN max-constraint + species embedding), followed post hoc by a **homology gate** (`stack.py`) and,
  in "HiGO-v2", a **per-aspect logistic grounding calibrator** over domain / ortholog / GOA / ligand evidence
  (`grounded.py`).
- Best numbers (IA-weighted Fmax mean over MF/BP/CC, cafaeval `prop=max`, `norm=cafa`):
  - CAFA-6 clustered test (8,058 proteins): HiGO 0.508, HiGO + gate 0.528, HiGO-v2 grounded **0.541**.
  - GO-GPT temporal holdout (8,630 proteins): GO-GPT released checkpoint **0.687**, HiGO-35M + gate 0.576,
    DIAMOND + naive 0.568, HiGO-v2 grounded 0.550.
- The gap to GO-GPT sits in the 88% of its test proteins with a >=50%-identity training homolog (0.730 vs 0.599).
  HiGO wins below 50% identity (0.480 vs 0.440 at 30-50%, 0.461 vs 0.395 below 30%).
- The HiGO side of that head-to-head is a **4-epoch, single-seed, 35M** model whose validation curve was still rising
  steeply (0.478 -> 0.518 -> 0.534 -> 0.545); the GO-GPT side is a 3B-backbone generative model.
- Main structural limits: 6,276/7,901-term vocabularies (>=30 annotations), only the last ESM layer, no retrieval
  inside the network, kNN signal entering only through a 3-feature gate or a near-zero-weight calibrator feature,
  random (not temporal) validation, single-GPU training loop with no data workers, dense float16 prediction matrices
  on disk, and 4.1 GB of free disk.

---

## 1. Architecture and data flow

### 1.1 Pipeline overview

```
data/raw (CAFA-6 Kaggle files)
   │  python -m cafa6.prepare                     (MMseqs2 clustering, propagation, vocab, Y)
   ▼
data/proc/{train,test}.parquet, Y.npz, vocab.tsv, go_ancestors.npy, go_edges.npy, gt_{val,test}.tsv
   │  python -m cafa6.baselines                    (DIAMOND kNN, naive, identity strata)
   ▼
data/proc/diamond_hits_valtest.parquet, diamond_{val,test}.npy, identity.parquet ; runs/baselines/
   │  python -m cafa6.train --config configs/X.yaml [k=v ...]
   ▼
runs/<name>/{config.yaml, taxa.json, train_log.jsonl, trainable_weights.pt, info.json, pred_{val,test}.npy}
   │  (if the run was killed at a fixed epoch budget) python -m cafa6.dump_preds --run <name>
   ▼
   ├── python -m cafa6.stack --runs ... --out eval_main          (homology gate, strata, error analysis)
   ├── python -m cafa6.grounded --run <name> --out grounded       (HiGO-v2 calibrator, ablations, stability)
   ├── python -m cafa6.prf_table --bench all                      (P/R/F1 tables)
   ├── python -m cafa6.cv_summary                                 (5-fold CV table)
   ├── python -m cafa6.explain --run <name>                       (evidence maps, faithfulness, InterPro overlap)
   ├── python -m cafa6.submit --run <name>                        (Kaggle submission, gate only)
   └── python -m cafa6.predict_grounded --split test|--fasta|--submission   (grounded per-protein output)

GO-GPT benchmark branch:
   benchmarks/gogpt_eval.py gen|score   -> runs/gogpt_cafa5/ (released GO-GPT reproduced)
   python -m cafa6.prepare_gogpt        -> data/proc_gogpt/ (same file layout as data/proc)
   CAFA6_PROC=data/proc_gogpt python -m cafa6.train ... name=gogptbench_35m_lora
   CAFA6_PROC=data/proc_gogpt python -m cafa6.bench_gogpt --runs gogptbench_35m_lora
   CAFA6_PROC=data/proc_gogpt python -m cafa6.grounded --bench gogpt ... --goa_date_max 20221130

Grounding resources:
   python -m cafa6.grounding_data --step all -> data/grounding/*
```

The dataset is switched by the environment variable `CAFA6_PROC`, read once at import time
(`src/cafa6/data.py:15`). `prf_table.py` works around this by spawning a subprocess per benchmark
(`prf_table.py:163-172`).

Orchestration scripts in `tools/`: `run_ablations.sh` (ablations + CV on one GPU), `run_rest.sh` (second wave),
`finish_fast.sh` (kills runs after 4 logged epochs with `pkill`, dumps predictions, runs `stack`/`cv_summary`/
`bench_gogpt`), `watch_done.sh` (agent wake-up loop on `logs/*.done`). These are how the "4-epoch budget" was
imposed: runs were killed externally, not configured to stop.

### 1.2 Module-by-module

#### `src/cafa6/data.py` (102 lines) — shared loaders
- Constants: `ROOT` (repo root), `RAW = data/raw`, `PROC = $CAFA6_PROC or data/proc` (l.13-15),
  `ASPECT_ROOT` (l.17), `AMBIGUOUS = XBZUOJ` (l.19), `PROP_RELS = ("is_a", "part_of")` (l.20).
- `read_fasta(path) -> DataFrame[id, taxon, seq, length, ambig_frac]` (l.23-51): parses UniProt headers
  (`sp|ACC|...OX=`) and Kaggle `>ID taxon` headers.
- `load_go(obo_path) -> (parents: dict[str,set], aspect: dict[str,'F'|'P'|'C'|'?'], alt: dict)` (l.54-66) via `obonet`.
- `ancestor_closure(parents) -> dict[str,set]` (l.69-86), memoised recursion, raises the recursion limit inside.
- `propagate(terms_df, closure, alt) -> DataFrame[id, term]` (l.89-97), true-path rule, alt_id remap.
- `load_ia(path) -> dict[term, float]` (l.100-102).

#### `src/cafa6/prepare.py` (131 lines) — dataset construction
- `mmseqs_cluster(df, min_id, cov, threads) -> Series member->rep` (l.29-37), `easy-cluster --cov-mode 0
  --cluster-mode 0`; temp dir inside `PROC`.
- `main()` knobs (l.41-48): `--min_id 0.3 --cov 0.8 --min_count 30 --max_ambig 0.05 --min_len 20 --seed 0 --threads 64`.
- Steps: cleaning on train only (l.57-62); propagation (l.64-72); cluster-level 80/10/10 random split with
  `np.random.default_rng(seed)` (l.74-84); length stats (l.86-88); vocabulary = terms with >= `min_count`
  propagated annotations among **train-split** proteins, roots excluded (l.90-99); IA-mass coverage (l.101-103);
  `Y` CSR float32 over **all** 82,038 proteins x vocab (l.105-109); `edges` = direct child->parent pairs **only if both
  are in the vocab** (l.110); `anc_pairs` = (child, ancestor) pairs from the full closure restricted to vocab (l.111).
- Outputs (l.113-127): `train.parquet` [id, taxon, seq, length, cluster, split], `test.parquet` (Kaggle superset,
  [id, taxon, seq, length]), `train_terms_propagated.parquet` [id, term, aspect], `vocab.tsv`
  [term, aspect, count, ia] sorted by aspect then count, `Y.npz`, `go_edges.npy` int32 (10,588 x 2),
  `go_ancestors.npy` int32 (56,432 x 2), `gt_{val,test}.tsv` (leaf terms, **not restricted to vocab**),
  `prepare_report.json`.

#### `src/cafa6/model.py` (155 lines) — HiGO
- `build_backbone(name, adapter="lora", r=8, alpha=16, dropout=0.05, targets=("query","value"))` (l.22-32):
  `AutoModel.from_pretrained(name, add_pooling_layer=False)`, all weights frozen, PEFT LoRA/DoRA mounted
  (`use_dora=adapter=="dora"`). Any other adapter string (e.g. `frozen`) leaves the backbone fully frozen.
- `class HiGO(nn.Module)` constructor (l.35-74) signature:
  `HiGO(plm_name, n_terms, anc_pairs, ia, n_taxa, adapter="lora", lora_r=8, d=256, hier_query=True,
  pooling="entmax", use_mcm=True, use_taxon=True, entmax_alpha=1.5, lora_targets=("query","value"),
  go_edges=None, use_go_gat=False, gat_layers=2, gat_heads=4, n_slots=8)`.
  - LoRA alpha is hard-wired to `2 * lora_r` (l.42).
  - `proj = Linear(h, d) + LayerNorm(d)` (l.44); `term_emb` (T x d, std 0.02, l.46); `term_bias` zeros (l.47);
    `key = Linear(d,d,bias=False)` (l.48); `out = Linear(d,d)` (l.49); `taxon_emb = Embedding(n_taxa, d)` (l.52).
  - Optional GAT (`go_graph.GOGraphEncoder`, l.54-58) and slot pooling parameters (l.60-63).
  - `A`: sparse row-normalised ancestor-mean operator built from `anc_pairs` (l.65-68); MCM gather indices
    `mcm_src/mcm_dst` = self pairs + anc pairs (l.70-73); `ia` buffer (l.74).
- `queries() -> T x d` (l.76-83): `q = e + A @ e` in fp32 (autocast off), then optional GAT.
- `mcm(p) -> B x T` (l.85-89): `scatter_reduce(amax)` of each term over itself and its descendants.
- `forward(input_ids, attention_mask, res_mask, taxon=None, return_attn=False, term_chunk=2048)` (l.91-135):
  - `H = plm(...).last_hidden_state` (l.94) — **only the last layer**; `H = proj(H.float())` (l.95), B x L x d fp32.
  - `K = key(H)` (l.97); scores `einsum("td,bld->btl")` per 2,048-term chunk (l.111-123), masked with -1e4 (l.124),
    `entmax_bisect(alpha=1.5, n_iter=30)` or softmax (l.125-126); `v = einsum("btl,bld->btd")` (l.127), `out(v)`,
    `+ taxon` (l.129-130); logit `z_t = <GELU(v_t), q_t> + b_t` (l.131).
  - Returns `(z: B x T fp32/bf16, attn: B x T x L or None)`.
  - `pooling="mean"` (l.113-117) and `pooling="slots"` (l.101-109, l.118-121) alternatives.
- `predict_proba(z)` (l.137-139): sigmoid, then MCM.
- `loss(z, y, ia_weight=True, ia_clip=5.0)` (l.141-155): MCLoss (`pos = mcm(y*p)`, `neg = mcm(p)`), clamped BCE,
  per-term weight `1 + min(IA,5)` normalised to mean 1, **mean over all B x T entries**.

#### `src/cafa6/go_graph.py` (41 lines) — pilot GAT
- `GOGraphEncoder(n_terms, d, edges, n_layers=2, heads=4)` (l.16-35): undirected child<->parent edges + self loops,
  `GATConv(d, d//heads, heads, concat=True, add_self_loops=False)`.
- `forward(q)` (l.37-41): `x = gelu(LayerNorm(conv(x) + x))` per layer; this replaces the 0.02-scale queries with
  unit-variance, GELU-clipped vectors (diagnosed cause of the -0.018 pilot result).

#### `src/cafa6/train.py` (282 lines) — training loop
- `Batcher(seqs, tok, max_len, tokens_per_batch, max_batch)` (l.25-76):
  - `batches(idx, shuffle)` (l.32-53): shuffles, then sorts within 4,096-protein chunks by length, packs to a padded
    token budget `max(L) * n <= tpb` and `n <= max_batch`.
  - `encode(pieces)` (l.55-61): ESM tokenizer; `res_mask` zeroes `<cls>` (pos 0) and `<eos>` (pos len-1).
  - `crop(s)` (l.63-67): random `max_len` crop for training.
  - `windows(s)` (l.69-76): stride `max_len // 2` sliding windows for inference.
- `fast_fmax_w(p, Y, ia, aspect) -> {F, P, C, mean}` (l.80-104): GPU IA-weighted Fmax over the vocab only, thresholds
  0.01..0.99; used for early stopping only.
- `predict(model, batcher, seqs, taxa, idx, device, bf16=True) -> np.float16 [len(idx), T]` (l.107-142): windows,
  own length-bucketing, **max over windows of logits** (l.139), then `predict_proba` (sigmoid + MCM) (l.141).
- `load_cfg()` (l.146-155): YAML + `key=value` overrides parsed with `yaml.safe_load`.
- `taxon_index(train_df, min_count=50)` (l.158-161): species with >=50 training proteins get an id, others 0
  (CAFA-6: 41 + 1; GO-GPT: 69 + 1).
- `main()` (l.164-278): seeds (l.166), `device = cuda` hard-coded (l.170), loads `PROC` files (l.172-182),
  optional k-fold over train+val clusters (`fold`, `n_folds`, l.184-195), `train_subset`/`val_subset` (l.196-199),
  model build (l.203-209), optional gradient checkpointing (l.210-213), two AdamW param groups
  (adapter `lr`, head `head_lr`, weight decay 0.01 on everything, l.221-224), cosine schedule with warm-up over
  `epochs * steps_per_epoch` (l.225-227), per-epoch validation (l.252-255), best-checkpoint saving of
  `requires_grad` tensors only (l.260-263), patience (l.264-266), reload + `pred_{val,test}.npy` dump **only when
  `fold` is None** (l.271-276), `logs/<name>.done` (l.278).
- Config knobs consumed: `name, plm, adapter, lora_r, lora_targets, d, pooling, hier_query, use_mcm, use_taxon,
  ia_weight, max_len, tokens_per_batch, max_batch, grad_accum, lr, head_lr, warmup_frac, epochs, patience,
  taxon_dropout, seed, fold, n_folds, train_subset, val_subset, grad_ckpt, use_go_gat, gat_layers, gat_heads, n_slots`.

#### `src/cafa6/dump_preds.py` (34 lines)
- Rebuilds a run via `explain.load_run`, writes `pred_{val,test}.npy` with the standard split, doubles the batch
  budget (l.25). Used after `finish_fast.sh` killed runs. For CV fold models the "val" split it predicts is part of
  that fold's training data (see §4.3).

#### `src/cafa6/homology.py` (61 lines)
- `diamond_search(db_df, q_df, threads=64, max_hits=50, exclude_self=True) -> DataFrame[q, s, pident, bits, evalue, qcov]`
  (l.23-37): `--more-sensitive -e 1e-3 -k 50`, temp files in `PROC`.
- `diamond_score(hits, q_ids, db_ids, Y_db) -> float32 [nq, T]` (l.40-51): bit-score-weighted average of neighbour
  label vectors (DiamondScore).
- `max_identity(hits, q_ids)` (l.54-57): max `pident` regardless of coverage.
- `identity_bin` (l.60-61): no_hit / <30 / 30-50 / >=50.

#### `src/cafa6/baselines.py` (69 lines)
- Naive frequency, DIAMOND kNN (train split as DB), DIAMOND + naive (0.8/0.2 if hit, naive otherwise, l.45-46).
- Writes `data/proc/diamond_{val,test}.npy` float16, `identity.parquet`, `runs/baselines/{metrics.tsv,summary.json}`.

#### `src/cafa6/evaluate.py` (76 lines)
- `dense_to_long(scores, ids, terms, top_k=1500, min_score=0.01) -> DataFrame[id, term, score]` (l.20-30), scores
  rounded to 3 decimals.
- `evaluate(preds: dict[name, df], gt_file, ids_subset=None, n_cpu=16, out_dir=None, obo=OBO, ia=IA)` (l.33-68):
  writes every prediction table as TSV into a temp dir **inside `PROC`** (l.41), runs `cafaeval.cafa_eval(...,
  norm="cafa", prop="max")`, returns per (method, aspect) Fmax_w, tau_w, Fmax, Smin, coverage, P_w, R_w, P, R.
- `summarize(res)` (l.71-76): wide MF/BP/CC table + `kaggle_mean_Fw`.

#### `src/cafa6/stack.py` (160 lines) — homology gate and test analysis
- `mcm_numpy(p, anc)` (l.25-30): GPU max-constraint on a dense numpy matrix.
- `features(ident, ids) -> [n,3]` = `[max_pident/100, has_hit, 1]` (l.33-36).
- `fit_gate(p_model, p_knn, Y, X, aspect, steps=300)` (l.39-56): per-aspect logistic `g = sigmoid(w·x)`, Adam lr 0.05,
  full-batch BCE of `g*p_knn + (1-g)*p_model` on validation.
- `apply_gate` (l.59-65); `per_protein_f1` (l.68-73).
- `main()` (l.76-156): gate per run, test evaluation, identity and IA-tercile strata (l.117-135), worst-decile error
  analysis at tau 0.3 and taxon enrichment (l.137-152). Outputs `runs/<out>/{metrics,summary_all,strata,gates,
  error_*}.tsv/json`.

#### `src/cafa6/submit.py` (87 lines)
- Gate fit on val (l.46-50), DIAMOND of the 224,309 Kaggle proteins against **all** 82,038 labelled proteins with
  self hits excluded (l.52-57), chunked prediction + gate + MCM, top 500 terms >= 0.02 (l.64-77), gzip TSV.

#### `src/cafa6/cv_summary.py` (38 lines)
- Reads the first `--max_epochs 4` lines of each `runs/cv_35m_fold*/train_log.jsonl` (l.23) and reports best epoch
  mean/std.

#### `src/cafa6/explain.py` (206 lines)
- `load_run(name, device="cuda") -> (model, cfg, vocab, tmap, tok)` (l.29-43): the generic run loader used by
  `dump_preds`, `submit`, `predict_grounded`.
- `forward_one` (l.46-56), `masked_prob(..., mode="input"|"pool")` (l.59-75), `interpro_regions(acc)` via EBI REST
  (l.80-94), `main()` (l.97-202): 300 test proteins with `50 <= length <= max_len` (l.110), most specific confident
  correct term per aspect, deletion faithfulness, InterPro overlap, figure.

#### `src/cafa6/comparative.py` (243 lines) — ortholog pilot
- `REFERENCE_TAXA` (10 model organisms, l.25-31), `build_reference_db` (l.34-36), `cross_species_hits` (l.39-44).
- Writes `data/proc/ortho_{val,test}.npy` (float16, 105/101 MB) and `runs/pilot_comparative_genomics/`.

#### `src/cafa6/sae.py` (418 lines) — SAE pilot
- Loads `dev_35m_lora` (`RUN`, l.42), extracts `H` for 3,000 val proteins in RAM (l.81-113), trains
  `SAE(256 -> 4096 -> 256)` (l.117-170), scores GO purity with permutation null (l.198-236) and contiguity (l.239-277).

#### `src/cafa6/prepare_gogpt.py` (102 lines) — GO-GPT benchmark data
- Downloads `wanglab/gogpt-training-data` train/validation parquet into `/tmp/gogpt_train` (l.38-42), test from
  `runs/gogpt_cafa5/ref/test.parquet` (l.43). Cleans train/val only (l.48). `taxon` = index of the organism **string**
  (3,130 organisms, l.49-50), not an NCBI id. GO 2023-01-01 obo + CAFA-5 `IA.txt` from `REF` (l.52, l.64).
  Vocab `min_count=30` -> 7,901 terms (l.65). `cluster = id` (no clustering, l.74). No `go_edges.npy`, no
  `test.parquet`, no `train_terms_propagated.parquet` are written. DIAMOND kNN against the train split (l.85-94).

#### `src/cafa6/bench_gogpt.py` (97 lines)
- `gogpt_preds(path_glob, sub)` (l.26-34) loads GO-GPT JSONL as score-1 predictions; `main()` fits the gate on the
  GO-GPT validation split, evaluates S3_all, S6_noknowledge (excludes `common_proteins.txt`, l.77-78) and identity
  strata (l.79-82) with GO 2023-01-01 + CAFA-5 IA.

#### `src/cafa6/prf_table.py` (177 lines)
- Builds ordered prediction sets per benchmark (l.58-98). Run names are hard-coded: `base_35m_frozen_mlp`,
  `dev_35m_lora` (l.69-71), `gogptbench_35m_lora` (l.82), `grounded` / `grounded_gogpt` (l.74, l.84). Extra rows via
  `--extra NAME=RUNDIR`. Writes `runs/prf/{cafa6,gogpt}_test.tsv` and `tables.md`.

#### `src/cafa6/grounding_data.py` (905 lines) — external resources
- Steps `static` (external2go, Rhea, ExPASy ENZYME, go-plus CHEBI logical definitions, ChEBI SMILES streamed,
  `go2ligand.tsv`; l.500-508), `uniprot` (REST features for every accession in both benchmarks and the Kaggle
  superset, resumable; l.676-726), `goa` (10 reference GAFs; l.773-804), `readme` (l.819-882).
- Disk guard: `MIN_FREE = 5 GiB` (l.57); `ensure_space` aborts when a download would leave < 5 GiB (l.114-118);
  `run_batches` aborts when free < 5 GiB (l.579-580, l.598-600). **With 4.1 GB free today, any step that has
  work left will stop immediately.**
- Leakage rule documented (l.20-22): curated `binding_chebi`, `cofactor_chebi`, `ec` only for training/reference
  proteins.

#### `src/cafa6/grounded.py` (390 lines) — HiGO-v2 calibrator
- `COMPONENTS` (13 features, l.44): `lp, lk, pid, hit, D, O, G, L, haslig, lfreq, lp_pid, lk_pid, D_lp`;
  `DROP` ablations (l.45-46).
- `Closure(obo_path, vocab_terms)` (l.50-78): term -> vocab indices of itself + ancestors (`nx.descendants` on the
  child->parent graph); `.matrix(go_terms)` sparse closure.
- `Grounding(vocab, closure, goa_date_max=None, exclude_ids=())` (l.103-182):
  - `IP` interpro x T (interpro2go closed over the ontology, l.109-114).
  - `LigT` T x ChEBI from `go2ligand.tsv`, currency metabolites from Rhea/EC removed (l.115-127).
  - `GOA` goa-protein x T, optional date cut, evaluation proteins removed (l.128-135).
  - `domains(ids, interpro_lists=None)` (l.142-145) -> sparse D.
  - `orthologs(hits, ids, q_taxon, db_df, Y_db)` (l.147-167) -> (dense O, sparse G, filtered hits); only hits whose
    subject is in `REFERENCE_TAXA` and of a different species.
  - `ligands(D, h, ids)` (l.169-182) -> (L_dom, L_hit).
- `logit(p)` clips to [1e-4, 1-1e-4] (l.186-188).
- `components(...)` -> dense float32 `n x t x 13` (l.191-208).
- `fit_calibrator(F, y, steps=400, lr=0.05)` (l.211-223): one logistic regression per aspect, standardised features,
  shared across all terms of that aspect, unweighted BCE, full batch on GPU.
- `calibrate(...)` (l.231-249), `violations` (l.253-254), `jaccard_across` (l.257-263).
- `main()` (l.267-386): for `--bench gogpt` maps organism codes to NCBI taxa from `uniprot_features.parquet`
  (l.280-283); evidence for val/test (l.312-321); variants `grounded`, `grounded_no_domain`, `grounded_no_ortholog`,
  `grounded_no_ligand`, `grounded_gate_only` (l.330-342); writes `pred_val.npy` (calibrator applied **in-sample** to
  the validation split it was fit on, l.337-340) and `pred_test.npy`; stability over the 5 CV models (l.356-385).

#### `src/cafa6/predict_grounded.py` (367 lines)
- `Explainer(vocab, anc, goa_exclude=(), goa_date_max=None)` (l.45-63) loads `runs/grounded/calibrator.json` and
  takes the per-aspect decision thresholds from `runs/grounded/metrics.tsv` (l.56-58), i.e. the **test-split** Fmax
  thresholds.
- `describe` (l.89-118) builds per-term records (path, domains, orthologs, ligands with SMILES, top residues).
- `uniparc_interpro` (l.157-171), `interproscan` (EBI REST, l.174-193).
- Modes: `--split test` (l.222-253), `--fasta` (l.256-303; truncates each sequence to `max_len`, l.281, no windows),
  `--submission` (l.306-341). Model run `dev_35m_lora` and `RUN_DIR = runs/grounded` are hard-coded
  (l.39, l.230, l.278, l.316).

#### `benchmarks/gogpt_eval.py` (393 lines)
- Re-implements `GOGPTPredictor.predict` with batching and an exact cached protein stream
  (`_protein_stream` l.73-90, `_go_stream` l.93-120, `install_protein_cache` l.123-136), ESM layer grab via a
  pre-hook that stops the forward early (`esm_layer`, l.143-162), greedy and beam decoding (l.165-207, truncation to
  1,024 tokens at l.190), organism resolver (l.210-224), scoring with the cafaeval CLI for protocols `S3_all`,
  `S6_noknowledge`, `cafa6_ia` (l.53-57, l.333-370).
- Depends on `/tmp/BioReason-Pro-main/gogpt/src` (l.47), which lives in `/tmp` (31 MB present today, but volatile).

#### Configs (`configs/*.yaml`)
| file | name | backbone | key settings |
|---|---|---|---|
| `dev.yaml` | dev_35m_lora | ESM2-35M | LoRA q,v r8, d 256, entmax, all head components on, max_len 1022, 16,384 tokens, max_batch 32, lr 3e-4 / head 1e-3, warm-up 6%, **epochs 15, patience 3** (l.18-19), taxon dropout 0.1, seed 0 |
| `final_3b.yaml` | final_3b | ESM2-3B | same head, `grad_ckpt: true` (l.16), lr 2e-4 (l.17), **epochs 5, patience 2** (l.20-21) |
| `pilot_baseline.yaml` | pilot_baseline_35m_lora | ESM2-35M | `train_subset 20000`, `val_subset 4000`, **epochs 4** (l.18-20) |
| `pilot_gat.yaml` | pilot_gat_35m_lora | ESM2-35M | as pilot + `use_go_gat`, 2 layers, 4 heads (l.24-26) |
| `pilot_slots.yaml` | pilot_slots_35m_lora | ESM2-35M | as pilot + `pooling: slots`, `n_slots: 8` (l.6-7) |

Ablations, CV and the GO-GPT run were launched from `dev.yaml` with command-line overrides (`tools/run_ablations.sh`,
`tools/run_rest.sh`), so their effective configs live only in `runs/<name>/config.yaml`.

#### `requirements.txt`
torch 2.13.0 (cu130), transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, entmax 1.3, cafaeval 1.3.0, torch_geometric
2.8.0.post1, esm 3.4.1.post1, obonet 1.3.0, goatools 1.6.5, networkx 3.7, pandas 3.0.6, numpy 2.3.5, scipy 1.18.1,
scikit-learn 1.9.1, datasets 4.3.0, safetensors 0.8.0, biopython 1.88, pytorch-lightning 2.6.6 (unused by the code).
`requests`, `matplotlib`, `pyyaml`, `huggingface_hub`, `pyarrow` are imported but not pinned. The venv is `PFA/bin/python`.
Binaries: `tools/diamond` (DIAMOND 2.1.10, 105 MB), `tools/mmseqs/bin/mmseqs`.

---

## 2. Data assets on disk

Free disk on `/`: **4.1 GB** of 1.4 TB (`df -h`, 100% used). RAM 251 GB, 128 CPUs, 2 x H100 NVL 95.8 GB (idle at
inspection time).

### 2.1 Raw inputs (`data/raw`, 191 MB)
| file | size | content |
|---|---|---|
| `Train/train_sequences.fasta` | 53 MB | 82,404 sequences |
| `Train/train_terms.tsv` | 10.8 MB | 537,027 leaf annotations (26,125 terms), no dates |
| `Train/train_taxonomy.tsv` | 1.0 MB | protein -> taxon |
| `Train/go-basic.obo` | 31 MB | GO 2025-06-01, 40,122 terms |
| `IA.tsv` | 1.0 MB | 40,122 terms, 28,294 with IA > 0 |
| `Test/testsuperset.fasta` | 101 MB | 224,309 Kaggle proteins, `>ID taxon` |
| `Test/testsuperset-taxon-list.tsv` | 0.2 MB | taxon id -> species |
| `sample_submission.tsv` | 0.5 MB | format reference |

### 2.2 CAFA-6 processed (`data/proc`, 616 MB)
| file | size | shape / dtype | notes |
|---|---|---|---|
| `train.parquet` | 44.5 MB | 82,038 rows [id str, taxon int64, seq str, length int64, cluster str, split str] | 1,293 taxa; length p50/p95/p99/max 411/1,320/2,376/35,213; 9.3% > 1,022 aa |
| `test.parquet` | 97.9 MB | 224,309 rows [id, taxon, seq, length] | 8,453 taxa; p95 1,095; 6.0% > 1,022 aa; **36.6% of ids also in `train.parquet`** |
| `train_terms_propagated.parquet` | 8.7 MB | 3,560,200 rows [id, term, aspect] | 29,384 distinct terms |
| `vocab.tsv` | 0.2 MB | 6,276 terms [term, aspect, count, ia] | BP 4,666 / MF 903 / CC 707; min count 30; 87.9% have IA > 0; IA median 0.65, max 10.65 |
| `Y.npz` | 3.7 MB | CSR 82,038 x 6,276 float32, 3,191,956 nnz | 38.9 vocab labels per protein on average |
| `go_edges.npy` | 85 KB | int32 (10,588, 2) | direct parent edges inside the vocab only |
| `go_ancestors.npy` | 452 KB | int32 (56,432, 2) | (child, ancestor) pairs inside the vocab |
| `gt_{val,test}.tsv` | 1.0 / 0.96 MB | leaf (id, term) | not restricted to vocab |
| `diamond_hits_valtest.parquet` | 3.2 MB | 284,565 hits [q, s, pident, bits, evalue, qcov] | val+test vs train split |
| `diamond_hits_kaggle_test.parquet` | 71.9 MB | 4,163,307 hits | Kaggle superset vs all labelled, self hits removed |
| `diamond_{val,test}.npy` | 105 / 101 MB | float16 (8,377 / 8,058, 6,276) | DiamondScore |
| `ortho_{val,test}.npy` | 105 / 101 MB | float16 | ortholog pilot (dropped from the model) |
| `identity.parquet` | 0.2 MB | 16,435 rows [id, split, max_pident, id_bin] | |
| `prepare_report.json` | | | all prepare statistics |

Split sizes: train 65,603 / val 8,377 / test 8,058 proteins from 33,003 MMseqs2 clusters (30% id, 80% cov).
Residual homology of test to train: >=50% 1,557; 30-50% 3,868; <30% 908; no hit 1,725 (val: 1,871 / 3,740 / 940 / 1,826).

Vocabulary coverage (computed for this inventory): 21,717 of the 27,996 terms seen in the train split have fewer than
30 annotations and are excluded. The vocab holds 94% of the val/test propagated annotations but only **89-90% of the
IA mass** (MF 83-84%, BP 90-91%, CC 92-93%). `GO:0005344` (oxygen carrier activity) has 14 train annotations and is
absent; `GO:0020037` (heme binding) is present.

### 2.3 GO-GPT benchmark processed (`data/proc_gogpt`, 441 MB)
| file | size | shape | notes |
|---|---|---|---|
| `train.parquet` | 84 MB | 141,684 rows [id, taxon, seq, length, split, cluster, organism] | train 119,749 / val 13,305 / test 8,630; `taxon` is an organism-string index (3,130); `cluster = id`; length p95 1,513 |
| `vocab.tsv` | 0.26 MB | 7,901 terms | BP 5,960 / MF 1,099 / CC 842 |
| `Y.npz` | 6.0 MB | 141,684 x 7,901, 4,873,787 nnz | 34.4 labels / protein |
| `go_ancestors.npy` | 0.66 MB | (82,293, 2) | **no `go_edges.npy`** |
| `gt_{val,test}.tsv` | 9.3 / 4.7 MB | propagated (id, term) | test GT IA mass inside vocab: 93.8% |
| `diamond_{val,test}.npy` | 210 / 136 MB | float16 | |
| `diamond_hits_valtest.parquet` | 10 MB | 590,952 hits | written by `grounded.py` |
| `identity.parquet` | 0.3 MB | [id, split, max_pident] | no `id_bin` column |

760 GO-GPT test proteins also occur in train/val (other aspects). Validation is the dataset's own random split, not
temporal.

Raw GO-GPT reference files (`runs/gogpt_cafa5/ref`, 46 MB): `test.parquet` (8,630 proteins with propagated GT and the
shipped `go_pred`), `go-basic_gogpt.obo` (GO 2023-01-01), `IA.txt` (CAFA-5), `common_proteins.txt` (471 ids),
`cafa_evals.py`, `run_cafa_eval.sh`, `data_README.md`. The training parquet is cached at `/tmp/gogpt_train` (148 MB,
volatile).

### 2.4 Grounding resources (`data/grounding`, 131 MB; see `data/grounding/README.md`)
| file | rows | content |
|---|---|---|
| `interpro2go.tsv` | 30,122 | InterPro -> GO (current release, not date-filtered) |
| `ec2go.tsv` / `rhea2go.tsv` | 4,831 / 7,875 | |
| `rhea_participants.tsv` | 87,619 | Rhea master id -> ChEBI |
| `go2chebi.tsv` | 27,398 | go-plus logical definitions |
| `chebi.tsv` | 15,251 | ChEBI name + SMILES (Rhea first, then ChEBI structures, descendant fallback) |
| `go2ligand.tsv` | 73,350 | MF term -> ChEBI (6,063 GO terms, 7,880 ChEBI ids, 99.9% with SMILES), `ubiquitous` flag |
| `ec.tsv` | 8,472 | ExPASy ENZYME names + transfers |
| `goa_reference.parquet` | 2,349,235 | GOA for 10 reference proteomes [id, go, evidence, date, taxon, aspect]; dates 1999-01-01 .. 2026-07-27; IEA 694k, ISO 370k, IBA 340k, IDA 271k, IMP 159k, ISS 140k, IPI 87k, TAS 53k |
| `uniprot_features.parquet` | 285,414 | [id, interpro, pfam, ec, binding_chebi, cofactor_chebi, taxon, found, primary_acc, feature_source] |
| `uniprot_partial.tsv` etc. | | resumable fetch state (23 MB) |
| `raw/` | | GAFs, Rhea, ChEBI, go-plus, go.obo sources (~70 MB) |

InterPro coverage: CAFA train/val/test 98%, GO-GPT test 98% (64% found in current UniProtKB, rest via UniParc),
Kaggle superset 95%. EC / binding / cofactor (curated, train/reference only): ~34% / 30% / 14% of CAFA proteins.

### 2.5 Cached backbones and models (`~/.cache/huggingface/hub`, blobs stored under `hub/blobs/<2-char>/`)
| model | on disk | usable? |
|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | 30 MB safetensors | yes |
| `facebook/esm2_t12_35M_UR50D` | 130 MB safetensors | yes (all HiGO runs) |
| `facebook/esm2_t33_650M_UR50D` | **absent** | needs ~2.6 GB download (fits only barely in 4.1 GB) |
| `facebook/esm2_t36_3B_UR50D` | **11.3 GB** fp32 `.bin` (9.3 + 1.3 GB shards) | yes (used by `final_3b` and GO-GPT) |
| `wanglab/gogpt` | 3.0 GB `model.ckpt` + tokenizer/organism maps | yes |
| `wanglab/bioreason-pro-rl` | 4.5 MB (configs only) | weights not cached |
| `biohub/esm3-sm-open-v1` | 5.2 GB | yes; structure-token VQ-VAE + sequence model (relevant for structure tokens) |
| `sentence-transformers/all-mpnet-base-v2` | 419 MB | yes; candidate GO-definition text encoder |
| `Qwen/Qwen2.5-0.5B-Instruct` | 954 MB | yes; candidate text encoder |
| `emilyalsentzer/Bio_ClinicalBERT` | 3.0 GB | biomedical text encoder (clinical, not ideal for GO) |
| others (Geneformer, MOMENT, chronos, ViT, BERT, distil*) | ~4 GB | not relevant |

`~/.cache/torch` holds 2.6 GB of unrelated vision checkpoints (BLIP-2, EVA-ViT, ResNet).

### 2.6 Embeddings
No per-protein or per-residue embedding cache exists anywhere on disk. NOTES.md Step 1 records that residue-level caches
were ruled out by disk; every run recomputes the backbone forward. The only stored activations are the SAE weights
(`runs/pilot_sae/sae_weights.pt`, 8.4 MB).

### 2.7 Model outputs reusable as features
Every completed run stores dense `pred_val.npy` / `pred_test.npy` (float16, 105 / 101 MB on CAFA-6; 210 / 136 MB on
GO-GPT). Available prediction sets: `dev_35m_lora`, `dev_35m_lora_all_r16`, `base_35m_frozen_mlp`, 6 ablations,
5 CV folds (test only is clean, see §4.3), 3 pilots (trained on 20k proteins), `gogptbench_35m_lora`, `grounded`,
`grounded_gogpt`. GO-GPT predictions: `runs/gogpt_cafa5/preds_shard{0,1,2}.jsonl` (beam, 8,630 proteins),
`runs/gogpt_cafa5/greedy/preds_shard0.jsonl`, `runs/gogpt_cafa6/preds_shard0.jsonl` (1,500 CAFA-6 proteins).
Grounded evidence: `runs/grounded/evidence_D_{val,test}.npz`, `has_ligand.npy`; per-protein grounded output
`runs/grounded/output/test_grounded.{jsonl,tsv}` (103 + 17 MB). Submissions: `runs/dev_35m_lora/submission/
submission.tsv` (1.1 GB) + `.gz` (213 MB, 45,957,813 rows), `runs/grounded/submission/submission.tsv.gz` (382 MB,
84,991,864 rows).

---

## 3. Experimental record

Validation numbers are the fast in-loop metric (`train.fast_fmax_w`, vocab-only, propagated labels); test numbers are
cafaeval. Times are wall-clock from `train_log.jsonl` / `info.json`; several runs shared a GPU, so per-epoch times vary.

### 3.1 Training runs (CAFA-6 data unless noted)

| run | config deltas vs `dev.yaml` | epochs logged / best | best val Fw (MF/BP/CC/mean) | test Kaggle raw / +gate | train time | converged? |
|---|---|---|---|---|---|---|
| `dev_35m_lora` | — | 12 / 8 (early stop) | 0.606 / 0.384 / 0.591 / **0.5270** | 0.5077 / 0.5277 | 7,764 s (~650 s/epoch alone) | yes on Fw; val loss minimum at epoch 4 (0.02023) then rises to 0.02083 |
| `dev_35m_lora_all_r16` | LoRA r16 on q,k,v,dense (1.66M params, 4.7%) | 10 / 6 | 0.600 / 0.387 / 0.589 / 0.5253 | 0.5087 / **0.5291** | 5,493 s | yes; val loss rises from epoch 4 |
| `dev_35m_dora` | DoRA r8 | 6 / 5 (stopped) | 0.597 / 0.376 / 0.584 / 0.5189 | no preds | ~5,100 s | no (killed) |
| `base_35m_frozen_mlp` | frozen, **mean pooling, no hier_query, no MCM, no taxon, no IA weight** | 15 / 12 | 0.575 / 0.347 / 0.544 / 0.4885 | 0.4766 / 0.5060 | 3,875 s (~260 s/epoch) | yes |
| `abl_no_hier_query` | flat queries | 12 / 11 | 0.600 / 0.383 / 0.586 / 0.5231 | 0.5068 / 0.5277 | 13,463 s | almost (still +0.001/epoch) |
| `abl_no_mcm` | plain BCE, no MCM | 11 / 10 | 0.594 / 0.375 / 0.577 / 0.5152 | 0.5052 / 0.5239 | 12,825 s | almost |
| `abl_mean_pool` | mean pooling | 15 / 12 | 0.597 / 0.385 / 0.583 / 0.5218 | 0.5047 / 0.5274 | 10,488 s | yes (ran full schedule) |
| `abl_softmax_pool` | softmax | 7 / 6 (killed) | 0.600 / 0.379 / 0.589 / 0.5224 | 0.5034 / 0.5276 | 7,901 s | no |
| `abl_no_taxon` | no species embedding | 4 / 3 (killed) | 0.592 / 0.346 / 0.549 / 0.4956 | 0.4760 / 0.5076 | 7,475 s | **no** (+0.011 in last epoch) |
| `abl_no_ia_weight` | unweighted MCLoss | 4 / 3 (killed) | 0.590 / 0.362 / 0.576 / 0.5095 | 0.4929 / 0.5191 | 7,501 s | **no** (+0.012 in last epoch) |
| `cv_35m_fold{0..4}` | k-fold over train+val clusters, `epochs 8 patience 2` | 4 / 3 each (killed) | mean 0.5135 ± 0.0038 (folds 0.515/0.518/0.516/0.509/0.510) | test raw 0.4948 ± 0.0063 across 6 models | 4,670-7,670 s | **no** (+0.008-0.009 in last epoch) |
| `pilot_baseline_35m_lora` | 20k train / 4k val, 4 epochs | 4 / 3 | 0.535 / 0.339 / 0.554 / 0.4761 | — | 531 s | no |
| `pilot_gat_35m_lora` | + GO GAT | 4 / 3 | 0.504 / 0.328 / 0.542 / 0.4582 | — | 534 s | no (epoch-0 train loss 0.49 vs 0.06) |
| `pilot_slots_35m_lora` | slot pooling (8) | 4 / 3 | 0.519 / 0.336 / 0.546 / 0.4665 | — | 423 s | no |
| `final_3b` | ESM2-3B, LoRA q,v r8 (2.95M, 0.10%), grad ckpt | 1 logged (killed in epoch 2) | 0.539 / 0.346 / 0.569 / **0.4848** after epoch 1 (35M: 0.445) | no preds | 8,749 s for epoch 0 (~3.3 s/step, ~2,450 steps) | **no**; only `trainable_weights.pt` (21.5 MB) of epoch 0 exists |
| `gogptbench_35m_lora` (proc_gogpt) | `epochs 10 patience 2`, 7,901 terms, 70 taxa | 4 / 3 (killed) | 0.637 / 0.405 / 0.592 / 0.5447 | GO-GPT test raw 0.4547 / +gate 0.5759 | 9,784 s | **no**: 0.478 -> 0.518 -> 0.534 -> 0.545 |

Notes on curves:
- Train loss keeps falling for every run while val loss bottoms out early (epoch 4-6 for 35M) and climbs again; the
  selection metric (Fw) keeps rising for several more epochs. The model overfits in calibration before ranking.
- The 4-epoch runs (no_taxon, no_ia_weight, CV, GO-GPT bench) all used the 15-/8-/10-epoch cosine schedule and were
  killed at epoch 4, so their learning rate never decayed. Their deltas mix component effects with schedule position.

### 3.2 Post-hoc combinations (CAFA-6 clustered test, Kaggle mean)
| method | MF | BP | CC | mean | source |
|---|---|---|---|---|---|
| naive | 0.363 | 0.224 | 0.362 | 0.316 | `runs/baselines/summary.json` |
| DIAMOND kNN | 0.523 | 0.314 | 0.450 | 0.429 | same |
| DIAMOND + naive | 0.575 | 0.328 | 0.481 | 0.461 | same |
| frozen ESM2 mean-pool head + gate | 0.604 | 0.362 | 0.552 | 0.506 | `runs/eval_main/summary_all.tsv` |
| HiGO + gate | 0.613 | 0.379 | 0.590 | 0.528 | same |
| HiGO LoRA-all r16 + gate | 0.613 | 0.384 | 0.589 | 0.529 | same |
| calibrator, gate features only | 0.588 | 0.380 | 0.586 | 0.518 | `runs/grounded/summary.tsv` |
| grounded, no domain | 0.591 | 0.379 | 0.586 | 0.519 | same |
| grounded, no ligand | 0.614 | 0.412 | 0.595 | 0.540 | same |
| grounded, no ortholog | 0.617 | 0.413 | 0.594 | 0.541 | same |
| **HiGO-v2 grounded** | 0.616 | 0.412 | 0.595 | **0.541** | same |

Gate weights (`runs/eval_main/gates.json`, `[identity, has_hit, bias]`): dev_35m_lora MF [0.81, 1.29, -2.44],
BP [1.15, 0.44, -2.47], CC [1.36, 0.60, -2.66] -> g ≈ 0.08 without hit, ≈ 0.3-0.4 at 100% identity.
On the GO-GPT data the fitted gate trusts kNN more: MF [1.56, 0.24, -1.14], BP [2.43, -0.05, -2.07],
CC [2.98, -0.12, -2.40] (`runs/bench_gogpt/gates.json`).

Calibrator weights (`runs/grounded/calibrator.json`, standardised): D 1.32-1.50, O 0.76-0.81, lp 0.34-0.50,
**lk 0.023-0.035**, lk_pid 0.09-0.18, G 0.33-0.42 (MF/BP), L -0.40 (MF). On GO-GPT (`runs/grounded_gogpt/
calibrator.json`): G 0.69 (MF) / 0.81 (BP), D 0.78-1.12.

Strata (`runs/eval_main/strata.tsv`, HiGO + gate vs DIAMOND + naive): no hit 0.495 vs 0.333 (n=1,725); <30% 0.509 vs
0.443 (908); 30-50% 0.530 vs 0.483 (3,868); >=50% 0.565 vs 0.533 (1,557). IA terciles: low 0.594 vs 0.523, mid 0.521
vs 0.455, high 0.480 vs 0.417. On no-hit proteins the gated and raw model are identical (gate ≈ 0).

Error analysis (`runs/eval_main/error_summary.tsv`, `error_taxa.tsv`): worst-decile proteins have 10 vs 28 labels,
32% vs 36% identity, mean label IA 1.23 vs 1.09; over-represented taxa 44689 (Dictyostelium, 2.2x), 7955 (zebrafish,
1.9x), 39947 (rice, 1.4x), rat 1.3x; human (0.47x), yeasts (0.27-0.29x) under-represented.

Stability (`runs/grounded/stability.tsv`, 6 models): raw 0.4948 ± 0.0063 (Jaccard 0.58), gate 0.5180 ± 0.0048 (0.65),
grounded 0.5323 ± 0.0043 (0.79), 0 violations.

### 3.3 GO-GPT benchmark (8,630 proteins, GO 2023-01-01, CAFA-5 IA)
| method | MF | BP | CC | mean | file |
|---|---|---|---|---|---|
| paper Table S3 | 0.701 | 0.539 | 0.713 | 0.651 | `runs/gogpt_cafa5/SUMMARY.md` |
| shipped `go_pred` | 0.689 | 0.542 | 0.717 | 0.649 | `runs/gogpt_cafa5/metrics.tsv` |
| released checkpoint, beam | 0.714 | 0.617 | 0.728 | **0.687** | same; S6 subset 0.702 |
| released checkpoint, greedy | 0.716 | 0.616 | 0.728 | 0.686 | `greedy/metrics.tsv` |
| HiGO-35M + gate | 0.647 | 0.434 | 0.647 | 0.576 | `runs/bench_gogpt/summary.tsv` |
| DIAMOND + naive | 0.650 | 0.416 | 0.638 | 0.568 | same |
| HiGO-35M raw | 0.502 | 0.339 | 0.522 | 0.455 | same |
| HiGO-v2 grounded | 0.618 | 0.406 | 0.627 | 0.550 | `runs/grounded_gogpt/summary.tsv` |
| calibrator, gate features only | 0.634 | 0.431 | 0.634 | 0.566 | same |

By identity: >=50% (7,585) GO-GPT 0.730 / HiGO+gate 0.599 / DIAMOND+naive 0.595; 30-50% (713) 0.440 / **0.480** /
0.457; <30% (82) 0.395 / **0.461** / 0.454; no hit (250) **0.374** / 0.359 / 0.203.

Unweighted Fmax of GO-GPT (beam): MF 0.777, BP 0.659, CC 0.812. GO-GPT zero-shot on 1,500 CAFA-6 test proteins:
0.887 / 0.871 / 0.933 (mean 0.897, `runs/gogpt_cafa6/metrics.tsv`); 96-97% of those proteins are in GO-GPT's training
set with labels, so this measures memorisation.

### 3.4 Interpretability and pilots
- Evidence maps (`runs/dev_35m_lora/explain/summary.json`, 511 cases): 39% non-zero residues; top 10% carry 76%;
  pool-deletion drop 0.067 vs 0.0005 (p = 0.007); input-masking drop -0.0003 vs 0.010 (p = 1.0); InterPro median
  enrichment 1.13 (163 cases). Residue-1 weight noted in NOTES Step 6 / REPORT §4.3.
- Ortholog channel (`runs/pilot_comparative_genomics/`): -0.004 vs DIAMOND + naive on val and test.
- SAE (`runs/pilot_sae/`): 99.9% variance explained, not sparse (371 dense latents), 50 analysed latents cover 15 terms.
- Grounded output (`runs/grounded/output/test_grounded_info.json`): 50 terms / protein, specific terms with domain
  support 12%, ortholog 82%, ligand 2.4%, 0 closure violations.

### 3.5 Timings and throughput
| job | measurement |
|---|---|
| ESM2-35M LoRA training | ~2,450 steps / epoch at 16,384 padded tokens per step, ~660 s for the training pass alone -> ~3.7 steps/s, ~60k padded tokens/s; validation pass ~110 s (`logs/dev_35m_lora.log`) |
| ESM2-35M frozen | ~230 s per training epoch (`logs/base_35m_frozen_mlp.log`) |
| ESM2-35M on GO-GPT data (119,749 proteins) | ~4,400 steps / epoch, 1,700-3,860 s / epoch on a shared GPU |
| ESM2-3B LoRA + grad ckpt | ~3.3 s/step (~5k padded tokens/s), 8,749 s for epoch 0 including ~800 s validation, ~36 GB GPU memory (NOTES Step 4) |
| GO-GPT generation (3B fp32 + decoder, beam 5) | 3 shards x ~2,877 proteins in 6,650-6,770 s each (~1.9 h wall on one H100 with 3 processes); 0.77-1.0 s/protein for short proteins up to 3.9 s/protein for long ones; peak 12.1-21.9 GiB |
| DIAMOND Kaggle superset vs 82k | ~4 min (`logs/diamond_kaggle.log` timestamps 02:25 -> 02:29) |
| grounded calibrator | ~1 h including the 5-model stability pass |

---

## 4. Weak spots, bugs, shortcuts and hard-coded assumptions

### 4.1 Training budget and statistics
1. **4-epoch budgets imposed by killing processes** (`tools/finish_fast.sh` `cap()`), with cosine schedules sized for
   8-15 epochs (`train.py:225-227`). Affects `abl_no_taxon`, `abl_no_ia_weight`, all CV folds, and the GO-GPT
   head-to-head (`gogptbench_35m_lora`, +0.011 in its last epoch). The GO-GPT comparison is therefore against an
   under-trained 35M model.
2. **Single seed** (`seed: 0` everywhere; `train.py:166`; fold assignment uses a fixed `default_rng(0)`, l.191;
   subsets fixed `default_rng(0)`, l.197-199). Ablation deltas of 0.001-0.008 sit inside the 0.004-0.006 spread
   measured across CV models.
3. **3B model never finished**: one epoch, no predictions, only the epoch-0 adapter (`runs/final_3b/`).
4. Validation loss rises from epoch ~4 while Fw rises until epoch 8 (`dev_35m_lora/train_log.jsonl`): probability
   calibration drifts; the post-hoc gate/calibrator then has to correct it.
5. `train_log.jsonl` is opened in append mode (`train.py:229`); re-running a name appends to the old log and
   `cv_summary.py:23` takes the first `max_epochs` lines, so a rerun would be silently ignored.
6. `REPORT.md` labels `base_35m_frozen_mlp` as "frozen ESM2-35M + HiGO head", but its config turns off pooling
   attention, hierarchical queries, MCM, taxon and IA weighting (`runs/base_35m_frozen_mlp/config.yaml`). The
   "LoRA vs frozen" gain (+0.031 raw, +0.022 gated) therefore confounds the adapter with the head components.
   `prf_table.py` labels it correctly as "Frozen ESM2 + MLP".

### 4.2 Model design limits (`model.py`)
7. **Only the last ESM layer** (`model.py:94`). GO-GPT uses layer 30 of 36 for the 3B model
   (`gogpt_eval.py:143-162`); intermediate layers are commonly better for function.
8. **Sliding-window max pooling of logits** (`train.py:139`): evidence from different windows is never combined
   before the nonlinearity; long multi-domain proteins (9.3% of train, 6% of Kaggle test above 1,022 aa) are scored as
   the max over independent windows. Training sees one random crop (`train.py:63-67, 236`) with **all** labels of the
   full protein, which is label noise for crops that miss the relevant region.
9. **Residue-1 attention artefact**: `res_mask` excludes `<cls>`/`<eos>` (`train.py:57-60`) but not the first residue;
   all maps put large weight on Met1 (NOTES Step 6). No positional prior or mask on residue 1.
10. Per-term attention cost is O(B x T x L) per chunk with `entmax_bisect(n_iter=30)` (`model.py:111-126`); this
    dominates head compute and memory and grows linearly with vocabulary size. Moving from 6k to ~30k terms
    multiplies this cost by ~5.
11. Term bias initialised at zero (`model.py:47`) instead of `logit(prior)`; epoch-0 loss is high (0.058) and the
    first epoch is partly spent learning the prior.
12. Loss is a mean over all B x T entries (`model.py:155`): dominated by easy negatives, no asymmetric/focal term, no
    per-protein normalisation. IA weight normalisation (`w / w.mean()`, l.153-154) fixes the scale but not imbalance.
13. Weight decay 0.01 is applied to term embeddings, biases and LayerNorm (`train.py:223-224`).
14. LoRA alpha fixed to 2r and dropout 0.05 (`model.py:42`, l.22); targets default to `query,value` only.
15. Taxon embedding limited to 41 species (`train.py:158-161`, `min_count=50`); 1,293 train taxa and 8,453 Kaggle taxa
    mostly fall into "other". On GO-GPT data the "taxon" is an organism-string index, not an NCBI id
    (`prepare_gogpt.py:49-50`), so no taxonomy-aware sharing is possible.
16. GAT pilot implementation replaces rather than refines queries (`go_graph.py:37-41`); `go_edges` only keeps direct
    parent edges whose both ends are in the vocab (`prepare.py:110`), so the graph is fragmented where intermediate
    terms were dropped. `proc_gogpt` has no edges file at all.

### 4.3 Data, splits and leakage
17. **Validation is not temporal.** CAFA-6: random cluster-level split (`prepare.py:74-84`); `train_terms.tsv` has no
    dates. GO-GPT: the dataset's random validation split, while the test is temporal (Mar 2023-Feb 2024). The gate and
    calibrator are fit on validation and transfer badly to the temporal test (grounded 0.550 vs gate 0.576, REPORT
    §8.4).
18. **Truncated vocabulary** (`prepare.py:94`, `prepare_gogpt.py:65`, `min_count=30`): 21,717 train terms dropped,
    10-11% of val/test IA mass unreachable on CAFA-6 (16-17% for MF), 6% on the GO-GPT test. Examples:
    `GO:0005344` oxygen carrier activity (14 train annotations). This is a hard ceiling on IA-weighted recall for
    exactly the high-IA terms that dominate the metric.
19. CV fold models: `dump_preds` writes `pred_val.npy` on the standard val split (`dump_preds.py:26-28`), which is
    inside the fold's training data (`train.py:188-195`). Only their `pred_test.npy` is clean.
20. `grounded/pred_val.npy` is the calibrator applied to its own training data (`grounded.py:337-340`); stacking on it
    would be in-sample.
21. `predict_grounded.Explainer` takes the decision thresholds from the **test** metrics file
    (`predict_grounded.py:56-58`); the reported Fmax is unaffected, but deployed term sets use test-tuned thresholds.
22. Gate and calibrator are fit on the same validation split used for early stopping (`train.py:255-263`,
    `stack.py:103`, `grounded.py:331`).
23. `interpro2go` and GOA are current releases (2026); GOA is date-filtered only for the GO-GPT bench
    (`--goa_date_max 20221130`), interpro2go never. Part of the domain gain may come from mappings curated after the
    test annotations (REPORT §8.4).
24. `max_identity` ignores coverage (`homology.py:54-57`): a short local hit at high identity counts as a close
    homolog, inflating the identity feature and strata.
25. Clustering at 30% identity / 80% coverage (`prepare.py:20-37`) leaves local homology across splits (79% of test
    proteins have a DIAMOND hit to train).
26. Kaggle submission excludes self hits (`submit.py:56`, `predict_grounded.py:311-313`): 36.6% of the 224,309 Kaggle
    proteins are in the training set, and their own known annotations are not used as a prior. Whether that helps
    depends on how CAFA-6 scores proteins that already have annotations; worth checking against the competition
    rules before the next submission.

### 4.4 Post-hoc combination
27. **kNN underweighted in the calibrator**: `lk = logit(clip(p_knn, 1e-4))` (`grounded.py:186-188, 194`). Most
    kNN scores are exactly 0, which map to the constant -9.21, so after standardisation the feature mostly encodes
    "has any neighbour with this term" and receives weight 0.02-0.04; the information enters through `lk_pid`. A raw
    score, `log(p+eps)`, or a per-identity-bin interaction would let near-identical neighbours dominate (globin demo
    misses heme binding despite a 100%-identity training neighbour, REPORT §8.6).
28. The homology gate has 3 features and is a per-protein scalar per aspect (`stack.py:33-36, 59-65`); it cannot
    express "trust kNN for this term because the neighbour has it at high identity but not for that term".
29. Calibrator weights are shared across all terms of an aspect (`grounded.py:211-223`); no term-level or
    depth-level parameters, unweighted BCE (not IA-aware, not Fmax-aware).
30. DIAMOND + naive uses a fixed 0.8/0.2 blend (`baselines.py:45-46`, `stack.py:99-100`).

### 4.5 Engineering debt that blocks scaling (ESM2-650M/3B, retrieval, structure, text)
31. No data loader workers, tokenisation and `Y[b].toarray()` on the main thread every step (`train.py:236-238`).
32. Single GPU, `device = cuda` hard-coded (`train.py:170`, `stack.py:26`, `grounded.py:211`); no DDP/FSDP, no
    `accelerate`, no `torch.compile`, no SDPA/flash attention configuration for ESM.
33. Validation prediction every epoch over the full split with windows (`train.py:252`); for 3B this was ~800 s per
    epoch.
34. Run reconstruction is duplicated in four places (`train.py:203-209`, `explain.load_run` l.34-41,
    `sae.load_higo` l.64-71, `predict_grounded` via `load_run`); new constructor arguments must be threaded through all
    of them or `load_state_dict(strict=False)` silently drops weights (`explain.py:42`, `sae.py:73`).
35. Predictions are dense float16 `N x T` matrices (`train.py:111`, `baselines.py:43`); at T ≈ 30k this is ~500 MB per
    split per run, impossible with 4.1 GB free.
36. `evaluate()` writes all prediction tables as TSV inside `PROC` (`evaluate.py:41-50`); each method can be
    hundreds of MB (up to 1,500 rows/protein). Large multi-method evaluations can fill the remaining disk.
37. `grounding_data.py` refuses to run with < 5 GiB free (l.57, l.114-118, l.579-580), so new UniProt/GOA fetches are
    currently blocked.
38. Hard-coded run names in `prf_table.py` (l.69-71, 82), `predict_grounded.py` (l.39, 230, 278, 316), `sae.py`
    (l.42).
39. `benchmarks/gogpt_eval.py` imports GO-GPT code from `/tmp/BioReason-Pro-main` (l.47) and `prepare_gogpt` caches to
    `/tmp/gogpt_train`; both disappear on reboot.
40. `predict_grounded --fasta` truncates to `max_len` (l.281) instead of using `train.predict` windows.
41. `config.yaml` overrides are parsed with `yaml.safe_load` per value (`train.py:152-154`); no schema validation,
    unknown keys are silently ignored.
42. Minor: NOTES.md quotes 10,609 parent edges / 56,371 ancestor pairs, `prepare_report.json` says 10,588 / 56,432.

---

## 5. Extension points

Effort estimates assume one engineer familiar with the code; "S" ≈ <1 day, "M" ≈ 1-3 days, "L" ≈ 1-2 weeks.

| extension | where it plugs in | what to change | effort | notes |
|---|---|---|---|---|
| **Larger backbone (ESM2-650M / 3B)** | `configs/*.yaml` `plm`; `model.build_backbone` (l.22-32) | config only for 3B (cached); add multi-layer feature tap (e.g. `output_hidden_states` + learned layer mix) at `model.py:94`; add DDP/`accelerate` in `train.main`; set `tokens_per_batch` per GPU | S (config) / M (layer mix + DDP) | 650M needs a 2.6 GB download; 3B is 11.3 GB fp32 already cached, ~3.3 s/step, ~2.5 h/epoch on one H100 |
| **New adapter** (e.g. LoRA on more layers, IA3, prefix, adapters on top-k layers only) | `model.build_backbone` (l.27-31) | add a branch on `adapter`; expose `alpha`, `dropout`, `layers_to_transform` from config; thread through `train.py:203-209` and `explain.load_run` | S | DoRA and LoRA-all were no better on 35M; probably more relevant at 3B |
| **Retrieval-augmented attention block** | `HiGO.forward` between pooling and logit (`model.py:127-131`) or as an extra key/value source | new module taking the top-k DIAMOND/embedding neighbours (ids from `diamond_hits_*.parquet`, labels from `Y.npz`, optionally neighbour embeddings) and letting term queries attend over `[residue evidence; neighbour label embeddings weighted by identity/bits]`; `Batcher`/training loop must pass neighbour indices per protein (`train.py:236-242`), excluding same-cluster neighbours during training to avoid leakage | M-L | this is the most direct route to GO-GPT's >=50%-identity regime; neighbours for train proteins need a train-vs-train DIAMOND run (not on disk; ~small, the 82k DB search took minutes) |
| **Label-text encoder** | `HiGO.queries()` (`model.py:76-83`) and `term_emb` (l.46) | precompute GO name+definition embeddings (from `data/raw/Train/go-basic.obo` / `data/grounding/raw/go.obo.gz`) with a cached text model (`all-mpnet-base-v2` 419 MB, or Qwen2.5-0.5B), store as `T x d_text` npy (6k x 768 fp16 ≈ 9 MB; 40k ≈ 60 MB), project into `d`, add to the ancestor-mean query | S-M | enables zero/few-shot terms below `min_count`, i.e. lifts the vocab ceiling |
| **Open / full vocabulary** | `prepare.py:90-99`, `prepare_gogpt.py:65` (`min_count`), `model.py` term chunking | lower `min_count` (or keep all terms with IA > 0) once text queries exist; switch to sparse top-k prediction storage (`train.predict` l.111, `dense_to_long`) | M | per-term attention cost grows linearly with T; consider sharing attention across terms (a few pooled heads) and scoring all terms with a dot product |
| **Structure-token input** | `Batcher.encode` (`train.py:55-61`) and `HiGO.forward` (`model.py:94-95`) | add a second token stream (e.g. ESM3 structure tokens or Foldseek 3Di) embedded and summed/concatenated with `H` before `proj`; requires precomputed tokens per protein (one byte per residue: ~37 MB for CAFA-6, ~100 MB for the Kaggle superset) | M-L | ESM3-sm-open (5.2 GB) is cached; AlphaFold structures are not on disk; a structure source is the main blocker |
| **Set / diffusion / autoregressive decoder** | replace `logits` assembly (`model.py:110-135`) and `loss` (l.141-155) | e.g. a small transformer over term embeddings conditioned on pooled evidence (set prediction with Hungarian loss or iterative refinement / discrete diffusion over the label vector), keeping MCM at the output | L | can still output per-term scores, so `predict`, `stack`, `evaluate` stay unchanged |
| **Gated GO-graph refinement** | `go_graph.GOGraphEncoder.forward` (l.37-41), `HiGO.queries` (l.81-82) | `q + g * GAT(q)` with `g` zero-initialised, no GELU on the last layer; export full-closure edges for `proc_gogpt` | S | fix described in REPORT §7.1 |
| **Temporal validation split** | `prepare.py:74-84` (CAFA-6), `prepare_gogpt.py:37-46` (GO-GPT) | CAFA-6 labels have no dates, but `goa_reference.parquet` has dates for 10 proteomes (1999-2026): build a time-sliced val set "annotations added after date X" for those proteomes; for GO-GPT, split train by annotation date if available in the HF dataset, otherwise hold out the newest UniProt entries | M | needed before refitting any gate/calibrator; also enables CAFA-style "no-knowledge vs limited-knowledge" analysis |
| **Better homology fusion** | `stack.features` (l.33-36), `grounded.components` (l.191-208) | per-term features: neighbour bits/identity for that term, coverage-aware identity (`qcov` is already in the hits), count of supporting neighbours, raw `p_knn` instead of clipped logit; per-depth or per-IA-bin weights | S-M | cheap and directly targets the >=50% identity bin |
| **Residue-1 fix** | `Batcher.encode` (`train.py:57-60`) or a learned positional bias in `HiGO.forward` (l.123-124) | zero `res[:, 1]` or subtract a learned per-position prior | S | changes evidence maps; check accuracy |
| **Multi-window aggregation** | `train.predict` (l.131-139) | aggregate window `H` (or window `v_t`) before the logit, e.g. concatenate window evidence and run entmax across all windows' residues | M | requires batching windows of the same protein together |
| **Loss** | `HiGO.loss` (`model.py:141-155`) | asymmetric loss / focal / per-protein normalisation, logit-prior bias init | S | |
| **Seeds and budgets** | configs + a small sweep script in `tools/` | 3 seeds per arm, schedules sized to the epochs actually run | S | compute-bound, not code-bound |

---

## 6. Compute and disk constraints

### 6.1 Disk
- Free on `/`: **4.1 GB**. Usage: `runs/` 6.2 GB, `data/` 1.38 GB (`proc` 616 MB, `proc_gogpt` 441 MB, `raw` 191 MB,
  `grounding` 131 MB), `tools/` 105 MB, HF cache ~32 GB (esm2-3B 11.3 GB, esm3 5.2 GB, GO-GPT 3.0 GB, unrelated
  models ~12 GB), torch cache 2.6 GB.
- Largest items in `runs/` (listed only, nothing removed): `dev_35m_lora/submission/submission.tsv` 1.1 GB (a 213 MB
  `.gz` copy exists), `grounded` 677 MB (382 MB submission gz + preds), `grounded_gogpt` 331 MB, `gogptbench_35m_lora`
  341 MB, each completed CAFA-6 run ~205 MB (two float16 prediction matrices), `gogpt_cafa5` 214 MB. `data/proc/
  ortho_{val,test}.npy` (206 MB) belong to the dropped ortholog pilot.
- Per-unit sizes: dense CAFA-6 prediction pair 206 MB per run; 35M adapter+head checkpoint 8.3 MB; 3B adapter+head
  21.5 MB; Kaggle submission 213-382 MB gzipped.
- Consequences: no ESM2-650M download without freeing space; no residue-level embedding cache (CAFA-6 alone is
  ~37M residues; at 2,560-d fp16 that is ~190 GB for 3B); per-protein 3B embeddings would be ~420 MB for CAFA-6 and
  ~1.15 GB for the Kaggle superset; `grounding_data.py` stops below 5 GiB free.

### 6.2 GPU
- 2 x H100 NVL (95.8 GB each). Observed: ESM2-3B LoRA training with gradient checkpointing ~36 GB at 16,384 padded
  tokens per step; GO-GPT inference 12.1-21.9 GiB peak per process (3 processes shared one GPU).
- ESM2-35M runs were routinely run 3-5 at a time on one GPU (`tools/run_ablations.sh`, `tools/run_rest.sh`), which is
  why per-epoch times range from 650 s to 1,950 s.
- Throughput: 35M ≈ 3.7 steps/s ≈ 60k padded tokens/s; 3B ≈ 0.3 steps/s ≈ 5k padded tokens/s (one GPU, shared).
  A 5-epoch 3B run is ~12-13 h on one GPU at that rate; the GO-GPT training set (1.8x more proteins) would be ~22 h.

### 6.3 CPU / network
- 128 CPUs; DIAMOND and cafaeval use 16-64 threads. InterProScan via EBI REST is too slow for bulk use (a single job
  did not finish in 20 minutes, NOTES Step 10); UniParc MD5 lookup takes seconds per sequence.

---

## 7. Leverage points ranked by expected impact on the GO-GPT benchmark

1. Close-homolog regime (88% of GO-GPT test): retrieval inside the model or a per-term homology fusion with
   coverage-aware identity; current fusion barely uses kNN (gate g ≤ 0.4, calibrator `lk` ≈ 0.03).
2. Backbone: ESM2-3B is cached and the 3B adapter already beat 35M by +0.04 after one epoch; add multi-layer features.
3. Training to convergence with schedules matched to the budget and ≥3 seeds; the GO-GPT HiGO run stopped at a
   still-rising curve.
4. Vocabulary ceiling: GO-definition text queries (encoder cached) plus a lower `min_count`.
5. Temporal validation for every post-hoc fit (gate, calibrator, thresholds), built from GOA dates.
6. Domain evidence is the one grounding feature with a clear gain (+0.013 CAFA-6, BP +0.033); feed InterPro
   (98% coverage, computed matches) into the network rather than only the calibrator, with a date-matched interpro2go.
7. Engineering: data workers, DDP over both GPUs, sparse prediction storage, a single run-loader, and recovering disk
   space before any of the above.
