# Code audit: HiGO / CAFA-6 codebase, inventory for a next-generation model that beats GO-GPT / BioReason-Pro

Audit date 2026-10-12. Read-only. Root: `/home/aj0486@students.ad.unt.edu/PFA/cafa6` (abbreviated `cafa6/`).
Python: `PFA/bin/python` (torch 2.13 cu130, transformers 5.17, peft 0.21, cafaeval 1.3.0, entmax 1.3, torch_geometric 2.8).
Line numbers refer to the files as of git HEAD `e230974`.

---------------------------------------------------------------------------------------------------------

## 0. Executive summary (what matters for beating GO-GPT)

1. **The single largest lever is retrieval, not the PLM.** On GO-GPT's temporal test (8,630 proteins), 51% (4,410)
   have a DIAMOND hit with >=95% identity in GO-GPT's training split (27% at 100%). Copying the labels of the
   *single best* hit (by bitscore) gives, pooled over aspects, IA-weighted recall / precision of
   0.915 / 0.739 (100% id, n=1,922) and 0.871 / 0.711 (95-99.9%, n=2,488) (cheap check run during this audit,
   precision is underestimated because the hit's labels in non-GT aspects count as false positives).
   The current kNN (`homology.diamond_score`) is a bitscore-weighted **mean over up to 50 hits** (median 44 hits per
   test query), which dilutes the near-identical neighbour; the gate only mixes it at g<=~0.6 and the grounded
   calibrator gives the kNN logit weight 0.02-0.13. Result: HiGO+gate ~= DIAMOND+naive on GO-GPT test
   (0.576 vs 0.568) and loses 0.730 vs 0.599 on the >=50% bin, which is 88% of that test set.
2. **Validation is not temporal.** CAFA split = random 30%-identity clusters (val has 0.6% proteins >=95% id);
   GO-GPT bench val = GO-GPT's random val split (30% >=95% id, 8.4% no hit). Gate/calibrator are fit on these and
   transfer badly to the temporal test (grounded calibrator: 0.550 vs gate 0.576 on GO-GPT test).
3. **Under-trained, small backbone.** GO-GPT-bench HiGO = ESM2-35M, 4 epochs (still improving: val Fw 0.4041->0.5447,
   train loss 0.0132 vs val 0.0141), ~41 min/epoch, single GPU. ESM2-3B weights *are cached* (11.4 GB `.bin`), 1 epoch
   was run on CAFA (val Fw 0.485 after epoch 1 vs 35M's 0.445), 2.4 h/epoch with grad checkpointing.
   No DDP/multi-GPU code although 2x H100 NVL 96 GB are available.
4. **Vocabulary ceiling.** GO-GPT bench vocab 7,901 terms (min_count 30) covers 93.8% of the test GT IA mass
   (MF only 88.7%, BP 94.2%, CC 95.8%); CAFA vocab 6,276 covers 89.9% (MF 83.8%). GO-GPT's tokenizer has 10,303 tokens.
5. **Large unused side information** in `wanglab/gogpt-training-data` (cached at `/tmp/gogpt_train/`, *volatile /tmp*):
   InterPro ids + residue spans (95% train / 83% test), AFDB structure ids (94% / 73%), STRING ids and PPI partners
   (75% / 66% train; `ppi_formatted` 14% test), protein names, function text, subcellular location. HiGO uses only
   sequence + organism code.
6. **Hard environment constraint: disk is full** (`/` 1.4 TB, 4.1 GB free right now). No residue-level embedding
   caches are possible; any new download (ESM2-650M, AFDB structures, InterProScan DB) needs space to be freed first.

---------------------------------------------------------------------------------------------------------

## 1. Data pipeline

### 1.1 `src/cafa6/data.py` (102 lines) - shared I/O and ontology
- `ROOT` = repo root (L13); `RAW = data/raw` (L14); `PROC = Path(os.environ.get("CAFA6_PROC", data/proc))` (L15).
  **PROC is fixed at import time**: switching benchmark requires `CAFA6_PROC=data/proc_gogpt` in the environment
  (prf_table.py re-spawns itself in a subprocess for that reason, prf_table.py L163-172).
- `ASPECT_ROOT` (L17), `AMBIGUOUS = set("XBZUOJ")` (L19), `PROP_RELS = ("is_a", "part_of")` (L20).
- `read_fasta(path) -> DataFrame[id, taxon, seq, length, ambig_frac]` (L23-51): UniProt header (`sp|ACC|...OX=`) or
  `>ID taxon` (Kaggle test).
- `load_go(obo_path=data/raw/Train/go-basic.obo) -> (parents, aspect, alt)` (L54-66), via obonet; only is_a/part_of.
- `ancestor_closure(parents)` (L69-86): memoised recursion, raises recursion limit to 100k.
- `propagate(terms, closure, alt)` (L89-97): true-path propagation (same rule as cafaeval).
- `load_ia(path=data/raw/IA.tsv)` (L100-102).
Reuse: all of it; small and correct.

### 1.2 `src/cafa6/prepare.py` (131 lines) - CAFA-6 split + label space
- CLI defaults (L40-49): `--min_id 0.3 --cov 0.8 --min_count 30 --max_ambig 0.05 --min_len 20 --seed 0 --threads 64`.
- `mmseqs_cluster(df, min_id, cov, threads)` (L29-37): `mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0
  --cluster-mode 0` (binary `tools/mmseqs/bin/mmseqs`, release 8cc5ce3).
- Cleaning, train only (L58-62): 36 ambiguous + 330 short dropped -> 82,038.
- Propagation (L65-72): 536,289 leaf -> 3,560,200 propagated annotations.
- Split (L75-84): **random shuffle of the 33,003 cluster representatives, 80/10/10** -> train 65,603 / val 8,377 /
  test 8,058. Not temporal.
- Vocab (L91-98): terms with >=30 propagated annotations in *train split*, roots excluded, sorted by aspect then
  count -> `vocab.tsv[term, aspect, count, ia]`: 6,276 (P 4,666 / F 903 / C 707). 12% of vocab terms have IA=0.
- `Y` (L106-109): `scipy.sparse.csr` float32 (82,038 x 6,276), nnz 3,191,956 (~39 labels/protein).
- `go_edges.npy` (L110): (10,588, 2) int32 **direct parent edges whose both ends are in vocab** (comment says "nearest
  vocab ancestors" but the code drops an edge when the direct parent is out of vocab, so the GAT graph is
  disconnected in places).
- `go_ancestors.npy` (L111): (56,432, 2) int32 (child, ancestor) pairs inside vocab -> used by the ancestor-mean
  query operator and MCM.
- Outputs (L113-126): `data/proc/{train.parquet[id,taxon,seq,length,cluster,split], test.parquet[id,taxon,seq,length]
  (224,309 Kaggle superset), train_terms_propagated.parquet, vocab.tsv, Y.npz, go_edges.npy, go_ancestors.npy,
  gt_{val,test}.tsv (leaf terms; cafaeval re-propagates), prepare_report.json}`.
- Length stats: train p50/p95/p99/max 411/1,320/2,376/35,213; Kaggle test p95 1,095.
- Taxa: 1,293 distinct NCBI taxa in train.parquet.

### 1.3 `src/cafa6/prepare_gogpt.py` (102 lines) - GO-GPT head-to-head data
- Downloads `wanglab/gogpt-training-data` train/validation parquet into **`/tmp/gogpt_train`** (L38-42; volatile).
  Test = `runs/gogpt_cafa5/ref/test.parquet` (8,630 proteins). Ontology `runs/gogpt_cafa5/ref/go-basic_gogpt.obo`
  (GO 2023-01-01), IA `runs/gogpt_cafa5/ref/IA.txt` (CAFA-5).
- Filters only train/val (ambig <=5%, len >=20; L47-48); test untouched.
- **`taxon` = integer index of the UniProt organism *string*** (L49-50), not an NCBI taxon id (3,130 organisms).
  grounded.py has to remap to NCBI taxa via `uniprot_features.parquet` (grounded.py L280-283).
- Labels = union of go_bp/go_mf/go_cc + closure (L54-60). Vocab >=30 train annotations -> 7,901 (P 5,960 / F 1,099
  / C 842); Y (141,684 x 7,901), nnz 4,873,787; anc pairs 82,293. **No `go_edges.npy`** (GAT unusable here).
- `cluster = id` (L74): **no redundancy clustering**; val = GO-GPT's own random validation split.
- `gt_{val,test}.tsv` = all propagated labels (L79-82).
- kNN (L84-93): DIAMOND val/test vs **train split only** (119,749; val 13,305 labels never used as neighbours);
  `diamond_{val,test}.npy` float16 (13,305 / 8,630 x 7,901); `identity.parquet[id, split, max_pident]` (no `id_bin`).
- `diamond_hits_valtest.parquet` (590,952 rows, 20,483 queries) is written later by grounded.py.
- 760 test proteins also occur in train/val (labels in other aspects); self hits excluded by id (homology.py L35-36),
  identical sequences under other accessions are kept.

### 1.4 Batching, cropping, windows (`src/cafa6/train.py` L25-76)
- `Batcher(seqs, tok, max_len, tokens_per_batch, max_batch)`:
  - `batches(idx, shuffle)` (L32-53): shuffle, then sort by length inside 4,096-protein chunks, greedy packing under
    `max(len)+2 * n <= tokens_per_batch` and `n <= max_batch`, then shuffle batches.
  - `encode(pieces)` (L55-61): ESM tokenizer with special tokens; `res_mask` = attention_mask with `<cls>` (pos 0) and
    `<eos>` zeroed -> pooling only over residues.
  - `crop(s)` (L63-67): train-time random 1,022-residue crop.
  - `windows(s)` (L69-76): eval-time windows of 1,022 with stride 511 (+ a final right-aligned window).
- `predict(...)` (L108-142): all windows of all proteins length-sorted and packed, BF16 autocast, **max-pool of
  logits over windows**, then `predict_proba` (sigmoid + MCM) per protein in a Python loop; returns float16
  (n x T).
- Nothing is cached: no embedding cache, the PLM is re-run every epoch (disk constraint, NOTES Step 1). DIAMOND hits and
  kNN score matrices *are* cached (see section 4.6).

---------------------------------------------------------------------------------------------------------

## 2. Model

### 2.1 `src/cafa6/model.py` (155 lines)
`build_backbone(name, adapter="lora", r=8, alpha=16, dropout=0.05, targets=("query","value"))` (L22-32):
`AutoModel.from_pretrained(name, add_pooling_layer=False)`, all params frozen, then PEFT `LoraConfig(r, lora_alpha,
lora_dropout, target_modules, bias="none", use_dora=(adapter=="dora"))`. Any other adapter string (e.g. `frozen`)
-> no adapter. Note HiGO passes `alpha=2*lora_r` (L42). `lora_all_r16` is not an adapter value; it is
`adapter=lora, lora_r=16, lora_targets=[query,key,value,dense]` (runs/dev_35m_lora_all_r16/config.yaml).

`class HiGO(nn.Module)` (L35-155), constructor signature:
```
HiGO(plm_name, n_terms, anc_pairs, ia, n_taxa, adapter="lora", lora_r=8, d=256, hier_query=True,
     pooling="entmax", use_mcm=True, use_taxon=True, entmax_alpha=1.5, lora_targets=("query","value"),
     go_edges=None, use_go_gat=False, gat_layers=2, gat_heads=4, n_slots=8)
```
Parameters / buffers:
- `proj = Sequential(Linear(h, d), LayerNorm(d))` (L44) on the **last hidden layer only**.
- `term_emb` (T x d, init N(0, 0.02)), `term_bias` (T), `key = Linear(d, d, bias=False)`, `out = Linear(d, d)`
  (L46-49), shared by all terms.
- `taxon_emb = Embedding(n_taxa, d)` if use_taxon (L52).
- GAT: `GOGraphEncoder` if use_go_gat (L54-58); slots: `slot_query (n_slots x d)`, `slot_key` (L60-63).
- `A`: sparse row-normalised ancestor-mean operator (T x T, COO) (L65-68).
- `mcm_src/mcm_dst` = self pairs + (desc, anc) pairs (L70-73); `ia` buffer (L74).
Head params at d=256, T=6,276: 1.88 M (info.json `head_params` 1,878,660).

`queries()` (L76-83): `q = e + A @ e` (fp32, autocast off) if hier_query; then `go_gat(q)` if enabled.

`forward(input_ids, attention_mask, res_mask, taxon=None, return_attn=False, term_chunk=2048)` (L91-135):
1. `H = plm(...).last_hidden_state`; `H = proj(H.float())` (B x L x d); `K = key(H)`.
2. Term chunks of 2,048:
   - `pooling="mean"`: masked mean, same vector for all terms.
   - `pooling="entmax"` (default) / `"softmax"`: scores `einsum("td,bld->btl", q, K) * d^-0.5`, masked with -1e4,
     `entmax_bisect(alpha=1.5, n_iter=30)` or softmax over residues; `v = einsum("btl,bld->btd", a, H)`.
   - `pooling="slots"`: n_slots learned queries pool residues with entmax, terms attend over slots (no residue maps).
3. `v = out(v) (+ taxon_emb)`; **logit z_t = sum(GELU(v) * q_t) + b_t** (L131).
Returns `(z [B x T], attn [B x T x L] or None)`.
- `predict_proba(z)` = sigmoid then MCM (L137-139). `mcm(p)` = `scatter_reduce(amax)` over descendants (L85-89).
- `loss(z, y, ia_weight=True, ia_clip=5.0)` (L141-155): C-HMCNN MCLoss (`pos = mcm(y*p)`, `neg = mcm(p)`) +
  BCE, per-term weights `1 + min(IA, 5)` normalised to mean 1, mean over all B x T entries. No positive/negative
  re-balancing (positives are ~0.6% of entries), no focal/ASL.

Memory/compute: the per-term attention tensor is B x T x L (e.g. 16 x 6,276 x 1,024 = 103 M elements per tensor,
several kept for backward, chunked by 2,048 terms but all chunks retained for autograd). This scales linearly with
vocab size; a 10k-30k vocab at L=1,024 will need either smaller term chunks with checkpointing, top-k residue
pre-selection, or a slot/latent bottleneck. Measured speed: ESM2-35M LoRA ~650 s/epoch on 65k CAFA proteins
(solo GPU), ~1,100-1,900 s/epoch when 2-3 jobs shared a GPU; GO-GPT bench (120k proteins) 2,446 s/epoch;
ESM2-3B + grad-ckpt 8,749 s/epoch (~3 s/step, 36 GB).

### 2.2 `src/cafa6/go_graph.py` (41 lines)
`GOGraphEncoder(n_terms, d, edges, n_layers=2, heads=4)`: undirected child<->parent + self-loops, `GATConv(d, d//heads,
heads, concat=True, add_self_loops=False)`, each layer `x = gelu(LayerNorm(conv(x) + x))` (L38-41). Known defect
(REPORT 7.1): LayerNorm rescales the 0.02-std queries and the final GELU clips negatives, so queries are replaced
rather than refined (pilot -0.018). Fix: `q + g * GAT(q)` with g=0 init and no final GELU. Requires `go_edges.npy`
(missing in proc_gogpt).

### 2.3 Where an in-house adapter can be inserted (HF `transformers/models/esm/modeling_esm.py`, v5.17)
Module tree of `EsmModel` (no pooler with `add_pooling_layer=False`):
- `embeddings.word_embeddings` (Embedding), `embeddings.token_dropout` (ESM mask-ratio rescaling, L221-252);
  rotary embeddings live at model level `rotary_embeddings` (L637-640) and are passed as `position_embeddings`.
- `encoder.layer.{i}` = `EsmLayer` (L466-521, pre-LN):
  - `attention.LayerNorm` (pre-attention LN, L418)
  - `attention.self.{query,key,value}` (nn.Linear, L336-338), attention via `config._attn_implementation`
    (eager/sdpa/flash, L381)
  - `attention.output.dense` (+dropout +residual, L399-409)
  - `LayerNorm` (pre-FFN LN, L480), `intermediate.dense` (h -> 4h, GELU), `output.dense` (4h -> h, +residual)
  - Optional `crossattention` (EsmAttention) exists if `config.is_decoder and config.add_cross_attention` (L473-477):
    a native slot for cross-attending to GO/neighbour/structure tokens inside the backbone.
- `encoder.emb_layer_norm_after` (final LN, L529) -> `last_hidden_state`.
- `contact_head.regression` (unused).
Hook points:
- PEFT `target_modules` names: `query`, `key`, `value`, `dense` (matches attention.output.dense, intermediate.dense,
  output.dense), or regex like `encoder.layer.(2[4-9]|3[0-5]).attention.self.(query|value)` to adapt top layers only.
- Bottleneck/parallel adapters: `register_forward_hook` on `encoder.layer.{i}` (output = hidden states tensor) or on
  `encoder.layer.{i}.output` (post-FFN residual).
- Intermediate-layer read-out: `register_forward_pre_hook(..., with_kwargs=True)` on `encoder.layer.{k}` as in
  `benchmarks/gogpt_eval.py` `esm_layer()` L143-162 (raises an exception to stop the forward early: GO-GPT reads
  **layer 30 of 36** of ESM2-3B; HiGO reads only the last layer). Alternatively `output_hidden_states=True`.
- Gradient checkpointing: `model.plm.gradient_checkpointing_enable()` + `enable_input_require_grads()`
  (train.py L210-213); `EsmLayer` subclasses `GradientCheckpointingLayer`.
Config sizes: 8M (6 layers, h=320), 35M (12 layers, h=480, 20 heads, ffn 1,920), 3B (36 layers, h=2,560, 40 heads).

---------------------------------------------------------------------------------------------------------

## 3. Training (`src/cafa6/train.py`, `configs/*.yaml`, `tools/*.sh`)

### 3.1 `train.py` (282 lines)
- `load_cfg()` (L146-155): YAML + `key=value` overrides (yaml-parsed).
- `taxon_index(train_df, min_count=50)` (L158-161): taxa with >=50 train proteins -> 1..K, else 0.
  CAFA: 41 taxa (n_taxa 42). Taxon dropout to 0 with prob `taxon_dropout` (0.1) per protein (L239-240).
- Folds (L184-195): optional `fold`/`n_folds` over train+val clusters (seeded shuffle, i % n_folds).
  `train_subset`/`val_subset` (L196-199).
- Optimiser (L221-224): AdamW two groups: adapter `lr` (3e-4; 2e-4 for 3B), head `head_lr` (1e-3); weight decay 0.01
  for both (incl. LayerNorm/bias/embeddings).
- Schedule (L225-227): `get_cosine_schedule_with_warmup`, warm-up `warmup_frac`=0.06 of `epochs * steps`; the cosine
  horizon is the max epoch count, so early-stopped or killed runs never reach the low-LR tail (all 4-epoch runs
  stopped at ~25% of a 10-15-epoch cosine).
- Loop (L232-267): BF16 autocast forward, loss in fp32, `grad_accum` (leftover micro-batches at epoch end are not
  stepped), clip-norm 1.0 over adapter+head.
- Validation each epoch: full `predict` on val + `fast_fmax_w` (L80-104: GPU IA-weighted Fmax, CAFA normalisation,
  thresholds 0.01..0.99, **vocab terms only**, so it ignores out-of-vocab GT and is optimistic vs cafaeval) and val
  BCE on post-MCM probabilities.
- Checkpoint (L260-263): only `requires_grad` params (adapter + head) to `runs/<name>/trainable_weights.pt` (8.3 MB for
  35M LoRA, 21.5 MB for 3B LoRA) when val Fw improves. No optimizer/scheduler state, no resume.
- Early stopping `patience` (3 dev, 2 others). After training (non-fold runs) dumps `pred_{val,test}.npy` (float16).
- Logs `runs/<name>/train_log.jsonl`, `info.json`, `logs/<name>.done`.
- **No multi-GPU**: `device = torch.device("cuda")` (L170); parallelism was done by launching several runs per GPU
  with `CUDA_VISIBLE_DEVICES` (tools/*.sh). No DDP, no FSDP, no torch.compile, no fused optimiser.
- Val set is used both for early stopping and (later) to fit the gate/calibrator.

### 3.2 Configs
| file | plm | key settings | epochs/patience | notes |
|---|---|---|---|---|
| `configs/dev.yaml` | esm2_t12_35M | lora r8 q,v; d 256; entmax; hier_query, mcm, taxon, ia_weight; max_len 1022; 16,384 tok/batch, max 32; lr 3e-4, head 1e-3, warmup 0.06; taxon_dropout 0.1 | 15 / 3 | the production run `dev_35m_lora` |
| `configs/final_3b.yaml` | esm2_t36_3B | same + `grad_ckpt: true`, lr 2e-4, explicit `lora_targets` | 5 / 2 | stopped after epoch 1 |
| `configs/pilot_baseline.yaml` | 35M | dev + train_subset 20k, val_subset 4k | 4 / 2 | |
| `configs/pilot_gat.yaml` | 35M | + use_go_gat, 2 layers, 4 heads | 4 / 2 | |
| `configs/pilot_slots.yaml` | 35M | pooling slots, n_slots 8 | 4 / 2 | |
The GO-GPT bench run (`runs/gogptbench_35m_lora/config.yaml`) = dev.yaml with epochs 10, patience 2 and was killed
after 4 epochs by `tools/finish_fast.sh`.

### 3.3 Epoch budget actually used (from `runs/*/train_log.jsonl`)
| run | epochs | best ep | best val Fw (MF/BP/CC) | s/epoch |
|---|---|---|---|---|
| dev_35m_lora | 12 (early stop) | 8 | 0.5270 (0.606/0.384/0.591) | 647 |
| dev_35m_lora_all_r16 | 10 | 6 | 0.5253 | 549 |
| dev_35m_dora | 6 (killed) | 5 | 0.5189 | 853 |
| base_35m_frozen_mlp | 15 | 12 | 0.4885 | 258 |
| abl_mean_pool / no_hier_query / no_mcm / softmax | 15 / 12 / 11 / 7 | 12 / 11 / 10 / 6 | 0.5218 / 0.5231 / 0.5152 / 0.5224 | 700-1,170 |
| abl_no_taxon / no_ia_weight | 4 (killed) | 3 | 0.4956 / 0.5095 | ~1,870 |
| cv_35m_fold0-4 | 4 (killed) | 3 | 0.509-0.518 | 1,170-1,920 |
| **gogptbench_35m_lora** | **4 (killed)** | 3 | **0.5447 (0.637/0.405/0.592)**, still rising | 2,446 |
| final_3b | 1 (killed) | 0 | 0.4848 | 8,749 |
| pilots (20k) | 4 | 3 | 0.476 / GAT 0.458 / slots 0.467 | 106-133 |

### 3.4 `tools/*.sh`
- `run_ablations.sh <adapter> <gpu>` (25 lines): 6 ablations (3 in parallel per wave) then 5-fold CV
  (`epochs=8 patience=2`) on one GPU; `run()` skips runs with an existing `.done`.
- `run_rest.sh <gpu>` (20 lines): second ablation wave + CV folds 0-4.
- `finish_fast.sh` (21 lines): enforces a 4-epoch cap by `pkill`-ing training processes (`cap()` polls the log),
  dumps predictions (`dump_preds`), runs `stack` (eval_main), `cv_summary`, `bench_gogpt`. Source of the 4-epoch
  truncation.
- `watch_done.sh` (13 lines): emits `AGENT_LOOP_WAKE_cafa6` per new `logs/*.done`.
- Binaries: `tools/diamond` (2.1.10), `tools/mmseqs/` (8cc5ce3, AVX2), git-ignored.

---------------------------------------------------------------------------------------------------------

## 4. Homology, stacking, grounding

### 4.1 `src/cafa6/homology.py` (61 lines)
- `diamond_search(db_df, q_df, threads=64, max_hits=50, exclude_self=True)` (L23-37): temp dir under PROC, `diamond
  makedb`, `blastp --more-sensitive -e 1e-3 -k 50`, outfmt `qseqid sseqid pident bitscore evalue qcovhsp`; drops
  `q == s`. No `--query-cover`/`--subject-cover` filter, no `--ultra-sensitive`, no iterative search.
- `diamond_score(hits, q_ids, db_ids, Y_db)` (L40-51): `S = (W @ Y) / rowsum(W)`, W = bitscores. **Plain weighted
  mean over all <=50 hits**; no identity sharpening, no top-k, no coverage weighting. This is the DeepGOPlus
  DiamondScore and is the root cause of the near-neighbour dilution (section 0).
- `max_identity(hits, q_ids)` (L54-57); `identity_bin(pid)` (L60-61): bins no_hit / <30 / 30-50 / >=50.

### 4.2 `src/cafa6/baselines.py` (69 lines)
Naive frequency (train split), DIAMOND kNN and `0.8 * kNN + 0.2 * naive` (naive if no hit). Writes
`data/proc/diamond_hits_valtest.parquet` (284,565 rows, 12,884 queries), `diamond_{val,test}.npy` (float16 n x 6,276),
`identity.parquet[id, split, max_pident, id_bin]`, `runs/baselines/{metrics.tsv, summary.json}`.
CAFA test: naive 0.316, DIAMOND kNN 0.429, DIAMOND+naive 0.461.

### 4.3 `src/cafa6/stack.py` (160 lines) - homology gate + test evaluation
- `mcm_numpy(p, anc)` (L25-30): GPU max over descendants.
- `features(ident, ids)` (L33-36): `[max_pident/100, has_hit, 1]` (3 features, protein-level).
- `fit_gate(p_model, p_knn, Y, X, aspect, steps=300)` (L39-56): per aspect, w in R^3, Adam lr 0.05, BCE of
  `g * p_knn + (1-g) * p_model` on validation. Gate is **per aspect, not per term**, and ignores neighbour count,
  coverage, label agreement among hits, and term depth.
- `apply_gate` (L59-65), then MCM.
- `main` (L76-156): gate on val, test eval by cafaeval, identity strata, IA terciles, worst-decile error analysis
  (`per_protein_f1` at tau=0.3), worst-taxa enrichment. Outputs `runs/<out>/{summary_all.tsv, gates.json, strata.tsv,
  error_*.tsv, metrics.tsv}`.
- Fitted gates: CAFA g ~0.1 without hit, ~0.3-0.4 at 100% id. GO-GPT bench (`runs/bench_gogpt/gates.json`):
  F [1.56, 0.24, -1.14], P [2.43, -0.05, -2.07], C [2.98, -0.12, -2.40] -> g(100% id) = 0.66 / 0.58 / 0.61.

### 4.4 `src/cafa6/comparative.py` (243 lines) - ortholog pilot
`REFERENCE_TAXA` (L25-31): 10 proteomes (9606, 10090, 3702, 559292, 10116, 284812, 83333, 7227, 6239, 83332).
`build_reference_db` (L34-36, train-split proteins from those taxa), `cross_species_hits` (L39-44, drops same-taxon
hits). Writes `data/proc/ortho_{val,test}.npy` (float16) and `runs/pilot_comparative_genomics/`. Result: -0.004 vs
DIAMOND+naive (subset of the same DB). Low reuse value.

### 4.5 `src/cafa6/grounding_data.py` (905 lines) + `data/grounding/` (137 MB)
Steps (`--step all|static|uniprot|goa|readme`), idempotent, with disk guard `MIN_FREE = 5 GB` (L57; **currently below
it**, so any re-run would abort):
- `static`: `interpro2go.tsv` (30,122), `ec2go.tsv` (4,831), `rhea2go.tsv` (7,875), `rhea_participants.tsv` (87,619),
  `go2chebi.tsv` (27,398; from go-plus logical definitions), `chebi.tsv` (15,251 with SMILES), `ec.tsv` (8,472),
  `go2ligand.tsv` (73,350 MF term-ligand rows, `ubiquitous` flag for ChEBI in >=300 Rhea reactions). BRENDA stub only.
- `uniprot`: `uniprot_features.parquet` (285,414 accessions: CAFA train/val/test, GO-GPT train/val/test, Kaggle
  superset) with `[id, interpro, pfam, ec, binding_chebi, cofactor_chebi, taxon, found, primary_acc,
  feature_source]`. InterPro coverage: CAFA 98%, GO-GPT train 96%, GO-GPT test 98% (only 63% via UniProtKB, rest via
  UniParc MD5), Kaggle 95%. **Leakage rule** (README): ec/binding/cofactor are curated -> training/reference proteins
  only; interpro/pfam are computed -> usable for any protein.
- `goa`: `goa_reference.parquet` (2,349,235 rows `[id, go, evidence, date, taxon, aspect]`, 10 proteomes, all evidence
  codes incl. IEA, NOT rows dropped). Has annotation **dates** -> usable to build a temporal validation split.
- All external files are **current (2026) releases**; interpro2go postdates both benchmarks' cutoffs (REPORT 8.4).

### 4.6 `src/cafa6/grounded.py` (390 lines) - grounded calibrator (HiGO-v2)
- `Closure(obo, vocab_terms)` (L50-78): term -> vocab closure indices; `.matrix(go_terms)`.
- `Grounding(vocab, closure, goa_date_max=None, exclude_ids=())` (L103-182):
  `IP` (interpro x T, interpro2go closed), `LigT` (T x ChEBI), `GOA` (goa protein x T, date filter, evaluation
  proteins excluded); `domains(ids, interpro_lists=None)` -> D; `orthologs(hits, ids, q_taxon, db_df, Y_db)` -> O
  (diamond_score over reference-taxa hits, same species removed), G (GOA of hits, max identity per term);
  `ligands(D, h, ids)` -> L_dom, L_hit.
- `components(...)` (L191-208): 13 features per (protein, term): `lp, lk, pid, hit, D, O, G, L, haslig, lfreq, lp_pid,
  lk_pid, D_lp` (`COMPONENTS` L44).
- `fit_calibrator(F, y, steps=400, lr=0.05)` (L211-223): standardised logistic regression (13 weights + bias) per
  aspect, shared across all terms of the aspect. `calibrate(...)` (L231-249), then MCM.
- `main` (L268-386): variants grounded / no_domain / no_ortholog / no_ligand / gate_only; stability over dev + 5 CV
  models. Outputs `runs/grounded{,_gogpt}/{calibrator.json, gate.json, summary.tsv, metrics.tsv, violations.json,
  stability.tsv, evidence_D_{val,test}.npz, has_ligand.npy, pred_{val,test}.npy}`.
- Fitted weights CAFA (standardised): D 1.32-1.50, O 0.76-0.81, lp 0.34-0.50, **lk 0.02-0.04**. GO-GPT bench: G
  (ortholog GOA) 0.69-0.81 for MF/BP (temporal mismatch), lk 0.03-0.13.
- Results: CAFA test 0.5408 (gate 0.5277, no domains 0.5187); GO-GPT test 0.5503 (< gate 0.5759).

### 4.7 `src/cafa6/predict_grounded.py` (367 lines)
`Explainer` (L45-119) loads `runs/grounded/calibrator.json` + per-aspect tau from `runs/grounded/metrics.tsv`;
`describe()` builds per-term records (parents path, domains, orthologs, ligands, top residues).
Modes: `--split test` (JSONL/TSV), `--fasta` (UniParc MD5 lookup L157-171, EBI InterProScan fallback L174-193, DIAMOND
vs all labelled proteins, **truncates to `max_len` without windows**, L281), `--submission` (Kaggle superset;
`runs/grounded/submission/submission.tsv.gz`, 84,991,864 rows). In the fasta/submission modes the kNN DB is all
82,038 labelled proteins, while the gate/calibrator were fit with a train-only DB (distribution shift in `lk`/`pid`).

### 4.8 `src/cafa6/submit.py` (87 lines)
Gate-only Kaggle submission (`runs/dev_35m_lora/submission/`): gate fit on val, kNN on all labelled proteins
(`diamond_hits_kaggle_test.parquet`: 4,163,307 rows, 198,155 queries with a hit = 88.3%), top 500 / score >= 0.02.

### 4.9 Cached homology artefacts (reusable, no recompute needed)
| file | shape / rows | content |
|---|---|---|
| `data/proc/diamond_hits_valtest.parquet` | 284,565 | CAFA val+test vs train split |
| `data/proc/diamond_hits_kaggle_test.parquet` | 4,163,307 | Kaggle superset vs all 82,038 labelled |
| `data/proc/diamond_{val,test}.npy` | (8,377 / 8,058, 6,276) f16 | bits-weighted kNN |
| `data/proc/ortho_{val,test}.npy` | same | reference-panel kNN |
| `data/proc/identity.parquet` | 16,435 | max_pident, id_bin |
| `data/proc_gogpt/diamond_hits_valtest.parquet` | 590,952 (20,483 queries) | GO-GPT val+test vs GO-GPT train |
| `data/proc_gogpt/diamond_{val,test}.npy` | (13,305 / 8,630, 7,901) f16 | |
| `data/proc_gogpt/identity.parquet` | 21,935 | max_pident (no bin) |
Missing: train-vs-train hits (needed to train a retrieval-aware model with neighbour features on training proteins).

---------------------------------------------------------------------------------------------------------

## 5. Evaluation

### 5.1 `src/cafa6/evaluate.py` (76 lines)
- `OBO = data/raw/Train/go-basic.obo`, `IA = data/raw/IA.tsv` (CAFA-6) defaults (L14-15).
- `dense_to_long(scores, ids, terms, top_k=1500, min_score=0.01)` (L20-30): rounds scores to 3 decimals.
- `evaluate(preds, gt_file, ids_subset=None, n_cpu=16, out_dir=None, obo=OBO, ia=IA)` (L33-68): writes temp
  pred/gt TSVs, `cafaeval.evaluation.cafa_eval(obo, pred_dir, gt, ia=ia, norm="cafa", prop="max", n_cpu)`; returns
  rows `[method, aspect, Fmax_w, tau_w, Fmax, Smin, cov, P_w, R_w, P, R]`. `summarize` (L71-76) -> MF/BP/CC/mean.

### 5.2 Exact protocol for comparing with GO-GPT (`src/cafa6/bench_gogpt.py`, 97 lines)
- Run with `CAFA6_PROC=data/proc_gogpt`. OBO = `runs/gogpt_cafa5/ref/go-basic_gogpt.obo` (GO 2023-01-01),
  IA = `runs/gogpt_cafa5/ref/IA.txt` (CAFA-5) (L23-24).
- GT = `data/proc_gogpt/gt_test.tsv` (propagated; cafaeval `prop=max`, `norm=cafa`).
- GO-GPT predictions: `runs/gogpt_cafa5/preds_shard{0,1,2}.jsonl` (beam) and `runs/gogpt_cafa5/greedy/preds_shard0.jsonl`
  (`{"protein_id", "organism", "MF": [...], "BP": [...], "CC": [...]}`), scored as binary (score 1.0) via
  `gogpt_preds()` (L27-34).
- Subsets (L77-82): `S3_all` (all 8,630; paper Table S3), `S6_noknowledge` (drop 471 ids in
  `runs/gogpt_cafa5/ref/common_proteins.txt`; 8,159; Tables S6-S9), identity bins.
- Gate fit on GO-GPT val, applied to test. Outputs `runs/bench_gogpt/{summary.tsv, gates.json, S3_all/metrics.tsv}`.
- Headline (S3_all, mean Fw): GO-GPT beam 0.6866 (MF 0.714 / BP 0.617 / CC 0.728), greedy 0.6864, HiGO+gate 0.5759,
  DIAMOND+naive 0.5678, HiGO 0.4547, naive 0.2438. S6: GO-GPT 0.7021, HiGO+gate 0.5849.
- Identity bins (mean Fw, GO-GPT beam vs HiGO+gate vs DIAMOND+naive): >=50% (7,585) 0.730 / 0.599 / 0.595;
  30-50% (713) 0.440 / 0.480 / 0.457; <30% (82) 0.395 / 0.461 / 0.454; no hit (250) 0.374 / 0.359 / 0.203.
- Reference numbers: paper 0.651; shipped `go_pred` 0.649; public checkpoint 0.687 (BP +0.08 over paper; REPORT
  suspects a different/more-data checkpoint). **A de novo model should report against all three**.

### 5.3 `benchmarks/gogpt_eval.py` (393 lines) - running GO-GPT itself
- Imports GO-GPT from **`/tmp/BioReason-Pro-main/gogpt/src`** (L47; volatile /tmp, 31 MB). Weights from HF cache
  `wanglab/gogpt` (`model.ckpt` 3.18 GB; config: 12 layers, 12 heads, n_embd 900, block 2,048, protein_embedding_dim
  2,560, `embed_model_path facebook/esm2_t36_3B_UR50D`, `protein_layer_index 30`, gated attention; tokenizer
  vocab 10,303, organism vocab 201).
- `PROTOCOLS` (L53-57): S3_all, S6_noknowledge, cafa6_ia (CAFA-6 obo + IA).
- `install_protein_cache` (L123-135), `_protein_stream` (L73-90), `_go_stream` (L93-120): exact KV cache of the
  protein stream (verified identical outputs).
- `esm_layer` (L143-162): stops ESM2-3B at layer 30 via pre-hook.
- `generate` (L185-206): **protein truncated to 1,024 tokens** (no windows), beam 5, length penalty 0.3, <=300 tokens
  per aspect; greedy option.
- `organism_resolver` (L210-224): GO-GPT organism vocabulary is the top-200 training species; unknown -> 0.
- `cmd_gen` (L284-313) sharded generation (`--shard/--n_shards/--batch 16/--sample`), `cmd_score` (L333-370) uses the
  `cafaeval` CLI (`-prop max -norm cafa -threads 16`).
- ~1.9 h for 8,630 proteins on one H100 with 3 processes.
- Other GO-GPT outputs: `runs/gogpt_cafa6/` (1,500 CAFA-6 test proteins zero-shot, 0.897 mean Fw, leaked: 96% of
  CAFA-6 labelled proteins are in GO-GPT's training data), `runs/gogpt_released/` (shipped go_pred scores).
- `benchmarks/globin_demo.fasta`: P02100-identical 147-aa globin.

### 5.4 `src/cafa6/prf_table.py` (177 lines)
IA-weighted P/R/F1 at each model's own tau_w and unweighted P/R/Fmax, for both benches; `--extra NAME=RUNDIR`
adds rows; writes `runs/prf/{cafa6_test.tsv, gogpt_test.tsv, tables.md}`. Re-spawns itself with the correct
`CAFA6_PROC` (L163-172). Directly reusable to score a new model against GO-GPT with one flag.

### 5.5 `src/cafa6/cv_summary.py` (38 lines), `src/cafa6/dump_preds.py` (34 lines)
`cv_summary`: best val Fw within `--max_epochs 4` per fold -> `runs/cv_summary/{folds,stats}.tsv` (0.5135 +- 0.0038).
`dump_preds --run X`: reload weights via `explain.load_run`, write `pred_{val,test}.npy` (tokens/batch doubled).

---------------------------------------------------------------------------------------------------------

## 6. Explainability, SAE, pilots

### 6.1 `src/cafa6/explain.py` (206 lines)
- `load_run(name, device="cuda") -> (model, cfg, vocab, tmap, tok)` (L29-43): canonical way to rebuild any HiGO run
  from `config.yaml` + `taxa.json` + `trainable_weights.pt` (`strict=False`). Used by submit, dump_preds,
  predict_grounded.
- `forward_one` (L46-56, single window, test proteins limited to 50 <= len <= max_len, L110), `masked_prob(mode=
  "input"|"pool")` (L59-74), `interpro_regions(acc)` (L80-94, live EBI API).
- Results (`runs/dev_35m_lora/explain/summary.json`): 511 cases, 39% residues non-zero, top-10% residues carry 76%
  mass; pool deletion drop 0.067 vs 0.0005 random (p=0.007); input masking no effect; InterPro median enrichment
  1.13; residue-1 (Met) artefact.
- Reuse: evidence-map tooling is a selling point vs GO-GPT (which has none); masking residue 1 and the special tokens
  from pooling is a one-line fix.

### 6.2 `src/cafa6/sae.py` (418 lines)
`SAE(d_in, d_hidden)` (L117-142, tied init, unit-norm decoder), `train_sae` (L145-170), `go_purity` (L196-236,
with permutation null), `contiguity` (L239-277). Run: 256->4,096 ReLU+L1 3e-3, 4,000 steps, 3,000 val proteins
(1.26 M residues, RAM only). Not sparse (371 latents >20% firing). Low value for accuracy; keep for interpretability.

### 6.3 Pilots (`configs/pilot_*.yaml`, `runs/pilot_*`, `runs/pilot_module_report.md`)
20k train / 4k val, 4 epochs, seed 0, val Fw: baseline 0.4761, GAT 0.4582 (-0.018, init bug), slots 0.4665 (-0.010,
inconclusive). Ortholog channel -0.004. Nothing to carry over except the fixed-GAT idea.

### 6.4 Ablations worth remembering (test, + gate): no taxon -0.020, no IA weight -0.009 (4-epoch confound),
no MCM -0.004, pooling type / ancestor sharing ~0. LoRA vs frozen +0.022. DoRA / all-linear r16 no gain on 35M.

---------------------------------------------------------------------------------------------------------

## 7. Hardware, environment, cached weights

- GPUs: 2x NVIDIA H100 NVL 95.8 GB (GPU0 had 1.9 GB in use at audit time, 0% util). 128 CPUs, 251 GB RAM.
- **Disk: `/` 1.4 TB, 4.1 GB free (100% used)**. `grounding_data.py` refuses to run below 5 GB. NOTES Step 1 lists
  the consequences (no residue caches, ESM2-650M never downloaded). `cafa6/runs` = 6.2 GB (`dev_35m_lora` 1.5 GB incl.
  1.1 GB uncompressed submission, `grounded` 677 MB incl. 382 MB submission; many 100-200 MB `pred_*.npy`); data/ =
  1.4 GB. Candidate space to reclaim (user decision): `runs/dev_35m_lora/submission/submission.tsv` (1.1 GB, gz copy
  exists), duplicated pred_*.npy of ablations, `~/.cache/pip` 2.9 GB, unrelated HF models (see below).
- HF cache `~/.cache/huggingface` (29 GB):
  | model | on disk | relevance |
  |---|---|---|
  | `facebook/esm2_t6_8M_UR50D` | 30 MB | smoke tests |
  | `facebook/esm2_t12_35M_UR50D` | 130 MB (safetensors) | current backbone |
  | `facebook/esm2_t36_3B_UR50D` | 11.4 GB (`pytorch_model-0000{1,2}-of-00002.bin`; second snapshot has only a safetensors index, no weights) | GO-GPT's backbone; ready to use |
  | `biohub/esm3-sm-open-v1` | 5.2 GB (`esm3_sm_open_v1.pth` 2.8 GB + function decoder 1.3 GB + structure enc/dec) | 1.4B multimodal PLM, can take structure tokens; `esm` 3.4.1 pip package is installed |
  | `wanglab/gogpt` | 3.2 GB `model.ckpt` + tokenizers | competitor, also usable as teacher / feature source |
  | `wanglab/bioreason-pro-rl` | 4.5 MB (only `protein_projection.pt`, `go_projection.pt`, `go_embedding.pt`, tokenizer, go-basic.obo; no LLM weights) | GO embeddings possibly reusable |
  | datasets `wanglab/bioreason-pro-test-data` | 15.5 MB parquet | test set copy |
  | ESM2-650M, ESM2-150M, ProtT5, Ankh, SaProt, ESM-C | **not present** | |
  Unrelated (reclaimable if allowed): chronos, MOMENT 1.3 GB, Geneformer 1.2 GB, Bio_ClinicalBERT 3.0 GB, Qwen2.5-0.5B,
  vit, mpnet, distil*, etc. `~/.cache/torch/hub/checkpoints`: blip2/eva_vit_g/resnet152 (2.6 GB, unrelated).
- Outside cafa6 (not audited in detail): `PFA/models/{go_graph_encoder.py, protein_encoder.py, protein_llm.py,
  go-basic.obo}` (BioReason-style GO graph encoder copy), `PFA/embeddings_P02100`, `PFA/external`.
- `/tmp` holds `BioReason-Pro-main` (GO-GPT + bioreason2 source, needed by gogpt_eval.py) and `gogpt_train/` (GO-GPT
  training parquet, 148 MB). Both can disappear on reboot; copy them into the repo or the HF cache if kept.

---------------------------------------------------------------------------------------------------------

## 8. GO-GPT training data: unused fields (`/tmp/gogpt_train/data/*.parquet`, `runs/gogpt_cafa5/ref/test.parquet`)
Columns (train, 120,143 rows before filtering): `protein_id, protein_names, protein_function, organism, length,
subcellular_location, sequence, go_ids, go_bp, go_mf, go_cc, structure_path, string_id, interaction_partners,
interpro_ids, interpro_location, ppi_formatted`. Test adds `interpro_formatted, go_pred`.
| field | train non-null | test non-null | note |
|---|---|---|---|
| interpro_ids + interpro_location (JSON spans) | 95% / 100% | 83% / 100% | spans allow domain-aware pooling without InterProScan |
| structure_path (AFDB model id, e.g. `AF-C0HLS4-F1-model_v6.pdb`) | 94% | 73% | structures not downloaded (disk) |
| string_id / interaction_partners | 75% / 66% | - | PPI graph for BP |
| ppi_formatted | 66% | 14% | BioReason-Pro input |
| protein_names | 100% | 100% | curated text, high leakage risk for test |
| protein_function / subcellular_location | 56% / 67% | 24% / 51% | curated text, leakage risk |
GO-GPT itself uses only sequence (ESM2-3B layer 30) + organism; BioReason-Pro adds InterPro, PPI and GO-GPT output.

---------------------------------------------------------------------------------------------------------

## 9. Weaknesses, bugs, technical debt (file:line)

Accuracy-limiting
1. kNN dilution: `homology.diamond_score` L40-51 averages <=50 hits by bitscore; best-hit label copy is far stronger
   at >=95% id (section 0). Gate (`stack.fit_gate` L39-56) is per aspect with 3 protein-level features; calibrator
   (`grounded.fit_calibrator` L211-223) is one logistic model per aspect shared by all terms.
2. kNN DB excludes the validation split at test time in both benches (prepare_gogpt.py L85-91; stack/bench use
   train-only npy). For the final model, val labels (13,305 GO-GPT proteins) are wasted as neighbours.
3. Non-temporal validation everywhere (prepare.py L75-84 random clusters; GO-GPT val random). Calibrator learns
   GOA overweighting (grounded_gogpt G weight 0.69-0.81).
4. 4-epoch truncation of the GO-GPT-bench run (`tools/finish_fast.sh` L14-19) inside a 10-epoch cosine schedule,
   i.e. LR still near peak when stopped.
5. Backbone: last layer of ESM2-35M only (`model.py` L94); GO-GPT uses layer 30/36 of ESM2-3B.
6. Vocab min_count 30 (prepare.py L44, prepare_gogpt.py L35): 6-10% of test IA mass unreachable; MF worst.
7. Loss: no handling of extreme label imbalance (model.py L141-155); IA weight clip 5.
8. Taxon: embedding for >=50-protein taxa only (train.py L158); GO-GPT bench uses organism *strings* (45% of the
   GO-GPT test set is zebrafish, 1,675 rat, 1,648 human); no taxonomic lineage sharing.
9. Domain evidence only post-hoc (calibrator), and via current interpro2go (temporal leakage on both benches).

Bugs / correctness debt
10. `go_edges.npy` only direct parents inside vocab (prepare.py L110), not nearest vocab ancestors as commented; no
    edges file for proc_gogpt.
11. `GOGraphEncoder.forward` (go_graph.py L38-41) LayerNorm+GELU replaces queries (documented).
12. `fast_fmax_w` (train.py L80-104) ignores out-of-vocab GT -> optimistic early-stopping metric; thresholds only
    0.01-0.99 in 0.01 steps.
13. `grad_accum` leftover micro-batches not stepped (train.py L245-247).
14. Weight decay applied to LayerNorm/bias/embeddings (train.py L223-224).
15. Validation split used for early stopping and for gate/calibrator fitting (double use).
16. `predict_grounded --fasta` truncates to max_len (L281) instead of windowing; `explain.forward_one` single window.
17. Gate/calibrator fit with a train-only kNN DB but applied with an all-labelled DB in submit.py L69 /
    predict_grounded.py L327 (feature shift).
18. Residue-1 (Met) attention artefact; `<cls>`-adjacent position not masked from pooling.
19. No resume / optimizer state in checkpoints; runs killed by `pkill` lose schedule state.
20. External dependencies in `/tmp` (BioReason-Pro source, GO-GPT train parquet).
21. `CAFA6_PROC` import-time global (data.py L15) makes multi-bench code awkward.
22. predict loop applies MCM per protein in Python (train.py L140-141) - slow for 224k proteins.

---------------------------------------------------------------------------------------------------------

## 10. Reusable assets (keep)
- Data prep (`data.py`, `prepare.py`, `prepare_gogpt.py`), Batcher with crop/windows, cafaeval wrapper and the exact
  GO-GPT protocol (`bench_gogpt.py`, `prf_table.py`, `benchmarks/gogpt_eval.py` incl. its fast GO-GPT generator for
  generating GO-GPT predictions on any new split, e.g. a temporal val split, or as distillation targets).
- HiGO head (hierarchical queries, entmax evidence pooling, MCM/MCLoss) - accurate enough and interpretable; the
  hierarchy-consistent output (0 violations) is a differentiator vs GO-GPT's free-text token sets.
- DIAMOND/MMseqs2 binaries and cached hits; grounding resources (InterPro/Pfam per accession, GOA with dates,
  ligand map); identity strata code.
- Trained checkpoints: `dev_35m_lora`, 5 CV folds, ablations, `gogptbench_35m_lora` (4 epochs), `final_3b` (1 epoch,
  21.5 MB LoRA+head) - the 3B one can warm-start a longer 3B run.

---------------------------------------------------------------------------------------------------------

## 11. Top 10 leverage points for beating GO-GPT on its own temporal benchmark (ordered by expected gain / cost)

1. **Identity-aware retrieval head** (cost: minutes, no GPU). Replace bits-weighted mean with e.g.
   `w_j = exp((pident_j - max_pident)/T) * qcov_j` (or top-1/top-3 label copy above ~90-95% id), per-term or per-depth
   gates with features [max id, id of best hit carrying the term, #hits carrying it, fraction of hits, coverage].
   Top-1 copy alone reaches IA-weighted recall ~0.87-0.92 / precision ~0.71-0.74 on the 51% of the GO-GPT test with
   >=95% hits, which is where GO-GPT's 0.730 lead comes from.
2. **Learned retrieval-augmented model instead of a post-hoc gate**: feed the neighbour label vector(s) + identity into
   the head (term-wise fusion), trained on train-vs-train hits with self/same-cluster exclusion and neighbour
   dropout. Needs a train-vs-train DIAMOND run (~minutes with 64 threads; output a few hundred MB, mind disk).
3. **Temporal validation** for model selection and calibration: build from GOA dates (`goa_reference.parquet`) or by
   holding out the latest annotations of GO-GPT's training proteins; or fit the gate on a temporal pseudo-split.
   Directly fixes the calibrator failure on GO-GPT test.
4. **Bigger backbone, read the right layer**: ESM2-3B is cached; use an intermediate layer (GO-GPT: 30/36) or a
   learned layer mix; consider ESM3-sm-open (cached) with structure tokens. Cache per-protein pooled embeddings only
   (141,684 x 2,560 fp16 = 0.73 GB) for fast head experiments; residue caches do not fit on disk.
5. **Train to convergence with multi-GPU**: add DDP (2x H100) and bf16 + flash/sdpa attention; 10-15 epochs for 35M,
   >=3-5 epochs for 3B (2.4 h/epoch single GPU with checkpointing). Fix scheduler horizon to the planned budget; use
   train+val for the final fit with a fixed epoch count.
6. **Larger vocabulary** (min_count 5-10 or GO-GPT's 10.3k terms; IA ceiling now 93.8%, MF 88.7%) with zero/few-shot
   term embeddings (GO definitions/names from `go.obo`/go-plus, BioReason `go_embedding.pt`) plus the ancestor-mean
   query; requires the term-chunked attention memory fix (section 2.1).
7. **Domain input inside the model**: InterPro ids (+ spans) as tokens/embeddings or as pooling priors (available for
   95% train / 83%+ test in GO-GPT data, 98% via UniParc). Domains gave the only post-hoc gain on CAFA (+0.013, BP
   +0.033). Use InterPro *ids* (computed) rather than current interpro2go mappings to limit temporal leakage, or pin
   an interpro2go release from before Nov 2022.
8. **Better organism modelling**: NCBI lineage embeddings (map GO-GPT organism strings via `uniprot_features.taxon`),
   shared across the 3,130 organisms; taxon was the largest ablation effect (-0.020 to -0.032).
9. **Ensembling and loss**: 6 HiGO models already exist (dev + 5 CV); ensembling + ASL/focal or positive re-weighting,
   fixed gated-residual GO-GAT. Cheap, +0.005-0.015 expected.
10. **PPI/structure context for BP** (lower priority, disk-bound): STRING partners (75% train, but `ppi_formatted`
    only 14% test) and AFDB structures (94% train / 73% test ids, not downloaded). Use only after 1-5.

Caveat for claims: the public GO-GPT checkpoint (0.687) scores 0.035 above the paper (0.651) and above its own
shipped predictions (0.649), mostly in BP, which suggests it was trained on more/later data. A de novo model should
report against all three numbers, on S3_all and S6_noknowledge, by identity bin, with `bench_gogpt.py`/`prf_table.py`.
