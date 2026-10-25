# fresh24 look #1: how each prediction file was made

Registration: `runs/fresh24_looks.md` (2026-10-19 18:40:00). Inputs: `data/fresh24/fresh24.fasta` (24,141 sequences)
with the NCBI taxon of `proteins.tsv`. Every file is `runs/fresh24/preds/<name>.npz` in the compare_paper format and
holds all 24,141 proteins unless noted. Runners: `src/cafa6/predict_fresh24.py`; external wrappers and converters
unchanged from `runs/compare_paper/extra_models/`. Step-by-step log with timings and failures:
`logs/fresh24_look1.log`. Nothing was fitted, tuned or selected on fresh24.

For the slow methods the input was split into `runs/fresh24/inputs/fresh24_P.fasta` (the 11,231 FR24-P proteins) and
`fresh24_rest.fasta` (the other 12,910), and FR24-P was run first. The split changes only the order of work, except
for SPROF-GO, whose label diffusion connects the query proteins of one call (see below).

## Ontologue and its parts
- `ontologue`: `cafa6.predict_r2.Predictor` as the web app builds it (default `data/serve` references = GO-GPT
  train + val, HiGO on cuda under bf16 autocast, twin stacker on CPU), called through `Predictor.predict` on chunks of
  6,100 proteins (GPU 0, 196 s for all 24,141 including model loading). Organism = NCBI taxon from `proteins.tsv`,
  resolved by `Predictor.resolve_organism` (what an `OX=` header gives).
- `ontologue_knn` (step 1a): the sharpened kNN prior (T 5, k 50, aspect-normalised, train + val references) and
  `ontologue_higo` (step 1b): the HiGO head's top 500 terms per protein, both captured inside the same `predict()`
  calls.
- Spot-check (`runs/fresh24/ontologue_spotcheck.json`): 20 random proteins (seed 0) through `predict()` one protein
  per request. 19 agree with the chunked rows to 1e-4 (the 4-decimal rounding of the `predict()` output). For
  Q4UWF4 the DIAMOND hits are identical, but its HiGO bf16 scores differ by 7.3e-4 between a one-protein batch and
  the 6,100-protein chunk (padding and batch numerics). That moves one term across the HiGO top-300 cut (316 vs 317
  terms; max difference 0.064 on one term). Multi-protein requests to the served predictor batch the same way.

## Homology baselines (definitions of `compare_paper.cmd_baselines`)
DIAMOND 2.1.10 `--more-sensitive -e 1e-3 -k 51 --dbsize 66,295,805` (GO-GPT training letters), self hits removed.
fresh24 was searched against the 119,749 training proteins (hits for 21,292 queries) and the 133,054 train + val
proteins (21,417 queries; same hit count as `data/fresh24/work/hits.parquet`).
- `blast_knn`: best hit by percent identity among training references (ties broken by bitscore), all propagated terms
  of the hit, score = identity / 100. `blast_knn_trainval`: the same rule with train + val references.
- `diamondscore`: bitscore-weighted label mean over the top 50 training hits.
- `naive`: per-aspect term frequency among the training proteins annotated in that aspect (terms >= 0.005).

## GO-GPT released checkpoint (`benchmarks/gogpt_eval.py gen --fasta ... --taxa protein_taxon.tsv --taxon_names taxon_names.tsv`)
- Organism = the UniProt organism name of `proteins.tsv`, mapped by `organism_resolver` (exact name, else the
  shortest vocabulary entry with the same name before " ("; else unknown, id 0). The organism is in GO-GPT's
  vocabulary for 68.3% of FR24-P (97.9% on the GO-GPT test) and 99.7% of the other proteins (mostly human).
- `gogpt_sample10`: the settings of `runs/gogpt_sample10/settings.json` (10 samples per aspect, T 0.7, top-k 20,
  300 new tokens, fp32, batch 128, token budget 66,000, seed 0). Score = fraction of the 10 samples containing the
  term. FR24-P on GPU 0 (2,627 s), the rest on GPU 1 (2,071 s), each one uninterrupted launch.
- `gogpt_beam`: the released predictor's beam (beam 5, length penalty 0.3), batch 8 as in `runs/gogpt_cafa5`,
  score 1.0. FR24-P in 3 shards, the rest in 6 shards, over both GPUs. Shards were stopped and resumed from their
  per-batch checkpoints when GPU memory ran short (a watchdog kept GPU 1, which hosts the web app, below 89 GB), and
  one FR24-P shard ended on a CUDA out-of-memory error at its longest proteins on 2026-10-19; all were resumed on
  2026-10-24 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator only; beam decoding is deterministic).

## External predictors (official code and weights fetched again into `/dev/shm/fresh24_xm`)
| name | source | run |
|---|---|---|
| `sprof_go` | SPROF-GO GitHub repo models; ProtT5-XL-U50 from HF `Rostlab/prot_t5_xl_uniref50` | `script/predict.py --gpu --feat_bs 1` through `run_sprof.py`, GPU 0, about 18 min per chunk |
| `deepgoplus` | `data-1.0.10.tar.gz` (2022-11-06; Swiss-Prot 2022_04, GO 2022-10-07) | `deepgoplus/main.py -t 0.01` (CNN + DIAMONDScore, alphas MF 0.63 BP 0.62 CC 0.54), CPU, 605 s |
| `proteinfer_e5` | GO random-split CNNs 13703706, 13703742, 13703997, 13704131, 13705631 | `run_proteinfer.py`, mean of the 5 elements, CPU, 1,048 s (FR24-P) and 871 s (rest) in parallel; 24,074 proteins have a score >= 0.01 |
| `deepfri_cnn` | `trained_models.tar.gz` (2023-01-12) | `predict.py --fasta_fn` (DeepCNN-MERGED) through `run_deepfri.py`, 6 CPU runs, 20-27 min |
| `deepgose` | deepgo2 `data.tar.gz` streamed (go.obo 2021-11-16, go-plus.norm, terms, 18 checkpoints); fair-esm ESM2-3B | `predict.py -t 0.01 -d cuda:0` through `run_deepgose.py`; only MF and BP terms are kept and scored, as registered |

- SPROF-GO accepts at most 5,000 sequences per call, so FR24-P and the rest were each split into three seeded random
  chunks (seed 0; 3,744/3,744/3,743 and 4,304/4,303/4,303). Its label diffusion has query-query edges only inside a
  chunk, as for the two halves on the GO-GPT test.
- DeepGO-SE: the first run (2026-10-19, GPU 1) was stopped because the ESM2-3B pass reserved up to 89 GB on the GPU
  that hosts the web app. It was rerun from scratch on 2026-10-24 on GPU 0 with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Scores below 0.01 were not kept (the scorer's lowest threshold). Terms are mapped to GO 2023-01-01 by
  `Ontology.canon`; terms missing from that release are dropped.
- Training-set overlap with fresh24 accessions: DeepGOPlus 1.0.10 training proteins 27 (13 in FR24-P); SPROF-GO
  train/valid/test MF 28, BP 75, CC 98 (FR24-P proteins with ground truth in that aspect: 11, 22, 23). These labels
  predate the 2024-02-29 cutoff, so no leakage flag. The other external models were trained on 2019-2021 data.

## Not scored
- BioReason-Pro RL: its prompt needs UniProt name, function and subcellular-location text, InterPro and PPI text and
  GO-GPT's `go_pred` text per protein (the columns of `wanglab/bioreason-pro-test-data`). The only UniProt text
  available for fresh24 is the current release (2026_03), written after the cutoff and partly from the annotations
  being scored, and no 2024-02 UniProt, InterPro or STRING snapshot is stored here. A leakage-free input set is not
  trivially available.
- InterLabelGO+: excluded by the registration (its released training labels come from a snapshot after the cutoff).
