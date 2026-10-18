# Serving HiGO-R2 on new protein sequences

This covers the final look-3 pipeline (test look #3 in `runs/test_looks.md`; `research_notes/HiGO-R loop/HiGO-R2_model.md`
section 0). It predicts GO terms for sequences that are not in the benchmark, through a command-line tool
(`cafa6.predict_r2`) and a small local web app (`cafa6.webapp`). Both run on CPU.

## Pipeline

For each protein:

1. **DIAMOND** blastp against the 133,054 GO-GPT train + val reference proteins (`data/serve/refs.dmnd`), with the
   settings of the benchmark hits: `--more-sensitive -e 1e-3 -k 51 --dbsize 66295805`. The `--dbsize` value is the
   letter count of the 119,749 training proteins, so e-values and the 1e-3 cut match the benchmark hits. The best HSP
   per pair is kept, then the top 50 subjects by bitscore.
2. **Sharpened kNN prior** `T5_k50_g0_na`, as in `knn_sharp.KNN.prior`: each hit is weighted by
   bits * exp((pident - max pident) / 5), each reference contributes its full propagated GO-GPT label set, and each
   aspect is normalised over the hits annotated in that aspect.
3. **HiGO network** `runs/l3_c0_tv2`: frozen ESM2-35M, entmax per-term pooling, hierarchical queries, MCM and a
   taxon embedding over 67 GO-GPT organisms plus "other". Sequences longer than 1,022 residues are cut into
   half-overlapping windows whose logits are max-pooled, as in `train.predict`.
4. **Twin stacker**: `stacker_tv` model b, refit on all of TV v2 for 800 steps and saved as
   `runs/stacker_tv/l3_c0/model_twin.pt`. It scores the union of the HiGO top 300 and every kNN term. The protein
   features are the max identity, whether there is a hit, the number of hits, the organism group (human, mouse, rat,
   zebrafish or other, taken from the taxon) and whether the top-bitscore hit among the training references comes
   from the same organism.
5. Scores >= 0.01 are max-propagated over GO with `knn_sharp.Ontology.propagate_max`.

## Command line

All commands run from `cafa6/` with `PYTHONPATH=src` and the project interpreter `../bin/python`.

```bash
# once: build data/serve (reference FASTA + DIAMOND db, labels, taxon map); about 2 min
python -m cafa6.predict_r2 --build

# predict
python -m cafa6.predict_r2 --fasta in.fa --taxon 9606 --out out_dir
```

- The organism of each record comes from `OX=<NCBI taxon id>` in its FASTA header (UniProt style). Records without
  it use `--taxon`. For a taxon that is not in the GO-GPT data, or no taxon at all, the record gets the generic
  taxon embedding, organism group "other" and no same-organism hit.
- Accepted residues are the 20 standard amino acids plus X, B, Z, U and O. Whitespace and `*` are removed.
- `out_dir/predictions.tsv` has the columns protein, go_term, score, aspect and term_name, with every propagated
  score >= 0.01.
- `out_dir/report.md` lists, for each protein, the terms at or above the per-aspect thresholds (MF 0.28, BP 0.42,
  CC 0.45) grouped by aspect, and the 5 nearest reference hits with identity, coverage, bitscore and e-value.
- `out_dir/predictions.json` has, per protein: length, taxon, organism, max identity and identity bin, number of hits,
  the 5 nearest references, the above-threshold terms, all scores and per-stage timings.
- `--exclude-self` drops hits to a reference whose accession equals the query id. This is the benchmark convention.
  It is off by default, so a protein already among the references can copy its own labels.
- `--threads` sets the torch threads (default min(16, cores)). `--diamond-threads` defaults to min(32, cores);
  DIAMOND returns identical hits at 16, 32 and 64 threads.

Checks (neither reads test labels):

```bash
python -m cafa6.predict_r2 --verify-stacker   # saved stacker on the stored test features vs pred_test_twin.npz
python -m cafa6.predict_r2 --self-test [--out runs/serve_selftest] [--no-cuda]
```

## Web app

```bash
nohup python -m cafa6.webapp --host 127.0.0.1 --port 7860 > logs/webapp.log 2>&1 &
```

- Open http://127.0.0.1:7860. The page has a textarea for a sequence or FASTA and an organism dropdown (human 9606,
  mouse 10090, rat 10116, zebrafish 7955, fly 7227, yeast 559292, E. coli 83333, Arabidopsis 3702, or another taxon
  id).
- The results page has one table per aspect. GO ids link to QuickGO, and scores are shown as bars. Terms below the
  threshold are folded. The page also lists the nearest reference hits (linked to UniProt) and a TSV download link.
- Limits: at most 20 sequences and 5,000 residues each per request. Requests outside the limits or with characters
  outside the alphabet are rejected (HTTP 422 on the API, an error list on the form).
- JSON API: `POST /api/predict` with `{"fasta": ">id OX=9606\nMKT...", "taxon": 9606}` or
  `{"sequences": [{"id": "p1", "seq": "MKT...", "taxon": 10090}]}` and an optional `min_score`. It returns
  `{"results": [...], "seconds": ...}` with the fields of `predictions.json`.
- `GET /health` returns the status. Requests run one at a time behind a lock.

Example:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"fasta": ">q1 OX=10090\nMSEQ...", "min_score": 0.1}' http://127.0.0.1:7860/api/predict
```

Access:

- The default host 127.0.0.1 accepts connections from the server only.
- `--host 0.0.0.0` serves the local network. There is no authentication, so anyone who can reach the port can submit
  jobs.
- To use it from a laptop without exposing it, keep 127.0.0.1 and tunnel over SSH, then open http://127.0.0.1:7860 on
  the laptop:

  ```bash
  ssh -N -L 7860:127.0.0.1:7860 user@server
  ```

## Data it needs

Needed at runtime:

| path | size | content |
|---|---|---|
| `data/serve/refs.dmnd` | 74 MB | DIAMOND db of the train + val references |
| `data/serve/labels.npz` | 6.8 MB | reference x GO label matrix (133,054 x 43,248, full propagated labels) |
| `data/serve/refs.parquet`, `taxa.tsv`, `meta.json` | 2 MB | reference ids, set and organism; organism to NCBI taxon map; dbsize |
| `runs/l3_c0_tv2/{config.yaml, taxa.json, trainable_weights.pt}` | 8.8 MB | HiGO head |
| `~/.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D` | 130 MB | ESM2-35M (loaded offline) |
| `runs/stacker_tv/l3_c0/model_twin.pt` | 23 MB | twin stacker, refit weights and constructor sizes |
| `runs/gogpt_cafa5/ref/go-basic_gogpt.obo`, `IA.txt` | 31 MB | GO 2023-01-01 (graph, names) and CAFA-5 IA |
| `data/proc_gogpt/{vocab.tsv, go_ancestors.npy, train.parquet}` | 82 MB | HiGO vocabulary, ancestor pairs, organism names |
| `tools/diamond` | 100 MB | DIAMOND 2.1.10 binary |

`data/serve/refs.fa` (72 MB) is used only to build the database.

`--build` also reads two more inputs:

- `external/gogpt_tmp_backup/gogpt_train/data/*.parquet`: the GO-GPT training data, for the full label sets.
- `data/grounding/uniprot_features.parquet`: for the NCBI taxon ids.

The process uses about 1.5 GB of RAM once loaded and needs no GPU. Start-up takes about 8 s.

## Runtime

Measured on this server with 16 torch threads and 32 DIAMOND threads, with the models already loaded:

| request | wall time | DIAMOND | kNN | HiGO | stacker |
|---|---|---|---|---|---|
| 1 protein, 563 aa (API, warm) | 4.2-4.3 s | 4.0 s | 0.06 s | 0.2 s | 0.04 s |
| 1 protein, 1,215 aa (API) | 4.7 s | 4.0 s | 0.05 s | 0.6 s | 0.04 s |
| 20 test proteins, 21,329 aa (CLI) | 24 s | 5.4 s | 0.08 s | 18.6 s | 0.06 s |

The first request after start-up takes about 5.4 s. DIAMOND dominates single requests because it loads the database
and builds its seed index on every call. Batches share that cost, so a batch averages about 1.2 s per protein on CPU.

## Verification

Neither check scores test labels; both compare prediction files.

**Refit reproduction.** I reran the refit for l3_c0 twin only:

```bash
CAFA6_KNN_DIR=$PWD/data/proc_gogpt_tv2/knn_trainval python -m cafa6.stacker_tv --higo-run runs/l3_c0_tv2 \
  --out runs/stacker_tv/l3_c0_refit_twin --models b --refit-steps 800 --boot 200 --save-model runs/stacker_tv/l3_c0
```

This was on GPU. The cross-fitting rerun writes to its own directory, so the files in `runs/stacker_tv/l3_c0` are
unchanged apart from the new `model_twin.pt`. `--save-model` defaults to off.

- Applying the saved model on CPU through `predict_r2`'s own feature and inference code to the stored test features
  reproduces `runs/stacker_tv/l3_c0/pred_test_twin.npz`: max abs diff 4.9e-4 after float16 rounding, which is one
  float16 step near 1.0. Both files have the same 1,731,753 entries.
- The rerun's own `pred_test_twin.npz` matches the original to the same 4.9e-4.
- Val Fmax is 0.6180, as in the original run.

**Self-test.** `--self-test` predicts 20 GO-GPT test proteins (seed 0) from their sequences. All three runs share one
DIAMOND search over `data/serve` and are compared with those proteins' rows of `pred_test_twin.npz`:

| run | HiGO computed by | max abs diff |
|---|---|---|
| `cpu_stored` | the stored HiGO rows (`pred_test_top500.npz`) | 2.4e-4 |
| `cuda_bf16` | GPU, bf16 autocast (how the benchmark file was made) | 2.4e-4 |
| `cpu` (serving path) | CPU, fp32 | 0.16 |

- The test passes when `cpu_stored` and, if a GPU is present, `cuda_bf16` are within 1e-3. Both are.
- So the new-sequence path (DIAMOND, kNN prior, organism features, stacker and propagation) matches the benchmarked
  pipeline, and so does the network when it runs in the benchmark's precision. The kNN prior from the new DIAMOND
  search differs from the stored train + val prior by at most 2e-5.
- The CPU fp32 path differs only through the precision of the HiGO network. The stored HiGO predictions were made
  under CUDA bf16 autocast. CUDA fp32 and CPU fp32 give identical HiGO scores, which differ from the bf16 ones by up
  to 0.049 (mean 0.0014). CPU bf16 autocast does not reproduce CUDA bf16 either (max 0.034).
- After the stacker, the fp32 path puts 68% of score entries within 1e-3 of the benchmark file (99th percentile
  0.027). 5 of 901 above-threshold decisions change, and the largest single difference (0.16) is in one protein of the
  50-80% identity bin.
- I did not measure whether fp32 is better or worse than bf16 in accuracy; that would need labels.

## Implementation notes

- **Same-organism flag.** This feature uses the top-bitscore hit among the training references only. That is how
  `data/proc_gogpt_tv2/cells.parquet` defines it for both TV v2 (the stacker's training data) and test, even though
  the kNN prior uses train + val.
- **Organism resolution.** NCBI taxon ids map to GO-GPT organism names through UniProt (`data/serve/taxa.tsv`, 3,002
  organisms). The organism name sets the HiGO taxon index (67 organisms with >= 50 training proteins, else 0) and the
  same-organism comparison.
- **Single database.** The benchmark ran two searches, train-only (`-k 50`) and val-only (`-k 51`, `--dbsize`), and
  merged them. Serving uses one combined database with `-k 51`. On the 20 self-test proteins, 5 had 1-2 tail hits
  that differ, and the effect on the prior was at most 2e-5.

## Licences to check before making it public

- ESM2 weights and code (facebook/esm2_t12_35M_UR50D): MIT.
- UniProt data (sequences, organism and taxon mapping): CC BY 4.0, with attribution.
- Gene Ontology (go-basic 2023-01-01, term names, the propagated labels): CC BY 4.0, with attribution.
- GO-GPT training data (wanglab/gogpt-training-data): its Hugging Face card says `license: apache-2.0` (checked
  2026-10-18). Its labels and sequences derive from UniProt and GO, so their CC BY 4.0 attribution applies as well.
  The card asks users to cite BioReason-Pro (Fallahpour et al., bioRxiv 2026.03.19.712954).
- CAFA-5 IA weights (`IA.txt`) come from the CAFA-5 evaluation material. Check its terms if the file is redistributed.
  It is only used as a stacker feature.
- DIAMOND: GPL-3.0. Calling the unmodified binary is fine, but redistributing it means shipping it under GPL terms.
- Python stack: PyTorch (BSD-3), transformers (Apache-2.0), FastAPI (MIT), uvicorn (BSD-3).

## Limitations

- **Accuracy without a close homologue is much lower.** IA-weighted Fmax on the GO-GPT temporal test (look 3), by
  identity to the nearest training protein:

  | identity bin | Fmax |
  |---|---|
  | all proteins | 0.718 |
  | 100% | 0.879 |
  | 30-50% | 0.480 |
  | <30% | 0.451 |
  | no hit | 0.342 |

  In the two lowest bins GO-GPT ten-sample scores higher (0.461 and 0.382). Reports and the web page flag proteins
  in these bins.
- **Thresholds.** The per-aspect thresholds (MF 0.28, BP 0.42, CC 0.45) give one operating point. Every score >= 0.01
  is in the TSV, so other cut-offs can be applied afterwards.
- **Root terms.** Root and near-root terms (for example molecular_function, binding, protein binding) score high
  for most proteins because of max-propagation. They carry little information.
- **Fixed references.** The reference set is the November 2022 GO-GPT snapshot. Annotations added since are not
  used, and a protein that is itself a reference copies its own labels unless `--exclude-self` is set.
- **Seed spread.** The HiGO contribution to the stacker comes from one training seed (l3_c0). Five 35M runs scored
  0.718-0.734 test-like Fmax on TV v2 through the same stacker, so another seed would give somewhat different
  scores.
