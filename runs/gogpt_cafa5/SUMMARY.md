# GO-GPT (released `wanglab/gogpt`) on its own temporal-holdout test set

**Bottom line.** Scored on the full test set (all 8,630 proteins) with the paper's protocol, the released checkpoint
matches or beats every per-aspect Fmax that BioReason-Pro reports for GO-GPT's single-run mode. The gap is small for
MF and CC (+0.01 to +0.02) and large for BP (+0.07 unweighted, +0.08 weighted). Beam search and greedy decoding
score within 0.002 of each other, so decoding does not explain the BP gap. The GO-GPT predictions shipped inside the
HF test dataset (`go_pred` column) do reproduce the paper to within ~0.01. So the public checkpoint is most likely not
the one behind the paper's GO-GPT tables (see Caveats).

## Paper (bioRxiv 10.64898/2026.03.19.712954 v2, posted 2026-07-19)

- **Test set.** CAFA-style temporal holdout: training on annotations up to Nov 2022; test proteins gained new
  experimental annotations between Mar 2023 and Feb 2024 in an aspect they had no annotations in before.
  - 8,630 proteins, 230,824 propagated annotations.
  - By aspect: MF 2,080, BP 5,819, CC 3,440 proteins.
  - Evidence codes: EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC.
- **Ontology and propagation.** GO release 2023-01-01; propagation over is_a and part_of.
- **Metric.** Official CAFA-evaluator: IA-weighted and unweighted Fmax per aspect, using CAFA-5 IA weights
  (`data/IA.txt` in the repo).
- **Scores.** GO-GPT's generated terms get score 1 ("greedy decoding, single run").
- **Other modes.** 10 samples per protein at T=0.7, top-k=20, scored either as term frequency or as a best-of-10
  oracle.
- **Headline.** "Weighted Fmax 0.65" is the mean over aspects of Table S3's weighted column.
- **Two reference subsets.**
  - Table S3 uses all 8,630 proteins.
  - Tables S6–S9 (BioReason comparisons) drop the 471 "limited-knowledge" proteins listed in
    `data/common_proteins.txt`, leaving 8,159 proteins (MF/BP/CC = 1,878/5,623/3,293; reproduced exactly here).
- **Training set.** 133,492 proteins from 3,135 organisms. GO-GPT's organism vocabulary is the top 200 species.

## Results (CAFA-evaluator, prop=max, norm=cafa, all proteins predicted, coverage = 1.0 in every aspect)

### Full test set, 8,630 proteins (paper Table S3; GO 2023-01-01, CAFA-5 IA)

| Aspect (n) | Paper Fmax (single run) | Paper Fmax_w | Reproduced, beam: Fmax | Fmax_w | Reproduced, greedy: Fmax | Fmax_w | Shipped `go_pred`: Fmax | Fmax_w |
|---|---|---|---|---|---|---|---|---|
| MF (2080) | 0.766 | 0.701 | **0.777** | **0.714** | 0.778 | 0.716 | 0.757 | 0.689 |
| BP (5819) | 0.587 | 0.539 | **0.659** | **0.617** | 0.658 | 0.616 | 0.586 | 0.542 |
| CC (3440) | 0.801 | 0.713 | **0.812** | **0.728** | 0.812 | 0.728 | 0.805 | 0.717 |
| mean | 0.718 | 0.651 | 0.749 | 0.686 | 0.749 | 0.687 | 0.716 | 0.649 |

Other paper modes, for reference.

| Mode | Fmax (MF / BP / CC) | Fmax_w (MF / BP / CC) |
|---|---|---|
| Probability from 10 samples (Table S5) | 0.766 / 0.600 / 0.816 | 0.706 / 0.547 / 0.730 |
| Best-of-10 oracle (Table S4) | 0.802 / 0.644 / 0.840 | 0.743 / 0.595 / 0.764 |
| InterLabelGO+ | 0.727 / 0.566 / 0.790 | 0.692 / 0.525 / 0.696 |
| ProtBoost | 0.702 / 0.533 / 0.777 | 0.658 / 0.490 / 0.669 |

The reproduced BP result beats even the paper's best-of-10 oracle.

### No-knowledge subset, 8,159 proteins (paper Tables S6–S9: GO-GPT pass@1)

| Aspect (n) | Paper Fmax | Paper Fmax_w | Reproduced beam: Fmax | Fmax_w | Greedy: Fmax | Fmax_w | Shipped `go_pred`: Fmax | Fmax_w |
|---|---|---|---|---|---|---|---|---|
| MF (1878) | 0.792 | 0.729 | 0.803 | 0.743 | 0.805 | 0.745 | 0.783 | 0.717 |
| BP (5623) | 0.593 | 0.545 | 0.668 | 0.626 | 0.666 | 0.625 | 0.593 | 0.550 |
| CC (3293) | 0.806 | 0.718 | 0.818 | 0.736 | 0.817 | 0.736 | 0.811 | 0.725 |

### Additional labelled variant: CAFA-6 `go-basic.obo` + CAFA-6 `IA.tsv`, all 8,630 proteins (beam)

| Aspect | Fmax | Fmax_w |
|---|---|---|
| MF | 0.776 | 0.716 |
| BP | 0.660 | 0.617 |
| CC | 0.813 | 0.729 |

These are nearly identical to the paper-IA numbers.

Precision and recall at Fmax are in `metrics.tsv` and `greedy/metrics.tsv`. For example, for full-set beam BP
unweighted precision is 0.662 and recall is 0.657.

## Protocol used here

- **Test data.** `ref/test.parquet`: 8,630 proteins with sequence, UniProt organism string, propagated GT in
  `go_mf`/`go_bp`/`go_cc`, and the shipped GO-GPT predictions in `go_pred`.
  - The file is byte-identical in size to the HF `wanglab/cafa5` config `interlabel_test_dataset_with_gogpt_memorized_copy`
    (test split). That dataset is gated for this account; the file was already present from an earlier stage of this
    job.
  - GT statistics match the paper exactly: 8,630 proteins, 230,824 annotations, and MF/BP/CC counts
    2080/19,773, 5819/169,459, 3440/41,592 (proteins/terms).
- **Reference files.** `ref/IA.txt`, `ref/common_proteins.txt`, `ref/go-basic_gogpt.obo` (GO 2023-01-01, md5-identical
  to `bioreason2/dataset/go-basic.obo`) and `ref/cafa_evals.py` all come from the BioReason-Pro repo zip.
- **Model settings.** Released `GOGPTPredictor.predict` settings:
  - ESM2-3B layer 30 in fp32; sequences truncated to 1,024 tokens (1,022 residues; 2,342 test proteins are longer).
  - Beam 5, length_penalty 0.3, at most 300 new tokens per aspect, one pass per aspect.
  - Organism is the UniProt organism string mapped to GO-GPT's 200-species vocabulary; 97.9% of test proteins
    are in it and the rest get `<UNKNOWN>` (id 0).
- **Greedy variant.** Argmax decoding, the paper's literal "greedy".
- **Scoring.** Predictions get score 1.0. GT is scored as given, and CAFA-evaluator propagates both GT and
  predictions (prop=max, norm=cafa).

## Deviations and implementation notes (all verified to be exact)

- **Batching.** Proteins are batched (8 for beam, 16 for greedy), padded to the longest sequence in the batch, with
  padded positions masked. The original pads to 1,024.
- **Protein-stream cache.** The decoder's protein stream is computed once per batch instead of at every decoding
  step. Protein residues never attend to GO tokens, so caching is exact. ESM2 also stops after layer 30 instead of
  keeping all 37 hidden states (max abs diff 0.0).
  - Beam: 36/36 aspect lists identical to the unmodified `GOGPTPredictor.predict` (12 proteins, 3 of them >1,100 aa).
  - Greedy: 18/18 identical to `GOGPT.generate(top_k=1)`, and a one-sync-per-step argmax loop gave 144/144
    identical lists.
- **No source patching.** The GO-GPT source in /tmp was not modified. The cache is installed by overriding
  `model.forward` on the instance.
- **Precision matters.** Running ESM2 in bf16 (an earlier aborted attempt) changed 2.2% of aspect lists, so all
  reported numbers use fp32.

## Runtime

Hardware: one H100 (GPU 1), shared with another user's job.
- **Beam.** Full test set (no sampling), 3 parallel processes: ~1 h 53 min wall-clock (6,650–6,770 s per shard;
  ~0.78 s/protein overall).
- **Greedy.** One process running concurrently, ~1 h 45 min.
- **Scoring.** ~4 min per decoding mode for the 3 protocols.

## Caveats

1. **The released checkpoint does not match the paper's GO-GPT numbers**, especially for BP. Several findings point
   to a different checkpoint:
   - The shipped `go_pred` predictions agree with the paper to about 0.01.
   - They agree with the released checkpoint's outputs for only 47–61% of aspect lists (mean Jaccard 0.60 for BP,
     0.74 for CC, 0.78 for MF), for both beam and greedy decoding.
   - Organism input was ruled out: switching to `<UNKNOWN>` makes the agreement worse.

   Leakage via the published training set was checked. `wanglab/gogpt-training-data` (133,492 proteins, train+val)
   shares 760 IDs with the test set, 471 of them the limited-knowledge proteins. For none of these do the training
   labels contain the held-out test-aspect terms. Whether the released checkpoint was trained only on that data
   cannot be verified. Treat the reproduced numbers as "public checkpoint" performance, not as a paper replication.
2. **Binary predictions.** Fmax is computed from a single operating point (no scores), exactly as in the paper's
   single-run setting.
3. **Organism sensitivity.** GO-GPT is strongly organism-conditioned. For 96 test proteins, replacing the true
   organism with `<UNKNOWN>` cut BP agreement with the shipped predictions from 0.58 to 0.27 Jaccard. For zero-shot
   CAFA-6, species outside the 200-species vocabulary will be degraded.
   - The resolver maps plain species names (for example "Homo sapiens" from the CAFA-6 taxon list) to the
     vocabulary's full UniProt strings ("Homo sapiens (Human)").
4. **Truncation.** Sequences longer than 1,022 residues are truncated, as in the released predictor.

## Files

- `preds_shard*.jsonl` (beam) and `greedy/preds_shard0.jsonl`: raw per-aspect GO lists plus the resolved organism.
- `cafaeval/<protocol>/pred/*.tsv` (CAFA format, protein<TAB>GO<TAB>score) and `cafaeval/<protocol>/gt.tsv`
  (protein<TAB>GO), with CAFA-evaluator outputs in `cafaeval/<protocol>/out/`. The same layout is under `greedy/`.
- `metrics.tsv` and `greedy/metrics.tsv`: per protocol, method and aspect: Fmax, Fmax_w, precision, recall,
  coverage, n.
- Code: `cafa6/benchmarks/gogpt_eval.py`. Log: `cafa6/logs/gogpt_cafa5.log`.
