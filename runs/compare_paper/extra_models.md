# External GO predictors added to the comparison (2026-10-18/19)

Each external method was run from its official code and released weights on the 8,630 GO-GPT test sequences
(`data/proc_gogpt/train.parquet`, split == test). Outputs were converted to `preds/<name>.npz` with
`python -m cafa6.compare_paper import-tsv` and scored by `compare_paper score` together with the existing
rows. Nothing was tuned on the test set. Scores below 0.01 were not kept; that is the scorer's lowest
threshold. Step-by-step timing and failures are in `logs/extra_models_progress.log`.

The test annotations date from 2023-03 to 2024-02. A model whose training snapshot is later than about
2023-01 gets a leakage flag. GO-release differences matter because the scorer uses GO 2023-01-01: predicted
terms missing from that release, after alternative-id mapping, are dropped. The GO terms a model cannot
predict at all are a fixed handicap for that model.

Wrappers and converters are in `runs/compare_paper/extra_models/`:
- `run_proteinfer.py`: the SavedModel loader. tensorflow_hub 0.16 dropped `hub.Module`.
- `run_deepgose.py`: stubs `dgl`, which predict.py does not use.
- `run_deepfri.py`: works around Python 3.12 rejecting the `'rU'` file mode.
- `run_sprof.py`: works around transformers 5 removing `batch_encode_plus`.
- `to_tsv.py`: converts the output formats.

Downloads and environments were kept in RAM under `/dev/shm/xm/`, not on disk, and were removed after the
runs. The raw prediction files (gzipped TSV) and run logs remain in `/dev/shm/xm/raw_outputs/` until
reboot.

## Results (S3_all, 8,630 proteins; Fw = IA-weighted Fmax)

S-min is IA-weighted and averaged over the three aspects. The difference column is HiGO-R2's Fw mean minus
the method's, with a paired protein bootstrap (1,000 reps, seed 0). Holm correction runs over the
13-method family.

| method | year, venue | predictions | training data, cutoff | leakage flag | Fw MF | Fw BP | Fw CC | Fw mean | S-min | HiGO-R2 − method [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| HiGO-R2 (ours) | — | frozen look-3 file | GO-GPT train + val | — | 0.738 | 0.658 | 0.758 | 0.718 | 7.56 | — |
| SPROF-GO | 2023, Brief. Bioinform. | official models, our run | GO data of 2020-01; training proteins annotated before 2018-01 | no | 0.615 | 0.400 | 0.609 | 0.541 | 13.06 | +0.177 [+0.169, +0.185], p_Holm 0.013 |
| DeepGOPlus 1.0.10 | 2019, Bioinformatics (release 2022-11) | official package, our run | Swiss-Prot 2022_04, GO 2022-10-07 | no | 0.539 | 0.264 | 0.481 | 0.428 | 15.80 | +0.290 [+0.283, +0.298], p_Holm 0.013 |
| ProteInfer GO, 5 models | 2023, eLife | official models, our run | Swiss-Prot 2019_01, random split | no | 0.514 | 0.253 | 0.484 | 0.417 | 17.20 | +0.301 [+0.292, +0.308], p_Holm 0.013 |
| DeepGO-SE | 2024, Nat. Mach. Intell. | official package, our run | Swiss-Prot 2021_04, GO 2021-11-16 | no | 0.522 | 0.253 | 0.131* | 0.302* | 16.39 | +0.416 [+0.408, +0.424], p_Holm 0.013 |
| DeepFRI, sequence CNN | 2021, Nat. Commun. | official models, our run | PDB + SWISS-MODEL chains, SIFTS 2019-06-18 | no | 0.282 | 0.179 | 0.360 | 0.274 | 17.52 | +0.444 [+0.436, +0.453], p_Holm 0.013 |
| *variant:* ProteInfer, 1 model | 2023, eLife | CLI default, our run | as above | no | 0.499 | 0.234 | 0.457 | 0.397 | 20.10 | +0.321 [+0.313, +0.329], p 0.001 |
| InterLabelGO+ | 2024, Bioinformatics | released package (Zenodo), our run | GOA snapshot after the test window (files 2024-09-26) | **yes: 96.1% of test GT pairs are in its training labels** | 0.750 | 0.685 | 0.748 | 0.728 | 7.80 | −0.010 [−0.016, −0.005], p 0.001; not comparable |

\* The DeepGO-SE CC value comes from faulty released checkpoints; see below.

On S6_noknowledge (8,159 proteins) the Fw means are:

| method | S6 Fw mean | HiGO-R2 − method [95% CI] |
|---|---|---|
| HiGO-R2 | 0.735 | — |
| SPROF-GO | 0.545 | +0.190 [+0.183, +0.198] |
| DeepGOPlus | 0.431 | +0.305 [+0.296, +0.313] |
| ProteInfer, 5 models | 0.418 | +0.317 [+0.309, +0.325] |
| DeepGO-SE | 0.304 | +0.432 [+0.424, +0.439] |
| DeepFRI | 0.276 | +0.460 [+0.451, +0.469] |
| InterLabelGO+ | 0.742 | −0.007 [−0.012, −0.001] |

Unweighted Fmax means on S3 are:

| method | S3 F mean |
|---|---|
| SPROF-GO | 0.628 |
| DeepGOPlus | 0.538 |
| ProteInfer, 5 models | 0.523 |
| DeepGO-SE | 0.440 |
| DeepFRI | 0.385 |
| InterLabelGO+ | 0.776 |
| HiGO-R2 | 0.765 |

Fw mean by best-hit identity to GO-GPT training proteins (threshold chosen within each bin; full table in
`report_tables.md`):

| method | no hit | <30 | 30–50 | 50–80 | 80–95 | 95–99.9 | 100 |
|---|---|---|---|---|---|---|---|
| HiGO-R2 | 0.342 | 0.451 | 0.480 | 0.506 | 0.675 | 0.858 | 0.878 |
| SPROF-GO | 0.403 | 0.529 | 0.514 | 0.512 | 0.572 | 0.582 | 0.549 |
| DeepGOPlus | 0.242 | 0.462 | 0.440 | 0.422 | 0.456 | 0.451 | 0.423 |
| ProteInfer, 5 models | 0.308 | 0.435 | 0.408 | 0.397 | 0.433 | 0.426 | 0.450 |
| DeepGO-SE | 0.268 | 0.333 | 0.306 | 0.291 | 0.321 | 0.329 | 0.295 |
| DeepFRI | 0.204 | 0.253 | 0.274 | 0.262 | 0.304 | 0.283 | 0.284 |
| InterLabelGO+ (leaked) | 0.436 | 0.574 | 0.571 | 0.628 | 0.714 | 0.821 | 0.816 |

The external predictors were trained on older and smaller annotation sets than GO-GPT's training proteins,
and their kNN references, where they have any, are those older sets. Their scores therefore depend much
less on identity to GO-GPT training: the Spearman ρ between per-protein F1 and identity is 0.14–0.16
(DeepGO-SE −0.09), against 0.56 for HiGO-R2 and 0.29 for the leaked InterLabelGO+. SPROF-GO is above HiGO-R2 below 80% identity and far below it at 80% and above.

Checks after the final scoring run:
- All 24 earlier rows of `summary.tsv` and all 96 earlier rows of `per_bin.tsv` are identical to the copies
  taken before this work, with a maximum absolute difference of 0.
- The only change is `p_holm`, because the Holm family grew from 8 to 13 methods.

## Per method

### SPROF-GO (biomed-AI/SPROF-GO)
- **Weights.** `model/{MF,BP,CC}_model_{0..4}.ckpt` from the GitHub repo.
- **Protein language model.** ProtT5-XL-UniRef50. The README points to Zenodo record 4644188, but that
  download ran at about 1 MB/s, so the same model was taken from HF `Rostlab/prot_t5_xl_uniref50`.
- **Run.**
  - `script/predict.py --gpu --feat_bs 1`, with only the documented `ProtTrans_path` set.
  - The output is after label diffusion over the training and validation proteins.
  - The tool accepts at most 5,000 sequences per call, so the test set was split into two seeded random
    halves of 4,315. The diffusion graph's test–test edges exist only within a half.
  - About 22 min per half on GPU 1.
- **Training data** (paper, CAFA protocol):
  - Swiss-Prot/GOA/GO annotations retrieved in 2020-01, experimental codes only.
  - Training proteins were annotated before 2018-01; validation proteins in 2018.
- **Vocabulary.** 790 MF, 4,766 BP and 667 CC terms; 114 predicted terms are not in GO 2023-01-01.
- **Overlap with the test set.** 139, 312 and 233 test proteins appear in its MF, BP and CC train+val sets,
  all with pre-2019 annotations. No leakage flag.

### DeepGOPlus (bio-ontology-research-group/deepgoplus)
- **Package.** `bio2vec.net/data/deepgo/data-1.0.10.tar.gz` (2022-11-06). Its `RELEASE.md` reads:
  "trained using the Gene Ontology released on 2022-10-07 and the SwissProt data with version 2022_04".
  - This is the newest release before the test window; later releases (1.0.11 and up) were not used.
  - The current `data.tar.gz` is release 1.0.18 (2024-04) and would leak.
- **Run.**
  - `deepgoplus/main.py`, the CNN plus DIAMONDScore with the release's alphas (MF 0.63, BP 0.62,
    CC 0.54).
  - Output threshold 0.01 instead of the default 0.1.
  - DIAMOND 2.1.10; CPU, a few minutes.
  - TF 2.19 with `tf_keras` (`TF_USE_LEGACY_KERAS=1`).
- **Training set.** 75,268 Swiss-Prot proteins, of which 448 are test proteins (pre-2022-10 annotations).
  DIAMOND found hits for 8,336 test proteins.
- **Vocabulary.** 58 predicted terms are not in GO 2023-01-01. No leakage flag.

### ProteInfer (google-research/proteinfer)
- **Models.** GO random-split CNNs from
  `storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/`.
  - The main row averages the first five elements of `utils.GO_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS`
    (13703706, 13703742, 13703997, 13704131, 13705631). Five is the maximum that `proteinfer.py` allows.
  - The single-model variant is the CLI default (13703706).
- **Run.**
  - `inference.Inferrer` builds a TF1 `hub.Module`, which tensorflow_hub 0.16 no longer has. Instead,
    `extra_models/run_proteinfer.py` loads the same SavedModel with `tf.compat.v1.saved_model.loader`.
  - It feeds `utils.residues_to_one_hot` encodings to the serving signature, as `in_graph_inferrer` does,
    and averages confidences over the elements as `get_preds_at_or_above_threshold` does.
  - CPU: 3 min for one model, 16 min for five.
- **Training data.** Swiss-Prot 2019_01, random split, 32,102 labels. 442 predicted GO-2019 terms are not in
  GO 2023-01-01. No leakage flag.

### DeepGO-SE (bio-ontology-research-group/deepgo2)
- **Package.** `bio2vec.net/data/deepgo/deepgo2/data.tar.gz` (15 GB, 2023-10-24). It was streamed, keeping
  only the files `predict.py` reads: `go.obo` (2021-11-16), `go-plus.norm`, `{mf,bp,cc}/terms.pkl` and
  the 18 `deepgozero_esm_plus_*.th` ensemble checkpoints.
- **Run.**
  - `predict.py -t 0.01 -d cuda:0`, ESM2-3B, mean-pooled layer 36. About 22 min on GPU 0.
  - `dgl` is stubbed: predict.py imports it but does not use it.
- **Training data** (paper): Swiss-Prot 2021_04, GO 2021-11-16. No leakage flag.
- **CC checkpoints faulty.** The CC output is implausible: GO:0005618 "cell wall" has a mean score of 0.885
  over all test proteins, while GO:0110165 "cellular anatomical entity" has 0.05. This was checked without
  our test labels:
  - On the authors' own CC test split (`training-data.tar.gz`, 4,221 proteins, their stored ESM2 features),
    the shipped CC checkpoints behave the same way: each model reaches F ≈ 0.05 and the top mean term is
    again "cell wall" (0.86).
  - `cc/terms.pkl` is byte-identical in both packages.
  - `go-plus.norm` matches the checkpoint shapes: 2,829 terms + 79,828 zero classes = 82,657 embeddings.
  - The reference example outputs the authors posted in issue #4 now return 404.
  - MF and BP top terms are sensible (binding, cellular process). The MF value is usable; the CC value and
    the mean are not representative of the method.
- **Vocabulary.** 228 predicted terms are not in GO 2023-01-01.

### DeepFRI (flatironinstitute/DeepFRI)
- **Models.** `users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz` (2023-01-12).
- **Run.**
  - `predict.py --fasta_fn` selects the `cnn` models (`DeepCNN-MERGED_*`): a one-hot sequence CNN with no
    language model (`lm_model_name` None). The LSTM language model is used only by the structure-based
    GCN, which needs contact maps.
  - Full `Y_hat` taken from `*_pred_scores.json`.
  - CPU, about 17 min.
  - The wrapper drops the `'U'` from `open(..., 'rU')`, which Python ≥ 3.11 rejects.
- **Training data** (paper): nrPDB and SWISS-MODEL chains, GO from SIFTS 2019-06-18 / UniProtKB 2019.06.
- **Vocabulary.** 489 MF, 1,943 BP and 320 CC terms; 42 are not in GO 2023-01-01. No leakage flag.

### InterLabelGO+ (QuanEvans/InterLabelGO)
- **Package source.**
  - The official host seq2fun.dcmb.med.umich.edu timed out again: three attempts of 40 s on 2026-10-18,
    after two earlier ones.
  - The same `InterLabelGO+.tar.xz` is on Zenodo, record 15508797, 7,449,545,408 bytes, md5
    94b94267fbfa34bf053d9a6b944d3dd3, which matches the record.
  - GitHub has the code only, no releases. Docker Hub has nothing. HF has only an unofficial MF-only model.
- **Run.**
  - `predict.py --use_gpu --seqid`: the README's dynamic weighting of the network and the DIAMOND kNN, top
    500 terms per aspect.
  - ESM2-3B, layers 34–36, from the package. About 35 min of run time on GPU 0. The run was paused once to
    give DeepGO-SE the GPU and resumed with `-c`, which reuses the cached embeddings.
- **Training data.** The shipped `Data/cafa5_raw_data/train_terms.tsv` is not the CAFA-5 Kaggle set:
  - 142,478 proteins and 5.18 M propagated rows; 227,839 sequences in `train_seq.fasta`.
  - Files dated 2024-09-26; `go-basic.obo` is 2023-01-01.
  - It contains 8,629 of the 8,630 test proteins and 96.1% of the test ground-truth (protein, term) pairs.
    It was evidently refreshed from a GOA release after 2024-02, which matches the repo's `update_data.py`.
  - **Test-label leakage confirmed.** The row is outside the Holm family and is not a comparison.
- **Vocabulary.** All predicted terms are in GO 2023-01-01.

## Not run

### ProtBoost (2024, arXiv 2412.04529; CAFA-5 2nd place)
No released predictions or weights exist. What was tried:
- **GitHub** btbpanda/CAFA5-protein-function-prediction-2nd-place: no releases. Every artifact URL in
  `Download.ipynb` returns 404, because the bucket storage.yandexcloud.net/cafa5embeds no longer exists.
  This includes `sub/submission.tsv`, `postproc/pred.tsv`, the GCN checkpoints and the NN models.
- **Kaggle.** The author's public datasets (`btbpanda/cafa5-go-annotations`, `btbpanda/cafa-terms-diff`)
  are input data. The neural-network prototype notebook (alexandervc/pytorch-keras-etc-3-blend-cafa-metric-etc)
  returns 403 without a Kaggle token, and it would be one base model, not ProtBoost.
- **Zenodo and Hugging Face:** no records.
- **Wayback Machine:** CDX query timed out.

The only remaining route is full retraining, which needs about 300 GB and V100-days. Per its paper, the
training data were the CAFA-5 set (UniProtKB 2022-11-17, GO 2023-01-01) plus GOA electronic annotations as
stacking features. Kept as paper-reported numbers only.

### Prot2Text-V2 (2025, NeurIPS)
Not a GO predictor.
- The official code (github.com/ColinFX/Prot2Text-V2) and weights (HF xiao-fei/Prot2Text-V2-11B-Instruct-hf)
  generate free-text function descriptions.
- `scripts/benchmark.py` computes only exact match, BLEU, ROUGE and BERTScore. The paper (arXiv 2505.11194)
  adds an LLM-as-judge and human review.
- Neither provides GO output or a text-to-GO protocol, so no GO mapping was constructed and it was not
  scored.
- Training data: Prot2Text-Data, 248,315 Swiss-Prot entries (Abdine et al., preprint 2023-07).
