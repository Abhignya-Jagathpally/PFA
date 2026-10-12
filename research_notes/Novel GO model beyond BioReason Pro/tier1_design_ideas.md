# Tier-one ML design ideas (ICLR / ICML / NeurIPS) for a GO prediction model that beats GO-GPT / BioReason-Pro

Scope: mechanisms from ICLR, ICML and NeurIPS main-track papers that could be adapted to HiGO (ESM2-35M + LoRA, GO-term queries with ancestor-mean embeddings, entmax per-term residue pooling, max-constraint module, IA-weighted loss, taxon embedding, post-hoc homology kNN gate). Every paper is tagged with venue and year. Non-tier-one items are labelled explicitly. "Reported" means a number stated by the paper; "Inference" means my reasoning, not something the paper tested.

Source-verification note: papers marked [fetched] had their venue page, proceedings page or PDF retrieved during this session. Papers marked [known] are well-established tier-one papers whose venue/year I am stating from prior knowledge with an arXiv link; they were not re-fetched here and the report writer should treat the venue tag as high-confidence but unverified in this session.

---

## Context: what the target (GO-GPT / BioReason-Pro) is

### Takeaway
GO-GPT is a frozen ESM2-3B encoder feeding a 12-layer, ~200M-parameter GPT decoder that emits GO tokens autoregressively; BioReason-Pro is an LLM layered on top that consumes GO-GPT outputs plus InterPro and context. Neither is a tier-one ML venue paper (bioRxiv 2026).

### Cited Findings
- GO-GPT: ESM2-3B (`esm2_t36_3B_UR50D`) encoder + 12-layer GPT decoder with prefix causal attention, ~3.2B parameters total (3B ESM2 + 200M decoder); organism conditioning; reported weighted Fmax 0.65–0.70 — [Hugging Face wanglab/gogpt](https://huggingface.co/wanglab/gogpt); [GitHub gogpt](https://github.com/bowang-lab/BioReason-Pro/tree/main/gogpt)
- GO-GPT config shows `freeze_esm: true` and gated attention; inference uses sampling (temperature 0.7, top-k 50) or beam search (beam 5, length penalty 0.3), max 300 new tokens — [GitHub gogpt](https://github.com/bowang-lab/BioReason-Pro/tree/main/gogpt)
- BioReason-Pro reports 73.6% Fmax on GO term prediction; held-out test set of 8,630 proteins; venue is bioRxiv (2026), not peer-reviewed tier-one — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v2); [GitHub README](https://github.com/bowang-lab/BioReason-Pro)
- The BioReason-Pro release includes Qwen3-Embedding-4B vectors for 43,248 GO terms (177 MB), used to initialise its GO graph encoder — [GitHub README](https://github.com/bowang-lab/BioReason-Pro)

### Inferences
- The released 177 MB GO-term text-embedding file fits within the ~4 GB disk budget and could seed a text-based label encoder (see the zero-shot section) without running a 4B text model locally.
- Because the ESM2-3B encoder in GO-GPT is frozen, its homolog advantage most likely comes from the 3B representation plus a decoder that memorises label co-occurrence for families seen in training, not from fine-tuned features. This suggests that in-network retrieval of labelled neighbours (below) is the most direct way to match it on close-homolog proteins.

### Gaps
- I did not find a published per-identity-bin breakdown of GO-GPT performance in a tier-one venue; the "88% close-homolog" figure comes from the project context, not a source I verified.

---

## Q1. Hierarchical / structured multi-label learning: which output structures beat autoregressive label-set generation?

### Takeaway
For protein-centric (instance-averaged) F-measure with a single global threshold, theory (Koyejo et al., NeurIPS 2015) says the Bayes-optimal predictor thresholds calibrated per-label marginals, and label dependence matters only through the threshold. That removes the main theoretical argument for autoregressive set generation under an Fmax metric. The structured-output methods with tier-one evidence (SPL, MBM boxes, C-HMCNN) mainly buy consistency and exact-match gains, not marginal-quality gains; none has been shown on GO-scale label spaces (~thousands of terms) with IA-weighted Fmax.

### Cited Findings
- **Consistent Multilabel Classification (Koyejo, Natarajan, Ravikumar, Dhillon; NeurIPS 2015) [fetched].** For instance-, micro- and macro-averaged metrics built from linear-fractional functions of TP/FP/FN/TN (including F-measure and Jaccard), the population-optimal classifier thresholds per-label conditional marginals; "label dependence in the underlying distribution is relevant to the optimal classifier only through the threshold parameter", and the threshold is shared across labels for instance- and micro-averaged metrics. They give a consistent plug-in algorithm — [NeurIPS 2015 PDF](https://proceedings.neurips.cc/paper/2015/file/85f007f8c50dd25f5a45fca73cad64bd-Paper.pdf); [NeurIPS poster page](https://neurips.cc/virtual/2015/poster/5749)
- Reported (author slides on the same work): shared-threshold plug-in gave the best micro- and instance-averaged F1 on Scene, Birds, Emotions, Cal500 versus per-label thresholds — [Koyejo slides](https://cs.stanford.edu/~sanmi/documents/koyejo-metrics.pdf)
- **Semantic Probabilistic Layers (Ahmed, Teso, Chang, Van den Broeck, Vergari; NeurIPS 2022) [fetched].** Output layer = neural probabilistic circuit × compiled constraint circuit, normalised, trained by exact maximum likelihood; guarantees 100% constraint satisfaction. Reported on 12 HMLC datasets (8 functional-genomics FunCat/GO-style sets, 56–4130 classes): SPL beats HMCNN (Giunchiglia & Lukasiewicz, NeurIPS 2020) on exact match on 11/12, e.g. SEQ 7.98 vs 5.24, but Hamming score is slightly *lower* for SPL on most sets (e.g. CELLCYCLE 97.84 vs 98.26) — [NeurIPS 2022 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/c182ec594f38926b7fcb827635b9a8f4-Paper-Conference.pdf); [arXiv 2206.00426](https://arxiv.org/html/2206.00426)
- **Coherent Hierarchical Multi-Label Classification Networks, C-HMCNN (Giunchiglia & Lukasiewicz; NeurIPS 2020) [known].** The max-constraint module (MCM) and max-constraint loss already used in HiGO — [arXiv 2010.10151](https://arxiv.org/abs/2010.10151)
- **Semantic Loss (Xu, Zhang, Friedman, Liang, Van den Broeck; ICML 2018) [known].** Loss = −log probability that the factorised output satisfies a logical constraint, computed by knowledge compilation — [arXiv 1711.11157](https://arxiv.org/abs/1711.11157)
- **A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints (Ahmed, Chang, Van den Broeck; NeurIPS 2023) [fetched].** Exact constraint likelihood under autoregressive distributions is #P-hard; they optimise a pseudolikelihood centred on a model sample. Reported gains in logical consistency on Sudoku and shortest-path generation and SoTA detoxification — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3accfe8332366a6f740d8740cd4cd653-Abstract-Conference.html)
- **Multi-label Box Model, MBM (Patel, Dangati, Lee, Boratko, McCallum; ICLR 2022) [fetched].** Labels are trainable hyper-rectangles; inputs are embedded into the same box space; containment encodes DAG relations and is read as calibrated conditional probability; partial taxonomy can be injected. Reported: significantly better taxonomic consistency (new metric, consistency-constrained mAP) "while preserving or surpassing" SoTA predictive performance on 12 datasets — [OpenReview](https://openreview.net/forum?id=tyTH9kOxcvh); [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6645); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10392233)
- **HypStructure (Sinha, Zeng, Yamada, Zhao; NeurIPS 2024) [fetched].** A regulariser: map features to the Poincaré ball, compute hyperbolic class prototypes, match pairwise tree distances (hyperbolic CPCC) plus a centring loss placing the root at the origin. Reported lower hierarchy distortion and better coarse/fine accuracy on CIFAR-10/100 and ImageNet-100, largest benefit at low embedding dimension — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/a5d2da376bab7624b3caeb9f78fcaa2f-Paper-Conference.pdf); [code](https://github.com/uiuctml/HypStructure)
- Older geometric label-hierarchy embeddings [known]: Order embeddings (Vendrov et al., ICLR 2016) — [arXiv 1511.06361](https://arxiv.org/abs/1511.06361); Poincaré embeddings (Nickel & Kiela, NeurIPS 2017) — [arXiv 1705.08039](https://arxiv.org/abs/1705.08039); Hyperbolic entailment cones (Ganea, Bécigneul, Hofmann, ICML 2018) — [arXiv 1804.01882](https://arxiv.org/abs/1804.01882)
- Set vs sequence [known]: "Order Matters: Sequence to sequence for sets" (Vinyals, Bengio, Kudlur; ICLR 2016) showed seq2seq performance on set outputs depends strongly on the chosen output order — [arXiv 1511.06391](https://arxiv.org/abs/1511.06391). Energy-based structured output: SPEN (Belanger & McCallum; ICML 2016), evaluated on multi-label classification — [arXiv 1511.06350](https://arxiv.org/abs/1511.06350)

### Inferences
- CAFA protein-centric Fmax is an instance-averaged F-measure maximised over one global threshold τ. Koyejo et al.'s result therefore suggests that a well-calibrated factorised (per-term sigmoid) head is already the right functional form; an autoregressive decoder can only help by producing *better marginals*, e.g. through implicit co-occurrence modelling, not because joint decoding is needed at decision time. Caveats: IA weighting and the CAFA convention of averaging precision only over proteins with ≥1 prediction make the metric not exactly the textbook instance-averaged F; I did not find a paper extending the result to that variant.
- SPL's pattern (better exact match, slightly worse Hamming) suggests it would not move Fmax much relative to HiGO's MCM, while compiling a constraint circuit over 6–8k GO terms is a large engineering risk. Low priority.
- MBM boxes give a principled replacement for "query = term embedding + mean of ancestor embeddings": parent boxes contain child boxes by construction, so P(child) ≤ P(parent) is geometric, and rare deep terms inherit volume from ancestors. Worth one pilot because it changes the label parameterisation (where the GAT failed) rather than adding message passing.
- HypStructure is a cheap add-on regulariser on the pooled protein representation using GO distances; expected effect is small but the cost is one extra loss term.
- Autoregressive decoding in GO-GPT needs sampling/beam (temperature 0.7, beam 5) to produce scores; this makes per-term probabilities a by-product of decoding rather than calibrated marginals, which plausibly hurts thresholding on rare deep terms (inference, untested).

### Gaps
- I found no tier-one paper on label-set diffusion for multi-label classification; I cannot recommend it with any evidence.
- No tier-one paper directly compares autoregressive label-set generation against a calibrated factorised head on GO or another large DAG-structured ontology.

---

## Q2. Retrieval-augmented protein models: attending to labelled homologs inside the network

### Takeaway
This is the most direct attack on GO-GPT's advantage on close homologs. The tier-one recipe that transfers best is TabR (ICLR 2024): retrieve k nearest training items with a learned key, then attend over them with values that include an embedding of the neighbour's *labels* plus a correction term for the key difference. Protein-specific evidence comes from PoET-2 (NeurIPS 2025) and Protriever (ICML 2025), plus ProtEx (preprint, not tier-one) which reports its largest gains on rare GO classes and far-from-training sequences.

### Cited Findings
- **TabR (Gorishniy, Rubachev, Kartashev, Shlenskii, Kotelnikov, Babenko; ICLR 2024) [fetched].** A feed-forward net with a kNN-like attention block in the middle: encoder → key projection → retrieve nearest training candidates (FAISS, L2) → softmax over negative squared L2 similarity → values = label embedding of neighbour + T(k − k_i). The query itself is excluded from its own context during training to prevent leakage. Reported: best average among tabular DL models, beats GBDT on the "GBDT-friendly" benchmark; ablation shows removing context labels, key-only representation or L2 distance drops performance back to MLP level — [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/4ef594af0d9a519db8fb292452c461fa-Paper-Conference.pdf); [code](https://github.com/yandex-research/tabular-dl-tabr/blob/d628ec7e1c0a66011473021034e7dd4a77740112/bin/tabr.py)
- **PoET-2 (Truong & Bepler; NeurIPS 2025) [fetched].** Retrieval-augmented protein foundation model; hierarchical encoder equivariant to the order of homologs in the context; dual causal + masked objectives; 182M parameters. Reported: SoTA zero-shot variant-effect prediction and SoTA supervised few-shot function learning; the paper states that scaling PLM parameters "primarily seems to benefit only structure prediction" while showing neutral or negative impact on function prediction; structure conditioning gave "little to no benefit" on supervised function prediction — [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/4d19160864e6a644496d61b21c7e015a-Paper-Conference.pdf); [poster](https://neurips.cc/virtual/2025/poster/116850)
- **Protriever (Weitzman et al.; ICML 2025) [fetched].** End-to-end differentiable homolog retrieval trained jointly with the downstream task via vector search; reported SoTA among sequence models that rely on MSA retrieval for fitness prediction while being two orders of magnitude faster — [PMLR v267](https://proceedings.mlr.press/v267/weitzman25a.html)
- **Tranception (Notin et al.; ICML 2022) [fetched].** Autoregressive PLM plus inference-time retrieval of homologs (MSA-derived) for fitness prediction — [ICML 2022 poster](https://icml.cc/virtual/2022/poster/18181)
- **ProtEx (Shaw et al., Google DeepMind) — NOT tier-one: bioRxiv 2024 preprint; an OpenReview entry exists but I found no acceptance record.** Retrieves class-conditioned positive and negative exemplars by BLAST and trains a model to decide whether the query shares the label of the positive exemplars; multi-sequence pretraining. Reported SoTA on EC, GO and Pfam, with the largest gains on rare classes and sequences far from training, plus generalisation to unseen labels — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.30.596539v2); [code](https://github.com/google-deepmind/protex)
- **kNN-LM (Khandelwal et al.; ICLR 2020) [known].** Interpolate model distribution with a kNN distribution over a datastore of hidden states — [arXiv 1911.00172](https://arxiv.org/abs/1911.00172)
- **Non-Parametric Transformers (Kossen et al.; NeurIPS 2021) [known].** Attention between datapoints, including their labels, so prediction for one example uses others in-context — [arXiv 2106.02584](https://arxiv.org/abs/2106.02584)
- **Memory Layers at Scale (Berges et al.; ICML 2025) [fetched].** Trainable product-key key–value memory adds parameters without FLOPs; reported to beat dense models with >2× compute and compute/parameter-matched MoE; gains "especially pronounced for factual tasks" — [PMLR v267](https://proceedings.mlr.press/v267/berges25a.html). Builds on product-key memory (Lample et al.; NeurIPS 2019) [known] — [arXiv 1907.05242](https://arxiv.org/abs/1907.05242)

### Inferences
- Concrete plug-in for HiGO ("TabR-for-GO"): store for each training protein a pooled ESM2 embedding (35M: 480-d; 3B: 2560-d) and its propagated GO label vector. For a query, retrieve top-k (k≈16–32) by L2/cosine on a learned key, excluding the protein itself and anything above an identity cutoff during training to simulate remote-homolog conditions. Value per neighbour = W·(label multi-hot, IA-weighted) + T(k_q − k_i). Feed the attended vector (a) as an extra token the GO-term queries can attend to, and (b) as a per-term prior logit added before the MCM. This replaces the post-hoc homology gate with a learned, query-dependent mixture, which is exactly what kNN-LM interpolation does in a fixed form.
- Cost: datastore of ~100k proteins × 480 floats ≈ 0.2 GB (fp32), well inside the disk budget; label matrix stored sparse. Retrieval on GPU with FAISS flat L2 is negligible at this size. Training cost roughly +10–20% (inference).
- Expected effect: largest on close-homolog proteins (GO-GPT's stronghold), plausibly small or negative on remote homologs unless the model learns to down-weight distant neighbours, which the TabR attention can do through the similarity term (inference; TabR showed context labels were the critical ingredient).
- Training trick: drop retrieval for a random fraction of batches and use identity-filtered retrieval for others so the model does not collapse onto copy-the-neighbour, preserving the remote-homolog strength.
- Memory layers are a speculative alternative: they give the 35M model cheap extra parametric capacity for memorising family→function facts. No protein evidence; low priority.

### Gaps
- No tier-one paper reports in-network retrieval specifically for GO term prediction with CAFA-style evaluation; ProtEx (the closest) is a preprint.
- I did not verify whether the "Retrieved Sequence Augmentation (RSA)" protein paper was accepted at a tier-one venue; treat it as unverified.

---

## Q3. Protein representation learning at ICLR/ICML/NeurIPS: structure tokens, text, scaling

### Takeaway
On the standard DeepFRI GO benchmark, moving ESM2 from 35M to 650M gives about +0.05 Fmax in every aspect, structure-aware tokens give a further +0.01–0.03, and the best BP/CC numbers come from fusing a PLM with a structure GNN (ESM-GearNet). The ICML 2024 transfer-learning study warns that many tasks do not improve with scale, but the GO numbers below show GO is not one of those tasks at the 35M→650M step.

### Cited Findings
- **SaProt (Su et al.; ICLR 2024, spotlight) [fetched].** Structure-aware vocabulary of 21×21=441 tokens combining amino acid and Foldseek 3Di tokens; ~40M sequences+structures. Reported GO Fmax (MF/BP/CC): ESM-2 35M 0.616/0.416/0.404; SaProt 35M 0.642/0.431/0.418; ESM-2 650M 0.670/0.473/0.470; SaProt 650M 0.682/0.486/0.479. SaProt using ESMFold structures instead of AF2: 0.678/0.480/0.474 — [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/1c42513b8895ab11fbbb5b7e8e6b6b02-Paper-Conference.pdf); [GitHub results table](https://github.com/westlake-repl/Saprot)
- Reported in the same table: ESM-GearNet 0.676/0.516/0.507; SaProt-GearNet 0.678/0.522/0.508; GearNet alone 0.644/0.481/0.476 — [SaProt ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/1c42513b8895ab11fbbb5b7e8e6b6b02-Paper-Conference.pdf). GearNet itself: Zhang et al., ICLR 2023 [known] — [arXiv 2203.06125](https://arxiv.org/abs/2203.06125)
- **ProSST (Li et al.; NeurIPS 2024) [fetched].** Learned structure quantiser (vocab up to 4096; 2048 used) + disentangled sequence/structure attention; 110M parameters. Reported GO Fmax 0.682/0.492/0.501, i.e. matches SaProt-650M on MF and beats it on BP and CC at ~1/6 the size; ESM-GearNet (690M) is slightly better on BP — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf); [code](https://github.com/ai4protein/ProSST)
- **Feature Reuse and Scaling (Li, Amini, Yue, Yang, Lu; ICML 2024) [fetched].** 370 transfer experiments with CARP and ESM: almost all tasks benefit from pretraining, but for the majority "performance does not scale with pretraining" and relies on low-level features learned early; secondary structure is the exception. GO was not among their tasks (they used e.g. subcellular localisation, fitness regression) — [PMLR v235](https://proceedings.mlr.press/v235/li24a.html); [bioRxiv full text](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v2.full-text)
- PoET-2 (NeurIPS 2025) states that parameter scaling shows "neutral or even negative impacts on fitness modeling and function prediction" — [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/4d19160864e6a644496d61b21c7e015a-Paper-Conference.pdf)
- **ProtST (Xu et al.; ICML 2023) [fetched].** Protein-text pretraining on ProtDescribe with unimodal masking, multimodal alignment and multimodal mask prediction; supports zero-shot classification and text-based functional retrieval — [PMLR v202](https://proceedings.mlr.press/v202/xu23t.html)
- **Prot2Text-V2 (NeurIPS 2025) [fetched].** Contrastive protein–text alignment plus instruction tuning to generate free-text function descriptions; emphasises low-similarity evaluation — [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/25a3e5c979a5507fd534360ab04f72bd-Paper-Conference.pdf)

### Inferences
- The SaProt table is the most direct tier-one evidence on PLM size for GO: 35M→650M ESM2 gives +0.054 MF, +0.057 BP, +0.066 CC Fmax (unweighted, DeepFRI split). This conflicts in spirit with the ICML 2024 "scaling does not help most tasks" message, but that study did not include GO. Given ESM2-3B embeddings are already cached, the cheapest large gain is likely to run HiGO's head on frozen (or lightly LoRA-tuned) 3B per-residue features, accepting that per-residue caching of 3B for all proteins may not fit the 4 GB disk (2560-d × average 400 residues × fp16 ≈ 2 MB per protein); computing on the fly on the H100s is the workaround.
- Structure tokens: Foldseek 3Di strings are one character per residue, so a SaProt-style input costs almost no disk; the bottleneck is obtaining AFDB structures (download or ESMFold). SaProt-35M weights exist, so swapping ESM2-35M→SaProt-35M in HiGO is a low-effort pilot with a reported +0.015–0.026 Fmax at 35M (reported on DeepFRI GO, not CAFA).
- ESM-GearNet's BP gain (+0.04 over ESM-2 650M) suggests structure helps BP most, but PoET-2 found structure conditioning gave little benefit on supervised function prediction; the evidence is mixed.

### Gaps
- No tier-one paper reports ESM2-3B vs 650M on GO with CAFA-style temporal splits and IA-weighted Fmax.
- I did not re-verify DPLM (ICML 2024) or "Training compute-optimal protein language models" (NeurIPS 2024) GO numbers this session.

---

## Q4. Zero-shot / open-vocabulary GO prediction with GO text so rare and unseen terms get parameters

### Takeaway
The tier-one pattern most applicable is SiameseXML (ICML 2021): a shared text encoder gives every label an embedding from its description, and a per-label free classifier vector is added on top for labels with enough data. Applied to GO, this gives rare deep terms (which dominate IA weight) a sensible initial classifier from their definitions, while frequent terms keep free parameters.

### Cited Findings
- **SiameseXML (Dahiya et al.; ICML 2021) [fetched].** Probabilistic model motivating a modular design: Siamese embeddings of inputs and label text, plus high-capacity per-label extreme classifiers; scales to 100M labels; targets few-shot labels ("<5 training points") where label text helps; reported 2–13% more accurate than leading XML methods on public benchmarks — [PMLR v139](https://proceedings.mlr.press/v139/dahiya21a.html); [PDF](https://proceedings.mlr.press/v139/dahiya21a/dahiya21a.pdf)
- Later work notes SiameseXML does not handle labels entirely absent from training (MACLR, arXiv preprint, not tier-one verified) — [arXiv 2112.08652](https://ar5iv.labs.arxiv.org/html/2112.08652)
- **OntoProtein (Zhang et al.; ICLR 2022) [fetched arXiv; venue known].** Jointly embeds GO knowledge graph (nodes described by GO text) and protein sequences with contrastive learning and knowledge-aware negative sampling — [arXiv 2201.11147](https://arxiv.org/pdf/2201.11147)
- **ProtST (ICML 2023) [fetched]** enables zero-shot protein classification by text — [PMLR v202](https://proceedings.mlr.press/v202/xu23t.html)
- Not tier-one ML venue (labelled): ProtNote (Bioinformatics 2025) uses GO text descriptions for zero-shot GO prediction and notes ~298 GO terms added and ~591 deprecated per year — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12054973/); FunBind (Bioinformatics) aligns five modalities for zero-shot unseen GO terms — [Ovid](https://www.ovid.com/journals/bioinf/fulltext/10.1093/bioinformatics/btag356~a-unified-multimodal-model-for-generalizable-zero-shot-and)
- The BioReason-Pro release provides Qwen3-Embedding-4B vectors for 43,248 GO terms (177 MB) — [GitHub README](https://github.com/bowang-lab/BioReason-Pro)

### Inferences
- Plug-in for HiGO: term query q_t = W_text·e_text(t) + mean_ancestors(q) + u_t, where e_text is the frozen text embedding of the GO name+definition and u_t is a free residual initialised to zero with stronger weight decay for rare terms (SiameseXML-style "text + free classifier"). This keeps the existing ancestor-mean design and adds information the GAT could not (the GAT only re-mixed existing term embeddings).
- Cost: negligible compute; 177 MB if the released Qwen3 vectors are reused, or a few MB if a small biomedical text encoder is run once.
- Expected gain concentrates on rare deep terms, which carry most IA weight; also enables predicting terms absent from training (useful because CAFA test sets accumulate new annotations).

### Gaps
- No tier-one paper reports text-initialised GO classifiers evaluated with IA-weighted Fmax.

---

## Q5. Long-tail / class-imbalance methods with theory, and positive-unlabelled learning for incomplete GO annotations

### Takeaway
Logit adjustment (ICLR 2021) and balanced softmax (NeurIPS 2020) give Fisher-consistent corrections for label-frequency skew via an additive log-prior term, which is a one-line change to HiGO's logits. For missing annotations, the tier-one single-positive multi-label (SPML) literature (NeurIPS 2022, ICML 2023) and PU-with-selection-bias (ICLR 2019) argue against "assume negative" and favour soft or ranked treatment of unobserved labels.

### Cited Findings
- **Logit adjustment (Menon et al.; ICLR 2021) [known].** Post-hoc or loss-time adjustment by τ·log prior; Fisher-consistent for balanced error — [arXiv 2007.07314](https://arxiv.org/abs/2007.07314)
- **Balanced Meta-Softmax (Ren et al.; NeurIPS 2020) [known]** — [arXiv 2007.10740](https://arxiv.org/abs/2007.10740)
- **LDAM-DRW (Cao et al.; NeurIPS 2019) [known].** Label-distribution-aware margin ∝ n_j^(−1/4) plus deferred re-weighting — [arXiv 1906.07413](https://arxiv.org/abs/1906.07413)
- **nnPU (Kiryo et al.; NeurIPS 2017) [known].** Non-negative risk estimator for deep PU learning — [arXiv 1703.00593](https://arxiv.org/abs/1703.00593)
- **PU learning with a selection bias (Kato, Teshima, Honda; ICLR 2019) [fetched].** When labelled positives are not selected completely at random, the Bayes classifier is only partially identified; they learn a score preserving the order of the class posterior and threshold it — [OpenReview](https://openreview.net/forum?id=rJzLciCqKm); [ICLR poster](https://iclr.cc/virtual/2019/poster/1024)
- **SMILE: One Positive Label is Sufficient (Xu et al.; NeurIPS 2022) [fetched].** Single-positive multi-label learning via label enhancement; reviews that "assume negative" (AN) introduces false negatives and that expected-positive regularisation and label smoothing reduce the damage — [NeurIPS 2022 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/888a66ed219c281f448babae80f3b8e8-Paper-Conference.pdf)
- **MIME: Revisiting Pseudo-Label for SPML (Liu et al.; ICML 2023) [fetched].** Gives conditions under which pseudo-labels make SPML learnable and proves MIME's pseudo-labels approximately converge to the fully-supervised case — [PMLR v202](https://proceedings.mlr.press/v202/liu23ar.html)
- Not tier-one (labelled): Cole et al. "Multi-Label Learning from Single Positive Labels" (CVPR 2021) introduced AN vs expected-positive regularisation baselines — [arXiv 2106.09708](https://ar5iv.labs.arxiv.org/html/2106.09708); entropy-maximisation for unobserved labels (Zhou et al., 2022, venue not verified); Kanehira & Harada multi-label PU ranking (CVPR 2016) — [CVF](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kanehira_Multi-Label_Ranking_From_CVPR_2016_paper.pdf)

### Inferences
- GO annotations are a textbook selected-not-at-random PU problem: well-studied organisms and assay-friendly functions are over-annotated. Kato et al.'s result suggests aiming for correct *ranking* of per-term scores and then tuning a threshold, which fits Fmax (a threshold sweep) well.
- Concrete plug-ins: (1) add τ·log(prior_t) logit adjustment, with priors computed on propagated annotations, and tune τ on validation IA-Fmax, since IA weighting already up-weights rare terms and the two may overlap; (2) for unannotated terms whose *ancestor* is annotated (the most likely missing positives), replace hard 0 targets with a soft target or down-weight their negative loss (nnPU-style); (3) self-distil: a teacher's high-confidence predictions on training proteins become soft positives for the student (MIME-style pseudo-labels).
- Cost: all are loss-level changes, essentially free.
- Risk: CAFA ground truth is also incomplete, so a model that predicts truly-correct-but-unannotated terms is penalised; PU corrections can lower measured Fmax even if they improve biology (inference).

### Gaps
- I found no tier-one paper that combines logit adjustment with an IA- or information-content-weighted loss, so their interaction is untested.
- No tier-one PU method is evaluated on GO; the selection-bias mechanism for GO is an inference.

---

## Q6. Calibration and selective prediction for Fmax thresholding

### Takeaway
Fmax tunes one global threshold, so cross-term calibration matters: a rare-term score of 0.4 must mean the same as a frequent-term 0.4. Koyejo et al. (NeurIPS 2015) justify a single shared threshold on calibrated marginals; Conformal Risk Control (ICLR 2024) gives a distribution-free way to pick that threshold to bound per-protein false-negative rate.

### Cited Findings
- **Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster; ICLR 2024, spotlight) [fetched].** Controls the expectation of any monotone loss; for multi-label sets C_λ(x) = {k : f(x)_k ≥ 1−λ} it picks λ̂ so that E[FNR] ≤ α. Reported on MS-COCO with α=0.1: mean FNR 0.0996 — [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html); [arXiv 2208.02814](https://arxiv.org/pdf/2208.02814); [code](https://github.com/aangelopoulos/conformal-risk)
- Koyejo et al. (NeurIPS 2015): shared threshold is Bayes-optimal for instance/micro-averaged F on calibrated marginals — [NeurIPS 2015 PDF](https://proceedings.neurips.cc/paper/2015/file/85f007f8c50dd25f5a45fca73cad64bd-Paper.pdf)
- **Temperature scaling (Guo et al.; ICML 2017) [known]** — [arXiv 1706.04599](https://arxiv.org/abs/1706.04599). **Selective classification for deep nets (Geifman & El-Yaniv; NeurIPS 2017) [known]** — [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)
- MBM (ICLR 2022) claims box containment probabilities are "calibrated conditional probabilities" — [OpenReview](https://openreview.net/forum?id=tyTH9kOxcvh)

### Inferences
- Plug-in: fit per-aspect (and optionally per-depth-bin) temperature or isotonic calibration on a held-out temporal split before the Fmax sweep. Because HiGO and a homology/retrieval branch produce scores on different scales, calibrating each branch before mixing is likely to matter more than the mixing rule itself.
- Selective prediction is directly relevant to CAFA's "precision averaged over proteins with ≥1 prediction" convention: abstaining on low-confidence proteins raises precision. A per-protein confidence (e.g. max score or retrieval similarity) with a learned abstain threshold exploits this (inference; check that the challenge rules count abstentions this way).
- Conformal risk control is useful for reporting guarantees, but optimising Fmax directly with a validation sweep is simpler and what the metric rewards.

### Gaps
- No tier-one paper studies calibration under IA-weighted Fmax specifically.

---

## Ranked shortlist (inference; synthesises the above)

| Rank | Mechanism (paper, venue/year) | Failure mode targeted | Cost | Evidence |
|---|---|---|---|---|
| 1 | In-network labelled-neighbour attention (TabR, ICLR 2024; PoET-2, NeurIPS 2025; ProtEx preprint) | Loses to GO-GPT on close homologs; post-hoc gate cannot learn per-term trust | Low–moderate (0.2 GB datastore, +10–20% train) | Strong in tabular; moderate in proteins; GO-specific only in a preprint |
| 2 | Larger encoder: frozen/LoRA ESM2-650M/3B features (SaProt ICLR 2024 table) | 35M capacity gap vs 3B | Moderate GPU, on-the-fly features due to disk | Reported +0.05–0.07 Fmax 35M→650M on DeepFRI GO |
| 3 | GO-text label encoder + free residual (SiameseXML, ICML 2021; OntoProtein ICLR 2022; ProtST ICML 2023) | Rare/deep/unseen terms have weak parameters | Very low (reuse 177 MB GO embeddings) | Strong in XML; indirect for GO |
| 4 | Logit adjustment / LDAM margins (ICLR 2021; NeurIPS 2019) | Frequency skew across terms | Free | Strong theory; untested with IA weights |
| 5 | PU / soft negatives for unannotated descendants-of-annotated (ICLR 2019; NeurIPS 2022; ICML 2023) | Missing labels treated as negatives | Free | Moderate; none on GO |
| 6 | Calibrate per branch, shared threshold (NeurIPS 2015; ICML 2017; ICLR 2024) | Scale mismatch across terms/branches | Free | Strong theory |
| 7 | Structure tokens (SaProt ICLR 2024; ProSST NeurIPS 2024) | Remote homologs where structure is conserved | Moderate (structures needed) | Reported +0.015–0.026 Fmax at 35M |
| 8 | Box label embeddings (MBM, ICLR 2022) | Hierarchy via geometry rather than MCM | Moderate | Moderate, not on GO |
| 9 | HypStructure regulariser (NeurIPS 2024) | Representation ignores GO distances | Low | Weak–moderate, vision only |
| 10 | SPL (NeurIPS 2022) / pseudo-semantic loss (NeurIPS 2023) | Consistency (already handled by MCM) | High | Gains in exact match, not Hamming |
