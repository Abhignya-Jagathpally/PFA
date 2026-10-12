# Structure, protein-text multimodality, and output decoding as levers to beat BioReason-Pro / GO-GPT

Scope: CAFA-style GO prediction (MF/BP/CC, IA/IC-weighted Fmax). Reference point is GO-GPT (ESM2-3B encoder + autoregressive GO-token decoder) and BioReason-Pro (Qwen3-4B reasoning LLM on top of GO-GPT + InterPro + ESM3). Our system: HiGO-35M (LoRA ESM2-35M, per-term entmax attention pooling, hierarchy max-constraint). Hardware: 2x H100 96 GB, ~4 GB free disk.

Labels used below: **[peer-reviewed]** = tier-one venue or journal; **[preprint]** = bioRxiv/arXiv only; **[workshop]** = workshop paper; **[inference]** = my reasoning, not a reported result.

Note on the 0.687 baseline: the BioReason-Pro preprint reports GO-GPT weighted Fmax 0.65 (greedy), 0.67 (10 samples, term frequency as probability), 0.70 (oracle best-of-10). The nearest published figure to 0.687 is 0.685 for BioReason-Pro (RL) on a 91-protein post-2026-03-20 temporal holdout in an independent analysis paper. I did not find 0.687 itself; the writer should treat it as the project's own measured number.

---

## (a) Structure-aware inputs: reported gains, costs, and cheap structure tokens

### Takeaway
On the DeepFRI/PDB GO benchmark, adding structure to a PLM yields +0.01 to +0.04 Fmax (largest for BP/CC with GearNet-style graphs; +0.01 to +0.03 for SaProt 3Di tokens), and the gain is larger for small models (SaProt-35M vs ESM-2-35M) than for 650M models. Structure tokens are cheap to obtain via ProstT5 (encoder + CNN, about 40 proteins/s on a mid-range GPU), so ~300k proteins is a few GPU-hours and ~150 MB of disk; the risk is that most reported gains come from AF2-derived 3Di and from benchmarks that are not temporal, high-homology splits like the GO-GPT test.

### Cited Findings
**SaProt (ICLR 2024, spotlight) [peer-reviewed]**
- SaProt combines residue tokens with Foldseek 3Di structure tokens into a "structure-aware vocabulary", pretrained on ~40M sequences+structures. — [ICLR 2024 poster](https://iclr.cc/virtual/2024/poster/19394)
- 650M models, Fmax (MF/BP/CC): ESM-2 0.670/0.473/0.470 vs SaProt 0.682/0.486/0.479 (+0.012/+0.013/+0.009); EC 0.868 → 0.882. — [SaProt paper, ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/1c42513b8895ab11fbbb5b7e8e6b6b02-Paper-Conference.pdf); [GitHub README](https://github.com/westlake-repl/SaProt?tab=readme-ov-file)
- 35M models, Fmax (MF/BP/CC): ESM-2-35M 0.616/0.416/0.404; SaProt-Seq-35M (same model, sequence only) 0.608/0.413/0.403; SaProt-35M 0.642/0.431/0.418 (+0.026/+0.015/+0.014 over ESM-2-35M). The SaProt-Seq control shows the gain comes from the 3Di input, not from the pretraining recipe. — [SaProt GitHub README](https://github.com/westlake-repl/SaProt?tab=readme-ov-file)
- Structure source matters: SaProt with ESMFold structures 0.678/0.480/0.474 vs AF2 structures 0.682/0.486/0.479. When evaluated on experimental PDB structures, the AF2-trained SaProt drops to 0.665/0.460/0.410 (CC −0.069), i.e. distribution shift between structure sources hurts. — [SaProt README](https://github.com/westlake-repl/SaProt?tab=readme-ov-file); [SaProt paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/1c42513b8895ab11fbbb5b7e8e6b6b02-Paper-Conference.pdf)
- SaProt's EC/GO numbers were re-evaluated in April 2024 after an evaluation bug (predictions flattened before Fmax); the paper's datasets differ from GearNet's, so cross-paper comparisons are not valid. — [SaProt issue #23](https://github.com/westlake-repl/SaProt/issues/23)
- SaProt uses official benchmark splits with "identity clustering and filtering"; no low-homology stratified GO result is reported. — [SaProt paper, App. D.3.2](https://proceedings.iclr.cc/paper_files/paper/2024/file/1c42513b8895ab11fbbb5b7e8e6b6b02-Paper-Conference.pdf)

**ProSST (NeurIPS 2024) [peer-reviewed]**
- Structure quantizer: residue-level local structure → dense embedding → k-means token (vocab 20 to 4096, default 2048), plus sequence-structure disentangled attention. — [ProSST NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf); [ProSST GitHub](https://github.com/ai4protein/ProSST)
- ProSST (110M) GO Fmax MF/BP/CC 0.682/0.492/0.501 vs ESM-2-650M 0.670/0.473/0.470 and SaProt-650M 0.682/0.486/0.479; ESM-GearNet (690M) 0.676/0.516/0.507. — [ProSST NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf)
- The bioRxiv v3 version reports different (lower) GO numbers for the same baselines (e.g. ESM-2 BP 0.345, ProSST BP 0.387), so evaluation code changed between versions. — [ProSST bioRxiv v3](https://www.biorxiv.org/content/10.1101/2024.04.15.589672v3.full.pdf)
- ProSST's quantizer requires a 3D structure (PDB file) as input, i.e. it needs full coordinates, not just sequence. — [ProSST GitHub](https://github.com/ai4protein/ProSST)

**ESM-GearNet / joint sequence-structure study [workshop + preprint]**
- ESM-2-650M 0.460/0.661/0.445 (BP/MF/CC) → ESM-GearNet serial fusion 0.488/0.681/0.464; with structure pretraining (Multiview Contrast) 0.514/0.683/0.497, SiamDiff 0.500/0.682/0.505. Serial fusion beat parallel and cross fusion; fine-tuning the PLM with a reduced learning rate was needed to avoid degradation. — [Zhang et al., arXiv 2303.06275 (preprint)](https://arxiv.org/html/2303.06275v2); original version at [ICLR 2023 MLDD workshop](https://iclr.cc/virtual/2023/12957)

**DPFunc (Nature Communications 2025) [peer-reviewed]**
- Domain-guided GNN over AlphaFold structures + ESM residue embeddings. CAFA-style time split, IC-weighted AUPR MF/CC/BP: DPFunc 0.606/0.535/0.367 vs ATGO+ 0.559/0.479/0.330, DeepGOPlus 0.512/0.468/0.305, BlastKNN 0.459/0.309/0.219. — [DPFunc supplementary](https://ora.ox.ac.uk/objects/uuid:33fc6776-167f-44ee-a62a-e97f5003e7b9/files/rcj82k866k); [DPFunc paper](http://www.nature.com/articles/s41467-024-54816-8.pdf)
- DPFunc "consistently outperforms other methods in nearly all cases" across sequence-identity thresholds to training, except BP at the 50% threshold where it is comparable to ATGO+. Structures were fetched from AFDB or predicted with AF2. — [DPFunc paper](http://www.nature.com/articles/s41467-024-54816-8.pdf); [Fig. 2](https://www.nature.com/articles/s41467-024-54816-8/figures/2)

**ESM3 (Science 2025) [peer-reviewed]**
- ESM3 tokenizes structure with a VQ-VAE (codebook 4096 + 4 special tokens) and has a function track built from InterPro/GO keywords (TF-IDF + LSH, 8 tokens per residue). — [ESM3 bioRxiv full text](https://www.biorxiv.org/content/10.1101/2024.07.01.600583.full); [Science](https://www.science.org/doi/10.1126/science.ads0018)
- GO-GPT authors tested ESM3 (without structure input) as the frozen encoder: "slightly worse than ESM2-3B"; ESM-C 600M, ESM2-650M, Profluent-E1 were close; ESM2-150M "noticeably weaker". — [BioReason-Pro bioRxiv, App. B.3.1 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- For BioReason-Pro, removing structure coordinates from ESM3 at inference "reduced performance marginally". — [BioReason-Pro bioRxiv, App. B.4.1 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)

**Getting structure tokens cheaply**
- ProstT5 (NAR Genomics & Bioinformatics 2024) translates AA → 3Di. An encoder + 2-layer CNN variant predicts 3Di for the 1,787-protein *M. jannaschii* proteome in 43.5 s (±2 s) on an RTX A5000 vs 48 h for ColabFold structure prediction (~3 orders of magnitude). Encoder embedding cost ~0.1 s/protein (human proteome ~35 min). — [ProstT5, NAR Genom Bioinform](https://academic.oup.com/nargab/article/6/4/lqae150/7901286); [PubMed](https://pubmed.ncbi.nlm.nih.gov/39633723/)
- Foldseek release 9 integrates ProstT5 (`createdb --prostt5-model weights --gpu 1`), "400-4000x faster than ColabFold"; resulting DBs contain only predicted 3Di (no Cα), so TM-score/LDDT and `--alignment-type 1` are unavailable. — [Foldseek release 9](https://github.com/steineggerlab/foldseek/releases/tag/9-427df8a); [Foldseek README](https://github.com/steineggerlab/foldseek)
- Foldseek AFDB50 (54M entries) needs ~151 GB RAM to search (35 GB with `--sort-by-structure-bits 0`); `foldseek databases` can download Alphafold/Swiss-Prot, Proteome, UniProt50, UniProt. — [Foldseek README](https://github.com/steineggerlab/foldseek)

### Inferences
- [inference] ProstT5-CNN at ~41 proteins/s on an A5000 implies ~2 GPU-hours for 300k proteins on one A5000, likely under 1 h per H100. 3Di strings cost 1 byte/residue, so ~300k × ~500 residues ≈ 150 MB, within the 4 GB budget. Full AFDB structures or AFDB50 Foldseek DBs are not feasible on our disk.
- [inference] SaProt's largest relative gain is at 35M scale (+0.015 to +0.026 Fmax), which matches our HiGO-35M backbone. Swapping the ESM2-35M LoRA backbone for SaProt-35M with ProstT5-predicted 3Di is the cheapest structure experiment. Caveat: SaProt was trained on AF2-derived 3Di, and its PDB-vs-AF2 drop (CC −0.069) shows sensitivity to token distribution shift; ProstT5-predicted 3Di is a third distribution, so the gain may shrink. I found no published SaProt + ProstT5-3Di GO result.
- [inference] All structure gains above are on the DeepFRI PDB benchmark (a few thousand PDB chains) or DPFunc's PDB/CAFA-style splits, not on a temporal test where 88% of proteins have ≥50%-identity training homologs. With close homologs, function is already well determined by sequence transfer; structure probably adds most where HiGO already wins (remote homologs) and least where GO-GPT wins (close homologs).
- [inference] Foldseek search over a self-built 3Di DB of our ~300k training proteins (via ProstT5) is a feasible structural kNN signal for the calibrator, and is aligned with the observation that grounded retrieval signals (InterPro) drive the CAFA-6 gain.

### Gaps
- No paper found that reports GO Fmax for SaProt/ProSST stratified by sequence identity (<30%, 30–50%, >50%).
- No reported GO result using ProstT5-predicted 3Di as SaProt input (vs AF2 3Di).
- ESM3 open-model GO Fmax on a standard benchmark: not found; only the GO-GPT authors' qualitative "slightly worse than ESM2-3B".
- DeepFRI-style contact-graph GO results were not re-fetched; DeepFRI numbers in this note come only via baselines in other papers.
- Exact disk size of Foldseek's Alphafold/Swiss-Prot DB was not found.

---

## (b) Protein-text multimodal models and GO definitions as label text

### Takeaway
Protein-text pretraining gives small supervised GO gains (ProtST: +0.006 to +0.015 Fmax over ESM-2-650M); its main benefits are retrieval and zero-shot classification. Using GO term definitions as label embeddings did not help GO-GPT and mainly helps rare or unseen terms in other methods, so it is unlikely to move IA-weighted Fmax on a high-homology temporal set by itself.

### Cited Findings
- ProtST (ICML 2023, oral): ProtST-ESM-2 vs ESM-2-650M on DeepFRI benchmark Fmax: GO-BP 0.472 → 0.482, GO-MF 0.662 → 0.668, GO-CC 0.472 → 0.487, EC 0.874 → 0.878. ProtST-ESM-2 beat vanilla ESM-2 on all 24 metrics. — [ProtST, PMLR v202](https://proceedings.mlr.press/v202/xu23t/xu23t.pdf); [arXiv 2301.12040](https://export.arxiv.org/pdf/2301.12040v2.pdf)
- ProtST enables zero-shot protein classification and function-based retrieval through text alignment. — [ProtST, PMLR](https://proceedings.mlr.press/v202/xu23t.html)
- OntoProtein (ICLR 2022): GO knowledge graph ProteinKG25 (612,483 entities, 4,990,097 triples), contrastive KG + masked LM pretraining; reports +4% (transductive) and +2% (inductive) on BPO over baselines in its function-prediction setting (ProtBert-era baselines). — [OntoProtein arXiv](https://arxiv.org/html/2201.11147v6); [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6505)
- ProtST surpasses OntoProtein on 22 of 24 benchmark metrics. — [ProtST, PMLR](https://proceedings.mlr.press/v202/xu23t/xu23t.pdf)
- ProTrek (Nature Biotechnology 2025): tri-modal (sequence/structure/text) contrastive model; outperforms ESM-2 in 9 of 11 downstream tasks; faster and more accurate than Foldseek and MMseqs2 at finding functionally related proteins; precomputed embeddings for >5B proteins on a server. — [ProTrek, Nat Biotechnol](https://www.nature.com/articles/s41587-025-02836-0); [ProTrek README](https://github.com/westlake-repl/ProTrek/blob/main/README.md)
- Evolla (10B protein-language generative model) is a preprint; I found no GO Fmax for it. — [Evolla GitHub](https://github.com/westlake-repl/Evolla)
- GO-GPT ablation: initializing GO token embeddings from text embeddings of term name + definition "did not yield measurable gains" for GO-GPT; in BioReason-Pro the same initialization only accelerated convergence of the GO graph encoder. — [BioReason-Pro bioRxiv, App. B.3.2 and B.4.1 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- STAR-GO [preprint]: SBERT-BioBERT definition embeddings + ontology autoencoder + topological-order transformer decoder; supervised Fmax 0.719 vs DeepGO-SE 0.722 on one ontology in its table; best zero-shot AUC on 13 of 16 held-out terms. — [STAR-GO arXiv 2512.05245](https://arxiv.org/html/2512.05245v1)
- MZSGO (Bioinformatics 2026): Qwen3-Embedding-4B encodes GO definitions and InterPro domain text, fused with ESM2-650M via gated fusion; reports gains on unseen labels. TransFew [preprint] similarly uses BioBERT definitions + GO graph autoencoder to help rare terms. — [MZSGO, Bioinformatics](https://academic.oup.com/bioinformatics/article-pdf/doi/10.1093/bioinformatics/btag168/67786304/btag168.pdf); [TransFew bioRxiv](https://www.biorxiv.org/content/10.1101/2024.03.11.584495v1-0)

### Inferences
- [inference] The protein-text gains (ProtST ≤ +0.015) are similar in size to SaProt's, but would require re-pretraining or switching backbones; ProtST checkpoints are ESM-2-650M scale, larger than HiGO-35M.
- [inference] Domain *text* (InterPro descriptions embedded with an LLM, as MZSGO does) is a cheaper way to add text than a protein-text foundation model, and is consistent with our finding that InterPro domains drive the CAFA-6 calibrator gain.
- [inference] GO definition embeddings are most useful as label-query initializations (see section c) and for rare/high-IA terms, which IA-weighted Fmax rewards; GO-GPT's null result suggests they will not help frequent terms.

### Gaps
- ProteinCLIP, Pinal and ProteinDT were not retrieved; I found no GO Fmax numbers for them in this session.
- ProTrek's GO Fmax values could not be extracted (the Nature Biotech page text did not include the table).
- No study found that measures protein-text pretraining gains on low-homology GO splits.

---

## (c) Output decoding: autoregressive vs set prediction, label queries, diffusion, DAG-constrained decoding

### Takeaway
No GO paper compares autoregressive, label-query, and diffusion decoders on the same data. GO-GPT's own numbers show its main weakness is calibration: greedy AR output is binary (0.65), sampling 10 times and using term frequencies gives 0.67, and an oracle best-of-10 reaches 0.70. In general multi-label learning, order-free or non-autoregressive decoders match or beat fixed-order AR decoders as label counts grow. A per-term scoring decoder with label-label interaction (label queries) plus a hierarchy constraint is the best-supported design for Fmax.

### Cited Findings
- GO-GPT: vocabulary of terms seen in ≥20 training proteins (1,470 MF + 7,500 BP + 1,007 CC = 9,977 tokens); depth-wise positional embeddings encode ontology order; organism embedding added to every GO token. — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- GO-GPT weighted Fmax: 0.65 greedy (all generated terms scored 1), 0.67 with 10 samples and term frequency as probability, 0.70 oracle best-of-10; InterLabelGO+ 0.63 (MF difference vs greedy GO-GPT not significant, p = 0.073). — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- GO-GPT: beam search did not outperform greedy or temperature sampling with probability aggregation; decoder scaling 70M → 270M gave small monotonic gains; intermediate encoder layers (~80–90% depth) beat the final layer. — [BioReason-Pro bioRxiv, App. B.3 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- Fmax "requires per-term probability estimates, but GO-GPT generates sequences of discrete GO tokens" (authors' own statement). — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- Order-Free Learning (AAAI 2020): seq2seq multi-label decoders depend on an imposed label order and suffer exposure bias; they "tend to overfit to label combinations and show poor generalization". — [Tsai & Lee, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/6066/5922)
- OTSeq2Set (EMNLP 2022): bipartite-matching (permutation-invariant) loss for an AR sequence-to-set model; +16.34% micro-F1 over the best seq2seq method on Wikipedia (31k labels). — [OTSeq2Set, ACL Anthology](https://aclanthology.org/2022.emnlp-main.377/)
- Non-autoregressive latent-variable multi-label model [preprint]: AR chain-order models work well for small label sets or top-ranked labels, but the NAR model "surpasses them by around 2% to 6%" when more labels are predicted or the label space is larger. — [arXiv 2012.01711](https://arxiv.org/html/2012.01711)
- Orderless recurrent models [preprint version of CVPR 2020 work]: fixed frequent-first/rare-first ordering improves over random ordering but still misaligns predictions with targets. — [arXiv 1911.09996](https://arxiv.org/pdf/1911.09996)
- Diffusion over label sets: CAD (NeurIPS 2025) denoises the multi-label matrix with co-occurrence guidance, but targets noisy-label learning, not large ontologies. — [CAD, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/d5c0f9585592bad5251133813893a6c0-Paper-Conference.pdf)
- Discrete-diffusion LLMs as multi-label classifiers [preprint, SDM 2026]: per-label yes/no scoring works; a joint set-refinement step "lowers F1" and is reported as a negative result. — [arXiv 2608.14649](https://arxiv.org/abs/2608.14649)
- STAR-GO [preprint] decodes GO terms in topological order with causal self-attention (DAG-aware AR) and reaches parity, not superiority, with DeepGO-SE on supervised Fmax. — [STAR-GO](https://arxiv.org/html/2512.05245v1)
- BioReason-Pro's GO graph encoder uses 200 learnable query embeddings via cross-attention (Perceiver-style compression of the ontology); 50 → 200 queries improved performance, saturating at 200. — [BioReason-Pro bioRxiv, App. B.4.1 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)

### Inferences
- [inference] The 0.65 → 0.67 → 0.70 ladder means GO-GPT loses ~0.02–0.05 Fmax to poor per-term scoring. HiGO already emits per-term probabilities and enforces the hierarchy max-constraint, so it does not pay this cost. The missing piece in HiGO is label-label dependency, which GO-GPT captures through autoregression.
- [inference] Best-supported decoder for us: a label-query (DETR/ML-Decoder-style) cross-attention head where each query is a GO term (initialized from definition text + ontology embedding), with a few self-attention layers among queries to model co-occurrence, followed by sigmoid per-term scores and the existing max-constraint. This keeps calibrated probabilities (good for Fmax) and adds the dependency modeling that AR provides. Cost: ~10k queries × residues cross-attention per protein; feasible on H100 if queries are restricted to candidate terms (e.g., top-k from HiGO or kNN).
- [inference] Discrete diffusion over ~10k-term label sets has no GO evidence and one negative result for joint refinement; low priority.
- [inference] A cheap test of the dependency hypothesis is to stack GO-GPT's sampled term frequencies with HiGO scores; the public BioReason-Pro test dataset includes GO-GPT predictions. — [wanglab/bioreason-pro-test-data](https://huggingface.co/datasets/wanglab/bioreason-pro-test-data)

### Gaps
- No paper found that compares AR vs set/label-query vs diffusion decoders on the same GO benchmark.
- No tier-one paper found that applies label-query (DETR/Query2Label/ML-Decoder) decoders to GO with Fmax results.
- GO-GPT's ordering ablation (depth-first vs frequency order) was not reported in detail.

---

## (d) Reasoning / LLM components: Fmax gain or mainly explanations?

### Takeaway
The LLM layer adds about +0.01 weighted Fmax over greedy GO-GPT (0.66 vs 0.65) and is no better than GO-GPT with 10-sample aggregation (0.67). An independent analysis shows the LLM's Fmax barely depends on the protein embedding and is driven by GO-GPT and InterPro text inputs. The reasoning layer's value is mainly explanations and expert preference, not Fmax.

### Cited Findings
- BioReason-Pro: SFT 0.64 greedy / 0.67 best-of-10; RL 0.66 greedy / 0.67 best-of-10 (weighted Fmax); GO-GPT greedy 0.65; InterLabelGO+ 0.63. Best unweighted Fmax 0.74 (abstract: 73.6%). — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full); [GitHub](https://github.com/bowang-lab/BioReason-Pro)
- The same preprint states GO-GPT reaches 0.76 unweighted Fmax vs 0.70 for InterLabelGO+, which is above BioReason-Pro's best unweighted 0.74; the decoding settings may differ, but as reported the LLM does not beat GO-GPT on unweighted Fmax. — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- RL shortened reasoning traces (mean −60 words) while increasing GO reward; human experts preferred or tied BioReason-Pro vs UniProt in 79% (SFT) and 73% (RL) of 162 proteins. — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- Training BioReason-Pro without GO-GPT inputs "produced weaker annotations"; GO-GPT predictions serve as scaffolding. — [BioReason-Pro bioRxiv, App. (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- Independent analysis [preprint, arXiv 2610.00898]: replacing BioReason-Pro's ESM3 embedding with one computed from a shuffled sequence leaves Fmax unchanged (0.855 vs 0.856 on 14,102 human proteins; 0.685 vs 0.682 on a 91-protein temporal holdout after 2026-03-20). RL raises Fmax from 0.829 (SFT) to 0.855 without increasing reliance on the protein embedding. — [When Do Biological Reasoning Models Use Their Biological Inputs?](https://arxiv.org/html/2610.00898)
- Homology dependence (mean per-protein F1 vs BLAST identity bins): BLAST ρ = 0.63, GO-GPT ρ = 0.46, BioReason-Pro SFT ρ = 0.41, RL ρ = 0.46. — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- Backbone: Qwen3-4B chosen over 8B (8B "moderately stronger" but much costlier); LoRA rank 128. — [BioReason-Pro bioRxiv, App. B.4.1 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)

### Inferences
- [inference] Reasoning is a ~0.01 Fmax lever at high cost (4B LLM, GPT-5 synthetic traces for 130k proteins, SFT + RL). It is not where a leap over GO-GPT will come from on Fmax; it acts as a learned re-ranker over GO-GPT + InterPro text, and a cheap calibrator over the same signals may capture most of that gain.

### Gaps
- No ablation found of BioReason-Pro with the reasoning trace removed (direct answer) at fixed inputs, which would isolate chain-of-thought's contribution.
- Fmax of BioReason-Pro stratified by identity bin (rather than mean per-protein F1) is not reported.

---

## Which combination is feasible on 2x H100 / ~4 GB disk and most likely to beat 0.687 on the GO-GPT temporal test?

### Takeaway
On a test where 88% of proteins have ≥50%-identity training homologs, the levers with the most evidence are: a larger sequence encoder, retrieval/homology signals, and better per-term calibration with label dependencies. Structure (ProstT5 3Di into SaProt-style input, plus Foldseek 3Di kNN) is cheap and feasible but has evidence for only +0.01 to +0.03 on non-temporal benchmarks, most likely concentrated on remote homologs. LLM reasoning and protein-text pretraining are low expected value for Fmax.

### Cited Findings
- GO-GPT found ESM2-150M "noticeably weaker" and larger encoders (650M/3B, ESM-C 600M) close to each other, with ESM2-3B best. — [BioReason-Pro bioRxiv, App. B.3.1 (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- SaProt gain over ESM-2 is larger at 35M (+0.015 to +0.026 Fmax) than at 650M (+0.009 to +0.013). — [SaProt README](https://github.com/westlake-repl/SaProt?tab=readme-ov-file)
- ProstT5-CNN 3Di generation is ~1000x faster than structure prediction (1,787 proteins in 43.5 s on an A5000). — [ProstT5](https://academic.oup.com/nargab/article/6/4/lqae150/7901286)
- GO-GPT gains 0.02 Fmax from sampling-based probabilities and has 0.05 oracle headroom over greedy. — [BioReason-Pro bioRxiv (preprint)](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full)
- GO-GPT predictions are released with the test data, enabling stacking/ensembling experiments. — [HF dataset](https://huggingface.co/datasets/wanglab/bioreason-pro-test-data)
- DPFunc's structure + domain model beats BlastKNN and ATGO+ on IC-weighted AUPR across identity thresholds, but requires full AF2 structures. — [DPFunc](http://www.nature.com/articles/s41467-024-54816-8.pdf)

### Inferences
- [inference] Ranked plan, by expected gain per unit cost on the GO-GPT temporal test:
  1. **Encoder scale-up**: HiGO's head on ESM2-650M or ESM-C 600M (LoRA, intermediate-layer features at ~80–90% depth). GO-GPT's ablation points to this as the main gap between a 35M model and a 3B-encoder model on close homologs. Fits easily on 2x H100.
  2. **Homology / retrieval blending tuned for close homologs**: DIAMOND/MMseqs2 kNN label transfer, weighted by identity, blended with HiGO per-term scores; plus a ProstT5 → Foldseek 3Di kNN over our own training set (~150 MB). Retrieval matters most where 88% of targets have close homologs.
  3. **Label-query decoder** with GO-definition-initialized queries over HiGO candidates, keeping per-term sigmoid scores and the max-constraint; captures the label dependencies that AR gives GO-GPT without losing calibration.
  4. **Structure tokens**: SaProt-35M or SaProt-650M with ProstT5-predicted 3Di as an extra ensemble member; expect gains mostly on remote homologs, possibly smaller than reported because of AF2-vs-ProstT5 3Di shift.
  5. **Ensemble with GO-GPT outputs** (released predictions) as a ceiling test: HiGO wins remote homologs, GO-GPT wins close ones, so a homology-gated ensemble is the most direct way past 0.687 [inference; complementary error profiles come from the project's own observations, not a cited source].
- [inference] Skip: LLM reasoning layer (~+0.01, very costly), protein-text pretraining from scratch (≤ +0.015), discrete diffusion decoding (no GO evidence), full-coordinate structure methods (DPFunc, GearNet, ProSST quantizer) because they need stored 3D structures that do not fit in 4 GB at 300k scale. ESMFold on the fly is possible in principle, but SaProt's ESMFold vs AF2 gap is small (0.678 vs 0.682 MF), so 3Di from ProstT5 is the cheaper route.

### Gaps
- No published result exists for any structure-aware or multimodal method on the GO-GPT temporal benchmark itself; all transfer estimates above are inferences.
- I found no study measuring whether structure gains persist when ≥50%-identity homologs are present (the GO-GPT test regime); DPFunc's identity-threshold analysis is the closest evidence.
