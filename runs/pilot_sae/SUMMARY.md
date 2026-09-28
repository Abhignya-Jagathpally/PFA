# Pilot SAE interpretability probe (runs/pilot_sae/)

Source checkpoint: `runs/dev_35m_lora` (HiGO-35M, LoRA r=8, d=256). Activations: `H = model.proj(model.plm(...).last_hidden_state)` (the LoRA-adapted, projected residue representation from `HiGO.forward`), residue positions only (`res_mask`), from 3000 validation proteins (1260244 residues total, kept in RAM only).

SAE: 256 -> 4096 -> 256, ReLU + L1 (0.003), 4000 Adam steps, unit-norm decoder columns, variance explained 1.00. 1/4096 latents never fire, 0 fire on 0.5%-20.0% of residues; the 50 latents closest to that range are analysed.

## Result

Best-term enrichment (P(term | latent fires in protein) / P(term), terms with >=30 proteins): mean 5.52x, against 2.59x for random protein sets of the same size (label permutation). 100% of latents exceed the 95th percentile of their null; 62% are above 5x. The 50 latents map to only 15 distinct terms (most common: GO:0004672 x17, GO:0140657 x8, GO:0015370 x5), so many latents are redundant detectors of a few large protein families.

This is protein-level co-occurrence. It shows that some directions in the adapted representation track specific functions; it does not show that particular residues cause a prediction, which is what the masking test in `explain.py` measured, so the two results are not directly comparable.

## Top 5 latents by GO-term enrichment

| latent | best GO term | aspect | enrichment | support (proteins) | P(term\|fires) | P(term) | n_fire_proteins | neighbor_frac | rel_runlen |
|---|---|---|---|---|---|---|---|---|---|
| 1063 | GO:0004497 | F | 10.15x | 50 | 0.169 | 0.017 | 260 | 0.03 | 0.85 |
| 1021 | GO:0015370 | F | 8.04x | 32 | 0.086 | 0.011 | 338 | 0.08 | 0.75 |
| 2334 | GO:0004497 | F | 7.75x | 50 | 0.129 | 0.017 | 271 | 0.00 | 0.80 |
| 3235 | GO:0015370 | F | 7.68x | 32 | 0.082 | 0.011 | 244 | 0.03 | 0.80 |
| 3191 | GO:0004672 | F | 7.51x | 38 | 0.095 | 0.013 | 284 | 0.13 | 0.70 |

## Contiguity (top 5 latents by GO enrichment, positional pattern)

`neighbor_frac` = fraction of fired residues (within a protein) that have an adjacent fired residue; `rel_runlen` = mean consecutive-run length / number of fired residues in that protein. Both close to 1 => one contiguous block (domain-like); both low => scattered single-residue hits.

The top latents fire on very few residues per protein (mean 1.8 among the top 5) and rarely on adjacent residues, so they look like point features (a site or short motif) rather than whole domains. With this few residues the contiguity numbers are noisy.

Caveats: pilot scale (3000 val proteins, 4000 SAE steps, 4096 latents); latents are selected and scored on the same sample; GO labels are propagated, so a latent's best term can be a broad ancestor; the representation was trained to predict these GO labels, so some enrichment is expected by construction; contiguity is a coarse proxy, not a domain-boundary check against InterPro.
