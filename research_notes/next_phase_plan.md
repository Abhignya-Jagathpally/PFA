# Next phase: Ontologue-Ladder (HiCLA) and FARO-PEFT, planned against current evidence

Date: 2026-10-26. Inputs read in full: `Ontologue_FARO_PEFT_Research_Pack.zip` (FARO_PEFT_README.md,
Ontologue_FARO_PEFT_Research_Prototype.ipynb = FARO_PEFT_Colab.ipynb, toy_smoke_test.log) and `ontologue_ladder.zip`
(README, DESIGN, LITERATURE, the `ontologue/` library, notebooks 01-03 = 02_train_and_evaluate.ipynb, the
random-stand-in run). Nothing has been run on real data in either pack.

## 1. What the two packs propose

| | Ontologue-Ladder (HiCLA) | FARO-PEFT |
|---|---|---|
| question | does using the GO hierarchy *inside* the adapter help proteins without close homologues? | can an adapter trained to survive deliberate retrieval removal recover functions retrieval misses? |
| encoder | frozen ESM-2, forward only, residue states from 4 intermediate layers | frozen ESM-2, last layer |
| adapter | ladder side network: one zero-gated residual bottleneck rung per read layer; rung k predicts depth band k; its beliefs FiLM-modulate rung k+1 | two rank-8 residue experts (de novo / evidence) mixed by a gate bounded by retrieval support |
| retrieval | none (deliberately) | 3-mer TF-IDF neighbours (toy stand-in for DIAMOND); retrieval logit added with a learned scale |
| training worlds | one | three: leave-self-out retrieval, leave-family-out retrieval, full outage; plus a "missed label" loss on terms the prior misses |
| head / output | per-term attention, ancestor-mean queries, max-constraint, IA-weighted MCLoss | per-term attention with GO-text n-gram queries, max-propagation |
| other levers | family-balanced weights n^-0.5; early stopping on the *remote* validation subset | ablations: no erasure, unrouted experts, no adapters |
| benchmark | DeepFRI nrPDB-GO 2019 (PDB chains; alignment-based identity flags; 1,717/3,416 test chains <30% identity; published numbers from PFresGO, TransFew, STAR-GO, DeepGO-SE) | toy synthetic; real mode on GO-GPT data with MMseqs2 30%/80% family split |
| evidence so far | random stand-in only; ladder rows slightly *below* the flat MLP (expected for a 3-epoch random encoder) | toy only; counterfactual training did *not* beat the no-erasure variant on validation AUPRC |

Both packs state their own limits clearly: no real-encoder result, proxies instead of alignment identity,
unknown evidence provenance of GO-GPT labels, and that the GO-GPT test and fresh24 are already inspected and must
not be used for model selection.

## 2. How they relate to what this repository has measured

- **FARO's central idea is partly in place already.** The forecasting model F1 withholds the kNN prior on 30% of
  batches (a retrieval-outage world) and is a residual on the prior. It now ties SPROF-GO on fresh24 (0.514 vs
  0.515) and leads it from 50% identity upward, but trails it below 50% (no hit 0.398 vs 0.454). What FARO adds
  that F1 lacks: a **leave-family-out** retrieval world, a **missed-label** loss, and **routed experts**. The
  family-out world targets exactly the bins where F1 is behind. The outage share itself was never ablated in F1.
- **The ladder's levers were partly tried here in other forms.** Frozen 3B with a layer mix did not pass the
  stacker rule (L5); LoRA/DoRA capacity did not matter (within 0.002); hierarchy heads add about 0.01 in the
  literature. New in HiCLA: per-rung reading of intermediate layers inside a side network, depth-band conditioning,
  family-balanced weights and remote-first early stopping. The last two are cheap and untested here.
- **Benchmarks.** GO-GPT test: inspected many times. fresh24: three of four registered looks used. Neither can
  select models. The forecasting validation splits (periods A and B) and TV v2 remain valid for selection, but the
  no-knowledge rows of A-valid are small (32/55/35). DeepFRI nrPDB-GO is a useful *external* benchmark (published
  comparators, many remote chains), but it is a 2019 cluster split of PDB chains, not temporal.
- **Encoder gap to SPROF-GO.** SPROF-GO uses ProtT5-XL; our best uses ESM2-650M. STAR-GO reports ProtT5 > ESM-2
  frozen by +0.044 MF / +0.019 CC. This alone may explain part of the remote-bin gap and must be controlled before
  attributing gains to an adapter.

## 3. Decisions the plan is built around

1. **Selection data only:** forecasting valid (A, B; all rows, no-knowledge rows secondary), a new
   family-disjoint "remote" validation subset (MMseqs2 30% identity / 80% coverage clusters, as FARO specifies), and
   DeepFRI valid. Never fresh24 or the GO-GPT test.
2. **Final confirmation:** a **prospective** time-forward test: freeze the models and code (checksums, seeds), then
   build the benchmark from GOA annotations dated after the freeze date with the fresh24 builder. The last fresh24
   look is kept for one retrospective number, clearly labelled as such.
3. **Every claim is per identity bin with the no-hit and <30% bins as the primary endpoint**, three seeds, paired
   bootstrap, Holm across the registered family, one control row per added idea.

## 4. Work plan

### Phase 0: intake and hygiene (about 1 day)
- Import both packs into the repo (`ladder/`, `faro/`), keep their code unchanged, add their unit/smoke tests to CI.
- Replace the k-mer and TF-IDF proxies with DIAMOND/MMseqs2 identity (both binaries are installed).
- Audit the evidence provenance of the GO-GPT training labels against the archived GOA extracts in `data/fresh24/`
  (share supported by experimental codes, share IEA-only or unsupported). Report; decide whether to train on the
  experimental subset as an ablation.
- Build the family-disjoint remote validation subset of the forecasting data.

### Phase 1: external benchmark baseline (about 1 day, GPU)
- Run the ladder pack's notebook 02 on DeepFRI nrPDB-GO with ESM2-650M, rows A-F, MF/BP/CC, 3 seeds (rows A and
  B first: they calibrate our setup against published frozen-embedding numbers).
- Add our own reference rows on the same split: sharpened kNN (DIAMOND), the F1 forecasting head adapted to a
  non-temporal task, and SPROF-GO if its released models run on the split.
- Gate: rows A/B within about 0.02 of published frozen-ESM-2 baselines, otherwise fix the pipeline before going on.

### Phase 2: pre-registered ablations inside the forecasting model (2-4 days, GPU)
All on F1-650M, selected on the Phase-0 validation sets, 3 seeds, kill threshold +0.005 on the primary endpoint
(no-hit + <30% IA-weighted Fmax on validation) with no loss above 0.005 at >= 95% identity:
- **E1 FARO worlds:** add the leave-family-out world; vary outage share {0, 0.15, 0.30, 0.5}; add the missed-label
  loss; then routed two-expert vs one adapter at matched parameters (FARO's own H1-H2).
- **E2 HiCLA encoder side:** replace the single-layer readout with the ladder (bands + conditioning); control: same
  ladder without conditioning (rows C/D).
- **E3 cheap levers:** family-balanced weights; remote-first early stopping.
- **E4 encoder:** ESM2-3B frozen; ProtT5-XL frozen (SPROF-GO's encoder) as the control for the SPROF-GO gap.
Run E3 and E4 first: they are cheap and change the baseline the other two must beat.

### Phase 3: combine, freeze, confirm (about 1 day + waiting time)
- Combine the passing components; one registered selection on validation.
- Freeze (code hash, weights, seeds); spend the last fresh24 look as a retrospective number.
- Register and build the prospective benchmark once enough post-freeze annotations exist (GOA releases are
  monthly; a first look is realistic after 4-6 months).
- Serve the frozen model in the web app after its self-test.

### Phase 4: writing
- The benchmark-hygiene draft stands on its own and can be submitted now.
- A method paper only if Phase 2 components pass and the prospective test confirms them; a negative result on
  FARO or HiCLA is reported as such (both packs say so explicitly).

## 5. Risks and how they are handled

| risk | handling |
|---|---|
| gains on validation do not transfer (seen with the stacker: +0.008 TV, -0.003 test) | larger, family-disjoint validation; three seeds; prospective final test |
| encoder choice explains the SPROF-GO gap | ProtT5-XL control row (E4) before attributing gains to adapters |
| GO-GPT labels include non-experimental evidence | provenance audit (Phase 0) and an experimental-only training ablation |
| ladder MCM is dense O(T^2) per protein; 7,901 terms is heavy | reuse the sparse MCM from `src/cafa6/model.py` |
| novelty overlap (retrieval-conditioned adapters, hierarchy conditioning) | literature check before any novelty claim; both packs already flag this |
