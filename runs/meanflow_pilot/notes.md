# MeanFlow one-step label refiner: pilot for the HiGO-R ladder

Date: 2026-10-12. Code: `src/cafa6/meanflow_refiner.py`. Outputs: this directory. GPU 0, under heavy sharing.
**Verdict: DROP.** On the selection split, the best MeanFlow beats the per-term fusion control by +0.0014 IA-weighted
Fmax, and the bar was +0.005. A non-generative network of the same size, trained with plain BCE, beats every
MeanFlow variant.

## 1. The method (MeanFlow, Geng, Deng, Bai, Kolter, He; NeurIPS 2025 oral; arXiv 2505.13447)

Flow Matching moves samples along z_t = (1 − t) x + t e, with data x at t = 0 and source e at t = 1. The conditional
velocity is v_t = e − x, and a network learns the instantaneous velocity v(z, t). MeanFlow instead learns the
**average velocity** over an interval [r, t]:

    u(z_t, r, t) = 1/(t − r) ∫_r^t v(z_τ, τ) dτ.

Differentiating (t − r) u = ∫_r^t v dτ with respect to t, holding r fixed, gives the **MeanFlow identity**

    u(z_t, r, t) = v(z_t, t) − (t − r) d/dt u(z_t, r, t),
    d/dt u = v · ∂_z u + ∂_t u,

which is a Jacobian-vector product of u_θ with the tangent (v, 0, 1) for the inputs (z, r, t). One
`torch.func.jvp` call computes it. The training loss is

    L(θ) = E ‖ u_θ(z_t, r, t) − sg(u_tgt) ‖²,   u_tgt = v_t − (t − r) (v_t ∂_z u_θ + ∂_t u_θ).

The stop-gradient avoids double backpropagation. When r = t the loss reduces to Flow Matching.

The paper's other choices, all taken from its ImageNet ablation defaults and used here:
- r and t are each drawn from a logit-normal lognorm(−0.4, 1.0), swapped so that t ≥ r, and r is set to t for 75%
  of the batch (25% have r ≠ t).
- The loss uses adaptive weighting: the weight sg(w) = 1/(‖Δ‖² + c)^p multiplies ‖Δ‖², with p = 1 and c = 1e-3.
- The network is conditioned on (t, t − r) through sinusoidal embeddings.
- The model weights are kept as an exponential moving average (EMA).

**One-step sampling** is z_0 = z_1 − u_θ(z_1, 0, 1). Classifier-free guidance was not used.

## 2. How it was adapted to GO labels

- **Data.** x = 2y − 1 ∈ {−1, +1}^7901 over the GO-GPT-bench vocabulary.
- **Condition c.** For each term: HiGO probability (`runs/gogptbench_35m_lora/pred_*.npy`), the sharpened kNN prior
  p_knn and the fusion prior p0. For each protein: max identity and has_hit.
- **Sharpened kNN.** Computed here from `diamond_hits_valtest.parquet` and `Y.npz`, with
  w_j = bits_j · exp((pident_j − max pident)/2) over the top 10 hits by bits, normalised, and
  p_knn = Σ_j w_j y_j. Only vocabulary terms are used, which is why test kNN scores 0.684 here against the red
  team's 0.698 with full label sets.
- **Network u_θ(z, r, t | c), 6.75M parameters.** A global path takes linear views of the whole state z and of
  p0, plus protein features and time embeddings, through an MLP to a 256-d protein vector g. That vector is where
  label co-occurrence can enter. A local path applies a shared two-layer per-term MLP to [z_j, term features],
  plus a learned 64-d term embedding and g. The output is the local head plus a low-rank decoder of g. Both output
  layers start at zero, so u = 0 at initialisation.
- **Source distribution (two flavours).**
  - `mf_noise`: e ~ N(0, I), the standard MeanFlow setup.
  - `mf_prior_sσ`: e = (2 p0 − 1) + σ ε, so the flow runs **from the fusion prior to the labels**. The derivation
    allows any source distribution as long as (x, e) pairs are drawn jointly given c. With u = 0 the one-step sample
    returns the prior, which makes this a residual refiner of the control.
  - On fit-half proteins, p0 comes from a 2-fold cross-fit, so the refiner trains on priors as noisy as the ones it
    sees at selection time.
- **Scores.** Draw S ∈ {1, 8, 32} one-step samples. Map each to [0, 1] with clip((x̂ + 1)/2) and average over
  samples. Apply MCM using `go_ancestors.npy`.
- **Training.** AdamW at 1e-3 with 200 warm-up steps, batch 512, EMA 0.995, fp32 with TF32. Checkpoints were scored
  at 500, 1,000 and 1,500 steps.

**Controls on the same split and features:**
1. HiGO alone.
2. Sharpened kNN alone.
3. **The kill-criterion control: a per-term logistic fusion.** For term j, logit = x_ij · (w_aspect + δ_j), where x
   = [HiGO logit, kNN logit, p_knn, max identity, has_hit, kNN logit × identity, 1]. An L2 penalty on δ_j shrinks
   rare terms toward the aspect weights. λ ∈ {1e-3, 1e-2, 1e-1} was selected on B, giving 1e-3.
4. Extra row, the **discriminative twin**: the same network with z = 2 p0 − 1 fixed, trained with BCE on
   logit(p0) + output. It separates the generative objective from the extra capacity.

**Selection data.** The temporal validation set (TV) appeared in `data/proc_gogpt_tv` during the pilot but cannot be
used here. TV proteins are GO-GPT training proteins: HiGO was trained on them, and no HiGO predictions exist for
them. Everything was therefore fit on half A and selected on half B of the **random** val split of
`data/proc_gogpt` (13,305 proteins, split 50/50 by sequence cluster: 6,652 in A and 6,653 in B). All
selection-stage numbers below are val-based.

**Scorer.** `knn_sharp.Scorer`, an in-memory reproduction of cafaeval: prop=max, norm=cafa, GO 2023-01-01, CAFA-5
IA, thresholds 0.01–0.99, top 1,500 terms per protein. It matches cafaeval to 1e-9 per
`runs/knn_sharp/scorer_check.json`. As a further check, the HiGO test row here (0.4547) equals the cafaeval value
in `runs/bench_gogpt/summary.tsv`.

## 3. Results

IA-weighted Fmax is the mean of MF, BP and CC. Identity bins use the max DIAMOND identity to the training split.
The full grid of 35 selection rows is in `summary.tsv`.

**Selection split: random val, half B (n = 6,653).**

| method | config | mean | MF | BP | CC | ≥95% | 50–95% | <50% / no hit |
|---|---|---|---|---|---|---|---|---|
| HiGO alone | | 0.5294 | 0.6106 | 0.3904 | 0.5871 | 0.5370 | 0.5351 | 0.5141 |
| sharpened kNN alone | T=2, k=10 | 0.5476 | 0.5907 | 0.4805 | 0.5715 | 0.6997 | 0.5324 | 0.4088 |
| **per-term logistic fusion (control)** | λ = 1e-3 | **0.6169** | 0.6742 | 0.5103 | 0.6664 | 0.7022 | 0.6007 | 0.5548 |
| MeanFlow, noise source | 1,000 steps, S=32 | 0.6166 | 0.6740 | 0.5148 | 0.6611 | 0.7210 | 0.5944 | 0.5472 |
| MeanFlow, prior source σ=0.1 | 1,500 steps, S=32 | 0.6151 | 0.6725 | 0.5088 | 0.6641 | 0.7045 | 0.5957 | 0.5489 |
| **MeanFlow, prior source σ=0.3 (best)** | 500 steps, S=32 | **0.6183** | 0.6748 | 0.5128 | 0.6674 | 0.7100 | 0.5992 | 0.5530 |
| same, S=8 | 500 steps | 0.6162 | | | | | | |
| same, S=1 | 500 steps | 0.5921 | | | | | | |
| discriminative twin (BCE) | 1,000 steps | 0.6246 | 0.6806 | 0.5229 | 0.6703 | 0.7229 | 0.6025 | 0.5586 |

**Kill criterion on B.** The best MeanFlow minus the fusion is **+0.0014**, with a paired protein bootstrap (300
reps) 95% CI of [−0.0001, +0.0029]. The bar was +0.005, so the result is **drop**. Two things make even this
number generous to MeanFlow:
- It is the best of 27 MeanFlow settings (3 sources × 3 checkpoints × 3 values of S) chosen on B itself, against 3
  fusion settings.
- σ = 1.0 had already been dropped after the 200-step smoke run (0.588).

The twin beats the best MeanFlow by 0.006 with the same inputs, network and data. Whatever the extra network
capacity buys, the generative objective and the sampling cost part of it.

**Single test look.** This is GO-GPT temporal test, n = 8,630, logged in `runs/test_looks.md`. The configuration
was frozen in `frozen.json` before the look, and all models were fit on val half A.

| method | mean | MF | BP | CC | ≥95% | 50–95% | <50% / no hit |
|---|---|---|---|---|---|---|---|
| HiGO alone | 0.4547 | 0.5022 | 0.3394 | 0.5225 | 0.4730 | 0.4521 | 0.3959 |
| **sharpened kNN alone** | **0.6843** | 0.7045 | 0.6261 | 0.7221 | 0.8557 | 0.5092 | 0.3552 |
| per-term logistic fusion | 0.6388 | 0.6555 | 0.5813 | 0.6798 | 0.7527 | 0.5411 | 0.4360 |
| MeanFlow, prior source σ=0.3, 500 steps, S=32 | 0.6544 | 0.6772 | 0.5959 | 0.6900 | 0.7781 | 0.5424 | 0.4360 |
| discriminative twin | 0.6719 | 0.6949 | 0.6206 | 0.7004 | 0.8062 | 0.5485 | 0.4443 |

On test, MeanFlow minus fusion is +0.0155 [+0.0137, +0.0175]. **This does not reverse the verdict, for three
reasons.**
- The decision rule is evaluated on the selection split only.
- The twin is a further +0.018 above MeanFlow on test, so the gain over fusion comes from capacity and
  nonlinearity, not from the one-step generative sampler.
- The dominant fact on test is that **every model fit on the random val split loses to the zero-training kNN**
  (0.684). The loss is concentrated at ≥95% identity: fusion 0.753 against kNN 0.856. Models fit on random val learn
  to distrust close neighbours, because the random split has much weaker identity-to-label transfer (kNN ≥95% bin:
  0.70 on val against 0.86 on test). This is the transfer failure the red-team note predicted (§0, point 3). Any
  learned stacker for HiGO-R has to be fit on TV, which needs HiGO predictions for TV proteins from a model that
  never saw their post-T0 labels.

## 4. What this says about the plan

- **The Koyejo argument holds up empirically here.** The best calibrated per-term combiner of the same features
  (the twin) beats sampling-based marginals from a one-step generative model, at equal capacity. On val B the
  ranking is twin 0.6246 > best MeanFlow 0.6183 ≈ fusion 0.6169, and on test it is twin 0.672 > MeanFlow 0.654 >
  fusion 0.639.
- **Sampling costs accuracy.** One sample (S=1) loses 0.02–0.11 against S=32, because a single draw is near-binary
  and wastes the threshold sweep. Averaging 32 draws only recovers what a direct marginal predictor outputs in one
  pass.
- **Starting from the prior helps somewhat but not enough.** Flowing from the fusion prior (σ = 0.3) is the best
  MeanFlow variant on B. It degrades with more training (0.618 at 500 steps, 0.613 at 1,500), which suggests the
  label-space flow overfits 6.6k proteins in 7.9k dimensions.
- **Side finding for the ladder.** A small nonlinear per-term refiner (the twin) gives +0.0077 over the per-term
  logistic fusion on val B. That supports the L1/L4 direction (richer stacker or in-network retrieval prior), but
  only once it is fit on TV.

## 5. Runtime

The total was about 50 minutes of wall time on a contended GPU 0, with under 11 GB of GPU memory:
- 200-step smoke run: 7 min.
- An aborted first full run, which was CPU-bound on host-side batch gathering: about 10 min.
- Selection run: 16 min. Each MeanFlow config took 1.5–3 min for 1,500 steps plus sampling, and each scoring call
  took 2–25 s.
- Test look: 2.7 min.

Outputs total 52 KB.
