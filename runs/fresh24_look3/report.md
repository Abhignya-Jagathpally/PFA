# fresh24 look #1: scores of the frozen methods

Registration: `runs/fresh24_looks.md` (2026-10-19 18:40:00). Scorer: `src/cafa6/score_fresh24.py` (cafaeval protocol, GO 2023-01-01, CAFA-5 IA). Predictions: `runs/fresh24/preds/` (how each was made: `runs/fresh24/methods.md`). This look selects nothing.

Bootstrap: 1000 paired protein resamples (seed 0). Permutation tests: 10000 swaps (seed 1), Ontologue minus each method on {Fmax_w, Fmax, Smin_w} x {MF, BP, CC, mean} on FR24-P, Holm over the whole family (48 tests). For S-min lower is better, so a negative difference favours Ontologue.

## FR24-P (primary): 11,231 proteins (MF 3,924, BP 3,975, CC 6,507)

| method | proteins with predictions | Fw MF | Fw BP | Fw CC | Fw mean [95% CI] | F mean | S-min MF/BP/CC | S-min mean | Ontologue − method, Fw mean [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| SPROF-GO | 11,231 | 0.602 | 0.313 | 0.629 | 0.515 [0.510, 0.519] | 0.624 | 8.19/22.98/4.69 | 11.95 | -0.001 [-0.006, +0.005], p_Holm 1.0000 |
| Forecasting model, round 2 (F1-650M ensemble) | 11,231 | 0.611 | 0.334 | 0.596 | 0.514 [0.508, 0.519] | 0.609 | 7.87/24.00/5.16 | 12.34 | — |
| Forecasting model F1 (3-seed ensemble) | 11,231 | 0.592 | 0.291 | 0.579 | 0.487 [0.482, 0.493] | 0.593 | 8.18/24.32/5.42 | 12.64 | +0.026 [+0.023, +0.030], p_Holm 0.0048 |
| Ontologue (served predictor) | 11,230 | 0.571 | 0.303 | 0.564 | 0.479 [0.474, 0.485] | 0.582 | 8.67/23.96/5.33 | 12.65 | +0.034 [+0.031, +0.039], p_Holm 0.0048 |
| GO-GPT ten-sample | 11,231 | 0.524 | 0.273 | 0.607 | 0.468 [0.462, 0.473] | 0.590 | 8.94/23.59/5.06 | 12.53 | +0.046 [+0.041, +0.051], p_Holm 0.0048 |

Fw = IA-weighted Fmax, F = unweighted Fmax, S-min = IA-weighted S-min. Proteins without predictions count with recall 0.

## Permutation tests on FR24-P (Ontologue minus method; Holm over the family)

| method | Fmax_w MF | Fmax_w BP | Fmax_w CC | Fmax_w mean | Fmax MF | Fmax BP | Fmax CC | Fmax mean | Smin_w MF | Smin_w BP | Smin_w CC | Smin_w mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ontologue (served predictor) | +0.040 (p 0.0048) | +0.031 (p 0.0048) | +0.033 (p 0.0048) | +0.034 (p 0.0048) | +0.040 (p 0.0048) | +0.037 (p 0.0048) | +0.005 (p 0.1290) | +0.027 (p 0.0048) | -0.802 (p 0.0048) | +0.043 (p 1.0000) | -0.168 (p 0.0048) | -0.309 (p 0.0048) |
| Forecasting model F1 (3-seed ensemble) | +0.019 (p 0.0048) | +0.043 (p 0.0048) | +0.018 (p 0.0048) | +0.026 (p 0.0048) | +0.008 (p 0.0063) | +0.026 (p 0.0048) | +0.014 (p 0.0048) | +0.016 (p 0.0048) | -0.311 (p 0.0048) | -0.323 (p 0.0048) | -0.262 (p 0.0048) | -0.299 (p 0.0048) |
| SPROF-GO | +0.009 (p 0.2920) | +0.020 (p 0.0048) | -0.032 (p 0.0048) | -0.001 (p 1.0000) | -0.018 (p 0.0048) | +0.013 (p 0.0048) | -0.040 (p 0.0048) | -0.015 (p 0.0048) | -0.328 (p 0.0050) | +1.022 (p 0.0048) | +0.472 (p 0.0048) | +0.389 (p 0.0048) |
| GO-GPT ten-sample | +0.088 (p 0.0048) | +0.061 (p 0.0048) | -0.011 (p 0.0420) | +0.046 (p 0.0048) | +0.047 (p 0.0048) | +0.032 (p 0.0048) | -0.021 (p 0.0048) | +0.019 (p 0.0048) | -1.071 (p 0.0048) | +0.408 (p 0.0088) | +0.098 (p 0.2920) | -0.188 (p 0.0088) |

Smallest attainable permutation p: 1.0e-04; Holm p values are those times up to 48. Tests with Holm p < 0.05: 43 of 48; of these, in Ontologue's favour (higher F, lower S-min): 33.

## Secondary sets (descriptive)

| set | n | method | Fw MF/BP/CC | Fw mean | S-min mean | Ontologue − method [95% CI] |
|---|---|---|---|---|---|---|
| NK-all | 22,700 | Forecasting model, round 2 (F1-650M ensemble) | 0.611/0.334/0.641 | 0.529 | 12.10 | — |
| NK-all | 22,700 | SPROF-GO | 0.602/0.313/0.658 | 0.524 | 11.65 | +0.004 [+0.000, +0.009] |
| NK-all | 22,700 | Ontologue (served predictor) | 0.571/0.303/0.654 | 0.510 | 12.23 | +0.019 [+0.016, +0.023] |
| NK-all | 22,700 | Forecasting model F1 (3-seed ensemble) | 0.592/0.291/0.609 | 0.498 | 12.43 | +0.031 [+0.028, +0.035] |
| NK-all | 22,700 | GO-GPT ten-sample | 0.524/0.273/0.599 | 0.465 | 12.39 | +0.064 [+0.059, +0.069] |
| LK | 1,441 | Ontologue (served predictor) | 0.568/0.410/0.514 | 0.497 | 13.77 | -0.004 [-0.017, +0.008] |
| LK | 1,441 | Forecasting model, round 2 (F1-650M ensemble) | 0.581/0.408/0.489 | 0.493 | 13.28 | — |
| LK | 1,441 | Forecasting model F1 (3-seed ensemble) | 0.575/0.372/0.444 | 0.464 | 13.60 | +0.029 [+0.019, +0.038] |
| LK | 1,441 | GO-GPT ten-sample | 0.437/0.292/0.463 | 0.397 | 14.22 | +0.096 [+0.079, +0.115] |
| LK | 1,441 | SPROF-GO | 0.513/0.332/0.321 | 0.389 | 14.41 | +0.104 [+0.088, +0.122] |
| FR24-P-noloads | 9,176 | SPROF-GO | 0.604/0.321/0.675 | 0.533 | 11.88 | -0.009 [-0.014, -0.004] |
| FR24-P-noloads | 9,176 | Forecasting model, round 2 (F1-650M ensemble) | 0.611/0.331/0.630 | 0.524 | 12.39 | — |
| FR24-P-noloads | 9,176 | Forecasting model F1 (3-seed ensemble) | 0.592/0.289/0.608 | 0.496 | 12.66 | +0.028 [+0.024, +0.031] |
| FR24-P-noloads | 9,176 | Ontologue (served predictor) | 0.571/0.300/0.587 | 0.486 | 12.68 | +0.038 [+0.033, +0.042] |
| FR24-P-noloads | 9,176 | GO-GPT ten-sample | 0.524/0.267/0.617 | 0.469 | 12.53 | +0.055 [+0.049, +0.060] |
| FR24-P-GOcurrent | 11,231 | SPROF-GO | 0.592/0.313/0.634 | 0.513 | 11.33 | -0.000 [-0.005, +0.005] |
| FR24-P-GOcurrent | 11,231 | Forecasting model, round 2 (F1-650M ensemble) | 0.606/0.331/0.602 | 0.513 | 11.65 | — |
| FR24-P-GOcurrent | 11,231 | Forecasting model F1 (3-seed ensemble) | 0.589/0.292/0.583 | 0.488 | 11.94 | +0.025 [+0.022, +0.028] |
| FR24-P-GOcurrent | 11,231 | Ontologue (served predictor) | 0.566/0.299/0.569 | 0.478 | 11.96 | +0.035 [+0.031, +0.039] |
| FR24-P-GOcurrent | 11,231 | GO-GPT ten-sample | 0.517/0.275/0.618 | 0.470 | 11.89 | +0.043 [+0.038, +0.049] |

## FR24-P: Fw mean per identity bin to GO-GPT train+val (threshold chosen inside the bin)

| method | no_hit (n=2270) | <30 (n=415) | 30-50 (n=2633) | 50-80 (n=3036) | 80-95 (n=1527) | 95-99.9 (n=897) | 100 (n=453) |
|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.454 | 0.488 | 0.532 | 0.558 | 0.531 | 0.469 | 0.436 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.398 | 0.460 | 0.521 | 0.564 | 0.534 | 0.521 | 0.528 |
| Forecasting model F1 (3-seed ensemble) | 0.356 | 0.444 | 0.493 | 0.544 | 0.520 | 0.494 | 0.507 |
| Ontologue (served predictor) | 0.334 | 0.420 | 0.479 | 0.527 | 0.513 | 0.518 | 0.573 |
| GO-GPT ten-sample | 0.385 | 0.425 | 0.486 | 0.508 | 0.472 | 0.458 | 0.383 |

Mean per-protein F1_w at each method's own Fmax_w threshold of the whole set (FR24-P):

| method | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 |
|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.379 | 0.418 | 0.472 | 0.509 | 0.476 | 0.407 | 0.364 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.253 | 0.353 | 0.423 | 0.490 | 0.460 | 0.440 | 0.439 |
| Forecasting model F1 (3-seed ensemble) | 0.218 | 0.348 | 0.397 | 0.465 | 0.439 | 0.403 | 0.414 |
| Ontologue (served predictor) | 0.227 | 0.314 | 0.375 | 0.440 | 0.428 | 0.438 | 0.504 |
| GO-GPT ten-sample | 0.314 | 0.361 | 0.418 | 0.457 | 0.414 | 0.383 | 0.324 |

### Ontologue on FR24-P by identity bin

| bin | proteins | n MF/BP/CC | Fw MF | Fw BP | Fw CC | Fw mean [95% CI] | F mean | S-min mean | per-protein F1_w |
|---|---|---|---|---|---|---|---|---|---|
| no_hit | 2270 | 394/460/1835 | 0.290 | 0.333 | 0.571 | 0.398 [0.388, 0.412] | 0.506 | 12.96 | 0.253 |
| <30 | 415 | 156/83/260 | 0.520 | 0.305 | 0.554 | 0.460 [0.433, 0.489] | 0.580 | 12.33 | 0.353 |
| 30-50 | 2633 | 1337/734/1275 | 0.617 | 0.362 | 0.584 | 0.521 [0.511, 0.533] | 0.630 | 11.13 | 0.423 |
| 50-80 | 3036 | 1169/1304/1396 | 0.677 | 0.331 | 0.684 | 0.564 [0.555, 0.575] | 0.654 | 10.82 | 0.490 |
| 80-95 | 1527 | 523/741/776 | 0.664 | 0.297 | 0.642 | 0.534 [0.522, 0.550] | 0.623 | 12.90 | 0.460 |
| 95-99.9 | 897 | 239/417/613 | 0.663 | 0.348 | 0.553 | 0.521 [0.504, 0.546] | 0.611 | 15.65 | 0.440 |
| 100 | 453 | 106/236/352 | 0.677 | 0.404 | 0.504 | 0.528 [0.501, 0.560] | 0.617 | 13.88 | 0.439 |

## NK-all: Fw mean per identity bin to GO-GPT train+val (threshold chosen inside the bin)

| method | no_hit (n=2653) | <30 (n=419) | 30-50 (n=2669) | 50-80 (n=3658) | 80-95 (n=3140) | 95-99.9 (n=2724) | 100 (n=7437) |
|---|---|---|---|---|---|---|---|
| Forecasting model, round 2 (F1-650M ensemble) | 0.396 | 0.459 | 0.520 | 0.562 | 0.539 | 0.546 | 0.588 |
| SPROF-GO | 0.441 | 0.489 | 0.530 | 0.550 | 0.541 | 0.514 | 0.533 |
| Ontologue (served predictor) | 0.332 | 0.420 | 0.480 | 0.531 | 0.531 | 0.544 | 0.616 |
| Forecasting model F1 (3-seed ensemble) | 0.356 | 0.442 | 0.493 | 0.538 | 0.519 | 0.512 | 0.561 |
| GO-GPT ten-sample | 0.372 | 0.424 | 0.485 | 0.501 | 0.470 | 0.468 | 0.427 |

Mean per-protein F1_w at each method's own Fmax_w threshold of the whole set (NK-all):

| method | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 |
|---|---|---|---|---|---|---|---|
| Forecasting model, round 2 (F1-650M ensemble) | 0.231 | 0.354 | 0.412 | 0.484 | 0.462 | 0.466 | 0.506 |
| SPROF-GO | 0.368 | 0.416 | 0.473 | 0.503 | 0.492 | 0.463 | 0.469 |
| Ontologue (served predictor) | 0.200 | 0.309 | 0.372 | 0.439 | 0.451 | 0.470 | 0.550 |
| Forecasting model F1 (3-seed ensemble) | 0.204 | 0.355 | 0.387 | 0.453 | 0.439 | 0.423 | 0.475 |
| GO-GPT ten-sample | 0.303 | 0.361 | 0.417 | 0.451 | 0.412 | 0.399 | 0.371 |

## LK: Fw mean per identity bin to GO-GPT train+val (threshold chosen inside the bin)

| method | no_hit (n=71) | <30 (n=5) | 30-50 (n=23) | 50-80 (n=86) | 80-95 (n=217) | 95-99.9 (n=403) | 100 (n=636) |
|---|---|---|---|---|---|---|---|
| Ontologue (served predictor) | 0.223 | 0.718 | 0.491 | 0.452 | 0.487 | 0.490 | 0.532 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.304 | 0.594 | 0.440 | 0.487 | 0.509 | 0.504 | 0.508 |
| Forecasting model F1 (3-seed ensemble) | 0.271 | 0.609 | 0.499 | 0.449 | 0.482 | 0.462 | 0.491 |
| GO-GPT ten-sample | 0.242 | 0.772 | 0.505 | 0.332 | 0.459 | 0.432 | 0.373 |
| SPROF-GO | 0.292 | 0.745 | 0.612 | 0.380 | 0.424 | 0.360 | 0.411 |

Mean per-protein F1_w at each method's own Fmax_w threshold of the whole set (LK):

| method | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 |
|---|---|---|---|---|---|---|---|
| Ontologue (served predictor) | 0.150 | 0.334 | 0.334 | 0.335 | 0.411 | 0.414 | 0.456 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.138 | 0.446 | 0.257 | 0.357 | 0.433 | 0.409 | 0.415 |
| Forecasting model F1 (3-seed ensemble) | 0.138 | 0.407 | 0.225 | 0.315 | 0.408 | 0.379 | 0.393 |
| GO-GPT ten-sample | 0.169 | 0.751 | 0.340 | 0.246 | 0.392 | 0.370 | 0.309 |
| SPROF-GO | 0.233 | 0.652 | 0.521 | 0.283 | 0.356 | 0.306 | 0.364 |

## FR24-P-noloads: Fw mean per identity bin to GO-GPT train+val (threshold chosen inside the bin)

| method | no_hit (n=1392) | <30 (n=275) | 30-50 (n=2228) | 50-80 (n=2910) | 80-95 (n=1429) | 95-99.9 (n=682) | 100 (n=260) |
|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.461 | 0.472 | 0.530 | 0.565 | 0.556 | 0.527 | 0.541 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.412 | 0.434 | 0.522 | 0.571 | 0.547 | 0.530 | 0.569 |
| Forecasting model F1 (3-seed ensemble) | 0.357 | 0.411 | 0.492 | 0.550 | 0.533 | 0.513 | 0.549 |
| Ontologue (served predictor) | 0.342 | 0.404 | 0.470 | 0.532 | 0.525 | 0.524 | 0.608 |
| GO-GPT ten-sample | 0.387 | 0.395 | 0.474 | 0.512 | 0.482 | 0.440 | 0.420 |

Mean per-protein F1_w at each method's own Fmax_w threshold of the whole set (FR24-P-noloads):

| method | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 |
|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.396 | 0.407 | 0.478 | 0.518 | 0.501 | 0.462 | 0.463 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.277 | 0.345 | 0.438 | 0.500 | 0.464 | 0.449 | 0.476 |
| Forecasting model F1 (3-seed ensemble) | 0.240 | 0.327 | 0.409 | 0.476 | 0.447 | 0.425 | 0.453 |
| Ontologue (served predictor) | 0.252 | 0.308 | 0.371 | 0.448 | 0.435 | 0.439 | 0.523 |
| GO-GPT ten-sample | 0.330 | 0.331 | 0.413 | 0.463 | 0.424 | 0.369 | 0.363 |

## FR24-P-GOcurrent: Fw mean per identity bin to GO-GPT train+val (threshold chosen inside the bin)

| method | no_hit (n=2270) | <30 (n=415) | 30-50 (n=2633) | 50-80 (n=3036) | 80-95 (n=1527) | 95-99.9 (n=897) | 100 (n=453) |
|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.453 | 0.486 | 0.532 | 0.558 | 0.526 | 0.465 | 0.429 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.396 | 0.466 | 0.521 | 0.566 | 0.529 | 0.518 | 0.527 |
| Forecasting model F1 (3-seed ensemble) | 0.355 | 0.449 | 0.495 | 0.547 | 0.516 | 0.491 | 0.504 |
| Ontologue (served predictor) | 0.328 | 0.422 | 0.481 | 0.528 | 0.510 | 0.513 | 0.568 |
| GO-GPT ten-sample | 0.383 | 0.427 | 0.487 | 0.513 | 0.471 | 0.457 | 0.381 |

Mean per-protein F1_w at each method's own Fmax_w threshold of the whole set (FR24-P-GOcurrent):

| method | no_hit | <30 | 30-50 | 50-80 | 80-95 | 95-99.9 | 100 |
|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.379 | 0.414 | 0.471 | 0.507 | 0.470 | 0.405 | 0.360 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.247 | 0.358 | 0.421 | 0.490 | 0.456 | 0.436 | 0.436 |
| Forecasting model F1 (3-seed ensemble) | 0.222 | 0.351 | 0.399 | 0.466 | 0.436 | 0.404 | 0.407 |
| Ontologue (served predictor) | 0.213 | 0.305 | 0.374 | 0.441 | 0.427 | 0.436 | 0.486 |
| GO-GPT ten-sample | 0.313 | 0.360 | 0.420 | 0.460 | 0.413 | 0.382 | 0.320 |

## FR24-P: Fw mean by organism group

| method | human (n=243) | mouse (n=1569) | other (n=8517) | rat (n=217) | zebrafish (n=685) | Arabidopsis thaliana (n=498) | Bacillus subtilis (strain 168) (n=139) | Bos taurus (n=127) | Candida albicans (strain SC5314 / ATCC MYA-2876) (n=141) | Danio rerio (n=685) | Drosophila melanogaster (n=373) | Homo sapiens (n=243) | Mus musculus (n=1569) | Rattus norvegicus (n=217) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPROF-GO | 0.518 | 0.402 | 0.546 | 0.463 | 0.513 | 0.481 | 0.467 | 0.601 | 0.515 | 0.513 | 0.515 | 0.518 | 0.402 | 0.463 |
| Forecasting model, round 2 (F1-650M ensemble) | 0.470 | 0.469 | 0.542 | 0.450 | 0.508 | 0.446 | 0.470 | 0.546 | 0.479 | 0.508 | 0.550 | 0.470 | 0.469 | 0.450 |
| Forecasting model F1 (3-seed ensemble) | 0.432 | 0.438 | 0.514 | 0.442 | 0.454 | 0.413 | 0.453 | 0.563 | 0.429 | 0.454 | 0.521 | 0.432 | 0.438 | 0.442 |
| Ontologue (served predictor) | 0.480 | 0.474 | 0.493 | 0.464 | 0.431 | 0.419 | 0.442 | 0.547 | 0.443 | 0.431 | 0.540 | 0.480 | 0.474 | 0.464 |
| GO-GPT ten-sample | 0.416 | 0.367 | 0.500 | 0.435 | 0.443 | 0.398 | 0.361 | 0.506 | 0.490 | 0.443 | 0.519 | 0.416 | 0.367 | 0.435 |

## NK-all: Fw mean by organism group

| method | human (n=11712) | mouse (n=1569) | other (n=8517) | rat (n=217) | zebrafish (n=685) | Arabidopsis thaliana (n=498) | Bacillus subtilis (strain 168) (n=139) | Bos taurus (n=127) | Candida albicans (strain SC5314 / ATCC MYA-2876) (n=141) | Danio rerio (n=685) | Drosophila melanogaster (n=373) | Homo sapiens (n=11712) | Mus musculus (n=1569) | Rattus norvegicus (n=217) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Forecasting model, round 2 (F1-650M ensemble) | 0.483 | 0.469 | 0.542 | 0.450 | 0.508 | 0.446 | 0.470 | 0.546 | 0.479 | 0.508 | 0.550 | 0.483 | 0.469 | 0.450 |
| SPROF-GO | 0.521 | 0.402 | 0.546 | 0.463 | 0.513 | 0.481 | 0.467 | 0.601 | 0.515 | 0.513 | 0.515 | 0.521 | 0.402 | 0.463 |
| Ontologue (served predictor) | 0.490 | 0.474 | 0.493 | 0.464 | 0.431 | 0.419 | 0.442 | 0.547 | 0.443 | 0.431 | 0.540 | 0.490 | 0.474 | 0.464 |
| Forecasting model F1 (3-seed ensemble) | 0.439 | 0.438 | 0.514 | 0.442 | 0.454 | 0.413 | 0.453 | 0.563 | 0.429 | 0.454 | 0.521 | 0.439 | 0.438 | 0.442 |
| GO-GPT ten-sample | 0.407 | 0.367 | 0.500 | 0.435 | 0.443 | 0.398 | 0.361 | 0.506 | 0.490 | 0.443 | 0.519 | 0.407 | 0.367 | 0.435 |

## Against the GO-GPT test (runs/compare_paper/summary.tsv, S3_all, 8,630 proteins)

On the GO-GPT test Ontologue is the frozen HiGO-R2 file (the same pipeline; its kNN and HiGO parts were scored in runs/eval_suite, not in compare_paper).

| method | GO-GPT test S3 Fw mean | FR24-P Fw mean | change | GO-GPT test Fw MF/BP/CC | FR24-P Fw MF/BP/CC |
|---|---|---|---|---|---|
| Forecasting model, round 2 (F1-650M ensemble) | — | 0.514 | — | — | 0.611/0.334/0.596 |
| Ontologue (served predictor) | 0.718 | 0.479 | -0.238 | 0.738/0.658/0.758 | 0.571/0.303/0.564 |
| Forecasting model F1 (3-seed ensemble) | — | 0.487 | — | — | 0.592/0.291/0.579 |
| SPROF-GO | 0.541 | 0.515 | -0.027 | 0.615/0.400/0.609 | 0.602/0.313/0.629 |
| GO-GPT ten-sample | 0.698 | 0.468 | -0.231 | 0.720/0.630/0.745 | 0.524/0.273/0.607 |

