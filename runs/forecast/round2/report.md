# Forecasting model, round 2: encoder size and routing

Registration: runs/test_looks.md, entry 2026-10-25 01:15:00. Code: `src/cafa6/forecast_train.py` (`train --plm`, `round2`, `predict --round2`). Log: `logs/forecast_train.log`.

Criterion = mean of the IA-weighted Fmax (mean over MF/BP/CC) on all rows of A valid and all rows of B valid (gained terms; input terms removed from predictions after propagation), seed ensembles (logit mean of seeds 0, 1, 2). Routing sends every aspect of a protein whose best input-time hit identity is >= theta to the input-time kNN prior. No-knowledge rows (empty aspect at input) are secondary.

- families trained: ['a', 'b']; dropped: none
- best criterion: b 0.2983; eligible within 0.003: b
- **chosen: b** (encoder facebook/esm2_t33_650M_UR50D, theta None), criterion 0.2983 (A all 0.2643, B all 0.3324; secondary nk 0.5048)
- kNN prior alone (reference): criterion 0.2466 (A all 0.1982, B all 0.2951)

## Selection table

| candidate | group | criterion | A all | B all | A nk | B nk | secondary nk | routed A | routed B | within 0.003 |
|---|---|---|---|---|---|---|---|---|---|---|
| a | 0 | 0.2947 | 0.2534 | 0.3360 | 0.4972 | 0.4919 | 0.4946 | 0.0000 | 0.0000 |  |
| a+route80 | 1 | 0.2723 | 0.2297 | 0.3150 | 0.4920 | 0.4763 | 0.4841 | 0.5103 | 0.4884 |  |
| a+route90 | 1 | 0.2738 | 0.2321 | 0.3154 | 0.4891 | 0.4713 | 0.4802 | 0.4007 | 0.3829 |  |
| a+route95 | 1 | 0.2755 | 0.2331 | 0.3179 | 0.4990 | 0.4761 | 0.4875 | 0.3199 | 0.3070 |  |
| a+route98 | 1 | 0.2799 | 0.2395 | 0.3203 | 0.4966 | 0.4786 | 0.4876 | 0.2503 | 0.2297 |  |
| b | 2 | 0.2983 | 0.2643 | 0.3324 | 0.5152 | 0.4944 | 0.5048 | 0.0000 | 0.0000 | yes |
| b+route80 | 3 | 0.2764 | 0.2389 | 0.3138 | 0.4958 | 0.4743 | 0.4851 | 0.5103 | 0.4884 |  |
| b+route90 | 3 | 0.2802 | 0.2421 | 0.3182 | 0.4973 | 0.4804 | 0.4889 | 0.4007 | 0.3829 |  |
| b+route95 | 3 | 0.2848 | 0.2463 | 0.3234 | 0.5042 | 0.4906 | 0.4974 | 0.3199 | 0.3070 |  |
| b+route98 | 3 | 0.2896 | 0.2528 | 0.3265 | 0.5096 | 0.4914 | 0.5005 | 0.2503 | 0.2297 |  |

## Per aspect (all rows and nk rows)

| candidate | period | all MF | all BP | all CC | all mean | nk MF | nk BP | nk CC | nk mean |
|---|---|---|---|---|---|---|---|---|---|
| kNN prior alone (reference, not a candidate) | A | 0.2289 | 0.1174 | 0.2483 | 0.1982 | 0.5045 | 0.3671 | 0.5288 | 0.4668 |
| kNN prior alone (reference, not a candidate) | B | 0.4229 | 0.2210 | 0.2414 | 0.2951 | 0.5663 | 0.3736 | 0.4044 | 0.4481 |
| a | A | 0.2866 | 0.1378 | 0.3357 | 0.2534 | 0.5556 | 0.4335 | 0.5026 | 0.4972 |
| a | B | 0.4376 | 0.2638 | 0.3065 | 0.3360 | 0.6196 | 0.4244 | 0.4318 | 0.4919 |
| a+route80 | A | 0.2772 | 0.1314 | 0.2804 | 0.2297 | 0.5525 | 0.4161 | 0.5075 | 0.4920 |
| a+route80 | B | 0.4399 | 0.2439 | 0.2612 | 0.3150 | 0.6085 | 0.4095 | 0.4108 | 0.4763 |
| a+route90 | A | 0.2701 | 0.1328 | 0.2934 | 0.2321 | 0.5366 | 0.4358 | 0.4950 | 0.4891 |
| a+route90 | B | 0.4376 | 0.2488 | 0.2598 | 0.3154 | 0.6105 | 0.4101 | 0.3932 | 0.4713 |
| a+route95 | A | 0.2741 | 0.1344 | 0.2907 | 0.2331 | 0.5527 | 0.4423 | 0.5019 | 0.4990 |
| a+route95 | B | 0.4364 | 0.2508 | 0.2667 | 0.3179 | 0.6205 | 0.4110 | 0.3968 | 0.4761 |
| a+route98 | A | 0.2748 | 0.1380 | 0.3058 | 0.2395 | 0.5500 | 0.4535 | 0.4863 | 0.4966 |
| a+route98 | B | 0.4393 | 0.2507 | 0.2710 | 0.3203 | 0.6230 | 0.4145 | 0.3984 | 0.4786 |
| b | A | 0.2945 | 0.1529 | 0.3456 | 0.2643 | 0.5768 | 0.4213 | 0.5476 | 0.5152 |
| b | B | 0.4396 | 0.2460 | 0.3116 | 0.3324 | 0.6201 | 0.4052 | 0.4577 | 0.4944 |
| b+route80 | A | 0.2856 | 0.1444 | 0.2867 | 0.2389 | 0.5814 | 0.4100 | 0.4961 | 0.4958 |
| b+route80 | B | 0.4388 | 0.2361 | 0.2666 | 0.3138 | 0.6025 | 0.3886 | 0.4318 | 0.4743 |
| b+route90 | A | 0.2779 | 0.1469 | 0.3016 | 0.2421 | 0.5725 | 0.4184 | 0.5012 | 0.4973 |
| b+route90 | B | 0.4401 | 0.2392 | 0.2752 | 0.3182 | 0.6084 | 0.3928 | 0.4401 | 0.4804 |
| b+route95 | A | 0.2877 | 0.1485 | 0.3026 | 0.2463 | 0.6045 | 0.4159 | 0.4923 | 0.5042 |
| b+route95 | B | 0.4443 | 0.2420 | 0.2841 | 0.3234 | 0.6204 | 0.3981 | 0.4534 | 0.4906 |
| b+route98 | A | 0.2906 | 0.1528 | 0.3150 | 0.2528 | 0.6019 | 0.4272 | 0.4998 | 0.5096 |
| b+route98 | B | 0.4462 | 0.2427 | 0.2906 | 0.3265 | 0.6221 | 0.3985 | 0.4537 | 0.4914 |

## Single seeds (no routing; descriptive)

| run | period | all mean | nk mean |
|---|---|---|---|
| F1_s0 | A | 0.2471 | 0.4931 |
| F1_s1 | A | 0.2491 | 0.4883 |
| F1_s2 | A | 0.2470 | 0.4908 |
| F1_s0 | B | 0.3337 | 0.4902 |
| F1_s1 | B | 0.3342 | 0.4903 |
| F1_s2 | B | 0.3339 | 0.4904 |
| F1_650M_s0 | A | 0.2570 | 0.5048 |
| F1_650M_s1 | A | 0.2558 | 0.4747 |
| F1_650M_s2 | A | 0.2548 | 0.5127 |
| F1_650M_s0 | B | 0.3245 | 0.4826 |
| F1_650M_s1 | B | 0.3223 | 0.4794 |
| F1_650M_s2 | B | 0.3268 | 0.4845 |
