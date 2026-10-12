# Who owns what (two sessions run the HiGO-R loop)

Set by the user on 2026-10-12 08:10. Read this before launching GPU jobs or editing shared code.

## Session A: stacking and calibration (CPU only)
- Owns `src/cafa6/stacker_tv.py`, `src/cafa6/calibrate_tv.py`, `runs/stacker_tv/`, `runs/calibrate_tv/`.
- Runs the stacker chain that starts automatically when a model's TV v2 predictions appear
  (`logs/stack_chain_{l3_c0,l3_c1,l4a}.log`, outputs `runs/stacker_tv/<run>/`).
- Does not launch GPU training jobs and does not edit `train.py`, `model.py` or `data.py`.

## Session B: GPU training changes
- Owns `train.py`, `model.py`, `data.py`, configs and all GPU launches.
- Optimiser: the user asked for both options head to head on the same setup (ideally the 3B feature cache):
  Schedule-Free AdamW (Defazio et al., NeurIPS 2024) and warmup-stable-decay with weight averaging at the end.
  Same seeds, same epochs budget, scored on TV v2 test-like mean through the stacker.
- Venue check (research_notes/HiGO-R loop/HiGO-R2_model.md, 09:03): SWA (UAI 2018) and ASAM (official repo 404) fail the
  venue/code rule. Justify weight averaging with WSD/SWAD (NeurIPS 2024/2021) and use plain SAM (ICLR 2021) for any
  sharpness arm. C-HMCNN code is GPL-3.0: keep our MCM implementation independent. flash_attn fails to import; use SDPA.
- New training runs should write `pred_{val,test,tv_all}_top500.npz` so Session A's stacker can pick them up.

## Runs in flight at 08:10
| run | GPU | owner |
|---|---|---|
| l3_c0_tv2, l3_c1_tv2 | 0 | A launched, B may manage |
| l5_3b_tv2 | 1 | A launched, B may manage |
| l4a_tv2 | 1 | A launched, B may manage |
| l5_3b_mix_tv2 | stopped (duplicate of l5_3b_tv2); checkpoints kept | - |

## Shared rules
- Test looks: 2 of 4 used (`runs/test_looks.md`). Only one look is taken per decision, after pre-registration.
- Kill processes by pattern only; stop wrapper shells first so they can't write false `.done` files.
- Nothing is deleted without the user's confirmation.
