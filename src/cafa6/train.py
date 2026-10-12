"""Step 4-5: train HiGO with a PEFT adapter. Usage: python -m cafa6.train --config configs/dev.yaml [key=value ...]

Opt-in keys added for HiGO-R (defaults keep the original behaviour):
- resume: true            save runs/<name>/checkpoint_last.pt (weights, optimizer, scheduler, RNG) at every epoch end
                          and every `ckpt_every` optimizer steps; a rerun with the same name continues from it.
- term_bias_init: prior   start term_bias at logit(train label frequency) instead of 0.
- wd_exclude: true        no weight decay on embeddings, LayerNorm, biases, gates and scalar-mix weights.
- sched_epochs: N         cosine horizon in epochs (default: epochs).
- pred_topk: K            write pred_{split}_top{K}.npz (sparse, float16) instead of dense pred_{split}.npy.
- pred_splits: [...]      splits to predict after training (default [val, test]; absent splits are skipped).
- pred_id_sets: {...}     extra prediction sets from id lists in PROC, e.g. {tv: tv_ids.txt} -> pred_tv*.
- knn_fusion: true        retrieval-conditioned logits; see data.load_knn for the files it reads, plus knn_caps,
                          knn_T, knn_topk, knn_gamma, knn_phi, knn_eps, knn_lambda_init, knn_delta_wd.
- readout: mix, readout_layers, plm_layers, plm_bf16, crla, crla_dim: see model.py.

Throughput and logging keys (outputs unchanged unless noted):
- prefetch: true          build the next training batches (crop, tokenise, label densify, pinned memory) in a background
                          thread, in the same order and with the same RNG draws as before; false = old inline path.
- log_every: 200          optimizer-step interval of the progress line (loss, running mean, lr, tokens/s, s/step, peak
                          GPU memory, epoch ETA); an epoch summary follows each epoch's JSON line.
- fast_entmax: false      torch.compile-fused entmax-1.5 bisection (model.py): ~3-10x faster pooling; equal to float
                          tolerance in fp32, but under bf16 autocast it differs from entmax_bisect at bf16 rounding level.
- fused_adamw: false      fused CUDA AdamW kernel; equal to float tolerance, not bit-identical.

Multi-GPU: run independent jobs, one per GPU (CUDA_VISIBLE_DEVICES=k), e.g. two seeds or two arms in parallel.
There is no DDP path; per-run throughput on one H100 is enough for 35M, and frozen 3B fits one GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from cafa6.data import KNN_CAPS, PROC, ROOT, KnnBank, load_knn, save_pred
from cafa6.model import build_higo, load_trainable

ASPECTS = ("F", "P", "C")


# ----------------------------------------------------------------------------- data
class Batcher:
    """Length-bucketed batches with a token budget; long proteins are cropped (train) or windowed (eval).
    With a kNN bank attached, retrieval(rows, cap) gives the per-protein retrieval inputs for a batch."""

    def __init__(self, seqs, tok, max_len, tokens_per_batch, max_batch, knn: KnnBank | None = None,
                 caps=None, seed: int = 0):
        self.seqs, self.tok, self.max_len = seqs, tok, max_len
        self.tpb, self.max_batch = tokens_per_batch, max_batch
        self.knn, self.caps = knn, caps
        self.cap_rng = np.random.default_rng(seed)

    def sample_cap(self):
        """Identity cap for one training batch: 0 = no retrieval, c = hits with identity <= c%, None = all hits."""
        if not self.caps:
            return None
        caps, probs = zip(*self.caps)
        return caps[self.cap_rng.choice(len(caps), p=np.asarray(probs) / sum(probs))]

    def retrieval(self, rows, cap=None, device="cuda"):
        if self.knn is None:
            return None
        out = self.knn.batch(rows, cap)
        return {k: torch.as_tensor(v, device=device) for k, v in out.items() if v is not None}

    def batches(self, idx, shuffle):
        idx = list(idx)
        if shuffle:
            random.shuffle(idx)
            chunks = [sorted(idx[i:i + 4096], key=lambda j: len(self.seqs[j])) for i in range(0, len(idx), 4096)]
        else:
            chunks = [sorted(idx, key=lambda j: len(self.seqs[j]))]
        out = []
        for ch in chunks:
            cur, cur_max = [], 0
            for j in ch:
                L = min(len(self.seqs[j]), self.max_len) + 2
                if cur and (max(cur_max, L) * (len(cur) + 1) > self.tpb or len(cur) >= self.max_batch):
                    out.append(cur)
                    cur, cur_max = [], 0
                cur.append(j)
                cur_max = max(cur_max, L)
            if cur:
                out.append(cur)
        if shuffle:
            random.shuffle(out)
        return out

    def encode(self, pieces):
        enc = self.tok(pieces, return_tensors="pt", padding=True, add_special_tokens=True)
        res = enc["attention_mask"].clone()
        res[:, 0] = 0
        lens = enc["attention_mask"].sum(1)
        res[torch.arange(len(pieces)), lens - 1] = 0
        return enc["input_ids"], enc["attention_mask"], res

    def crop(self, s):
        if len(s) <= self.max_len:
            return s
        st = random.randint(0, len(s) - self.max_len)
        return s[st:st + self.max_len]

    def windows(self, s):
        if len(s) <= self.max_len:
            return [s]
        stride = self.max_len // 2
        starts = list(range(0, len(s) - self.max_len + 1, stride))
        if starts[-1] != len(s) - self.max_len:
            starts.append(len(s) - self.max_len)
        return [s[a:a + self.max_len] for a in starts]

    def train_tensors(self, b, Y, taxa, pin: bool):
        """CPU tensors for one training batch: (b, ids, att, res, y, tx, n_tokens); draws the crop RNG."""
        ids, att, res = self.encode([self.crop(self.seqs[j]) for j in b])
        out = [ids, att, res, torch.from_numpy(Y[b].toarray()), torch.as_tensor(taxa[b])]
        if pin:
            out = [t.pin_memory() for t in out]
        return (b, *out, int(att.sum()))


def iter_train_batches(batches, skip, make, prefetch: bool, depth: int = 4):
    """Yields (bi, make(b)) for bi >= skip. With prefetch a single background thread runs `make` ahead of the
    consumer in batch order, so every RNG it uses is drawn in exactly the order of the inline path."""
    todo = [(bi, b) for bi, b in enumerate(batches) if bi >= skip]
    if not prefetch:
        for bi, b in todo:
            yield bi, make(b)
        return
    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        try:
            for bi, b in todo:
                q.put((bi, make(b)))
        except BaseException as e:   # surfaced in the consumer
            q.put(e)
            return
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def fmt_dur(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def fmt_diag(d: dict) -> str:
    parts = []
    for k, v in d.items():
        parts.append(f"{k} {v}" if isinstance(v, list) else f"{k} {v:.4f}")
    return " | ".join(parts)


# ----------------------------------------------------------------------------- metric
def fast_fmax_w(p: np.ndarray, Y: np.ndarray, ia: np.ndarray, aspect: np.ndarray) -> dict:
    """IA-weighted Fmax (CAFA normalisation) on the vocab label space; used for early stopping only."""
    out = {}
    P, Yt, w = torch.as_tensor(p).cuda(), torch.as_tensor(Y).cuda(), torch.as_tensor(ia).cuda()
    for a in ASPECTS:
        m = torch.as_tensor(aspect == a).cuda()
        pa, ya, wa = P[:, m], Yt[:, m], w[m]
        has_gt = (ya.sum(1) > 0)
        pa, ya = pa[has_gt], ya[has_gt]
        gt_w = (ya * wa).sum(1)
        best = 0.0
        for tau in np.arange(0.01, 1.0, 0.01):
            pr_mask = (pa >= tau).float()
            tp = (pr_mask * ya * wa).sum(1)
            pred_w = (pr_mask * wa).sum(1)
            covered = pred_w > 0
            if covered.sum() == 0:
                continue
            prec = (tp[covered] / pred_w[covered]).mean()
            rec = (tp / gt_w.clamp(min=1e-9)).mean()
            f = (2 * prec * rec / (prec + rec)).item() if (prec + rec) > 0 else 0.0
            best = max(best, f)
        out[a] = best
    out["mean"] = float(np.mean([out[a] for a in ASPECTS]))
    return out


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, batcher, seqs, taxa, idx, device, bf16=True):
    model.eval()
    scores = np.zeros((len(idx), model.n_terms), np.float16)
    pos = {j: i for i, j in enumerate(idx)}
    pieces, owners = [], []
    for j in idx:
        for w in batcher.windows(seqs[j]):
            pieces.append(w)
            owners.append(j)
    order = sorted(range(len(pieces)), key=lambda k: len(pieces[k]))
    best = {}
    b, cur_max = [], 0
    groups = []
    for k in order:
        L = len(pieces[k]) + 2
        if b and (max(cur_max, L) * (len(b) + 1) > batcher.tpb or len(b) >= batcher.max_batch):
            groups.append(b)
            b, cur_max = [], 0
        b.append(k)
        cur_max = max(cur_max, L)
    if b:
        groups.append(b)
    for g in groups:
        ids, att, res = batcher.encode([pieces[k] for k in g])
        tx = torch.as_tensor([taxa[owners[k]] for k in g], device=device)
        kn = batcher.retrieval([owners[k] for k in g], None, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            z, _ = model(ids.to(device), att.to(device), res.to(device), tx, knn=kn)
        z = z.float()
        for r, k in enumerate(g):
            o = owners[k]
            best[o] = z[r] if o not in best else torch.maximum(best[o], z[r])   # max-pool windows
    for o, z in best.items():
        scores[pos[o]] = model.predict_proba(z[None])[0].cpu().numpy().astype(np.float16)
    return scores


# ----------------------------------------------------------------------------- main
def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def taxon_index(train_df, min_count=50):
    vc = train_df.taxon.value_counts()
    keep = [t for t, c in vc.items() if c >= min_count]
    return {t: i + 1 for i, t in enumerate(keep)}   # 0 = other


def main():
    cfg = load_cfg()
    torch.manual_seed(cfg.get("seed", 0)); random.seed(cfg.get("seed", 0)); np.random.seed(cfg.get("seed", 0))
    run = ROOT / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    device = torch.device("cuda")

    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    anc = np.load(PROC / "go_ancestors.npy")
    go_edges = np.load(PROC / "go_edges.npy") if cfg.get("use_go_gat") else None   # proc_gogpt has no edges file
    ia = vocab.ia.values.astype(np.float32)
    aspect = vocab.aspect.values
    tmap = taxon_index(df[df.split == "train"])
    json.dump({str(k): v for k, v in tmap.items()}, open(run / "taxa.json", "w"))
    taxa = df.taxon.map(lambda t: tmap.get(t, 0)).values
    seqs = df.seq.tolist()

    fold = cfg.get("fold")   # optional k-fold over train+val clusters
    if fold is None:
        tr_idx = np.where(df.split == "train")[0]
        va_idx = np.where(df.split == "val")[0]
    else:
        pool = df[df.split != "test"]
        cl = np.array(sorted(pool.cluster.unique()), dtype=object)
        rng = np.random.default_rng(0); rng.shuffle(cl)
        fold_of = {c: i % cfg["n_folds"] for i, c in enumerate(cl)}
        f = df.cluster.map(fold_of)
        tr_idx = np.where((df.split != "test") & (f != fold))[0]
        va_idx = np.where((df.split != "test") & (f == fold))[0]
    if cfg.get("train_subset"):
        tr_idx = np.random.default_rng(0).choice(tr_idx, cfg["train_subset"], replace=False)
    if cfg.get("val_subset"):
        va_idx = np.random.default_rng(0).choice(va_idx, min(cfg["val_subset"], len(va_idx)), replace=False)

    tok = AutoTokenizer.from_pretrained(cfg["plm"])
    knn = knn_bank(cfg, df, Y, tr_idx, aspect)
    batcher = Batcher(seqs, tok, cfg["max_len"], cfg["tokens_per_batch"], cfg["max_batch"], knn=knn,
                      caps=cfg.get("knn_caps", KNN_CAPS) if knn is not None else None, seed=cfg.get("seed", 0))
    model = build_higo(cfg, vocab, anc, len(tmap) + 1, go_edges).to(device)
    if cfg.get("term_bias_init", "zero") == "prior":
        prior = np.asarray(Y[tr_idx].mean(0)).ravel().clip(1e-4, 1 - 1e-4)
        model.term_bias.data.copy_(torch.as_tensor(np.log(prior / (1 - prior)), dtype=torch.float32))
    if cfg.get("grad_ckpt"):
        model.plm.gradient_checkpointing_enable()
        if hasattr(model.plm, "enable_input_require_grads"):
            model.plm.enable_input_require_grads()
    n_tr = sum(p.numel() for p in model.plm.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.plm.parameters())
    n_head = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("plm.") and p.requires_grad)
    info = {"backbone_params": n_all, "adapter_trainable": n_tr, "adapter_frac": n_tr / n_all, "head_params": n_head,
            "n_train": len(tr_idx), "n_val": len(va_idx), "n_terms": len(vocab), "n_taxa": len(tmap) + 1}
    print(json.dumps(info))

    groups = param_groups(model, cfg)
    all_params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.01, **({"fused": True} if cfg.get("fused_adamw") else {}))
    accum = cfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(batcher.batches(tr_idx, True)) / accum)
    total = steps_per_epoch * cfg.get("sched_epochs", cfg["epochs"])
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.get("warmup_frac", 0.06) * total), total)
    Yd = Y[va_idx].toarray()
    log = open(run / "train_log.jsonl", "a")
    best, bad, start_ep, skip, elapsed0 = -1.0, 0, 0, 0, 0.0
    ckpt_path = run / "checkpoint_last.pt"
    if cfg.get("resume") and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_trainable(model, ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, bad, start_ep, skip, elapsed0 = ck["best"], ck["bad"], ck["epoch"], ck["step"], ck["elapsed"]
        random.setstate(ck["ep_rng"] if skip else ck["py_rng"])
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        print(f"resumed from {ckpt_path}: epoch {start_ep}, batch {skip}, best {best:.4f}, bad {bad}", flush=True)
    t0 = time.time() - elapsed0
    stop = bad >= cfg.get("patience", 3)
    prefetch, log_every, patience = cfg.get("prefetch", True), cfg.get("log_every", 200), cfg.get("patience", 3)
    make = lambda b: batcher.train_tensors(b, Y, taxa, pin=prefetch)
    for ep in range(start_ep, cfg["epochs"]):
        if stop:
            break
        model.train()
        nb = 0
        tot_t = torch.zeros((), dtype=torch.float64, device=device)   # running loss sum without a per-step sync
        ep_rng = random.getstate()
        ep_batches = batcher.batches(tr_idx, True)
        torch.cuda.reset_peak_memory_stats()
        ep_t0 = win_t0 = time.time()
        win_tok, win_nb, win_tot = 0, 0, 0.0
        for bi, (b, ids, att, res, y, tx, n_tok) in iter_train_batches(ep_batches, skip, make, prefetch):
            ids, att, res, y, tx = (t.to(device, non_blocking=prefetch) for t in (ids, att, res, y, tx))
            if cfg.get("taxon_dropout", 0.1) > 0:
                tx = torch.where(torch.rand_like(tx, dtype=torch.float) < cfg.get("taxon_dropout", 0.1), 0, tx)
            kn = batcher.retrieval(b, batcher.sample_cap(), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, _ = model(ids, att, res, tx, knn=kn)
            loss = model.loss(z.float(), y, ia_weight=cfg.get("ia_weight", True)) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if cfg.get("resume") and cfg.get("ckpt_every") and ((bi + 1) // accum) % cfg["ckpt_every"] == 0:
                    save_checkpoint(ckpt_path, model, opt, sched, ep, bi + 1, best, bad, ep_rng, time.time() - t0)
            tot_t += loss.detach().double() * accum; nb += 1; win_tok += n_tok
            if bi % log_every == 0:
                tot = tot_t.item()
                now = time.time()
                win_dt = max(now - win_t0, 1e-9)
                recent = (tot - win_tot) / max(nb - win_nb, 1)
                lrs = "/".join(f"{l:.2e}" for l in sorted({g["lr"] for g in opt.param_groups}))
                eta = (now - ep_t0) / nb * (len(ep_batches) - bi - 1)
                print(f"ep{ep} step{bi}/{len(ep_batches)} loss {recent:.5f} avg {tot / max(nb, 1):.5f} lr {lrs} "
                      f"{win_tok / win_dt:.0f} tok/s {win_dt / max(nb - win_nb, 1):.3f} s/step "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB eta {fmt_dur(eta)} "
                      f"{now - t0:.0f}s", flush=True)
                win_t0, win_tok, win_nb, win_tot = now, 0, nb, tot
        tot = tot_t.item()
        skip = 0
        train_s = time.time() - ep_t0
        # validation: loss + fast weighted Fmax
        pv = predict(model, batcher, seqs, taxa, va_idx, device).astype(np.float32)
        pvc = np.clip(pv, 1e-6, 1 - 1e-6)
        vloss = float(-(Yd * np.log(pvc) + (1 - Yd) * np.log(1 - pvc)).mean())
        fm = fast_fmax_w(pv, Yd, ia, aspect)
        val_s = time.time() - ep_t0 - train_s
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vloss,
               **{f"val_Fw_{k}": v for k, v in fm.items()}, "elapsed_s": time.time() - t0, **diagnostics(model)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        improved = fm["mean"] > best
        if improved:
            best, bad = fm["mean"], 0
            state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(state, run / "trainable_weights.pt")   # adapter + head only
        else:
            bad += 1
            stop = bad >= patience
        diag = fmt_diag(diagnostics(model))
        print(f"== epoch {ep + 1}/{cfg['epochs']} | train {rec['train_loss']:.5f} val {vloss:.5f} | val Fw "
              f"F {fm['F']:.4f} P {fm['P']:.4f} C {fm['C']:.4f} mean {fm['mean']:.4f} | best {best:.4f}"
              f"{' (new)' if improved else ''} patience {bad}/{patience} | train {fmt_dur(train_s)} "
              f"val {fmt_dur(val_s)}{' | ' + diag if diag else ''}", flush=True)
        if stop:
            print("early stop")
        if cfg.get("resume"):
            save_checkpoint(ckpt_path, model, opt, sched, ep + 1, 0, best, bad, None, time.time() - t0)
    info["best_val_Fw_mean"] = best
    info["train_time_s"] = time.time() - t0
    # reload best and dump predictions for evaluation & stacking
    load_trainable(model, torch.load(run / "trainable_weights.pt"))
    if cfg.get("fold") is None:
        for split, idx in prediction_sets(cfg, df).items():
            save_pred(run, split, predict(model, batcher, seqs, taxa, idx, device), cfg.get("pred_topk", 0))
    (run / "info.json").write_text(json.dumps(info, indent=2))
    (ROOT / "logs" / f"{cfg['name']}.done").touch()


def prediction_sets(cfg, df) -> dict:
    """Rows to predict after training: every split in pred_splits, plus pred_id_sets {name: ids file in PROC}, whose
    rows follow the file's order (first row per id), e.g. {tv: tv_ids.txt} for TV proteins kept in the train split."""
    out = {}
    for split in cfg.get("pred_splits", ["val", "test"]):
        idx = np.where(df.split == split)[0]
        if len(idx):
            out[split] = idx
        else:
            print(f"no proteins in split {split!r}; skipped")
    first = pd.Series(np.arange(len(df)), index=df.id.values)
    first = first[~first.index.duplicated()]
    for name, fname in (cfg.get("pred_id_sets") or {}).items():
        ids = [l.strip() for l in open(PROC / fname) if l.strip()]
        missing = [i for i in ids if i not in first.index]
        if missing:
            raise KeyError(f"{len(missing)} ids of {fname} not in train.parquet, e.g. {missing[0]}")
        out[name] = first.loc[ids].values
    return out


def knn_bank(cfg, df, Y, tr_idx, aspect):
    """Train source (capped retrieval from train_hits.parquet, else the contract file) plus val/test sources."""
    if not cfg.get("knn_fusion"):
        return None
    assert cfg.get("fold") is None, "knn_fusion is not wired for k-fold runs"
    caps = cfg.get("knn_caps", KNN_CAPS)
    sources = [load_knn(cfg, df, Y, "train", tr_idx, aspect)]
    if sources[0].kind == "contract" and any(c not in (0, None) for c, _ in caps):
        raise FileNotFoundError("identity caps need train_hits.parquet in the knn dir (train-vs-train DIAMOND hits, "
                                "self excluded); set knn_caps to [[0, p], [null, 1-p]] to train without it")
    for split in ("val", "test", "tv"):
        rows = np.where(df.split == split)[0]
        if len(rows):
            try:
                sources.append(load_knn(cfg, df, Y, split, rows, aspect))
            except FileNotFoundError as e:
                if split == "val":
                    raise
                print(f"warning: {e}; {split} proteins will fail at prediction time", flush=True)
    for s in sources:
        print(f"knn source {s.kind}: {len(s.row)} proteins", flush=True)
    return KnnBank(sources)


NO_DECAY_NAMES = ("term_emb", "term_bias", "taxon_emb.weight", "slot_query")


def param_groups(model, cfg):
    """Adapter (lr) and head (head_lr) groups, weight decay 0.01. With wd_exclude, embeddings, LayerNorms, biases and
    scalar gates get no decay; the per-term kNN offsets always get knn_delta_wd."""
    lr, head_lr = cfg["lr"], cfg.get("head_lr", 1e-3)
    groups = {"adapter": {"params": [], "lr": lr}, "adapter_nd": {"params": [], "lr": lr, "weight_decay": 0.0},
              "head": {"params": [], "lr": head_lr}, "head_nd": {"params": [], "lr": head_lr, "weight_decay": 0.0},
              "knn_delta": {"params": [], "lr": head_lr, "weight_decay": cfg.get("knn_delta_wd", 1.0)}}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        side = "adapter" if n.startswith("plm.") else "head"
        if n in ("knn_delta", "knn_g_delta"):
            key = "knn_delta"
        elif cfg.get("wd_exclude") and (p.ndim < 2 or n in NO_DECAY_NAMES or n.startswith(("knn_", "mix."))):
            key = side + "_nd"
        else:
            key = side
        groups[key]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


def save_checkpoint(path, model, opt, sched, epoch, step, best, bad, ep_rng, elapsed):
    """Atomic write, so a kill during saving never leaves a truncated checkpoint behind."""
    state = {"model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
             "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "step": step, "best": best,
             "bad": bad, "ep_rng": ep_rng, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(), "elapsed": elapsed}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def diagnostics(model) -> dict:
    out = {}
    if getattr(model, "knn_fusion", False):
        lam = torch.nn.functional.softplus(model.knn_a[model.term_aspect] + model.knn_delta)
        for a, i in (("F", 0), ("P", 1), ("C", 2)):
            out[f"knn_lambda_{a}"] = float(lam[model.term_aspect == i].mean())
    if getattr(model, "mix", None) is not None:
        out["mix_weights"] = [round(float(w), 4) for w in torch.softmax(model.mix.w.float(), 0)]
    if getattr(model, "crla", None) is not None:
        out["crla_gate"] = float(torch.tanh(model.crla.alpha))
    return out


if __name__ == "__main__":
    main()
