"""Step 4-5: train HiGO with a PEFT adapter. Usage: python -m cafa6.train --config configs/dev.yaml [--key value]"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from cafa6.data import PROC, ROOT
from cafa6.model import HiGO

ASPECTS = ("F", "P", "C")


# ----------------------------------------------------------------------------- data
class Batcher:
    """Length-bucketed batches with a token budget; long proteins are cropped (train) or windowed (eval)."""

    def __init__(self, seqs, tok, max_len, tokens_per_batch, max_batch):
        self.seqs, self.tok, self.max_len = seqs, tok, max_len
        self.tpb, self.max_batch = tokens_per_batch, max_batch

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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            z, _ = model(ids.to(device), att.to(device), res.to(device), tx)
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
    batcher = Batcher(seqs, tok, cfg["max_len"], cfg["tokens_per_batch"], cfg["max_batch"])
    model = HiGO(cfg["plm"], len(vocab), anc, ia, n_taxa=len(tmap) + 1, adapter=cfg["adapter"],
                 lora_r=cfg.get("lora_r", 8), d=cfg.get("d", 256), hier_query=cfg.get("hier_query", True),
                 pooling=cfg.get("pooling", "entmax"), use_mcm=cfg.get("use_mcm", True),
                 use_taxon=cfg.get("use_taxon", True),
                 lora_targets=cfg.get("lora_targets", ["query", "value"])).to(device)
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

    adapter_params = [p for n, p in model.named_parameters() if n.startswith("plm.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("plm.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": adapter_params, "lr": cfg["lr"]},
                             {"params": head_params, "lr": cfg.get("head_lr", 1e-3)}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(batcher.batches(tr_idx, True)) / cfg.get("grad_accum", 1))
    total = steps_per_epoch * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.get("warmup_frac", 0.06) * total), total)
    Yd = Y[va_idx].toarray()
    log = open(run / "train_log.jsonl", "a")
    best, bad = -1.0, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for bi, b in enumerate(batcher.batches(tr_idx, True)):
            ids, att, res = batcher.encode([batcher.crop(seqs[j]) for j in b])
            y = torch.as_tensor(Y[b].toarray(), device=device)
            tx = torch.as_tensor(taxa[b], device=device)
            if cfg.get("taxon_dropout", 0.1) > 0:
                tx = torch.where(torch.rand_like(tx, dtype=torch.float) < cfg.get("taxon_dropout", 0.1), 0, tx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, _ = model(ids.to(device), att.to(device), res.to(device), tx)
            loss = model.loss(z.float(), y, ia_weight=cfg.get("ia_weight", True)) / cfg.get("grad_accum", 1)
            loss.backward()
            if (bi + 1) % cfg.get("grad_accum", 1) == 0:
                torch.nn.utils.clip_grad_norm_(adapter_params + head_params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * cfg.get("grad_accum", 1); nb += 1
            if bi % 200 == 0:
                print(f"ep{ep} step{bi} loss {tot / nb:.5f} {time.time() - t0:.0f}s", flush=True)
        # validation: loss + fast weighted Fmax
        pv = predict(model, batcher, seqs, taxa, va_idx, device).astype(np.float32)
        pvc = np.clip(pv, 1e-6, 1 - 1e-6)
        vloss = float(-(Yd * np.log(pvc) + (1 - Yd) * np.log(1 - pvc)).mean())
        fm = fast_fmax_w(pv, Yd, ia, aspect)
        rec = {"epoch": ep, "train_loss": tot / nb, "val_loss": vloss, **{f"val_Fw_{k}": v for k, v in fm.items()},
               "elapsed_s": time.time() - t0}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if fm["mean"] > best:
            best, bad = fm["mean"], 0
            state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(state, run / "trainable_weights.pt")   # adapter + head only
        else:
            bad += 1
            if bad >= cfg.get("patience", 3):
                print("early stop"); break
    info["best_val_Fw_mean"] = best
    info["train_time_s"] = time.time() - t0
    # reload best and dump val/test predictions for evaluation & stacking
    state = torch.load(run / "trainable_weights.pt")
    model.load_state_dict(state, strict=False)
    if cfg.get("fold") is None:
        for split in ("val", "test"):
            idx = np.where(df.split == split)[0]
            np.save(run / f"pred_{split}.npy", predict(model, batcher, seqs, taxa, idx, device))
    (run / "info.json").write_text(json.dumps(info, indent=2))
    (ROOT / "logs" / f"{cfg['name']}.done").touch()


if __name__ == "__main__":
    main()
