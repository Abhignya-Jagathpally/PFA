"""Pilot: a one-step MeanFlow label refiner on top of HiGO + sharpened kNN (GO-GPT benchmark).

MeanFlow (Geng, Deng, Bai, Kolter, He; NeurIPS 2025 oral, arXiv 2505.13447) learns the average velocity
u(z, r, t) = 1/(t - r) * int_r^t v(z_s, s) ds of a flow z_t = (1 - t) x + t e. It trains on the MeanFlow identity
u = v - (t - r) du/dt with du/dt = v . d_z u + d_t u (one torch.func.jvp, tangent (v, 0, 1)) and a stop-gradient
target, and samples in one step: x = e - u(e, 0, 1).

Here x = 2y - 1 is the label vector over the 7,901-term vocabulary, and the condition c is HiGO's probabilities,
the sharpened kNN prior, max identity and has_hit. The source e is either N(0, I) ("noise") or the per-term fusion
prior plus noise, e = 2 p0 - 1 + sigma * eps ("prior"); the flow identity does not care what the source
distribution is, as long as (x, e) pairs are sampled jointly given c. Scores are the mean of S one-step samples
mapped back to [0, 1], then max-constrained (MCM).

Controls on the same split: HiGO alone, sharpened kNN alone, a per-term logistic fusion of the same features,
and a "discriminative twin" (the same network trained with BCE).

Selection data: GO-GPT's random val split, split 50/50 by sequence cluster (A = fit, B = select). TV cannot be
used: TV proteins are GO-GPT training proteins that HiGO was trained on, and there are no HiGO predictions for them.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m cafa6.meanflow_refiner [--test]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

from cafa6.data import ROOT
from cafa6.knn_sharp import Ontology, Scorer, gt_frame

PROC = ROOT / "data" / "proc_gogpt"
RUN = ROOT / "runs" / "meanflow_pilot"
HIGO = ROOT / "runs" / "gogptbench_35m_lora"
TEST_LOOKS = ROOT / "runs" / "test_looks.md"
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(4)
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(RUN / "log.txt", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- data

def sharp_knn(qids, hits, tr_pos, Ytr, T=2.0, k=10):
    """w_j = bits_j * exp((pident_j - max pident)/T) over the top-k hits by bits, normalised; p = sum_j w_j y_j."""
    h = hits[hits.q.isin(set(qids)) & hits.s.isin(tr_pos.keys())].sort_values(["q", "bits"], ascending=[True, False])
    maxid_all = h.groupby("q").pident.max()
    h = h[h.groupby("q").cumcount() < k].copy()
    h["w"] = h.bits * np.exp((h.pident - h.groupby("q").pident.transform("max")) / T)
    h["w"] /= h.groupby("q").w.transform("sum")
    qpos = {q: i for i, q in enumerate(qids)}
    W = sp.csr_matrix((h.w.values, (h.q.map(qpos).values, h.s.map(tr_pos).values)), shape=(len(qids), Ytr.shape[0]))
    P = np.minimum((W @ Ytr).toarray(), 1.0).astype(np.float32)
    maxid = maxid_all.reindex(qids).fillna(0).values.astype(np.float32)
    return P, maxid, (maxid > 0).astype(np.float32)


def load_split(split, df, Y, hits, tr_pos, Ytr):
    idx = np.where(df.split.values == split)[0]
    ids = np.asarray(df.id.values[idx], dtype=object)
    ph = np.load(HIGO / f"pred_{split}.npy", mmap_mode="r")
    pk, maxid, has = sharp_knn(ids, hits, tr_pos, Ytr)
    return {"ids": ids, "cluster": np.asarray(df.cluster.values[idx]),
            "ph": torch.from_numpy(np.ascontiguousarray(ph, dtype=np.float32)),
            "pk": torch.from_numpy(pk), "maxid": torch.from_numpy(maxid / 100), "has": torch.from_numpy(has),
            "y": torch.from_numpy(Y[idx].toarray().astype(np.float32))}


def sub(d, m):
    mt = torch.as_tensor(m)
    return {k: (v[mt.to(v.device)] if isinstance(v, torch.Tensor) else v[m] if isinstance(v, np.ndarray) else v)
            for k, v in d.items()}


def gpu(d):
    return {k: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def mcm(P, src, dst):
    return P.clone().scatter_reduce(1, dst.expand(P.size(0), -1), P[:, src], reduce="amax", include_self=True)


def logit(p, eps):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


# ---------------------------------------------------------------- per-term logistic fusion (control)

def fusion_feats(d):
    lh = logit(d["ph"], 1e-4) / 4
    lk = logit(d["pk"], 1e-3) / 4
    m, h = d["maxid"][:, None].expand_as(lh), d["has"][:, None].expand_as(lh)
    return torch.stack([lh, lk, d["pk"], m, h, lk * m, torch.ones_like(lh)], -1)  # [n, T, 7]


class Fusion(nn.Module):
    """logit_ij = x_ij . (w_aspect(j) + delta_j); delta has an L2 penalty (shrinks rare terms to the aspect fit)."""

    def __init__(self, aspect_id, nf=7):
        super().__init__()
        self.register_buffer("a", aspect_id)
        self.Wa = nn.Parameter(torch.zeros(3, nf))
        self.D = nn.Parameter(torch.zeros(len(aspect_id), nf))
        with torch.no_grad():
            self.Wa[:, 0], self.Wa[:, 1], self.Wa[:, -1] = 2.0, 2.0, -0.0

    def forward(self, X):
        return (X * (self.Wa[self.a] + self.D)).sum(-1)


def fit_fusion(d, aspect_id, lam, steps=300, bs=2048):
    m = Fusion(aspect_id).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    n = len(d["ids"])
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        b = torch.randint(0, n, (min(bs, n),), generator=g).to(d["ph"].device)
        X = fusion_feats({k: d[k][b].to(DEV) for k in ("ph", "pk", "maxid", "has")})
        loss = F.binary_cross_entropy_with_logits(m(X), d["y"][b].to(DEV)) + lam * m.D.pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def apply_fusion(m, d, bs=1024):
    out = []
    for i in range(0, len(d["ids"]), bs):
        X = fusion_feats({k: d[k][i:i + bs].to(DEV) for k in ("ph", "pk", "maxid", "has")})
        out.append(torch.sigmoid(m(X)).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- MeanFlow refiner

def temb(x, dim=32):
    f = torch.exp(torch.arange(dim // 2, device=x.device) * (-math.log(1000.0) / (dim // 2 - 1)))
    a = 1000 * x[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Refiner(nn.Module):
    """u_theta(z, r, t | c), parameterised as net(z, t, t - r | c).

    Global path: linear views of the whole state z and of the prior p0 -> protein vector g (label co-occurrence).
    Local path: a shared per-term MLP over [z_j, term features, g] with a learned term embedding.
    Output = local head + low-rank decoder of g. Both output layers start at zero, so u = 0 at init.
    """

    def __init__(self, T, d=256, h=64, nf=6):
        super().__init__()
        self.zin, self.pin = nn.Linear(T, d), nn.Linear(T, d)
        self.qin, self.tin = nn.Linear(2, d), nn.Linear(64, d)
        # no dropout: the JVP pass and the loss pass must see the same function
        self.glob = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.E = nn.Parameter(torch.randn(T, h) * 0.02)
        self.loc_in, self.loc_g = nn.Linear(nf + 1, h), nn.Linear(d, h)
        self.loc_mid = nn.Linear(h, h)
        self.loc_out = nn.Linear(h, 1)
        self.U = nn.Linear(d, T)
        for lin in (self.loc_out, self.U):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, z, t, dt, Fc, q, p0):
        g = self.glob(self.zin(z) + self.pin(p0) + self.qin(q) + self.tin(torch.cat([temb(t), temb(dt)], -1)))
        x = F.gelu(self.loc_in(torch.cat([z[..., None], Fc], -1)) + self.E + self.loc_g(g)[:, None])
        x = F.gelu(self.loc_mid(x)) + x
        return self.loc_out(x).squeeze(-1) + self.U(g)


def cond(d, p0, idx):
    ph, pk = d["ph"][idx].to(DEV), d["pk"][idx].to(DEV)
    m, h = d["maxid"][idx].to(DEV), d["has"][idx].to(DEV)
    p0 = p0[idx].to(DEV)
    Fc = torch.stack([logit(ph, 1e-4) / 4, logit(pk, 1e-3) / 4, pk, logit(p0, 1e-4) / 4,
                      m[:, None].expand_as(ph), h[:, None].expand_as(ph)], -1)
    return Fc, torch.stack([m, h], -1), p0


def sample_tr(n, mu=-0.4, sigma=1.0, frac_neq=0.25):
    a, b = torch.sigmoid(torch.randn(2, n, device=DEV) * sigma + mu)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    eq = torch.rand(n, device=DEV) >= frac_neq
    r = torch.where(eq, t, r)
    return r, t


def source(p0, mode, sigma):
    eps = torch.randn_like(p0)
    return eps if mode == "noise" else (2 * p0 - 1) + sigma * eps


def train_meanflow(dA, p0A, cfg, steps, bs=512, lr=1e-3, p_adapt=1.0, c_adapt=1e-3, seed=0, evalfn=None,
                   eval_at=()):
    torch.manual_seed(seed)
    T = dA["y"].shape[1]
    model = Refiner(T).to(DEV)
    ema = Refiner(T).to(DEV); ema.load_state_dict(model.state_dict()); ema.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200))
    n = len(dA["ids"])
    hist = {}
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (bs,), device=DEV)
        Fc, q, p0 = cond(dA, p0A, idx)
        x = 2 * dA["y"][idx].to(DEV) - 1
        if cfg["kind"] == "twin":
            out = model(2 * p0 - 1, torch.ones(bs, device=DEV), torch.ones(bs, device=DEV), Fc, q, p0)
            loss = F.binary_cross_entropy_with_logits(logit(p0, 1e-4) + out, (x + 1) / 2)
        else:
            e = source(p0, cfg["mode"], cfg["sigma"])
            r, t = sample_tr(bs)
            z = (1 - t[:, None]) * x + t[:, None] * e
            v = e - x
            fn = lambda z_, r_, t_: model(z_, t_, t_ - r_, Fc, q, p0)
            with torch.no_grad():
                _, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
            u = fn(z, r, t)
            u_tgt = v - (t - r)[:, None] * dudt
            l = (u - u_tgt).pow(2).sum(-1)
            w = 1.0 / (l.detach() + c_adapt).pow(p_adapt)
            loss = (w * l).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.995)
        if s % 500 == 0:
            log(f"  {cfg['name']} step {s} loss {loss.item():.4f}")
        if evalfn is not None and s in eval_at:
            hist[s] = evalfn(ema)
    return ema, hist


@torch.no_grad()
def generate(model, d, p0, cfg, S=1, bs=256, seed=0):
    torch.manual_seed(seed)
    n, out = len(d["ids"]), []
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=DEV)
        Fc, q, pp = cond(d, p0, idx)
        b = len(idx)
        if cfg["kind"] == "twin":
            o = model(2 * pp - 1, torch.ones(b, device=DEV), torch.ones(b, device=DEV), Fc, q, pp)
            out.append(torch.sigmoid(logit(pp, 1e-4) + o).cpu())
            continue
        acc = torch.zeros_like(pp)
        for _ in range(S):
            e = source(pp, cfg["mode"], cfg["sigma"])
            xh = e - model(e, torch.ones(b, device=DEV), torch.ones(b, device=DEV), Fc, q, pp)  # z0 = z1 - u(z1, 0, 1)
            acc += ((xh + 1) / 2).clamp(0, 1)
        out.append((acc / S).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- evaluation

class Eval:
    def __init__(self, onto, split, ids, vocab_terms, src, dst, maxid):
        gt = gt_frame(split)
        self.sc = Scorer(onto, gt[gt.id.isin(set(ids))])
        self.ids = ids
        self.cols = np.array([onto.idx.get(t, -1) for t in vocab_terms])
        self.keep = self.cols >= 0
        self.src, self.dst, self.nt = src, dst, len(onto.terms)
        mx = pd.Series(maxid * 100, index=ids)
        self.bins = {">=95%": ids[mx.values >= 95], "50-95%": ids[(mx.values >= 50) & (mx.values < 95)],
                     "<50%/no hit": ids[mx.values < 50]}

    def curves(self, P):
        P = mcm(P.to(DEV).float(), self.src, self.dst).cpu().numpy()
        P = np.where(P >= 0.01, P, 0)[:, self.keep]
        M = sp.csr_matrix(P)
        M = sp.csr_matrix((M.data, self.cols[self.keep][M.indices], M.indptr), shape=(P.shape[0], self.nt))
        # the vocab is ancestor-closed up to 3 zero-IA roots, so MCM over go_ancestors equals cafaeval's prop=max
        return self.sc.curves(M, self.ids, closed=True)

    def score(self, P):
        cur = self.curves(P)
        f = self.sc.fmax(cur)
        row = {a: f[a] for a in ("MF", "BP", "CC")} | {"mean": f["mean"]}
        for b, ids in self.bins.items():
            row[b] = self.sc.fmax(cur, subset=ids)["mean"]
        return row, cur


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="the single test look, for the frozen configuration")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    log(f"== meanflow pilot {'(TEST LOOK)' if args.test else '(selection on val half B)'}")

    df = pd.read_parquet(PROC / "train.parquet", columns=["id", "split", "cluster"])
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    aspect_id = torch.as_tensor(vocab.aspect.map({"F": 0, "P": 1, "C": 2}).values).to(DEV)
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    tr = np.where(df.split.values == "train")[0]
    tr_pos = {p: i for i, p in enumerate(df.id.values[tr])}
    Ytr = Y[tr]
    hits = pd.read_parquet(PROC / "diamond_hits_valtest.parquet", columns=["q", "s", "pident", "bits"])
    anc = np.load(PROC / "go_ancestors.npy")
    src = torch.as_tensor(anc[:, 0], device=DEV).long()
    dst = torch.as_tensor(anc[:, 1], device=DEV).long()
    onto = Ontology()

    va = load_split("val", df, Y, hits, tr_pos, Ytr)
    cl = np.unique(va["cluster"])
    rng = np.random.default_rng(0)
    clA = set(rng.permutation(cl)[: len(cl) // 2])
    inA = np.array([c in clA for c in va["cluster"]])
    dA, dB = sub(va, inA), sub(va, ~inA)
    log(f"val: {len(va['ids'])} proteins -> A (fit) {inA.sum()}, B (select) {(~inA).sum()}, split by cluster")
    if args.test:
        dE = load_split("test", df, Y, hits, tr_pos, Ytr)
        ev = Eval(onto, "test", dE["ids"], vocab.term.values, src, dst, dE["maxid"].numpy())
    else:
        dE = dB
        ev = Eval(onto, "val", dB["ids"], vocab.term.values, src, dst, dB["maxid"].numpy())
    split_name = "test" if args.test else "valB"
    dA, dE = gpu(dA), gpu(dE)
    log("data loaded")

    rows, curs = [], {}

    def add(method, config, P, extra=None):
        r, cur = ev.score(P)
        curs[method + "|" + config] = cur
        rows.append({"split": split_name, "method": method, "config": config, **r, **(extra or {})})
        log(f"{split_name} {method:22s} {config:34s} mean {r['mean']:.4f}  MF {r['MF']:.4f} BP {r['BP']:.4f} "
            f"CC {r['CC']:.4f}  >=95 {r['>=95%']:.4f} 50-95 {r['50-95%']:.4f} <50 {r['<50%/no hit']:.4f}")
        return r

    add("higo", "pred_*.npy", dE["ph"])
    add("knn_sharp", "T=2,k=10", dE["pk"])

    # fusion control: lambda chosen on B (frozen for the test look)
    frozen = json.loads((RUN / "frozen.json").read_text()) if args.test else None
    lams = [frozen["fusion_lambda"]] if args.test else [1e-3, 1e-2, 1e-1]
    best = None
    for lam in lams:
        fm = fit_fusion(dA, aspect_id, lam)
        r = add("fusion_logistic", f"lambda={lam}", apply_fusion(fm, dE))
        if best is None or r["mean"] > best[0]:
            best = (r["mean"], lam, fm)
    lam = best[1]
    fus_key = f"fusion_logistic|lambda={lam}"
    log(f"fusion lambda = {lam}")

    # out-of-fold fusion prior on A (2-fold by cluster) so the refiner trains on priors as noisy as at test time
    p0A = torch.zeros_like(dA["ph"])
    ucl = np.unique(dA["cluster"])
    fold = pd.Series(np.random.default_rng(1).integers(0, 2, len(ucl)), index=ucl).loc[dA["cluster"]].values
    for f in (0, 1):
        m = fit_fusion(sub(dA, fold != f), aspect_id, lam)
        p0A[torch.as_tensor(fold == f, device=DEV)] = apply_fusion(m, sub(dA, fold == f)).to(DEV)
    p0E = apply_fusion(best[2], dE).to(DEV)

    # sigma = 1.0 for the prior source was dropped after the 200-step smoke run (0.588 vs 0.615 at sigma 0.3)
    configs = [{"name": "twin_bce", "kind": "twin"},
               {"name": "mf_noise", "kind": "mf", "mode": "noise", "sigma": 1.0},
               {"name": "mf_prior_s0.1", "kind": "mf", "mode": "prior", "sigma": 0.1},
               {"name": "mf_prior_s0.3", "kind": "mf", "mode": "prior", "sigma": 0.3}]
    if args.test:
        configs = [c for c in configs if c["name"] in (frozen["meanflow"]["name"], "twin_bce")]
    steps_grid = [args.steps // 3, 2 * args.steps // 3, args.steps]
    for cfg in configs:
        t0 = time.time()
        if args.test:
            st = frozen["meanflow"]["steps"] if cfg["kind"] == "mf" else frozen["twin_steps"]
            model, _ = train_meanflow(dA, p0A, cfg, st)
            Ss = [frozen["meanflow"]["S"]] if cfg["kind"] == "mf" else [1]
            for S in Ss:
                add(cfg["name"], f"steps={st},S={S}", generate(model, dE, p0E, cfg, S=S))
            continue

        def evalfn(m, cfg=cfg):
            return {S: generate(m, dE, p0E, cfg, S=S) for S in ([1] if cfg["kind"] == "twin" else [1, 8, 32])}
        _, hist = train_meanflow(dA, p0A, cfg, args.steps, evalfn=evalfn, eval_at=steps_grid)
        log(f"{cfg['name']} trained+sampled in {time.time() - t0:.0f}s")
        for st, outs in hist.items():
            for S, P in outs.items():
                add(cfg["name"], f"steps={st},S={S}", P, {"sigma": cfg.get("sigma")})

    out = pd.DataFrame(rows)
    mf = out[out.method.str.startswith("mf_")]
    bmf = mf.loc[mf["mean"].idxmax()]
    fus = out[(out.method == "fusion_logistic") & (out.config == f"lambda={lam}")].iloc[0]
    bs = ev.sc.bootstrap(curs[bmf.method + "|" + bmf.config], curs[fus_key], reps=300)
    gain = float(bmf["mean"] - fus["mean"])
    res = {"split": split_name, "best_meanflow": f"{bmf.method}|{bmf.config}", "meanflow_mean": float(bmf["mean"]),
           "fusion_mean": float(fus["mean"]), "gain_vs_fusion": gain, "bootstrap_vs_fusion": bs,
           "keep": bool(gain >= 0.005), "runtime_s": round(time.time() - T0, 1)}
    tw = out[out.method == "twin_bce"]
    if len(tw):
        btw = tw.loc[tw["mean"].idxmax()]
        res["twin_best"] = f"{btw.config}: {btw['mean']:.4f}"
    log(json.dumps(res, indent=1))
    path = RUN / "summary.tsv"
    if path.exists():
        prev = pd.read_csv(path, sep="\t")
        out = pd.concat([prev[prev.split != split_name], out])
    out.round(5).to_csv(path, sep="\t", index=False)
    (RUN / f"result_{split_name}.json").write_text(json.dumps(res, indent=2))
    if not args.test:
        st, S = bmf.config.split(",")
        twin_steps = int(btw.config.split(",")[0].split("=")[1])
        (RUN / "frozen.json").write_text(json.dumps({
            "fusion_lambda": lam, "twin_steps": twin_steps,
            "meanflow": {"name": bmf.method, "steps": int(st.split("=")[1]), "S": int(S.split("=")[1])},
            "selected_on": "val half B (random val split, cluster-level 50/50), frozen before any test look"}, indent=2))
    else:
        with open(TEST_LOOKS, "a") as f:
            f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | MeanFlow label-refiner pilot | TEST LOOK #1 (only) for "
                    f"the frozen config in runs/meanflow_pilot/frozen.json: {res['best_meanflow']} test mean Fmax "
                    f"{res['meanflow_mean']:.4f} vs per-term fusion {res['fusion_mean']:.4f} (diff {gain:+.4f} "
                    f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}], 300 reps); verdict was fixed on val B beforehand\n")


if __name__ == "__main__":
    main()
