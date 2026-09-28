"""Step 6: homology gate, test evaluation, strata and error analysis for trained runs.

The gate is fit on the validation split only (kNN computed against the train split), then frozen and
applied to the test split:  p = g * p_knn + (1 - g) * p_model,  g = sigmoid(w_a . [id/100, has_hit, 1])
per aspect a, followed by the max-constraint (MCM) so the blend stays hierarchy-consistent.

Usage: python -m cafa6.stack --runs dev_35m_lora dev_35m_dora base_35m_frozen_mlp
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from cafa6.data import PROC, ROOT
from cafa6.evaluate import dense_to_long, evaluate, summarize

ASPECTS = ("F", "P", "C")


def mcm_numpy(p: np.ndarray, anc: np.ndarray) -> np.ndarray:
    P = torch.as_tensor(p, dtype=torch.float32).cuda()
    src = torch.as_tensor(anc[:, 0], device="cuda").long()
    dst = torch.as_tensor(anc[:, 1], device="cuda").long()
    out = P.clone().scatter_reduce(1, dst.expand(P.size(0), -1), P[:, src], reduce="amax", include_self=True)
    return out.cpu().numpy()


def features(ident: pd.DataFrame, ids) -> np.ndarray:
    m = ident.set_index("id").loc[ids]
    pid = m.max_pident.values / 100.0
    return np.stack([pid, (pid > 0).astype(np.float32), np.ones_like(pid)], 1).astype(np.float32)


def fit_gate(p_model, p_knn, Y, X, aspect, steps=300):
    """Per-aspect logistic gate trained with BCE on validation labels."""
    W = {}
    for a in ASPECTS:
        m = aspect == a
        pm = torch.as_tensor(p_model[:, m]).cuda().float()
        pk = torch.as_tensor(p_knn[:, m]).cuda().float()
        y = torch.as_tensor(Y[:, m]).cuda().float()
        x = torch.as_tensor(X).cuda()
        w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(steps):
            g = torch.sigmoid(x @ w)[:, None]
            p = (g * pk + (1 - g) * pm).clamp(1e-6, 1 - 1e-6)
            loss = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        W[a] = w.detach().cpu().numpy()
    return W


def apply_gate(p_model, p_knn, X, aspect, W):
    out = p_model.copy()
    for a in ASPECTS:
        m = aspect == a
        g = 1 / (1 + np.exp(-(X @ W[a])))
        out[:, m] = g[:, None] * p_knn[:, m] + (1 - g[:, None]) * p_model[:, m]
    return out


def per_protein_f1(p, Y, tau):
    pred = p >= tau
    tp = (pred & (Y > 0)).sum(1)
    prec = tp / np.maximum(pred.sum(1), 1)
    rec = tp / np.maximum((Y > 0).sum(1), 1)
    return np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="eval_main")
    args = ap.parse_args()
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    terms, aspect = vocab.term.values, vocab.aspect.values
    anc = np.load(PROC / "go_ancestors.npy")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    ident = pd.read_parquet(PROC / "identity.parquet")
    va, te = np.where(df.split == "val")[0], np.where(df.split == "test")[0]
    va_ids, te_ids = df.id.values[va], df.id.values[te]
    knn_va = np.load(PROC / "diamond_val.npy").astype(np.float32)
    knn_te = np.load(PROC / "diamond_test.npy").astype(np.float32)
    Xva, Xte = features(ident, va_ids), features(ident, te_ids)
    Yva, Yte = Y[va].toarray(), Y[te].toarray()

    preds, gates, test_scores = {}, {}, {}
    freq = np.asarray(Y[np.where(df.split == "train")[0]].mean(0)).ravel().astype(np.float32)
    has = knn_te.sum(1) > 0
    test_scores["diamond+naive"] = np.where(has[:, None], 0.8 * knn_te + 0.2 * freq, freq)
    for r in args.runs:
        pv = np.load(ROOT / "runs" / r / "pred_val.npy").astype(np.float32)
        pt = np.load(ROOT / "runs" / r / "pred_test.npy").astype(np.float32)
        W = fit_gate(pv, knn_va, Yva, Xva, aspect)
        gates[r] = {a: W[a].round(4).tolist() for a in ASPECTS}
        test_scores[r] = pt
        test_scores[f"{r}+knn_gate"] = mcm_numpy(apply_gate(pt, knn_te, Xte, aspect, W), anc)
    for k, s in test_scores.items():
        preds[k] = dense_to_long(s, te_ids, terms)

    res = evaluate(preds, PROC / "gt_test.tsv", out_dir=out)
    summ = summarize(res)
    print("== test (all)\n", summ.round(4).to_string())
    summ.to_csv(out / "summary_all.tsv", sep="\t")
    (out / "gates.json").write_text(json.dumps(gates, indent=2))

    # strata: max identity to train
    idt = ident[ident.split == "test"].set_index("id")
    strata = []
    for b in ["no_hit", "<30%", "30-50%", ">=50%"]:
        ids_b = idt.index[idt.id_bin == b]
        rb = evaluate(preds, PROC / "gt_test.tsv", ids_subset=ids_b)
        s = summarize(rb); s["stratum"] = f"identity {b}"; s["n"] = len(ids_b)
        strata.append(s)
    # strata: proteins whose annotations are dominated by rare (high-IA) terms vs common ones
    ia_prot = pd.Series((Yte * vocab.ia.values).sum(1) / np.maximum(Yte.sum(1), 1), index=te_ids)
    q = ia_prot.quantile([1 / 3, 2 / 3]).values
    for name, sel in [("IA low", ia_prot <= q[0]), ("IA mid", (ia_prot > q[0]) & (ia_prot <= q[1])), ("IA high", ia_prot > q[1])]:
        ids_b = ia_prot.index[sel]
        rb = evaluate(preds, PROC / "gt_test.tsv", ids_subset=ids_b)
        s = summarize(rb); s["stratum"] = name; s["n"] = len(ids_b)
        strata.append(s)
    strata = pd.concat(strata).reset_index()
    strata.to_csv(out / "strata.tsv", sep="\t", index=False)
    print(strata.round(4).to_string())

    # error analysis for the best gated model: worst proteins by per-protein F1 at tau=0.3
    best = summ.index[0]
    f1 = per_protein_f1(test_scores[best], Yte, 0.3)
    ea = pd.DataFrame({"id": te_ids, "f1": f1, "length": df.length.values[te], "taxon": df.taxon.values[te],
                       "n_labels": Yte.sum(1), "max_pident": idt.loc[te_ids].max_pident.values,
                       "mean_label_ia": ia_prot.values})
    ea.sort_values("f1").to_csv(out / "error_analysis.tsv", sep="\t", index=False)
    worst = ea[ea.f1 <= ea.f1.quantile(0.1)]
    cmp = pd.DataFrame({"worst_decile": worst.drop(columns=["id", "taxon"]).median(),
                        "all": ea.drop(columns=["id", "taxon"]).median()})
    cmp.to_csv(out / "error_summary.tsv", sep="\t")
    print("error analysis (medians)\n", cmp.round(3).to_string())
    tx = pd.DataFrame({"share_worst": worst.taxon.value_counts(normalize=True),
                       "share_all": ea.taxon.value_counts(normalize=True)}).fillna(0)
    tx["enrichment"] = tx.share_worst / tx.share_all.clip(lower=1e-9)
    tx = tx[tx.share_all >= 0.01].sort_values("enrichment", ascending=False)
    tx.to_csv(out / "error_taxa.tsv", sep="\t")
    print("taxa over-represented among the worst decile (>=1% of test)\n", tx.head(8).round(3).to_string())
    print("best:", best)
    (ROOT / "logs" / f"{args.out}.done").touch()


if __name__ == "__main__":
    main()
