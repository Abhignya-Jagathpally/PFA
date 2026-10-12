"""Unit tests for the HiGO-R blocks. Run: PYTHONPATH=src ../bin/python tests/test_higo_r.py (pytest also works).

Uses the cached ESM2-8M and synthetic labels, so it needs no dataset; GPU optional."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from cafa6.data import KnnBank, KnnPrior, load_pred, save_pred
from cafa6.model import CRLA, HiGO, load_trainable

PLM = "facebook/esm2_t6_8M_UR50D"
T = 12
ANC = np.array([[1, 0], [2, 0], [3, 1], [3, 0], [5, 4], [6, 4], [7, 6], [7, 4], [9, 8], [10, 8], [11, 10], [11, 8]])
ASPECT = np.array(list("FFFFPPPPCCCC"))
IA = np.linspace(0.1, 3.0, T).astype(np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(**kw):
    torch.manual_seed(0)
    return HiGO(PLM, T, ANC, IA, n_taxa=3, adapter="frozen", d=32, aspect=ASPECT, **kw).to(DEV).eval()


def batch():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PLM)
    enc = tok(["MKTAYIAKQRQISFVKSHFSRQ", "MSEQLLK"], return_tensors="pt", padding=True)
    res = enc["attention_mask"].clone()
    res[:, 0] = 0
    res[torch.arange(2), enc["attention_mask"].sum(1) - 1] = 0
    return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV), res.to(DEV), torch.tensor([1, 0], device=DEV)


def test_crla_identity_at_init():
    torch.manual_seed(0)
    blk = CRLA(48, 8, 5)
    h, c = torch.randn(2, 7, 48), torch.randn(2, 5)
    assert torch.equal(blk(h, c), h)
    blk(h, c).pow(2).sum().backward()
    assert blk.alpha.grad.abs() > 0, "alpha must receive gradient at init (W_up is not zeroed)"
    assert blk.up.weight.abs().sum() > 0


def test_crla_model_matches_plain_head():
    ids, att, res, tx = batch()
    plain, with_crla = tiny(), tiny(crla=True, crla_dim=16)
    missing = with_crla.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert all(k.startswith("crla.") or k == "term_aspect" for k in missing)
    with torch.no_grad():
        z0, _ = plain(ids, att, res, tx)
        z1, _ = with_crla(ids, att, res, tx)
    assert torch.equal(z0, z1)


def test_mix_readout_and_truncation():
    ids, att, res, tx = batch()
    m = tiny(plm_layers=4, readout="mix", readout_layers=[2, 4])
    assert len(m.plm.encoder.layer) == 4
    with torch.no_grad():
        z, _ = m(ids, att, res, tx)
    assert z.shape == (2, T) and torch.isfinite(z).all()
    w = torch.softmax(m.mix.w, 0)
    assert w[-1] > 0.95, "top layer dominates at init"


def test_knn_fusion_no_hit_fallback_and_hierarchy():
    ids, att, res, tx = batch()
    m = tiny(knn_fusion=True)
    p = torch.rand(2, T, device=DEV)
    kn = {"p": p, "max_id": torch.tensor([0.9, 0.0], device=DEV), "has_hit": torch.tensor([True, False], device=DEV),
          "s_star": p * 0.9, "frac": p, "count": (p * 10).round()}
    with torch.no_grad():
        z0, _ = m(ids, att, res, tx)
        z1, _ = m(ids, att, res, tx, knn=kn)
    assert torch.equal(z1[1], z0[1].float()), "no-hit protein must keep the HiGO logits"
    lam = torch.nn.functional.softplus(m.knn_a[0])
    lp = torch.logit(p[0].clamp(1e-3, 1 - 1e-3))
    assert torch.allclose(z1[0] - z0[0].float(), lam * lp, atol=1e-4), "g = 0 at init, so the shift is lambda*logit"
    q = m.predict_proba(z1)
    assert (q[:, ANC[:, 1]] >= q[:, ANC[:, 0]]).all(), "MCM output must be hierarchy-consistent"


def test_load_trainable_is_loud():
    m = tiny(knn_fusion=True)
    state = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    assert load_trainable(m, state) == []
    for bad in ({k: v for k, v in state.items() if k != "knn_delta"}, {**state, "not_a_param": torch.zeros(1)}):
        try:
            load_trainable(m, bad)
        except RuntimeError:
            continue
        raise AssertionError("mismatch was not reported")
    assert load_trainable(m, {k: v for k, v in state.items() if k != "knn_delta"}, optional=("knn_delta",)) == ["knn_delta"]


def toy_df():
    df = pd.DataFrame({"id": [f"P{i}" for i in range(8)], "split": ["train"] * 5 + ["val"] * 3})
    Y = sp.csr_matrix((np.random.default_rng(0).random((8, T)) > 0.6).astype(np.float32))
    return df, Y


def test_knn_contract_loader():
    df, Y = toy_df()
    val = np.where(df.split == "val")[0]
    P = sp.csr_matrix(np.array([[0.5] * T, [0] * T, [0.2] * T], np.float32))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "val_knn.npz"
        np.savez(path, data=P.data, indices=P.indices, indptr=P.indptr, shape=np.array(P.shape),
                 max_identity=np.array([95.0, 0.0, 40.0]), ids=np.array(["P7", "P5", "P6"]))
        src = KnnPrior.from_contract(path, df, val, T)
        out = src.batch([5, 6, 7])
        assert np.allclose(out["max_id"], [0.0, 0.4, 0.95]) and out["has_hit"].tolist() == [False, True, True]
        assert np.allclose(out["p"][2], 0.5) and out["s_star"] is None
        assert not src.batch([7], cap=0)["has_hit"].any()
        for call in (lambda: src.batch([7], cap=30), lambda: KnnBank([src]).batch([0])):
            try:
                call()
            except (ValueError, KeyError):
                continue
            raise AssertionError("expected a loud failure")
        np.savez(path, data=P.data, indices=P.indices, indptr=P.indptr, shape=np.array(P.shape), max_id=np.zeros(3))
        assert KnnPrior.from_contract(path, df, val, T).row == {5: 0, 6: 1, 7: 2}


def test_knn_hits_caps_and_self_exclusion():
    df, Y = toy_df()
    tr = np.where(df.split == "train")[0]
    hits = pd.DataFrame({"q": ["P0", "P0", "P0", "P0"], "s": ["P0", "P1", "P2", "P3"],
                         "pident": [100.0, 92.0, 45.0, 25.0], "bits": [500.0, 300.0, 120.0, 60.0],
                         "qcov": [100.0, 90.0, 80.0, 70.0]})
    src = KnnPrior.from_hits(hits, df, Y, tr, tr, T=2.0, top_k=10)
    full = src.batch([0])
    assert np.isclose(full["max_id"][0], 0.92), "self hit must be excluded"
    w = np.array([300.0, 120.0 * np.exp(-47 / 2), 60.0 * np.exp(-67 / 2)])
    want = w @ Y[[1, 2, 3]].toarray() / w.sum()
    assert np.allclose(full["p"][0], want, atol=1e-6)
    assert np.isclose(src.batch([0], cap=50)["max_id"][0], 0.45)
    assert np.isclose(src.batch([0], cap=30)["max_id"][0], 0.25)
    assert not src.batch([0], cap=0)["has_hit"][0] and not src.batch([1])["has_hit"][0]
    assert np.allclose(full["count"][0], Y[[1, 2, 3]].toarray().sum(0))


def test_sparse_pred_roundtrip():
    rng = np.random.default_rng(0)
    p = rng.random((5, 300)).astype(np.float16)
    with tempfile.TemporaryDirectory() as d:
        save_pred(Path(d), "val", p, top_k=20)
        q = load_pred(Path(d), "val")
    for i in range(5):
        kept = q[i] > 0
        assert 0 < kept.sum() <= 20 and np.array_equal(q[i, kept], p[i, kept])
        assert p[i, ~kept].max() <= p[i, kept].min()
    tie = np.array([[0.9, 0.5, 0.5, 0.1]], np.float16)          # term 1 is the parent of term 2, tied at the cut
    with tempfile.TemporaryDirectory() as d:
        save_pred(Path(d), "val", tie, top_k=2)
        q = load_pred(Path(d), "val")
    assert (q[0, 1] > 0) == (q[0, 2] > 0), "a tie at the cut must keep or drop both"


def test_fast_entmax_matches_bisect():
    from entmax import entmax_bisect
    from cafa6.model import entmax15_bisect_fused
    torch.manual_seed(0)
    X = torch.randn(3, 5, 40, device=DEV, dtype=torch.float64) * 2
    X[:, :, 30:] = -1e4                                   # padding, as in HiGO.forward
    G = torch.randn_like(X)
    xa, xb = X.clone().requires_grad_(), X.clone().requires_grad_()
    ya, yb = entmax_bisect(xa, alpha=1.5, dim=-1, n_iter=30), entmax15_bisect_fused(xb, 30)
    ya.backward(G); yb.backward(G)
    assert yb.dtype == X.dtype and torch.equal(ya > 0, yb > 0)
    assert torch.allclose(ya.float(), yb.float(), atol=1e-6) and torch.allclose(xa.grad, xb.grad, atol=1e-5)
    ids, att, res, tx = batch()
    with torch.no_grad():
        z0, _ = tiny()(ids, att, res, tx)
        z1, _ = tiny(fast_entmax=True)(ids, att, res, tx)
    assert torch.allclose(z0, z1, atol=1e-5)


def test_prefetch_matches_inline_batches():
    import random
    from cafa6.train import Batcher, iter_train_batches
    from transformers import AutoTokenizer
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), n)) for n in rng.integers(5, 80, 60)]
    Y = sp.csr_matrix(rng.random((60, T)) < 0.2, dtype=np.float32)
    taxa = rng.integers(0, 3, 60)
    bt = Batcher(seqs, AutoTokenizer.from_pretrained(PLM), 32, 256, 8)
    runs = []
    for prefetch in (False, True):
        random.seed(1)
        batches = bt.batches(range(60), True)
        runs.append(list(iter_train_batches(batches, 2, lambda b: bt.train_tensors(b, Y, taxa, prefetch), prefetch)))
    assert [bi for bi, _ in runs[0]] == [bi for bi, _ in runs[1]] and runs[0][0][0] == 2
    for (_, a), (_, b) in zip(*runs):
        assert a[0] == b[0] and a[-1] == b[-1] and all(torch.equal(x, y) for x, y in zip(a[1:-1], b[1:-1]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
