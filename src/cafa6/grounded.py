"""HiGO-v2: ground HiGO's GO scores in computed evidence, then recalibrate per aspect.

Evidence per (protein, term), all restricted to the model vocabulary and propagated up the ontology
(is_a + part_of):
  D  domain support: InterPro matches of the protein -> interpro2go. InterPro matches are computed by
     InterProScan, so they are allowed for test proteins.
  O  ortholog support: DIAMOND hits in the 10 reference proteomes (same species removed), scored with the
     training labels of the hit proteins (bits-weighted, as `diamond_score`).
  G  GOA support: GOA annotations of those ortholog hits (hits that are themselves evaluation proteins are
     dropped; `--goa_date_max` removes annotations newer than a benchmark's cutoff), weighted by identity.
  L  ligand consistency: the term's ChEBI ligands (go2ligand: GO logical definitions, Rhea) intersect the
     ligands implied for the protein, i.e. ligands of its domain-supported MF terms and curated
     binding-site / cofactor ligands of its ortholog hits (training or reference proteins only).
The calibrator is a per-aspect logistic model over [logit p_model, logit p_knn, identity, has_hit, D, O,
G, L, term has ligand, log term frequency, interactions], fit on validation, then MCM keeps the output
hierarchy-consistent.

Usage: python -m cafa6.grounded --run dev_35m_lora --out grounded
       CAFA6_PROC=data/proc_gogpt python -m cafa6.grounded --bench gogpt --run gogptbench_35m_lora \
           --out grounded_gogpt --goa_date_max 20221130
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import networkx as nx
import numpy as np
import obonet
import pandas as pd
import scipy.sparse as sp
import torch

from cafa6.comparative import REFERENCE_TAXA
from cafa6.data import PROC, ROOT
from cafa6.evaluate import IA, OBO, dense_to_long, evaluate, summarize
from cafa6.homology import diamond_score, diamond_search
from cafa6.stack import ASPECTS, apply_gate, features, fit_gate, mcm_numpy

GR = ROOT / "data" / "grounding"
REF = ROOT / "runs" / "gogpt_cafa5" / "ref"
COMPONENTS = ["lp", "lk", "pid", "hit", "D", "O", "G", "L", "haslig", "lfreq", "lp_pid", "lk_pid", "D_lp"]
DROP = {"no_domain": ["D", "D_lp", "L_dom"], "no_ortholog": ["O", "G", "L_hit"], "no_ligand": ["L", "L_dom", "L_hit"],
        "gate_only": ["D", "O", "G", "L", "haslig", "D_lp", "L_dom", "L_hit"]}


# ----------------------------------------------------------------------------- ontology helpers
class Closure:
    """Maps any GO id (incl. alt ids) to the vocab indices of itself and its is_a/part_of ancestors."""

    def __init__(self, obo_path, vocab_terms):
        g = obonet.read_obo(str(obo_path))
        self.alt = {a: n for n, d in g.nodes(data=True) for a in d.get("alt_id", [])}
        self.h = nx.DiGraph([(u, v) for u, v, k in g.edges(keys=True) if k in ("is_a", "part_of")])
        self.h.add_nodes_from(g.nodes)
        self.idx = {t: i for i, t in enumerate(vocab_terms)}
        self.cache: dict[str, tuple] = {}

    def __call__(self, term: str) -> tuple:
        term = self.alt.get(term, term)
        if term not in self.cache:
            if term not in self.h:
                self.cache[term] = ()
            else:
                anc = {term} | nx.descendants(self.h, term)
                self.cache[term] = tuple(sorted(self.idx[a] for a in anc if a in self.idx))
        return self.cache[term]


    def matrix(self, go_terms) -> sp.csr_matrix:
        """len(go_terms) x T binary matrix of each term's vocab closure."""
        r, c = [], []
        for i, t in enumerate(go_terms):
            for k in self(t):
                r.append(i); c.append(k)
        return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(go_terms), len(self.idx)))


def incidence(keys_per_row, vocab_keys) -> sp.csr_matrix:
    """Binary rows x len(vocab_keys) matrix from a list of key lists."""
    kidx = {k: i for i, k in enumerate(vocab_keys)}
    r, c = [], []
    for i, keys in enumerate(keys_per_row):
        for k in keys:
            j = kidx.get(k)
            if j is not None:
                r.append(i); c.append(j)
    m = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(keys_per_row), len(vocab_keys)))
    m.data[:] = 1.0
    return m


def binarize(m: sp.spmatrix) -> sp.csr_matrix:
    m = m.tocsr()
    m.data[:] = 1.0
    m.eliminate_zeros()
    return m


# ----------------------------------------------------------------------------- grounding resources
class Grounding:
    """Vectorised evidence builder; all matrices are proteins x vocab terms."""

    def __init__(self, vocab, closure, goa_date_max=None, exclude_ids=()):
        self.vocab, self.closure, self.T = vocab, closure, len(vocab)
        self.uf = pd.read_parquet(GR / "uniprot_features.parquet").drop_duplicates("id").set_index("id")
        ip2go = pd.read_csv(GR / "interpro2go.tsv", sep="\t")
        ip2go.columns = ["interpro", "go"] + list(ip2go.columns[2:])
        self.ips = sorted(ip2go.interpro.unique())
        gos = sorted(ip2go.go.unique())
        IG = incidence(ip2go.groupby("interpro").go.apply(list).reindex(self.ips).tolist(), gos)
        self.IP = binarize(IG @ closure.matrix(gos))                       # interpro x T
        lig = pd.read_csv(GR / "go2ligand.tsv", sep="\t").dropna(subset=["chebi"])
        # currency metabolites (H2O, H+, ATP, NAD+ ...) from reactions would link most enzyme terms to each other;
        # they are kept only when the GO definition itself names them (e.g. ATP binding -> ATP)
        lig = lig[~(lig.ubiquitous & lig.source.isin(["rhea", "ec"]))]
        self.lig = lig
        self.chebis = sorted(lig.chebi.unique())
        self.cidx = {c: i for i, c in enumerate(self.chebis)}
        tidx = {t: i for i, t in enumerate(vocab.term.values)}
        lv = lig[lig.go.isin(tidx)]
        self.LigT = binarize(sp.csr_matrix((np.ones(len(lv), np.float32),
                                            (lv.go.map(tidx).values, lv.chebi.map(self.cidx).values)),
                                           shape=(self.T, len(self.chebis))))   # T x chebi
        self.has_lig = np.asarray(self.LigT.sum(1)).ravel() > 0
        goa = pd.read_parquet(GR / "goa_reference.parquet", columns=["id", "go", "date"])
        if goa_date_max is not None:
            goa = goa[goa.date <= goa_date_max]
        goa = goa[~goa.id.isin(set(exclude_ids))].drop_duplicates(["id", "go"])
        self.goa_ids = np.array(sorted(goa.id.unique()))
        gos = sorted(goa.go.unique())
        GG = incidence(goa.groupby("id").go.apply(list).reindex(self.goa_ids).tolist(), gos)
        self.GOA = binarize(GG @ closure.matrix(gos))                      # goa protein x T
        self.goa_row = {p: i for i, p in enumerate(self.goa_ids)}

    def _lists(self, ids, col):
        s = self.uf[col].reindex(ids)
        return [list(v) if isinstance(v, (list, np.ndarray)) and len(v) else [] for v in s.values]

    def domains(self, ids, interpro_lists=None) -> sp.csr_matrix:
        """interpro_lists overrides the UniProt matches (used for new sequences run through InterProScan)."""
        lists = interpro_lists if interpro_lists is not None else self._lists(ids, "interpro")
        return binarize(incidence(lists, self.ips) @ self.IP)

    def orthologs(self, hits, ids, q_taxon, db_df, Y_db):
        """hits: DIAMOND hits (q, s, pident, bits) of `ids` against db_df (training proteins).
        Returns O (dense, training labels of cross-species reference hits), G (GOA of those hits, max identity
        per term) and the filtered hit table."""
        tax = pd.Series(db_df.taxon.values, index=db_df.id.values)
        tax = tax[~tax.index.duplicated()]
        h = hits[hits.q.isin(set(ids)) & hits.s.isin(tax.index)].copy()
        h["st"] = h.s.map(tax).values
        h["qt"] = h.q.map(q_taxon).values
        h = h[h.st.isin(REFERENCE_TAXA) & (h.st != h.qt)]
        O = diamond_score(h, ids, db_df.id.values, Y_db).astype(np.float32)
        qi = {p: i for i, p in enumerate(ids)}
        hg = h[h.s.isin(self.goa_row)].sort_values("pident", ascending=False)
        hg = hg.assign(rank=hg.groupby("q").cumcount())
        G = sp.csr_matrix((len(ids), self.T), dtype=np.float32)
        for _, grp in hg.groupby("rank"):
            rows = grp.q.map(qi).values
            S = sp.csr_matrix((grp.pident.values.astype(np.float32) / 100, (rows, grp.s.map(self.goa_row).values)),
                              shape=(len(ids), len(self.goa_ids)))
            G = G.maximum(S @ self.GOA)
        return O, G.tocsr(), h

    def ligands(self, D, h, ids, return_protein_ligands=False):
        """Returns (L_dom, L_hit): term-level ligand consistency from domain-implied and ortholog ligands
        (plus the protein x ChEBI matrices when return_protein_ligands)."""
        mf = sp.diags((self.vocab.aspect.values == "F").astype(np.float32))
        PL_dom = binarize(D @ mf @ self.LigT)
        s_ids = np.array(sorted(h.s.unique()))
        s_lig = [a + b for a, b in zip(self._lists(s_ids, "binding_chebi"), self._lists(s_ids, "cofactor_chebi"))]
        SL = incidence(s_lig, self.chebis)
        qi, si = {p: i for i, p in enumerate(ids)}, {p: i for i, p in enumerate(s_ids)}
        H = binarize(sp.csr_matrix((np.ones(len(h), np.float32), (h.q.map(qi).values, h.s.map(si).values)),
                                   shape=(len(ids), len(s_ids))))
        PL_hit = binarize(H @ SL)
        L = binarize(PL_dom @ self.LigT.T), binarize(PL_hit @ self.LigT.T)
        return (*L, PL_dom, PL_hit) if return_protein_ligands else L


# ----------------------------------------------------------------------------- calibrator
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def components(p_model, p_knn, X, ev, lfreq, has_lig, cols, drop=()):
    """Dense per-component arrays for the term columns `cols` (numpy, n x len(cols))."""
    n = p_model.shape[0]
    lp, lk = logit(p_model[:, cols]), logit(p_knn[:, cols])
    pid = np.repeat(X[:, :1], len(cols), 1)
    hit = np.repeat(X[:, 1:2], len(cols), 1)
    dense = lambda m: m[:, cols].toarray() if sp.issparse(m) else m[:, cols]
    D = dense(ev["D"]) if "D" not in drop else np.zeros((n, len(cols)), np.float32)
    O = dense(ev["O"]) if "O" not in drop else np.zeros_like(D)
    G = dense(ev["G"]) if "G" not in drop else np.zeros_like(D)
    L_dom = dense(ev["L_dom"]) if "L_dom" not in drop else np.zeros_like(D)
    L_hit = dense(ev["L_hit"]) if "L_hit" not in drop else np.zeros_like(D)
    L = np.maximum(L_dom, L_hit) if "L" not in drop else np.zeros_like(D)
    hl = np.repeat(has_lig[None, cols].astype(np.float32), n, 0) if "haslig" not in drop else np.zeros_like(D)
    comp = {"lp": lp, "lk": lk, "pid": pid, "hit": hit, "D": D, "O": O, "G": G, "L": L, "haslig": hl,
            "lfreq": np.repeat(lfreq[None, cols], n, 0), "lp_pid": lp * pid, "lk_pid": lk * pid,
            "D_lp": D * lp if "D_lp" not in drop else np.zeros_like(D)}
    return np.stack([comp[k] for k in COMPONENTS], -1).astype(np.float32)   # n x t x F


def fit_calibrator(F, y, steps=400, lr=0.05, device="cuda"):
    Ft = torch.as_tensor(F, device=device)
    mu, sd = Ft.mean((0, 1)), Ft.std((0, 1)).clamp(min=1e-3)
    Fn = (Ft - mu) / sd
    yt = torch.as_tensor(y, device=device, dtype=torch.float32)
    w = torch.zeros(F.shape[-1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        z = Fn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return {"w": (w / sd).detach().cpu().numpy(), "b": float((b - (w / sd * mu).sum()).item())}


def apply_calibrator(F, params):
    z = F @ params["w"] + params["b"]
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def calibrate(p_model_va, p_knn_va, Xva, ev_va, Yva, p_model_te, p_knn_te, Xte, ev_te, vocab, lfreq, has_lig,
              drop=(), params=None):
    aspect = vocab.aspect.values
    out = np.zeros_like(p_model_te)
    fitted = {}
    for a in ASPECTS:
        cols = np.where(aspect == a)[0]
        if params is None:
            Fva = components(p_model_va, p_knn_va, Xva, ev_va, lfreq, has_lig, cols, drop)
            fitted[a] = fit_calibrator(Fva, Yva[:, cols])
            del Fva
        else:
            fitted[a] = params[a]
        for s in range(0, len(p_model_te), 2048):
            sl = slice(s, s + 2048)
            ev_s = {k: v[sl] for k, v in ev_te.items()}
            Fte = components(p_model_te[sl], p_knn_te[sl], Xte[sl], ev_s, lfreq, has_lig, cols, drop)
            out[sl, cols] = apply_calibrator(Fte, fitted[a])
    return out, fitted


# ----------------------------------------------------------------------------- stability
def violations(p, anc, tol=1e-6) -> int:
    return int((p[:, anc[:, 0]] > p[:, anc[:, 1]] + tol).sum())


def jaccard_across(preds: list[np.ndarray], tau: np.ndarray) -> float:
    sets = [p >= tau[None, :] for p in preds]
    vals = []
    for A, B in itertools.combinations(sets, 2):
        inter, union = (A & B).sum(1), (A | B).sum(1)
        vals.append(np.where(union > 0, inter / np.maximum(union, 1), 1.0).mean())
    return float(np.mean(vals))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="cafa6", choices=["cafa6", "gogpt"])
    ap.add_argument("--run", default="dev_35m_lora")
    ap.add_argument("--out", default="grounded")
    ap.add_argument("--goa_date_max", type=int, default=None)
    ap.add_argument("--stability_runs", nargs="*", default=[f"cv_35m_fold{i}" for i in range(5)])
    args = ap.parse_args()
    out = ROOT / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    obo, ia = (OBO, IA) if args.bench == "cafa6" else (REF / "go-basic_gogpt.obo", REF / "IA.txt")

    df = pd.read_parquet(PROC / "train.parquet")
    if args.bench == "gogpt":
        # proc_gogpt stores GO-GPT's organism codes; the ortholog filter needs NCBI taxon ids
        uf_tax = pd.read_parquet(GR / "uniprot_features.parquet", columns=["id", "taxon"]).drop_duplicates("id")
        df["taxon"] = df.id.map(uf_tax.set_index("id").taxon).fillna(-1).astype(int).values
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    terms, aspect = vocab.term.values, vocab.aspect.values
    anc = np.load(PROC / "go_ancestors.npy")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    ident = pd.read_parquet(PROC / "identity.parquet")
    tr, va, te = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
    va_ids, te_ids = df.id.values[va], df.id.values[te]
    Yva, Yte = Y[va].toarray(), Y[te].toarray()
    Xva = features(ident[ident.split == "val"], va_ids)
    Xte = features(ident[ident.split == "test"], te_ids)
    knn_va = np.load(PROC / "diamond_val.npy").astype(np.float32)
    knn_te = np.load(PROC / "diamond_test.npy").astype(np.float32)
    pv = np.load(ROOT / "runs" / args.run / "pred_val.npy").astype(np.float32)
    pt = np.load(ROOT / "runs" / args.run / "pred_test.npy").astype(np.float32)
    freq = np.asarray(Y[tr].mean(0)).ravel().astype(np.float32)
    lfreq = np.log(freq + 1e-5).astype(np.float32)

    hits_path = PROC / "diamond_hits_valtest.parquet"
    if not hits_path.exists():
        hits = diamond_search(df.iloc[tr], df[df.split != "train"])
        hits.to_parquet(hits_path)
    hits = pd.read_parquet(hits_path)

    closure = Closure(obo, terms)
    eval_ids = set(va_ids) | set(te_ids)
    gr = Grounding(vocab, closure, goa_date_max=args.goa_date_max, exclude_ids=eval_ids)
    db = df.iloc[tr]
    evs = {}
    for name, idx, ids in (("val", va, va_ids), ("test", te, te_ids)):
        q_tax = pd.Series(df.taxon.values[idx], index=ids)
        q_tax = q_tax[~q_tax.index.duplicated()]
        D = gr.domains(ids)
        O, G, h = gr.orthologs(hits, ids, q_tax, db, Y[tr])
        L_dom, L_hit = gr.ligands(D, h, ids)
        evs[name] = {"D": D, "O": O, "G": G, "L_dom": L_dom, "L_hit": L_hit}
        cov = {k: float((np.asarray((v > 0).sum(1)).ravel() > 0).mean()) for k, v in evs[name].items()}
        print(f"{name}: evidence coverage (proteins with any support) {cov}", flush=True)
        sp.save_npz(out / f"evidence_D_{name}.npz", D.tocsr())
    np.save(out / "has_ligand.npy", gr.has_lig)

    scores = {"naive": np.tile(freq, (len(te), 1)),
              "diamond+naive": np.where((knn_te.sum(1) > 0)[:, None], 0.8 * knn_te + 0.2 * freq, freq),
              args.run: pt}
    W = fit_gate(pv, knn_va, Yva, Xva, aspect)
    scores[f"{args.run}+knn_gate"] = mcm_numpy(apply_gate(pt, knn_te, Xte, aspect, W), anc)
    params_all = {}
    for variant, drop in [("grounded", ())] + [(f"grounded_{k}", v) for k, v in DROP.items()]:
        p, params = calibrate(pv, knn_va, Xva, evs["val"], Yva, pt, knn_te, Xte, evs["test"], vocab, lfreq,
                              gr.has_lig, drop=drop)
        scores[variant] = mcm_numpy(p, anc)
        params_all[variant] = {a: {"w": dict(zip(COMPONENTS, np.round(v["w"], 4).tolist())), "b": round(v["b"], 4)}
                               for a, v in params.items()}
        if variant == "grounded":
            best_params = params
            pv_g, _ = calibrate(pv, knn_va, Xva, evs["val"], Yva, pv, knn_va, Xva, evs["val"], vocab, lfreq,
                                gr.has_lig, params=params)
            np.save(out / "pred_val.npy", mcm_numpy(pv_g, anc).astype(np.float16))
            np.save(out / "pred_test.npy", scores[variant].astype(np.float16))
        print(f"calibrated {variant}", flush=True)
    (out / "calibrator.json").write_text(json.dumps(params_all, indent=2))
    (out / "gate.json").write_text(json.dumps({a: W[a].tolist() for a in ASPECTS}))

    res = evaluate({k: dense_to_long(v, te_ids, terms) for k, v in scores.items()}, PROC / "gt_test.tsv",
                   out_dir=out, obo=obo, ia=ia)
    summ = summarize(res)
    summ.to_csv(out / "summary.tsv", sep="\t")
    print(summ.round(4).to_string(), flush=True)
    viol = {k: violations(v, anc) for k, v in scores.items()}
    (out / "violations.json").write_text(json.dumps(viol, indent=2))
    print("ontology violations:", viol)

    # stability: same gate / calibrator (fit on the main run's validation predictions) applied to models
    # trained on different folds; measures score spread and agreement of the predicted term sets
    stab_runs = [args.run] + [r for r in args.stability_runs if (ROOT / "runs" / r / "pred_test.npy").exists()]
    if args.bench == "cafa6" and len(stab_runs) > 1:
        per = {"raw": [], "gate": [], "grounded": []}
        preds = {}
        for r in stab_runs:
            p = np.load(ROOT / "runs" / r / "pred_test.npy").astype(np.float32)
            g = mcm_numpy(apply_gate(p, knn_te, Xte, aspect, W), anc)
            q, _ = calibrate(pv, knn_va, Xva, evs["val"], Yva, p, knn_te, Xte, evs["test"], vocab, lfreq,
                             gr.has_lig, params=best_params)
            q = mcm_numpy(q, anc)
            for k, v in (("raw", p), ("gate", g), ("grounded", q)):
                per[k].append(v)
                preds[f"{k}__{r}"] = dense_to_long(v, te_ids, terms)
        rs = evaluate(preds, PROC / "gt_test.tsv", obo=obo, ia=ia)
        rs["variant"], rs["run"] = zip(*rs.method.str.split("__"))
        tau_of = res.set_index(["method", "aspect"]).tau_w
        name_of = {"raw": args.run, "gate": f"{args.run}+knn_gate", "grounded": "grounded"}
        rows = []
        for k in per:
            tau = np.array([tau_of[(name_of[k], {"F": "MF", "P": "BP", "C": "CC"}[a])] for a in aspect], np.float32)
            s = rs[rs.variant == k].pivot(index="run", columns="aspect", values="Fmax_w")
            kag = s[["MF", "BP", "CC"]].mean(1)
            rows.append({"variant": k, "n_models": len(stab_runs), "kaggle_mean": kag.mean(), "kaggle_std": kag.std(),
                         "MF_std": s.MF.std(), "BP_std": s.BP.std(), "CC_std": s.CC.std(),
                         "set_jaccard": jaccard_across(per[k], tau),
                         "violations": int(sum(violations(v, anc) for v in per[k]))})
        stab = pd.DataFrame(rows)
        stab.to_csv(out / "stability.tsv", sep="\t", index=False)
        print(stab.round(4).to_string())
    (ROOT / "logs" / f"{args.out}.done").touch()


if __name__ == "__main__":
    main()
