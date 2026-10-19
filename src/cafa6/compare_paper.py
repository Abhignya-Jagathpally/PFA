"""Method comparison of the BioReason-Pro paper (bioRxiv 2026.03.19.712954) under one scorer.

Every method of the paper's GO comparison is put on the GO-GPT temporal test set (8,630 proteins,
data/proc_gogpt/gt_test.tsv; GO 2023-01-01 runs/gogpt_cafa5/ref/go-basic_gogpt.obo; CAFA-5 IA
runs/gogpt_cafa5/ref/IA.txt) and scored with `knn_sharp.Scorer`, the in-memory copy of cafaeval
(prop=max, norm=cafa, thresholds 0.01..0.99, IA-weighted), so HiGO-R2 (frozen look-3 file
runs/stacker_tv/l3_c0/pred_test_twin.npz) is compared with every method on equal terms. Nothing here
fits or selects anything on test.

Methods and sources of predictions
  HiGO-R2                 frozen file (not altered)
  GO-GPT beam / greedy    runs/gogpt_cafa5 (released checkpoint, our generation)
  GO-GPT ten-sample       runs/gogpt_sample10 term frequencies (T 0.7, top-k 20)
  GO-GPT (authors' file)  `go_pred` column shipped in the HF test set (the predictions behind the paper)
  BioReason-Pro RL        released wanglab/bioreason-pro-rl, run with the authors' eval.py and the settings
                          of scripts/sh_eval.sh for the published benchmark (greedy, max_new_tokens 3000,
                          ask_all_go_aspects False, force_uniprot_summary False); GO IDs are taken from the
                          whole response with score 1.0 as evals/cafa_evals.py does (a final-answer-only
                          variant is scored as well)
  BLAST-KNN               paper section 4.4.2: best hit by percent identity, all its GO terms, score = identity
                          (DIAMOND --more-sensitive e<=1e-3 hits against GO-GPT training proteins stand in for
                          blastp)
  DIAMONDScore            DeepGOPlus: sum of bitscores of hits carrying the term / sum of all hit bitscores
  Naive                   per-aspect term frequency among training proteins annotated in that aspect
  SPROF-GO, DeepGOPlus, ProteInfer, DeepGO-SE, DeepFRI (sequence CNN), InterLabelGO+ (released package,
                          test labels in its training data)   official weights, our runs, imported with
                          import-tsv (runs/compare_paper/extra_models.md)
  ProtBoost, BioReason-Pro SFT   paper-reported only (see runs/compare_paper/report.md)

Subsets: S3_all (8,630) and S6_noknowledge (without the 471 limited-knowledge proteins of
data/common_proteins.txt); identity bins from data/proc_gogpt_tv2/knn/test_knn.npz max_pident.
Paired protein bootstrap (1,000 reps, seed 0) of HiGO-R2 minus each method on the weighted Fmax mean,
Holm-corrected two-sided bootstrap p-values across methods.

Usage (PYTHONPATH=src, from cafa6/):
  python -m cafa6.compare_paper bioreason-gen --gpu 0 --evals_path runs/compare_paper/bioreason_rl/evals
         [--num_chunks 2 --chunk_id 0] [--max_samples 20] [--batch_size 64]
  python -m cafa6.compare_paper baselines      writes runs/compare_paper/preds/{naive,blast_knn,diamondscore}.npz
  python -m cafa6.compare_paper collect-bioreason --evals_path ...   -> preds/bioreason_rl*.npz
  python -m cafa6.compare_paper import-tsv --tsv preds.tsv --name NAME [--header] [--meta JSON]
                                               external (protein, GO id, score) rows -> preds/NAME.npz
  python -m cafa6.compare_paper score          summary.tsv, per_bin.tsv, report tables
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "compare_paper"
PREDS = OUT / "preds"
BR_REPO = ROOT / "external" / "BioReason-Pro-github"
RL_CKPT = Path("/dev/shm/cmp_paper_bioreason_rl")
GO_EMB = Path("/dev/shm/cmp_paper_go_emb/go_embeddings")
DS_CACHE = Path("/dev/shm/cmp_paper_ds_cache")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------- BioReason-Pro

def _transformers_shims():
    """transformers 5 dropped processing_utils.CommonKwargs and changed ProcessorMixin kwarg merging; the
    authors' PLProcessor only uses them to pass text kwargs through (it pops `padding` and tokenizes with
    fixed arguments), so an empty text_kwargs dict reproduces its behaviour."""
    from typing import Optional, TypedDict

    import types

    import transformers.processing_utils as pu
    if not hasattr(pu, "CommonKwargs"):
        pu.CommonKwargs = TypedDict("CommonKwargs", {"return_tensors": Optional[str]}, total=False)
    # bioreason2/models/__init__ imports the training module, which imports unsloth at module level; unsloth
    # patches transformers/vLLM on import and is never called on the vLLM inference path.
    stub = types.ModuleType("unsloth")
    stub.FastLanguageModel = None
    sys.modules["unsloth"] = stub
    # the installed flash_attn wheel was built against another torch (undefined symbol on import); vLLM only
    # tolerates ModuleNotFoundError when probing for it, which a None entry produces
    sys.modules["flash_attn"] = None


def _patch_processor():
    from bioreason2.models.pl import processing_pl

    def merge(self, *args, **kwargs):
        return {"text_kwargs": {}}
    processing_pl.PLProcessor._merge_kwargs = merge

    # esm 3.4 transformer blocks return (x, attn_weights); the authors' hook expects the tensor x
    from bioreason2.models import protein_encoder

    def register(self, layer_idx):
        def hook_fn(module, inp, output):
            self.layer_outputs[layer_idx] = output[0] if isinstance(output, tuple) else output
        h = self.model.transformer.blocks[layer_idx].register_forward_hook(hook_fn)
        self.hooks.append(h)
        return h
    protein_encoder.ESM3Encoder._register_layer_hook = register


def bioreason_args(a) -> list[str]:
    """scripts/sh_eval.sh of the authors' repo, with the two flags its comments say reproduce the published
    benchmark (ASK_ALL_GO_ASPECTS=False, FORCE_UNIPROT_SUMMARY=False)."""
    args = {
        "ckpt_dir": RL_CKPT, "protein_model_name": "esm3_sm_open_v1", "protein_embedding_layer": 37,
        "go_obo_path": BR_REPO / "bioreason2/dataset/go-basic.obo", "precomputed_embeddings_path": GO_EMB,
        "unified_go_encoder": True, "go_hidden_dim": 512, "go_num_gat_layers": 3, "go_num_heads": 8,
        "go_num_reduced_embeddings": 200, "go_embedding_dim": 2560, "max_model_len": 8192,
        "gpu_memory_utilization": a.gpu_mem, "max_num_seqs": 256, "batch_size": a.batch_size,
        "cafa5_dataset": "wanglab/bioreason-pro-test-data", "cafa5_dataset_name": "default",
        "reasoning_dataset_name": "default", "go_gpt_predictions_column": "go_pred",
        "dataset_cache_dir": DS_CACHE, "structure_dir": "", "split_go_aspects": False,
        "interpro_in_prompt": True, "predict_interpro": False, "ppi_in_prompt": True,
        "ask_all_go_aspects": a.ask_all, "force_uniprot_summary": a.ask_all, "add_uniprot_summary": True,
        "min_go_mf_freq": 1, "min_go_bp_freq": 1, "min_go_cc_freq": 1, "apply_go_filtering_to_val_test": False,
        "seed": 23, "debug": False, "max_samples": a.max_samples, "max_length_protein": 2000,
        "max_new_tokens": a.max_new_tokens, "temperature": 0, "top_p": 0.95, "pass_at_k": 1,
        "repetition_penalty": 1.0, "evals_path": Path(a.evals_path).resolve(),
    }
    if a.num_chunks > 1:
        args.update(num_chunks=a.num_chunks, chunk_id=a.chunk_id)
    out = []
    for k, v in args.items():
        out += [f"--{k}", str(v)]
    return out


def cmd_bioreason_gen(a):
    a.evals_path = str(Path(a.evals_path).resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["HF_DATASETS_CACHE"] = str(DS_CACHE)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # a spawned engine-core process would re-run eval.py as __main__ without the shims below
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # FlashInfer's sampler JIT-compiles kernels at warm-up; decoding is greedy, so vLLM's torch sampler is used
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    for p in (RL_CKPT / "model.safetensors.index.json", GO_EMB):
        if not p.exists():
            sys.exit(f"missing {p}: download wanglab/bioreason-pro-rl and bioreason-pro-go-embeddings first")
    sys.path.insert(0, str(BR_REPO))
    os.chdir(BR_REPO)
    _transformers_shims()
    _patch_processor()
    Path(a.evals_path).mkdir(parents=True, exist_ok=True)
    argv = ["eval.py"] + bioreason_args(a)
    log("eval.py " + " ".join(argv[1:]))
    sys.argv = argv
    t0 = time.time()
    runpy.run_path(str(BR_REPO / "eval.py"), run_name="__main__")
    log(f"eval.py finished in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------- shared data

ASPECTS = ("MF", "BP", "CC")
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]


def _np():
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    return np, pd, sp


def ontology(unweighted: bool = False):
    from cafa6.knn_sharp import Ontology
    o = Ontology()
    if unweighted:
        o.ia = o.ia * 0 + 1.0
    return o


def save_pred(name: str, P, ids, onto, meta: dict):
    """CSR scores over the ontology columns that occur, compressed."""
    np, _, sp = _np()
    P = P.tocsr().astype(np.float32)
    P.eliminate_zeros()
    used = np.unique(P.indices)
    Q = P[:, used].tocsr()
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS / f"{name}.npz", ids=np.asarray(ids, dtype=str), data=Q.data, indices=Q.indices,
                        indptr=Q.indptr, shape=np.array(Q.shape), terms=np.array(onto.terms)[used].astype(str),
                        meta=json.dumps(meta))
    log(f"wrote preds/{name}.npz: {len(ids)} proteins, {Q.nnz} scores, {Q.nnz / max(len(ids), 1):.1f} per protein")


def load_npz_pred(path: Path, onto):
    """(P over ontology columns, ids) from any {ids, data, indices, indptr, shape, terms} file."""
    np, _, sp = _np()
    z = np.load(path, allow_pickle=False)
    Q = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    cols = np.array([onto.canon(t)[0] if onto.canon(t) else -1 for t in z["terms"]])
    keep = cols >= 0
    Q = Q[:, np.where(keep)[0]].tocoo()
    P = sp.csr_matrix((Q.data, (Q.row, cols[keep][Q.col])), shape=(Q.shape[0], len(onto.terms)))
    return P, z["ids"].astype(str)


def sets_matrix(sets: dict, onto, score: float = 1.0):
    """{protein: iterable of GO ids} -> CSR with `score` on every listed term (not propagated)."""
    np, _, sp = _np()
    ids = np.array(list(sets))
    M = onto.label_matrix([sets[p] for p in ids], close=False)
    M.data[:] = score
    return M, ids


# ----------------------------------------------------------------------------------------- baselines

def cmd_baselines(a):
    """Naive, BLAST-KNN (paper section 4.4.2) and DIAMONDScore from GO-GPT training proteins only."""
    np, pd, sp = _np()
    from cafa6.knn_sharp import KNN, SRC, split_ids
    from cafa6.temporal_val import GO, full_labels
    onto = ontology()
    df = pd.read_parquet(SRC / "train.parquet", columns=["id", "split"])
    db_ids = df.id.values[(df.split == "train").values]
    full = full_labels(GO(), "train")
    L = onto.label_matrix([full.get(p, set()) for p in db_ids])
    log(f"training references: {len(db_ids)} proteins, {L.nnz} propagated labels")
    kn = KNN(onto, "test", db_ids, L)
    te = kn.ids
    log(f"test queries: {len(te)}, with a DIAMOND hit: {int(kn.has_hit.sum())}")

    # naive: per aspect, fraction of training proteins annotated in that aspect that carry the term
    cnt = np.asarray(L.sum(0)).ravel()
    roots = [onto.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    n_asp = np.asarray(L[:, roots].sum(0)).ravel()
    freq = cnt / n_asp[onto.ns]
    keep = np.where(freq >= 0.005)[0]
    row = sp.csr_matrix((freq[keep].astype(np.float32), (np.zeros(len(keep), int), keep)), shape=(1, len(onto.terms)))
    save_pred("naive", sp.vstack([row] * len(te)).tocsr(), te, onto,
              {"method": "naive", "def": "per-aspect term frequency among training proteins annotated in the aspect"})

    # BLAST-KNN: best hit by percent identity (ties: higher bitscore), all its terms, score = identity / 100
    order = np.lexsort((-kn.bits, -kn.pid, kn.q))
    q, s, pid = kn.q[order], kn.s[order], kn.pid[order]
    first = np.r_[True, q[1:] != q[:-1]]
    q, s, pid = q[first], s[first], pid[first]
    W = sp.csr_matrix((pid / 100.0, (q, s)), shape=(len(te), len(db_ids)))
    save_pred("blast_knn", (W @ L).tocsr(), te, onto,
              {"method": "BLAST-KNN", "def": "paper 4.4.2; best hit by pident among DIAMOND hits e<=1e-3 to "
                                             "training proteins; all propagated terms of the hit, score = pident/100"})

    # DIAMONDScore (DeepGOPlus): bitscore-weighted label mean over all hits (top 50 by bitscore)
    P, _ = kn.prior(np.inf, 50, 0, "all")
    save_pred("diamondscore", P, te, onto,
              {"method": "DIAMONDScore", "def": "sum_j bits_j y_jt / sum_j bits_j over DIAMOND hits (top 50, e<=1e-3)"})

    # sensitivity: BLAST-KNN with GO-GPT training + validation proteins as references (HiGO-R2's kNN uses both)
    h = pd.read_parquet(ROOT / "data/proc_gogpt_tv2/knn_trainval/diamond_hits_test_trainval.parquet")
    full_v = full_labels(GO(), "val")
    ref = {**{p: full[p] for p in db_ids if p in full}, **full_v}
    h = h[h.q.isin(set(te)) & h.s.isin(set(ref))].sort_values(["q", "pident", "bits"], ascending=[True, False, False])
    best = h.drop_duplicates("q")
    qi = {p: i for i, p in enumerate(te)}
    rows = [qi[q] for q in best.q]
    Lb = onto.label_matrix([ref[s] for s in best.s])
    Lb = sp.diags(best.pident.values / 100.0) @ Lb
    P = sp.csr_matrix(Lb.tocoo(), shape=Lb.shape)
    P = sp.csr_matrix((P.tocoo().data, (np.array(rows)[P.tocoo().row], P.tocoo().col)), shape=(len(te), len(onto.terms)))
    log(f"train+val references: {len(ref)}; test queries with a hit {best.q.nunique()}")
    save_pred("blast_knn_trainval", P, te, onto,
              {"method": "BLAST-KNN (train+val references)", "def": "as blast_knn, references = GO-GPT train + val"})


# ----------------------------------------------------------------------------------------- BioReason-Pro outputs

def parse_bioreason(evals_path: Path, final_answer_only: bool):
    """{protein: GO ids} as evals/cafa_evals.py extracts them (regex GO:\\d{7} over the whole response, or only
    after </think>), plus per-protein generation stats."""
    import re
    rx = re.compile(r"GO:\d{7}")
    sets, stats = {}, []
    for f in sorted(Path(evals_path).glob("*_k00.json")):
        r = json.loads(f.read_text())
        txt = r.get("generated_response", "") or ""
        has_close = "</think>" in txt
        part = txt.split("</think>")[-1] if (final_answer_only and has_close) else txt
        sets[r["protein_id"]] = set(rx.findall(part))
        stats.append({"protein_id": r["protein_id"], "chars": len(txt), "has_think_close": has_close,
                      "n_go": len(set(rx.findall(txt))), "n_go_final": len(set(rx.findall(txt.split("</think>")[-1])))
                      if has_close else 0, "is_empty": not txt.strip()})
    return sets, stats


def cmd_collect_bioreason(a):
    """JSON outputs of eval.py -> preds/{tag}.npz (whole response) and {tag}_final.npz (after </think>), plus
    a compressed copy of the responses."""
    import gzip
    np, pd, sp = _np()
    onto = ontology()
    ev = Path(a.evals_path)
    for final, suffix in ((False, ""), (True, "_final")):
        sets, stats = parse_bioreason(ev, final)
        P, ids = sets_matrix(sets, onto)
        save_pred(a.tag + suffix, P, ids, onto,
                  {"method": "BioReason-Pro RL", "source": str(ev), "extraction": "after </think>" if final
                   else "whole response (evals/cafa_evals.py --final_answer_only False)", "score": 1.0})
    st = pd.DataFrame(stats)
    st.to_csv(OUT / f"{a.tag}_gen_stats.tsv", sep="\t", index=False)
    log(f"{len(st)} responses; empty {int(st.is_empty.sum())}; without </think> {int((~st.has_think_close).sum())}; "
        f"mean GO ids {st.n_go.mean():.1f} (final block {st.n_go_final.mean():.1f}); mean chars {st.chars.mean():.0f}")
    with gzip.open(PREDS / f"{a.tag}_responses.jsonl.gz", "wt") as f:
        for p in sorted(ev.glob("*_k00.json")):
            r = json.loads(p.read_text())
            f.write(json.dumps({k: r.get(k) for k in ("protein_id", "go_aspect", "generated_response")}) + "\n")
    log(f"wrote preds/{a.tag}_responses.jsonl.gz")


def cmd_import_tsv(a):
    """External predictor output (protein, GO id, score; tab-separated, no header unless --header) ->
    preds/{name}.npz. Scores are kept as the tool wrote them; terms outside our ontology are dropped and
    alternative ids are mapped by Ontology.canon; duplicate (protein, term) pairs keep the maximum."""
    np, pd, sp = _np()
    onto = ontology()
    d = pd.read_csv(a.tsv, sep="\t", header=0 if a.header else None, usecols=list(a.cols), dtype=str,
                    comment=None)
    d.columns = ["id", "term", "score"]
    d["score"] = d.score.astype(np.float32)
    ids = np.array(sorted(d.id.unique()))
    pos = {p: i for i, p in enumerate(ids)}
    canon = {t: onto.canon(t) for t in d.term.unique()}
    miss = sum(1 for t, c in canon.items() if not c)
    r, c, v = [], [], []
    for p, t, s in zip(d.id, d.term, d.score):
        for j in canon[t]:
            r.append(pos[p]); c.append(j); v.append(s)
    D = pd.DataFrame({"r": r, "c": c, "v": v}).groupby(["r", "c"]).v.max().reset_index()
    M = sp.csr_matrix((D.v.values.astype(np.float32), (D.r.values, D.c.values)), shape=(len(ids), len(onto.terms)))
    log(f"{a.tsv}: {len(d)} rows, {len(ids)} proteins, {len(canon)} distinct terms, {miss} not in our ontology")
    meta = json.loads(a.meta) if a.meta else {}
    meta.update({"source_file": str(a.tsv), "terms_not_in_ontology": miss})
    save_pred(a.name, M, ids, onto, meta)


def cmd_catalogue(a):
    """BioReason-Pro SFT generations released by the authors (wanglab/protein_catalogue, staged in
    /dev/shm/cmp_paper_catalogue) for the test proteins they cover -> preds/bioreason_sft_catalogue.npz and
    the protein list catalogue_ids.txt (a supplementary subset; the catalogue is not a random sample)."""
    import glob
    import re
    np, pd, sp = _np()
    from cafa6.knn_sharp import split_ids
    onto = ontology()
    te = set(split_ids("test"))
    rx = re.compile(r"GO:\d{7}")
    sets = {}
    for f in sorted(glob.glob("/dev/shm/cmp_paper_catalogue/data/*.parquet")):
        d = pd.read_parquet(f, columns=["protein_id", "generation"])
        d = d[d.protein_id.isin(te)]
        for p, g in zip(d.protein_id, d.generation):
            sets[p] = set(rx.findall(g or ""))
    P, ids = sets_matrix(sets, onto)
    save_pred("bioreason_sft_catalogue", P, ids, onto,
              {"method": "BioReason-Pro SFT", "source": "wanglab/protein_catalogue (authors' generations)",
               "extraction": "regex GO:\\d{7} over the whole generation", "score": 1.0})
    (OUT / "catalogue_ids.txt").write_text("\n".join(sorted(ids)) + "\n")
    log(f"catalogue covers {len(ids)} of {len(te)} test proteins")


# ----------------------------------------------------------------------------------------- scoring

def curves_full(sc, P, ids):
    """knn_sharp.Scorer.curves plus the raw IA sums (predicted, true positive, ground truth) per protein and
    threshold, which S-min needs."""
    np, _, sp = _np()
    from cafa6.knn_sharp import NS, NS_SHORT, TAU, to_submission
    ids = np.asarray(ids)
    m = np.array([p in sc.pos for p in ids])
    P = P.tocsr()[np.where(m)[0]]
    gi = np.array([sc.pos[p] for p in ids[m]], dtype=np.int64)
    P = sc.o.propagate_max(to_submission(P))
    rows = gi[np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))]
    cols, vals = P.indices, P.data
    w, nsc = sc.o.ia[cols], sc.o.ns[cols]
    ing = np.isin(rows * len(sc.o.terms) + cols, sc.gkey)
    kk = np.searchsorted(TAU, vals, side="right")
    out = {}
    for k in range(3):
        sel = (nsc == k) & (w > 0) & sc.in_ns[k, rows]
        r, kq, ww, g = rows[sel], kk[sel], w[sel], ing[sel]
        C = np.bincount(r * 100 + kq, weights=ww, minlength=sc.n * 100).reshape(sc.n, 100)
        I = np.bincount(r * 100 + kq, weights=ww * g, minlength=sc.n * 100).reshape(sc.n, 100)
        predw = np.cumsum(C[:, ::-1], axis=1)[:, ::-1][:, 1:]
        interw = np.cumsum(I[:, ::-1], axis=1)[:, ::-1][:, 1:]
        idx = np.where(sc.in_ns[k])[0]
        predw, interw, gw = predw[idx], interw[idx], sc.gtw[k][idx][:, None]
        has = predw > 0
        out[NS_SHORT[NS[k]]] = {
            "idx": idx, "has": has,
            "prec": np.divide(interw, predw, out=np.zeros_like(predw), where=has).astype(np.float32),
            "rec": np.divide(interw, gw, out=np.zeros_like(interw), where=gw > 0).astype(np.float32),
            "ru": (gw - interw).astype(np.float32), "mi": (predw - interw).astype(np.float32)}
    return out


def metrics(cur, wv):
    """Per aspect: Fmax, its precision / recall / threshold, S-min (cafaeval: remaining uncertainty and
    misinformation averaged over all proteins with ground truth in the aspect), coverage; plus means."""
    np, _, _ = _np()
    from cafa6.knn_sharp import TAU
    res = {}
    for a, c in cur.items():
        ww = wv[c["idx"]]
        ne = ww.sum()
        if ne <= 0:
            continue
        nh = ww @ c["has"]
        P = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
        R = (ww @ c["rec"]) / ne
        F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
        F[nh <= 0] = 0
        i = int(F.argmax())
        S = np.sqrt(((ww @ c["ru"]) / ne) ** 2 + ((ww @ c["mi"]) / ne) ** 2)
        res[a] = {"F": float(F[i]), "P": float(P[i]), "R": float(R[i]), "tau": float(TAU[i]),
                  "Smin": float(S.min()), "cov": float(nh[0] / ne), "n": int((ww > 0).sum())}
    return res


def fmax_mean(cur, wv):
    """Mean over aspects of Fmax only (fast path for the bootstrap)."""
    np, _, _ = _np()
    vals = []
    for c in cur.values():
        ww = wv[c["idx"]]
        ne = ww.sum()
        nh = ww @ c["has"]
        P = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
        R = (ww @ c["rec"]) / ne
        F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
        F[nh <= 0] = 0
        vals.append(F.max())
    return float(np.mean(vals))


def per_protein_f1(cur, n):
    """Weighted F1 of each protein at its aspect's global Fmax threshold, averaged over the aspects in which
    the protein has ground truth (paper eq. 4.29, used for its Fig. 3F / Table S12)."""
    np, _, _ = _np()
    s, k = np.zeros(n), np.zeros(n)
    ones = np.ones(n)
    for c in cur.values():
        ww = ones[c["idx"]]
        nh = ww @ c["has"]
        P = np.divide(ww @ c["prec"], nh, out=np.zeros(99), where=nh > 0)
        R = (ww @ c["rec"]) / ww.sum()
        F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
        i = int(F.argmax())
        p, r = c["prec"][:, i], c["rec"][:, i]
        s[c["idx"]] += np.divide(2 * p * r, p + r, out=np.zeros(len(p)), where=(p + r) > 0)
        k[c["idx"]] += 1
    return s / np.maximum(k, 1)


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


METHODS = [
    # key, label, source of predictions
    ("higo_r2", "HiGO-R2 (ours)", "frozen look-3 file runs/stacker_tv/l3_c0/pred_test_twin.npz"),
    ("bioreason_rl", "BioReason-Pro RL (greedy)", "released wanglab/bioreason-pro-rl, authors' eval.py, our run"),
    ("bioreason_rl_final", "BioReason-Pro RL (greedy, final answer only)", "same run, GO ids after </think>"),
    ("bioreason_rl_askall", "BioReason-Pro RL (greedy, label-independent prompt)",
     "same checkpoint, eval.py with ask_all_go_aspects=force_uniprot_summary=True (repo default, predict.py)"),
    ("gogpt_sample10", "GO-GPT ten-sample frequency", "released checkpoint, our generation (T 0.7, top-k 20)"),
    ("gogpt_greedy", "GO-GPT greedy", "released checkpoint, our generation"),
    ("gogpt_beam", "GO-GPT beam 5", "released checkpoint, our generation (predictor default)"),
    ("gogpt_authors", "GO-GPT (authors' predictions)", "go_pred column of wanglab/bioreason-pro-test-data"),
    ("diamondscore", "DIAMONDScore", "our DIAMOND hits to GO-GPT training proteins"),
    ("blast_knn", "BLAST-KNN", "our DIAMOND hits to GO-GPT training proteins (paper 4.4.2 rule)"),
    ("blast_knn_trainval", "BLAST-KNN (train+val references)",
     "same rule, GO-GPT training + validation proteins as references (sensitivity row)"),
    ("naive", "Naive", "GO-GPT training label frequencies"),
    # external predictors, official weights, our runs (runs/compare_paper/extra_models.md)
    ("deepgoplus", "DeepGOPlus (release 1.0.10)",
     "official data-1.0.10 package (SwissProt 2022_04, GO 2022-10-07), deepgoplus/main.py, our run"),
    ("proteinfer", "ProteInfer GO (one model)",
     "official GO random-split CNN 13703706 (SwissProt 2019_01), our run"),
    ("proteinfer_e5", "ProteInfer GO (5-model ensemble)",
     "official GO random-split CNNs 13703706/3742/3997/4131/5631 averaged (proteinfer.py maximum), our run"),
    ("deepgose", "DeepGO-SE", "official deepgo2 data package (SwissProt 2021_04, GO 2021-11-16), predict.py, our run"),
    ("deepfri_cnn", "DeepFRI sequence-only CNN",
     "official trained_models (PDB + SWISS-MODEL, SIFTS 2019-06-18), predict.py --fasta_fn, our run"),
    ("sprof_go", "SPROF-GO", "official repo models + ProtT5-XL-U50 (GO data of 2020-01), predict.py, our run"),
    ("interlabelgo_plus", "InterLabelGO+ (released package; test labels in its training data)",
     "Zenodo 15508797 package (training labels contain 96% of test GT pairs), predict.py --seqid, our run"),
]


def load_method(key, onto):
    np, pd, sp = _np()
    if key == "higo_r2":
        return load_npz_pred(ROOT / "runs/stacker_tv/l3_c0/pred_test_twin.npz", onto)
    if key in ("gogpt_beam", "gogpt_greedy"):
        sub = "" if key == "gogpt_beam" else "greedy"
        sets = {}
        for f in sorted((ROOT / "runs/gogpt_cafa5" / sub).glob("preds_shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                sets[r["protein_id"]] = {t for k in ASPECTS for t in r.get(k, [])}
        return sets_matrix(sets, onto)
    if key == "gogpt_authors":
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from gogpt_eval import released_preds
        return sets_matrix({p: {t for k in ASPECTS for t in s[k]} for p, s in released_preds().items()}, onto)
    if key == "gogpt_sample10":
        d = pd.read_csv(ROOT / "runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz", sep="\t", header=None,
                        names=["id", "term", "score"])
        ids = np.array(sorted(d.id.unique()))
        pos = {p: i for i, p in enumerate(ids)}
        r, c, v = [], [], []
        for p, t, s in zip(d.id, d.term, d.score):
            for j in onto.canon(t):
                r.append(pos[p]); c.append(j); v.append(s)
        M = sp.csr_matrix((np.array(v, np.float32), (r, c)), shape=(len(ids), len(onto.terms)))
        M.sum_duplicates()
        return M, ids
    f = PREDS / f"{key}.npz"
    return load_npz_pred(f, onto) if f.exists() else (None, None)


def cmd_score(a):
    np, pd, sp = _np()
    from cafa6.knn_sharp import REF, Scorer, gt_frame
    from cafa6.temporal_val import ident_bins
    onto, onto_u = ontology(), ontology(unweighted=True)
    gt = gt_frame("test")
    sc, scu = Scorer(onto, gt), Scorer(onto_u, gt)
    n = sc.n
    common = set(open(REF / "common_proteins.txt").read().split())
    s6 = np.array([p not in common for p in sc.ids], float)
    k = np.load(ROOT / "data/proc_gogpt_tv2/knn/test_knn.npz")
    ident = pd.Series(k["max_pident"], index=k["ids"]).reindex(sc.ids).fillna(0).values
    bins = ident_bins(ident).values
    log(f"GT proteins {n}; S6 {int(s6.sum())}; bins " + ", ".join(f"{b} {int((bins == b).sum())}" for b in BINS))

    curs, cursu, info = {}, {}, {}
    methods = METHODS + ([("bioreason_sft_catalogue", "BioReason-Pro SFT (authors' catalogue)",
                           "wanglab/protein_catalogue generations")] if a.subset_file else [])
    for key, label, src in methods:
        P, ids = load_method(key, onto)
        if P is None:
            log(f"skip {key}: no predictions")
            continue
        cov = len(set(ids) & set(sc.ids))
        curs[key], cursu[key] = curves_full(sc, P, ids), curves_full(scu, P, ids)
        info[key] = {"label": label, "source": src, "n_pred_proteins": cov}
        m = metrics(curs[key], np.ones(n))
        log(f"{key:20s} proteins with predictions {cov:5d}  S3 Fmax_w " +
            " ".join(f"{x} {m[x]['F']:.4f}" for x in ASPECTS) + f"  mean {np.mean([m[x]['F'] for x in ASPECTS]):.4f}")

    subsets = {"S3_all": np.ones(n), "S6_noknowledge": s6}
    if a.subset_file:
        keep = set(open(a.subset_file).read().split())
        subsets = {f"{k}_sub": v * np.array([p in keep for p in sc.ids]) for k, v in subsets.items()}
    rows = []
    for sname, wv in subsets.items():
        for key in curs:
            mw, mu = metrics(curs[key], wv), metrics(cursu[key], wv)
            r = {"subset": sname, "method": key, "label": info[key]["label"], "source": info[key]["source"]}
            for x in ASPECTS:
                r.update({f"Fw_{x}": mw[x]["F"], f"F_{x}": mu[x]["F"], f"Smin_{x}": mw[x]["Smin"],
                          f"tau_w_{x}": mw[x]["tau"], f"cov_{x}": mw[x]["cov"], f"n_{x}": mw[x]["n"]})
            r["Fw_mean"] = np.mean([mw[x]["F"] for x in ASPECTS])
            r["F_mean"] = np.mean([mu[x]["F"] for x in ASPECTS])
            r["Smin_mean"] = np.mean([mw[x]["Smin"] for x in ASPECTS])
            rows.append(r)

    # paired protein bootstrap: the same resamples for every method
    rng = np.random.default_rng(0)
    ref = "higo_r2"
    for sname, wv in subsets.items():
        base = wv / wv.sum()
        draws = {key: [] for key in curs}
        for _ in range(a.reps):
            c = rng.multinomial(int(wv.sum()), base).astype(float)
            for key in curs:
                draws[key].append(fmax_mean(curs[key], c))
        d0 = np.array(draws[ref])
        pv, stat = {}, {}
        for key in curs:
            if key == ref:
                continue
            d = d0 - np.array(draws[key])
            pv[key] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1.0 / a.reps)
            stat[key] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        # Holm family: one row per method; the extraction / prompt variants of the BioReason-Pro run are
        # reported with their raw bootstrap p only, as are the single-model ProteInfer variant and
        # InterLabelGO+, whose released training labels contain the test annotations
        ph = holm({k: v for k, v in pv.items()
                   if k not in ("bioreason_rl_final", "bioreason_rl_askall", "blast_knn_trainval",
                                "interlabelgo_plus", "proteinfer")})
        for r in rows:
            if r["subset"] == sname and r["method"] in stat:
                hr = [x for x in rows if x["subset"] == sname and x["method"] == ref][0]
                r["diff_higo_minus_method"] = hr["Fw_mean"] - r["Fw_mean"]
                r["ci_lo"], r["ci_hi"] = stat[r["method"]]
                r["p_boot"] = pv[r["method"]]
                r["p_holm"] = ph.get(r["method"], np.nan)
        log(f"bootstrap {sname}: {a.reps} reps done")
    summ = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_sub" if a.subset_file else ""
    summ.to_csv(OUT / f"summary{tag}.tsv", sep="\t", index=False, float_format="%.4f")

    # identity bins: Fmax_w within each bin (threshold chosen inside the bin, as in earlier per-bin tables),
    # and the paper's per-protein weighted F1 at the global threshold
    brow, pp = [], {}
    for key in curs:
        f1 = per_protein_f1(curs[key], n)
        pp[key] = f1
        for b in BINS + ["all"]:
            wv = np.ones(n) if b == "all" else (bins == b).astype(float)
            if a.subset_file:
                wv = wv * subsets["S3_all_sub"]
            if wv.sum() == 0:
                continue
            m = metrics(curs[key], wv)
            brow.append({"method": key, "bin": b, "n": int(wv.sum()),
                         **{f"Fw_{x}": m[x]["F"] if x in m else np.nan for x in ASPECTS},
                         "Fw_mean": np.nanmean([m[x]["F"] for x in ASPECTS if x in m]),
                         "perprot_F1w_mean": float(f1[wv > 0].mean())})
    pd.DataFrame(brow).to_csv(OUT / f"per_bin{tag}.tsv", sep="\t", index=False, float_format="%.4f")
    rho = {key: float(pd.Series(pp[key]).rank().corr(pd.Series(ident).rank())) for key in pp}
    (OUT / f"spearman_identity{tag}.json").write_text(json.dumps(rho, indent=2))
    show = summ.pivot_table(index="method", columns="subset", values="Fw_mean").sort_values(list(subsets)[0],
                                                                                         ascending=False)
    log("weighted Fmax mean:\n" + show.round(4).to_string())
    log("Spearman(per-protein F1_w, best-hit identity): " + json.dumps({k: round(v, 3) for k, v in rho.items()}))


# ----------------------------------------------------------------------------------------- report tables

# paper numbers (runs/compare_paper/paper_table.md): (subset, weighted MF/BP/CC, unweighted MF/BP/CC, where)
PAPER = {
    "GO-GPT greedy": ("S3_all", (0.701, 0.539, 0.713), (0.766, 0.587, 0.801), "Table S3"),
    "GO-GPT probability k=10": ("S3_all", (0.706, 0.547, 0.730), (0.766, 0.600, 0.816), "Table S5"),
    "GO-GPT pass@1 (S6)": ("S6_noknowledge", (0.729, 0.545, 0.718), (0.792, 0.593, 0.806), "Tables S6/S8"),
    "BioReason-Pro RL pass@1": ("S6_noknowledge", (0.718, 0.544, 0.727), (0.783, 0.587, 0.812), "Table S8"),
    "BioReason-Pro SFT pass@1": ("S6_noknowledge", (0.717, 0.516, 0.689), (0.773, 0.557, 0.781), "Table S6"),
    "InterLabelGO+": ("S3_all", (0.692, 0.525, 0.696), (0.727, 0.566, 0.790), "Tables S3-S5"),
    "ProtBoost": ("S3_all", (0.658, 0.490, 0.669), (0.702, 0.533, 0.777), "Tables S3-S5"),
}
# our method that scores the same (or the corresponding) predictions under our scorer
PAPER_MATCH = [
    ("GO-GPT greedy", "gogpt_authors", "same predictions (authors' go_pred file)"),
    ("GO-GPT pass@1 (S6)", "gogpt_authors", "same predictions (authors' go_pred file)"),
    ("GO-GPT greedy", "gogpt_greedy", "released checkpoint, our greedy run"),
    ("GO-GPT probability k=10", "gogpt_sample10", "released checkpoint, our ten samples"),
    ("BioReason-Pro RL pass@1", "bioreason_rl", "released RL checkpoint, our run of eval.py"),
    ("BioReason-Pro RL pass@1", "bioreason_rl_final", "same run, final-answer extraction"),
]


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def cmd_tables(a):
    """Markdown tables from summary.tsv / per_bin.tsv for report.md (written to report_tables.md)."""
    np, pd, _ = _np()
    s = pd.read_csv(OUT / "summary.tsv", sep="\t")
    b = pd.read_csv(OUT / "per_bin.tsv", sep="\t")
    rho = json.loads((OUT / "spearman_identity.json").read_text())
    lab = dict(zip(s.method, s.label))
    src = dict(zip(s.method, s.source))
    s3, s6 = s[s.subset == "S3_all"].set_index("method"), s[s.subset == "S6_noknowledge"].set_index("method")
    order = s3.sort_values("Fw_mean", ascending=False).index.tolist()
    parts = []

    def ci(sub, m):
        r = sub.loc[m]
        if m == "higo_r2" or pd.isna(r.get("diff_higo_minus_method", np.nan)):
            return "—"
        ph = r["p_holm"]
        if pd.isna(ph):
            pr = r["p_boot"]
            pt = f"p_boot {pr:.3f} (variant, not in Holm family)"
        else:
            pt = f"p_Holm {ph:.3f}"
        return f"{r.diff_higo_minus_method:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], {pt}"

    rows = []
    for m in order:
        r3, r6 = s3.loc[m], s6.loc[m]
        rows.append({"method": lab[m], "predictions": src[m], "S3 Fw MF": r3.Fw_MF, "S3 Fw BP": r3.Fw_BP,
                     "S3 Fw CC": r3.Fw_CC, "S3 Fw mean": r3.Fw_mean, "S6 Fw mean": r6.Fw_mean,
                     "S3 S-min mean": r3.Smin_mean, "HiGO-R2 minus method, S3 [95% CI]": ci(s3, m),
                     "HiGO-R2 minus method, S6 [95% CI]": ci(s6, m)})
    for k, (sub, w, u, where) in PAPER.items():
        if k in ("InterLabelGO+", "ProtBoost", "BioReason-Pro SFT pass@1"):
            rows.append({"method": k + " (paper-reported, different scorer)", "predictions": where,
                         "S3 Fw MF": w[0] if sub == "S3_all" else "—", "S3 Fw BP": w[1] if sub == "S3_all" else "—",
                         "S3 Fw CC": w[2] if sub == "S3_all" else "—",
                         "S3 Fw mean": float(np.mean(w)) if sub == "S3_all" else "—",
                         "S6 Fw mean": float(np.mean(w)) if sub != "S3_all" else "—", "S3 S-min mean": "—",
                         "HiGO-R2 minus method, S3 [95% CI]": "not computable (no predictions)",
                         "HiGO-R2 minus method, S6 [95% CI]": "—"})
    parts.append("### Main table (IA-weighted Fmax; S-min in IA units)\n\n" + md_table(pd.DataFrame(rows)))

    rows = []
    for m in order:
        r3, r6 = s3.loc[m], s6.loc[m]
        rows.append({"method": lab[m], "S3 F MF": r3.F_MF, "S3 F BP": r3.F_BP, "S3 F CC": r3.F_CC,
                     "S3 F mean": r3.F_mean, "S6 Fw MF": r6.Fw_MF, "S6 Fw BP": r6.Fw_BP, "S6 Fw CC": r6.Fw_CC,
                     "S6 F mean": r6.F_mean, "S3 S-min MF": r3.Smin_MF, "S3 S-min BP": r3.Smin_BP,
                     "S3 S-min CC": r3.Smin_CC})
    parts.append("### Unweighted Fmax, S6 per aspect, S-min per aspect\n\n" + md_table(pd.DataFrame(rows)))

    rows = []
    for pk, m, how in PAPER_MATCH:
        if m not in s3.index:
            continue
        sub, w, u, where = PAPER[pk]
        r = (s3 if sub == "S3_all" else s6).loc[m]
        ow, ou = [r[f"Fw_{x}"] for x in ASPECTS], [r[f"F_{x}"] for x in ASPECTS]
        rows.append({"paper row": f"{pk} ({where})", "subset": sub, "ours": lab[m], "relation": how,
                     "paper Fw mean": float(np.mean(w)), "ours Fw mean": float(np.mean(ow)),
                     "offset Fw (ours - paper)": float(np.mean(ow) - np.mean(w)),
                     "offset Fw MF/BP/CC": "/".join(f"{o - p:+.3f}" for o, p in zip(ow, w)),
                     "paper F mean": float(np.mean(u)), "ours F mean": float(np.mean(ou)),
                     "offset F (ours - paper)": float(np.mean(ou) - np.mean(u))})
    parts.append("### Our scorer vs the paper's numbers\n\n" + md_table(pd.DataFrame(rows)))

    piv = b.pivot_table(index="method", columns="bin", values="Fw_mean")
    n = b.groupby("bin").n.first()
    cols = [c for c in BINS + ["all"] if c in piv.columns]
    piv = piv.loc[order, cols]
    piv.index = [lab[m] for m in piv.index]
    t = piv.reset_index().rename(columns={"index": "method"})
    t.columns = ["method"] + [f"{c} (n={int(n[c])})" for c in cols]
    parts.append("### Weighted Fmax mean per identity bin (threshold chosen within the bin)\n\n" + md_table(t))
    piv = b.pivot_table(index="method", columns="bin", values="perprot_F1w_mean").loc[order, cols]
    piv.index = [lab[m] for m in piv.index]
    t = piv.reset_index().rename(columns={"index": "method"})
    t.columns = ["method"] + [f"{c}" for c in cols]
    t["Spearman rho vs identity"] = [rho[m] for m in order]
    parts.append("### Mean per-protein weighted F1 at the global Fmax threshold per bin (paper eq. 4.29, "
                 "Fig. 3F) and its Spearman correlation with best-hit identity (paper Table S12)\n\n" + md_table(t))
    sub_f = OUT / "summary_sub.tsv"
    if sub_f.exists():
        ss = pd.read_csv(sub_f, sep="\t")
        rows = []
        for m in ss[ss.subset == "S3_all_sub"].sort_values("Fw_mean", ascending=False).method:
            r3 = ss[(ss.subset == "S3_all_sub") & (ss.method == m)].iloc[0]
            r6 = ss[(ss.subset == "S6_noknowledge_sub") & (ss.method == m)].iloc[0]
            rows.append({"method": r3.label, "Fw MF": r3.Fw_MF, "Fw BP": r3.Fw_BP, "Fw CC": r3.Fw_CC,
                         "Fw mean": r3.Fw_mean, "S6 Fw mean": r6.Fw_mean,
                         "HiGO-R2 minus method [95% CI]": "—" if m == "higo_r2" else
                         f"{r3.diff_higo_minus_method:+.4f} [{r3.ci_lo:+.4f}, {r3.ci_hi:+.4f}]",
                         "n MF/BP/CC": f"{int(r3.n_MF)}/{int(r3.n_BP)}/{int(r3.n_CC)}"})
        parts.append("### Supplementary: the 1,213 test proteins in the authors' SFT catalogue\n\n"
                     + md_table(pd.DataFrame(rows)))
    (OUT / "report_tables.md").write_text("\n\n".join(parts) + "\n")
    log(f"wrote {OUT / 'report_tables.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("bioreason-gen")
    g.add_argument("--gpu", default="0")
    g.add_argument("--evals_path", default=str(OUT / "bioreason_rl" / "evals"))
    g.add_argument("--num_chunks", type=int, default=1)
    g.add_argument("--chunk_id", type=int, default=0)
    g.add_argument("--max_samples", type=int, default=-1)
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--max_new_tokens", type=int, default=3000)
    g.add_argument("--gpu_mem", type=float, default=0.5)
    g.add_argument("--ask_all", action="store_true",
                   help="ask_all_go_aspects=force_uniprot_summary=True (label-independent prompt, predict.py)")
    sub.add_parser("baselines")
    sub.add_parser("catalogue")
    sub.add_parser("tables")
    t = sub.add_parser("import-tsv")
    t.add_argument("--tsv", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--cols", type=int, nargs=3, default=[0, 1, 2], help="columns of protein, GO id, score")
    t.add_argument("--header", action="store_true")
    t.add_argument("--meta", default="", help="JSON dict stored in the npz")
    c = sub.add_parser("collect-bioreason")
    c.add_argument("--evals_path", default=str(OUT / "bioreason_rl" / "evals"))
    c.add_argument("--tag", default="bioreason_rl")
    s = sub.add_parser("score")
    s.add_argument("--reps", type=int, default=1000)
    s.add_argument("--subset_file", default="", help="score only these proteins (one id per line)")
    a = ap.parse_args()
    {"bioreason-gen": cmd_bioreason_gen, "baselines": cmd_baselines, "collect-bioreason": cmd_collect_bioreason,
     "catalogue": cmd_catalogue, "score": cmd_score, "tables": cmd_tables, "import-tsv": cmd_import_tsv}[a.cmd](a)


if __name__ == "__main__":
    main()
