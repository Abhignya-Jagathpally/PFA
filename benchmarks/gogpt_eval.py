"""Benchmark the released GO-GPT (HF wanglab/gogpt) on its own temporal-holdout test set, or zero-shot on any FASTA.

Generation reproduces GOGPTPredictor.predict (ESM2-3B layer 30 in fp32, sequences truncated to 1024 tokens,
beam 5, length_penalty 0.3, max 300 new tokens per aspect, organism name -> organism id, unknown -> 0) but
  * batches proteins (padding to the longest sequence in the batch; padded residues are masked), and
  * computes the protein stream of the decoder once per batch instead of at every decoding step. Protein residues
    never attend to GO tokens (PrefixCausalAttention masks them out), so the per-layer protein keys/values are
    independent of the generated tokens and caching them is exact.
Predictions are unscored GO sets; they are written with score 1.0 (the paper's "greedy / single run" setting).

--decode beam (default) is the released predictor's setting; --decode greedy is argmax decoding (model.generate
with top_k=1), the paper's literal "greedy decoding".

--decode sample is the paper's "probability" mode (section 4.2.3): --n_samples sequences per protein and aspect drawn
as in GOGPT.generate (temperature 0.7, top-k 20, up to 300 new tokens), with a key/value cache for the GO tokens as
well (with top_k=1 it reproduces --decode greedy token for token). Samples go to samples_shard*.jsonl.gz; scoring
gives each term the fraction of samples that contain it ("_freq"), plus a variant where every sample is first
propagated to its ancestors ("_freq_prop"), and evaluates both with the bench_gogpt evaluator and subsets
(S3_all, S6_noknowledge, identity bins), next to the beam and greedy runs from runs/gogpt_cafa5.

Scoring uses the CAFA-evaluator CLI (prop=max, norm=cafa). Protocols:
  S3_all          go-basic 2023-01-01 + CAFA-5 IA.txt from the BioReason-Pro repo, all 8,630 test proteins
                  (paper Table S3: MF/BP/CC n = 2080/5819/3440)
  S6_noknowledge  same, without the 471 limited-knowledge proteins in data/common_proteins.txt (excluded by the
                  authors' evals/cafa_evals.py; paper Tables S6-S9: n = 1878/5623/3293)
  cafa6_ia        CAFA-6 go-basic.obo + CAFA-6 IA.tsv, all proteins (for comparability with our CAFA-6 numbers)

Usage
  paper test set:  python benchmarks/gogpt_eval.py all
  any FASTA:       python benchmarks/gogpt_eval.py all --fasta X.fasta --taxa taxa.tsv --gt gt.tsv --out runs/X
                   --taxa: either a taxon-id -> species table with an "ID<TAB>Species" header (taxon read from the
                   FASTA header, e.g. CAFA-6 testsuperset-taxon-list.tsv) or a header-less protein<TAB>taxon-id map
                   (species names then come from --taxon_names).
  stages:          gen | score | all
  ten samples:     python benchmarks/gogpt_eval.py all --decode sample --out runs/gogpt_sample10 --batch 128
                   --token_budget 66000
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/gogpt_tmp_backup/BioReason-Pro-main/gogpt/src"))
sys.path.insert(0, str(ROOT / "src"))
REF = ROOT / "runs" / "gogpt_cafa5" / "ref"
CAFAEVAL = Path(sys.executable).parent / "cafaeval"
ASPECTS = ("MF", "BP", "CC")
NS_SHORT = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
GO_RE = re.compile(r"GO:\d{7}")
PROTOCOLS = {
    "S3_all": (REF / "go-basic_gogpt.obo", REF / "IA.txt", False),
    "S6_noknowledge": (REF / "go-basic_gogpt.obo", REF / "IA.txt", True),
    "cafa6_ia": (ROOT / "data/raw/Train/go-basic.obo", ROOT / "data/raw/IA.tsv", False),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- model

def load_predictor():
    from gogpt import GOGPTPredictor
    pred = GOGPTPredictor.from_pretrained("wanglab/gogpt", verbose=False)
    install_protein_cache(pred.model)
    return pred


def _protein_stream(model, emb, mask):
    """Per-layer protein keys/values; identical to the protein half of every Block.forward."""
    ps = model.protein_projection(emb)
    B, T, C = ps.shape
    H = model.config.n_head
    attn_mask = mask.bool()[:, None, None, :]
    kv = []
    for blk in model.transformer.h:
        a = blk.attn
        x = blk.ln_1_protein(ps)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in a.protein_qkv(x).split(C, dim=2))
        kv.append((k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2).reshape(B, T, C)
        if a.use_gated_attention:
            o = o * torch.sigmoid(a.protein_gate_proj(x))
        ps = ps + a.protein_proj(o)
        ps = ps + blk.mlp_protein(blk.ln_2_protein(ps))
    return kv


def _go_stream(model, kv, go_tokens, protein_mask, go_mask, organism_id):
    """GO half of GOGPT.forward (targets=None) using cached protein keys/values."""
    tr, cfg = model.transformer, model.config
    b, t = go_tokens.shape
    C, H = cfg.n_embd, cfg.n_head
    dev = go_tokens.device
    x = tr.wte(go_tokens) + tr.wpe(torch.arange(t, device=dev))[None]
    if organism_id is not None:
        x = x + model.organism_embedding(organism_id)[:, None, :]
    if go_mask is None:
        go_mask = torch.ones(b, t, dtype=torch.bool, device=dev)
    gm = go_mask.bool()
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=dev))
    m = torch.cat([protein_mask.bool()[:, None, None, :].expand(b, 1, t, protein_mask.shape[1]),
                   (causal[None, None] & gm[:, None, None, :])], dim=-1)
    valid = gm.view(b, 1, t, 1).to(x.dtype)
    for blk, (pk, pv) in zip(tr.h, kv):
        a = blk.attn
        xn = blk.ln_1_go(x)
        q, k, v = (tt.view(b, t, H, C // H).transpose(1, 2) * valid for tt in a.go_qkv(xn).split(C, dim=2))
        o = F.scaled_dot_product_attention(q, torch.cat([pk, k], 2), torch.cat([pv, v], 2), attn_mask=m)
        o = (o * valid).transpose(1, 2).reshape(b, t, C)
        if a.use_gated_attention:
            o = o * torch.sigmoid(a.go_gate_proj(xn))
        o = o * gm[..., None].to(o.dtype)
        x = x + a.go_proj(o)
        x = x + blk.mlp_go(blk.ln_2_go(x))
    return model.lm_head(tr.ln_f_go(x)[:, [-1], :])


def install_protein_cache(model):
    """Replaces model.forward (used by generate_beam_search) with a protein-stream-cached equivalent.
    The cache must be reset (model._pcache.clear()) whenever the protein batch changes."""
    model._pcache = {}

    def forward(protein_tokens=None, go_tokens=None, targets=None, protein_mask=None, go_mask=None,
                organism_id=None, protein_embeddings=None):
        assert targets is None and protein_embeddings is not None
        key = tuple(protein_embeddings.shape)
        if model._pcache.get("key") != key:
            model._pcache.update(key=key, kv=_protein_stream(model, protein_embeddings, protein_mask))
        return _go_stream(model, model._pcache["kv"], go_tokens, protein_mask, go_mask, organism_id), None

    model.forward = forward


class _StopESM(Exception):
    pass


def esm_layer(model, toks, mask):
    """hidden_states[protein_layer_index] of the ESM2 forward, without running (or storing) the later layers."""
    li = model.config.protein_layer_index
    layers = model.esm.encoder.layer
    if li >= len(layers):
        return model.esm(input_ids=toks, attention_mask=mask, output_hidden_states=True).hidden_states[li]
    box = {}

    def grab(_, args, kwargs):
        box["h"] = args[0] if args else kwargs["hidden_states"]
        raise _StopESM

    h = layers[li].register_forward_pre_hook(grab, with_kwargs=True)
    try:
        model.esm(input_ids=toks, attention_mask=mask)
    except _StopESM:
        pass
    finally:
        h.remove()
    return box["h"]


@torch.no_grad()
def greedy(model, protein_mask, go, organism_id, emb, max_new_tokens):
    """Same output as GOGPT.generate(temperature=1, top_k=1) with one host sync per step instead of one per row."""
    cfg = model.config
    end_ids = torch.tensor([cfg.mf_end_token_id, cfg.bp_end_token_id, cfg.cc_end_token_id], device=go.device)
    finished = torch.zeros(go.size(0), dtype=torch.bool, device=go.device)
    go_mask = torch.ones_like(go, dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits, _ = model(protein_tokens=None, go_tokens=go, protein_mask=protein_mask, go_mask=go_mask,
                          organism_id=organism_id, protein_embeddings=emb)
        nxt = logits[:, -1, :].argmax(-1)
        nxt = torch.where(finished, torch.zeros_like(nxt), nxt)            # finished rows emit pad (id 0)
        go = torch.cat([go, nxt[:, None]], dim=1)
        go_mask = torch.cat([go_mask, (~finished)[:, None]], dim=1)
        finished = finished | torch.isin(nxt, end_ids)
        if finished.all():
            break
    return go


def _go_step(model, cache, mask, tp, tok, pos, organism_id):
    """One GO token per row through the GO stream. `cache` holds one preallocated (K, V) buffer per layer of shape
    (rows, heads, tp + max_new_tokens, head_dim) with the protein keys/values in the first tp slots; this step's GO
    key/value is written at slot tp + pos. `mask` (rows, 1, 1, slots) marks valid protein residues and GO slots.
    Exact incremental form of _go_stream for rows that are still decoding: such rows have no masked GO positions,
    so each new token attends to the valid protein residues, all earlier GO tokens and itself."""
    tr, cfg = model.transformer, model.config
    b = tok.size(0)
    C, H = cfg.n_embd, cfg.n_head
    x = tr.wte(tok)[:, None] + tr.wpe.weight[pos][None, None]
    if organism_id is not None:
        x = x + model.organism_embedding(organism_id)[:, None, :]
    L = tp + pos + 1
    m = mask[..., :L]
    for blk, (K, V) in zip(tr.h, cache):
        a = blk.attn
        xn = blk.ln_1_go(x)
        q, k, v = (tt.view(b, 1, H, C // H).transpose(1, 2) for tt in a.go_qkv(xn).split(C, dim=2))
        K[:, :, L - 1:L] = k
        V[:, :, L - 1:L] = v
        o = F.scaled_dot_product_attention(q.to(K.dtype), K[:, :, :L], V[:, :, :L], attn_mask=m)
        o = o.transpose(1, 2).reshape(b, 1, C)
        if a.use_gated_attention:
            o = o * torch.sigmoid(a.go_gate_proj(xn))
        x = x + a.go_proj(o)
        x = x + blk.mlp_go(blk.ln_2_go(x))
    return model.lm_head(tr.ln_f_go(x))[:, -1, :]


@torch.no_grad()
def sample(model, cache, mask, tp, start, organism_id, max_new_tokens, temperature, top_k):
    """Ancestral sampling as in GOGPT.generate(temperature, top_k): logits / temperature, keep the top_k logits
    (ties with the k-th kept), softmax, multinomial; finished rows emit pad. Uses a GO key/value cache."""
    cfg = model.config
    end_ids = torch.tensor([cfg.mf_end_token_id, cfg.bp_end_token_id, cfg.cc_end_token_id], device=start.device)
    b = start.size(0)
    finished = torch.zeros(b, dtype=torch.bool, device=start.device)
    out = [start]
    tok = start
    for pos in range(max_new_tokens):
        logits = _go_step(model, cache, mask, tp, tok, pos, organism_id).float() / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.zeros_like(nxt), nxt)
        out.append(nxt)
        finished = finished | torch.isin(nxt, end_ids)
        tok = nxt
        if pos % 8 == 7 and finished.all():
            break
    return torch.stack(out, 1)


@torch.no_grad()
def generate_samples(pred, seqs, organisms, n_samples=10, temperature=0.7, top_k=20, max_new_tokens=300,
                     bf16=False):
    """n_samples sampled GO sequences per protein and aspect; returns per protein {aspect: [list of terms] * n}."""
    info = pred.tokenizer_info
    start = {"MF": info["mf_start_token_id"], "BP": info["bp_start_token_id"], "CC": info["cc_start_token_id"]}
    model, dev = pred.model, pred.device
    enc = pred.protein_tokenizer(list(seqs), return_tensors="pt", padding="longest", truncation=True, max_length=1024)
    toks, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    org = torch.tensor([pred.organism_mapper.map_organism(o) for o in organisms], device=dev)
    emb = esm_layer(model, toks, mask) * mask[..., None].float()
    del toks
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        kv = _protein_stream(model, emb, mask)
        del emb
        rows, tp = len(seqs) * n_samples, mask.size(1)
        cache = []
        for k, v in kv:
            bufs = []
            for t in (k, v):
                buf = t.new_empty(rows, t.size(1), tp + max_new_tokens, t.size(3))
                buf[:, :, :tp] = t.repeat_interleave(n_samples, 0)
                bufs.append(buf)
            cache.append(bufs)
        del kv
        full_mask = torch.ones(rows, 1, 1, tp + max_new_tokens, dtype=torch.bool, device=dev)
        full_mask[:, 0, 0, :tp] = mask.bool().repeat_interleave(n_samples, 0)
        org_r = org.repeat_interleave(n_samples, 0)
        out = [{a: [] for a in ASPECTS} for _ in seqs]
        for asp in ASPECTS:
            go = torch.full((rows,), start[asp], device=dev)
            gen = sample(model, cache, full_mask, tp, go, org_r, max_new_tokens, temperature, top_k).cpu()
            for r in range(gen.size(0)):
                out[r // n_samples][asp].append(pred._decode_tokens(gen[r:r + 1], asp))
    return out


@torch.no_grad()
def generate(pred, seqs, organisms, decode="beam", beam_size=5, length_penalty=0.3, max_new_tokens=300):
    info = pred.tokenizer_info
    start = {"MF": info["mf_start_token_id"], "BP": info["bp_start_token_id"], "CC": info["cc_start_token_id"]}
    model, dev = pred.model, pred.device
    enc = pred.protein_tokenizer(list(seqs), return_tensors="pt", padding="longest", truncation=True, max_length=1024)
    toks, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    org = torch.tensor([pred.organism_mapper.map_organism(o) for o in organisms], device=dev)
    emb = esm_layer(model, toks, mask) * mask[..., None].float()
    model._pcache.clear()   # same batch (and same beam expansion) for all three aspects -> stream computed once
    out = [dict() for _ in seqs]
    for asp in ASPECTS:
        go = torch.full((len(seqs), 1), start[asp], device=dev)
        if decode == "greedy":
            gen = greedy(model, mask, go, org, emb, max_new_tokens)
        else:
            gen = model.generate_beam_search(protein_tokens=toks, protein_mask=mask, go_tokens=go,
                                             max_new_tokens=max_new_tokens, beam_size=beam_size,
                                             length_penalty=length_penalty, organism_id=org, protein_embeddings=emb)
        for i in range(len(seqs)):
            out[i][asp] = pred._decode_tokens(gen[i:i + 1], asp)
    model._pcache.clear()
    return out


def organism_resolver(pred):
    """Maps a species name to GO-GPT's organism vocabulary (full UniProt organism strings, top-200 training species).
    Exact match first, else the shortest vocabulary entry whose part before ' (' equals the name; else unknown (id 0)."""
    vocab = pred.organism_mapper.organism_to_idx
    by_base = {}
    for k in sorted(vocab, key=len):
        by_base.setdefault(k.split(" (")[0], k)

    def resolve(name):
        if not isinstance(name, str):
            return "<UNKNOWN>"
        if name in vocab:
            return name
        return by_base.get(name.split(" (")[0], "<UNKNOWN>")
    return resolve


# ----------------------------------------------------------------------------------------------- data

def parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        return list(ast.literal_eval(x)) if x.strip() not in ("", "nan") else []
    return list(x)


def load_inputs(a) -> pd.DataFrame:
    """protein_id, sequence, organism (species name)."""
    if not a.fasta:
        d = pd.read_parquet(REF / "test.parquet")
        return d[["protein_id", "sequence", "organism"]].copy()
    from cafa6.data import read_fasta
    df = read_fasta(Path(a.fasta)).rename(columns={"id": "protein_id", "seq": "sequence"})
    if a.taxa:
        t = pd.read_csv(a.taxa, sep="\t", header=None, dtype=str)
        if str(t.iloc[0, 1]).strip().lower() == "species":            # taxon-id -> species table
            tax2name = dict(zip(t.iloc[1:, 0].astype(int), t.iloc[1:, 1]))
        else:                                                          # protein -> taxon-id map
            df["taxon"] = df.protein_id.map(dict(zip(t.iloc[:, 0], t.iloc[:, 1].astype(int)))).fillna(-1).astype(int)
            n = pd.read_csv(a.taxon_names, sep="\t")
            tax2name = dict(zip(n.iloc[:, 0].astype(int), n.iloc[:, 1]))
        df["organism"] = df.taxon.map(lambda x: tax2name.get(int(x)))
    else:
        df["organism"] = None
    return df[["protein_id", "sequence", "organism"]]


def load_gt(a) -> pd.DataFrame:
    """protein<TAB>GO ground truth (already propagated for the paper test set; cafaeval propagates anyway)."""
    if a.gt:
        return pd.read_csv(a.gt, sep="\t", header=None, names=["protein_id", "term"], usecols=[0, 1], comment="#")
    d = pd.read_parquet(REF / "test.parquet")
    rows = [(pid, t) for pid, mf, bp, cc in zip(d.protein_id, d.go_mf, d.go_bp, d.go_cc)
            for col in (mf, bp, cc) for t in parse_list(col)]
    return pd.DataFrame(rows, columns=["protein_id", "term"]).drop_duplicates()


def released_preds() -> dict:
    """GO-GPT predictions shipped with the test set (go_pred text column of the HF dataset)."""
    d = pd.read_parquet(REF / "test.parquet")
    out = {}
    for pid, txt in zip(d.protein_id, d.go_pred):
        s = {k: [] for k in ASPECTS}
        for line in str(txt).splitlines():
            key = next((k for k in ASPECTS if f"({k})" in line), None)
            if key:
                s[key] += GO_RE.findall(line)
        out[pid] = s
    return out


# ----------------------------------------------------------------------------------------------- stages

def cmd_gen(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_inputs(a)
    if a.sample:
        df = df.sample(n=min(a.sample, len(df)), random_state=0)
    df = df.iloc[a.shard::a.n_shards].reset_index(drop=True)
    pred = load_predictor()
    resolve = organism_resolver(pred)
    df["organism_resolved"] = df.organism.map(resolve)
    known = (df.organism_resolved != "<UNKNOWN>").mean()
    log(f"{len(df)} proteins; organism in GO-GPT vocabulary for {known:.1%}")
    sampling = a.decode == "sample"
    if sampling:
        torch.manual_seed(a.seed + a.shard)
        path = out / f"samples_shard{a.shard}.jsonl.gz"
        opener = lambda p, m: gzip.open(p, m + "t")   # appended gzip members read back as one stream
    else:
        path = out / f"preds_shard{a.shard}.jsonl"
        opener = open
    done = {json.loads(l)["protein_id"] for l in opener(path, "r")} if path.exists() else set()
    todo = df[~df.protein_id.isin(done)]
    todo = todo.loc[todo.sequence.str.len().sort_values().index]
    if a.limit:
        todo = todo.iloc[:a.limit] if a.limit > 0 else todo.iloc[a.limit:]
    log(f"{len(done)} already done, {len(todo)} to go (batch {a.batch})")
    t0 = time.time()
    # padded protein length, plus the GO slots of the preallocated sampling cache
    lens = (todo.sequence.str.len().clip(upper=1022) + 2 + (300 if sampling else 0)).to_numpy()
    starts, s = [], 0
    while s < len(todo):
        starts.append(s)
        e = s + 1
        while e < len(todo) and e - s < a.batch and (e - s + 1) * lens[e] <= (a.token_budget or np.inf):
            e += 1
        s = e
    starts.append(len(todo))
    with opener(path, "a") as f:
        for bi, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
            chunk = todo.iloc[s:e]
            if sampling:
                res = generate_samples(pred, chunk.sequence.tolist(), chunk.organism_resolved.tolist(),
                                       n_samples=a.n_samples, temperature=a.temperature, top_k=a.top_k, bf16=a.bf16)
            else:
                res = generate(pred, chunk.sequence.tolist(), chunk.organism_resolved.tolist(), decode=a.decode)
            for pid, org, r in zip(chunk.protein_id, chunk.organism_resolved, res):
                f.write(json.dumps({"protein_id": pid, "organism": org, **r}) + "\n")
            f.flush()
            n = s + len(chunk)
            el = time.time() - t0
            if bi % 10 == 0 or n == len(todo):
                log(f"shard {a.shard}: {n}/{len(todo)}  {el:.0f}s  {el / n:.3f}s/protein  "
                    f"maxlen {chunk.sequence.str.len().max()}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
    (out / f"gen_time_shard{a.shard}.json").write_text(json.dumps({"n": len(todo), "seconds": time.time() - t0}))


def run_cafaeval(obo, pred_dir, gt_file, ia, out_dir):
    cmd = [str(CAFAEVAL), str(obo), str(pred_dir), str(gt_file), "-ia", str(ia), "-prop", "max", "-norm", "cafa",
           "-out_dir", str(out_dir), "-threads", "16"]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    bf = pd.read_csv(Path(out_dir) / "evaluation_best_f.tsv", sep="\t")
    bw = pd.read_csv(Path(out_dir) / "evaluation_best_f_w.tsv", sep="\t")
    rows = []
    for _, r in bf.iterrows():
        w = bw[(bw.filename == r.filename) & (bw.ns == r.ns)].iloc[0]
        rows.append({"method": Path(r["filename"]).stem, "aspect": NS_SHORT[r["ns"]], "Fmax": r["f"],
                     "Fmax_w": w["f_w"], "precision": r["pr"], "recall": r["rc"], "coverage": r["cov"],
                     "precision_w": w["pr_w"], "recall_w": w["rc_w"], "n_gt": int(r["n"])})
    return rows


def cmd_score(a):
    out = Path(a.out)
    preds = {}
    for p in sorted(out.glob("preds_shard*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            preds[r["protein_id"]] = {k: r[k] for k in ASPECTS}
    gt = load_gt(a)
    methods = {"gogpt_reproduced": preds}
    if not a.gt:
        methods["gogpt_released_preds"] = released_preds()
    protocols = list(PROTOCOLS) if not a.gt else ["cafa6_ia"]
    exclude = set(open(REF / "common_proteins.txt").read().split())
    evaldir = out / "cafaeval"
    rows = []
    for proto in protocols:
        obo, ia, excl = PROTOCOLS[proto]
        # scored on proteins that have reproduced predictions (the full set unless --sample was used)
        keep = set(preds) - (exclude if excl else set())
        g = gt[gt.protein_id.isin(keep)]
        pdir = evaldir / proto / "pred"
        pdir.mkdir(parents=True, exist_ok=True)
        g.to_csv(evaldir / proto / "gt.tsv", sep="\t", header=False, index=False)
        for m, ps in methods.items():
            with open(pdir / f"{m}.tsv", "w") as f:
                for pid in keep:
                    for asp in ASPECTS:
                        for t in ps.get(pid, {}).get(asp, []):
                            f.write(f"{pid}\t{t}\t1.0\n")
        for r in run_cafaeval(obo, pdir, evaldir / proto / "gt.tsv", ia, evaldir / proto / "out"):
            r.update(protocol=proto, n_proteins=g.protein_id.nunique())
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.tsv", sep="\t", index=False)
    wide = res.pivot_table(index=["protocol", "method"], columns="aspect", values=["Fmax", "Fmax_w", "coverage"])
    for v in ("Fmax", "Fmax_w"):
        wide[(v, "mean")] = wide[v][list(ASPECTS)].mean(axis=1)
    log(f"proteins with predictions: {len(preds)}, GT proteins: {gt.protein_id.nunique()}")
    print(wide.sort_index(axis=1).round(4).to_string(), flush=True)


def load_samples(out: Path) -> dict:
    s = {}
    for p in sorted(out.glob("samples_shard*.jsonl.gz")):
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            s[r["protein_id"]] = {k: r[k] for k in ASPECTS}
    return s


def frequency_scores(samples: dict, closure=None, alt=None) -> pd.DataFrame:
    """Score of a term = fraction of a protein's samples that contain it. With `closure`, every sample is first
    propagated to its ancestors (alt ids mapped), so a term also counts when only a descendant was sampled."""
    rows = []
    for pid, asp in samples.items():
        for runs in asp.values():
            n = len(runs)
            cnt = {}
            for terms in runs:
                ts = set(terms)
                if closure is not None:
                    ts = {alt.get(t, t) for t in ts}
                    ts = ts.union(*(closure.get(t, ()) for t in ts))
                for t in ts:
                    cnt[t] = cnt.get(t, 0) + 1
            rows += [(pid, t, round(c / n, 3)) for t, c in cnt.items()]
    return pd.DataFrame(rows, columns=["id", "term", "score"]).drop_duplicates(["id", "term"])


def cmd_score_sample(a):
    """Scores ten-sample frequencies on the paper test set with the bench_gogpt evaluator and subsets
    (S3_all, S6_noknowledge, identity bins), alongside the beam and greedy runs as a consistency check."""
    os.environ.setdefault("CAFA6_PROC", str(ROOT / "data" / "proc_gogpt"))
    from cafa6.data import PROC, ancestor_closure, load_go
    from cafa6.evaluate import evaluate
    from cafa6.homology import identity_bin
    out = Path(a.out)
    samples = load_samples(out)
    n = {len(r) for asp in samples.values() for r in asp.values()}
    log(f"{len(samples)} proteins with samples; samples per aspect: {sorted(n)}")
    parents, _, alt = load_go(REF / "go-basic_gogpt.obo")
    closure = ancestor_closure(parents)
    tag = f"gogpt_sample{a.n_samples}"
    preds = {f"{tag}_freq": frequency_scores(samples), f"{tag}_freq_prop": frequency_scores(samples, closure, alt)}
    for name, df in preds.items():
        df.to_csv(out / f"{name}.tsv.gz", sep="\t", index=False, header=False, compression="gzip")
        log(f"{name}: {len(df)} rows, {df.groupby('id').size().mean():.1f} terms/protein")
    for name, sub in (("gogpt_beam", ""), ("gogpt_greedy", "greedy")):
        rows = []
        for f in sorted((ROOT / "runs" / "gogpt_cafa5" / sub).glob("preds_shard*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                rows += [(r["protein_id"], t, 1.0) for k in ASPECTS for t in r.get(k, [])]
        preds[name] = pd.DataFrame(rows, columns=["id", "term", "score"]).drop_duplicates(["id", "term"])

    df = pd.read_parquet(PROC / "train.parquet", columns=["id", "split"])
    te_ids = df.id.values[df.split.values == "test"]
    missing = set(te_ids) - set(samples)
    if missing:
        log(f"WARNING: {len(missing)} test proteins have no samples (scored as empty predictions)")
    common = set(open(REF / "common_proteins.txt").read().split())
    subsets = {"S3_all": list(te_ids), "S6_noknowledge": [p for p in te_ids if p not in common]}
    ident = pd.read_parquet(PROC / "identity.parquet")
    idt = ident[ident.split == "test"].set_index("id").loc[te_ids]
    bins = identity_bin(idt.max_pident)
    for b in ["no_hit", "<30%", "30-50%", ">=50%"]:
        subsets[f"identity {b}"] = list(idt.index[bins == b])
    tables, long = [], []
    for name, ids in subsets.items():
        res = evaluate(preds, PROC / "gt_test.tsv", ids_subset=ids, obo=REF / "go-basic_gogpt.obo", ia=REF / "IA.txt")
        res["subset"], res["n"] = name, len(ids)
        long.append(res)
        w = res.pivot(index="method", columns="aspect", values="Fmax_w")[list(ASPECTS)]
        u = res.pivot(index="method", columns="aspect", values="Fmax")[list(ASPECTS)]
        s = w.add_prefix("Fw_").join(u.add_prefix("F_"))
        s["mean_Fw"], s["mean_F"] = w.mean(axis=1), u.mean(axis=1)
        s["subset"], s["n"] = name, len(ids)
        s = s.sort_values("mean_Fw", ascending=False)
        tables.append(s)
        log(f"== {name} (n={len(ids)})\n{s.drop(columns=['subset', 'n']).round(4).to_string()}")
    pd.concat(tables).reset_index().to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.concat(long).to_csv(out / "metrics.tsv", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "score", "all"])
    ap.add_argument("--fasta"); ap.add_argument("--taxa"); ap.add_argument("--gt")
    ap.add_argument("--taxon_names", default=str(ROOT / "data/raw/Test/testsuperset-taxon-list.tsv"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "gogpt_cafa5"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--token_budget", type=int, default=0,
                    help="cap proteins x padded length per batch (0 = only --batch); lets short proteins batch wider")
    ap.add_argument("--sample", type=int, default=0, help="random subset (seed 0); 0 = all")
    ap.add_argument("--decode", choices=["beam", "greedy", "sample"], default="beam")
    ap.add_argument("--n_samples", type=int, default=10, help="--decode sample: samples per protein and aspect")
    ap.add_argument("--temperature", type=float, default=0.7, help="--decode sample (paper section 4.2.3: 0.7)")
    ap.add_argument("--top_k", type=int, default=20, help="--decode sample (paper section 4.2.3: 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", help="--decode sample: bf16 autocast for the GO decoder")
    ap.add_argument("--limit", type=int, default=0, help="only the first N (shortest) proteins still to do, "
                                                         "or the last |N| (longest) if negative; for profiling")
    a = ap.parse_args()
    if a.cmd in ("gen", "all"):
        cmd_gen(a)
    if a.cmd in ("score", "all"):
        cmd_score_sample(a) if a.decode == "sample" else cmd_score(a)


if __name__ == "__main__":
    main()
