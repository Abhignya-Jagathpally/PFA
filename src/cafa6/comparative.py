"""Cross-species orthology pilot: does true comparative-genomics signal add anything on top of
plain same-corpus DIAMOND homology transfer?

`baselines.py`'s `diamond_knn` BLASTs every val/test protein against the *entire* training split and
transfers labels from whatever hits it finds, regardless of species -- that conflates "close homolog,
possibly same species" with cross-species evolutionary conservation. This module isolates the
latter: DIAMOND search restricted to a curated panel of well-annotated reference-genome model organisms
(`REFERENCE_TAXA`, spanning animals/plants/fungi/bacteria), with same-taxon hits explicitly dropped, so
any signal found is attributable to conservation *across* species rather than within-species sequence
redundancy that `diamond_knn` already captures.

Usage: python -m cafa6.comparative
Writes runs/pilot_comparative_genomics/{metrics.tsv,SUMMARY.md}, data/proc/ortho_{val,test}.npy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cafa6.data import PROC, ROOT
from cafa6.evaluate import dense_to_long, evaluate, summarize
from cafa6.homology import diamond_score, diamond_search, identity_bin, max_identity

REFERENCE_TAXA = {
    9606: "Homo sapiens", 10090: "Mus musculus", 3702: "Arabidopsis thaliana",
    559292: "Saccharomyces cerevisiae S288C", 10116: "Rattus norvegicus",
    284812: "Schizosaccharomyces pombe", 83333: "Escherichia coli K-12",
    7227: "Drosophila melanogaster", 6239: "Caenorhabditis elegans",
    83332: "Mycobacterium tuberculosis H37Rv",
}


def build_reference_db(train_df: pd.DataFrame) -> pd.DataFrame:
    """Train-split proteins from the curated reference-genome panel only (never val/test: no leakage)."""
    return train_df[(train_df.split == "train") & (train_df.taxon.isin(REFERENCE_TAXA))]


def cross_species_hits(hits: pd.DataFrame, query_taxon: pd.Series, db_taxon: pd.Series) -> pd.DataFrame:
    """Drops hit rows where the query and the hit protein share a taxon (same-species redundancy,
    already captured by the same-corpus `diamond_knn` baseline) -- what remains is cross-species only."""
    qt = hits.q.map(query_taxon)
    st = hits.s.map(db_taxon)
    return hits[(qt != st) & qt.notna() & st.notna()]


def main():
    out = ROOT / "runs" / "pilot_comparative_genomics"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "train.parquet")
    vocab = pd.read_csv(PROC / "vocab.tsv", sep="\t")
    Y = sp.load_npz(PROC / "Y.npz").tocsr()
    terms = vocab.term.values
    tr = np.where(df.split == "train")[0]
    taxon_map = pd.Series(df.taxon.values, index=df.id.values)

    freq = np.asarray(Y[tr].mean(0)).ravel().astype(np.float32)

    ref_db = build_reference_db(df)
    ref_idx = ref_db.index.values  # RangeIndex positions align with Y rows (verified against train.parquet)
    db_ids = df.id.values[ref_idx]
    Y_db = Y[ref_idx]

    # Single DIAMOND run (reference panel db vs all val+test queries), mirroring baselines.py's pattern.
    q_df = df[df.split != "train"]
    raw_hits = diamond_search(ref_db, q_df)
    hits = cross_species_hits(raw_hits, taxon_map, taxon_map)

    results = []
    ident = []
    scores = {}
    for split in ("val", "test"):
        idx = np.where(df.split == split)[0]
        ids = df.id.values[idx]
        osc = diamond_score(hits, ids, db_ids, Y_db)
        np.save(PROC / f"ortho_{split}.npy", osc.astype(np.float16))
        scores[split] = osc
        mi = max_identity(hits, ids)
        ident.append(pd.DataFrame({"id": ids, "split": split, "max_pident": mi.values,
                                   "id_bin": identity_bin(mi).astype(str).values}))
    ident = pd.concat(ident, ignore_index=True)

    diamond_available = (PROC / "diamond_val.npy").exists() and (PROC / "diamond_test.npy").exists()

    coverage = {}
    for split in ("val", "test"):
        idx = np.where(df.split == split)[0]
        ids = df.id.values[idx]
        naive = np.tile(freq, (len(idx), 1))
        ortho = scores[split]
        has_ortho_hit = np.isin(ids, hits.q.unique())
        coverage[split] = float(has_ortho_hit.mean())

        preds = {"naive": dense_to_long(naive, ids, terms), "ortho_knn": dense_to_long(ortho, ids, terms)}

        if diamond_available:
            dmd = np.load(PROC / f"diamond_{split}.npy").astype(np.float32)

            def with_naive(s):
                return np.where((s.sum(1) > 0)[:, None], 0.8 * s + 0.2 * naive, naive)

            # both arms get the same naive blend, otherwise the prior is credited to the ortholog channel
            preds["diamond_knn"] = dense_to_long(dmd, ids, terms)
            preds["diamond+naive"] = dense_to_long(with_naive(dmd), ids, terms)
            preds["diamond+ortho"] = dense_to_long(with_naive(np.maximum(dmd, ortho)), ids, terms)

        r = evaluate(preds, PROC / f"gt_{split}.tsv")
        r["split"] = split
        results.append(r)

    res = pd.concat(results, ignore_index=True)
    res.to_csv(out / "metrics.tsv", sep="\t", index=False)

    print(f"reference DB: {len(ref_db)} train proteins across {ref_db.taxon.nunique()} reference taxa")
    print(f"cross-species hits remaining after same-taxon filter: {len(hits)} (raw: {len(raw_hits)})")
    print("ortho coverage (any cross-species hit):", coverage)
    for s in ("val", "test"):
        print(f"== {s}")
        print(summarize(res[res.split == s]).round(4).to_string())
    print(ident.groupby(["split", "id_bin"], observed=False).size().unstack())

    # ---- SUMMARY.md ----
    summ_val = summarize(res[res.split == "val"]).round(4)
    summ_test = summarize(res[res.split == "test"]).round(4)

    def fmt_table(summ: pd.DataFrame) -> str:
        cols = list(summ.columns)
        header = "| method | " + " | ".join(cols) + " |"
        sep = "|---|" + "---|" * len(cols)
        lines = [header, sep]
        for method, row in summ.iterrows():
            lines.append(f"| {method} | " + " | ".join(f"{row[c]:.4f}" for c in cols) + " |")
        return "\n".join(lines)

    bin_counts = ident.groupby(["split", "id_bin"], observed=False).size().unstack(fill_value=0)
    hit_bins = ident[ident.id_bin != "no_hit"]
    frac_remote_of_hits = {}
    for split in ("val", "test"):
        h = hit_bins[hit_bins.split == split]
        frac_remote_of_hits[split] = float((h.id_bin == "<30%").mean()) if len(h) else float("nan")

    beats = {}
    for split, summ in (("val", summ_val), ("test", summ_test)):
        if "diamond+ortho" in summ.index and "diamond+naive" in summ.index:
            beats[split] = (float(summ.loc["diamond+ortho", "kaggle_mean_Fw"]),
                            float(summ.loc["diamond+naive", "kaggle_mean_Fw"]))

    lines = []
    lines.append("# Pilot: cross-species orthology signal for CAFA-6 GO prediction")
    lines.append("")
    lines.append("Research question: does DIAMOND search restricted to a curated cross-species reference "
                 "panel, with same-taxon hits excluded, add incremental signal on top of the existing "
                 "same-corpus DIAMOND-kNN homology baseline (`diamond_knn` in `baselines.py`, which searches "
                 "the whole training split regardless of species) -- or is it redundant with it?")
    lines.append("")
    lines.append(f"Reference panel: {len(REFERENCE_TAXA)} model organisms "
                 f"({', '.join(REFERENCE_TAXA.values())}); {len(ref_db)} train-split proteins "
                 f"({ref_db.taxon.nunique()} distinct reference taxa actually present).")
    lines.append("")
    lines.append("## Coverage: fraction of proteins with ANY cross-species ortholog hit")
    lines.append("")
    lines.append("| split | n proteins | fraction with >=1 cross-species hit |")
    lines.append("|---|---|---|")
    for split in ("val", "test"):
        n = int((df.split == split).sum())
        lines.append(f"| {split} | {n} | {coverage[split]:.4f} |")
    lines.append("")
    lines.append(f"Cross-species hits surviving the same-taxon filter: {len(hits)} rows "
                 f"(raw DIAMOND hits against the reference panel, before filtering: {len(raw_hits)}; "
                 f"{len(hits) / max(len(raw_hits), 1):.1%} kept). Note {(q_df.taxon.isin(REFERENCE_TAXA)).mean():.1%} "
                 "of val/test proteins are themselves from a reference taxon, so a large share of raw hits "
                 "against the panel are same-species (excluded).")
    lines.append("")
    lines.append("## Fmax_w per aspect and Kaggle mean")
    lines.append("")
    lines.append("### Validation")
    lines.append("")
    lines.append(fmt_table(summ_val))
    lines.append("")
    lines.append("### Test")
    lines.append("")
    lines.append(fmt_table(summ_test))
    lines.append("")
    if not diamond_available:
        lines.append("`data/proc/diamond_{val,test}.npy` were not found, so the `diamond_knn` and "
                     "`diamond+ortho` arms were skipped (recomputing them is `baselines.py`'s job, which "
                     "this module must not touch). Only `naive` and `ortho_knn` are reported above.")
        lines.append("")
    lines.append("## Does diamond+ortho beat diamond+naive?")
    lines.append("")
    lines.append("Both arms use the same rule (0.8 x kNN score + 0.2 x naive, naive when there is no hit), so the "
                 "difference is the ortholog channel alone. Raw `diamond_knn` is listed for reference only; "
                 "comparing against it would credit the naive prior to the ortholog channel.")
    lines.append("")
    if beats:
        for split, (combo, base) in beats.items():
            delta = combo - base
            verdict = "beats" if delta > 0 else ("ties" if delta == 0 else "loses to")
            lines.append(f"- **{split}**: diamond+ortho {combo:.4f} vs diamond+naive {base:.4f} "
                         f"(delta {delta:+.4f}), so diamond+ortho {verdict} diamond+naive.")
        lines.append("")
        lines.append("The reference panel is a subset of the training split that `diamond_knn` already searches, "
                     "so the ortholog channel cannot find new neighbours; it only re-weights a subset of the "
                     "existing ones, and taking the elementwise max inflates scores for terms carried by "
                     "those neighbours.")
    else:
        lines.append("Not evaluated (diamond_knn baseline scores were unavailable; see above).")
    lines.append("")
    lines.append("## Ortholog hit quality by percent-identity bin (`identity_bin`)")
    lines.append("")
    lines.append("Identity of each protein's best cross-species hit (all val/test proteins):")
    lines.append("")
    lines.append("| split | no_hit (all proteins) | <30% | 30-50% | >=50% |")
    lines.append("|---|---|---|---|---|")
    for split in ("val", "test"):
        row = bin_counts.loc[split] if split in bin_counts.index else pd.Series(dtype=int)
        total = int(row.sum()) if len(row) else int((df.split == split).sum())
        no_hit = int(row.get("no_hit", 0))
        b30 = int(row.get("<30%", 0))
        b50 = int(row.get("30-50%", 0))
        bge = int(row.get(">=50%", 0))
        lines.append(f"| {split} | {no_hit}/{total} ({no_hit/total:.1%}) | {b30} ({b30/total:.1%}) | "
                     f"{b50} ({b50/total:.1%}) | {bge} ({bge/total:.1%}) |")
    lines.append("")
    lines.append("Share of proteins with a hit whose best hit is below 30% identity (where homology transfer "
                 "is least reliable):")
    lines.append("")
    for split in ("val", "test"):
        v = frac_remote_of_hits[split]
        lines.append(f"- **{split}**: {v:.1%}" if v == v else f"- **{split}**: n/a (no hits)")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is a pilot, not a trained model. The reference panel (10 model organisms) covers "
                 f"{len(ref_db)}/{len(tr)} ({len(ref_db)/len(tr):.1%}) of the training split, and "
                 f"{(q_df.taxon.isin(REFERENCE_TAXA)).mean():.1%} of val/test proteins come from one of these "
                 "taxa, so the corpus behind `diamond_knn` is already dominated by the same species.")

    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
