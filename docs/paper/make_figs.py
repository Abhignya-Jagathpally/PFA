"""Figures for docs/paper/benchmark_hygiene_draft.md, drawn only from tracked result tables.

    ../bin/python docs/paper/make_figs.py            # from cafa6/; writes docs/paper/figs/

Inputs (all tracked under runs/):
  runs/knn_trainval/stats_test_bins.tsv   GO-GPT test, best DIAMOND identity to GO-GPT train(+val) proteins
  runs/knn_trainval/stats_tv_bins.tsv     TV v2, same definition
  runs/fresh24/per_bin.tsv                fresh24 look #1, per-bin scores (n_proteins gives the composition)
  runs/fresh24_look2/per_bin.tsv          fresh24 look #2 (forecasting model F1)
  runs/compare_paper/per_bin.tsv          GO-GPT test, per-bin scores under one scorer
  runs/compare_paper/summary.tsv          GO-GPT test, whole-set scores
  runs/fresh24/summary.tsv                fresh24, whole-set scores

Outputs: figs/fig1_identity_composition.png, figs/fig2_fmax_by_bin.png, figs/fig3_benchmark_shift.png and one
figN_data.tsv per figure with the plotted numbers. No scoring is done here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
BIN_LABELS = ["no hit", "<30%", "30-50%", "50-80%", "80-95%", "95-99.9%", "100%"]


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel, sep="\t")


def fig1() -> None:
    test = read("runs/knn_trainval/stats_test_bins.tsv")
    test = test.set_index(test.columns[0])
    tv = read("runs/knn_trainval/stats_tv_bins.tsv")
    tv = tv.set_index(tv.columns[0])
    f24 = read("runs/fresh24/per_bin.tsv")
    f24 = f24[(f24["grouping"] == "bin") & (f24["method"] == "ontologue")]

    rows = {
        "GO-GPT temporal test\n(8,630; train refs)": test.loc[BINS, "before"].astype(float),
        "GO-GPT temporal test\n(8,630; train+val refs)": test.loc[BINS, "after"].astype(float),
        "TV v2 selection set\n(7,566; train+val refs)": tv.loc[BINS, "after"].astype(float),
        "fresh24 NK-all, with HPA load\n(22,700; train+val refs)": f24[f24["set"] == "NK-all"].set_index("group").loc[BINS, "n_proteins"].astype(float),
        "fresh24 FR24-P, primary\n(11,231; train+val refs)": f24[f24["set"] == "FR24-P"].set_index("group").loc[BINS, "n_proteins"].astype(float),
    }
    counts = pd.DataFrame(rows).T[BINS]
    shares = counts.div(counts.sum(axis=1), axis=0)
    out = counts.copy()
    out.columns = [f"n_{b}" for b in BINS]
    for b in BINS:
        out[f"share_{b}"] = shares[b].round(4)
    out.index = [i.replace("\n", " ") for i in out.index]
    out.to_csv(OUT / "fig1_data.tsv", sep="\t", index_label="set")

    colors = plt.cm.viridis([i / (len(BINS) - 1) for i in range(len(BINS))])
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    left = pd.Series(0.0, index=shares.index)
    for b, lab, c in zip(BINS, BIN_LABELS, colors):
        ax.barh(range(len(shares)), shares[b], left=left, color=c, label=lab, edgecolor="white", linewidth=0.5)
        for i, (v, l0) in enumerate(zip(shares[b], left)):
            if v >= 0.06:
                ax.text(l0 + v / 2, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if b in BINS[:4] else "black")
        left += shares[b]
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels(shares.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of proteins (numbers in bars: %)")
    ax.set_title("Best DIAMOND identity to the nearest GO-GPT training protein", fontsize=10)
    ax.legend(ncol=7, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_identity_composition.png", dpi=200)
    plt.close(fig)


def fig2() -> None:
    cp = read("runs/compare_paper/per_bin.tsv")
    gt_methods = [
        ("higo_r2", "HiGO-R2 / Ontologue"),
        ("gogpt_sample10", "GO-GPT ten-sample"),
        ("bioreason_rl", "BioReason-Pro RL"),
        ("blast_knn", "BLAST-KNN (train refs)"),
        ("sprof_go", "SPROF-GO"),
    ]
    f1 = read("runs/fresh24/per_bin.tsv")
    f2 = read("runs/fresh24_look2/per_bin.tsv")
    f = pd.concat([f1, f2[f2["method"] == "forecast_F1_ens"]])
    f = f[(f["set"] == "FR24-P") & (f["grouping"] == "bin")]
    fr_methods = [
        ("ontologue", "HiGO-R2 / Ontologue"),
        ("gogpt_sample10", "GO-GPT ten-sample"),
        ("blast_knn", "BLAST-KNN (train refs)"),
        ("sprof_go", "SPROF-GO"),
        ("ontologue_knn", "sharpened kNN prior (train+val)"),
        ("forecast_F1_ens", "forecasting model F1"),
    ]
    style = {
        "HiGO-R2 / Ontologue": ("tab:blue", "o", "-"),
        "GO-GPT ten-sample": ("tab:orange", "s", "-"),
        "BioReason-Pro RL": ("tab:red", "^", "--"),
        "BLAST-KNN (train refs)": ("tab:gray", "v", ":"),
        "SPROF-GO": ("tab:green", "D", "-"),
        "sharpened kNN prior (train+val)": ("tab:purple", "x", "--"),
        "forecasting model F1": ("tab:brown", "P", "-."),
    }
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, (title, df, key, methods, ncol) in zip(axes, [
        ("GO-GPT temporal test (bins: train refs)", cp, "method", gt_methods, "Fw_mean"),
        ("fresh24 FR24-P (bins: train+val refs)", f, "method", fr_methods, "Fmax_w_mean"),
    ]):
        for m, lab in methods:
            d = df[df[key] == m].set_index("bin" if "bin" in df.columns else "group")
            ys = [float(d.loc[b, ncol]) for b in BINS]
            if m in ("blast_knn", "ontologue_knn"):
                ys_plot = [None if (b == "no_hit" and y == 0.0) else y for b, y in zip(BINS, ys)]
            else:
                ys_plot = ys
            c, mk, ls = style[lab]
            ax.plot(range(len(BINS)), [float("nan") if y is None else y for y in ys_plot], color=c, marker=mk,
                    linestyle=ls, label=lab, linewidth=1.4, markersize=5)
            for b, y in zip(BINS, ys):
                records.append({"benchmark": title, "method": lab, "bin": b, "Fmax_w_mean": y})
        nrow = df[df[key] == methods[0][0]].set_index("bin" if "bin" in df.columns else "group")
        ncolname = "n" if "n" in nrow.columns else "n_proteins"
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels([f"{l}\n(n={int(nrow.loc[b, ncolname]):,})" for b, l in zip(BINS, BIN_LABELS)],
                           fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("IA-weighted Fmax, mean of MF/BP/CC\n(threshold chosen within each bin)")
    axes[0].set_ylim(0.0, 0.95)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_fmax_by_bin.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT / "fig2_data.tsv", sep="\t", index=False)


def fig3() -> None:
    cp = read("runs/compare_paper/summary.tsv")
    cp = cp[cp["subset"] == "S3_all"].set_index("method")
    fr = read("runs/fresh24/summary.tsv")
    fr = fr[fr["set"] == "FR24-P"].set_index("method")
    pairs = [
        ("higo_r2", "ontologue", "HiGO-R2 / Ontologue"),
        ("gogpt_sample10", "gogpt_sample10", "GO-GPT ten-sample"),
        ("gogpt_beam", "gogpt_beam", "GO-GPT beam"),
        ("blast_knn", "blast_knn", "BLAST-KNN (train refs)"),
        ("blast_knn_trainval", "blast_knn_trainval", "BLAST-KNN (train+val refs)"),
        ("diamondscore", "diamondscore", "DIAMONDScore"),
        ("sprof_go", "sprof_go", "SPROF-GO"),
        ("deepgoplus", "deepgoplus", "DeepGOPlus 1.0.10"),
        ("proteinfer_e5", "proteinfer_e5", "ProteInfer (5 models)"),
        ("deepfri_cnn", "deepfri_cnn", "DeepFRI (sequence CNN)"),
        ("naive", "naive", "Naive"),
    ]
    rows = []
    for a, b, lab in pairs:
        rows.append({"method": lab, "gogpt_test_S3_Fw_mean": round(float(cp.loc[a, "Fw_mean"]), 4),
                     "fr24p_Fw_mean": round(float(fr.loc[b, "Fmax_w_mean"]), 4)})
    d = pd.DataFrame(rows)
    d["change"] = (d["fr24p_Fw_mean"] - d["gogpt_test_S3_Fw_mean"]).round(4)
    d.to_csv(OUT / "fig3_data.tsv", sep="\t", index=False)

    built_on_gogpt = {"HiGO-R2 / Ontologue", "GO-GPT ten-sample", "GO-GPT beam", "BLAST-KNN (train refs)",
                      "BLAST-KNN (train+val refs)", "DIAMONDScore", "Naive"}
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    label_y = {}
    last = None
    for i in d.sort_values("fr24p_Fw_mean", ascending=False).index:
        y = d.loc[i, "fr24p_Fw_mean"]
        if last is not None and last - y < 0.013:
            y = last - 0.013
        label_y[i] = last = y
    for i, r in d.iterrows():
        c = "tab:red" if r["method"] in built_on_gogpt else "tab:green"
        ax.plot([0, 1], [r["gogpt_test_S3_Fw_mean"], r["fr24p_Fw_mean"]], color=c, marker="o", linewidth=1.2,
                markersize=4, alpha=0.85)
        ax.plot([1, 1.03], [r["fr24p_Fw_mean"], label_y[i]], color="0.6", linewidth=0.5)
        ax.text(1.04, label_y[i], f"{r['method']} ({r['change']:+.3f})", fontsize=6.5, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GO-GPT temporal test\n(8,630)", "fresh24 FR24-P\n(11,231)"], fontsize=8)
    ax.set_xlim(-0.15, 1.9)
    ax.set_ylabel("IA-weighted Fmax, mean of MF/BP/CC")
    ax.set_title("Same frozen predictors on two temporal benchmarks", fontsize=10)
    ax.plot([], [], color="tab:red", label="trained on / references from GO-GPT data")
    ax.plot([], [], color="tab:green", label="external tool, own older training data")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_benchmark_shift.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
