"""Convert full-score outputs of external predictors to (protein, GO id, score) TSV rows with score >= THR,
the input of `compare_paper import-tsv`.

  python to_tsv.py deepfri OUT.tsv THR prefix_MF_pred_scores.json [..._BP_... ..._CC_...]
      DeepFRI save_predictions JSON (pdb_chains, Y_hat, goterms)
  python to_tsv.py sprof OUT.tsv THR run_all_preds.txt [more chunks]
      SPROF-GO export_predictions *_all_preds.txt (term header, then per protein MF/BP/CC score lines)
"""
import json
import sys

import numpy as np

kind, out, thr, files = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
n = 0
with open(out, "w") as w:
    if kind == "deepfri":
        for fn in files:
            d = json.load(open(fn))
            Y, terms = np.asarray(d["Y_hat"], np.float32), np.asarray(d["goterms"])
            for p, y in zip(d["pdb_chains"], Y):
                for j in np.where(y >= thr)[0]:
                    w.write(f"{p}\t{terms[j]}\t{y[j]:.4f}\n"); n += 1
    elif kind == "sprof":
        for fn in files:
            lines = open(fn).read().split("\n")
            terms, i = {}, 1
            for _ in range(3):  # "MF:", ids, names
                asp = lines[i].rstrip(":"); terms[asp] = lines[i + 1].split("; "); i += 3
            while i < len(lines):
                if not lines[i].strip():
                    i += 1; continue
                p = lines[i]; i += 1
                for _ in range(3):
                    asp = lines[i].rstrip(":")
                    s = np.array(lines[i + 1].split("; "), np.float32)
                    assert len(s) == len(terms[asp]), (p, asp)
                    for j in np.where(s >= thr)[0]:
                        w.write(f"{p}\t{terms[asp][j]}\t{s[j]:.3f}\n"); n += 1
                    i += 2
    else:
        sys.exit(f"unknown kind {kind}")
print(f"wrote {n} rows to {out}")
