#!/usr/bin/env bash
# Regenerate the result tables of HiGO-R2 from the frozen prediction files already under runs/ (no training,
# no GPU, no new test look), write everything to a scratch directory and compare the key numbers with the
# tracked result files.
#
#   bash scripts/reproduce_tables.sh                       # all stages, outputs in /tmp/repro
#   OUT=/tmp/r2 STAGES="look3 novel" bash scripts/reproduce_tables.sh
#   CHECK_ONLY=1 STAGES="eval eval_ext" bash scripts/reproduce_tables.sh   # syntax / input check, no scoring
#
# Stages (STAGES, space separated; default: all, in this order):
#   look3     test look #3 table          cafa6.test_look --no-log    vs runs/test_look/look3_l3c0_twin
#   novel     beyond-homology terms       cafa6.novel_terms           vs runs/novel_terms
#   compare   paper comparison scoring    cafa6.compare_paper score   vs runs/compare_paper
#   eval      evaluation suite            cafa6.eval_suite            vs runs/eval_suite
#   eval_ext  evaluation suite, extended  cafa6.eval_suite --extended vs runs/eval_suite_v2 (untracked, if present)
#
# Environment: OUT (default /tmp/repro), PY (default ../bin/python), STAGES, CHECK_ONLY=1 (only check inputs and
# that each module imports), CAFA6_THREADS (eval_suite threads, default 24), EVAL_ARGS (extra eval_suite flags,
# e.g. "--skip-retrieval"). Approximate CPU times: look3 30 s, novel 1-2 min, compare and eval tens of minutes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="${PY:-../bin/python}"
OUT="${OUT:-/tmp/repro}"
STAGES="${STAGES:-look3 novel compare eval eval_ext}"
CHECK_ONLY="${CHECK_ONLY:-0}"
export PYTHONPATH=src
export CAFA6_THREADS="${CAFA6_THREADS:-24}"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
case "$OUT/" in
  "$HERE"/runs/*|"$HERE"/data/*) echo "OUT=$OUT is inside runs/ or data/; choose a scratch directory" >&2; exit 2 ;;
esac
RESULTS="$OUT/compare_results.tsv"
: > "$RESULTS"

section() { echo; echo "=== $* ==="; }
need() {
  local f
  for f in "$@"; do
    [ -e "$f" ] || { echo "missing input: $f" >&2; exit 3; }
  done
}

# compare REF NEW KEYS TOL [COLS]: join two TSV files on KEYS (comma separated) and compare the numeric COLS
# (comma separated; default: every numeric column the two files share). Prints and records MATCH or DIFF.
compare() {
  "$PY" - "$@" <<'EOF' | tee -a "$RESULTS"
import sys
import numpy as np
import pandas as pd
ref, new, keys, tol = sys.argv[1], sys.argv[2], sys.argv[3].split(","), float(sys.argv[4])
cols = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else None
a, b = pd.read_csv(ref, sep="\t"), pd.read_csv(new, sep="\t")
if cols is None:
    cols = [c for c in a.columns if c in b.columns and c not in keys
            and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
m = a.merge(b, on=keys, how="outer", suffixes=("_ref", "_new"), indicator=True)
missing = int((m["_merge"] != "both").sum())
m = m[m["_merge"] == "both"]
worst, where = 0.0, ""
for c in cols:
    x, y = m[c + "_ref"].astype(float).values, m[c + "_new"].astype(float).values
    nan_mismatch = np.isnan(x) != np.isnan(y)
    d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
    d[nan_mismatch] = np.inf
    if len(d) and np.nanmax(d) > worst:
        worst = float(np.nanmax(d))
        r = m.iloc[int(np.nanargmax(d))]
        where = f" (worst: {c} at " + ",".join(str(r[k]) for k in keys) + f": ref {r[c + '_ref']} new {r[c + '_new']})"
ok = missing == 0 and worst <= tol
print(f"{'MATCH' if ok else 'DIFF'}\t{new}\trows {len(m)}, unmatched {missing}, cols {len(cols)}, "
      f"max |diff| {worst:.2e} (tol {tol:g}){'' if ok else where}")
EOF
}

# compare_json REF NEW TOL: every numeric leaf of two JSON files.
compare_json() {
  "$PY" - "$@" <<'EOF' | tee -a "$RESULTS"
import json, sys
ref, new, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, float(o)
a, b = dict(leaves(json.load(open(ref)))), dict(leaves(json.load(open(new))))
missing = sorted(set(a) ^ set(b))
d = {k: abs(a[k] - b[k]) for k in set(a) & set(b)}
worst = max(d.values(), default=0.0)
ok = not missing and worst <= tol
print(f"{'MATCH' if ok else 'DIFF'}\t{new}\tvalues {len(d)}, unmatched {len(missing)}, max |diff| {worst:.2e} "
      f"(tol {tol:g})" + ("" if ok else f" (worst: {max(d, key=d.get) if d else missing[:3]})"))
EOF
}

importable() { "$PY" -c "import importlib, sys; importlib.import_module(sys.argv[1])" "$1" && echo "  import $1: ok"; }

LOOKS_MD5_BEFORE="$(md5sum runs/test_looks.md | cut -d' ' -f1)"
section "setup"
echo "repo:    $HERE"
echo "python:  $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
echo "out:     $OUT"
echo "stages:  $STAGES"
[ "$CHECK_ONLY" = 1 ] && echo "mode:    CHECK_ONLY (inputs and imports only, nothing is scored)"
need runs/stacker_tv/l3_c0/pred_test_twin.npz runs/gogpt_sample10/gogpt_sample10_freq.tsv.gz \
     data/proc_gogpt/gt_test.tsv data/proc_gogpt_tv2/knn/test_knn.npz runs/gogpt_cafa5/ref/go-basic_gogpt.obo \
     runs/gogpt_cafa5/ref/IA.txt
ls runs/gogpt_cafa5/preds_shard*.jsonl > /dev/null

for stage in $STAGES; do
  case "$stage" in

  look3)
    section "(a) test look #3 table (cafa6.test_look --no-log; runs/test_looks.md is not modified)"
    REF=runs/test_look/look3_l3c0_twin
    D="$OUT/test_look/look3_l3c0_twin"
    if [ "$CHECK_ONLY" = 1 ]; then importable cafa6.test_look; continue; fi
    "$PY" -m cafa6.test_look --pred runs/stacker_tv/l3_c0/pred_test_twin.npz --name look3_l3c0_twin --look 3 \
      --out "$D" --no-log
    echo "-- compare with $REF"
    compare "$REF/summary.tsv" "$D/summary.tsv" method 1e-9
    compare "$REF/per_bin.tsv" "$D/per_bin.tsv" method 1e-9
    compare_json "$REF/result.json" "$D/result.json" 1e-9
    ;;

  novel)
    section "(d) beyond-homology terms (cafa6.novel_terms)"
    need runs/test_look/look3_l3c0_twin/cafaeval/best_f_w.tsv
    D="$OUT/novel_terms"
    if [ "$CHECK_ONLY" = 1 ]; then importable cafa6.novel_terms; continue; fi
    "$PY" -m cafa6.novel_terms --out "$D"
    echo "-- compare with runs/novel_terms"
    compare runs/novel_terms/summary.tsv "$D/summary.tsv" model,aspect,bin 1e-9
    compare runs/novel_terms/by_bin.tsv "$D/by_bin.tsv" model,aspect,bin 1e-9
    ;;

  compare)
    section "(c) paper comparison, scoring step (cafa6.compare_paper score + tables; predictions read from runs/compare_paper/preds)"
    need runs/compare_paper/preds/blast_knn.npz runs/compare_paper/preds/bioreason_rl.npz \
         runs/compare_paper/catalogue_ids.txt
    D="$OUT/compare_paper"
    if [ "$CHECK_ONLY" = 1 ]; then importable cafa6.compare_paper; continue; fi
    "$PY" -m cafa6.compare_paper score --out "$D"
    "$PY" -m cafa6.compare_paper score --out "$D" --subset_file runs/compare_paper/catalogue_ids.txt
    "$PY" -m cafa6.compare_paper tables --out "$D"
    echo "-- compare with runs/compare_paper (files are written with 4 decimals)"
    compare runs/compare_paper/summary.tsv "$D/summary.tsv" subset,method 1e-4
    compare runs/compare_paper/per_bin.tsv "$D/per_bin.tsv" method,bin 1e-4
    compare_json runs/compare_paper/spearman_identity.json "$D/spearman_identity.json" 1e-9
    compare runs/compare_paper/summary_sub.tsv "$D/summary_sub.tsv" subset,method 1e-4
    if cmp -s runs/compare_paper/report_tables.md "$D/report_tables.md"; then
      echo -e "MATCH\t$D/report_tables.md\tidentical to runs/compare_paper/report_tables.md" | tee -a "$RESULTS"
    else
      echo -e "DIFF\t$D/report_tables.md\tdiffers from runs/compare_paper/report_tables.md" | tee -a "$RESULTS"
    fi
    ;;

  eval|eval_ext)
    # eval_suite has no output flag; its module-level output paths are pointed at the scratch directory.
    if [ "$stage" = eval ]; then
      section "(b) evaluation suite (cafa6.eval_suite)"
      D="$OUT/eval_suite"; REF=runs/eval_suite; VAR=OUT; FLAG=""
    else
      section "(b) evaluation suite, extended (cafa6.eval_suite --extended)"
      D="$OUT/eval_suite_v2"; REF=runs/eval_suite_v2; VAR=OUT_EXT; FLAG="--extended"
    fi
    need runs/test_look/look3_l3c0_twin/cafaeval/best_f_w.tsv
    if [ "$CHECK_ONLY" = 1 ]; then
      "$PY" -c "import cafa6.eval_suite as e, pathlib; setattr(e, '$VAR', pathlib.Path('$D')); \
print('  import cafa6.eval_suite: ok; $VAR ->', e.$VAR)"
      continue
    fi
    # shellcheck disable=SC2086
    "$PY" -c "import sys, pathlib, cafa6.eval_suite as e; setattr(e, '$VAR', pathlib.Path(sys.argv[1])); \
sys.argv = ['eval_suite'] + sys.argv[2:]; e.main()" "$D" $FLAG ${EVAL_ARGS:-}
    if [ -f "$REF/headline.tsv" ]; then
      echo "-- compare with $REF$([ "$stage" = eval_ext ] && echo ' (untracked; check that it is not being rewritten)')"
      compare "$REF/headline.tsv" "$D/headline.tsv" model,metric,aspect 1e-9
      compare "$REF/identity_bins.tsv" "$D/identity_bins.tsv" identity,model,f1,aspect,bin 1e-9
      compare "$REF/consistency.tsv" "$D/consistency.tsv" model,aspect 1e-9
    else
      echo "no $REF/headline.tsv to compare with"
    fi
    ;;

  *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done

section "summary"
if [ "$(md5sum runs/test_looks.md | cut -d' ' -f1)" = "$LOOKS_MD5_BEFORE" ]; then
  echo "runs/test_looks.md unchanged"
else
  echo "runs/test_looks.md CHANGED during this run (another process may have written to it)"
fi
if [ -s "$RESULTS" ]; then
  cut -f1,2 "$RESULTS"
  n_diff="$(grep -c '^DIFF' "$RESULTS" || true)"
  echo "$(grep -c '^MATCH' "$RESULTS" || true) MATCH, $n_diff DIFF (details in $RESULTS)"
  [ "$n_diff" = 0 ]
else
  echo "no comparisons were run"
fi
