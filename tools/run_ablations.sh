#!/usr/bin/env bash
# Ablations of HiGO components on ESM2-35M, then 5-fold CV of the full model.
# Usage: tools/run_ablations.sh <adapter> <gpu>
set -u
cd "$(dirname "$0")/.."
ADAPTER=${1:-lora}
GPU=${2:-0}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
PY=../bin/python
run() { local name=$1; shift; [ -e "logs/$name.done" ] && return; $PY -m cafa6.train --config configs/dev.yaml name=$name adapter=$ADAPTER "$@" > logs/$name.log 2>&1; }

run abl_no_hier_query hier_query=false &
run abl_no_mcm use_mcm=false &
run abl_mean_pool pooling=mean &
wait
run abl_no_taxon use_taxon=false &
run abl_no_ia_weight ia_weight=false &
run abl_softmax_pool pooling=softmax &
wait
for f in 0 1 2 3 4; do
  run cv_35m_fold$f fold=$f n_folds=5 epochs=8 patience=2 &
  if [ $f -eq 2 ]; then wait; fi
done
wait
touch logs/ablations_cv.done
