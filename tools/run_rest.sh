#!/usr/bin/env bash
# Second ablation wave and 5-fold CV of the full HiGO-35M model on a given GPU.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
PY=../bin/python
run() { local name=$1; shift; [ -e "logs/$name.done" ] && return; $PY -m cafa6.train --config configs/dev.yaml name=$name adapter=lora "$@" > logs/$name.log 2>&1; }

run abl_no_taxon use_taxon=false &
run abl_no_ia_weight ia_weight=false &
run abl_softmax_pool pooling=softmax &
run cv_35m_fold0 fold=0 n_folds=5 epochs=8 patience=2 &
run cv_35m_fold1 fold=1 n_folds=5 epochs=8 patience=2 &
wait
run cv_35m_fold2 fold=2 n_folds=5 epochs=8 patience=2 &
run cv_35m_fold3 fold=3 n_folds=5 epochs=8 patience=2 &
run cv_35m_fold4 fold=4 n_folds=5 epochs=8 patience=2 &
wait
touch logs/ablations_cv.done
