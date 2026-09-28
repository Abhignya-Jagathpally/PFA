#!/usr/bin/env bash
# Fixed 4-epoch budget: dump ablation predictions, stop CV folds / GO-GPT-benchmark run at 4 epochs, then evaluate.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=../bin/python
stop() { pkill -f "[c]afa6.train .*name=$1( |$)"; }

(for r in abl_no_hier_query abl_no_mcm abl_mean_pool abl_no_taxon abl_no_ia_weight abl_softmax_pool; do
   [ -e runs/$r/pred_test.npy ] || CUDA_VISIBLE_DEVICES=1 $PY -m cafa6.dump_preds --run $r >> logs/dump_ablations.log 2>&1
 done
 $PY -m cafa6.stack --runs dev_35m_lora dev_35m_lora_all_r16 base_35m_frozen_mlp abl_no_hier_query abl_no_mcm \
     abl_mean_pool abl_no_taxon abl_no_ia_weight abl_softmax_pool --out eval_main > logs/stack_main.log 2>&1) &

cap() { while [ "$(grep -c '"epoch"' logs/$1.log)" -lt 4 ]; do sleep 30; done; stop $1; }
(for f in 2 3 4; do cap cv_35m_fold$f & done; wait
 $PY -m cafa6.cv_summary > logs/cv_summary.log 2>&1) &
(cap gogptbench_35m_lora; sleep 5
 CAFA6_PROC=data/proc_gogpt CUDA_VISIBLE_DEVICES=1 $PY -m cafa6.dump_preds --run gogptbench_35m_lora > logs/dump_gogptbench.log 2>&1
 CAFA6_PROC=data/proc_gogpt CUDA_VISIBLE_DEVICES=1 $PY -m cafa6.bench_gogpt --runs gogptbench_35m_lora --out bench_gogpt > logs/bench_gogpt.log 2>&1) &
wait
touch logs/finish_fast.done
