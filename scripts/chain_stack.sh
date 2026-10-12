#!/usr/bin/env bash
# Usage: scripts/chain_stack.sh <stack_name> <run_dir> <done_file>
# Waits (max 16 h) for <done_file> and the run's TV/val/test predictions, then runs the TV stacker and the isotonic
# calibration with the train + val kNN priors. Progress lines go to logs/stack_chain_<stack_name>.log; the
# "AGENT_LOOP_WAKE_stack" lines are what the loop watches.
set -u
cd "$(dirname "$0")/.."
name=$1; run=$2; done_file=$3
log=logs/stack_chain_${name}.log
export PYTHONPATH=src CAFA6_KNN_DIR="$PWD/data/proc_gogpt_tv2/knn_trainval"
end=$((SECONDS + 57600))
echo "$(date '+%F %T') waiting for $done_file and $run/pred_{tv_all,val,test}_top500.npz" >> "$log"
while [ $SECONDS -lt $end ] && [ ! -f "$done_file" ]; do sleep 60; done
for s in tv_all val test; do
  if [ ! -f "$run/pred_${s}_top500.npz" ]; then
    echo "AGENT_LOOP_WAKE_stack $name missing $run/pred_${s}_top500.npz" >> "$log"; exit 1
  fi
done
echo "$(date '+%F %T') stacker start (CAFA6_KNN_DIR=$CAFA6_KNN_DIR)" >> "$log"
../bin/python -m cafa6.stacker_tv --higo-run "$run" --out "runs/stacker_tv/$name" --threads 16 \
  > "runs/stacker_tv/$name.stdout" 2>&1
rc=$?
echo "AGENT_LOOP_WAKE_stack $name stacker exit $rc" >> "$log"
[ $rc -eq 0 ] || exit $rc
mkdir -p runs/calibrate_tv
../bin/python -m cafa6.calibrate_tv --method iso --stacker-run "runs/stacker_tv/$name" --name "${name}_iso" \
  > "runs/calibrate_tv/${name}_iso.stdout" 2>&1
echo "AGENT_LOOP_WAKE_stack calib $name exit $?" >> "$log"
