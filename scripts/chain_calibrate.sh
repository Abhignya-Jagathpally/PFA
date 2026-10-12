#!/usr/bin/env bash
# Waits (60 s polls, at most 12 h) for runs/stacker_tv/<run>/result.json, then runs the isotonic calibration
# (src/cafa6/calibrate_tv.py --method iso, CPU only) on that stacker run and appends a wake line to
# logs/stack_chain_<run>.log. result.json is the last file stacker_tv writes, so pred_tv_oof_*.npz exist by then.
# Usage: scripts/chain_calibrate.sh <run>      e.g. scripts/chain_calibrate.sh l3_c0
set -u
run=${1:?usage: scripts/chain_calibrate.sh <run>}
cd /home/aj0486@students.ad.unt.edu/PFA/cafa6
end=$((SECONDS + 43200))
while [ $SECONDS -lt $end ] && [ ! -f "runs/stacker_tv/$run/result.json" ]; do
  sleep 60
done
mkdir -p runs/calibrate_tv logs
if [ -f "runs/stacker_tv/$run/result.json" ]; then
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= ../bin/python -m cafa6.calibrate_tv --method iso \
    --stacker-run "runs/stacker_tv/$run" --name "${run}_iso" --threads 16 \
    > "runs/calibrate_tv/${run}_iso.stdout" 2>&1
  code=$?
else
  echo "timed out after 12 h waiting for runs/stacker_tv/$run/result.json" > "runs/calibrate_tv/${run}_iso.stdout"
  code=124
fi
echo "AGENT_LOOP_WAKE_stack calib $run exit $code" >> "logs/stack_chain_$run.log"
