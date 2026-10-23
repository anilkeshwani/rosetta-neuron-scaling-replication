#!/usr/bin/env bash
# Phase 1 GPU schedule: the three small Pythia-OPT pairs on all 4 GPUs, one after
# another, then the 9 random-init null pairs as 4 parallel single-GPU workers.
#
# Token caches come from `run_experiment.py --caches-only` (see logs/prebuild_caches.log);
# each pair waits until its cache build has exited so the two never write the same file.
# NPROC is fixed within each curve: 4 for phase1_pythia_opt (the FSDP pairs need all 4),
# 1 for the null.
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
LOGS=logs

wait_for_cache() {
  until grep -q "^$1 exit=" "$LOGS/prebuild_caches.log" 2>/dev/null; do sleep 30; done
  grep "^$1 exit=" "$LOGS/prebuild_caches.log"
}

for run in pythia-160m__opt-125m pythia-410m__opt-350m pythia-1.4b__opt-1.3b; do
  wait_for_cache "$run"
  NPROC_PER_NODE=4 uv run python scripts/run_experiment.py configs/experiments/phase1_pythia_opt.yaml \
    --only "$run" > "$LOGS/phase1_$run.log" 2>&1
  echo "[phase1] $run exit=$? $(date -Is)"
done

null_worker() {
  local gpu=$1; shift
  for run in "$@"; do
    wait_for_cache "$run"
    CUDA_VISIBLE_DEVICES=$gpu NPROC_PER_NODE=1 uv run python scripts/run_experiment.py \
      configs/experiments/phase1_null_random_init.yaml --only "$run" > "$LOGS/null_$run.log" 2>&1
    echo "[null] gpu$gpu $run exit=$? $(date -Is)"
  done
}

# Largest pairs spread across workers first so the tail is short
null_worker 0 pythia-1.4b-seed0__opt-1.3b-seed0 pythia-160m-seed0__opt-125m-seed0 pythia-160m-seed2__opt-125m-seed2 &
null_worker 1 pythia-1.4b-seed1__opt-1.3b-seed1 pythia-410m-seed0__opt-350m-seed0 &
null_worker 2 pythia-1.4b-seed2__opt-1.3b-seed2 pythia-410m-seed1__opt-350m-seed1 &
null_worker 3 pythia-410m-seed2__opt-350m-seed2 pythia-160m-seed1__opt-125m-seed1 &
wait
echo "[phase1] all done $(date -Is)"
