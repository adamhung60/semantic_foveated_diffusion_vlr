#!/usr/bin/env bash
# Run a command under hard resource caps, so a runaway job is killed instead of freezing the machine.
#
#   MEM_GB=16 THREADS=4 scripts/run_capped.sh python scripts/run_ablation_benchmark.py --phase base
#
# MEM_GB   hard RAM limit for the whole process tree (no swap); exceeding it kills the job (exit 137)
# THREADS  CPU threads for torch / BLAS / OpenMP
set -euo pipefail

MEM_GB="${MEM_GB:-16}"
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

exec systemd-run --user --scope -q -p "MemoryMax=${MEM_GB}G" -p MemorySwapMax=0 nice -n 10 "$@"
