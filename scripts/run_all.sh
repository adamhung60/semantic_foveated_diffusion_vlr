#!/usr/bin/env bash
# Unattended driver for the T4 ablation study. Every phase is resumable; finished phases leave logs/phase_<name>.done.
# Started (and restarted if it dies) by scripts/watchdog.sh from cron, inside tmux session "exp".
cd "$(dirname "$0")/.." || exit 1
source env.sh
mkdir -p logs results/ablations
status() { echo "$(date '+%F %T') $*" | tee -a logs/status.txt; }

phase() {  # phase <name> <command...>
  local name=$1; shift
  [ -f "logs/phase_${name}.done" ] && return 0
  status "START $name"
  if "$@" >> "logs/${name}.log" 2>&1; then
    touch "logs/phase_${name}.done"; status "DONE  $name"
  else
    status "FAIL  $name (exit $?) -- see logs/${name}.log"; return 1
  fi
}
evalsum() {  # evaluate everything new, then refresh SUMMARY.md / RESULTS.md section / plot
  $PY scripts/eval_ablations.py --root outputs/abl --csv results/ablations/metrics.csv &&
  { [ ! -d outputs/abl_seed1 ] || $PY scripts/eval_ablations.py --root outputs/abl_seed1 --csv results/ablations/metrics_seed1.csv; } &&
  $PY scripts/summarize_ablations.py > /dev/null
}
gen() { $PY scripts/run_ablations.py "$@"; }

set -o pipefail
phase core      gen --group core --subset 30                                   || exit 1
phase eval1     evalsum                                                        || exit 1
phase curve     gen --group curve --subset 30                                  || exit 1
phase eval2     evalsum                                                        || exit 1
phase fhr28     gen --group fhr28 --subset 30                                  || exit 1
phase eval3     evalsum                                                        || exit 1
# extend the key comparisons to all 60 prompts
phase full60    gen --subset 60 --variants draft ours_t95_r7 fhr7 fhr10 allhr_fov_t95_r7 allhr_nofov_t95_r7 \
                    mask_random_t95_r7 mask_none_t95_r7 ours_esr_t95_r7 ours_cache_t95_r10 ours_t95_r10 || exit 1
phase eval4     evalsum                                                        || exit 1
# second seed on 30 prompts for the headline pairs
phase seed1     gen --subset 30 --seed_offset 1 --out outputs/abl_seed1 --variants draft ours_t95_r7 fhr7 \
                    allhr_fov_t95_r7 allhr_nofov_t95_r7 mask_random_t95_r7 ours_esr_t95_r7 ours_cache_t95_r10 || exit 1
phase eval5     evalsum                                                        || exit 1
[ -f scripts/extra_phases.sh ] && source scripts/extra_phases.sh
status "ALL_DONE"
touch logs/ALL_DONE
