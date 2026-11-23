#!/usr/bin/env bash
# cron (every 15 min): keep the ablation driver alive in tmux session "exp". Safe to run any time.
REPO=/home/ubuntu/semantic_foveated_diffusion_vlr
cd $REPO || exit 1
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
log() { echo "$(date '+%F %T') $*" >> logs/watchdog.log; }
[ -f logs/ALL_DONE ] && [ ! -f logs/RERUN ] && exit 0
[ -f logs/HOLD ] && { log "HOLD present, not restarting"; exit 0; }
if ! nvidia-smi >/dev/null 2>&1; then
  log "GPU not visible, reloading nvidia modules"
  sudo modprobe -r nouveau 2>/dev/null; sudo modprobe nvidia && sudo modprobe nvidia_uvm && sudo modprobe nvidia_modeset
  sudo nvidia-smi -pm 1 >/dev/null 2>&1
fi
if ! pgrep -f "bash scripts/run_all.sh" >/dev/null; then
  tmux has-session -t exp 2>/dev/null || tmux new-session -d -s exp -n main
  tmux kill-window -t exp:driver 2>/dev/null
  tmux new-window -d -t exp -n driver "bash scripts/run_all.sh; echo DRIVER_EXITED; sleep 600"
  rm -f logs/RERUN
  log "driver (re)started"
fi
