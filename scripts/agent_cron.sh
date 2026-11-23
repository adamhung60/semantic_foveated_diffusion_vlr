#!/usr/bin/env bash
# cron (every 2 h): wake a headless Claude Code agent to babysit the study and write up results.
REPO=/home/ubuntu/semantic_foveated_diffusion_vlr
cd $REPO || exit 1
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin HOME=/home/ubuntu
grep -q "STUDY COMPLETE" logs/agent_notes.md 2>/dev/null && [ "$(date +%H)" != "09" ] && exit 0
echo "===== $(date '+%F %T') agent wake-up =====" >> logs/agent.log
flock -n /tmp/sfd_agent.lock timeout 5400 claude -p "$(cat scripts/agent_prompt.md)" \
  --allowedTools "Bash" "Read" "Edit" "Write" "Glob" "Grep" \
  --model opus --max-turns 80 >> logs/agent.log 2>&1
echo "===== $(date '+%F %T') agent exit $? =====" >> logs/agent.log
