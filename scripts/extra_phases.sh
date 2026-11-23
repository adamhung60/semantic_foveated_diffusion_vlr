# Extra phases (added 2026-11-23 after the curve results): combine the three individually-supported wins
# (Real-ESRGAN init, residual cache at 10 steps, CFG off on last 2) and confirm on all 60 prompts.
phase best60    gen --subset 60 --variants ours_esr_cache_cfgoff2_t95_r10 ours_cache_cfgoff2_t95_r10 ours_esr_t95_r7 || exit 1
phase eval6     evalsum                                                        || exit 1
# Second extra phase (cron agent, 2026-11-23 20:10): second-seed check of the headline claim
# "ours_cache_cfgoff2_t95_r10 ties Full HR 10-step at ~2.1x lower latency" (seed 0, n=60, null-equivalence).
# Same 30 prompts / seed_offset 1 as phase seed1 (drafts cached). ~30-40 min GPU.
phase seed1b    gen --subset 30 --seed_offset 1 --out outputs/abl_seed1 --variants ours_cache_cfgoff2_t95_r10 fhr10 || exit 1
phase eval7     evalsum                                                        || exit 1
