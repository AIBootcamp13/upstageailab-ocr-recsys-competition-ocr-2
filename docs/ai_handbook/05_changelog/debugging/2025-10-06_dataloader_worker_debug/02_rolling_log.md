# Dataloader Debug Rolling Log

- **Session folder:** `run_summary_2025-10-06T122125Z_dataloader_worker_debug`
- **Timezone:** UTC
- **Legend:**
  - `status`: ✅ success, ⚠️ partial, ❌ failure, ⏸ blocked
  - `owner`: initials or agent handle executing the step

| timestamp (UTC) | status | owner | action | details & evidence | follow-up |
|-----------------|--------|-------|--------|---------------------|-----------|
| 2025-10-06T12:21:25Z | ⚠️ | Copilot | Session bootstrap | Created debugging folder per handbook schema guidance (run_summary + timestamp). Logged prior context from last session to anchor investigation. | Draft detailed plan (`01_debugging_plan.md`) and schedule first reproduction run with HYDRA_FULL_ERROR override. |
| 2025-10-06T21:34:21Z | ❌ | Copilot | Reproduced short train probe (user run) | Training aborted after validation sanity check while saving last checkpoint. `UniqueModelCheckpoint.format_checkpoint_name` attempted `metrics.get("step")` but dataloader worker exited (pid 4163604) with exit code 1; details lost due to multiprocessing. Manual Ctrl+C triggered graceful shutdown; Lightning created `.FAILURE` sentinel. Full command and traceback captured in shared console log (see session transcript). | Do **not** rerun training. Collect stack trace by single-worker dataset probe (`num_workers=0`) or offline dataset inspection instead of full trainer. |
