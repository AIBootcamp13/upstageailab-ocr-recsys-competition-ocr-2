# Dataloader Worker Crash Debug Plan

- **Session start:** 2025-10-06T12:21:25Z
- **Owner:** GitHub Copilot (agent)
- **Location:** `docs/debugging/run_summary_2025-10-06T122125Z_dataloader_worker_debug`
- **Related failure:** `RuntimeError: DataLoader worker ... exited unexpectedly with exit code 1`

## Current Signal
- Crash occurs during PyTorch Lightning validation sanity check when the **training** dataloader spins workers.
- Validation dataloader completes instantly; issue isolated to training pipeline.
- Previous mitigations (W&B throttling, smaller media uploads) no longer the limiting factor.
- Worker termination happens before batch 0 is processed, consistent with a dataset/transform failure or resource spike at worker start.
- 2025-10-06T21:34Z probe reproduced the failure: Lightning entered checkpoint cleanup while the worker (pid 4163604) exited with code 1. `.FAILURE` sentinel created; checkpoint callback attempted to format name without a valid `step` metric because training never advanced.

## Objectives
1. Capture a full worker-side traceback to identify the originating transform/dataset code path.
2. Reproduce and isolate the fault in a single-process context (`num_workers=0`).
3. Determine whether the failure is data-dependent, transform-dependent, or resource-related.
4. Implement instrumentation that survives future sessions (lightweight logging toggles, probes).
5. Produce actionable follow-ups (fix PR, data correction, or upstream issue) or a narrowed root cause for handoff.

### Exit Criteria
- ✅ Full stack trace collected and stored in the rolling log.
- ✅ Fault reproduced with deterministic steps (seed, config overrides documented).
- ✅ Hypothesis validated or narrowed to a single component (dataset file, transform, collate, external resource).
- ✅ Proposed fix or next experiment encoded as a follow-up task in the rolling log.

## Constraints & Guardrails
- Training run may hang; terminate manually after 120s of no progress if worker is silent.
- Do **not** rerun the full trainer for now—the latest attempt already captured its failure mode and adds noise without new signal.
- Keep Hydra overrides minimal and always copy them verbatim into the rolling log when executed.
- Avoid modifying shared configs until root cause is confirmed.

## Hypotheses (Descending Priority)
1. **Transform assumption breakage:** New augmentation/logging path expects RGB tensors but receives grayscale or channel-last arrays.
2. **Dataset artifact:** Specific training sample references missing/corrupt image or annotation, crashing when decoded.
3. **Resource exhaustion:** Worker hits CPU OOM or open-file limit under multiprocessing, leading to signal kill.
4. **W&B side effect:** Residual WandB artifact conversion runs inside dataset pipeline and crashes intermittently.

## Investigation Track
1. ### Reproduce with verbose trace *(completed 2025-10-06T21:34Z)*
   - Latest run already executed the trainer command with `num_workers=4` and captured the failure (see rolling log). No further trainer attempts needed until we have new instrumentation.
   - Outcome: Worker exit persists; checkpoint cleanup raises due to missing `step` metric when formatting last checkpoint name.

2. ### Single-worker isolation *(prefer offline probe)*
   - Instead of full trainer, instantiate the training dataloader with `num_workers=0` in a standalone script or notebook cell.
   - Iterate through the first few batches synchronously to surface the underlying exception without multiprocessing swallowing the trace.
   - If the synchronous pass succeeds, incrementally reintroduce specific transforms/components to pinpoint the failing stage.

3. ### Dataset probe script
   - Create ad-hoc probe in `scripts/debug/` (pending) that instantiates training dataset and calls `__getitem__` on suspicious indices.
   - Log image path, annotation snippet, transform pipeline sequence for each call.
   - Use `limit_train_batches=2` run output to identify failing index offset.

4. ### Augmentation stack audit
   - Inspect configs: `configs/preset/datasets/...` and `configs/transforms`. Verify new transforms added recently.
   - Add try/except instrumentation around transforms to log shapes/dtypes.
   - Consider toggling suspicious transforms via Hydra override (e.g., disable random rotation) to test hypothesis quickly.

5. ### Data integrity check
   - Run checksum/exists validation on first 500 training samples via probe script.
   - For any missing image or malformed polygon JSON, capture details in rolling log.

6. ### Resource monitoring (parallel track)
   - While reproducing, run `htop` (CPU) and `nvidia-smi --loop-ms=500` to observe spikes.
   - If worker dies with SIGKILL and memory spikes, experiment with reduced `batch_size` and `prefetch_factor`.

7. ### Regression confirmation
   - Once a fix hypothesis emerges, rerun short train with original settings to confirm no worker deaths (2 train batches, 1 val batch).
   - Document command + outcome in log.

## Artifacts to Capture
- Commands executed (with overrides).
- Stack traces or tracebacks.
- Dataset sample identifiers involved in failures.
- Transform configuration diffs.
- Resource observations (CPU/GPU utilization snapshots).

## Handover Template (update before ending session)
```
Current focus:
Latest command & result:
Suspected root cause:
Next steps:
Blocking issues:
```

---
Keep the rolling log synchronized whenever new evidence appears. Use concise timestamps (UTC) and note whether actions succeeded, failed, or were aborted.
