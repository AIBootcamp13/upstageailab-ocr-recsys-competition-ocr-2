# **filename: docs/ai_handbook/02_protocols/05_iterative_debugging.md**

# **Protocol: Iterative Debugging and Root Cause Analysis**

This protocol is to be used when a code change introduces a complex bug that is not immediately obvious and requires a systematic investigation. It is designed to manage context efficiently and produce a valuable summary, even if the root cause is not found.

It complements the standard "Context Logging" protocol but is specific to a single, focused debugging session.

## **Phase 1: Isolate the Regression with git bisect**

Before manual investigation, you must first attempt to automatically find the exact commit that introduced the bug. This is the most efficient way to pinpoint a regression.

1. **Identify a "bad" commit:** This is typically the current HEAD where the bug is present.
2. **Identify a "good" commit:** Find a recent commit hash where the bug did not exist.
3. **Identify a test command:** Find a reliable command that can automatically detect the bug (e.g., a specific pytest command from the Command Registry that now fails).
4. **Execute the git bisect process:**
   `git bisect start`
   `git bisect bad <bad_commit_hash>`
   `git bisect good <good_commit_hash>`
   # The agent will now be in a bisect session.
   # For each step, run the test command:
   `uv run pytest tests/path/to/failing_test.py`
   # If the test fails, run:
   `git bisect bad`
   # If the test passes, run:
   `git bisect good`

5. **Conclusion:** Continue this process until git identifies the first bad commit. This is often the complete solution. If it is, log this finding in an experiment summary and conclude. If git bisect is not feasible or does not reveal the cause, proceed to Phase 2.

## **Phase 2: The Structured Debugging Log**

If the root cause is more complex than a single bad commit, you will start a dedicated debugging log for this session.

* **File Location:** `logs/debugging_sessions/<YYYY-MM-DD_HH-MM-SS>_debug.jsonl`

This log is **not** for every single action. It is for recording the **scientific method** of your debugging: hypothesis, test, observation, conclusion.

For each hypothesis you test, you must log a single JSON object to the _debug.jsonl file with the following schema:

{
  "timestamp": "2025-09-28T16:00:00Z",
  "hypothesis": "I believe the channel mismatch error is caused by the UNet decoder's `output_channels` not matching the DBNet head's `in_channels`.",
  "test_action": {
    "type": "code_modification",
    "file": "configs/preset/models/decoder/unet.yaml",
    "change": "Set `output_channels` to 256."
  },
  "observation": "After modifying the config and re-running the test, the channel mismatch `RuntimeError` disappeared, but a new `CUDA out of memory` error occurred.",
  "conclusion": "The hypothesis was correct about the channel mismatch, but this has revealed a downstream memory issue. The new hypothesis is that the larger feature maps from the corrected decoder are too large for the current batch size."
}

## **Phase 3: Generate the Debugging Summary (RCA)**

At the end of your debugging session—whether you succeed, fail, or exhaust your attempts—you **must** generate a summary of your investigation.

Execute the new summarization tool (which will be added to the Command Registry):

`uv run python scripts/agent_tools/summarize_debugging_log.py --log-file <path_to_your_debug_log.jsonl>`

This script will read your structured debugging log and generate a concise, human-readable Markdown summary of the entire investigation.

* **Output Location:** docs/ai_handbook/04_experiments/debug_summary_<timestamp>.md.

## **Phase 4: Using the Summary for Hand-off and Context**

This generated debug_summary.md is now the single most important artifact from your session.

* **If you solved the bug:** The summary serves as a permanent record of the root cause and the fix.
* **If you failed to solve the bug:** When escalating to a human, provide **only this summary**. It contains the full narrative of your investigation in a condensed format, allowing a human to get up to speed in seconds.
* **For the next session:** This summary becomes the primary context for you or another agent to continue the investigation, ensuring no prior work is lost or repeated.
