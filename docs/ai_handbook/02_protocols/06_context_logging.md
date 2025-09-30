# **filename: docs/ai_handbook/02_protocols/04_context_logging.md**

# **Protocol: Context Logging and Summarization**

To ensure that your actions are observable and can be used as context for future tasks without overloading the context window, you must follow this logging and summarization protocol.

> **What does “wrap-up” mean?** In this protocol, wrap-up refers to the end of an agent work session (the period of active coding or debugging during a conversation). Long-running training experiments should still be captured with the [experiment template](../04_experiments/TEMPLATE.md); you can link the experiment record from your context summary when needed.

## **1. The Principle: Log Everything, Summarize for Context**

* **Logging:** Every significant action you take must be recorded in a structured JSON log file. This provides a detailed, machine-readable audit trail.
* **Summarization:** Raw logs are too verbose for context. After every run, you will use a tool to generate a concise Markdown summary of the log. **This summary is the primary artifact used for context.**

## **2. Structured Logging**

For each run, a log file will be created at `logs/agent_runs/<YYYY-MM-DD_HH-MM-SS>.jsonl`.

Each action you take should be logged as a new line in this file. Each line is a JSON object with the following schema:

```json
{
  "timestamp": "2025-09-28T15:30:00Z",
  "action": "execute_script",
  "parameters": {
    "script_name": "scripts/agent_tools/validate_config.py",
    "args": ["--path", "configs/experiments/new_decoder.yaml"]
  },
  "thought": "The user asked me to validate a new config. I will use the 'validate_config.py' tool from the command registry to check for errors before proceeding.",
  "outcome": "success",
  "output_snippet": "Validation successful. No errors found."
}
```

A helper function (`log_agent_action`) ships in `scripts/agent_tools/context_log.py`; see the helper tooling section below.

## **3. Generating and Using Summaries**

At the end of your run (whether it succeeds or fails), you **must** call the summarization script.

### **3.1. How to Generate a Summary**

Execute the following command (or use the `make context-log-summarize` shortcut described below):
```bash
uv run python scripts/agent_tools/context_log.py summarize --log-file <path_to_your_log_file.jsonl>
```
This command will:

1. Read the structured log file.
2. Use an LLM to generate a concise summary.
3. Save the summary as a Markdown file in `docs/ai_handbook/04_experiments/run_summary_<timestamp>.md`.

### **3.2. How to Use Summaries**

Summaries are the key to effective, long-term context.

* **For Debugging:** When a task fails, your first step is to locate the summary of the failed run. It will provide the most efficient overview of what went wrong.
* **For Multi-step Tasks:** If you are continuing a multi-part task, the summary from the previous part should be used as your primary context.

## **4. Helper Tooling**

To simplify adoption, the `scripts/agent_tools/context_log.py` CLI exposes three subcommands:

| Command | Purpose |
| --- | --- |
| `uv run python scripts/agent_tools/context_log.py start [--label my-task]` | Creates `logs/agent_runs/<timestamp>[_my-task].jsonl` and prints its path. |
| `uv run python scripts/agent_tools/context_log.py log --log-file <path> --action ...` | Manually append an entry (useful for shell-only workflows). |
| `uv run python scripts/agent_tools/context_log.py summarize --log-file <path>` | Generates the Markdown summary via the LLM helper. |

Shortcuts exist in the `Makefile`:

```bash
make context-log-start LABEL="streamlit-maintenance"
make context-log-summarize LOG=logs/agent_runs/2025-09-30_18-00-00_streamlit-maintenance.jsonl
```

Both commands appear in `make help` and are listed in the Command Registry for easy agent invocation.

### Environment defaults

- `.env` (checked into git) defines non-sensitive defaults such as `AGENT_CONTEXT_LOG_DIR` and an optional blank `AGENT_CONTEXT_LOG_LABEL`.
- `.env.local` (ignored by git) should contain private keys and, if desired, a default label override (for example `AGENT_CONTEXT_LOG_LABEL="streamlit-maintenance"`). Copy `.env.template` to `.env.local` and fill in the placeholders.
- When you invoke `context_log.py start` without `--label`, the tool reads `AGENT_CONTEXT_LOG_LABEL` from the environment. This makes it easy to set a per-session label once (export it or place it in `.env.local`).
