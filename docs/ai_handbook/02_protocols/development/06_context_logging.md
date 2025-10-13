# **filename: docs/ai_handbook/02_protocols/development/06_context_logging.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=context_logging,summarization,agent_workflow -->

# **Protocol: Context Logging and Summarization**

## **Overview**
This protocol ensures agent actions are observable and can be used as context for future tasks without overloading context windows. It establishes structured logging and summarization practices for maintaining comprehensive audit trails and efficient knowledge transfer.

## **Prerequisites**
- Access to project logging infrastructure and scripts
- Understanding of JSON logging format and Markdown summaries
- Familiarity with the context logging CLI tools
- Environment configuration (`.env` and `.env.local` files)

## **Procedure**

### **Step 1: Initialize Context Logging**
Start structured logging at the beginning of any significant work session:

**Create Log File:**
```bash
uv run python scripts/agent_tools/context_log.py start [--label my-task]
# Or use Makefile shortcut:
make context-log-start LABEL="my-task"
```

This creates `logs/agent_runs/<timestamp>[_my-task].jsonl` and prints the path for reference.

**Configure Environment:**
- `.env` contains non-sensitive defaults like `AGENT_CONTEXT_LOG_DIR`
- `.env.local` contains private keys and optional default labels
- Set `AGENT_CONTEXT_LOG_LABEL` for per-session consistency

### **Step 2: Log All Significant Actions**
Record every meaningful action in structured JSON format throughout the session:

**Log Format:**
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

**Use Helper Function:**
```python
from scripts.agent_tools.context_log import log_agent_action

log_agent_action(
    log_file_path="logs/agent_runs/2025-09-28_15-30-00.jsonl",
    action="execute_script",
    parameters={"script_name": "validate_config.py", "args": ["--config-name", "train"]},
    thought="Validating training configuration before starting experiment",
    outcome="success",
    output_snippet="Configuration validated successfully"
)
```

### **Step 3: Generate Session Summary**
Create concise Markdown summary at the end of each work session:

**Run Summarization:**
```bash
uv run python scripts/agent_tools/context_log.py summarize --log-file <path_to_log_file.jsonl>
# Or use Makefile shortcut:
make context-log-summarize LOG=<path_to_log_file.jsonl>
```

**Summary Process:**
- Reads the structured JSON log file
- Uses LLM to generate concise Markdown summary
- Saves as `docs/ai_handbook/04_experiments/run_summary_<timestamp>.md`
- Provides efficient overview for future context

### **Step 4: Utilize Summaries for Context**
Leverage generated summaries for efficient knowledge transfer and debugging:

**For Debugging:**
- Locate summary of failed runs for efficient problem diagnosis
- Use summary as primary context when investigating issues
- Reference specific actions and outcomes without full log verbosity

**For Multi-step Tasks:**
- Use previous session summaries as primary context for continuation
- Link experiment records to context summaries when applicable
- Maintain continuity across long-running development efforts

## **Validation**
- Log file created and contains structured JSON entries for all significant actions
- Each log entry includes timestamp, action, parameters, thought process, and outcome
- Summary generation completes successfully and produces readable Markdown
- Summaries provide sufficient context for understanding session outcomes
- Environment configuration properly set up for logging workflow

## **Troubleshooting**
- If log file creation fails, check environment variables and directory permissions
- When summary generation fails, verify LLM access and log file format
- For missing context, ensure all significant actions were logged during session
- If summaries are too verbose, review logging granularity and focus on key actions
- When debugging past sessions, start with summary before diving into raw logs

## **Related Documents**
- [Command Registry](02_command_registry.md) - Available logging and context tools
- [Experiment Template](../../experiments/TEMPLATE.md) - Long-running experiment documentation
- [Context Checkpointing](08_context_checkpointing.md) - Advanced context management
- [Iterative Debugging](07_iterative_debugging.md) - Debugging workflow integration
