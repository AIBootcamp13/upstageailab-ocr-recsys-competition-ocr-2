# **filename: docs/ai_handbook/02_protocols/development/07_iterative_debugging.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=debugging,root_cause_analysis,regression_testing -->

# **Protocol: Iterative Debugging and Root Cause Analysis**

## **Overview**
This protocol provides systematic investigation methodology for complex bugs that require methodical hypothesis testing and root cause analysis. It complements standard debugging workflows by focusing on structured investigation, efficient context management, and valuable documentation even when issues remain unresolved.

## **Prerequisites**
- Clear reproduction case for the bug
- Access to git repository with commit history
- Understanding of hypothesis-testing debugging approach
- Familiarity with project testing framework and validation scripts
- Knowledge of the debugging log format and summarization tools

## **Procedure**

### **Step 1: Isolate Regression with Git Bisect**
Automatically identify the exact commit that introduced the bug before manual investigation:

**Identify Commits:**
- "Bad" commit: Current HEAD where bug is present
- "Good" commit: Recent commit where bug did not exist

**Find Test Command:**
- Locate reliable automated test that detects the bug
- Use commands from Command Registry (e.g., specific pytest commands)

**Execute Bisect Process:**
```bash
git bisect start
git bisect bad <bad_commit_hash>
git bisect good <good_commit_hash>

# For each step, run test and mark result:
uv run pytest tests/path/to/failing_test.py

# If test fails: git bisect bad
# If test passes: git bisect good
```

**Evaluate Results:**
- If bisect identifies the first bad commit, log finding and conclude
- If bisect is inconclusive, proceed to structured debugging

### **Step 2: Conduct Structured Hypothesis Testing**
Create dedicated debugging log for systematic investigation when bisect fails:

**Initialize Debug Log:**
- Location: `logs/debugging_sessions/<YYYY-MM-DD_HH-MM-SS>_debug.jsonl`
- Focus: Scientific method recording, not every action

**Hypothesis Testing Format:**
```json
{
  "timestamp": "2025-09-28T16:00:00Z",
  "hypothesis": "Channel mismatch error caused by UNet decoder output_channels not matching DBNet head in_channels",
  "test_action": {
    "type": "code_modification",
    "file": "configs/preset/models/decoder/unet.yaml",
    "change": "Set output_channels to 256"
  },
  "observation": "Channel mismatch RuntimeError disappeared, but CUDA out of memory error occurred",
  "conclusion": "Hypothesis correct about channel mismatch, revealed downstream memory issue with larger feature maps"
}
```

**Testing Methodology:**
- Formulate clear, testable hypotheses
- Execute minimal changes to validate each hypothesis
- Record observations objectively
- Draw conclusions that lead to next hypothesis

### **Step 3: Generate Root Cause Analysis Summary**
Create comprehensive documentation of the debugging investigation:

**Run Summarization Tool:**
```bash
uv run python scripts/agent_tools/summarize_debugging_log.py --log-file <path_to_debug_log.jsonl>
```

**Summary Generation:**
- Reads structured debugging log entries
- Produces concise, human-readable Markdown summary
- Output: `docs/ai_handbook/04_experiments/debug_summary_<timestamp>.md`

### **Step 4: Utilize Summary for Context and Hand-off**
Leverage the debugging summary as the primary artifact for knowledge transfer:

**For Successful Resolution:**
- Summary serves as permanent record of root cause and fix
- Documents complete investigation narrative
- Enables future reference and similar issue prevention

**For Escalation:**
- When investigation exhausts attempts, provide only the summary
- Contains full investigation narrative in condensed format
- Allows rapid knowledge transfer to human experts

**For Continuation:**
- Summary becomes primary context for resuming investigation
- Prevents work duplication across sessions
- Maintains investigation continuity

## **Validation**
- Git bisect attempted and results documented (successful or inconclusive)
- Debug log contains structured hypothesis-testing entries
- Each hypothesis includes clear test action, observation, and conclusion
- Summary generation completes successfully
- Summary provides comprehensive investigation overview
- Investigation follows systematic, documented approach

## **Troubleshooting**
- If git bisect cannot find suitable good/bad commits, proceed directly to hypothesis testing
- When hypotheses are too broad, break them into smaller, testable components
- If debug log becomes too verbose, focus on key hypothesis transitions
- When summary generation fails, verify log format and tool access
- For complex multi-factor bugs, consider separate investigations for each factor

## **Related Documents**
- [Debugging Workflow](03_debugging_workflow.md) - General debugging techniques and tools
- [Context Logging](06_context_logging.md) - Session logging and summarization
- [Context Checkpointing](08_context_checkpointing.md) - Advanced context management
- [Command Registry](02_command_registry.md) - Available testing and debugging tools
