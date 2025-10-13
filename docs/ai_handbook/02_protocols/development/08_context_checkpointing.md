# **filename: docs/ai_handbook/02_protocols/development/08_context_checkpointing.md**
<!-- ai_cue:priority=medium -->
<!-- ai_cue:use_when=context_management,long_running_tasks,performance_optimization -->

# **Protocol: Context Checkpointing & Restoration**

## **Overview**
This protocol provides systematic management for long-running tasks to prevent performance degradation from context window limits. It treats conversations like transactional processes, enabling efficient state preservation and restoration across multiple sessions.

## **Prerequisites**
- Internal token counting capability to monitor context usage
- Access to conversation history or structured action logs
- Understanding of JSON state summary format
- Familiarity with continuation prompt formatting
- Knowledge of logical breakpoints in task workflows

## **Procedure**

### **Step 1: Monitor Context Usage**
Track conversation context to identify checkpoint triggers:

**Context Limit Monitoring:**
- Maintain internal token count of conversation
- Trigger checkpoint when exceeding 80% of model's context window
- Monitor for performance degradation indicators

**Logical Breakpoint Identification:**
- Complete significant sub-tasks in larger projects
- Reach natural stopping points (module completion, phase ends)
- Identify points where fresh context would be beneficial

### **Step 2: Generate State Summary**
Create comprehensive summary of current session state:

**Pause Current Task:**
- Stop forward progress on current work
- Ensure no partial changes are in progress

**Create Summary Object:**
```json
{
  "overall_goal": "High-level objective of entire session",
  "last_completed_task": "Most recent successfully finished work",
  "key_findings": [
    "Important facts, file paths, or conclusions",
    "Critical decisions made",
    "Key files modified or created"
  ],
  "next_immediate_step": "Very next action required to continue"
}
```

**Validate Completeness:**
- Ensure all critical context is captured
- Include file paths, key decisions, and findings
- Specify exact next step for seamless continuation

### **Step 3: Format Continuation Prompt**
Transform state summary into user-ready restoration prompt:

**Structure Continuation Prompt:**
```markdown
**Goal:** [overall_goal]

**Previous Session Summary:**
- **Completed:** [last_completed_task]
- **Key Files:** [relevant file paths]
- **Reference Document:** [applicable documentation]

**Next Step:**
[next_immediate_step]
```

**Save to File:**
- Location: `logs/continuation_prompts/<descriptive_name>.md`
- Include clear instructions for user
- Preserve all critical context elements

### **Step 4: Provision for Restoration**
Deliver checkpoint to user and prepare for new session:

**User Notification:**
```bash
### --- CONTEXT CHECKPOINT CREATED ---

The context window is nearing its limit. To continue this task with optimal performance, please start a new conversation and use the following prompt.

This file has been saved to: logs/continuation_prompts/<filename>.md
```

**Session Transition:**
- Provide complete continuation prompt
- Ensure user understands restoration process
- Maintain work continuity across sessions

## **Validation**
- Context usage monitored and checkpoint triggered appropriately
- State summary captures all essential session information
- Continuation prompt is clear and actionable
- File saved to designated location with proper naming
- User receives complete restoration instructions

## **Troubleshooting**
- If context monitoring is unavailable, use logical breakpoints as primary trigger
- When summary becomes too large, prioritize most critical findings
- If continuation prompt is confusing, test with example restoration
- For complex multi-part tasks, consider intermediate checkpoints
- When performance degradation occurs unexpectedly, implement immediate checkpointing

## **Related Documents**
- [Context Logging](06_context_logging.md) - Session logging and summarization
- [Iterative Debugging](07_iterative_debugging.md) - Complex debugging session management
- [Experiment Template](../../experiments/TEMPLATE.md) - Long-running experiment documentation
- [Command Registry](02_command_registry.md) - Context management tools
