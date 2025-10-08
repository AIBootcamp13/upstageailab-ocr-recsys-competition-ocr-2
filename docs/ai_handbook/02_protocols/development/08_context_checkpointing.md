# **filename: docs/ai_handbook/02_protocols/06_context_checkpointing.md**

# **Protocol: Context Checkpointing & Restoration**

This protocol provides a systematic method for managing long-running tasks and avoiding performance degradation due to context window limits. Its purpose is to automate the process of "saving your work" and starting a fresh, efficient conversation.

## **The Problem**

* LLM performance degrades as the context window fills up.
* A sliding window approach can silently drop important early context.
* Manually restarting and summarizing conversations is inefficient and error-prone.

## **The Solution: Context Checkpointing**

The solution is to treat a long conversation like a transactional process. At a logical breakpoint, you will **checkpoint** the current state by summarizing it and then provide the user with an automated way to **restore** that state in a new conversation.

### **When to Create a Checkpoint**

You should trigger the checkpointing process under two conditions:

1. **Nearing the Context Limit:** You must maintain an internal token count of the conversation. When this count exceeds a threshold (e.g., **80%** of the model's known context window), you must initiate a checkpoint.
2. **Logical Task Completion:** After completing a significant sub-task in a larger project (e.g., finishing the implementation of one module before starting the next), it is best practice to create a checkpoint.

### **How to Create a Checkpoint**

When a checkpoint is triggered, you must perform the following steps:

1. **Pause the current task.** Do not proceed with the next action.
2. **Generate a State Summary.** Read the conversation history (or your structured action logs) and generate a concise summary object. The summary MUST contain:
   * **overall_goal**: What is the high-level objective of the entire session?
   * **last_completed_task**: What was the last thing that was successfully finished?
   * **key_findings**: A bulleted list of the most important facts, file paths, or conclusions from the completed work.
   * **next_immediate_step**: What is the very next action that needs to be taken?
3. **Generate the Continuation Prompt.** Format the State Summary into a pre-packaged prompt that can be used to start a new conversation.
4. **Provision the Prompt.** Save the Continuation Prompt to a file and output a message to the user, instructing them on how to proceed.

### **Example Workflow**

**1. Context Limit Reached.** The agent determines it has used 85% of its context.

**2. State Summary Generation (Internal).** The agent generates the following JSON object:

```json
{
  "overall_goal": "Refactor the OCR model's decoder to be a plug-and-play component.",
  "last_completed_task": "Successfully created the `BaseDecoder` abstract class in `src/ocr_framework/core/base_decoder.py` and implemented the UNetDecoder.",
  "key_findings": [
    "The BaseDecoder requires a `forward` method and an `output_channels` property.",
    "The existing UNet decoder was migrated successfully.",
    "All existing tests are still passing after the refactor."
  ],
  "next_immediate_step": "Create a new `PANDecoder` class that also inherits from `BaseDecoder`."
}
```

**3. Continuation Prompt Provisioning.** The agent saves the following to `logs/continuation_prompts/continue_decoder_refactor.md` and prints a message to the user:

```bash
### --- CONTEXT CHECKPOINT CREATED ---

The context window is nearing its limit. To continue this task with optimal performance, please start a new conversation and use the following prompt.

This file has been saved to: `logs/continuation_prompts/continue_decoder_refactor.md`

---
**Continuation Prompt:**

**Goal:** Refactor the OCR model's decoder to be a plug-and-play component.

**Previous Session Summary:**
- **Completed:** We successfully created the `BaseDecoder` abstract class and migrated the `UNetDecoder` to use it. All tests are passing.
- **Key Files:** `src/ocr_framework/core/base_decoder.py`
- **Reference Document:** `docs/ai_handbook/03_references/01_architecture.md`

**Next Step:**
Your next task is to implement the `PANDecoder`. Create a new file at `src/ocr_framework/architectures/decoders/pan_decoder.py` and define a `PANDecoder` class that inherits from `BaseDecoder`. I am ready to begin.
```
