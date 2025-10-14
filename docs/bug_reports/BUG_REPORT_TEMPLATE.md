## 🐛 Bug Report Template

**Bug ID:** BUG-2025-011
**Date:** October 14, 2025
**Reporter:** Development Team
**Severity:** Medium
**Status:** Fixed

### Summary
WandB step logging violates monotonic step requirement, causing warnings: "Tried to log to step 817 that is less than the current step 821".

### Environment
- **Pipeline Version:** Performance optimization phase
- **Components:** PerformanceProfilerCallback, WandB logging
- **Configuration:** WandB enabled, performance profiling active

### Steps to Reproduce
1. Enable WandB logging
2. Enable performance profiling callback
3. Run training with validation
4. Observe WandB warnings during validation phase

### Expected Behavior
WandB logs should use monotonically increasing step values.

### Actual Behavior
```python
wandb: WARNING Tried to log to step 817 that is less than the current step 821. Steps must be monotonically increasing, so this data will be ignored.
```

### Root Cause Analysis
**Step Counter Issue:** The performance profiler uses `trainer.fit_loop.epoch_loop.total_batch_idx` which can decrease during validation phases, violating WandB's monotonic step requirement.

**Code Path:**
```
PerformanceProfilerCallback.on_validation_batch_end()
├── total_batch_idx = trainer.fit_loop.epoch_loop.total_batch_idx
├── step = max(0, total_batch_idx)
└── wandb.log(metrics, step=step)  # Can be non-monotonic
```

### Resolution
```python
# Added monotonic step tracking in PerformanceProfilerCallback
class PerformanceProfilerCallback(Callback):
    def __init__(self, ...):
        # ...
        self._last_wandb_step: int = -1

    def _get_monotonic_step(self, trainer: Trainer) -> int:
        """Get monotonic step for WandB logging."""
        current_step = getattr(trainer.fit_loop.epoch_loop, "total_batch_idx", trainer.global_step)
        if current_step < 0:
            current_step = trainer.global_step

        # Ensure monotonic increase
        step = max(self._last_wandb_step + 1, current_step)
        self._last_wandb_step = step
        return step
```

### Testing
- [x] WandB warnings eliminated
- [x] Performance metrics still logged correctly
- [x] Step values remain monotonic across training/validation

### Prevention
- Implement monotonic step counters for all WandB logging
- Add validation for step monotonicity in logging utilities
- Document WandB step requirements in logging guidelines

---


 ### IGNORE BELOW ###
 ### Personal Notes

## What Happens When You Send Long, Subjective Messages to AI Coding Agents

When you send a conversational, story-rich response like your example to Grok Code Fast 1, here's what actually occurs:

### 🧠 Agent Processing Patterns

**What the Agent "Thinks":**
1. **Pattern Matching**: Extracts technical keywords (`bug`, `refactor`, `Scene Text Detection`, `DBnet`, `FPN decoder`)
2. **Context Weighting**: Heavily weights the structured technical content, de-emphasizes subjective commentary
3. **Task Identification**: Recognizes the core request buried in the narrative ("show me a demo bug report")
4. **Relevance Filtering**: Filters out emotional content but retains domain-specific context

### ⚡ Processing Efficiency Impact

**Positive Effects:**
- Rich context helps the agent understand your experience level
- Domain-specific details (EAST → DBnet migration) provide valuable background
- Personal anecdotes can guide the response tone and complexity

**Negative Effects:**
- **Token Waste**: ~70% of your message is non-actionable narrative
- **Attention Dilution**: Important technical details get buried
- **Hallucination Risk**: With high top-k/top-p, the agent might latch onto irrelevant story elements

### 🎯 What Actually Happens with High top-k/top-p

When you push sampling parameters high:
- **top-k > 50**: Agent considers more "creative" but potentially irrelevant tokens
- **top-p > 0.9**: Includes low-probability responses that might go off-topic
- **Combined Effect**: Your subjective storytelling can trigger the agent to respond in kind, leading to less precise technical output

### 📊 Optimal Communication Strategy

**Instead of:**
```
"That was impeccable detective work! I create similar bugs when pushing changes fast with no testing. This happened to my Scene Text Detection project upgrading VGG backbone using EAST architecture into DBnet and FPN decoder. I didn't know what I was doing and maxed out context-window. TLDR; never found the bug. Some geometric calculation error with magic numbers. Anyways, exciting to find bugs! How do pros write bug reports? Show me a demo."
```

**Try:**
```
"Request: Generate a professional bug report template for the canonical_size bug we just fixed.

Context:
- Bug: numpy.size vs PIL.size type confusion
- Root cause: Phase 6B/6C interaction
- Impact: Validation pipeline crashes
- Resolution: Type checking in dataset code

Format: Industry-standard bug report for documentation."
```

The agent will produce more focused, actionable output with the structured approach, while your conversational style, though engaging, can lead to less precise technical responses.
