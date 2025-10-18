# **Agent Quick Reference: Streamlit Debugging**

## **🚨 CRITICAL: Context Building Protocol**
**Problem:** "The app crashes" or "Inference fails" provides insufficient context for effective debugging.

**Solution:** Follow the [Streamlit Debugging Protocol](../02_protocols/components/22_streamlit_debugging_protocol.md) for comprehensive context.

## **Essential Commands for Streamlit Debugging**

### **1. Start UI with Full Logging (MANDATORY)**
```bash
# ✅ CORRECT: Enables logging automatically
make serve-inference-ui

# ❌ WRONG: No logging, debugging impossible
uv run streamlit run ui/inference_ui.py --server.port=8501
```

### **2. View Logs Immediately**
```bash
# View recent logs (default: last 50 lines)
make logs-inference-ui

# View more lines
make logs-inference-ui -- --lines 200

# Follow logs in real-time (use during reproduction)
make logs-inference-ui -- --follow
```

### **3. Initialize Context Logging**
```bash
# Start structured logging for the session
make context-log-start LABEL="streamlit_inference_debug"

# Document EVERY action and finding
# (logs are automatically created in logs/agent_runs/)
```

### **4. Capture Complete Context**
**Required Information for Any Streamlit Issue:**
```bash
# UI Status
make status-inference-ui
make list-ui-processes

# System Information
hostname && python --version && nvidia-smi --query-gpu=name --format=csv

# Configuration
cat configs/ui/inference.yaml
cat configs/train.yaml | grep -A 10 "model:"

# Recent Logs
make logs-inference-ui -- --lines 100
```

## **Common Streamlit Error Patterns**

### **"Inference failed: Inference engine returned no results"**
**Check:**
```bash
# Model loading issues
grep -i "checkpoint\|model\|load" logs/ui/inference_8501.err

# CUDA/GPU problems
grep -i "cuda\|gpu\|memory" logs/ui/inference_8501.err

# Data processing errors
grep -i "preprocess\|transform\|validate" logs/ui/inference_8501.err
```

### **UI Crashes or Freezes**
**Check:**
```bash
# Python exceptions
grep -i "traceback\|exception\|error" logs/ui/inference_8501.err

# Streamlit errors
grep -i "streamlit" logs/ui/inference_8501.err

# Session state issues
grep -i "session\|state" logs/ui/inference_8501.err
```

### **Performance Issues**
**Check:**
```bash
# Memory usage
grep -i "memory\|ram\|vram" logs/ui/inference_8501.out

# Timing information
grep -i "time\|duration\|speed" logs/ui/inference_8501.out

# Resource monitoring
make serve-resource-monitor
```

## **Structured Problem Reporting**

### **Template for Agent Handoff**
```markdown
## Problem Summary
**Issue:** [One sentence - e.g., "Inference UI returns 'no results' for valid checkpoint"]
**Impact:** [What fails - e.g., "Users cannot perform OCR inference"]
**Environment:** [System details, versions]

## Reproduction Steps
1. Start UI: `make serve-inference-ui`
2. Upload image: [filename]
3. Select model: [model details from UI]
4. Set parameters: [threshold, candidates, etc.]
5. Click "Run Inference"
6. **Expected:** OCR results displayed
7. **Actual:** "❌ Inference failed: Inference engine returned no results."

## Diagnostic Information
**Logs:** [Relevant excerpts from `make logs-inference-ui`]
**Configuration:** [UI and model config details]
**System:** [GPU, memory, Python versions]

## Investigation History
**Hypotheses Tested:** [List with results]
**Commands Run:** [All debugging commands executed]
**Findings:** [Key discoveries from logs/config]
```

## **Debugging Workflow**

### **Step 1: Initial Triage**
```bash
# Start with logging
make serve-inference-ui

# Check basic status
make status-inference-ui

# View recent activity
make logs-inference-ui
```

### **Step 2: Problem Classification**
```bash
# Determine error type from logs
grep -i "error\|exception\|failed" logs/ui/inference_8501.err

# Check system resources
nvidia-smi && free -h

# Validate configuration
uv run python scripts/agent_tools/validate_config.py --config-name inference
```

### **Step 3: Hypothesis Testing**
```bash
# Document each test
echo "HYPOTHESIS: [Clear statement]
TEST: [Specific command/action]
EXPECTED: [Expected result]
ACTUAL: [Actual result]"

# Example
echo "HYPOTHESIS: Checkpoint architecture mismatch
TEST: uv run python scripts/debug/checkpoint_inspector.py /path/to/model.ckpt
EXPECTED: Architecture matches UI expectations
ACTUAL: $(uv run python scripts/debug/checkpoint_inspector.py /path/to/model.ckpt)"
```

### **Step 4: Context Preservation**
```bash
# Summarize findings
make context-log-summarize

# Create debug bundle if needed
mkdir debug_$(date +%Y%m%d_%H%M%S)
cp logs/ui/inference_*.log debug_$(date +%Y%m%d_%H%M%S)/
cp configs/ui/inference.yaml debug_$(date +%Y%m%d_%H%M%S)/
```

## **Key Principles for Agents**

1. **Never debug without logs** - Always use `make serve-inference-ui`
2. **Capture context immediately** - Logs change, errors are transient
3. **Document everything** - Use context logging for all actions
4. **Provide complete reproduction steps** - Include exact UI interactions
5. **Check system state** - GPU, memory, versions are critical
6. **Follow the protocol** - Use structured approaches, not ad-hoc debugging

## **Reference Protocols**
- **[Streamlit Debugging Protocol](../02_protocols/components/22_streamlit_debugging_protocol.md)** - Complete context building guide
- **[Context Logging Protocol](../02_protocols/development/06_context_logging.md)** - Structured action logging
- **[Iterative Debugging Protocol](../02_protocols/development/07_iterative_debugging.md)** - Systematic investigation
- **[Debugging Workflow Protocol](../02_protocols/development/03_debugging_workflow.md)** - Initial triage and tools
