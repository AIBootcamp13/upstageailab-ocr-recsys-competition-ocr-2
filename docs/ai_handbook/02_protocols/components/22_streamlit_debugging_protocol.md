# **filename: docs/ai_handbook/02_protocols/components/22_streamlit_debugging_protocol.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=streamlit_debugging,ui_troubleshooting,inference_failures -->

# **Protocol: Streamlit UI Debugging and Context Building**

## **Overview**
This protocol establishes systematic debugging practices for Streamlit UI applications, with special emphasis on building comprehensive context for agents. It addresses the common problem of insufficient problem descriptions by providing structured approaches to capture, document, and communicate debugging context effectively.

## **Prerequisites**
- Process manager installed and configured (`scripts/process_manager.py`)
- Access to UI log files in `logs/ui/`
- Understanding of Streamlit app architecture (entry points, components, services)
- Familiarity with context logging protocol (`06_context_logging.md`)
- Access to UI configuration files in `configs/ui/`

## **Core Problem: Insufficient Context**
**Common Issues:**
- "The app crashes" → Missing: error logs, reproduction steps, environment details
- "Inference fails" → Missing: model details, input data, error traces, configuration
- "UI is slow" → Missing: performance metrics, component profiling, resource usage

**Solution:** Structured context building with mandatory checkpoints

## **Procedure**

### **Step 1: Initialize Comprehensive Context Logging**
Start with full context capture before any debugging actions:

**Initialize Context Log:**
```bash
# Start structured logging with descriptive label
make context-log-start LABEL="streamlit_inference_debug"

# Document initial problem statement with REQUIRED details
echo "PROBLEM: Inference UI returns 'no results' error
CONTEXT: Using checkpoint ep999 hmean0.953, uploaded image drp.en_ko.in_house.selectstar_002281_shadow_filtered.jpg
ENVIRONMENT: $(hostname), Python $(python --version)
REPRODUCTION: Upload image → Select model → Click 'Run Inference' → Error appears"
```

**Capture Baseline State:**
```bash
# Document current UI state
make status-inference-ui
make list-ui-processes

# Capture configuration
cat configs/ui/inference.yaml
cat configs/train.yaml | grep -A 10 "model:"

# Log current git state
git status
git log --oneline -5
```

### **Step 2: Enable Full Logging and Reproduce Issue**
Set up comprehensive logging before reproducing the problem:

**Start UI with Logging:**
```bash
# Stop any existing instances
make stop-inference-ui

# Start with full logging enabled
make serve-inference-ui

# Verify logging is active
ls -la logs/ui/inference_8501.*
```

**Document Reproduction Steps:**
```bash
# Log the exact reproduction sequence
echo "REPRODUCTION STEPS:
1. Open http://localhost:8501
2. Upload: drp.en_ko.in_house.selectstar_002281_shadow_filtered.jpg
3. Select model: dbnet-mobilenetv3 · ep999 · hmean0.953
4. Set parameters: threshold=0.5, max_candidates=10
5. Click 'Run Inference'
6. Observe: '❌ Inference failed: Inference engine returned no results.'"
```

**Reproduce and Capture Logs:**
```bash
# Reproduce the issue
# (perform steps in UI)

# Immediately capture logs
make logs-inference-ui -- --lines 100

# If needed, follow logs in real-time
make logs-inference-ui -- --follow
```

### **Step 3: Structured Error Analysis**
Analyze captured logs using systematic approach:

**Log Analysis Checklist:**
```bash
# Check for common error patterns
grep -i "error\|exception\|traceback\|failed" logs/ui/inference_8501.err
grep -i "cuda\|gpu\|memory" logs/ui/inference_8501.err
grep -i "checkpoint\|model\|load" logs/ui/inference_8501.out
grep -i "inference\|predict" logs/ui/inference_8501.out
```

**Error Classification:**
- **Model Loading Errors:** Checkpoint path, architecture mismatch, dependencies
- **Data Processing Errors:** Image format, preprocessing failures, validation errors
- **Inference Engine Errors:** CUDA issues, memory problems, prediction failures
- **UI State Errors:** Session management, component state, user interaction issues

**Context Enhancement:**
```bash
# Capture additional diagnostic information
echo "MODEL DETAILS:
$(cat logs/ui/inference_8501.out | grep -A 20 "checkpoint_path")"

echo "SYSTEM RESOURCES:
$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv)"
$(free -h)

echo "PYTHON ENVIRONMENT:
$(uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')")"
```

### **Step 4: Hypothesis Testing with Full Context**
Test hypotheses while maintaining complete audit trail:

**Hypothesis Template:**
```bash
echo "HYPOTHESIS: Model checkpoint architecture mismatch
EVIDENCE: UI shows dbnet-mobilenetv3 but logs show different architecture
TEST: Compare checkpoint metadata with UI expectations
EXPECTED: Architecture validation error in logs"
```

**Testing Protocol:**
```bash
# Document each test attempt
echo "TEST 1: Checkpoint validation
COMMAND: uv run python scripts/validate_checkpoint.py /path/to/checkpoint.ckpt
RESULT: $(uv run python scripts/validate_checkpoint.py /path/to/checkpoint.ckpt 2>&1)
CONCLUSION: [PASS/FAIL] - [explanation]"

# Log all actions
make context-log-summarize
```

### **Step 5: Comprehensive Problem Documentation**
Create agent-ready problem description:

**Required Context Elements:**
```markdown
## Problem Summary
**Issue:** [One sentence description]
**Impact:** [What fails to work]
**Environment:** [System, versions, configuration]

## Reproduction Steps
1. [Step-by-step instructions]
2. [Expected vs actual behavior]

## Diagnostic Information
**Logs:** [Relevant log excerpts with timestamps]
**Configuration:** [UI config, model config, system settings]
**Data:** [Input data details, preprocessing results]

## Investigation Results
**Hypotheses Tested:** [List with results]
**Root Cause:** [Identified or suspected cause]
**Failed Attempts:** [What didn't work]

## Environment Details
**System:** $(uname -a)
**Python:** $(python --version)
**Dependencies:** $(uv run python -c "import streamlit, torch, numpy; print(f'Streamlit: {streamlit.__version__}')")
**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader)
```

### **Step 6: Agent Handoff Preparation**
Package complete context for agent consumption:

**Context Bundle Creation:**
```bash
# Create comprehensive context package
mkdir -p debug_context/$(date +%Y%m%d_%H%M%S)
cd debug_context/$(date +%Y%m%d_%H%M%S)

# Copy all relevant files
cp ../../logs/ui/inference_*.log ./
cp ../../configs/ui/inference.yaml ./
cp ../../configs/train.yaml ./
cp -r ../../logs/agent_runs/ ./

# Create summary report
cat > DEBUG_CONTEXT.md << 'EOF'
# Streamlit Inference UI Debug Context

## Problem
[Complete problem description]

## Files in this bundle:
- inference_8501.out/err: UI logs
- inference.yaml: UI configuration
- train.yaml: Training configuration
- agent_runs/: Context logging from investigation

## Key Findings
[Summary of investigation results]

## Next Steps for Agent
[Specific tasks or hypotheses to test]
EOF
```

## **Validation**
- [ ] Context log initialized with descriptive label
- [ ] UI started with logging enabled
- [ ] Reproduction steps documented in detail
- [ ] All log files captured and analyzed
- [ ] Error classification completed
- [ ] Hypotheses tested with full documentation
- [ ] Comprehensive problem summary created
- [ ] Context bundle prepared for agent handoff

## **Common Pitfalls**
- **Insufficient Logging:** Always use `make serve-inference-ui` (with logging) instead of direct streamlit commands
- **Missing Reproduction Steps:** Document exact UI interactions, not just "it crashes"
- **Incomplete Environment Context:** Include versions, configurations, system resources
- **Log Timing Issues:** Capture logs immediately after reproducing the error
- **Context Fragmentation:** Use single context log session for entire investigation

## **Integration with Existing Protocols**
- **Context Logging (06):** Use for all debugging actions
- **Iterative Debugging (07):** For complex root cause analysis
- **Debugging Workflow (03):** For initial triage and lightweight inspection
- **Feature Implementation (21):** For any code changes resulting from debugging

## **Tools and Commands**
```bash
# Process management
make serve-inference-ui          # Start with logging
make logs-inference-ui           # View logs
make logs-inference-ui -- --follow  # Follow logs
make stop-inference-ui           # Clean shutdown

# Context management
make context-log-start LABEL="debug_session"
make context-log-summarize

# Diagnostic tools
uv run python scripts/agent_tools/validate_config.py --config-name inference
uv run python scripts/debug/checkpoint_validator.py /path/to/model.ckpt
```

## **Success Criteria**
- Agent can reproduce the issue without additional questions
- All relevant logs and configuration are provided
- Investigation history is fully documented
- Context is structured for efficient analysis
- Next steps are clearly defined for the agent
