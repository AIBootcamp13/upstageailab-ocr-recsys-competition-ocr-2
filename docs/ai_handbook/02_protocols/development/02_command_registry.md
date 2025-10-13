# **filename: docs/ai_handbook/02_protocols/development/02_command_registry.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=automation,commands,scripts -->

# **Protocol: Command Registry for AI Agents**

This protocol provides the standardized approach for managing and executing approved scripts and commands for AI agent automation within the project.

## **Overview**

This protocol establishes the authoritative registry of approved, safe-to-run scripts and commands for autonomous AI agent execution. All commands are designed to use `uv run` to ensure proper environment isolation and dependency management. The registry is organized by functional categories and includes safety guidelines, resource requirements, and expected outputs for each command.

## **Prerequisites**

- Access to development environment with uv package manager installed
- Understanding of project structure and configuration system
- Knowledge of Hydra configuration patterns
- Familiarity with the project's data formats and model architectures

## **Procedure**

### **Step 1: Command Selection & Safety Assessment**
Before executing any command, verify:
- Command is listed in this approved registry
- Required parameters are available and valid
- Resource requirements match available environment
- Command aligns with current task objectives

### **Step 2: Environment Preparation**
Ensure proper setup before execution:
- Activate project environment: `uv sync`
- Verify configuration files exist and are valid
- Check required data directories and checkpoints are accessible
- Confirm output directories exist or will be created

### **Step 3: Command Execution**
Execute commands following these patterns:

**Validation & Smoke Tests:**
- `uv run python scripts/agent_tools/validate_config.py --config-name <name>`
- `uv run python runners/train.py --config-name train trainer.fast_dev_run=true`

**Data & Preprocessing:**
- `uv run python scripts/agent_tools/generate_samples.py --num-samples 5`

**Querying Information:**
- `uv run python scripts/agent_tools/list_checkpoints.py`

**Data Diagnostics:**
- `uv run python tests/debug/data_analyzer.py --mode orientation|polygons|both [--limit N]`
- `uv run python ui/visualize_predictions.py --image_dir <path> --checkpoint <path> [--max_images N] [--save_dir <path>] [--score_threshold T]`

### **Step 4: Result Validation & Documentation**
After execution:
- Verify expected outputs were generated
- Check for error messages or warnings
- Document results in context logs if applicable
- Update any relevant documentation with findings

## **Validation**

Run the following validation checks:

```bash
# Registry integrity validation
uv run python scripts/agent_tools/validate_manifest.py

# Command availability check
uv run python scripts/agent_tools/validate_config.py --config-name train

# Environment readiness
uv run python -c "import torch; print('PyTorch available')"
```

## **Troubleshooting**

### **Common Issues**
- **Command Not Found**: Ensure `uv sync` has been run to install dependencies
- **Configuration Errors**: Validate config files with `validate_config.py` before use
- **Permission Errors**: Check file/directory permissions for input/output paths
- **Resource Exhaustion**: Monitor GPU/CPU usage, especially for inference commands
- **Path Not Found**: Verify relative paths are correct from project root

### **Debugging Steps**
1. Test basic environment: `uv run python -c "print('Environment ready')"`
2. Validate configuration: `uv run python scripts/agent_tools/validate_config.py --config-name train`
3. Check file paths: `ls -la <path/to/check>`
4. Monitor resources: `nvidia-smi` (for GPU) or `top` (for CPU)
5. Review command logs for specific error messages

## **Related Documents**

- `docs/ai_handbook/02_protocols/development/01_coding_standards.md` - Coding standards and environment setup
- `docs/ai_handbook/02_protocols/development/03_debugging_workflow.md` - Debugging procedures for command issues
- `docs/ai_handbook/02_protocols/governance/20_bug_fix_protocol.md` - Issue reporting for command failures
- `docs/ai_handbook/03_references/architecture/01_architecture.md` - Project architecture overview
- `docs/ai_handbook/_templates/development.md` - Development template

---

*This document follows the development protocol template. Last updated: October 13, 2025*
