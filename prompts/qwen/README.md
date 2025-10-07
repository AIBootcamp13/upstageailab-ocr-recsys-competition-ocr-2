# Qwen Coder Prompt Templates

This directory contains specialized prompts for delegating isolated coding tasks to Qwen Coder.

## Usage

Each prompt file is a self-contained specification for a coding task that can be sent to Qwen via stdin.

### Execution Pattern

```bash
# Send prompt to Qwen in yolo mode
cat prompts/qwen/<task_name>.md | qwen --yolo

# Or using the helper script
uv run python scripts/agent_tools/delegate_to_qwen.py \
  --prompt prompts/qwen/<task_name>.md \
  --output <output_file>
```

### Prompt Structure

Each prompt follows this template:

```markdown
# Task: <Short Description>

## Context
- Project: Receipt OCR Text Detection
- Framework: PyTorch Lightning + Hydra
- Code Style: Follow pyproject.toml (ruff, mypy)

## Objective
<Clear, specific goal>

## Requirements
1. <Requirement 1>
2. <Requirement 2>
...

## Input Files
- Read: <file1>
- Read: <file2>

## Output Files
- Create: <new_file>
- Modify: <existing_file>

## Implementation Details
<Specific guidance, examples, edge cases>

## Validation
- [ ] Tests pass: <test command>
- [ ] Linting passes: uv run ruff check <file>
- [ ] Type checking: uv run mypy <file>

## Example Usage
<How to use the implemented code>
```

## Active Prompts

### Phase 1: Foundation & Monitoring
- `01_performance_profiler_callback.md` - Performance profiling callback
- `02_performance_profiler_config.md` - Hydra config for profiler
- `03_baseline_report_generator.md` - Performance baseline report script
- `04_regression_test_suite.md` - Performance regression tests

### Phase 2: PyClipper Caching
- `05_polygon_cache_implementation.md` - PolygonCache class (TDD)
- `06_cache_integration.md` - Integrate cache with DBCollateFN

## Guidelines for Creating Prompts

1. **Be Specific**: Include exact file paths, function signatures, expected behavior
2. **Provide Context**: Reference existing code patterns in the project
3. **Include Validation**: Specify exact test commands and expected outcomes
4. **Show Examples**: Include code snippets showing desired patterns
5. **Limit Scope**: Each prompt should be a 1-3 hour coding task maximum

## Validation Before Delegation

Before sending a prompt to Qwen:
- [ ] Prompt is self-contained (all context included)
- [ ] Input files are specified with full paths
- [ ] Expected output is clearly defined
- [ ] Validation criteria are testable
- [ ] Example usage is provided
