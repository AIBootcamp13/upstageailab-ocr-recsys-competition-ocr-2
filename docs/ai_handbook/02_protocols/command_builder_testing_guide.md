# Command Builder Testing Guide

This document provides guidance for testing the refactored Command Builder module to ensure stability as new features are added.

## Test Strategy Overview

### Unit Testing Approach
Each module in the `ui/utils/command/` package should be tested independently:

- **models.py**: Test dataclass instantiation and default values
- **quoting.py**: Test override quoting logic with various edge cases
- **builder.py**: Test command construction logic
- **executor.py**: Test execution methods (using mocks for actual execution)
- **validator.py**: Test command validation logic

### Integration Testing Approach
- Test the interaction between components
- Test UI component integration with the new modules
- Test end-to-end command workflows

## Recommended Test Implementation

### 1. Unit Tests for Each Module

Create `tests/test_command_modules.py`:

```python
import pytest
from ui.utils.command.models import TrainCommandParams, TestCommandParams, PredictCommandParams
from ui.utils.command.quoting import quote_override
from ui.utils.command.builder import CommandBuilder
from ui.utils.command.validator import CommandValidator
from ui.utils.command.executor import CommandExecutor

class TestCommandModels:
    def test_train_command_params_defaults(self):
        params = TrainCommandParams(exp_name='test')
        assert params.exp_name == 'test'
        assert params.max_epochs == 10  # default value

    def test_predict_command_params_with_minified_json(self):
        params = PredictCommandParams(minified_json=True)
        assert params.minified_json is True

class TestQuotingUtils:
    def test_quote_override_with_special_chars(self):
        result = quote_override('model.encoder.model_name=test=value')
        assert '"' in result  # Should be quoted due to special chars

    def test_quote_override_no_special_chars(self):
        result = quote_override('exp_name=test')
        assert result == 'exp_name=test'  # Should not be quoted

class TestCommandBuilder:
    def test_build_train_command(self):
        builder = CommandBuilder()
        params = TrainCommandParams(exp_name='test', encoder='resnet18')
        command = builder.build_train_command(params)
        assert 'train.py' in command
        assert 'resnet18' in command

    def test_build_command_from_overrides(self):
        builder = CommandBuilder()
        command = builder.build_command_from_overrides('train.py', ['exp_name=test'])
        assert 'uv run python' in command
        assert 'train.py' in command

class TestCommandValidator:
    def test_validate_command_success(self):
        validator = CommandValidator()
        result, message = validator.validate_command('uv run python runners/train.py exp_name=test')
        assert result is True

    def test_validate_command_invalid_structure(self):
        validator = CommandValidator()
        result, message = validator.validate_command('invalid command')
        assert result is False

class TestCommandExecutor:
    def test_execute_command_streaming_method_exists(self):
        executor = CommandExecutor()
        assert hasattr(executor, 'execute_command_streaming')
```

### 2. UI Component Integration Tests

Create `tests/test_command_builder_ui.py`:

```python
import pytest
from ui.utils.command import CommandBuilder, CommandValidator, CommandExecutor
from ui.utils.command.models import TrainCommandParams
from ui.apps.command_builder.components.training import render_training_page
from ui.apps.command_builder.components.test import render_test_page
from ui.apps.command_builder.components.predict import render_predict_page
from ui.apps.command_builder.components.execution import render_execution_panel

class TestTrainingComponent:
    def test_validation_call_uses_validator_instance(self):
        # Ensure the fix for validate_command location is maintained
        builder = CommandBuilder()
        validator = CommandValidator()

        # Test that validation is done via validator, not builder
        assert hasattr(validator, 'validate_command')
        assert not hasattr(builder, 'validate_command')  # This was moved

class TestCommandGeneration:
    def test_all_command_types_generate_properly(self):
        builder = CommandBuilder()

        # Test train command
        train_params = TrainCommandParams(exp_name='test_train', encoder='resnet18', max_epochs=5)
        train_cmd = builder.build_train_command(train_params)
        assert 'train.py' in train_cmd
        assert 'resnet18' in train_cmd

        # Test test command
        test_cmd = builder.build_test_command(TestCommandParams(exp_name='test_test'))
        assert 'test.py' in test_cmd

        # Test predict command
        predict_cmd = builder.build_predict_command(PredictCommandParams(exp_name='test_predict'))
        assert 'predict.py' in predict_cmd
```

### 3. Regression Test for Fixed Issue

Create `tests/test_regression_validation_fix.py`:

```python
import pytest
from ui.utils.command import CommandBuilder, CommandValidator

def test_validation_method_location_fix():
    """
    Regression test for the issue where validate_command was called on CommandBuilder
    but had been moved to CommandValidator during refactoring.
    """
    builder = CommandBuilder()
    validator = CommandValidator()

    # The validation method should exist on CommandValidator, not CommandBuilder
    assert hasattr(validator, 'validate_command')
    assert not hasattr(builder, 'validate_command')

    # The validation method should work properly
    is_valid, error_msg = validator.validate_command('uv run python runners/train.py exp_name=test')
    assert is_valid  # Should be valid
```

### 4. Smoke Tests

Create `tests/test_command_builder_smoke.py`:

```python
def test_command_builder_smoke_test():
    """Basic smoke test to ensure Command Builder modules load and work together"""
    from ui.utils.command import CommandBuilder, CommandValidator, CommandExecutor
    from ui.utils.command.models import TrainCommandParams

    # Create instances
    builder = CommandBuilder()
    validator = CommandValidator()
    executor = CommandExecutor()

    # Test basic functionality
    params = TrainCommandParams(exp_name='smoke_test', encoder='resnet18')
    command = builder.build_train_command(params)

    # Validate command
    is_valid, msg = validator.validate_command(command)
    assert is_valid

    # Check that executor methods exist
    assert hasattr(executor, 'execute_command_streaming')
    assert hasattr(executor, 'terminate_process_group')

    print("✓ Smoke test passed - Command Builder modules work together correctly")
```

## CI/CD Integration Recommendations

### GitHub Actions Workflow
Add to `.github/workflows/test-command-builder.yml`:

```yaml
name: Test Command Builder
on:
  push:
    paths:
      - 'ui/utils/command/**'
      - 'ui/apps/command_builder/**'
      - 'tests/**'
  pull_request:
    paths:
      - 'ui/utils/command/**'
      - 'ui/apps/command_builder/**'
      - 'tests/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev
    - name: Run tests
      run: |
        uv run pytest tests/test_command_builder* -v
```

## Testing Best Practices

1. **Run Tests Before Each Release**: Always run the full test suite before deploying changes
2. **Add Tests for New Features**: For every new feature, add corresponding unit and integration tests
3. **Mock External Dependencies**: For command execution tests, use mocks to avoid actually running commands
4. **Test Edge Cases**: Test with unusual inputs to quoting and validation functions
5. **Monitor Performance**: Add benchmarks for command generation to catch performance regressions
6. **Regular Clean-up**: Periodically clean up deprecated tests when removing backward compatibility

## Monitoring and Alerting

Consider adding a simple health check that can be run periodically:

```python
def command_builder_health_check():
    """Health check for command builder functionality"""
    try:
        from ui.utils.command import CommandBuilder, CommandValidator
        from ui.utils.command.models import TrainCommandParams

        # Test core functionality
        builder = CommandBuilder()
        validator = CommandValidator()

        params = TrainCommandParams(exp_name='health_check', encoder='resnet18')
        command = builder.build_train_command(params)
        is_valid, msg = validator.validate_command(command)

        if not is_valid:
            return False, f"Command validation failed: {msg}"

        return True, "Command Builder is healthy"
    except Exception as e:
        return False, f"Command Builder health check failed: {str(e)}"
```

This approach will help you catch regressions early and ensure the Command Builder remains stable as you add new features.
