"""
Regression tests to prevent re-introduction of the validation method issue.

This test specifically addresses the issue where validate_command was called on
CommandBuilder but had been moved to CommandValidator during refactoring.
"""

from ui.utils.command import CommandBuilder, CommandExecutor, CommandValidator


def test_validation_method_location_fix():
    """
    Regression test for the issue where validate_command was called on CommandBuilder
    but had been moved to CommandValidator during refactoring.
    """
    builder = CommandBuilder()
    validator = CommandValidator()

    # The validation method should exist on CommandValidator, not CommandBuilder
    assert hasattr(validator, "validate_command")
    assert not hasattr(builder, "validate_command"), (
        "validate_command should not exist on CommandBuilder after refactoring. "
        "If this fails, the method may have been moved back incorrectly."
    )

    # The validation method should work properly
    is_valid, error_msg = validator.validate_command("uv run python runners/train.py exp_name=test")
    assert is_valid, f"Command validation should succeed, but got error: {error_msg}"


def test_execution_method_location_fix():
    """
    Regression test for the issue where execute_command_streaming was called on
    CommandBuilder but had been moved to CommandExecutor during refactoring.
    """
    builder = CommandBuilder()
    executor = CommandExecutor()

    # The execution method should exist on CommandExecutor, not CommandBuilder
    assert hasattr(executor, "execute_command_streaming")
    assert not hasattr(builder, "execute_command_streaming"), (
        "execute_command_streaming should not exist on CommandBuilder after refactoring. "
        "If this fails, the method may have been moved back incorrectly."
    )

    # The execution method should be callable
    assert callable(executor.execute_command_streaming)


def test_all_method_locations_after_refactor():
    """Test that all methods are in their new correct locations after refactoring."""
    builder = CommandBuilder()
    validator = CommandValidator()
    executor = CommandExecutor()

    # Builder should have command construction methods
    assert hasattr(builder, "build_train_command")
    assert hasattr(builder, "build_test_command")
    assert hasattr(builder, "build_predict_command")
    assert hasattr(builder, "build_command_from_overrides")

    # Validator should have validation methods
    assert hasattr(validator, "validate_command")

    # Executor should have execution methods
    assert hasattr(executor, "execute_command_streaming")
    assert hasattr(executor, "terminate_process_group")

    # Methods should NOT be in wrong locations
    method_location_assertions = [
        (builder, "validate_command", "should not be on CommandBuilder"),
        (builder, "execute_command_streaming", "should not be on CommandBuilder"),
        (validator, "build_train_command", "should not be on CommandValidator"),
        (validator, "execute_command_streaming", "should not be on CommandValidator"),
        (executor, "build_train_command", "should not be on CommandExecutor"),
        (executor, "validate_command", "should not be on CommandExecutor"),
    ]

    for obj, method_name, description in method_location_assertions:
        assert not hasattr(obj, method_name), f"{method_name} {description}"


def test_command_builder_still_works_with_proper_workflow():
    """
    Test that the main command building workflow still works correctly
    with the new modular structure.
    """
    # Create all necessary components
    builder = CommandBuilder()
    validator = CommandValidator()

    # Test building and validation workflow
    from ui.utils.command.models import TrainCommandParams

    params = TrainCommandParams(exp_name="regression_test", encoder="resnet18")
    command = builder.build_train_command(params)

    # Should be a valid command
    is_valid, error = validator.validate_command(command)
    assert is_valid, f"Command should be valid: {error}"

    # Command should contain expected elements
    assert "train.py" in command
    assert "resnet18" in command
    assert "exp_name=regression_test" in command


def test_ui_component_pattern_still_works():
    """
    Test that the pattern used in UI components (creating validator/executor instances) works.
    """
    builder = CommandBuilder()

    # This is how UI components now use the modules
    validator = CommandValidator()
    executor = CommandExecutor()

    from ui.utils.command.models import PredictCommandParams

    params = PredictCommandParams(exp_name="ui_pattern_test", checkpoint_path="test.ckpt")
    command = builder.build_predict_command(params)

    # Validate using the validator instance (not builder)
    is_valid, error = validator.validate_command(command)
    assert is_valid, f"Command validation should work with validator instance: {error}"

    # Executor should be available
    assert hasattr(executor, "execute_command_streaming")
    assert hasattr(executor, "terminate_process_group")
