"""
Integration tests for Command Builder UI components to ensure they work with the refactored modules.
"""

from ui.utils.command import CommandBuilder, CommandExecutor, CommandValidator
from ui.utils.command.models import PredictCommandParams, TestCommandParams, TrainCommandParams


class TestCommandGenerationIntegration:
    """Test the integration between command builder and UI components."""

    def test_command_builder_validator_integration(self):
        """Test that CommandBuilder and CommandValidator work together properly."""
        builder = CommandBuilder()
        validator = CommandValidator()

        # Create a command
        params = TrainCommandParams(exp_name="integration_test", encoder="resnet18")
        command = builder.build_train_command(params)

        # Validate the command
        is_valid, error_msg = validator.validate_command(command)
        assert is_valid, f"Command validation failed: {error_msg}"
        assert "resnet18" in command
        assert "train.py" in command

    def test_all_command_types_integration(self):
        """Test that all command types work with validation."""
        builder = CommandBuilder()
        validator = CommandValidator()

        # Test train command
        train_params = TrainCommandParams(exp_name="train_test", encoder="resnet18", max_epochs=5)
        train_cmd = builder.build_train_command(train_params)
        is_valid, _ = validator.validate_command(train_cmd)
        assert is_valid
        assert "train.py" in train_cmd

        # Test test command
        test_params = TestCommandParams(exp_name="test_test", checkpoint_path="test.ckpt")
        test_cmd = builder.build_test_command(test_params)
        is_valid, _ = validator.validate_command(test_cmd)
        assert is_valid
        assert "test.py" in test_cmd

        # Test predict command
        predict_params = PredictCommandParams(exp_name="predict_test", checkpoint_path="predict.ckpt", minified_json=True)
        predict_cmd = builder.build_predict_command(predict_params)
        is_valid, _ = validator.validate_command(predict_cmd)
        assert is_valid
        assert "predict.py" in predict_cmd


class TestUIComponentIntegration:
    """Test scenarios that mimic how UI components use the command modules."""

    def test_training_component_workflow_simulation(self):
        """Simulate the workflow of the training component."""
        builder = CommandBuilder()
        validator = CommandValidator()

        # Simulate what the training component does
        params = TrainCommandParams(exp_name="training_simulation", encoder="resnet18", max_epochs=10, learning_rate=0.001)

        # Build command
        command = builder.build_train_command(params)
        assert command.startswith("uv run python")
        assert "train.py" in command
        assert "resnet18" in command

        # Validate command
        is_valid, error = validator.validate_command(command)
        assert is_valid, f"Generated command should be valid: {error}"

    def test_test_component_workflow_simulation(self):
        """Simulate the workflow of the test component."""
        builder = CommandBuilder()
        validator = CommandValidator()

        # Simulate what the test component does
        TestCommandParams(exp_name="test_simulation", checkpoint_path="outputs/checkpoint.ckpt")

        # Build command using overrides approach (used in UI)
        overrides = ["checkpoint_path=outputs/checkpoint.ckpt", "exp_name=test_simulation"]
        command = builder.build_command_from_overrides("test.py", overrides)

        assert "test.py" in command
        assert "checkpoint_path=outputs/checkpoint.ckpt" in command

        # Validate command
        is_valid, error = validator.validate_command(command)
        assert is_valid, f"Generated command should be valid: {error}"

    def test_predict_component_workflow_simulation(self):
        """Simulate the workflow of the predict component."""
        builder = CommandBuilder()
        validator = CommandValidator()

        # Simulate what the predict component does
        params = PredictCommandParams(exp_name="predict_simulation", checkpoint_path="outputs/predict.ckpt", minified_json=True)

        command = builder.build_predict_command(params)
        assert "predict.py" in command
        assert "minified_json=true" in command

        # Validate command
        is_valid, error = validator.validate_command(command)
        assert is_valid, f"Generated command should be valid: {error}"


class TestExecutorIntegration:
    """Test integration with the command executor."""

    def test_all_command_types_with_executor_methods(self):
        """Test that all command types can be used with executor methods."""
        builder = CommandBuilder()
        validator = CommandValidator()
        executor = CommandExecutor()

        # Test train command
        train_params = TrainCommandParams(exp_name="executor_test", encoder="resnet18")
        train_cmd = builder.build_train_command(train_params)

        # Verify the command is valid
        is_valid, _ = validator.validate_command(train_cmd)
        assert is_valid

        # Verify executor methods exist and are accessible
        assert hasattr(executor, "execute_command_streaming")
        assert hasattr(executor, "terminate_process_group")

        # Test test command
        test_params = TestCommandParams(exp_name="test_executor", checkpoint_path="test.ckpt")
        test_cmd = builder.build_test_command(test_params)
        is_valid, _ = validator.validate_command(test_cmd)
        assert is_valid

        # Test predict command
        predict_params = PredictCommandParams(exp_name="predict_executor", checkpoint_path="predict.ckpt")
        predict_cmd = builder.build_predict_command(predict_params)
        is_valid, _ = validator.validate_command(predict_cmd)
        assert is_valid
