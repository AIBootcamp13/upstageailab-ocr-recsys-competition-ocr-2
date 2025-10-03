"""
Unit tests for the command modules to ensure each component works correctly in isolation.
"""

from ui.utils.command.builder import CommandBuilder
from ui.utils.command.executor import CommandExecutor
from ui.utils.command.models import PredictCommandParams, TestCommandParams, TrainCommandParams
from ui.utils.command.quoting import quote_override
from ui.utils.command.validator import CommandValidator


class TestCommandModels:
    """Test the data models for command parameters."""

    def test_train_command_params_defaults(self):
        """Test that TrainCommandParams has correct default values."""
        params = TrainCommandParams(exp_name="test")
        assert params.exp_name == "test"
        assert params.max_epochs == 10  # default value
        assert params.learning_rate == 0.001  # default value
        assert params.batch_size == 16  # default value

    def test_train_command_params_with_values(self):
        """Test that TrainCommandParams accepts custom values."""
        params = TrainCommandParams(exp_name="custom_exp", encoder="resnet50", max_epochs=20, learning_rate=0.01)
        assert params.exp_name == "custom_exp"
        assert params.encoder == "resnet50"
        assert params.max_epochs == 20
        assert params.learning_rate == 0.01

    def test_test_command_params(self):
        """Test TestCommandParams functionality."""
        params = TestCommandParams(exp_name="test_exp", checkpoint_path="path/to/checkpoint.ckpt")
        assert params.exp_name == "test_exp"
        assert params.checkpoint_path == "path/to/checkpoint.ckpt"

    def test_predict_command_params_with_minified_json(self):
        """Test PredictCommandParams with minified_json."""
        params = PredictCommandParams(exp_name="predict_exp", checkpoint_path="path/to/checkpoint.ckpt", minified_json=True)
        assert params.minified_json is True
        assert params.exp_name == "predict_exp"


class TestQuotingUtils:
    """Test the quoting utilities for Hydra overrides."""

    def test_quote_override_no_special_chars(self):
        """Test quoting with no special characters."""
        result = quote_override("exp_name=test")
        assert result == "exp_name=test"  # Should not be quoted

    def test_quote_override_with_special_chars(self):
        """Test quoting with special characters."""
        result = quote_override("model.encoder.model_name=test=value")
        assert '"' in result  # Should be quoted due to equals sign
        assert result.startswith("'") and result.endswith("'")  # Should be shell-quoted

    def test_quote_override_with_spaces(self):
        """Test quoting with spaces."""
        result = quote_override("data.path=/path with spaces")
        assert '"' in result  # Should be quoted due to spaces
        assert result.startswith("'") and result.endswith("'")  # Should be shell-quoted

    def test_quote_override_already_quoted(self):
        """Test handling of already quoted overrides."""
        result = quote_override("'exp_name=test'")
        assert result == "'exp_name=test'"  # Should remain unchanged if already shell-quoted


class TestCommandBuilder:
    """Test the command builder functionality."""

    def test_build_train_command(self):
        """Test building a training command."""
        builder = CommandBuilder()
        params = TrainCommandParams(exp_name="test_exp", encoder="resnet18", max_epochs=5)
        command = builder.build_train_command(params)
        assert "train.py" in command
        assert "resnet18" in command
        assert "trainer.max_epochs=5" in command

    def test_build_test_command(self):
        """Test building a test command."""
        builder = CommandBuilder()
        params = TestCommandParams(exp_name="test_exp", checkpoint_path="test.ckpt")
        command = builder.build_test_command(params)
        assert "test.py" in command
        assert "test.ckpt" in command

    def test_build_predict_command(self):
        """Test building a prediction command."""
        builder = CommandBuilder()
        params = PredictCommandParams(exp_name="predict_exp", checkpoint_path="test.ckpt", minified_json=True)
        command = builder.build_predict_command(params)
        assert "predict.py" in command
        assert "test.ckpt" in command
        assert "minified_json=true" in command

    def test_build_command_from_overrides(self):
        """Test building command from generic overrides."""
        builder = CommandBuilder()
        command = builder.build_command_from_overrides("train.py", ["exp_name=test", "trainer.max_epochs=3"])
        assert "train.py" in command
        assert "exp_name=test" in command
        assert "trainer.max_epochs=3" in command


class TestCommandValidator:
    """Test the command validator functionality."""

    def test_validate_command_success(self):
        """Test successful command validation."""
        validator = CommandValidator()
        result, message = validator.validate_command("uv run python runners/train.py exp_name=test")
        assert result is True
        assert message == ""

    def test_validate_command_invalid_structure(self):
        """Test validation of invalid command structure."""
        validator = CommandValidator()
        result, message = validator.validate_command("invalid command")
        assert result is False
        assert "Command must start with" in message

    def test_validate_command_missing_script(self):
        """Test validation of command with non-existent script."""
        validator = CommandValidator()
        result, message = validator.validate_command("uv run python non_existent_script.py")
        assert result is False
        assert "Script not found" in message


class TestCommandExecutor:
    """Test the command executor functionality."""

    def test_executor_methods_exist(self):
        """Test that executor methods exist."""
        executor = CommandExecutor()
        assert hasattr(executor, "execute_command_streaming")
        assert hasattr(executor, "terminate_process_group")

    def test_executor_methods_callable(self):
        """Test that executor methods are callable."""
        executor = CommandExecutor()
        assert callable(executor.execute_command_streaming)
        assert callable(executor.terminate_process_group)
