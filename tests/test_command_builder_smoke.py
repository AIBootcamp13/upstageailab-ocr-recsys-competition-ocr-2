"""
Smoke tests for Command Builder functionality.

These tests provide a quick verification that the core functionality works
without running the full test suite. Useful for pre-deployment checks.
"""

from ui.utils.command import CommandBuilder, CommandExecutor, CommandValidator
from ui.utils.command.models import PredictCommandParams, TestCommandParams, TrainCommandParams


def command_builder_health_check():
    """Health check for command builder functionality."""
    try:
        # Test core functionality
        builder = CommandBuilder()
        validator = CommandValidator()

        params = TrainCommandParams(exp_name="health_check", encoder="resnet18")
        command = builder.build_train_command(params)
        is_valid, msg = validator.validate_command(command)

        if not is_valid:
            return False, f"Command validation failed: {msg}"

        return True, "Command Builder is healthy"
    except Exception as e:
        return False, f"Command Builder health check failed: {str(e)}"


class TestCommandBuilderSmokeTests:
    """Basic smoke tests to verify Command Builder functionality."""

    def test_command_builder_basic_functionality(self):
        """Test basic command builder functionality."""
        builder = CommandBuilder()
        validator = CommandValidator()
        executor = CommandExecutor()

        # Test that all components can be instantiated
        assert builder is not None
        assert validator is not None
        assert executor is not None

        # Test basic command generation
        params = TrainCommandParams(exp_name="smoke_test", encoder="resnet18")
        command = builder.build_train_command(params)

        # Verify command structure
        assert "uv run python" in command
        assert "train.py" in command
        assert "resnet18" in command

        # Verify validation works
        is_valid, msg = validator.validate_command(command)
        assert is_valid, f"Command should be valid: {msg}"

        # Verify executor has required methods
        assert hasattr(executor, "execute_command_streaming")
        assert hasattr(executor, "terminate_process_group")

        print("✓ Command Builder basic functionality test passed")

    def test_all_command_types_smoke(self):
        """Test all command types work in a basic smoke test."""
        builder = CommandBuilder()
        validator = CommandValidator()

        # Test train command
        train_params = TrainCommandParams(exp_name="smoke_train", encoder="resnet18")
        train_cmd = builder.build_train_command(train_params)
        assert "train.py" in train_cmd
        train_valid, _ = validator.validate_command(train_cmd)
        assert train_valid

        # Test test command
        test_params = TestCommandParams(exp_name="smoke_test", checkpoint_path="test.ckpt")
        test_cmd = builder.build_test_command(test_params)
        assert "test.py" in test_cmd
        test_valid, _ = validator.validate_command(test_cmd)
        assert test_valid

        # Test predict command
        predict_params = PredictCommandParams(exp_name="smoke_predict", checkpoint_path="predict.ckpt")
        predict_cmd = builder.build_predict_command(predict_params)
        assert "predict.py" in predict_cmd
        predict_valid, _ = validator.validate_command(predict_cmd)
        assert predict_valid

        print("✓ All command types smoke test passed")

    def test_quoting_functionality_smoke(self):
        """Test that quoting functionality works."""
        from ui.utils.command.quoting import quote_override

        # Test basic quoting
        result = quote_override("exp_name=test")
        assert result == "exp_name=test"

        # Test quoting with special chars
        result = quote_override("model.encoder.model_name=test=value")
        assert '"' in result or "'" in result  # Should be quoted

        print("✓ Quoting functionality smoke test passed")

    def test_module_imports_smoke(self):
        """Test that all modules can be imported without errors."""
        # This is already implicitly tested by importing above, but let's be explicit
        from ui.utils.command import CommandBuilder, CommandExecutor, CommandValidator
        from ui.utils.command.models import TrainCommandParams
        from ui.utils.command.quoting import quote_override

        # Verify classes exist
        assert CommandBuilder is not None
        assert CommandValidator is not None
        assert CommandExecutor is not None
        assert TrainCommandParams is not None
        assert quote_override is not None

        print("✓ Module imports smoke test passed")

    def test_complete_workflow_smoke_test(self):
        """Test the complete workflow from parameter creation to validation."""
        # Create all necessary components
        builder = CommandBuilder()
        validator = CommandValidator()

        # Create parameters
        params = TrainCommandParams(exp_name="complete_workflow_test", encoder="resnet18", max_epochs=1, learning_rate=0.001)

        # Build command
        command = builder.build_train_command(params)

        # Validate command
        is_valid, error_msg = validator.validate_command(command)

        # Verify everything worked
        assert is_valid, f"Command should be valid: {error_msg}"
        assert "complete_workflow_test" in command
        assert "resnet18" in command
        assert "trainer.max_epochs=1" in command

        print("✓ Complete workflow smoke test passed")


def test_command_builder_health_check_function():
    """Test the health check function directly."""
    is_healthy, message = command_builder_health_check()
    assert is_healthy, f"Health check should pass: {message}"
    print(f"✓ Health check function test passed: {message}")
