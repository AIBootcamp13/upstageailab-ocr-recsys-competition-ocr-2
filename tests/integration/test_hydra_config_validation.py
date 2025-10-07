import pytest

from ui.utils.command import CommandBuilder, CommandValidator
from ui.utils.config_parser import ConfigParser


class TestHydraConfigurationValidation:
    """Test Hydra configuration validation and compatibility."""

    @pytest.fixture
    def config_parser(self):
        return ConfigParser()

    @pytest.fixture
    def command_builder(self):
        return CommandBuilder()

    @pytest.fixture
    def validator(self):
        return CommandValidator()

    def test_preprocessing_profiles_valid(self, config_parser, command_builder, validator):
        """Test that all preprocessing profiles generate valid commands."""
        profiles = config_parser.get_preprocessing_profiles()

        for profile_name, profile_data in profiles.items():
            overrides = profile_data.get("overrides", [])

            # Build minimal command with this profile
            test_overrides = [
                "exp_name=test_config_validation",
                "model.architecture_name=dbnet",
                "model.encoder.model_name=resnet18",
                "trainer.max_epochs=1",
            ] + overrides

            command = command_builder.build_command_from_overrides(script="train.py", overrides=test_overrides)

            is_valid, error = validator.validate_command(command)
            assert is_valid, f"Profile {profile_name} failed: {error}"

    @pytest.mark.parametrize(
        "architecture,encoder",
        [
            ("dbnet", "resnet18"),
            ("dbnet", "mobilenetv3_small_050"),
            ("craft", "resnet34"),
            # Add more combinations as needed
        ],
    )
    def test_architecture_encoder_compatibility(self, architecture, encoder, command_builder, validator):
        """Test architecture and encoder compatibility."""
        overrides = [
            f"model.architecture_name={architecture}",
            f"model.encoder.model_name={encoder}",
            "exp_name=test_compat",
            "trainer.max_epochs=1",
        ]

        command = command_builder.build_command_from_overrides("train.py", overrides)
        is_valid, error = validator.validate_command(command)
        assert is_valid, f"{architecture} + {encoder} incompatible: {error}"

    def test_optimizer_scheduler_compatibility(self, command_builder, validator):
        """Test optimizer and scheduler compatibility."""
        # Test common combinations
        test_cases = [
            ("adam", "step"),
            ("adamw", "cosine"),
            ("sgd", "linear"),
        ]

        for optimizer, scheduler in test_cases:
            overrides = [
                f"model/optimizers={optimizer}",
                f"model/schedulers={scheduler}",
                "exp_name=test_optimizer_sched",
                "trainer.max_epochs=1",
            ]

            command = command_builder.build_command_from_overrides("train.py", overrides)
            is_valid, error = validator.validate_command(command)
            assert is_valid, f"{optimizer} + {scheduler} incompatible: {error}"

    def test_dataset_transform_compatibility(self, command_builder, validator):
        """Test dataset and transform compatibility."""
        # Test that transforms work with different datasets
        test_cases = [
            ("preprocessing_docTR_demo", "enhanced_transforms"),
            ("preprocessing_standard", "basic_transforms"),
        ]

        for dataset, transforms in test_cases:
            overrides = [
                f"+preset/datasets={dataset}",
                f'data.transforms.train_transform="${{{transforms}.train_transform}}"',
                f'data.transforms.val_transform="${{{transforms}.val_transform}}"',
                "exp_name=test_dataset_transform",
                "trainer.max_epochs=1",
            ]

            command = command_builder.build_command_from_overrides("train.py", overrides)
            is_valid, error = validator.validate_command(command)
            assert is_valid, f"{dataset} + {transforms} incompatible: {error}"

    def test_full_config_integration(self, config_parser, command_builder, validator):
        """Test full configuration integration with realistic training setup."""
        # Build a complete configuration similar to what would be used in production
        overrides = [
            "exp_name=test_full_integration",
            "model.architecture_name=dbnet",
            "model/architectures=dbnet",
            "model.encoder.model_name=mobilenetv3_small_050",
            "model.component_overrides.decoder.name=fpn_decoder",
            "model.component_overrides.head.name=db_head",
            "model.component_overrides.loss.name=db_loss",
            "model/optimizers=adamw",
            "model.optimizer.lr=0.0003",
            "model.optimizer.weight_decay=0.0001",
            "dataloaders.train_dataloader.batch_size=8",
            "dataloaders.val_dataloader.batch_size=8",
            "trainer.max_epochs=1",
            "trainer.accumulate_grad_batches=1",
            "trainer.gradient_clip_val=5.0",
            "trainer.precision=32",
            "seed=42",
            "data=preprocessing",
            "+preset/datasets=preprocessing_docTR_demo",
            'data.transforms.train_transform="${enhanced_transforms.train_transform}"',
            'data.transforms.val_transform="${enhanced_transforms.val_transform}"',
            'data.transforms.test_transform="${enhanced_transforms.test_transform}"',
            'data.transforms.predict_transform="${enhanced_transforms.predict_transform}"',
        ]

        command = command_builder.build_command_from_overrides("train.py", overrides)
        is_valid, error = validator.validate_command(command)
        assert is_valid, f"Full config integration failed: {error}"
