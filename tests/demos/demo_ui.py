#!/usr/bin/env python3
"""
Demo script for the Command Builder UI

This script demonstrates how the command builder generates CLI commands
from configuration dictionaries.
"""

# Setup project paths automatically
from ocr.utils.path_utils import setup_project_paths

setup_project_paths()

from ui.utils.command_builder import CommandBuilder
from ui.utils.config_parser import ConfigParser


def demo_train_command():
    """Demonstrate training command generation."""
    print("🚀 Training Command Demo")
    print("=" * 50)

    ConfigParser()
    command_builder = CommandBuilder()

    # Example training configuration
    config = {
        "encoder": "resnet50",
        "decoder": "unet",
        "head": "db_head",
        "loss": "db_loss",
        "learning_rate": 0.001,
        "batch_size": 8,
        "max_epochs": 20,
        "seed": 42,
        "exp_name": "demo_training",
        "wandb": True,
    }

    command = command_builder.build_train_command(config)
    print("Generated Command:")
    print(command)
    print()

    # Validate command
    is_valid, error_msg = command_builder.validate_command(command)
    print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if not is_valid:
        print(f"Error: {error_msg}")
    print()


def demo_test_command():
    """Demonstrate testing command generation."""
    print("🧪 Testing Command Demo")
    print("=" * 50)

    command_builder = CommandBuilder()

    # Example testing configuration
    config = {
        "checkpoint_path": "outputs/demo_training/checkpoints/epoch=19-step=1000.ckpt",
        "exp_name": "demo_testing",
    }

    command = command_builder.build_test_command(config)
    print("Generated Command:")
    print(command)
    print()

    # Validate command
    is_valid, error_msg = command_builder.validate_command(command)
    print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if not is_valid:
        print(f"Error: {error_msg}")
    print()


def demo_predict_command():
    """Demonstrate prediction command generation."""
    print("🔮 Prediction Command Demo")
    print("=" * 50)

    command_builder = CommandBuilder()

    # Example prediction configuration
    config = {
        "checkpoint_path": "outputs/demo_training/checkpoints/epoch=19-step=1000.ckpt",
        "exp_name": "demo_prediction",
        "minified_json": False,
    }

    command = command_builder.build_predict_command(config)
    print("Generated Command:")
    print(command)
    print()

    # Validate command
    is_valid, error_msg = command_builder.validate_command(command)
    print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if not is_valid:
        print(f"Error: {error_msg}")
    print()


def demo_available_options():
    """Demonstrate available configuration options."""
    print("🔧 Available Configuration Options")
    print("=" * 50)

    config_parser = ConfigParser()

    print("Model Components:")
    models = config_parser.get_available_models()
    for component, options in models.items():
        print(f"  {component}: {options}")
    print()

    print("Training Parameters:")
    train_params = config_parser.get_training_parameters()
    for param, info in train_params.items():
        if isinstance(info, dict):
            default = info.get("default", "N/A")
            min_val = info.get("min", "N/A")
            max_val = info.get("max", "N/A")
            print(f"  {param}: default={default}, range=[{min_val}, {max_val}]")
        else:
            print(f"  {param}: {info}")
    print()

    print("Available Datasets:")
    datasets = config_parser.get_available_datasets()
    print(f"  {datasets}")
    print()

    print("Available Presets:")
    presets = config_parser.get_available_presets()
    print(f"  {presets}")
    print()


if __name__ == "__main__":
    print("OCR Command Builder Demo")
    print("=" * 60)
    print()

    demo_available_options()
    demo_train_command()
    demo_test_command()
    demo_predict_command()

    print("✨ Demo completed! Run 'python run_ui.py command_builder' to launch the interactive UI.")
