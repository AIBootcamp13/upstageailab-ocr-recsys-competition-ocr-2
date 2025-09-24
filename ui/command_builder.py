"""
OCR Training Command Builder - Streamlit UI

A Streamlit application for building and executing OCR training, testing,
and prediction commands through an intuitive UI.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ui.utils.command_builder import CommandBuilder
from ui.utils.config_parser import ConfigParser


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")

    st.title("🔍 OCR Training Command Builder")
    st.markdown(
        "Build and execute training, testing, and prediction commands with ease"
    )

    # Initialize utilities
    config_parser = ConfigParser()
    command_builder = CommandBuilder()

    # Sidebar for command type selection
    st.sidebar.header("Command Type")
    command_type = st.sidebar.selectbox(
        "Select Command",
        ["train", "test", "predict"],
        help="Choose the type of command to build",
    )

    # Main content area
    if command_type == "train":
        render_train_interface(config_parser, command_builder)
    elif command_type == "test":
        render_test_interface(config_parser, command_builder)
    elif command_type == "predict":
        render_predict_interface(config_parser, command_builder)


def render_train_interface(
    config_parser: ConfigParser, command_builder: CommandBuilder
):
    """Render the training command interface."""
    st.header("🚀 Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Architecture")

        # Get available models
        models = config_parser.get_available_models()

        # Encoder selection
        encoder = st.selectbox(
            "Encoder",
            models.get("backbones", ["resnet18"]),
            index=0,
            help="Choose the backbone encoder architecture",
        )

        # Decoder selection
        decoder = st.selectbox(
            "Decoder",
            models.get("decoders", ["unet"]),
            index=0,
            help="Choose the decoder architecture",
        )

        # Head selection
        head = st.selectbox(
            "Head",
            models.get("heads", ["db_head"]),
            index=0,
            help="Choose the detection head",
        )

        # Loss selection
        loss = st.selectbox(
            "Loss Function",
            models.get("losses", ["db_loss"]),
            index=0,
            help="Choose the loss function",
        )

    with col2:
        st.subheader("Training Parameters")

        # Get training parameters
        train_params = config_parser.get_training_parameters()

        # Learning rate
        lr_param = train_params.get("learning_rate", {})
        learning_rate = st.slider(
            "Learning Rate",
            min_value=lr_param.get("min", 1e-6),
            max_value=lr_param.get("max", 1e-2),
            value=lr_param.get("default", 0.001),
            step=1e-5,
            format="%.1e",
            help="Learning rate for training",
        )

        # Batch size
        batch_param = train_params.get("batch_size", {})
        batch_size = st.slider(
            "Batch Size",
            min_value=batch_param.get("min", 1),
            max_value=batch_param.get("max", 64),
            value=batch_param.get("default", 4),
            step=1,
            help="Batch size for training",
        )

        # Max epochs
        epochs_param = train_params.get("max_epochs", {})
        max_epochs = st.slider(
            "Max Epochs",
            min_value=epochs_param.get("min", 1),
            max_value=epochs_param.get("max", 100),
            value=epochs_param.get("default", 10),
            step=1,
            help="Maximum number of training epochs",
        )

        # Random seed
        seed_param = train_params.get("seed", {})
        seed = st.number_input(
            "Random Seed",
            value=seed_param.get("default", 42),
            min_value=0,
            help="Random seed for reproducibility",
        )

    st.subheader("Experiment Settings")

    col3, col4 = st.columns(2)

    with col3:
        exp_name = st.text_input(
            "Experiment Name",
            value="ocr_training",
            help="Name for this training experiment",
        )

        use_wandb = st.checkbox(
            "Use Weights & Biases",
            value=False,
            help="Enable W&B logging for experiment tracking",
        )

    with col4:
        resume_training = st.checkbox(
            "Resume Training", value=False, help="Resume from a checkpoint"
        )

        checkpoint_path = ""
        if resume_training:
            checkpoint_path = st.text_input(
                "Checkpoint Path",
                value="",
                help="Path to checkpoint file to resume from",
            )

    # Build configuration
    config = {
        "encoder": encoder,
        "decoder": decoder,
        "head": head,
        "loss": loss,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "seed": seed,
        "exp_name": exp_name,
        "wandb": use_wandb,
    }

    if resume_training and checkpoint_path:
        config["resume"] = checkpoint_path

    # Generate command
    command = command_builder.build_train_command(config)

    st.subheader("Generated Command")
    st.code(command, language="bash")

    # Validation
    is_valid, error_msg = command_builder.validate_command(command)
    if not is_valid:
        st.error(f"Command validation failed: {error_msg}")
    else:
        st.success("Command is valid")

        # Execute button
        if st.button("🚀 Execute Training", type="primary"):
            execute_command(command_builder, command)


def render_test_interface(config_parser: ConfigParser, command_builder: CommandBuilder):
    """Render the testing command interface."""
    st.header("🧪 Testing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Settings")

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="outputs/ocr_training/checkpoints/epoch-0-step-205.ckpt",
            help="Path to the trained model checkpoint",
        )

    with col2:
        st.subheader("Experiment Settings")

        exp_name = st.text_input(
            "Experiment Name",
            value="ocr_testing",
            help="Name for this testing experiment",
        )

    # Build configuration
    config = {
        "checkpoint_path": checkpoint_path,
        "exp_name": exp_name,
    }

    # Generate command
    command = command_builder.build_test_command(config)

    st.subheader("Generated Command")
    st.code(command, language="bash")

    # Validation
    is_valid, error_msg = command_builder.validate_command(command)
    if not is_valid:
        st.error(f"Command validation failed: {error_msg}")
    else:
        st.success("Command is valid")

        # Execute button
        if st.button("🧪 Execute Testing", type="primary"):
            execute_command(command_builder, command)


def render_predict_interface(
    config_parser: ConfigParser, command_builder: CommandBuilder
):
    """Render the prediction command interface."""
    st.header("🔮 Prediction Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Settings")

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="outputs/ocr_training/checkpoints/epoch-0-step-205.ckpt",
            help="Path to the trained model checkpoint",
        )

    with col2:
        st.subheader("Output Settings")

        exp_name = st.text_input(
            "Experiment Name",
            value="ocr_prediction",
            help="Name for this prediction experiment",
        )

        minified_json = st.checkbox(
            "Minified JSON", value=False, help="Output minified JSON format"
        )

    # Build configuration
    config = {
        "checkpoint_path": checkpoint_path,
        "exp_name": exp_name,
        "minified_json": minified_json,
    }

    # Generate command
    command = command_builder.build_predict_command(config)

    st.subheader("Generated Command")
    st.code(command, language="bash")

    # Validation
    is_valid, error_msg = command_builder.validate_command(command)
    if not is_valid:
        st.error(f"Command validation failed: {error_msg}")
    else:
        st.success("Command is valid")

        # Execute button
        if st.button("🔮 Execute Prediction", type="primary"):
            execute_command(command_builder, command)


def execute_command(command_builder: CommandBuilder, command: str):
    """Execute a command and display results."""
    with st.spinner("Executing command..."):
        return_code, stdout, stderr = command_builder.execute_command(command)

    if return_code == 0:
        st.success("Command executed successfully!")

        if stdout:
            with st.expander("Standard Output"):
                st.code(stdout)

        if stderr:
            with st.expander("Standard Error (Warnings)"):
                st.code(stderr)
    else:
        st.error(f"Command failed with return code {return_code}")

        if stdout:
            with st.expander("Standard Output"):
                st.code(stdout)

        if stderr:
            with st.expander("Standard Error"):
                st.code(stderr)


if __name__ == "__main__":
    main()
