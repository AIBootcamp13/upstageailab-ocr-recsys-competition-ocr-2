"""
Preset management system for Streamlit Preprocessing Viewer.

Provides functionality to load, save, and delete preprocessing configurations
as Hydra-compatible YAML files.
"""

from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from omegaconf import OmegaConf

# Import viewer contracts for validation
from .viewer_contracts import validate_viewer_config


class PresetManager:
    """
    Manages preprocessing configuration presets stored as YAML files.

    Presets are stored in a dedicated directory and can be loaded, saved,
    and deleted through the UI.
    """

    def __init__(self, presets_dir: str = "preprocessing_presets"):
        """
        Initialize preset manager.

        Args:
            presets_dir: Directory to store preset files
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(exist_ok=True)

    def list_presets(self) -> list[str]:
        """List all available preset names."""
        return [f.stem for f in self.presets_dir.glob("*.yaml")]

    def load_preset(self, name: str) -> dict[str, Any] | None:
        """
        Load a preset configuration.

        Args:
            name: Preset name (without .yaml extension)

        Returns:
            Configuration dictionary or None if not found
        """
        preset_file = self.presets_dir / f"{name}.yaml"
        if not preset_file.exists():
            return None

        try:
            with open(preset_file) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            st.error(f"Error loading preset '{name}': {e}")
            return None

    @validate_viewer_config
    def save_preset(self, name: str, config: dict[str, Any]) -> bool:
        """
        Save a preset configuration.

        Args:
            name: Preset name (without .yaml extension)
            config: Configuration dictionary to save

        Returns:
            True if successful, False otherwise
        """
        preset_file = self.presets_dir / f"{name}.yaml"

        try:
            # Convert to OmegaConf for validation and then back to dict for YAML
            omega_config = OmegaConf.create(config)
            yaml_str = OmegaConf.to_yaml(omega_config)

            with open(preset_file, "w") as f:
                f.write(yaml_str)

            return True
        except Exception as e:
            st.error(f"Error saving preset '{name}': {e}")
            return False

    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset configuration.

        Args:
            name: Preset name (without .yaml extension)

        Returns:
            True if successful, False otherwise
        """
        preset_file = self.presets_dir / f"{name}.yaml"
        if not preset_file.exists():
            st.error(f"Preset '{name}' not found")
            return False

        try:
            preset_file.unlink()
            return True
        except Exception as e:
            st.error(f"Error deleting preset '{name}': {e}")
            return False

    def get_default_config(self) -> dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            # Core processing flags
            "enable_document_detection": True,
            "enable_perspective_correction": True,
            "enable_enhancement": True,
            "enhancement_method": "conservative",
            # Color preprocessing
            "enable_color_preprocessing": True,
            "convert_to_grayscale": False,
            "color_inversion": False,
            # Advanced processing
            "enable_document_flattening": False,
            "enable_orientation_correction": False,
            "enable_noise_elimination": True,
            "enable_brightness_adjustment": True,
            # Size settings
            "target_size": [640, 640],
            "enable_final_resize": True,
            # Detection parameters
            "document_detection_min_area_ratio": 0.18,
            "document_detection_use_adaptive": True,
            "document_detection_use_fallback_box": True,
            # Orientation settings
            "orientation_angle_threshold": 2.0,
            "orientation_expand_canvas": True,
            "orientation_preserve_original_shape": False,
            # DOCTR settings
            "use_doctr_geometry": False,
            "doctr_assume_horizontal": False,
            # Advanced settings
            "enable_padding_cleanup": False,
        }


def render_preset_management(preset_manager: PresetManager) -> dict[str, Any] | None:
    """
    Render the preset management UI component.

    Args:
        preset_manager: PresetManager instance

    Returns:
        Selected preset configuration or None
    """
    st.subheader("🎛️ Preprocessing Presets")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Load preset
        available_presets = preset_manager.list_presets()
        if available_presets:
            selected_preset = st.selectbox("Load Preset", [""] + available_presets, help="Select a saved preset to load")

            if selected_preset and st.button("Load Preset", key="load_preset"):
                config = preset_manager.load_preset(selected_preset)
                if config:
                    st.success(f"Preset '{selected_preset}' loaded successfully!")
                    return config
                else:
                    st.error(f"Failed to load preset '{selected_preset}'")
        else:
            st.info("No presets available. Create your first preset below.")

    with col2:
        # Save preset
        preset_name = st.text_input("Preset Name", key="save_preset_name", help="Enter a name for the new preset")

    with col3:
        # Delete preset
        if available_presets:
            preset_to_delete = st.selectbox(
                "Delete Preset", [""] + available_presets, key="delete_preset_select", help="Select a preset to delete"
            )

            if preset_to_delete and st.button("Delete", key="delete_preset"):
                if preset_manager.delete_preset(preset_to_delete):
                    st.success(f"Preset '{preset_to_delete}' deleted successfully!")
                    st.rerun()  # Refresh the UI
                else:
                    st.error(f"Failed to delete preset '{preset_to_delete}'")

    # Save button (separate row for better layout)
    if preset_name and st.button("Save Current Config", key="save_preset"):
        # Get current config from session state
        current_config = st.session_state.get("current_config", preset_manager.get_default_config())

        if preset_manager.save_preset(preset_name, current_config):
            st.success(f"Preset '{preset_name}' saved successfully!")
            st.rerun()  # Refresh the UI
        else:
            st.error(f"Failed to save preset '{preset_name}'")

    return None
