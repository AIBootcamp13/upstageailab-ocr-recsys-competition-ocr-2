"""
Export utilities for Streamlit Preprocessing Viewer.

Provides functionality to export preprocessing configurations
as Hydra-compatible YAML files for direct integration with the main OCR pipeline.
"""

from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from omegaconf import OmegaConf

# Import viewer contracts for validation
from .viewer_contracts import validate_viewer_config


class ConfigExporter:
    """
    Exports preprocessing configurations in Hydra-compatible YAML format.

    Ensures exported configurations can be directly used in training and prediction pipelines.
    """

    def __init__(self, export_dir: str = "exported_configs"):
        """
        Initialize config exporter.

        Args:
            export_dir: Directory to store exported configurations
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)

    @validate_viewer_config
    def export_to_yaml(self, config: dict[str, Any], filename: str) -> bool:
        """
        Export configuration to Hydra-compatible YAML.

        Args:
            config: Preprocessing configuration dictionary
            filename: Output filename (without .yaml extension)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to OmegaConf for validation
            omega_config = OmegaConf.create(config)

            # Create full file path
            yaml_file = self.export_dir / f"{filename}.yaml"

            # Export to YAML
            yaml_str = OmegaConf.to_yaml(omega_config)

            with open(yaml_file, "w") as f:
                f.write(yaml_str)

            return True

        except Exception as e:
            st.error(f"Error exporting configuration: {e}")
            return False

    def get_export_path(self, filename: str) -> Path:
        """Get full path for exported configuration file."""
        return self.export_dir / f"{filename}.yaml"


def render_export_interface(config_exporter: ConfigExporter, current_config: dict[str, Any]) -> None:
    """
    Render the configuration export interface.

    Args:
        config_exporter: ConfigExporter instance
        current_config: Current preprocessing configuration
    """
    st.subheader("📤 Export Configuration")

    st.markdown("""
    Export the current preprocessing configuration as a Hydra-compatible YAML file.
    This configuration can be directly used in your training and prediction pipelines.
    """)

    # Configuration preview
    with st.expander("Preview Configuration", expanded=False):
        st.code(yaml.dump(current_config, default_flow_style=False), language="yaml")

    # Export options
    col1, col2 = st.columns([2, 1])

    with col1:
        export_name = st.text_input("Configuration Name", value="preprocessing_config", help="Name for the exported configuration file")

    with col2:
        if st.button("Export to YAML", key="export_yaml", type="primary"):
            if config_exporter.export_to_yaml(current_config, export_name):
                export_path = config_exporter.get_export_path(export_name)
                st.success("✅ Configuration exported successfully!")
                st.info(f"📁 File saved to: `{export_path}`")

                # Provide download link
                with open(export_path) as f:
                    yaml_content = f.read()

                st.download_button(
                    label="📥 Download YAML File",
                    data=yaml_content,
                    file_name=f"{export_name}.yaml",
                    mime="application/x-yaml",
                    key="download_yaml",
                )
            else:
                st.error("❌ Failed to export configuration")

    # Integration instructions
    st.markdown("---")
    st.subheader("🔗 Integration Instructions")

    st.markdown("""
    **To use this configuration in your OCR pipeline:**

    1. **Training Pipeline:**
       ```bash
       python run.py +preprocessing=<your_config_name>
       ```

    2. **Prediction Pipeline:**
       ```bash
       python predict.py +preprocessing=<your_config_name>
       ```

    3. **Custom Integration:**
       ```python
       from omegaconf import OmegaConf
       config = OmegaConf.load("exported_configs/your_config.yaml")
       # Use config in your preprocessing pipeline
       ```

    **Configuration Structure:**
    - All parameters follow the same naming convention as the main pipeline
    - Boolean flags control which preprocessing steps are enabled
    - Numeric parameters can be tuned for your specific use case
    - The configuration is fully compatible with Hydra's configuration system
    """)

    # Validation info
    st.markdown("---")
    st.subheader("✅ Configuration Validation")

    try:
        # Validate with OmegaConf
        OmegaConf.create(current_config)
        st.success("✅ Configuration is valid and Hydra-compatible")

        # Show key statistics
        enabled_steps = [k for k, v in current_config.items() if isinstance(v, bool) and v and k.startswith("enable_")]
        st.info(f"📊 Enabled preprocessing steps: {len(enabled_steps)}")

    except Exception as e:
        st.error(f"❌ Configuration validation failed: {e}")
