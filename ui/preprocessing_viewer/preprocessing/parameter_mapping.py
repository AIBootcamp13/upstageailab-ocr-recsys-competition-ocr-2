"""
Parameter Mapping for Streamlit Preprocessing Viewer.

This module provides comprehensive parameter definitions and mappings for all
preprocessing modules, enabling dynamic UI generation and validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UIControlType(str, Enum):
    """UI control types for parameter rendering."""

    SLIDER = "slider"
    CHECKBOX = "checkbox"
    SELECTBOX = "selectbox"
    RADIO = "radio"
    NUMBER_INPUT = "number_input"
    TEXT_INPUT = "text_input"
    MULTISELECT = "multiselect"


class UIParameterDefinition(BaseModel):
    """Definition of a UI parameter for dynamic control generation."""

    key: str = Field(..., description="Parameter key/attribute name")
    label: str = Field(..., description="Human-readable label")
    type: UIControlType = Field(..., description="UI control type")
    default: Any = Field(..., description="Default value")
    help: str | None = Field(default=None, description="Help text/tooltips")

    # Control-specific options
    min_value: float | None = Field(default=None, description="Minimum value for sliders/numbers")
    max_value: float | None = Field(default=None, description="Maximum value for sliders/numbers")
    step: float | None = Field(default=None, description="Step size for sliders/numbers")
    options: list[str] | None = Field(default=None, description="Options for selectbox/multiselect/radio")
    multiple: bool = Field(default=False, description="Allow multiple selections for selectbox")

    # Validation
    required: bool = Field(default=False, description="Whether parameter is required")
    validation_rules: dict[str, Any] = Field(default_factory=dict, description="Additional validation rules")

    # Dependencies
    depends_on: str | None = Field(default=None, description="Parameter this depends on")
    show_when: dict[str, Any] | None = Field(default=None, description="Conditions to show this parameter")


class ProcessingStage(str, Enum):
    """Processing stages for parameter organization."""

    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class ModuleParameterMapping(BaseModel):
    """Parameter mapping for a preprocessing module."""

    module_name: str = Field(..., description="Name of the preprocessing module")
    stage: ProcessingStage = Field(..., description="Processing stage this module belongs to")
    description: str = Field(..., description="Human-readable description")
    parameters: list[UIParameterDefinition] = Field(default_factory=list, description="Parameter definitions")
    enabled_by_default: bool = Field(default=True, description="Whether this module is enabled by default")
    enable_parameter: str | None = Field(default=None, description="Parameter that enables this module")


class PreprocessingParameterMapping:
    """Central registry of all preprocessing module parameters."""

    def __init__(self):
        self._mappings: dict[str, ModuleParameterMapping] = {}
        self._load_parameter_mappings()

    def _load_parameter_mappings(self):
        """Load parameter mappings for all preprocessing modules."""

        # Document Preprocessor Config (Main Pipeline)
        self._mappings["document_preprocessor"] = ModuleParameterMapping(
            module_name="document_preprocessor",
            stage=ProcessingStage.INITIAL,
            description="Main document preprocessing pipeline configuration",
            parameters=[
                UIParameterDefinition(
                    key="enable_document_detection",
                    label="Enable Document Detection",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Enable automatic document boundary detection",
                ),
                UIParameterDefinition(
                    key="enable_perspective_correction",
                    label="Enable Perspective Correction",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Enable perspective distortion correction",
                ),
                UIParameterDefinition(
                    key="enable_enhancement",
                    label="Enable Enhancement",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Enable image enhancement techniques",
                ),
                UIParameterDefinition(
                    key="enhancement_method",
                    label="Enhancement Method",
                    type=UIControlType.SELECTBOX,
                    default="conservative",
                    options=["conservative", "moderate", "aggressive"],
                    help="Enhancement algorithm to use",
                ),
                UIParameterDefinition(
                    key="enable_orientation_correction",
                    label="Enable Orientation Correction",
                    type=UIControlType.CHECKBOX,
                    default=False,
                    help="Enable automatic orientation correction",
                ),
                UIParameterDefinition(
                    key="orientation_angle_threshold",
                    label="Orientation Angle Threshold",
                    type=UIControlType.SLIDER,
                    default=2.0,
                    min_value=0.0,
                    max_value=45.0,
                    step=0.5,
                    help="Minimum angle threshold for orientation correction (degrees)",
                    depends_on="enable_orientation_correction",
                ),
                UIParameterDefinition(
                    key="document_detection_min_area_ratio",
                    label="Min Document Area Ratio",
                    type=UIControlType.SLIDER,
                    default=0.18,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    help="Minimum area ratio for valid document detection",
                ),
                UIParameterDefinition(
                    key="document_detection_use_adaptive",
                    label="Use Adaptive Thresholding",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Use adaptive thresholding for document detection",
                ),
                UIParameterDefinition(
                    key="document_detection_use_camscanner",
                    label="Use CamScanner Detection",
                    type=UIControlType.CHECKBOX,
                    default=False,
                    help="Use CamScanner-style document detection",
                ),
            ],
        )

        # Color preprocessing
        self._mappings["color_preprocessing"] = ModuleParameterMapping(
            module_name="color_preprocessing",
            stage=ProcessingStage.INITIAL,
            description="Color preprocessing options including grayscale and inversion",
            parameters=[
                UIParameterDefinition(
                    key="enable_color_preprocessing",
                    label="Enable Color Preprocessing",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Enable color preprocessing stages before detection",
                ),
                UIParameterDefinition(
                    key="convert_to_grayscale",
                    label="Convert to Grayscale",
                    type=UIControlType.CHECKBOX,
                    default=False,
                    help="Convert the image to grayscale prior to detection",
                    depends_on="enable_color_preprocessing",
                ),
                UIParameterDefinition(
                    key="color_inversion",
                    label="Invert Colors",
                    type=UIControlType.CHECKBOX,
                    default=False,
                    help="Invert image colors to highlight text regions",
                    depends_on="enable_color_preprocessing",
                ),
            ],
        )

        # Brightness Adjustment
        self._mappings["brightness"] = ModuleParameterMapping(
            module_name="brightness",
            stage=ProcessingStage.FINAL,
            description="Intelligent brightness adjustment and contrast enhancement",
            enable_parameter="enable_brightness_adjustment",
            parameters=[
                UIParameterDefinition(
                    key="enable_brightness_adjustment",
                    label="Enable Brightness Adjustment",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Enable intelligent brightness adjustment",
                ),
                UIParameterDefinition(
                    key="brightness_method",
                    label="Brightness Method",
                    type=UIControlType.SELECTBOX,
                    default="auto",
                    options=["adaptive_histogram", "gamma_correction", "clahe", "content_aware", "auto"],
                    help="Brightness adjustment method to use",
                    depends_on="enable_brightness_adjustment",
                ),
                UIParameterDefinition(
                    key="clahe_clip_limit",
                    label="CLAHE Clip Limit",
                    type=UIControlType.SLIDER,
                    default=2.0,
                    min_value=1.0,
                    max_value=5.0,
                    step=0.1,
                    help="CLAHE clipping limit",
                    depends_on="enable_brightness_adjustment",
                    show_when={"brightness_method": ["clahe", "auto"]},
                ),
                UIParameterDefinition(
                    key="gamma_value",
                    label="Gamma Value",
                    type=UIControlType.SLIDER,
                    default=1.0,
                    min_value=0.5,
                    max_value=2.0,
                    step=0.1,
                    help="Gamma correction value (1.0 = no change)",
                    depends_on="enable_brightness_adjustment",
                    show_when={"brightness_method": ["gamma_correction", "auto"]},
                ),
                UIParameterDefinition(
                    key="auto_gamma",
                    label="Auto Gamma Estimation",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Automatically estimate optimal gamma value",
                    depends_on="enable_brightness_adjustment",
                    show_when={"brightness_method": ["gamma_correction", "auto"]},
                ),
                UIParameterDefinition(
                    key="preserve_text_regions",
                    label="Preserve Text Regions",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Preserve text regions during brightness adjustment",
                    depends_on="enable_brightness_adjustment",
                ),
                UIParameterDefinition(
                    key="target_mean_brightness",
                    label="Target Mean Brightness",
                    type=UIControlType.SLIDER,
                    default=180.0,
                    min_value=0.0,
                    max_value=255.0,
                    step=5.0,
                    help="Target mean brightness value",
                    depends_on="enable_brightness_adjustment",
                ),
            ],
        )

        # Noise Elimination
        self._mappings["noise_elimination"] = ModuleParameterMapping(
            module_name="noise_elimination",
            stage=ProcessingStage.INTERMEDIATE,
            description="Advanced noise elimination and shadow removal",
            enable_parameter="enable_noise_elimination",
            parameters=[
                UIParameterDefinition(
                    key="enable_noise_elimination",
                    label="Enable Noise Elimination",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Enable advanced noise elimination",
                ),
                UIParameterDefinition(
                    key="noise_method",
                    label="Noise Reduction Method",
                    type=UIControlType.SELECTBOX,
                    default="combined",
                    options=["adaptive_background", "shadow_removal", "morphological", "combined"],
                    help="Noise reduction method to use",
                    depends_on="enable_noise_elimination",
                ),
                UIParameterDefinition(
                    key="adaptive_block_size",
                    label="Adaptive Block Size",
                    type=UIControlType.NUMBER_INPUT,
                    default=15,
                    min_value=3,
                    help="Block size for adaptive thresholding (must be odd)",
                    depends_on="enable_noise_elimination",
                    show_when={"noise_method": ["adaptive_background", "combined"]},
                ),
                UIParameterDefinition(
                    key="shadow_detection_threshold",
                    label="Shadow Detection Threshold",
                    type=UIControlType.SLIDER,
                    default=0.7,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    help="Threshold for shadow detection",
                    depends_on="enable_noise_elimination",
                    show_when={"noise_method": ["shadow_removal", "combined"]},
                ),
                UIParameterDefinition(
                    key="shadow_removal_strength",
                    label="Shadow Removal Strength",
                    type=UIControlType.SLIDER,
                    default=0.8,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    help="Strength of shadow removal",
                    depends_on="enable_noise_elimination",
                    show_when={"noise_method": ["shadow_removal", "combined"]},
                ),
                UIParameterDefinition(
                    key="preserve_text_regions",
                    label="Preserve Text Regions",
                    type=UIControlType.CHECKBOX,
                    default=True,
                    help="Preserve text regions during noise elimination",
                    depends_on="enable_noise_elimination",
                ),
                UIParameterDefinition(
                    key="morph_kernel_size",
                    label="Morphological Kernel Size",
                    type=UIControlType.NUMBER_INPUT,
                    default=2,
                    min_value=1,
                    max_value=10,
                    help="Kernel size for morphological operations",
                    depends_on="enable_noise_elimination",
                    show_when={"noise_method": ["morphological", "combined"]},
                ),
            ],
        )

        # Document Flattening
        self._mappings["document_flattening"] = ModuleParameterMapping(
            module_name="document_flattening",
            stage=ProcessingStage.INTERMEDIATE,
            description="Document flattening and distortion correction",
            enable_parameter="enable_document_flattening",
            parameters=[
                UIParameterDefinition(
                    key="enable_document_flattening",
                    label="Enable Document Flattening",
                    type=UIControlType.CHECKBOX,
                    default=False,
                    help="Enable document flattening for crumpled paper",
                ),
                UIParameterDefinition(
                    key="flattening_method",
                    label="Flattening Method",
                    type=UIControlType.SELECTBOX,
                    default="thin_plate_spline",
                    options=["thin_plate_spline", "geometric", "adaptive"],
                    help="Document flattening algorithm to use",
                    depends_on="enable_document_flattening",
                ),
                UIParameterDefinition(
                    key="grid_size",
                    label="Grid Size",
                    type=UIControlType.NUMBER_INPUT,
                    default=20,
                    min_value=5,
                    max_value=100,
                    help="Grid size for surface estimation",
                    depends_on="enable_document_flattening",
                ),
                UIParameterDefinition(
                    key="smoothing_factor",
                    label="Smoothing Factor",
                    type=UIControlType.SLIDER,
                    default=0.1,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    help="Smoothing factor for surface interpolation",
                    depends_on="enable_document_flattening",
                ),
                UIParameterDefinition(
                    key="edge_preservation_strength",
                    label="Edge Preservation Strength",
                    type=UIControlType.SLIDER,
                    default=0.8,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    help="Strength of edge preservation during flattening",
                    depends_on="enable_document_flattening",
                ),
                UIParameterDefinition(
                    key="min_curvature_threshold",
                    label="Min Curvature Threshold",
                    type=UIControlType.SLIDER,
                    default=0.05,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    help="Minimum curvature to trigger flattening",
                    depends_on="enable_document_flattening",
                ),
            ],
        )

    def get_module_mapping(self, module_name: str) -> ModuleParameterMapping | None:
        """Get parameter mapping for a specific module."""
        return self._mappings.get(module_name)

    def get_all_mappings(self) -> dict[str, ModuleParameterMapping]:
        """Get all parameter mappings."""
        return self._mappings.copy()

    def get_mappings_by_stage(self, stage: ProcessingStage) -> dict[str, ModuleParameterMapping]:
        """Get parameter mappings for a specific processing stage."""
        return {name: mapping for name, mapping in self._mappings.items() if mapping.stage == stage}

    def get_ui_schema(self, module_name: str) -> dict[str, Any]:
        """Generate UI schema for a specific module."""
        mapping = self.get_module_mapping(module_name)
        if not mapping:
            return {}

        schema = {"module_name": mapping.module_name, "description": mapping.description, "stage": mapping.stage.value, "parameters": []}

        for param in mapping.parameters:
            param_schema = param.model_dump()
            schema["parameters"].append(param_schema)

        return schema

    def validate_parameter_value(self, module_name: str, param_key: str, value: Any) -> bool:
        """Validate a parameter value against its definition."""
        mapping = self.get_module_mapping(module_name)
        if not mapping:
            return False

        param_def = next((p for p in mapping.parameters if p.key == param_key), None)
        if not param_def:
            return False

        # Basic validation based on parameter definition
        try:
            if param_def.type == UIControlType.SLIDER:
                if param_def.min_value is not None and value < param_def.min_value:
                    return False
                if param_def.max_value is not None and value > param_def.max_value:
                    return False
            elif param_def.type in [UIControlType.SELECTBOX, UIControlType.RADIO]:
                if param_def.options and value not in param_def.options:
                    return False
            elif param_def.type == UIControlType.CHECKBOX:
                if not isinstance(value, bool):
                    return False
            return True
        except Exception:
            return False


# Global instance for easy access
parameter_mapping = PreprocessingParameterMapping()
