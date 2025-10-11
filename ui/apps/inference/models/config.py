from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class _ConfigBase(BaseModel):
    """Common configuration for UI config models."""

    model_config = ConfigDict(validate_assignment=True)


class SliderConfig(_ConfigBase):
    key: str
    label: str
    min: float
    max: float
    step: float
    default: float
    help: str = ""

    @classmethod
    def from_mapping(cls, key: str, data: dict[str, Any]) -> SliderConfig:
        return cls(
            key=key,
            label=data.get("label", key.replace("_", " ").title()),
            min=float(data.get("min", 0.0)),
            max=float(data.get("max", 1.0)),
            step=float(data.get("step", 0.1)),
            default=float(data.get("default", 0.5)),
            help=data.get("help", ""),
        )

    def cast_value(self, value: float | int) -> float:
        return float(value)

    def is_integer_domain(self) -> bool:
        values = (self.min, self.max, self.step, self.default)
        return all(float(v).is_integer() for v in values)


class AppSection(_ConfigBase):
    title: str
    subtitle: str
    page_icon: str = "🧩"
    layout: str = "centered"
    initial_sidebar_state: str = "auto"


class ModelSelectorConfig(_ConfigBase):
    sort_by: list[str] = ["architecture", "backbone"]
    demo_label: str = "No trained models found - using Demo Mode"
    success_message: str = ""
    unavailable_message: str = ""
    empty_message: str = ""


class UploadConfig(_ConfigBase):
    enabled_file_types: list[str] = ["jpg", "jpeg", "png"]
    multi_file_selection: bool = True
    immediate_inference_for_single: bool = True


class ResultsConfig(_ConfigBase):
    expand_first_result: bool = True
    show_summary: bool = True
    show_raw_predictions: bool = True
    image_width: str = "stretch"


class NotificationConfig(_ConfigBase):
    inference_complete_delay_seconds: float = 1.0


class PathConfig(_ConfigBase):
    outputs_dir: Path = Path("outputs")
    hydra_config_filenames: list[str] = [
        "config.yaml",
        "hparams.yaml",
        "train.yaml",
        "predict.yaml",
    ]


class PreprocessingConfig(_ConfigBase):
    enable_label: str = "Enable docTR preprocessing"
    enable_help: str = "Run docTR geometry (orientation, rcrops, padding cleanup) before inference and capture visuals."
    default_enabled: bool = False
    enable_document_detection: bool = True
    enable_perspective_correction: bool = True
    enable_enhancement: bool = True
    enhancement_method: str = "office_lens"
    target_size: tuple[int, int] = (640, 640)
    enable_final_resize: bool = True
    enable_orientation_correction: bool = True
    orientation_angle_threshold: float = 1.0
    orientation_expand_canvas: bool = True
    orientation_preserve_original_shape: bool = False
    use_doctr_geometry: bool = True
    doctr_assume_horizontal: bool = False
    enable_padding_cleanup: bool = True
    show_metadata: bool = True
    show_corner_overlay: bool = True
    document_detection_min_area_ratio: float = 0.18
    document_detection_use_adaptive: bool = True
    document_detection_use_fallback_box: bool = True
    document_detection_use_camscanner: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "enable_document_detection": self.enable_document_detection,
            "enable_perspective_correction": self.enable_perspective_correction,
            "enable_enhancement": self.enable_enhancement,
            "enhancement_method": self.enhancement_method,
            "target_size": self.target_size,
            "enable_final_resize": self.enable_final_resize,
            "enable_orientation_correction": self.enable_orientation_correction,
            "orientation_angle_threshold": self.orientation_angle_threshold,
            "orientation_expand_canvas": self.orientation_expand_canvas,
            "orientation_preserve_original_shape": self.orientation_preserve_original_shape,
            "use_doctr_geometry": self.use_doctr_geometry,
            "doctr_assume_horizontal": self.doctr_assume_horizontal,
            "enable_padding_cleanup": self.enable_padding_cleanup,
            "document_detection_min_area_ratio": self.document_detection_min_area_ratio,
            "document_detection_use_adaptive": self.document_detection_use_adaptive,
            "document_detection_use_fallback_box": self.document_detection_use_fallback_box,
            "document_detection_use_camscanner": self.document_detection_use_camscanner,
        }


class UIConfig(_ConfigBase):
    app: AppSection
    model_selector: ModelSelectorConfig
    hyperparameters: dict[str, SliderConfig]
    upload: UploadConfig
    results: ResultsConfig
    notifications: NotificationConfig
    paths: PathConfig
    preprocessing: PreprocessingConfig

    def slider(self, key: str) -> SliderConfig:
        return self.hyperparameters[key]
