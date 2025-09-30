from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SliderConfig:
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


@dataclass
class AppSection:
    title: str
    subtitle: str
    page_icon: str = "🧩"
    layout: str = "centered"
    initial_sidebar_state: str = "auto"


@dataclass
class ModelSelectorConfig:
    sort_by: list[str] = field(default_factory=lambda: ["architecture", "backbone"])
    demo_label: str = "No trained models found - using Demo Mode"
    success_message: str = ""
    unavailable_message: str = ""
    empty_message: str = ""


@dataclass
class UploadConfig:
    enabled_file_types: list[str] = field(default_factory=lambda: ["jpg", "jpeg", "png"])
    multi_file_selection: bool = True
    immediate_inference_for_single: bool = True


@dataclass
class ResultsConfig:
    expand_first_result: bool = True
    show_summary: bool = True
    show_raw_predictions: bool = True
    image_width: str = "stretch"


@dataclass
class NotificationConfig:
    inference_complete_delay_seconds: float = 1.0


@dataclass
class PathConfig:
    outputs_dir: Path = Path("outputs")
    hydra_config_filenames: list[str] = field(
        default_factory=lambda: [
            "config.yaml",
            "hparams.yaml",
            "train.yaml",
            "predict.yaml",
        ]
    )


@dataclass
class PreprocessingConfig:
    enable_label: str = "Enable docTR preprocessing"
    enable_help: str = "Run docTR geometry (orientation, rcrops, padding cleanup) before inference and capture visuals."
    default_enabled: bool = False
    enable_document_detection: bool = True
    enable_perspective_correction: bool = True
    enable_enhancement: bool = True
    enhancement_method: str = "office_lens"
    enable_text_enhancement: bool = True
    target_size: tuple[int, int] = (640, 640)
    enable_orientation_correction: bool = True
    orientation_angle_threshold: float = 1.0
    orientation_expand_canvas: bool = True
    orientation_preserve_original_shape: bool = False
    use_doctr_geometry: bool = True
    doctr_assume_horizontal: bool = False
    enable_padding_cleanup: bool = True
    show_metadata: bool = True
    show_corner_overlay: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "enable_document_detection": self.enable_document_detection,
            "enable_perspective_correction": self.enable_perspective_correction,
            "enable_enhancement": self.enable_enhancement,
            "enhancement_method": self.enhancement_method,
            "enable_text_enhancement": self.enable_text_enhancement,
            "target_size": self.target_size,
            "enable_orientation_correction": self.enable_orientation_correction,
            "orientation_angle_threshold": self.orientation_angle_threshold,
            "orientation_expand_canvas": self.orientation_expand_canvas,
            "orientation_preserve_original_shape": self.orientation_preserve_original_shape,
            "use_doctr_geometry": self.use_doctr_geometry,
            "doctr_assume_horizontal": self.doctr_assume_horizontal,
            "enable_padding_cleanup": self.enable_padding_cleanup,
        }


@dataclass
class UIConfig:
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
