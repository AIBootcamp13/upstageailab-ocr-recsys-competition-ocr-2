from __future__ import annotations

from typing import Any

from ui.utils.config_parser import ConfigParser


def build_additional_overrides(values: dict[str, Any], config_parser: ConfigParser | None = None) -> list[str]:
    """Translate high-level UI toggles into concrete Hydra overrides."""

    cp = config_parser or ConfigParser()
    overrides: list[str] = []

    profile_key = values.get("preprocessing_profile")
    if isinstance(profile_key, str) and profile_key and profile_key != "none":
        profiles = cp.get_preprocessing_profiles()
        profile = profiles.get(profile_key)
        if isinstance(profile, dict):
            overrides.extend(profile.get("overrides", []))

    return overrides


def maybe_suffix_exp_name(overrides: list[str], values: dict[str, Any], append_suffix: bool) -> list[str]:
    """Optionally append architecture/encoder info to exp_name to avoid collisions."""

    if not append_suffix or not values.get("encoder"):
        return overrides

    if values.get("resume_training"):
        return overrides

    encoder = str(values.get("encoder"))
    architecture = str(values.get("architecture")) if values.get("architecture") else None
    decoder = str(values.get("decoder")) if values.get("decoder") else None

    new_overrides = list(overrides)
    for idx, ov in enumerate(new_overrides):
        if ov.startswith("exp_name="):
            base_name = ov.split("=", 1)[1]
            suffix_parts = [part for part in [architecture, decoder, encoder] if part]
            new_overrides[idx] = f"exp_name={base_name}-{'-'.join(suffix_parts)}" if suffix_parts else ov
            break
    else:
        # If exp_name override wasn't present, append with suffix
        base_name = values.get("exp_name", "ocr_training")
        suffix_parts = [part for part in [architecture, decoder, encoder] if part]
        suffix = f"-{'-'.join(suffix_parts)}" if suffix_parts else ""
        new_overrides.append(f"exp_name={base_name}{suffix}")
    return new_overrides
