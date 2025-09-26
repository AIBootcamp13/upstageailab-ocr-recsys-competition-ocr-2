import yaml

from ui.utils.ui_generator import compute_overrides
from ui.utils.ui_validator import validate_inputs


def test_compute_overrides_basic():
    schema = {
        "constant_overrides": ["foo=1"],
        "ui_elements": [
            {"key": "exp_name", "hydra_override": "exp_name"},
            {"key": "wandb", "hydra_override": "wandb"},
        ],
    }
    values = {"exp_name": "run1", "wandb": True}
    overrides, constants = compute_overrides(schema, values)
    assert "exp_name=run1" in overrides
    assert "wandb=true" in overrides
    assert constants == ["foo=1"]


def test_validate_inputs_required_and_required_if(tmp_path):
    schema = {
        "ui_elements": [
            {
                "key": "name",
                "label": "Name",
                "validation": {"required": True, "min_length": 3},
            },
            {
                "key": "resume",
                "label": "Resume",
                "validation": {},
            },
            {
                "key": "ckpt",
                "label": "Checkpoint",
                "validation": {"required_if": "resume == true"},
            },
        ]
    }
    p = tmp_path / "schema.yaml"
    p.write_text(yaml.safe_dump(schema))

    # Missing required name
    errors = validate_inputs({"name": ""}, str(p))
    assert any("Name" in e for e in errors)

    # Required_if triggered
    errors = validate_inputs({"name": "abc", "resume": True, "ckpt": ""}, str(p))
    assert any("Checkpoint" in e for e in errors)
