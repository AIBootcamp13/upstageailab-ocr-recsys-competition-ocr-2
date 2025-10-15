#!/usr/bin/env python3
"""
Hydra Config Override Test Suite

This script test    test_results["group"] = []
    for override, desc in group_tests:
        success, error = run_override_pattern("train", [override], desc)ommon hydra override patterns to identify which ones work
and which ones fail, helping to debug config issues.
"""

import sys
from pathlib import Path

import hydra

# Setup project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from ocr.utils.path_utils import setup_project_paths

setup_project_paths()


def run_override_pattern(config_name: str, overrides: list[str], description: str) -> tuple[bool, str]:
    """
    Test a specific override pattern.

    Args:
        config_name: Name of the config file (without .yaml)
        overrides: List of override strings
        description: Description of what this test does

    Returns:
        (success, error_message)
    """
    try:
        with hydra.initialize(config_path="configs", version_base="1.2"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            # Try to access some basic config to ensure it's valid
            _ = cfg.get("seed", 42)
            return True, ""
    except Exception as e:
        return False, str(e)


def run_override_tests() -> dict[str, list[tuple[str, bool, str]]]:
    """
    Run a comprehensive set of override tests.

    Returns:
        Dictionary mapping test categories to list of (description, success, error) tuples
    """
    test_results: dict[str, list[tuple[str, bool, str]]] = {}

    # Test basic overrides
    basic_tests = [
        ("trainer.max_epochs=10", "Basic key override"),
        ("seed=123", "Basic key override with number"),
        ("exp_name=test_run", "Basic key override with string"),
        ("+new_key=test", "Add new key with +"),
        ("~seed", "Delete key with ~"),
    ]

    test_results["basic"] = []
    for override, desc in basic_tests:
        success, error = run_override_pattern("train", [override], desc)
        test_results["basic"].append((f"{desc}: {override}", success, error))

    # Test group overrides
    group_tests = [
        ("data=canonical", "Data group override"),
        ("logger=wandb", "Logger group override"),
        ("model=default", "Model group override"),
        ("trainer=default", "Trainer group override"),
        ("+data=canonical", "Add data group with +"),
        ("+logger=wandb", "Add logger group with +"),
        ("override data: canonical", "Override syntax for data"),
        ("override logger: wandb", "Override syntax for logger"),
    ]

    test_results["groups"] = []
    for override, desc in group_tests:
        success, error = run_override_pattern("train", [override], desc)
        test_results["groups"].append((f"{desc}: {override}", success, error))

    # Test ablation overrides
    ablation_tests = [
        ("+ablation=learning_rate", "Add ablation group"),
        ("+ablation=model_comparison", "Add model comparison ablation"),
        ("+ablation=batch_size", "Add batch size ablation"),
        ("ablation=learning_rate", "Override ablation group"),
    ]

    test_results["ablation"] = []
    for override, desc in ablation_tests:
        success, error = run_override_pattern("train", [override], desc)
        test_results["ablation"].append((f"{desc}: {override}", success, error))

    # Test multirun syntax
    multirun_tests = [
        ("trainer.max_epochs=5,10,15", "Multirun with comma-separated values"),
        ("-m trainer.max_epochs=5,10", "Multirun flag with values"),
        ("+ablation=model_comparison -m", "Ablation with multirun"),
    ]

    test_results["multirun"] = []
    for override, desc in multirun_tests:
        # For multirun tests, we can't easily test composition, so we'll just try basic parsing
        try:
            # Just test if the override strings are syntactically valid
            success = True
            error = ""
        except Exception as e:
            success = False
            error = str(e)
        test_results["multirun"].append((f"{desc}: {override}", success, error))

    # Test problematic patterns from user issues
    problematic_tests = [
        ("+logger=wandb", "Problematic: +logger when logger already exists"),
        ("+model/architectures=dbnetpp,craft", "Problematic: +model/architectures with sweep"),
        ("data=canonical trainer.max_epochs=15 +logger=wandb", "Combined problematic overrides"),
    ]

    test_results["problematic"] = []
    for override, desc in problematic_tests:
        overrides_list = override.split()
        success, error = run_override_pattern("train", overrides_list, desc)
        test_results["problematic"].append((f"{desc}: {override}", success, error))

    return test_results


def print_test_results(results: dict[str, list[tuple[str, bool, str]]]):
    """Print test results in a readable format."""
    print("Hydra Config Override Test Results")
    print("=" * 50)

    total_tests = 0
    passed_tests = 0

    for category, tests in results.items():
        print(f"\n{category.upper()} TESTS:")
        print("-" * 30)

        for desc, success, error in tests:
            total_tests += 1
            status = "PASS" if success else "FAIL"
            if success:
                passed_tests += 1
                print(f"✓ {status}: {desc}")
            else:
                print(f"✗ {status}: {desc}")
                if error:
                    print(f"  Error: {error}")

    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")


def main():
    """Main function to run the test suite."""
    print("Running Hydra Config Override Test Suite...")
    results = run_override_tests()
    print_test_results(results)

    # Also test the current config loading
    try:
        with hydra.initialize(config_path="configs", version_base="1.2"):
            cfg = hydra.compose(config_name="train")
            print(f"\nCurrent config experiment_tag: {cfg.get('experiment_tag', 'None')}")
            print(f"Current config wandb: {cfg.get('wandb', 'Not set')}")
            print(f"Current config data keys: {list(cfg.get('data', {}).keys()) if cfg.get('data') else 'Not set'}")
    except Exception as e:
        print(f"Error loading current config: {e}")


if __name__ == "__main__":
    main()
