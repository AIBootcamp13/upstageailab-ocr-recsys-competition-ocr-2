#!/usr/bin/env python3
"""
OCR Lightning Module Refactor CLI - YOLO Mode
Automated refactoring with parallel test generation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


class OCRRefactorCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent

    def run_command(self, cmd: str, description: str) -> bool:
        """Run command with status reporting."""
        print(f"🔧 {description}")
        try:
            result = subprocess.run(cmd, shell=True, cwd=self.project_root, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                return True
            else:
                print(f"❌ {description} - FAILED")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ {description} - ERROR: {e}")
            return False

    def phase_1_extract_utils(self) -> bool:
        """Execute Phase 1: Extract Utilities."""
        print("\n🎯 Phase 1: Extracting Utilities")

        commands = [
            ("mkdir -p ocr/lightning_modules/utils", "Create utils directory"),
            ("mkdir -p ocr/lightning_modules/processors", "Create processors directory"),
            ("mkdir -p ocr/lightning_modules/evaluators", "Create evaluators directory"),
            ("mkdir -p ocr/lightning_modules/loggers", "Create loggers directory"),
        ]

        success = True
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                success = False

        if success:
            print("📝 Phase 1 requires manual code extraction. See refactor plan for details.")
            print("🔍 Run: qwen --generate-tests --target ocr/lightning_modules/utils/config_utils.py")

        return success

    def phase_2_extract_evaluators(self) -> bool:
        """Execute Phase 2: Extract Evaluators."""
        print("\n🎯 Phase 2: Extracting Evaluators")

        print("📝 Phase 2 requires manual code extraction. See refactor plan for details.")
        print("🔍 Run: qwen --generate-tests --target ocr/lightning_modules/evaluators/cl_evaluator.py")

        # Run validation test
        return self.run_command(
            "python scripts/train.py --config-name=test trainer.max_epochs=1 trainer.limit_val_batches=5", "Quick validation test"
        )

    def phase_3_extract_loggers(self) -> bool:
        """Execute Phase 3: Extract Loggers."""
        print("\n🎯 Phase 3: Extracting Loggers")

        print("📝 Phase 3 requires manual code extraction. See refactor plan for details.")
        print("🔍 Run: qwen --generate-tests --target ocr/lightning_modules/loggers/")

        return True

    def phase_4_cleanup(self) -> bool:
        """Execute Phase 4: Clean Up."""
        print("\n🎯 Phase 4: Clean Up & Documentation")

        commands = [
            ("python -m pytest tests/unit/ -v --tb=short", "Run unit tests"),
            ("python -m pytest tests/integration/ -v --tb=short", "Run integration tests"),
        ]

        success = True
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                success = False

        if success:
            print("🎉 Refactor completed successfully!")
            print("📊 Run full test suite: python -m pytest tests/ -v --tb=short")

        return success

    def run_yolo_mode(self, start_phase: int = 1, stop_on_error: bool = True):
        """Run all phases in YOLO mode."""
        print("🚀 Starting OCR Lightning Module Refactor - YOLO Mode")
        print("=" * 60)

        phases = [
            ("Phase 1: Extract Utilities", self.phase_1_extract_utils),
            ("Phase 2: Extract Evaluators", self.phase_2_extract_evaluators),
            ("Phase 3: Extract Loggers", self.phase_3_extract_loggers),
            ("Phase 4: Clean Up", self.phase_4_cleanup),
        ]

        for i, (name, func) in enumerate(phases, 1):
            if i < start_phase:
                continue

            try:
                if func():
                    print(f"✅ {name} completed successfully")
                else:
                    print(f"❌ {name} failed")
                    if stop_on_error:
                        print("🛑 Stopping due to error (use --continue-on-error to continue)")
                        return False
            except Exception as e:
                print(f"💥 {name} crashed: {e}")
                if stop_on_error:
                    return False

        print("\n🎉 Refactor completed! Run full test suite:")
        print("python -m pytest tests/ -v --tb=short")
        return True


def main():
    parser = argparse.ArgumentParser(description="OCR Lightning Module Refactor CLI")
    parser.add_argument("--yolo", action="store_true", help="Run all phases automatically")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run specific phase")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if phase fails")
    parser.add_argument("--start-phase", type=int, default=1, help="Start from specific phase")

    args = parser.parse_args()

    cli = OCRRefactorCLI()

    if args.yolo:
        success = cli.run_yolo_mode(args.start_phase, not args.continue_on_error)
        sys.exit(0 if success else 1)
    elif args.phase:
        phase_methods = {
            1: cli.phase_1_extract_utils,
            2: cli.phase_2_extract_evaluators,
            3: cli.phase_3_extract_loggers,
            4: cli.phase_4_cleanup,
        }
        success = phase_methods[args.phase]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
