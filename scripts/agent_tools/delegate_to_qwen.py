#!/usr/bin/env python3
"""
Delegate coding tasks to Qwen Coder via stdin.

This script reads a prompt file and sends it to Qwen in yolo mode,
capturing the output and validating the result.

Usage:
    uv run python scripts/agent_tools/delegate_to_qwen.py \
        --prompt prompts/qwen/01_performance_profiler_callback.md \
        --validate

    # Or pipe directly
    cat prompts/qwen/01_performance_profiler_callback.md | qwen --yolo
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delegate task to Qwen Coder")
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Path to prompt file (e.g., prompts/qwen/01_*.md)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation commands after Qwen completes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for Qwen execution in seconds (default: 600)",
    )
    parser.add_argument(
        "--validation-timeout",
        type=int,
        default=60,
        help="Timeout for each validation command in seconds (default: 60)",
    )

    args = parser.parse_args()

    # Validate prompt file exists
    if not args.prompt.exists():
        print(f"Error: Prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    # Read prompt
    prompt_content = args.prompt.read_text()

    if args.verbose:
        print(f"📝 Reading prompt: {args.prompt}")
        print(f"📏 Prompt length: {len(prompt_content)} chars")

    # Send to Qwen via stdin
    print("🤖 Delegating task to Qwen Coder...")
    print(f"📋 Prompt: {args.prompt.name}")
    print("=" * 60)

    try:
        # Use qwen in yolo mode (assumes qwen is in PATH)
        result = subprocess.run(
            ["qwen", "--yolo"],
            input=prompt_content,
            text=True,
            capture_output=True,
            timeout=args.timeout,  # Configurable timeout
        )

        # Print Qwen's output
        print(result.stdout)

        if result.stderr:
            print("⚠️  Qwen stderr:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"❌ Qwen exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

        print("=" * 60)
        print("✅ Qwen completed successfully!")

        # Run validation if requested
        if args.validate:
            print("\n🔍 Running validation commands...")
            validate_implementation(args.prompt, verbose=args.verbose, timeout=args.validation_timeout)

    except subprocess.TimeoutExpired:
        print(f"❌ Qwen timed out after {args.timeout} seconds", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: 'qwen' command not found in PATH", file=sys.stderr)
        print("Please ensure Qwen Coder is installed and accessible", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def validate_implementation(prompt_path: Path, verbose: bool = False, timeout: int = 60):
    """
    Extract and run validation commands from the prompt file.

    Looks for validation section in the prompt and executes the commands.
    """
    prompt_content = prompt_path.read_text()

    # Extract validation commands (simple parsing)
    in_validation = False
    validation_commands = []

    for line in prompt_content.split("\n"):
        if "## Validation" in line or "### Run These Commands:" in line:
            in_validation = True
            continue

        if in_validation:
            # Stop at next section
            if line.startswith("##") and "Validation" not in line:
                break

            # Extract bash commands
            if line.strip().startswith("uv run") or line.strip().startswith("pytest"):
                cmd = line.strip().lstrip("# ").strip()
                if cmd:
                    validation_commands.append(cmd)

    if not validation_commands:
        print("⚠️  No validation commands found in prompt")
        return

    print(f"Found {len(validation_commands)} validation commands:")
    for i, cmd in enumerate(validation_commands, 1):
        print(f"  {i}. {cmd}")

    print()

    # Run each validation command
    all_passed = True
    for i, cmd in enumerate(validation_commands, 1):
        print(f"[{i}/{len(validation_commands)}] Running: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                print("  ✅ Passed")
                if verbose and result.stdout:
                    print(f"     {result.stdout[:200]}")
            else:
                print(f"  ❌ Failed (exit code {result.returncode})")
                all_passed = False
                if result.stderr:
                    print(f"     Error: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            print("  ❌ Timeout")
            all_passed = False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False

    print()
    if all_passed:
        print("🎉 All validations passed!")
    else:
        print("⚠️  Some validations failed - review may be needed")
        sys.exit(1)


if __name__ == "__main__":
    main()
