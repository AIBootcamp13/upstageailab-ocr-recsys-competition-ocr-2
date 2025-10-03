import argparse
import subprocess
from pathlib import Path

# This script demonstrates how to automate context injection for an AI agent.
# It determines the task, fetches relevant documentation, and constructs
# a complete prompt for the LLM.


def get_bundle_for_task(task_description: str) -> str:
    """
    Determines the appropriate documentation bundle based on keywords in the task.
    This is a simple rule-based example. A more advanced version could use an LLM call.
    """
    task_lower = task_description.lower()
    if any(keyword in task_lower for keyword in ["ui", "streamlit", "interface"]):
        return "streamlit-maintenance"
    elif any(keyword in task_lower for keyword in ["refactor", "decouple", "cleanup"]):
        return "refactor"
    elif any(keyword in task_lower for keyword in ["debug", "fix", "error", "bug"]):
        return "debugging"
    elif any(keyword in task_lower for keyword in ["train", "experiment", "model", "run"]):
        return "training-analysis"
    else:
        # A safe, general-purpose default bundle
        return "onboarding"


def fetch_context_files(bundle: str) -> list[Path]:
    """
    Calls the existing get_context.py script to get the file paths for a given bundle.
    """
    print(f"--- Fetching context files for bundle: '{bundle}' ---")
    script_path = Path(__file__).parent / "scripts/agent_tools/get_context.py"
    try:
        # We run the script and capture its output.
        # Note: The output of get_context.py is human-readable text, not JSON.
        # A more robust implementation would have get_context.py output JSON.
        result = subprocess.run(
            ["uv", "run", "python", str(script_path), "--bundle", bundle],
            capture_output=True,
            text=True,
            check=True,
        )

        # Simple parsing of the text output to find file paths.
        paths = []
        for line in result.stdout.splitlines():
            if line.strip().startswith("path:"):
                path_str = line.split("path:", 1)[1].strip()
                # Assuming paths in index.md are relative to docs/ai_handbook/
                full_path = Path("docs/ai_handbook") / path_str
                if full_path.exists():
                    paths.append(full_path)

        print(f"--- Found {len(paths)} context files. ---\n")
        return paths

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: Could not fetch context for bundle '{bundle}'. {e}")
        return []


def construct_final_prompt(task_description: str, context_files: list[Path]) -> str:
    """
    Constructs the final prompt to be sent to the LLM, including the fetched context.
    """
    print("--- Constructing final prompt for LLM ---")

    context_content = ""
    if context_files:
        context_content += "You MUST use the following documentation as your primary source of truth for this project. Adhere to all protocols and conventions described within.\n\n"
        for path in context_files:
            context_content += f"--- START OF DOCUMENT: {path} ---\n\n"
            context_content += path.read_text()
            context_content += f"\n\n--- END OF DOCUMENT: {path} ---\n\n"

    final_prompt = f"""
{context_content}
Based on the provided documentation, complete the following task.
Think step-by-step and use the tools and protocols mentioned in the context.

TASK: "{task_description}"
"""
    print("--- Prompt construction complete. ---\n")
    return final_prompt


def main():
    parser = argparse.ArgumentParser(description="Demo of automated context injection for an Agentic AI.")
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        help="The high-level task for the AI (e.g., 'refactor the dataloader').",
    )
    args = parser.parse_args()

    # 1. Determine the context bundle from the task description.
    bundle = get_bundle_for_task(args.task)

    # 2. Fetch the relevant documentation file paths for that bundle.
    context_files = fetch_context_files(bundle)

    # 3. Read the content of those files and construct the final prompt.
    final_prompt = construct_final_prompt(args.task, context_files)

    # 4. (Simulation) Print the final prompt that would be sent to the LLM.
    print("================ FINAL PROMPT TO LLM (SIMULATED) ================")
    print(final_prompt)
    print("======================== END OF PROMPT ========================")


if __name__ == "__main__":
    main()
