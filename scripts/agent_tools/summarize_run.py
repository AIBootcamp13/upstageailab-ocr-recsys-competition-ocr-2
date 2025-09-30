# filename: scripts/agent_tools/summarize_run.py

import argparse
import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def call_llm_for_summary(text_content: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to summarize logs.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": f"Summarize the following agent run log:\n\n{text_content}"}]
    )
    content = response.choices[0].message.content
    if content is None:  # pragma: no cover - defensive guard for API changes
        raise RuntimeError("LLM response did not contain summary content.")
    return content


def summarize_log_file(log_path: Path):
    """
    Reads a JSONL log file, formats its content for an LLM,
    and generates a Markdown summary.
    """
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    # Read the JSONL file line by line
    actions = []
    with open(log_path) as f:
        for line in f:
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")
                continue

    if not actions:
        print("Error: No valid actions found in log file.")
        return

    # Format the log content into a single string for the LLM prompt.
    # This step is crucial for providing clear context to the summarization model.
    formatted_log_content = "Agent Run Log:\n\n"
    for i, action in enumerate(actions, 1):
        formatted_log_content += f"Step {i}:\n"
        formatted_log_content += f"- Thought: {action.get('thought', 'N/A')}\n"
        formatted_log_content += f"- Action: {action.get('action', 'N/A')}\n"
        if "parameters" in action:
            formatted_log_content += f"- Parameters: {json.dumps(action['parameters'])}\n"
        formatted_log_content += f"- Outcome: {action.get('outcome', 'N/A')}\n"
        formatted_log_content += f"- Output Snippet: {action.get('output_snippet', 'N/A')}\n\n"

    # Call the LLM to generate the summary
    summary_md = call_llm_for_summary(formatted_log_content)

    # Determine the output path for the summary
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"run_summary_{timestamp}.md"
    # Assuming this script is run from the project root
    output_path = Path("docs/ai_handbook/04_experiments") / summary_filename

    # Save the summary to a Markdown file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary_md)

    print(f"Successfully generated summary at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize an agent's run log using an LLM.")
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to the .jsonl log file to be summarized.",
    )
    args = parser.parse_args()

    summarize_log_file(args.log_file)
