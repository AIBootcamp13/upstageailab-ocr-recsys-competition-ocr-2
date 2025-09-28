# filename: scripts/agent_tools/summarize_run.py

import argparse
import datetime
import json
from pathlib import Path


# --- Placeholder for your LLM API call ---
# You would replace this with your actual client, e.g., from openai, anthropic, or a custom library.
def call_llm_for_summary(text_content: str) -> str:
    """
    Placeholder function to simulate calling a Large Language Model.
    In a real implementation, this would make an API call to a model
    to generate a summary of the provided text.
    """
    print("--- SIMULATING LLM CALL ---")
    # In a real scenario, you would construct a prompt and send `text_content` to the LLM.
    # For example:
    # prompt = f"Please summarize the following agent run log into a concise, human-readable report... {text_content}"
    # response = your_llm_client.generate(prompt)
    # return response.text
    summary = (
        "# Agent Run Summary\n\n"
        "## Objective\n"
        "The agent attempted to debug a CUDA out-of-memory error.\n\n"
        "## Key Actions\n"
        "- **Hypothesis 1:** The UNet decoder channel mismatch was the root cause.\n"
        "- **Test:** Modified `unet.yaml` to set `output_channels: 256`.\n"
        "- **Observation:** This fixed the channel error but introduced a new CUDA memory error.\n"
        "- **Hypothesis 2:** The batch size was too large for the corrected feature maps.\n"
        "- **Test:** Reduced `data.batch_size` from 16 to 8.\n"
        "- **Observation:** The training run completed successfully.\n\n"
        "## Conclusion\n"
        "The root cause was a combination of an incorrect channel configuration and an overly large batch size for the resulting feature map dimensions. The issue is now resolved.\n\n"
        "## Status\n"
        "Completed"
    )
    print("--- SIMULATION COMPLETE ---")
    return summary


# -----------------------------------------


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
