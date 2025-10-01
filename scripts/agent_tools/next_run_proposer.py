import argparse
import sys

import wandb

# Configure the Weights & Biases API
# In a real environment, you might configure this differently
# For now, this assumes the user is logged in via the CLI
api = wandb.Api()


def analyze_run(run_id: str):
    """
    Analyzes a Weights & Biases run and proposes the next experiment.

    Args:
        run_id: The ID of the W&B run to analyze (e.g., "entity/project/runid").
    """
    try:
        run = api.run(run_id)
        summary = run.summary
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}

        print("## ✅ Run Analysis Complete")
        print(f"**Run Name:** {run.name}")
        print(f"**W&B Link:** {run.url}\n")

        # --- Rule-Based Analysis ---
        recall = summary.get("test/recall", 0)
        precision = summary.get("test/precision", 0)
        summary.get("val/loss", float("inf"))

        recommendations = []

        # Rule 1: High Precision, Low Recall -> Adjust post-processing
        if recall < 0.6 and precision > 0.9:
            current_box_thresh = config.get("model", {}).get("head", {}).get("postprocess", {}).get("box_thresh", 0.4)
            new_box_thresh = round(current_box_thresh - 0.1, 2)
            recommendations.append(
                {
                    "priority": 1,
                    "hypothesis": f"The model is too conservative (Precision={precision:.2f}, Recall={recall:.2f}). Lowering the bounding box threshold should increase recall.",
                    "override": f"model.head.postprocess.box_thresh={new_box_thresh}",
                }
            )

        # Rule 2: Stagnant Validation Loss -> Change Scheduler
        # This is a simplified check. A real implementation would look at the loss curve.
        if "val/loss" in summary and "StepLR" in config.get("model", {}).get("scheduler", {}).get("_target_", ""):
            recommendations.append(
                {
                    "priority": 2,
                    "hypothesis": "The validation loss is not improving significantly. A dynamic learning rate scheduler like CosineAnnealingLR may help escape local minima.",
                    "override": "model.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR model.scheduler.T_max=29",  # Assuming max_epochs
                }
            )

        # Rule 3: General low performance -> Suggest smaller learning rate
        if summary.get("val/hmean", 0) < 0.7:
            current_lr = config.get("model", {}).get("optimizer", {}).get("lr", 1e-3)
            new_lr = current_lr / 5
            recommendations.append(
                {
                    "priority": 3,
                    "hypothesis": "Overall performance is low. A smaller learning rate might lead to more stable convergence.",
                    "override": f"model.optimizer.lr={new_lr:.1e}",
                }
            )

        if not recommendations:
            print("No specific recommendations triggered. Consider manual analysis or architectural changes.")
            return

        # --- Output Proposed Next Run ---
        print("## 🚀 Proposed Next Experiment\n")
        # Sort by priority and take the top recommendation
        best_recommendation = sorted(recommendations, key=lambda x: x["priority"])[0]

        print("### Objective\n")
        print(f"* **Hypothesis:** {best_recommendation['hypothesis']}")
        print("\n### Configuration\n")
        print("**Key Overrides:**")
        print("```yaml")
        print(best_recommendation["override"])
        print("```\n")
        print("**Full Command:**")
        # Attempt to reconstruct the original command
        original_command = f"uv run python runners/train.py {' '.join(run.args)} {best_recommendation['override']}"
        print(f"```bash\n{original_command}\n```")

    except Exception as e:
        print(f"Error analyzing run {run_id}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a W&B run and propose the next experiment.")
    parser.add_argument("run_id", type=str, help="The Weights & Biases run ID (e.g., 'username/project-name/abcdef12').")
    args = parser.parse_args()
    analyze_run(args.run_id)
