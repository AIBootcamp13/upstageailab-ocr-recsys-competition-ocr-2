#!/usr/bin/env python3
"""Auto-generate and update the AI handbook index.json from directory structure."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Directory structure mapping
DIRECTORY_STRUCTURE = {
    "01_onboarding": {
        "purpose": "Contains introductory documents for new contributors (human or AI).",
        "naming_convention": "NN_descriptive_topic.md",
        "allowed_content": "Only Markdown (.md) documents.",
        "prohibited_content": "Code, outputs, logs, experimental notes.",
    },
    "02_protocols": {
        "purpose": "Actionable, step-by-step instructions for recurring tasks, organized by category.",
        "naming_convention": "NN_protocol_name.md",
        "allowed_content": "Only Markdown (.md) documents defining a process.",
        "prohibited_content": "Generated outputs, work-in-progress notes. Outputs from protocol execution belong in the root /outputs directory.",
        "subdirectories": {
            "development": "Core development practices and workflows",
            "components": "Component-specific protocols",
            "configuration": "Configuration management protocols",
            "governance": "Documentation and maintenance protocols",
        },
    },
    "03_references": {
        "purpose": "Factual, descriptive information about the project's architecture, tools, and data, organized by category.",
        "naming_convention": "NN_reference_topic.md",
        "allowed_content": "Only Markdown (.md) documents.",
        "prohibited_content": "Tutorials, step-by-step guides (these are protocols).",
        "subdirectories": {
            "architecture": "Core system architecture documentation",
            "guides": "Usage guides and tutorials",
            "integrations": "Third-party integration documentation",
            "workflows": "Workflow examples and patterns",
            "frameworks": "Development frameworks and methodologies",
        },
    },
    "04_experiments": {
        "purpose": "Logs and summaries of completed experiments, organized by type.",
        "naming_convention": "YYYY-MM-DD_experiment_summary.md or run_summary_TIMESTAMP.md",
        "allowed_content": "Markdown (.md) summaries of experimental results. Templates for logging are also allowed.",
        "prohibited_content": "Protocols, planning documents, raw output files.",
        "subdirectories": {
            "experiment_logs": "Core experiment documentation and results",
            "debugging": "Debug sessions and troubleshooting documentation",
            "sessions": "Session handovers and knowledge transfer",
            "agent_runs": "Automated agent activity logs",
        },
    },
    "05_changelog": {
        "purpose": "A chronological record of significant, completed changes to the project.",
        "naming_convention": "YYYY-MM-DD_description_of_change.md",
        "allowed_content": "Only Markdown (.md) documents summarizing completed work.",
        "prohibited_content": "Planning documents, work-in-progress notes, deprecated files.",
    },
    "06_concepts": {
        "purpose": "High-level explanations of the 'why' behind key design decisions and theories.",
        "naming_convention": "NN_concept_name.md",
        "allowed_content": "Only Markdown (.md) documents explaining concepts.",
        "prohibited_content": "Actionable protocols or factual references.",
    },
    "07_planning": {
        "purpose": "Planning documents, assessments, and operational logs for development activities.",
        "naming_convention": "NN_planning_topic.md",
        "allowed_content": "Only Markdown (.md) documents for planning purposes.",
        "prohibited_content": "Generated outputs, protocols, factual references.",
        "subdirectories": {
            "assessments": "AI agent and system assessments",
            "plans": "Development and optimization plans",
            "logs": "Delegation and operational logs",
            "data": "Data generation and management plans",
            "phases": "Phase-specific documentation and prompts",
        },
    },
}

# Priority mapping for different types of content
PRIORITY_MAP = {
    "high": ["training", "debugging", "architecture", "command", "registry"],
    "medium": ["workflow", "guide", "reference", "experiment", "template"],
    "low": ["utility", "adoption", "context", "checkpoint"],
}


def extract_title_from_file(file_path: Path) -> str:
    """Extract title from markdown file."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    # Remove markdown heading and clean up
                    title = line[2:].strip()
                    # Remove any markdown formatting
                    title = re.sub(r"\*\*([^*]+)\*\*", r"\1", title)
                    return title
    except Exception:
        pass
    # Fallback to filename
    return file_path.stem.replace("_", " ").replace("-", " ").title()


def determine_priority(filename: str, content_type: str) -> str:
    """Determine priority based on filename and content type."""
    filename_lower = filename.lower()

    for priority, keywords in PRIORITY_MAP.items():
        if any(keyword in filename_lower for keyword in keywords):
            return priority

    # Default priorities by content type
    if content_type in ["protocols", "architecture"]:
        return "high"
    elif content_type in ["references", "experiments"]:
        return "medium"
    else:
        return "low"


def determine_owner(filename: str) -> str:
    """Determine ownership based on filename."""
    filename_lower = filename.lower()

    if any(term in filename_lower for term in ["training", "experiment", "model"]):
        return "ml-platform"
    elif any(term in filename_lower for term in ["ui", "streamlit", "interface"]):
        return "frontend"
    elif any(term in filename_lower for term in ["debug", "logging", "monitoring"]):
        return "platform"
    elif any(term in filename_lower for term in ["command", "registry", "automation"]):
        return "automation"
    else:
        return "core-team"


def scan_directory(handbook_dir: Path) -> dict[str, Any]:
    """Scan the handbook directory and build the index structure."""
    entries = []
    bundles: dict[str, list[str]] = defaultdict(list)

    for dir_name in DIRECTORY_STRUCTURE.keys():
        dir_path = handbook_dir / dir_name
        if not dir_path.exists():
            continue

        # Scan for markdown files
        for md_file in dir_path.rglob("*.md"):
            if md_file.name.startswith(".") or md_file.name == "README.md":
                continue

            # Calculate relative path
            rel_path = md_file.relative_to(handbook_dir)
            path_str = f"./{rel_path}"

            # Extract metadata
            title = extract_title_from_file(md_file)
            filename = md_file.stem

            # Determine section and other metadata
            section = f"{dir_name.split('_')[1].title()} ({'How-To Guides' if dir_name == '02_protocols' else 'Factual Information' if dir_name == '03_references' else dir_name.split('_')[1].title()})"

            priority = determine_priority(filename, dir_name.split("_")[1])
            owner = determine_owner(filename)

            # Create entry ID from filename
            entry_id = filename.lower().replace(" ", "_").replace("-", "_")

            # Determine tags based on content
            tags = []
            filename_lower = filename.lower()
            if "debug" in filename_lower:
                tags.append("debugging")
            if "train" in filename_lower or "experiment" in filename_lower:
                tags.append("training")
            if "ui" in filename_lower or "streamlit" in filename_lower:
                tags.append("ui")
            if "config" in filename_lower or "hydra" in filename_lower:
                tags.append("configuration")
            if not tags:
                tags.append(dir_name.split("_")[1])

            entry = {
                "id": entry_id,
                "title": title,
                "path": path_str,
                "section": section,
                "tags": tags,
                "priority": priority,
                "summary": f"Documentation for {title.lower()}",
                "last_reviewed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "owner": owner,
                "bundles": [],
            }

            entries.append(entry)

            # Add to bundles based on tags
            for tag in tags:
                if tag not in bundles:
                    bundles[tag] = []
                bundles[tag].append(entry_id)

    # Convert bundles to the expected format (objects with entries arrays)
    formatted_bundles = {}
    for bundle_name, entry_ids in bundles.items():
        formatted_bundles[bundle_name] = {"entries": entry_ids}

    return {
        "source": "docs/ai_handbook/index.md",
        "version": "1.11 (2025-10-09)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": DIRECTORY_STRUCTURE,
        "entries": entries,
        "bundles": formatted_bundles,
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-generate AI handbook index.json from directory structure")
    parser.add_argument("--handbook-dir", type=Path, default=Path("docs/ai_handbook"), help="Path to the ai_handbook directory")
    parser.add_argument("--output", type=Path, default=Path("docs/ai_handbook/index.json"), help="Output path for the generated index.json")
    parser.add_argument("--validate", action="store_true", help="Run validation after generation")

    args = parser.parse_args()

    # Generate the index
    index_data = scan_directory(args.handbook_dir)

    # Write to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"Generated index.json with {len(index_data['entries'])} entries")

    # Run validation if requested
    if args.validate:
        import subprocess

        result = subprocess.run(["python", "scripts/agent_tools/validate_manifest.py", str(args.output)], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed:")
            print(result.stdout)
            print(result.stderr)


if __name__ == "__main__":
    main()
