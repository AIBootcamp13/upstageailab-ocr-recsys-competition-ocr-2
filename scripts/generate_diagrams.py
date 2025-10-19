#!/usr/bin/env python3
"""
Automated Mermaid Diagram Generation and Validation System

This script analyzes the codebase to generate and update Mermaid diagrams
automatically when architecture changes are detected.

Usage:
    python scripts/generate_diagrams.py --update
    python scripts/generate_diagrams.py --validate
    python scripts/generate_diagrams.py --check-changes
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiagramMetadata:
    """Metadata for a diagram file."""

    file_path: Path
    diagram_type: str
    last_updated: str
    source_files: list[Path]
    checksum: str


class DiagramGenerator:
    """Automated diagram generation system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.diagrams_dir = project_root / "docs" / "ai_handbook" / "03_references" / "architecture" / "diagrams"
        self.generated_dir = self.diagrams_dir / "_generated"
        self.metadata_file = self.generated_dir / "diagram_metadata.json"

        # Ensure directories exist
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def analyze_codebase(self) -> dict[str, list[Path]]:
        """Analyze codebase to identify source files for each diagram type."""
        sources = {
            "component_registry": [
                self.project_root / "ocr" / "models" / "core" / "registry.py",
                self.project_root / "ocr" / "models" / "core" / "architecture.py",
                self.project_root / "configs" / "model",
            ],
            "data_pipeline": [
                self.project_root / "ocr" / "datasets" / "preprocessing",
                self.project_root / "ocr" / "datasets" / "db_collate_fn.py",
                self.project_root / "configs" / "data",
            ],
            "training_inference": [
                self.project_root / "ocr" / "lightning_modules" / "ocr_pl.py",
                self.project_root / "runners" / "train.py",
                self.project_root / "configs" / "trainer",
            ],
            "ui_flow": [
                self.project_root / "ui" / "apps" / "inference",
                self.project_root / "ui" / "services",
                self.project_root / "configs" / "ui",
            ],
            "data_loading_complexity": [
                self.project_root / "ocr" / "datasets" / "preprocessing",
                self.project_root / "ocr" / "datasets" / "db_collate_fn.py",
                self.project_root / "ocr" / "datasets",
                self.project_root / "configs" / "data",
            ],
        }

        # Expand directory paths to file lists
        expanded_sources = {}
        for diagram_type, paths in sources.items():
            file_list: list[Path] = []
            for path in paths:
                if path.is_dir():
                    file_list.extend(path.rglob("*.py"))
                    file_list.extend(path.rglob("*.yaml"))
                    file_list.extend(path.rglob("*.yml"))
                elif path.exists():
                    file_list.append(path)
            expanded_sources[diagram_type] = file_list

        return expanded_sources

    def calculate_checksum(self, files: list[Path]) -> str:
        """Calculate checksum of source files."""
        hasher = hashlib.sha256()
        for file_path in sorted(files):
            if file_path.exists():
                hasher.update(file_path.read_bytes())
        return hasher.hexdigest()

    def check_for_changes(self) -> dict[str, bool]:
        """Check if diagrams need updating based on source file changes."""
        if not self.metadata_file.exists():
            return dict.fromkeys(["component_registry", "data_pipeline", "training_inference", "ui_flow", "data_loading_complexity"], True)

        with open(self.metadata_file) as f:
            metadata = json.load(f)

        sources = self.analyze_codebase()
        changes_needed = {}

        for diagram_type, source_files in sources.items():
            current_checksum = self.calculate_checksum(source_files)
            stored_checksum = metadata.get(diagram_type, {}).get("checksum", "")

            changes_needed[diagram_type] = current_checksum != stored_checksum

        return changes_needed

    def generate_component_registry_diagram(self) -> str:
        """Generate component registry diagram from code analysis."""
        registry_file = self.project_root / "ocr" / "models" / "core" / "registry.py"

        if not registry_file.exists():
            return self._get_fallback_diagram("component_registry")

        content = registry_file.read_text()

        # Extract registered components
        encoders = re.findall(r'@register_encoder\([\'"](.*?)[\'"]\)', content)
        decoders = re.findall(r'@register_decoder\([\'"](.*?)[\'"]\)', content)
        heads = re.findall(r'@register_head\([\'"](.*?)[\'"]\)', content)
        losses = re.findall(r'@register_loss\([\'"](.*?)[\'"]\)', content)

        diagram = f"""```mermaid
graph TD
    A[ModelFactory] --> B[Component Registry]
    B --> C[Encoder Registry]
    B --> D[Decoder Registry]
    B --> E[Head Registry]
    B --> F[Loss Registry]

    C --> C1[{len(encoders)} Encoders]
    D --> D1[{len(decoders)} Decoders]
    E --> E1[{len(heads)} Heads]
    F --> F1[{len(losses)} Losses]

    C1 --> G[Assembly]
    D1 --> G
    E1 --> G
    F1 --> G
    G --> H[OCRModel]

    subgraph "Available Encoders"
"""

        for encoder in encoders[:5]:  # Show first 5
            diagram += f"        C2[{encoder}]\n"
        if len(encoders) > 5:
            diagram += f"        C3[+{len(encoders) - 5} more]\n"

        diagram += """    end

    subgraph "Available Decoders"
"""

        for decoder in decoders[:3]:
            diagram += f"        D2[{decoder}]\n"
        if len(decoders) > 3:
            diagram += f"        D3[+{len(decoders) - 3} more]\n"

        diagram += """    end

    subgraph "Available Heads"
"""

        for head in heads[:3]:
            diagram += f"        E2[{head}]\n"
        if len(heads) > 3:
            diagram += f"        E3[+{len(heads) - 3} more]\n"

        diagram += """    end

    subgraph "Available Losses"
"""

        for loss in losses[:3]:
            diagram += f"        F2[{loss}]\n"
        if len(losses) > 3:
            diagram += f"        F3[+{len(losses) - 3} more]\n"

        diagram += """    end
```"""

        return diagram

    def generate_data_pipeline_diagram(self) -> str:
        """Generate data pipeline diagram from preprocessing analysis."""
        preprocess_dir = self.project_root / "ocr" / "datasets" / "preprocessing"

        if not preprocess_dir.exists():
            return self._get_fallback_diagram("data_pipeline")

        # Analyze preprocessing files
        transforms = []
        for py_file in preprocess_dir.rglob("*.py"):
            content = py_file.read_text()
            class_matches = re.findall(r"class (\w+).*?Transform", content)
            transforms.extend(class_matches)

        diagram = f"""```mermaid
graph LR
    A[Raw Image] --> B[Load Image]
    B --> C[Geometric Preprocessing]

    subgraph "Transform Pipeline"
        C --> D[{len(transforms)} Transforms]
"""

        for i, transform in enumerate(transforms[:8]):  # Show first 8
            next_id = chr(ord("E") + i)
            diagram += f"        D --> {next_id}[{transform}]\n"
            if i < len(transforms) - 1:
                diagram += f"        {next_id} --> {chr(ord(next_id) + 1)}[{transforms[i + 1]}]\n"

        if len(transforms) > 8:
            diagram += f"        {chr(ord('E') + 7)} --> Z[+{len(transforms) - 8} more]\n"

        diagram += """    end

    Z --> F[DB Collate Function]
    F --> G[Batch Assembly]
    G --> H[DataLoader]
    H --> I[Lightning Module]

    subgraph "Data Contracts"
        J[DatasetSample] --> K[(H,W,3) uint8]
        J --> L[List of polygons]
        J --> M[Metadata dict]
    end

    I --> N[Training Loop]
```"""

        return diagram

    def generate_training_inference_diagram(self) -> str:
        """Generate training/inference diagram from Lightning module analysis."""
        lightning_file = self.project_root / "ocr" / "lightning_modules" / "ocr_pl.py"

        if not lightning_file.exists():
            return self._get_fallback_diagram("training_inference")

        content = lightning_file.read_text()

        # Extract method signatures
        methods = re.findall(r"def (\w+)\(self", content)
        key_methods = [m for m in methods if m in ["training_step", "validation_step", "predict_step", "configure_optimizers"]]

        # Build dynamic diagram based on implemented methods
        diagram = """```mermaid
graph TD
    A[Lightning Module] --> B[OCRModel]
    B --> C[Encoder]
    B --> D[Decoder]
    B --> E[Head]

"""

        # Add method nodes dynamically
        method_nodes = []
        connections = []
        node_id = ord("F")

        for method in ["training_step", "validation_step", "predict_step", "configure_optimizers"]:
            if method in key_methods:
                node_letter = chr(node_id)
                diagram += f"    A --> {node_letter}[{method}]\n"
                method_nodes.append(node_letter)
                if method in ["training_step", "validation_step", "predict_step"]:
                    connections.append(f"    {node_letter} --> J[Forward Pass]")
                node_id += 1

        # Add common processing nodes if any training methods exist
        if any(m in key_methods for m in ["training_step", "validation_step", "predict_step"]):
            diagram += """
    J --> K[Loss Calculation]
    K --> L[Backpropagation]
    L --> M[Optimizer Step]

"""
            if "predict_step" in key_methods:
                diagram += """    H --> N[Inference Output]
    N --> O[Post-processing]

"""

        # Add training loop if training_step exists
        if "training_step" in key_methods:
            diagram += """    subgraph "Training Loop"
        P[Epoch] --> Q[Dataloader]
        Q --> R[Batch]
        R --> F
        F --> S[Metrics]
        S --> T[Logging]
        T --> U[Checkpoint]
    end

"""

        # Add inference pipeline if predict_step exists
        if "predict_step" in key_methods:
            diagram += """    subgraph "Inference Pipeline"
        V[Input Image] --> W[Preprocessing]
        W --> H
        H --> N
        N --> X[Results]
    end

```
"""

        return diagram

    def generate_ui_flow_diagram(self) -> str:
        """Generate UI flow diagram from Streamlit analysis."""
        ui_dir = self.project_root / "ui" / "apps" / "inference"

        if not ui_dir.exists():
            return self._get_fallback_diagram("ui_flow")

        # Analyze UI components
        components = []
        for py_file in ui_dir.rglob("*.py"):
            content = py_file.read_text()
            func_matches = re.findall(r"def (\w+)\(", content)
            components.extend(func_matches)

        diagram = f"""```mermaid
graph TD
    A[User] --> B[Streamlit App]
    B --> C[Sidebar]
    B --> D[Main Content]

    C --> E[Model Selection]
    C --> F[Parameter Controls]
    C --> G[File Upload]

    D --> H[Image Display]
    D --> I[Results Viewer]
    D --> J[Export Options]

    G --> K[Inference Service]
    K --> L[Model Loading]
    K --> M[Batch Processing]
    K --> N[Results Formatting]

    N --> I

    subgraph "UI Components"
        O[{len(components)} Functions]
        O --> P[State Management]
        O --> Q[Service Integration]
    end

    subgraph "Service Layer"
        K --> R[Validation]
        K --> S[Preprocessing]
        K --> T[Post-processing]
    end
```"""

        return diagram

    def generate_data_loading_complexity_diagram(self) -> str:
        """Generate data loading complexity explanation diagram."""
        diagram = """```mermaid
graph TD
    A[JPG on Disk] --> B{File Exists?}
    B -->|No| C[FileNotFoundError]
    B -->|Yes| D[Read Bytes<br/>cv2.imread/OpenCV]

    D --> E{Valid Image?}
    E -->|No| F[CorruptFileError]
    E -->|Yes| G[Decode Image<br/>JPEG decompression]

    G --> H[Color Space Check<br/>RGB/BGR/Grayscale]
    H --> I[Geometric Preprocessing<br/>Document detection & correction]

    I --> J[Polygon Loading<br/>JSON/annotation parsing]
    J --> K[Shape Validation<br/>Polygon format checks]

    K --> L[Batch Collation<br/>Tensor stacking & alignment]
    L --> M[GPU Transfer<br/>Pinned memory & async copy]

    M --> N[Training Ready<br/>But wait, there's more...]

    subgraph "Filesystem Layer"
        D1[Path Resolution] --> D2[Permission Checks]
        D2 --> D3[File Locking]
    end

    subgraph "Image Processing"
        G1[EXIF Orientation] --> G2[Color Profile]
        G2 --> G3[Bit Depth Conversion]
    end

    subgraph "Geometric Transforms"
        I1[Document Detection<br/>Corner finding] --> I2[Perspective Correction<br/>Homography matrix]
        I2 --> I3[Orientation Fix<br/>Rotation detection]
        I3 --> I4[Canvas Expansion<br/>Size normalization]
    end

    subgraph "Annotation Processing"
        J1[JSON Parsing] --> J2[Coordinate Validation]
        J2 --> J3[Shape Normalization<br/>(N,2) format]
        J3 --> J4[Area Filtering<br/>Degenerate removal]
    end

    subgraph "Batch Optimization"
        L1[Memory Layout<br/>Contiguous tensors] --> L2[Type Conversion<br/>float32/int64]
        L2 --> L3[Shape Padding<br/>Variable length handling]
    end
```"""

        return diagram

    def _get_fallback_diagram(self, diagram_type: str) -> str:
        """Return fallback diagram when source files not found."""
        fallbacks = {
            "component_registry": """```mermaid
graph TD
    A[ModelFactory] --> B[Component Registry]
    B --> C[Available Components]
    C --> D[OCRModel Assembly]
```""",
            "data_pipeline": """```mermaid
graph LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[DataLoader]
    C --> D[Model]
```""",
            "training_inference": """```mermaid
graph TD
    A[Training] --> B[Model]
    B --> C[Inference]
```""",
            "ui_flow": """```mermaid
graph TD
    A[User] --> B[UI]
    B --> C[Service]
    C --> D[Results]
```""",
        }
        return fallbacks.get(diagram_type, "```mermaid\ngraph TD\n    A[Component] --> B[System]\n```")

    def update_diagram(self, diagram_type: str) -> bool:
        """Update a specific diagram file."""
        # Map diagram types to actual filenames
        filename_map = {
            "component_registry": "01_component_registry.md",
            "data_pipeline": "02_data_pipeline.md",
            "training_inference": "03_training_inference.md",
            "ui_flow": "04_ui_flow.md",
            "data_loading_complexity": "05_data_loading_complexity.md",
        }

        filename = filename_map.get(diagram_type, f"{diagram_type}.md")
        diagram_file = self.diagrams_dir / filename

        if not diagram_file.exists():
            print(f"Warning: Diagram file {diagram_file} not found")
            return False

        # Generate new diagram
        if diagram_type == "component_registry":
            new_diagram = self.generate_component_registry_diagram()
        elif diagram_type == "data_pipeline":
            new_diagram = self.generate_data_pipeline_diagram()
        elif diagram_type == "training_inference":
            new_diagram = self.generate_training_inference_diagram()
        elif diagram_type == "ui_flow":
            new_diagram = self.generate_ui_flow_diagram()
        elif diagram_type == "data_loading_complexity":
            new_diagram = self.generate_data_loading_complexity_diagram()
        else:
            return False

        # Read current content
        content = diagram_file.read_text()

        # Replace diagram section - replace ALL mermaid blocks
        pattern = r"```mermaid.*?```"
        updated_content = re.sub(pattern, new_diagram, content, flags=re.DOTALL)

        # Write back
        diagram_file.write_text(updated_content)

        # Update metadata
        sources = self.analyze_codebase()
        checksum = self.calculate_checksum(sources[diagram_type])

        metadata = self._load_metadata()
        metadata[diagram_type] = {
            "file_path": str(diagram_file.relative_to(self.project_root)),
            "diagram_type": diagram_type,
            "last_updated": subprocess.run(["date", "+%Y-%m-%d %H:%M:%S"], capture_output=True, text=True).stdout.strip(),
            "source_files": [str(p.relative_to(self.project_root)) for p in sources[diagram_type]],
            "checksum": checksum,
        }
        self._save_metadata(metadata)

        return True

    def validate_diagrams(self) -> list[str]:
        """Validate that all diagrams are syntactically correct."""
        errors = []

        for diagram_file in self.diagrams_dir.glob("*.md"):
            if diagram_file.name.startswith("_"):
                continue

            content = diagram_file.read_text()

            # Extract mermaid blocks
            mermaid_blocks = re.findall(r"```mermaid(.*?)```", content, re.DOTALL)

            for i, block in enumerate(mermaid_blocks):
                # Basic syntax validation
                lines = block.strip().split("\n")
                if not lines[0].startswith("graph "):
                    errors.append(f"{diagram_file.name}: Block {i + 1} missing 'graph' declaration")

                # Check for common syntax errors
                if "->>" in block and "graph TD" not in block:
                    errors.append(f"{diagram_file.name}: Block {i + 1} uses '->>' in non-TD graph")

        return errors

    def _load_metadata(self) -> dict:
        """Load diagram metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: dict):
        """Save diagram metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Automated Mermaid Diagram Generation")
    parser.add_argument("--update", nargs="*", help="Update specific diagrams (or all if none specified)")
    parser.add_argument("--validate", action="store_true", help="Validate diagram syntax")
    parser.add_argument("--check-changes", action="store_true", help="Check which diagrams need updates")
    parser.add_argument("--force", action="store_true", help="Force update even if no changes detected")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    generator = DiagramGenerator(project_root)

    if args.check_changes:
        changes = generator.check_for_changes()
        print("Diagrams needing updates:")
        for diagram_type, needs_update in changes.items():
            status = "YES" if needs_update else "NO"
            print(f"  {diagram_type}: {status}")

    elif args.validate:
        errors = generator.validate_diagrams()
        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("All diagrams validated successfully!")

    elif args.update is not None:
        diagrams_to_update = (
            args.update
            if args.update
            else ["component_registry", "data_pipeline", "training_inference", "ui_flow", "data_loading_complexity"]
        )

        if not args.force:
            changes = generator.check_for_changes()
            diagrams_to_update = [dt for dt in diagrams_to_update if changes.get(dt, True)]

        if not diagrams_to_update:
            print("No diagrams need updating.")
            return

        print(f"Updating diagrams: {', '.join(diagrams_to_update)}")

        for diagram_type in diagrams_to_update:
            success = generator.update_diagram(diagram_type)
            status = "SUCCESS" if success else "FAILED"
            print(f"  {diagram_type}: {status}")

        # Re-validate after updates
        errors = generator.validate_diagrams()
        if errors:
            print("Post-update validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
