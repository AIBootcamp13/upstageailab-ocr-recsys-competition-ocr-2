# OCR Project Streamlit UI

This directory contains Streamlit applications for managing OCR training workflows and real-time inference.
## Table of Contents

- [Applications](#applications)
    - [Command Builder (`command_builder.py`)](#command-builder-command_builderpy)
        - [Features](#features)
        - [Process Safety](#process-safety)
        - [Usage](#usage)
    - [Inference UI (`inference_ui.py`)](#inference-ui-inference_uipy)
        - [Features](#features-1)
        - [Setup Requirements](#setup-requirements)
        - [Usage](#usage-1)
        - [Inference Workflow](#inference-workflow)
        - [Demo Mode](#demo-mode)
    - [Evaluation Viewer (`evaluation_viewer.py`)](#evaluation-viewer-evaluation_viewerpy)
        - [주요 기능](#주요-기능)
        - [사용법](#사용법)
        - [분석 기능](#분석-기능)
- [Applications](#applications-1)
    - [Command Builder (`command_builder.py`)](#command-builder-command_builderpy-1)
    - [Evaluation Viewer (`evaluation_viewer.py`)](#evaluation-viewer-evaluation_viewerpy-1)
    - [Resource Monitor (`resource_monitor.py`)](#resource-monitor-resource_monitorpy)
        - [Features](#features-2)
        - [Process Management](#process-management)
        - [Usage](#usage-2)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Development](#development)
- [Future Enhancements](#future-enhancements)

## Applications

### Command Builder (`command_builder.py`)
A user-friendly interface for building and executing training, testing, and prediction commands.

**Features:**
- Interactive model architecture selection (encoders, decoders, heads, losses)
- Training parameter adjustment (learning rate, batch size, epochs)
- Experiment configuration (W&B integration, checkpoint resuming)
- Real-time command validation and preview
- One-click command execution with progress monitoring
- **Improved process management** - Safe process group handling prevents orphaned training processes

**Process Safety:**
- Uses process groups for complete cleanup on interruption
- Automatic termination of DataLoader worker processes
- Graceful shutdown handling for interrupted training sessions
- Integration with process monitoring utilities

**Usage:**
```bash
# Run the command builder UI
python run_ui.py command_builder

# Or directly with streamlit
uv run streamlit run ui/command_builder.py
```

### Inference UI (`inference_ui.py`) - ✅ New!
Real-time OCR inference interface for instant predictions on uploaded images.

**Features:**
- Drag-and-drop image upload (supports JPG, PNG, BMP)
- Model checkpoint selection from trained models
- Real-time inference with progress tracking
- Interactive visualization of OCR predictions
- Batch processing for multiple images
- Demo mode with mock predictions when models aren't available

**Setup Requirements:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or use your preferred method

# Install dependencies
uv sync

# For real inference (optional - demo mode works without this)
# Train a model first using the command builder UI
```

**Usage:**
```bash
# Run the inference UI
python run_ui.py inference

# Or directly with streamlit
uv run streamlit run ui/inference_ui.py
```

**Inference Workflow:**
1. Upload one or more images via drag-and-drop
2. Select a trained model checkpoint (or use demo mode)
3. Click "Run Inference" for instant results
4. View predictions overlaid on images
5. Extract and copy recognized text

**Demo Mode:**
If no trained models are available, the UI automatically switches to demo mode with mock predictions, allowing you to test the interface and workflow before training models.

### Evaluation Viewer (`ui/evaluation/`) - ✅ Implemented
A modular interface for viewing and analyzing OCR evaluation results.

**Architecture:**
- `ui/evaluation/app.py` - Main application entry point
- `ui/evaluation/single_run.py` - Single model analysis view
- `ui/evaluation/comparison.py` - Model comparison view
- `ui/evaluation/gallery.py` - Image gallery with filtering
- `ui/evaluation/__init__.py` - Package initialization

**주요 기능:**
- 예측 결과 CSV 파일 로드 및 분석
- 데이터셋 통계 및 분포 차트 표시
- 예측 분석 (바운딩 박스 면적, 종횡비 등)
- 이미지별 예측 결과 시각화 (바운딩 박스 오버레이)
- 모델 간 비교 및 차이 분석
- 이미지 갤러리 with 필터링 (높은 신뢰도, 낮은 신뢰도 등)
- 대화형 차트 및 통계 테이블

**사용법:**
```bash
# 평가 결과 뷰어 실행
python run_ui.py evaluation_viewer

# 데모 실행
python demo_evaluation_viewer.py
```

**분석 기능:**
- 전체 데이터셋 통계 (이미지 수, 예측 수, 평균 예측/이미지)
- 예측 분포 히스토그램
- 바운딩 박스 면적 및 종횡비 분석
- 개별 이미지 예측 결과 시각화

## Applications

### Command Builder (`command_builder.py`)
A user-friendly interface for building and executing training, testing, and prediction commands.

**Features:**
- Interactive model architecture selection (encoders, decoders, heads, losses)
- Training parameter adjustment (learning rate, batch size, epochs)
- Experiment configuration (W&B integration, checkpoint resuming)
- Real-time command validation and preview
- One-click command execution with progress monitoring

**Usage:**
```bash
# Run the command builder UI
python run_ui.py command_builder

# Or directly with streamlit
uv run streamlit run ui/command_builder.py
```

### Evaluation Viewer (`evaluation_viewer.py`) - Coming Soon
An interface for viewing and analyzing evaluation results.

### Resource Monitor (`resource_monitor.py`) - ✅ New!
A comprehensive monitoring interface for system resources, training processes, and GPU utilization.

**Features:**
- Real-time CPU, memory, and GPU resource monitoring
- Training process and worker process status display
- Process management with safe termination and force kill options
- GPU memory usage visualization with progress bars
- Auto-refresh capability (5-second intervals)
- Quick action buttons for process cleanup and emergency stops

**Process Management:**
- View all training processes and their worker processes
- Terminate processes gracefully (SIGTERM) or forcefully (SIGKILL)
- Confirmation dialogs for dangerous operations
- Integration with the process monitor utility script

**Usage:**
```bash
# Run the resource monitor UI
python run_ui.py resource_monitor

# Or directly with streamlit
uv run streamlit run ui/resource_monitor.py
```

## Architecture

The UI is built with a modular design:

```
ui/
├── command_builder.py          # Main command builder app
├── evaluation/                 # Evaluation results viewer (modular)
│   ├── __init__.py
│   ├── app.py                  # Main application
│   ├── single_run.py           # Single model analysis
│   ├── comparison.py           # Model comparison
│   └── gallery.py              # Image gallery
├── evaluation_viewer.py        # Legacy wrapper for evaluation/
├── inference_ui.py            # Real-time inference interface
├── resource_monitor.py         # System resource and process monitor
├── components/                 # Reusable UI components
├── utils/                      # Utility modules
│   ├── config_parser.py        # Parses Hydra configurations
│   └── command_builder.py      # Builds CLI commands
└── __init__.py
```

## Dependencies

- `streamlit >= 1.28.0` - Web UI framework
- Project dependencies (PyTorch, Lightning, etc.)

## Development

The UI applications are designed to be:
- **Modular**: Separate concerns with clear interfaces
- **Extensible**: Easy to add new features and components
- **Integrated**: Works seamlessly with existing CLI tools
- **User-friendly**: Intuitive interface for complex configurations

## Future Enhancements

- Advanced ablation study configuration
- Real-time training progress monitoring
- Model comparison tools
- Automated hyperparameter optimization
- Custom dataset upload and validation
