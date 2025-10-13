# 🧾 OCR Receipt Text Detection

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.8+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Competition-Upstage_AI_Lab-blue.svg" alt="Competition">
  <br>
  <strong>AI Competition: Receipt Text Detection with DBNet baseline</strong>
</div>

## 🎯 Competition Overview

This project focuses on extracting text locations from receipt images. The goal is to build a model that can accurately identify and generate bounding polygons around text elements in given receipt images.

- **Competition Period:** September 22, 2025 (10:00) - October 16, 2025 (19:00)
- **Main Challenge:** Identify text regions in receipt images and draw contours

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- UV package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2.git
cd upstageailab-ocr-recsys-competition-ocr-2

# Install dependencies
uv sync

# Run training
uv run python runners/train.py
```

## 📊 Key Features

### 🏗️ Architecture
- **DBNet Baseline**: State-of-the-art text detection model
- **PyTorch Lightning**: Modern deep learning framework
- **Hydra Configuration**: Flexible experiment management

### ⚡ Performance Optimizations
- Mixed precision training (FP16)
- Image preloading and caching
- Tensor caching for faster iterations
- **6-8x speedup** with full optimizations

### 🔧 Development Tools
- **Comprehensive testing** with pytest
- **Type checking** with mypy
- **Code formatting** with ruff
- **Pre-commit hooks** for quality assurance

## 📈 Benchmark Results

| Configuration | Time/Epoch | Speedup | Memory |
|---------------|------------|---------|--------|
| Baseline (FP32) | ~180-200s | 1x | Standard |
| **Optimized (FP16 + Cache)** | **~20-30s** | **6-8x** | ~3-4GB |

## 🗂️ Documentation Structure

### [📋 Documentation Guide](README.md)
Complete guide to all project documentation, organized by intent and use case.

### [🏗️ Project Overview](project/project-overview.md)
High-level project information, competition details, and architecture overview.

### [⚙️ Setup Guide](setup/SETUP.md)
Environment setup, dependency installation, and development environment configuration.

### [🔄 Pipeline Documentation](pipeline/data_contracts.md)
Data processing pipeline, model training workflow, and inference procedures.

### [📊 Performance Analysis](performance/BENCHMARK_COMMANDS.md)
Benchmarking commands, performance optimization guides, and timing analysis.

### [🧪 Testing Framework](testing/pipeline_validation.md)
Unit tests, integration tests, and validation procedures.

### [🔍 Troubleshooting](troubleshooting/common_errors.md)
Common issues, debugging guides, and problem resolution.

### [🤖 AI Handbook](ai_handbook/index.md)
Agent operating procedures, debugging frameworks, and development workflows.

## 🏆 Team

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/156163982?v=4" width="100" height="100"/><br>
      <a href="https://github.com/SuWuKIM">AI13_이상원</a><br>
      <em>팀장, 일정관리, 성능 최적화</em>
    </td>
    <td align="center">
      <img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/hskimh1982.png" width="100" height="100"/><br>
      <a href="https://github.com/YOUR_GITHUB">AI13_김효석</a><br>
      <em>EDA, 데이터셋 증강</em>
    </td>
    <td align="center">
      <img src="https://github.com/Wchoi189/document-classifier/blob/dev-hydra/docs/images/team/AI13_%EC%B5%9C%EC%9A%A9%EB%B9%84.png?raw=true" width="100" height="100"/><br>
      <a href="https://github.com/Wchoi189">AI13_최용비</a><br>
      <em>베이스라인, CI</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/YeonkyungKang.png" width="100" height="100"/><br>
      <a href="https://github.com/YeonkyungKang">AI13_강연경</a><br>
      <em>문서화, 평가</em>
    </td>
    <td align="center">
      <img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/jungjaehoon.jpg" width="100" height="100"/><br>
      <a href="https://github.com/YOUR_GITHUB">AI13_정재훈</a><br>
      <em>모델링, 실험</em>
    </td>
  </tr>
</table>

## 📚 Additional Resources

- [Competition Page](https://upstage.ai) - Official competition website
- [PyTorch Lightning](https://lightning.ai) - Deep learning framework
- [Hydra](https://hydra.cc) - Configuration management
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

## 🤝 Contributing

We welcome contributions! Please see our [setup guide](setup/SETUP.md) for development environment setup and our [AI handbook](ai_handbook/index.md) for development workflows.

## 📄 License

© 2025 AI Bootcamp Team 13. All rights reserved.

---

<div align="center">
  <sub>Built with ❤️ using <a href="https://squidfunk.github.io/mkdocs-material/">Material for MkDocs</a></sub>
</div>
