# OCR Project Streamlit U#### Evaluation Viewer (`ui/evaluation_viewer.py`) - ✅ Implemented
An interface for viewing and analyzing OCR evaluation results.

**주요 기능:**
- 예측 결과 CSV 파일 로드 및 분석
- 데이터셋 통계 및 분포 차트 표시
- 예측 분석 (바운딩 박스 면적, 종횡비 등)
- 이미지별 예측 결과 시각화 (바운딩 박스 오버레이)
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
This directory contains Streamlit applications for managing OCR training workflows.

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

## Architecture

The UI is built with a modular design:

```
ui/
├── command_builder.py          # Main command builder app
├── evaluation_viewer.py        # Evaluation results viewer (planned)
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