<!-- Github Decorative Badges -->
<div align="center">

[![CI](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![UV](https://img.shields.io/badge/UV-0.8+-purple.svg)](https://github.com/astral-sh/uv)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.1+-orange.svg)](https://lightning.ai)
[![Competition](https://img.shields.io/badge/Competition-Upstage_AI_Lab-blue.svg)](https://upstage.ai)
</div>

# AI 경진대회: 영수증 텍스트 검출

이 프로젝트는 영수증 이미지에서 텍스트 위치를 추출하는 것에 중점을 둡니다. 주어진 영수증 이미지에서 텍스트 요소 주변에 경계 다각형을 정확하게 식별하고 생성할 수 있는 모델을 구축하는 것이 목표입니다.

* **대회 기간:** 2025년 9월 22일 (10:00) - 2025년 10월 16일 (19:00)
* **주요 과제:** 영수증 이미지 내 텍스트 영역 식별 및 윤곽 그리기

## 📋 목차

- [0. 개요](#0-개요)
- [1. 대회 정보](#1-대회-정보)
- [2. 구성 요소](#2-구성-요소)
- [3. 데이터 설명](#3-데이터-설명)
- [4. 모델링](#4-모델링)
- [5. 결과](#5-결과)
- [6. 결론 및 향후 과제](#6-결론-및-향후-과제)


## 👥 팀 소개
<table>
    <tr>
        <td align="center"><img src="https://avatars.githubusercontent.com/u/156163982?v=4" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/hskimh1982.png" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/Wchoi189/document-classifier/blob/dev-hydra/docs/images/team/AI13_%EC%B5%9C%EC%9A%A9%EB%B9%84.png?raw=true" width="180" height="180"/>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/YeonkyungKang.png" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/jungjaehoon.jpg" width="180" height="180"/></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/SuWuKIM">AI13_이상원</a></td>
        <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_김효석</a></td>
        <td align="center"><a href="https://github.com/Wchoi189">AI13_최용비</a></td>
        <td align="center"><a href="https://github.com/YeonkyungKang">AI13_강연경</a></td>
        <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_정재훈</a></td>
    </tr>
    <tr>
        <td align="center">팀장, 일정관리, 성능 최적화</td>
        <td align="center">EDA, 데이터셋 증강</td>
        <td align="center">베이스라인, CI</td>
        <td align="center">연구/실험 설계 및 추론 분석</td>
        <td align="center">아키텍처 리뷰 및 설계</td>
    </tr>
 </table>

## 0. 개요

### 개발 환경
- **Python:** 3.10+
- **패키지 관리자:** UV 0.8+
- **딥러닝:** PyTorch 2.8+, PyTorch Lightning 2.1+
- **구성 관리:** Hydra 1.3+

### 요구사항
- Python 3.10 이상
- UV 패키지 관리자
- CUDA 호환 GPU (훈련 시 권장)

## 1. 대회 정보

### 개요
영수증 텍스트 검출에 중점을 둔 AI 경진대회입니다. 참가자들은 경계 다각형을 사용하여 영수증 이미지에서 텍스트 영역을 정확하게 검출하고 위치를 파악할 수 있는 모델을 개발해야 합니다.

### 일정
- **시작일:** 2025년 9월 22일 (10:00)
- **최종 제출 마감일:** 2025년 10월 16일 (19:00)

## 2. 구성 요소

### 디렉토리 구조

```
├── augmentation-patterns.yaml
├── configs/
│   ├── predict.yaml
│   ├── test.yaml
│   ├── train.yaml
│   └── preset/
│       ├── base.yaml
│       ├── example.yaml
│       ├── datasets/
│       │   └── db.yaml
│       ├── lightning_modules/
│       │   └── base.yaml
│       └── models/
│           ├── model_example.yaml
│           ├── decoder/
│           ├── encoder/
│           ├── head/
│           └── loss/
├── data/
│   ├── datasets/
│   │   ├── sample_submission.csv
│   │   └── images/
│   │       ├── test/
│   │       └── ...
│   └── jsons/
├── docs/
│   ├── api-reference.md
│   ├── architecture-overview.md
│   ├── process-management-guide.md
│   ├── component-diagrams.md
│   ├── workflow-diagram.md
│   ├── maintenance/
│   │   └── project-state.md
│   └── development/
│       ├── coding-standards.md
│       ├── naming-conventions.md
│       └── testing-guide.md
├── ocr/
│   ├── datasets/
│   ├── lightning_modules/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── outputs/
├── runners/
│   ├── predict.py
│   ├── test.py
│   └── train.py
├── scripts/
│   └── process_monitor.py
├── ui/
│   ├── command_builder.py
│   ├── evaluation_viewer.py
│   ├── inference_ui.py
│   ├── resource_monitor.py
│   ├── components/
│   ├── utils/
│   └── README.md
└── tests/
```

### UI 도구

프로젝트에는 명령어 구축과 결과 분석을 위한 Streamlit 기반 UI 도구가 포함되어 있습니다.

#### Command Builder (`ui/command_builder.py`)
훈련, 테스트, 예측 명령어를 직관적인 UI로 구축하고 실행할 수 있는 도구입니다.

**주요 기능:**
- 모델 아키텍처 선택 (인코더, 디코더, 헤드, 손실 함수)
- 학습 파라미터 조정 (학습률, 배치 크기, 에폭 수)
- 실험 설정 (W&B 통합, 체크포인트 재개)
- 실시간 명령어 검증 및 미리보기
- 원클릭 명령어 실행 및 진행 상황 모니터링

**사용법:**
```bash
# 명령어 구축 UI 실행
python run_ui.py command_builder

# 또는 직접 실행
uv run streamlit run ui/command_builder.py
```

#### Evaluation Viewer (`ui/evaluation_viewer.py`) - ✅ Implemented
평가 결과를 시각화하고 분석하는 도구입니다.

### 유틸리티 스크립트

#### Process Monitor (`scripts/process_monitor.py`) - ✅ Implemented
훈련 프로세스와 작업자 프로세스를 모니터링하고 정리하는 유틸리티입니다.

**주요 기능:**
- 고아 프로세스 감지 및 정리
- 훈련 프로세스와 DataLoader 작업자 프로세스 모니터링
- 안전한 프로세스 종료 (SIGTERM) 및 강제 종료 (SIGKILL) 지원
- 드라이런 모드로 미리보기 기능

**사용법:**
```bash
# 현재 실행 중인 훈련 프로세스 목록 보기
python scripts/process_monitor.py --list

# 모든 고아 프로세스 정리 (안전 모드)
python scripts/process_monitor.py --cleanup

# 강제 정리 (SIGKILL 사용)
python scripts/process_monitor.py --cleanup --force

# 정리할 프로세스 미리보기 (실제로는 정리하지 않음)
python scripts/process_monitor.py --cleanup --dry-run
```
<!--
#### Resource Monitor (`ui/resource_monitor.py`) - ✅ New!
시스템 리소스, 훈련 프로세스, GPU 사용량을 실시간으로 모니터링하는 도구입니다.

**주요 기능:**
- CPU, 메모리, GPU 리소스 실시간 모니터링
- 훈련 프로세스와 작업자 프로세스 상태 표시
- 프로세스 관리 (안전 종료 및 강제 종료)
- GPU 메모리 사용량 시각화
- 자동 새로고침 기능

**사용법:**
```bash
# 리소스 모니터 UI 실행
python run_ui.py resource_monitor
```

#### AI System Monitor (`scripts/monitoring/monitor.sh`) - ✅ New!
AI 에이전트가 시스템 리소스를 모니터링하고 프로세스를 관리할 수 있는 도구입니다.

**주요 기능:**
- AI 기반 자연어 쿼리로 시스템 모니터링
- CPU, 메모리, 디스크 사용량 종합 분석
- 고아 프로세스와 좀비 프로세스 감지
- 안전한 프로세스 종료 기능
- Qwen MCP 서버를 통한 AI 에이전트 통합

**사용법:**
```bash
# 시스템 상태 확인
./scripts/monitoring/monitor.sh "Show system health status"

# 고아 프로세스 검사
./scripts/monitoring/monitor.sh "Monitor system resources and check for orphaned processes"

# 상위 CPU 사용 프로세스 목록
./scripts/monitoring/monitor.sh "List top 10 processes by CPU usage"

# 메모리 사용량 분석
./scripts/monitoring/monitor.sh "Check memory usage and identify high consumers"
``` -->

### 주요 구성 파일

- `train.yaml`, `test.yaml`, `predict.yaml`: 러너 실행 설정
- `preset/example.yaml`: 각 모듈의 구성 파일 지정
- `preset/datasets/db.yaml`: Dataset, Transform, 데이터 관련 설정
- `preset/lightning_modules/base.yaml`: PyTorch Lightning 실행 설정
- `preset/metrics/cleval.yaml`: CLEval 평가 설정
- `preset/models/model_example.yaml`: 각 모델 모듈과 Optimizer의 구성 파일 지정
- `preset/models/*`: 모델 구성에 필요한 각 모듈 설정

## 3. 데이터 설명

### 데이터셋 개요

데이터는 이미지 폴더와 주석을 위한 해당 JSON 파일로 구성됩니다. 데이터셋은 영수증 이미지와 텍스트 영역 주석을 포함하는 train/validation/test 분할로 구성되어 있습니다.

### 디렉토리 구조

```
.
├── images/
│   ├── train/
│   │   └── ...jpg
│   ├── val/
│   │   └── ...jpg
│   └── test/
│       └── ...jpg
└── jsons/
     ├── train.json
     ├── val.json
     └── test.json
```

### JSON 주석 형식

JSON 파일은 이미지 파일명을 텍스트 경계 상자의 좌표에 매핑합니다.

* **IMAGE_FILENAME**: 각 이미지 레코드의 키
* **words**: 이미지에 대해 감지된 모든 텍스트 인스턴스를 포함하는 객체
* **nnnn**: 각 단어 인스턴스의 고유한 4자리 인덱스 (0001부터 시작)
* **points**: 텍스트 주변의 다각형을 정의하는 [X, Y] 좌표 쌍의 배열. 원점 (0,0)은 이미지의 왼쪽 상단 모서리. 평가에 유효한 다각형이 되려면 최소 4개의 점이 필요

### EDA

- _EDA 과정과 단계별 결론을 설명하세요_

### 데이터 처리

- 이미지는 JPG 형식으로 저장
- 주석은 다각형 좌표가 포함된 JSON 형식으로 제공
- 텍스트 영역은 정확한 경계 다각형으로 주석 처리
- 데이터셋은 train, validation, test 분할을 포함

### 데이터 전처리 (Pre-processing)

이 프로젝트는 훈련 성능을 크게 향상시키는 오프라인 전처리 시스템을 사용합니다.

#### 전처리가 필요한 이유

DBNet 모델은 확률 맵(probability map)과 임계값 맵(threshold map)을 필요로 합니다. 이전에는 이러한 맵을 훈련 중 실시간으로 생성했으나, 다음과 같은 문제가 있었습니다:

- 계산 비용이 높은 pyclipper 연산과 거리 계산
- 에포크마다 동일한 맵을 반복 계산
- 효과적이지 못한 캐싱 메커니즘

오프라인 전처리를 통해 **5-8배 빠른 검증 속도**를 달성했습니다.

#### 전처리 실행 방법

전체 데이터셋을 전처리하려면 프로젝트 루트에서 다음 명령을 실행하세요:

```bash
uv run python scripts/preprocess_maps.py
```

샘플 수를 제한하여 테스트하려면:

```bash
uv run python scripts/preprocess_maps.py data.train_num_samples=100 data.val_num_samples=20
```

전처리 스크립트는 다음을 생성합니다:
- `data/datasets/images/train_maps/`: 훈련 데이터의 전처리된 맵
- `data/datasets/images_val_canonical_maps/`: 검증 데이터의 전처리된 맵

각 이미지에 대해 압축된 `.npz` 파일이 생성되며, 확률 맵과 임계값 맵이 포함됩니다.

#### 자동 폴백 (Fallback)

전처리된 맵이 없어도 훈련은 정상적으로 작동합니다. 시스템이 자동으로 실시간 맵 생성으로 전환되지만, 속도가 느려집니다.

더 자세한 내용은 [데이터 전처리 데이터 컨트랙트](docs/preprocessing-data-contracts.md)와 [파이프라인 데이터 컨트랙트](docs/pipeline/data_contracts.md)를 참조하세요.

## 4. 모델링

### 모델 설명

베이스라인 코드는 장면 텍스트 검출에서 효과적인 것으로 알려진 **DBNet** 아키텍처를 기반으로 구축되었습니다. DBNet은 실시간 장면 텍스트 검출을 위해 미분 가능한 이진화를 사용합니다.

#### DBNet: 미분 가능한 이진화를 통한 실시간 장면 텍스트 검출

![DBNet](docs/assets/images/00_refactor_bsaeline/flow-chart-of-the-dbnet.png)

### 베이스라인 성능

V100 GPU에서 10 에포크 훈련 후, 베이스라인 모델은 공개 테스트 세트에서 다음과 같은 성능을 달성했습니다:

* **훈련 시간:** 약 22분
* **H-Mean:** 0.8818
* **정밀도:** 0.9651
* **재현율:** 0.8194

### 평가 지표

이 대회는 텍스트 검출 결과 평가를 위해 **CLEval**을 사용합니다.

#### CLEval: 텍스트 검출 및 인식 작업을 위한 문자 수준 평가

![CLEval](https://github.com/clovaai/CLEval/raw/master/resources/screenshots/explanation.gif)

### 모델링 과정

#### 훈련
```bash
uv run python runners/train.py preset=example
```

#### 테스트
```bash
# 사용 예시
uv run python runners/test.py preset=example checkpoint_path=\"outputs/ocr_training/checkpoints/epoch-9-step-1030.ckpt\"
```

#### 예측
```bash
# 사용 예시
uv run python runners/predict.py preset=example checkpoint_path=\"outputs/ocr_training/checkpoints/epoch-8-step-1845.ckpt\"
```

### Ablation Studies ( ablation studies )

이 프로젝트는 체계적인 ablation studies를 위한 완전한 워크플로우를 제공합니다. 하이퍼파라미터 튜닝, 모델 아키텍처 비교, 데이터 증강 실험 등을 자동화하여 연구 효율성을 높입니다.

#### Quick Start

```bash
# 1. Learning rate ablation study
python ablation_workflow.py --ablation learning_rate --tag lr_study

# 2. Batch size ablation study
python ablation_workflow.py --ablation batch_size --tag batch_study

# 3. Model architecture comparison
python ablation_workflow.py --ablation model_architecture --tag model_study
```

#### Available Ablation Types

- **`learning_rate`**: 학습률 스윕 (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5)
- **`batch_size`**: 배치 크기 스윕 (4, 8, 16, 32)
- **`model_architecture`**: 백본 아키텍처 비교 (resnet18, resnet34, resnet50, mobilenetv3_small_050, efficientnet_b0)
- **`custom`**: 사용자 정의 설정

#### Manual Control

각 단계별로 개별 실행 가능:

```bash
# 1. Run experiments only
python run_ablation.py +ablation=learning_rate experiment_tag=my_lr_study -m

# 2. Collect results from wandb
python collect_results.py --project OCR_Ablation --tag my_lr_study --output results.csv

# 3. Generate comparison table
python generate_ablation_table.py --input results.csv --ablation-type learning_rate --metric val/hmean --output-md table.md
```

#### Configuration

Ablation 설정은 `configs/ablation/` 디렉토리에 YAML 파일로 정의:

```yaml
# configs/ablation/learning_rate.yaml
defaults:
  - _self_

training:
  learning_rate: [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

trainer:
  max_epochs: 5

wandb: true
experiment_tag: "lr_ablation"
```

#### Custom Ablations

Hydra의 강력한 설정 시스템을 활용하여 복잡한 ablation studies 생성:

```bash
# Multiple parameter sweep
python run_ablation.py \
  training.learning_rate=1e-3,5e-4,1e-4 \
  data.batch_size=8,16,32 \
  experiment_tag=multi_param_study \
  -m

# Architecture + augmentation sweep
python run_ablation.py \
  +ablation=custom \
  model.backbone.name=resnet18,resnet50 \
  augmentation.rotate.limit=15,30 \
  experiment_tag=arch_aug_study \
  -m
```

#### Results Analysis

자동 생성되는 결과물:
- **CSV 파일**: 모든 실험의 상세 메트릭
- **마크다운 테이블**: 발표용 비교 테이블
- **LaTeX 테이블**: 논문용 테이블
- **시각화**: 성능 추이 그래프

#### wandb Integration

모든 ablation studies는 자동으로 wandb에 기록되며:
- 실험별 메트릭 추적
- 하이퍼파라미터 로깅
- 모델 체크포인트 저장
- 실시간 비교 대시보드

---

#### 사용 예시
```bash
# 1. Run unit tests
uv run pytest tests/ -v

# 2. Train model (adjust epochs as needed)
uv run python runners/train.py preset=example trainer.max_epochs=10 dataset_base_path="/path/to/data/datasets/"

# 3. Generate predictions
uv run python runners/predict.py preset=example checkpoint_path="outputs/ocr_training/checkpoints/best.ckpt" dataset_base_path="/path/to/data/datasets/"

# 4. Convert to submission format
uv run python ocr/utils/convert_submission.py --json_path outputs/ocr_training/submissions/latest.json --output_path submission.csv


```

### 모델 개선 사항

- _모델 아키텍처 변경 사항을 설명하세요_
- _하이퍼파라미터 튜닝 과정을 설명하세요_
- _데이터 증강 기법을 설명하세요_
- _앙상블 방법을 설명하세요 (해당하는 경우)_

## 5. 결과

### 최종 성능

- _최종 모델의 성능 지표를 기록하세요_
- _공개/비공개 리더보드 점수를 기록하세요_
- _베이스라인 대비 개선 사항을 설명하세요_

### 제출 과정

#### 제출 파일 생성

예측 스크립트는 JSON 파일을 생성합니다. 이 파일은 제출하기 전에 제공된 유틸리티 스크립트를 사용하여 필요한 CSV 형식으로 변환해야 합니다.

```bash
# 사용 예시
uv run python ocr/utils/convert_submission.py --json_path outputs/ocr_training/submissions/your_submission.json --output_path submission.csv
```

#### CSV 형식

제출 파일은 `filename`과 `polygons` 두 열이 있는 CSV여야 합니다.

* **filename**: 테스트 세트의 이미지 파일명
* **polygons**: 해당 이미지에서 예측된 모든 텍스트 영역의 좌표를 포함하는 단일 문자열
  * 하나의 다각형에 대한 좌표는 공백으로 구분 (예: `X1 Y1 X2 Y2 X3 Y3 X4 Y4`)
  * 같은 이미지의 다른 다각형들은 파이프 문자(`|`)로 구분

#### 출력 파일 구조

```
└─── outputs
    └── {exp_name}
        ├── .hydra
        │   ├── overrides.yaml
        │   ├── config.yaml
        │   └── hydra.yaml
        ├── checkpoints
        │   └── epoch={epoch}-step={step}.ckpt
        ├── logs
        │   └── {exp_name}
        │       └── {exp_version}
        │           └── events.out.tfevents.{timestamp}.{hostname}.{pid}.v2
        └── submissions
            └── {timestamp}.json
```

### 평가 기준

대회 리더보드는 공개 및 비공개 순위로 나뉩니다. 대회 기간 중에는 공개 세트에 대한 점수가 표시됩니다. 최종 우승자는 대회 종료 후 공개되는 비공개 테스트 세트에서의 모델 성능으로 결정됩니다. 테스트 데이터는 공개 및 비공개 세트 간에 동등하게(50/50) 분할됩니다.

### 실험 결과 분석

- _다양한 실험 결과를 비교 분석하세요_
- _실패한 케이스들을 분석하세요_
- _모델의 한계점을 설명하세요_

## 6. 결론 및 향후 과제

### 주요 성과

- _프로젝트의 주요 성과를 요약하세요_
- _팀원별 기여도를 설명하세요_
- _배운 점들을 정리하세요_

### 향후 개선 방향

- _모델 성능 개선을 위한 아이디어를 제시하세요_
- _데이터 품질 향상 방안을 제시하세요_
- _시스템 최적화 방안을 제시하세요_

## 설치 및 설정

### 🚨 환경 설정 (중요)

이 프로젝트는 **UV** 패키지 매니저를 사용합니다. 다른 패키지 매니저(pip, conda, poetry)를 사용하지 마세요.

```bash
# 자동 환경 설정 (권장)
./scripts/setup/00_setup-environment.sh

# 또는 수동으로:
# 1. 의존성 설치
uv sync --group dev

# 2. 환경 확인
uv run python -c "import torch; print('PyTorch:', torch.__version__)"
```

### VS Code 설정

프로젝트를 VS Code에서 열면 자동으로 다음 설정이 적용됩니다:
- Python 인터프리터: `./.venv/bin/python`
- 터미널: 자동으로 가상환경 활성화
- 모든 Python 명령어는 `uv run` 접두사 사용

### 모든 명령어는 `uv run` 사용

```bash
# ❌ 잘못된 사용
python runners/train.py
pytest tests/

# ✅ 올바른 사용
uv run python runners/train.py
uv run pytest tests/
```

### 로컬 테스트

```bash
# 모든 테스트 실행
uv run pytest tests/

```markdown
# 특정 테스트 파일 실행
uv run pytest tests/test_metrics.py

# 커버리지와 함께 실행 (선택사항)
uv run pytest tests/ --cov=ocr
```

### 구성 파일 설정

```bash
# 데이터셋 경로 설정
# configs/preset/datasets/db.yaml 파일에서 base_path 수정
base_path: /path/to/your/extracted/data

# 환경 변수 설정 (선택사항)
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 문제 해결

### 일반적인 문제들

#### CUDA 메모리 부족
```bash
# 배치 크기 줄이기
# configs/preset/datasets/db.yaml에서 batch_size 조정
batch_size: 8  # 기본값에서 줄이기
```

#### 패키지 의존성 문제
```bash
# UV 캐시 정리
uv cache clean

# 의존성 재설치
uv sync --reinstall
```

#### 데이터 경로 오류
```bash
# 데이터 경로 확인
ls -la data/images/train/
ls -la data/jsons/

# 구성 파일에서 경로 확인
cat configs/preset/datasets/db.yaml
```

### 성능 최적화 팁

- GPU 메모리에 맞게 배치 크기 조정
- 데이터 로더의 num_workers 설정 최적화
- 혼합 정밀도 훈련 사용 고려
- 체크포인트 저장 빈도 조정

## 기여 가이드라인

### 코드 스타일

- Python PEP 8 스타일 가이드 준수
- 타입 힌트 사용 권장
- Docstring 작성 (Google 스타일)

### 테스트 작성

```bash
# 새로운 기능에 대한 테스트 작성
# tests/ 디렉토리에 테스트 파일 추가

# 테스트 실행 확인
uv run pytest tests/test_new_feature.py -v
```

### Pull Request 가이드라인

1. 기능 브랜치 생성
2. 코드 변경 및 테스트 작성
3. 모든 테스트 통과 확인
4. 명확한 커밋 메시지 작성
5. PR 설명에 변경 사항 상세 기록

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 참고 자료

- [프로세스 관리 가이드](docs/process-management-guide.md) - 훈련 프로세스 관리 및 고아 프로세스 방지
- [DBNet](https://github.com/MhLiao/DB)
- [Hydra](https://hydra.cc/docs/intro/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [CLEval](https://github.com/clovaai/CLEval)
- [UV 패키지 관리자](https://github.com/astral-sh/uv)

## 참고 논문:
- CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks
  https://arxiv.org/pdf/2006.06244.pdf


## 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 다음을 통해 연락해 주세요:

- **이슈 트래커**: [GitHub Issues](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues)
<!-- - **이메일**: [팀 대표 이메일 주소]
- **디스코드**: [팀 디스코드 채널] -->

---

**마지막 업데이트**: 2025년 9월 23일

## 🤖 코드 품질 자동화

이 프로젝트는 코드 품질을 자동으로 유지하기 위한 포괄적인 자동화 시스템을 갖추고 있습니다.

### 자동화 수준

#### 1. **Pre-commit Hooks** (로컬 개발자용)
커밋 전에 자동으로 코드 품질 검사를 수행합니다.

```bash
# 설치 (최초 1회)
make pre-commit
# 또는
pre-commit install

# 수동 실행
pre-commit run --all-files
```

#### 2. **CI/CD 자동화** (GitHub Actions)
- **푸시/PR 시**: 자동 코드 품질 검사
- **PR 시**: 자동 수정 적용 및 커밋
- **주간**: 정기적인 코드 품질 유지보수

#### 3. **로컬 자동화 스크립트**
```bash
# 코드 품질 자동 수정
make quality-fix
# 또는
./scripts/code-quality.sh

# 품질 검사만 수행
make quality-check
```

### 자동화되는 작업

| 도구 | 목적 | 자동화 수준 |
|------|------|-------------|
| **Black** | 코드 포맷팅 | ✅ 자동 적용 |
| **isort** | import 정렬 | ✅ 자동 적용 |
| **flake8** | 린팅 | ✅ 자동 적용 |
| **autoflake** | 미사용 import 제거 | ✅ 자동 적용 |
| **mypy** | 타입 체크 | ✅ 검사만 |

### 워크플로우

#### 개발자 워크플로우
```bash
# 1. 코드 작성
# 2. 자동 품질 검사 (pre-commit)
git add .
git commit  # 자동으로 품질 검사가 실행됨

# 3. 수동 품질 개선 (필요시)
make quality-fix
```

#### CI 워크플로우
- **푸시**: 코드 품질 검사
- **PR**: 자동 수정 적용
- **매주 월요일**: 유지보수 PR 생성

### 설정 파일들

- `.pre-commit-config.yaml` - Pre-commit 훅 설정
- `.github/workflows/ci.yml` - CI/CD 파이프라인
- `.github/workflows/scheduled-maintenance.yml` - 주간 유지보수
- `setup.cfg` - Flake8 설정
- `pyproject.toml` - Black, isort, mypy 설정
- `Makefile` - 편의 명령어들

### 수동 실행

```bash
# 모든 품질 도구 실행
make quality-check

# 코드 포맷팅만
make format

# 린팅만
make lint

# 테스트와 품질 검사
make ci
```

### 자동화의 이점

- **일관성**: 모든 기여자가 동일한 코드 스타일 유지
- **품질 보장**: 자동으로 버그 유발 코드를 방지
- **시간 절약**: 수동 코드 리뷰 시간 감소
- **지속적 개선**: 정기적인 코드 품질 유지보수
