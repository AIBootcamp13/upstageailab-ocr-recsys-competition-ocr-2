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
└── tests/
```

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

### 데이터 처리

- 이미지는 JPG 형식으로 저장
- 주석은 다각형 좌표가 포함된 JSON 형식으로 제공
- 텍스트 영역은 정확한 경계 다각형으로 주석 처리
- 데이터셋은 train, validation, test 분할을 포함
- _데이터 처리 과정을 설명하세요 (예: 데이터 라벨링, 데이터 정제 등)_

## 4. 모델링

### 모델 설명

베이스라인 코드는 장면 텍스트 검출에서 효과적인 것으로 알려진 **DBNet** 아키텍처를 기반으로 구축되었습니다. DBNet은 실시간 장면 텍스트 검출을 위해 미분 가능한 이진화를 사용합니다.

#### DBNet: 미분 가능한 이진화를 통한 실시간 장면 텍스트 검출

![DBNet](https://www.researchgate.net/publication/369783176/figure/fig1/AS:11431281137414188@1680649387586/Structure-of-DBNet-DBNet-is-a-novel-network-architecture-for-real-time-scene-text.png)

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
uv run python runners/test.py preset=example "checkpoint_path=outputs/ocr_training/checkpoints/epoch--step-1845.ckpt"
```

#### 예측
```bash
# 사용 예시
uv run python runners/predict.py preset=example "checkpoint_path=outputs/ocr_training/checkpoints/epoch-8-step-1845.ckpt"
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

### 환경 설정

```bash
# 의존성 설치
uv sync

# 개발 의존성 설치 (개발용)
uv sync --extra dev
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

- [DBNet](https://github.com/MhLiao/DB)
- [Hydra](https://hydra.cc/docs/intro/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [CLEval](https://github.com/clovaai/CLEval)
- [UV 패키지 관리자](https://github.com/astral-sh/uv)

## 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 다음을 통해 연락해 주세요:

- **이슈 트래커**: [GitHub Issues](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues)
<!-- - **이메일**: [팀 대표 이메일 주소]
- **디스코드**: [팀 디스코드 채널] -->

---

**마지막 업데이트**: 2025년 9월 23일
```
