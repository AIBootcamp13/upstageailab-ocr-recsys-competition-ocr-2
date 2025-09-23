# **AI 경진대회: 영수증 텍스트 검출**

이 프로젝트는 영수증 이미지에서 텍스트의 위치를 추출하는 것에 중점을 둡니다. 주어진 영수증 이미지에서 텍스트 요소 주변에 경계 다각형을 정확하게 식별하고 생성할 수 있는 모델을 구축하는 것이 목표입니다.

* **대회 기간:** 2025년 9월 22일 (10:00) - 2025년 10월 16일 (19:00)
* **주요 과제:** 영수증 이미지 내 텍스트 영역 식별 및 윤곽 그리기

## **1. 프로젝트 설정**

### **환경 설정**

대회 환경은 사전 구성되어 있지만, 필요한 패키지를 설치하여 로컬에서 설정을 복제할 수 있습니다.

```bash
uv sync
```

호환성을 보장하기 위해 Python 3.8 이상을 사용하세요. UV가 자동으로 가상 환경을 생성하고 종속성을 설치합니다.

### **데이터 및 코드 다운로드**

`wget`을 사용하여 대회 데이터셋과 베이스라인 소스 코드를 다운로드합니다.

1. **데이터셋 다운로드:**
    ```bash
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000293/data/code.tar.gz
    ```

2. **베이스라인 코드 다운로드:**
    ```bash
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000361/data/data.tar.gz
    ```

3. **파일 압축 해제:**
    ```bash
    # 데이터셋 압축 해제
    tar xvfz ./data.tar.gz

    # 베이스라인 코드 압축 해제
    tar xvfz ./code.tar.gz
    ```

### **구성**

구성 파일에서 데이터셋 경로를 업데이트해야 합니다. `configs/preset/datasets/db.yaml`의 `base_path` 변수를 데이터를 압축 해제한 위치로 수정하세요.

## **2. 모델 및 훈련 설정**

### **파일 구조**

```
└─── configs
    ├── preset
    │   ├── example.yaml
    │   ├── base.yaml
    │   ├── datasets
    │   │   └── db.yaml
    │   ├── lightning_modules
    │   │   └── base.yaml
    │   ├── metrics
    │   │   └── cleval.yaml
    │   └── models
    │       ├── decoder
    │       │   └── unet.yaml
    │       ├── encoder
    │       │   └── timm_backbone.yaml
    │       ├── head
    │       │   └── db_head.yaml
    │       ├── loss
    │       │   └── db_loss.yaml
    │       ├── postprocess
    │       │   └── base.yaml
    │       └── model_example.yaml
    ├── train.yaml
    ├── test.yaml
    └── predict.yaml
```

### **주요 구성 파일**

- `train.yaml`, `test.yaml`, `predict.yaml`: 러너 실행에 필요한 설정
- `preset/example.yaml`: 각 모듈의 구성 파일 지정
- `preset/datasets/db.yaml`: Dataset, Transform, 데이터 관련 설정
- `preset/lightning_modules/base.yaml`: PyTorch Lightning 실행 관련 설정
- `preset/metrics/cleval.yaml`: CLEval 평가 관련 설정
- `preset/models/model_example.yaml`: 각 모델 모듈과 Optimizer의 구성 파일 지정
- `preset/models/*`: 모델 구성에 필요한 각 모듈 관련 설정

## **3. 데이터셋 개요**

데이터는 이미지 폴더와 주석을 위한 해당 JSON 파일로 구성됩니다.

### **디렉토리 구조**

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

### **JSON 주석 형식**

JSON 파일은 이미지 파일명을 텍스트 경계 상자의 좌표에 매핑합니다.

* **IMAGE_FILENAME**: 각 이미지 레코드의 키
* **words**: 이미지에 대해 감지된 모든 텍스트 인스턴스를 포함하는 객체
* **nnnn**: 0001부터 시작하는 각 단어 인스턴스의 고유한 4자리 인덱스
* **points**: 텍스트 주변의 다각형을 정의하는 [X, Y] 좌표 쌍의 배열. 원점 (0,0)은 이미지의 왼쪽 상단 모서리입니다. 평가에 유효한 다각형이 되려면 최소 4개의 점이 필요합니다.

## **4. 베이스라인 모델**

제공된 베이스라인 코드는 장면 텍스트 검출에서 효과적인 것으로 알려진 **DBNet** 아키텍처를 기반으로 구축되었습니다.

### **성능**

V100 GPU에서 10 에포크 훈련 후, 베이스라인 모델은 공개 테스트 세트에서 다음과 같은 성능을 달성했습니다:

* **훈련 시간:** 약 22분
* **H-Mean:** 0.8818
* **정밀도:** 0.9651
* **재현율:** 0.8194

### **모델 아키텍처**

이 베이스라인 코드는 DBNet을 기반으로 합니다.

#### **DBNet: 미분 가능한 이진화를 통한 실시간 장면 텍스트 검출**

![DBNet](https://www.researchgate.net/publication/369783176/figure/fig1/AS:11431281137414188@1680649387586/Structure-of-DBNet-DBNet-is-a-novel-network-architecture-for-real-time-scene-text.png)

### **평가 지표**

이 대회는 텍스트 검출 결과를 평가하기 위해 **CLEval**을 사용합니다.

#### **CLEval: 텍스트 검출 및 인식 작업을 위한 문자 수준 평가**

![CLEval](https://github.com/clovaai/CLEval/raw/master/resources/screenshots/explanation.gif)

## **5. 실행 방법**

일관되고 재현 가능한 결과를 위해 고정된 랜덤 SEED를 사용하는 것이 권장됩니다. 다음 스크립트들은 훈련, 테스트, 예측 워크플로를 처리합니다.

### **모델 훈련**

```bash
python runners/train.py preset=example
```

### **모델 테스트**

평가를 실행하려면 저장된 체크포인트의 경로를 제공하세요.

```bash
# 사용 예시
python runners/test.py preset=example "checkpoint_path=outputs/ocr_training/checkpoints/epoch--step-1845.ckpt"
```

### **예측 생성**

이 명령은 테스트 세트에서 모델을 실행하고 예측 결과가 포함된 JSON 파일을 생성합니다.

```bash
# 사용 예시
python runners/predict.py preset=example "checkpoint_path=outputs/ocr_training/checkpoints/epoch-8-step-1845.ckpt"
```

## **6. 제출 과정**

### **제출 파일 생성**

예측 스크립트는 JSON 파일을 생성합니다. 이 파일은 제출하기 전에 제공된 유틸리티 스크립트를 사용하여 필요한 CSV 형식으로 변환해야 합니다.

```bash
# 사용 예시
python ocr/utils/convert_submission.py --json_path outputs/ocr_training/submissions/your_submission.json --output_path submission.csv
```

### **CSV 형식**

제출 파일은 `filename`과 `polygons` 두 열이 있는 CSV여야 합니다.

* **filename**: 테스트 세트의 이미지 파일명
* **polygons**: 해당 이미지에서 예측된 모든 텍스트 영역의 좌표를 포함하는 단일 문자열
  * 하나의 다각형에 대한 좌표는 공백으로 구분됩니다 (예: `X1 Y1 X2 Y2 X3 Y3 X4 Y4`)
  * 같은 이미지의 다른 다각형들은 파이프 문자(`|`)로 구분됩니다

### **출력 파일 저장 경로**

#### **파일 구조**

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

#### **주요 파일**

- `outputs/{exp_name}/submissions/{timestamp}.json`: 제출 파일
- `outputs/{exp_name}/checkpoints/epoch={epoch}-step={step}.ckpt`: 훈련된 모델 체크포인트 파일
- `outputs/{exp_name}/.hydra/*.yaml`: 실행 중 입력된 구성 값

## **7. 평가 기준**

대회 리더보드는 공개 및 비공개 순위로 나뉩니다. 대회 기간 중에는 공개 세트에 대한 점수가 표시됩니다. 최종 우승자는 대회 종료 후 공개되는 비공개 테스트 세트에서의 모델 성능으로 결정됩니다. 테스트 데이터는 공개 및 비공개 세트 간에 동등하게(50/50) 분할됩니다.

## **참고 자료**

- [DBNet](https://github.com/MhLiao/DB)
- [Hydra](https://hydra.cc/docs/intro/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [CLEval](https://github.com/clovaai/CLEval)
