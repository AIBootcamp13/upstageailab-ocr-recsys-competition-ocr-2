## Fast dev run
```bash
uv run python runners/train.py --config-name train trainer.fast_dev_run=true
```

## Train run CLI
```bash
uv run python /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py \
    exp_name=transforms_test_caching-dbnetpp-dbnetpp_decoder-resnet518 \
    logger.wandb.enabled=true \
    model/architectures=dbnetpp \
    model.encoder.model_name=resnet50 \
    model.component_overrides.decoder.name=dbnetpp_decoder \
    model.component_overrides.head.name=dbnetpp_head \
    model.component_overrides.loss.name=dbnetpp_loss \
    model/optimizers=adamw \
    model.optimizer.lr=0.000305 \
    model.optimizer.weight_decay=0.0001 \
    model."scheduler._target_"=torch.optim.lr_scheduler.CosineAnnealingLR \
    model.scheduler.T_max=1000 \
    model.scheduler.eta_min=0.00001 \
    dataloaders.train_dataloader.batch_size=16 \
    dataloaders.val_dataloader.batch_size=16 \
    trainer.max_epochs=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.gradient_clip_val=5.0 \
    trainer.precision=32 \
    seed=42 \
    data=canonical
```
----


## 요약 보고서용 스크립트

### 1. **실험 분석** (analyze_experiment.py)
- **목적**: W&B CSV 내보내기에서 훈련 메트릭을 분석
- **출력**: 성능 메트릭, 이상 현상 및 데이터 인사이트가 포함된 JSON 요약
- **사용법**: W&B의 CSV 내보내기가 필요; 성능 저하를 감지하고 구조화된 요약을 생성

### 2. **절제 연구 결과 수집기** (collect_results.py)
- **목적**: 비교를 위해 여러 W&B 실행에서 결과를 수집
- **출력**: 통계(평균, 표준편차, 최대, 최소)가 포함된 CSV 파일 및 표 형식 요약
- **사용법**:
  ```bash
  uv run python ablation_study/collect_results.py --project "receipt-text-recognition-ocr-project" --output training_summary.csv
  ```
- **참고**: 여러 실행을 위해 설계되었지만 단일 실행에서도 작동 가능

### 3. **성능 기준 보고서** (generate_baseline_report.py)
- **목적**: W&B 프로파일링 실행에서 포괄적인 마크다운 보고서를 생성
- **출력**: 메트릭, 병목 현상 및 권장 사항이 포함된 마크다운 보고서
- **사용법**:
  ```bash
  uv run python scripts/performance/generate_baseline_report.py \
    --run-id jn3156xt \
    --project "receipt-text-recognition-ocr-project" \
    --output training_report.md
  ```

### 4. **대화형 평가 UI** (single_run.py)
- **목적**: 단일 실행 예측의 스트림릿 기반 대화형 분석
- **출력**: 예측, 메트릭 및 시각화를 탐색하는 웹 인터페이스
- **사용법**:
  ```bash
  uv run python run_ui.py evaluation_viewer
  ```

## 귀하의 특정 실행에 대해

W&B URL에서 실행 ID(`jn3156xt`)를 가지고 있으므로 다음을 수행할 수 있습니다:

1. **상세한 마크다운 요약을 위해 성능 보고서 스크립트 사용**
2. **W&B에서 데이터를 내보내고 실험 분석 스크립트 사용**
3. **평가 UI를 사용하여 대화형으로 보기**

성능 기준 보고서 스크립트는 독립형 요약 문서를 생성하는 데 가장 포괄적인 옵션입니다.




----
