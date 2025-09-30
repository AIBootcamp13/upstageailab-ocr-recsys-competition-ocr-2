# OCR 프로젝트 Bash 별칭 및 함수 설정 가이드

## 개요

이 문서는 OCR 프로젝트에서 사용하는 편리한 Bash 별칭(alias)과 함수들을 설정하고 사용하는 방법을 설명합니다. 이러한 도구들은 개발 생산성을 향상시키기 위해 만들어졌습니다.

## 설치 방법

### 자동 설치 (권장)

프로젝트 루트 디렉토리에서 다음 명령을 실행하세요:

```bash
./scripts/setup/02_setup-bash-aliases.sh
```

이 스크립트는 다음 작업을 수행합니다:
- 기존 `.bashrc` 파일을 백업합니다
- 중복 추가를 방지하면서 필요한 함수와 별칭을 추가합니다
- 설치 완료 후 터미널을 재시작하거나 `source ~/.bashrc`를 실행하라고 안내합니다

### 수동 설치

`.bashrc` 파일 끝에 다음 내용을 추가하세요:

```bash
# Generic make wrapper
function omake() {
  (cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2 && make "$@")
}

alias mcb='omake serve-ui'
alias mev='omake serve-evaluation-ui'
alias minf='omake serve-inference-ui'

function ui-train() { omake serve-ui PORT="${1:-8502}"; }
function ui-eval()  { omake serve-evaluation-ui PORT="${1:-8503}"; }
function ui-infer() { omake serve-inference-ui PORT="${1:-8504}"; }
```

## 사용법

### 1. `omake` 함수 - 범용 Make 래퍼

**목적**: 프로젝트 디렉토리로 자동 이동하여 make 명령을 실행합니다.

**사용법**:
```bash
omake [make-target]
```

**예시**:
```bash
omake lint          # 코드 린팅 실행
omake test          # 테스트 실행
omake serve-ui      # UI 서버 시작
omake install       # 의존성 설치
```

**장점**: 어느 디렉토리에서든 프로젝트의 make 명령을 실행할 수 있습니다.

### 2. 간단한 UI 별칭들

**`mcb`** - Command Builder UI 시작 (기본 포트: 8501)
```bash
mcb
# 또는
mcb PORT=8505  # 다른 포트 지정
```

**`mev`** - Evaluation Results Viewer 시작
```bash
mev
```

**`minf`** - OCR Inference UI 시작
```bash
minf
```

### 3. 고급 UI 함수들

**`ui-train`** - Training Command Builder (기본 포트: 8502)
```bash
ui-train        # 포트 8502에서 시작
ui-train 8506   # 지정된 포트에서 시작
```

**`ui-eval`** - Evaluation Results Viewer (기본 포트: 8503)
```bash
ui-eval         # 포트 8503에서 시작
ui-eval 8507    # 지정된 포트에서 시작
```

**`ui-infer`** - OCR Inference UI (기본 포트: 8504)
```bash
ui-infer        # 포트 8504에서 시작
ui-infer 8508   # 지정된 포트에서 시작
```

## 사용 시나리오

### 개발 중 자주 사용하는 명령들

```bash
# 코드 품질 검사
omake lint
omake format

# 테스트 실행
omake test

# UI 도구들 실행
mcb          # 명령어 빌더
mev          # 평가 결과 뷰어
minf         # 추론 UI

# 여러 UI를 동시에 실행 (다른 포트 사용)
ui-train 8502
ui-eval 8503
ui-infer 8504
```

### CI/CD 파이프라인에서

```bash
# 빌드 및 테스트
omake install
omake test
omake quality-check
```

## 문제 해결

### 변경사항이 적용되지 않는 경우

설정 변경 후 새 터미널을 열거나 다음 명령을 실행하세요:

```bash
source ~/.bashrc
```

### 별칭이 작동하지 않는 경우

1. `.bashrc` 파일이 제대로 로드되었는지 확인:
   ```bash
   echo $SHELL
   cat ~/.bashrc | tail -20
   ```

2. 함수가 정의되었는지 확인:
   ```bash
   type omake
   type mcb
   ```

### 백업 복원

설치 스크립트는 자동으로 백업을 생성합니다. 문제가 발생하면:

```bash
# 백업 파일 찾기
ls -la ~/.bashrc.backup.*

# 백업 복원
cp ~/.bashrc.backup.20250129_120000 ~/.bashrc
source ~/.bashrc
```

## 고급 사용법

### 사용자 정의 포트 설정

환경변수를 사용하여 기본 포트를 변경할 수 있습니다:

```bash
PORT=8505 mcb
PORT=8506 mev
```

### 여러 프로젝트에서 사용

다른 프로젝트에서도 유사한 함수를 만들 수 있습니다:

```bash
function project2-make() {
  (cd /path/to/project2 && make "$@")
}
```

## 주의사항

- 이 별칭들은 VS Code 개발 환경용으로 최적화되어 있습니다
- `/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2` 경로가 올바른지 확인하세요
- 팀원 모두가 동일한 설정을 사용하도록 하세요

## 문의

설정이나 사용법에 대한 질문이 있으면 팀 리드에게 문의하세요.
