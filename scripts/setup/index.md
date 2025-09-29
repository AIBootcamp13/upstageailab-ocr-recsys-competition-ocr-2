# Setup Scripts

이 디렉토리에는 프로젝트 설정을 위한 스크립트들이 포함되어 있습니다. 스크립트들은 실행 권장 순서대로 번호가 매겨져 있습니다.

## 스크립트 목록

### 00_setup-environment.sh
프로젝트 환경 설정 스크립트입니다. UV를 사용하여 Python 가상환경을 생성하고, 의존성을 설치하며, 기본적인 임포트 테스트를 수행합니다.

### 01_setup-professional-linting.sh
전문적인 Python 린팅 환경을 설정합니다. Ruff, pre-commit 훅, VS Code 설정, CI 파이프라인을 구성합니다.

### 02_setup-bash-aliases.sh
프로젝트용 Bash 별칭과 함수를 설정합니다. `omake`, `mcb`, `mev`, `minf` 등의 편리한 명령어를 추가합니다.

## 사용법

스크립트들은 순서대로 실행하는 것을 권장합니다:

```bash
./00_setup-environment.sh
./01_setup-professional-linting.sh
./02_setup-bash-aliases.sh
```

## 관련 문서

- [Bash 별칭 및 함수 설정 가이드](../../docs/setup/BASH_ALIASES_KO.md)
