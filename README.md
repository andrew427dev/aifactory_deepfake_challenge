# AI Factory Deepfake Challenge

본 리포지토리는 AI Factory 주최 딥페이크 탐지 대회의 베이스라인을 확장하기 위한 실험용 코드베이스입니다. Vision Transformer(ViT)를 기반으로 한 이미지·동영상 이중 처리 파이프라인과, 데이터 준비부터 제출 파일 생성까지 이어지는 End-to-End(E2E) 워크플로우를 목표로 합니다.

## 리포지토리 구조

```
configs/                 # 데이터·모델·추론 파이프라인 설정(YAML)
scripts/                 # prepare/train/validate/infer/submission 스크립트
src/                     # 공용 모듈(utils, evaluation, inference 등)
tests/                   # pytest 기반 단위 테스트
Makefile                 # 주요 작업을 자동화한 명령 모음
requirements.txt         # 필수 Python 패키지 목록
data/{raw,processed}/    # 로컬 데이터 디렉터리(버전관리 제외)
experiments/             # 체크포인트와 로그가 저장될 위치(생성 시 자동)
submission/              # 제출용 CSV 생성 위치
```

> `data/`, `experiments/`, `model/`, `submission/` 디렉터리는 `.gitignore`에 의해 버전관리에서 제외되어 있으며, 로컬에서 명령 실행 시 자동 생성됩니다.

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

CUDA, torchvision, opencv-python, transformers 등을 포함하므로 GPU 환경을 권장합니다.

### 2. 데이터 준비

`configs/data.yaml`을 수정하여 `raw` 데이터 경로와 전처리 옵션을 지정합니다. 이후 다음 명령으로 프레임 추출·이미지 복사 등의 전처리를 수행합니다.

```bash
make prepare
```

### 3. 학습

`configs/train.yaml`과 `configs/model.yaml`을 조정한 뒤 다음 명령으로 학습을 실행합니다. 현재는 더미 루프만 구현되어 있으며, 추후 Dataset/Dataloader/Optimizer가 추가될 예정입니다.

```bash
make train
```

### 4. 검증

체크포인트를 로드하여 Macro F1 평가를 수행하도록 확장할 예정입니다. 현 버전은 구조 확인용 로그만 출력합니다.

```bash
make val
```

### 5. 추론 및 제출 파일 생성

```bash
make infer     # submission/submission.csv 생성
make submit    # 스키마 검증
```

`tests/test_submission_schema.py`에서 제출 CSV가 요구 형식을 만족하는지 단위 테스트를 제공하고 있습니다.

## 설정 파일 개요

- `configs/data.yaml`: 원본/전처리 경로, 이미지 크기, 동영상 프레임 샘플링 옵션.
- `configs/model.yaml`: 사용할 Hugging Face 모델 아티팩트 또는 체크포인트 경로, 클래스/레이블 매핑.
- `configs/train.yaml`: 학습 관련 하이퍼파라미터, 시드, 실험 로그/체크포인트 경로.
- `configs/infer.yaml`: 추론 배치 크기, 비디오 확률 집계 방식, 제출 파일 저장 경로.

각 설정 파일에는 주석을 포함한 템플릿 값이 제공되며, 필요에 따라 사용자 환경에 맞게 덮어쓰면 됩니다.

## 테스트

단위 테스트는 `pytest`로 실행할 수 있습니다.

```bash
pytest
```

향후 Dataset과 학습 루프가 추가되면 E2E 테스트가 확장될 예정입니다.

## 참고 문서

- [프로젝트 개요 PID](PID.md) – 진행 상황과 향후 계획 요약 *(추후 추가 예정)*

---
