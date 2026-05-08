# 제조산업 품질검사를 위한 딥러닝 기반 비전검사 최적 모델 탐색 대시보드 PRD

---

## 1. 문서 개요

| 항목 | 내용 |
|---|---|
| **프로젝트명** | 제조산업 품질검사를 위한 딥러닝 기반 비전검사 최적 모델 탐색 대시보드 |
| **버전** | v1.0 (MVP) |
| **작성일** | 2026-05-08 |
| **작성 목적** | 자동차 볼트 외관 결함 탐지를 위한 비전 검사 모델(EfficientAD, PatchCore)의 학습·평가·비교를 단일 Streamlit 대시보드에서 수행할 수 있도록 기능·비기능 요구사항을 명세 |
| **대상 독자** | 데이터 분석가, ML 엔지니어, 프로젝트 매니저, MLOps 엔지니어 |
| **MVP 제외 항목** | 앙상블(EfficientAD + PatchCore 가중 평균), GAN 기반 이미지 증강, 전처리·증강 적용 순서 설정 |

---

## 2. 배경 및 문제 정의

### 2.1 현재 문제점
- 자동차 볼트 외관 검사가 작업자의 **육안 검사**에 의존하여 검사자별 판정 기준이 상이함 (주관성)
- 검사 공정이 **병목**으로 작용하여 전체 생산 라인의 처리량 저하
- 결함 유형(crack, scratch 등)이 다양하고 미세하여 작업자의 누적 피로도에 따라 판정 정확도가 변동

### 2.2 해결 목표
- 딥러닝 기반 이상 탐지(Anomaly Detection) 모델을 활용하여 **객관적·일관된 판정 기준** 확보
- 데이터 분석가/ML 엔지니어가 코드 작성 없이 **다양한 모델·전처리 조합을 실험**하고 최적 모델을 탐색
- 학습된 모델을 **state_dict + configs.yaml** 형태로 저장하여 추론 애플리케이션과 연계 가능한 자산화

### 2.3 성공 지표
- MVTec AD Screw 데이터셋 기준 **AUC ≥ 0.95** 달성 모델 탐색 가능
- 단일 실험 사이클(설정 → 학습 → 평가) **30분 이내** 완료 (g4dn.xlarge 기준)
- 실험 히스토리 기반으로 **모델·전처리 조합 비교**가 한 화면에서 가능

---

## 3. 사용자 정의 (User Persona)

### 3.1 주 사용자: 데이터 분석가 / ML 엔지니어 (전문 사용자)

| 속성 | 내용 |
|---|---|
| **기술 수준** | Python·PyTorch 기본 이해, 이상 탐지 모델 개념 이해, Docker 기본 사용 가능 |
| **사용 환경** | 로컬 개발 머신 또는 AWS EC2 GPU 인스턴스(g4dn.xlarge) |
| **주요 목표** | 다양한 전처리·모델·하이퍼파라미터 조합 실험 → 최적 조합 탐색 → 모델 자산화 |
| **목표 행동** | ① 데이터 폴더 구조 검증 → ② 전처리·모델 파라미터 설정 → ③ 학습 실행·모니터링 → ④ 실험 결과 비교 → ⑤ Anomaly Map으로 모델 행동 검증 → ⑥ 최적 모델 저장 |
| **페인 포인트** | 모델별 별도 코드 작성 부담, 실험 재현성 확보 어려움, 파라미터 조합 비교의 수작업 과정 |

---

## 4. 기능 요구사항 (Functional Requirements)

### 우선순위 정의
- **Must Have (M)**: MVP 출시 필수 기능
- **Should Have (S)**: MVP 포함 권장 기능
- **Nice to Have (N)**: MVP 이후 추가 검토 기능 (제외 항목 포함)

---

### 4.1 탭1 — 데이터 폴더 구조

| 우선순위 | 기능 |
|---|---|
| **M** | 로컬 디렉토리 경로 입력, MVTec AD 스타일 폴더 구조 검증, 트리 시각화, 폴더별 이미지 수 표시, 대표 샘플 썸네일, Grayscale 자동 감지 후 RGB 변환 안내 |
| **S** | 폴더 구조 오류 시 경고 메시지, 지원 포맷(`.jpg`, `.png`, `.bmp`) 외 파일 필터링 |
| **N** | 디렉토리 탐색(browse) 다이얼로그 (Streamlit 한계로 대안 UI 검토) |

#### Input / Process / Output

| 단계 | 내용 |
|---|---|
| **Input** | 로컬 디렉토리 경로 (텍스트 입력) |
| **Process** | ① 폴더 구조 스캔 → ② MVTec AD 스타일 검증(`train/good`, `test/<defect>`, `ground_truth/<defect>`) → ③ 이미지 수 카운트 → ④ 채널 수 감지(Grayscale → RGB 변환) → ⑤ 대표 이미지 썸네일 추출 |
| **Output** | 폴더 트리 시각화, 폴더별 이미지 수 테이블, 썸네일 그리드, 채널 변환 안내 메시지, 구조 오류 시 경고, `session_state.dataset_path` 저장 |

---

### 4.2 탭2 — 전처리 파라미터 설정

| 우선순위 | 기능 |
|---|---|
| **M** | 전처리 모델 라디오 선택(선택 안 함 / Homomorphic / HE / CLAHE), 적용 전·후 미리보기, Resize+Padding 고정, resize 크기 입력, 정규화 방식(ImageNet/커스텀) 선택, configs.yaml 저장/불러오기 |
| **S** | 비선택 모델 파라미터 UI 완전 숨김, Grayscale 자동 RGB 변환 |
| **N (MVP 제외)** | **GAN 기반 이미지 증강 (적용 여부 토글, 생성 이미지 수 입력, 생성 샘플 미리보기)** — *MVP 제외* |
| **N (MVP 제외)** | **전처리 필터와 GAN 증강의 적용 순서 설정** — *MVP 제외, 향후 확장 고려사항* |

#### 전처리 모델별 파라미터

| 모델 | 파라미터 | UI |
|---|---|---|
| 선택 안 함 | - | 파라미터 UI 없음 |
| Homomorphic Filter | sigma, gamma_H, gamma_L, cutoff, normalize | 슬라이더 |
| HE | 없음 | 안내 텍스트만 표시 |
| CLAHE | clipLimit | 슬라이더 |

#### Input / Process / Output

| 단계 | 내용 |
|---|---|
| **Input** | 전처리 모델 선택, 모델별 파라미터, resize 목표 크기, 정규화 방식(ImageNet / 커스텀 mean·std) |
| **Process** | ① 선택된 필터로 샘플 이미지 변환 → ② Resize + Padding (검정 0 고정, 비율 유지) → ③ 정규화 적용 → ④ 좌(원본)/우(필터 후) 비교 미리보기 생성 |
| **Output** | 미리보기 이미지, `session_state.preprocessing_config`, configs.yaml 전처리 섹션 저장 파일 |

---

### 4.3 탭3 — 모델 파라미터 설정

| 우선순위 | 기능 |
|---|---|
| **M** | 모델 라디오 선택(EfficientAD / PatchCore), 공통·전용 파라미터 입력, 고급 설정 expander, Threshold 방식 선택(Percentile/Absolute), 디바이스 자동 감지 표시, configs.yaml 저장/불러오기 |
| **S** | image_size 탭2 연동 자동 반영, 현재 threshold 기준 정상/결함 판정 비율 실시간 표시 |
| **N (MVP 제외)** | **앙상블(EfficientAD + PatchCore) 선택 옵션 및 가중치 슬라이더(합산 1.0 고정, 기본값 0.5:0.5)** — *MVP 제외* |
| **N (MVP 제외)** | **앙상블 모드 선택 시 Percentile 자동 전환 및 안내 메시지** — *MVP 제외* |

#### 공통 파라미터
- `image_size` (탭2 연동), `batch_size`, `random_seed`

#### EfficientAD 전용 파라미터

| 구분 | 파라미터 |
|---|---|
| **기본 노출** | model_size(small/medium), train_steps, optimizer(Adam/AdamW/SGD), learning_rate, weight_decay, out_channels, padding, ae_loss_weight ↔ st_loss_weight (합산 1.0 슬라이더) |
| **고급 설정 (st.expander)** | autoencoder_lr, autoencoder_weight_decay, lr_decay_epochs, lr_decay_factor, 스케줄러(StepLR/CosineAnnealingLR), imagenet_penalty_weight, penalty_batch_size |

#### PatchCore 전용 파라미터

| 구분 | 파라미터 |
|---|---|
| **기본 노출** | backbone(WideResNet50 등), Pretrained Weights 방식(torchvision 자동 / 로컬 경로), coreset_sampling_ratio, neighbourhood_kernel_size |
| **고급 설정 (st.expander)** | max_train, k-NN, Top-k Ratio |

#### Anomaly Score Threshold

| 방식 | 기본값 | UI |
|---|---|---|
| Percentile (기본) | 95 | 슬라이더 0~100% |
| Absolute Score | 0.5 | 슬라이더 0.0~1.0 |

#### Input / Process / Output

| 단계 | 내용 |
|---|---|
| **Input** | 모델 종류, 공통 파라미터, 모델 전용 파라미터, threshold 방식·값 |
| **Process** | ① 디바이스 감지(`torch.cuda.is_available()`) → ② 파라미터 유효성 검증 → ③ threshold 기준 임시 통계 계산 |
| **Output** | 디바이스 정보(예: "현재 디바이스: CUDA (Tesla T4)"), 정상/결함 비율 실시간 표시, `session_state.model_config`, configs.yaml 모델 섹션 저장 |

---

### 4.4 탭4 — 학습 시작 + 학습 로그

| 우선순위 | 기능 |
|---|---|
| **M** | [학습 시작] 버튼, 진행률 Progress Bar, 실시간 Loss 곡선, 학습 로그 텍스트 박스, [학습 중지] 버튼(완전 중단), 완료 알림 및 총 소요 시간, 실험 이름 입력란 |
| **S** | 실험 이름 자동 생성(`{모델명}_{날짜}_{시간}`), 중단 시 히스토리에 "중단" 상태 기록 |
| **N (MVP 제외)** | **앙상블 선택 시 EfficientAD → PatchCore 순차 실행 후 Score 결합** — *MVP 제외* |

#### Input / Process / Output

| 단계 | 내용 |
|---|---|
| **Input** | 탭2·탭3 설정값(`session_state`), 실험 이름 |
| **Process** | ① 데이터 로더 구성 → ② 모델 초기화 → ③ 학습 루프 실행 (step별 loss 갱신, UI 비동기 업데이트) → ④ 중지 신호 감시 → ⑤ 완료 시 평가 메트릭 계산 |
| **Output** | 진행률 바, 실시간 Loss 차트, 로그 텍스트(`[Step 5000/70000] Train Loss: 0.0342 | Time: 12.3s`), 완료 알림, `session_state.experiments[exp_id]` 갱신 |

#### 중단 처리
- 완전 중단: 그 시점까지의 학습 결과 **저장하지 않고 폐기**
- 히스토리에 `status: "중단"`으로 기록, `metrics`는 `null`

---

### 4.5 탭5 — 실험 히스토리 + 결과 상세 + 모델 저장

| 우선순위 | 기능 |
|---|---|
| **M** | 실험 목록 테이블, 실험 선택 시 상세 결과 표시(Confusion Matrix, ROC, Anomaly Score 분포), 모델 저장(state_dict + configs.yaml), 실험 삭제 |
| **S** | 다중 실험 선택 비교 차트(Accuracy, Precision, Recall, F1, F2), 저장 완료 시 경로·파일명·용량 출력 |
| **N** | 실험 태그·메모 기능 |

#### 실험 목록 테이블 컬럼
실험명 / 모델 종류 / 주요 파라미터 요약 / Accuracy / Precision / Recall / F1 / F2 / AUC / 실행 시각 / 상태(완료/중단)

#### 결과 상세 출력
- 성능 지표 카드: Accuracy, Precision, Recall, F1
- Confusion Matrix (heatmap)
- ROC Curve + AUC
- Anomaly Score 분포 히스토그램 (정상 vs 결함 중첩 비교)

#### 모델 저장 명세
- 저장 방식: **state_dict + configs.yaml 고정** (옵션 UI 없음)
- 저장 항목: `model_state_dict.pth`, `configs.yaml`
- 저장 경로 입력란 + [모델 저장] 버튼
- 저장 완료 시 경로·파일명·용량 정보 출력

#### Input / Process / Output

| 단계 | 내용 |
|---|---|
| **Input** | 실험 목록 클릭, 비교 대상 다중 선택, 저장 경로 |
| **Process** | ① JSON 히스토리 파일 로드 → ② 선택 실험 메트릭 계산·렌더링 → ③ 저장 시 state_dict + configs.yaml 직렬화 |
| **Output** | 테이블, 메트릭 카드/차트, 비교 차트, 저장 결과 메시지 |

---

### 4.6 탭6 — 이상 영역 시각화 (Anomaly Map)

| 우선순위 | 기능 |
|---|---|
| **M** | 탭5 실험 선택 연동, 테스트 이미지 목록 테이블, 3분할 시각화(원본/GT 마스크/Heatmap), Threshold 슬라이더 실시간 갱신, PNG 저장 |
| **S** | 결함 유형 필터 드롭다운, 이미지별 최대/평균 Anomaly Score 표시, FP/FN 표시 |
| **N** | 다중 이미지 일괄 PNG 내보내기 |

#### 테스트 이미지 목록 컬럼
이미지명 / Anomaly Score / OK·NG 판정 / GT 일치 여부 / 오분류(FP/FN)

#### Input / Process / Output

| 단계 | 내용 |
|---|---|
| **Input** | 선택된 실험 ID, 테스트 이미지, threshold 슬라이더 값, 결함 유형 필터 |
| **Process** | ① 모델 로드 → ② 테스트 이미지 추론 → ③ Anomaly Score·Heatmap 생성 → ④ threshold 기준 이진화 → ⑤ GT 마스크와 비교 |
| **Output** | 3분할 이미지(원본 / GT / jet colormap heatmap), 점수 수치, 이진화 결과, PNG 다운로드 파일 |

---

## 5. 비기능 요구사항 (Non-Functional Requirements)

| 항목 | 요구사항 |
|---|---|
| **성능 (UI)** | 학습 중 UI 블로킹 없음 — Streamlit `st.empty()` + 주기적 `rerun` 또는 background thread + queue 패턴 적용 |
| **성능 (학습)** | g4dn.xlarge(Tesla T4) 기준 EfficientAD-medium 70,000 steps **20분 이내**, PatchCore(coreset 10%) **10분 이내** 목표 |
| **디바이스 자동 감지** | `torch.cuda.is_available()` 기반 자동 전환, UI에 GPU/CPU 명시 |
| **재현성** | Docker 이미지 단일화(로컬·서버 동일), `random_seed` 고정 시 동일 결과 보장 |
| **배포 환경** | AWS EC2 g4dn.xlarge (Tesla T4 GPU, 16GB VRAM, 4 vCPU, 16GB RAM) |
| **이미지 정규화** | 다양한 해상도 자동 처리 — Resize + Padding (검정 0 고정, 비율 유지) |
| **채널 처리** | Grayscale 자동 감지 후 RGB 3채널 변환 |
| **로깅** | 학습 로그·에러 로그 파일 저장 (`./logs/{experiment_id}.log`) |
| **데이터 무결성** | 폴더 구조 검증 실패 시 학습 진입 차단 |

---

## 6. 기술 스택 및 의존성

### 6.1 핵심 스택

| 카테고리 | 기술 |
|---|---|
| **UI** | Streamlit ≥ 1.30 |
| **ML 프레임워크** | PyTorch ≥ 2.1, torchvision |
| **이상 탐지 모델** | EfficientAD, PatchCore (Anomalib 또는 자체 구현) |
| **이미지 처리** | OpenCV (`opencv-python-headless`), Pillow |
| **평가 지표** | scikit-learn |
| **시각화** | Matplotlib, Plotly |
| **설정 관리** | PyYAML |
| **컨테이너** | Docker, NVIDIA Container Toolkit |
| **클라우드** | AWS EC2 (g4dn.xlarge) |

### 6.2 환경 명세

| 항목 | 버전 |
|---|---|
| **Python** | 3.12 |
| **CUDA** | 12.4 |
| **cuDNN** | 9.1 |
| **베이스 Docker 이미지** | `nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04` |

### 6.3 지원 이미지 포맷
`.jpg`, `.png`, `.bmp`

---

## 7. 데이터 흐름도 (Data Flow)

### 7.1 탭 간 흐름

```
[탭1: 데이터 폴더]
  ↓ session_state.dataset_path
[탭2: 전처리 설정]
  ↓ session_state.preprocessing_config
[탭3: 모델 설정]
  ↓ session_state.model_config
[탭4: 학습 실행]
  ↓ session_state.experiments[exp_id]
[탭5: 히스토리·결과]
  ↓ session_state.selected_experiment_id
[탭6: Anomaly Map]
```

### 7.2 session_state 키 명세

| 탭 | 생성/갱신 키 | 소비 키 |
|---|---|---|
| **탭1** | `dataset_path`, `dataset_meta` (이미지 수, 채널) | - |
| **탭2** | `preprocessing_config` | `dataset_path` |
| **탭3** | `model_config`, `device_info` | `preprocessing_config.image_size` |
| **탭4** | `experiments[exp_id]`, `current_run_status` | `dataset_path`, `preprocessing_config`, `model_config` |
| **탭5** | `selected_experiment_id` | `experiments` |
| **탭6** | `anomaly_map_threshold` | `selected_experiment_id`, `experiments` |

### 7.3 데이터 없음 안내 표준 문구

| 상황 | 메시지 |
|---|---|
| 탭1 미설정 상태로 탭2 진입 | "먼저 탭1에서 데이터 폴더를 설정해 주세요." |
| 탭2 미설정 상태로 탭3 진입 | "먼저 탭2에서 전처리 설정을 완료해 주세요." |
| 탭3 미설정 상태로 탭4 진입 | "먼저 탭3에서 모델 파라미터를 설정해 주세요." |
| 실험 없음 상태로 탭5/탭6 진입 | "아직 실행된 실험이 없습니다. 탭4에서 학습을 먼저 실행해 주세요." |

---

## 8. 실험 히스토리 관리 명세

### 8.1 저장 포맷
- 파일: `./experiments/history.json` (전체 실험 배열)
- 개별 실험 자산: `./models/{exp_id}/` 디렉토리

### 8.2 저장 항목
- `experiment_id`, `name`, `status`(completed / 중단), `created_at`
- `model`, `preprocessing`, `model_params`
- `metrics` (중단 시 `null`)
- `threshold`, `model_path`, `configs_path`

### 8.3 JSON 예시

```json
{
  "experiment_id": "exp_001",
  "name": "efficientad_20260508_1400",
  "status": "completed",
  "created_at": "2026-05-08 14:00:00",
  "model": "efficientad",
  "preprocessing": {
    "method": "homomorphic",
    "resize_mode": "padding",
    "image_size": 256
  },
  "model_params": {
    "model_size": "medium",
    "train_steps": 70000,
    "lr": 0.0001
  },
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.91,
    "f1_score": 0.92,
    "f2_score": 0.91,
    "auc": 0.97
  },
  "threshold": {
    "method": "percentile",
    "value": 95
  },
  "model_path": "./models/exp_001/",
  "configs_path": "./models/exp_001/configs.yaml"
}
```

### 8.4 중단 실험 처리
- `status: "중단"`, `metrics: null`, `model_path: null`
- 히스토리 테이블에 별도 시각 표시(예: 회색 처리)

### 8.5 비교 UI 명세
- 다중 선택 체크박스로 실험 선택
- 비교 메트릭 다중 선택: Accuracy / Precision / Recall / F1 / F2
- 막대 차트 또는 레이더 차트로 비교 시각화

---

## 9. configs.yaml 명세

### 9.1 원칙
- 전처리 + 모델 파라미터를 **단일 파일**에 통합, **섹션 분리** 운영
- 탭2·탭3 각각에서 저장/불러오기 가능, **동일 파일에 해당 섹션만 업데이트**
- 모델 저장 시 함께 직렬화되어 추론 애플리케이션이 동일 설정으로 재현 가능

### 9.2 전체 구조 예시

```yaml
experiment:
  name: "exp_001"
  created_at: "2026-05-08 14:00:00"

preprocessing:
  method: "homomorphic"      # none / homomorphic / he / clahe
  resize_mode: "padding"     # 고정값
  image_size: 256
  homomorphic:
    sigma: 10
    gamma_H: 1.5
    gamma_L: 0.5
    cutoff: 30
    normalize: true
  clahe:
    clipLimit: 2.0
  normalization: "imagenet"  # imagenet / custom
  custom_mean: [0.5, 0.5, 0.5]
  custom_std: [0.5, 0.5, 0.5]

model:
  type: "efficientad"        # efficientad / patchcore
  common:
    image_size: 256
    batch_size: 16
    random_seed: 42
  threshold:
    method: "percentile"     # percentile / absolute
    value: 95
  efficientad:
    model_size: "medium"
    train_steps: 70000
    optimizer: "adam"
    learning_rate: 0.0001
    weight_decay: 0.0001
    out_channels: 384
    padding: false
    ae_loss_weight: 0.5
    st_loss_weight: 0.5
    advanced:
      autoencoder_lr: 0.0001
      autoencoder_weight_decay: 0.00001
      lr_decay_epochs: 50000
      lr_decay_factor: 0.1
      scheduler: "StepLR"
      imagenet_penalty_weight: 1.0
      penalty_batch_size: 8
  patchcore:
    backbone: "wide_resnet50_2"
    pretrained_source: "torchvision"   # torchvision / local
    pretrained_path: null
    coreset_sampling_ratio: 0.1
    neighbourhood_kernel_size: 3
    advanced:
      max_train: 1000
      knn: 9
      top_k_ratio: 0.1
```

---

## 10. UI/UX 가이드라인

### 10.1 한국어 UI 원칙
- 모든 라벨·버튼·안내 메시지·에러 메시지를 한국어로 작성
- 기술 용어는 한국어 + 영문 병기 (예: "학습 단계 (train_steps)")

### 10.2 레이아웃 원칙
- 좌측 사이드바: 데이터셋 경로·디바이스 정보 등 상시 노출 정보
- 메인 영역: 6개 탭(`st.tabs`)
- 컴포넌트 간 충분한 여백, 카드형 강조 표시 활용

### 10.3 컴포넌트 스타일 기준

| 요소 | 스타일 |
|---|---|
| 메트릭 카드 | `st.metric` 활용, 색상 강조 |
| 폴더 트리 | 들여쓰기 텍스트 또는 `streamlit-tree-select` |
| 미리보기 이미지 | 좌·우 2분할 컬럼 (`st.columns(2)`) |
| 진행률 | `st.progress` |
| 실시간 차트 | `st.line_chart` 또는 Plotly |
| 고급 설정 | `st.expander("고급 설정")` |

### 10.4 비선택 모델 파라미터 처리
- 라디오 미선택 모델의 파라미터는 **DOM 자체 미렌더링** (단순 `disabled` 처리 금지)

### 10.5 표준 안내 메시지

| 상황 | 메시지 |
|---|---|
| 데이터 없음 | "먼저 탭1에서 데이터 폴더를 설정해 주세요." (7.3절 표 참조) |
| Grayscale 감지 | "Grayscale 이미지가 감지되었습니다. 모델 입력을 위해 RGB 3채널로 자동 변환됩니다." |
| 폴더 구조 오류 | "MVTec AD 형식의 폴더 구조가 아닙니다. (필수: train/good, test/, ground_truth/)" |
| 학습 중단 완료 | "학습이 중단되었습니다. 해당 실험은 '중단' 상태로 히스토리에 기록되었습니다." |
| 앙상블 모드 Threshold (MVP 제외) | *"앙상블 모드에서는 모델 간 Score 스케일이 달라 Percentile 방식을 권장합니다." — MVP 제외 항목* |

---

## 11. 배포 명세 (Deployment Spec)

### 11.1 Dockerfile 기본 구성 예시

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit 실행
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### 11.2 AWS EC2 권장 사양

| 항목 | 권장 값 |
|---|---|
| **인스턴스 타입** | g4dn.xlarge |
| **GPU** | NVIDIA Tesla T4 (16GB VRAM) |
| **vCPU / RAM** | 4 vCPU / 16GB RAM |
| **EBS 볼륨** | 100GB gp3 (데이터셋·모델 저장 고려) |
| **AMI** | Deep Learning AMI (Ubuntu 22.04) 또는 NVIDIA Container Toolkit 설치된 Ubuntu 22.04 |
| **보안 그룹** | 8501/tcp (Streamlit), 22/tcp (SSH) |

### 11.3 로컬 실행 명령어 예시

```bash
# 1. 이미지 빌드
docker build -t vision-inspection-dashboard:latest .

# 2. 컨테이너 실행 (GPU + 데이터 볼륨 마운트)
docker run -d \
  --name vision-dashboard \
  --gpus all \
  -p 8501:8501 \
  -v /path/to/dataset:/app/dataset \
  -v /path/to/models:/app/models \
  -v /path/to/experiments:/app/experiments \
  vision-inspection-dashboard:latest

# 3. 접속
# 브라우저에서 http://localhost:8501
```

### 11.4 EC2 배포 명령어 예시

```bash
# EC2 SSH 접속 후
git clone <repository_url>
cd vision-inspection-dashboard

docker build -t vision-inspection-dashboard:latest .

docker run -d \
  --name vision-dashboard \
  --gpus all \
  -p 8501:8501 \
  -v /home/ubuntu/dataset:/app/dataset \
  -v /home/ubuntu/models:/app/models \
  -v /home/ubuntu/experiments:/app/experiments \
  --restart unless-stopped \
  vision-inspection-dashboard:latest
```

---

## 12. 완료 조건 (Definition of Done)

### 12.1 탭별 완료 기준

#### 탭1 — 데이터 폴더 구조
- [ ] MVTec AD 폴더 구조 검증 동작
- [ ] 폴더별 이미지 수 정확 표시
- [ ] 대표 샘플 썸네일 렌더링
- [ ] Grayscale 자동 감지 및 RGB 변환 안내
- [ ] 잘못된 구조에 대한 경고 메시지 출력
- [ ] `.jpg`/`.png`/`.bmp` 외 파일 필터링

#### 탭2 — 전처리 파라미터 설정
- [ ] 라디오 선택에 따라 비선택 모델 파라미터 UI 완전 숨김
- [ ] Homomorphic / HE / CLAHE 적용 전·후 미리보기 정상 동작
- [ ] Resize + Padding(검정 0) 고정 적용
- [ ] ImageNet/커스텀 정규화 선택 가능
- [ ] configs.yaml 전처리 섹션 저장/불러오기 동작

#### 탭3 — 모델 파라미터 설정
- [ ] EfficientAD / PatchCore 라디오 선택 동작
- [ ] image_size 탭2 연동 자동 반영
- [ ] EfficientAD ae/st loss weight 합산 1.0 슬라이더 동작
- [ ] 고급 설정 expander 동작
- [ ] Threshold 방식 선택 및 정상/결함 비율 실시간 표시
- [ ] 디바이스 자동 감지 표시
- [ ] configs.yaml 모델 섹션 저장/불러오기 동작

#### 탭4 — 학습 시작 + 학습 로그
- [ ] 진행률 Progress Bar 실시간 갱신
- [ ] Loss 곡선 실시간 표시
- [ ] 학습 로그 텍스트 박스 스트리밍
- [ ] 학습 중지 시 데이터 폐기·"중단" 상태 기록
- [ ] 완료 시 알림 + 총 소요 시간 표시
- [ ] 실험명 자동 생성 또는 입력

#### 탭5 — 실험 히스토리 + 결과 + 모델 저장
- [ ] 실험 목록 테이블 렌더링 및 정렬
- [ ] 실험 선택 시 상세 결과(Confusion Matrix, ROC, Anomaly Score 분포) 표시
- [ ] 다중 실험 메트릭 비교 차트 동작
- [ ] state_dict + configs.yaml 저장 동작
- [ ] 저장 완료 시 경로·파일명·용량 출력
- [ ] 실험 삭제 동작

#### 탭6 — 이상 영역 시각화
- [ ] 탭5 실험 선택 연동 동작
- [ ] 테스트 이미지 목록 테이블 렌더링
- [ ] 결함 유형별 필터링 동작
- [ ] 3분할 시각화(원본/GT/Heatmap) 정상 표시
- [ ] Threshold 슬라이더 실시간 이진화 갱신
- [ ] 3분할 PNG 저장 동작

### 12.2 비기능 완료 기준
- [ ] Docker 이미지 빌드 및 GPU 컨테이너 정상 실행
- [ ] 학습 중 UI 블로킹 없음 확인
- [ ] g4dn.xlarge에서 EfficientAD-medium 70k steps 20분 이내 완료
- [ ] 동일 random_seed 재현성 확인
- [ ] 한국어 UI 일관성 검토 완료

---

## 13. 향후 확장 고려사항 (MVP 이후)

| 항목 | 설명 |
|---|---|
| **앙상블 (EfficientAD + PatchCore 가중 평균)** | UI 슬라이더로 비율 조절(기본 0.5:0.5), 학습은 EfficientAD → PatchCore 순차 실행 후 Score 결합. 앙상블 시 Percentile 자동 전환 안내 메시지 적용 |
| **GAN 기반 이미지 증강** | 적용 여부 토글, 추가 생성 이미지 수 입력, 생성 샘플 썸네일 그리드 |
| **전처리 필터 ↔ GAN 증강 적용 순서 설정** | 사용자가 파이프라인 순서를 직접 구성하는 옵션 |
| **공장 실사용 추론 애플리케이션 연동** | state_dict + configs.yaml을 S3로 업로드하여 추론 앱이 자동 폴링·로드 |
| **실시간 카메라 연동** | RTSP/USB 카메라 입력으로 실시간 추론 및 알람 |
| **재학습 파이프라인 자동화** | 신규 데이터 누적 시 트리거 기반 자동 재학습 (Airflow / SageMaker Pipelines 등) |
| **사용자 권한 관리** | 다중 사용자 환경에서 실험·모델 접근 권한 분리 |
| **클라우드 모델 레지스트리 통합** | MLflow / SageMaker Model Registry 등과 연동하여 모델 버전·메타데이터 중앙 관리 |

---

## 부록 A — MVP 제외 항목 요약

| # | 제외 항목 | 사유 | 향후 고려 |
|---|---|---|---|
| 1 | 앙상블 (EfficientAD + PatchCore 가중 평균) | 단일 모델 검증 우선, 스케일 정합성·Threshold 정책 추가 설계 필요 | ✅ |
| 2 | GAN 기반 이미지 증강 | 데이터셋 의존적 효과·학습 안정성 검증 필요 | ✅ |
| 3 | 전처리 필터와 GAN 증강 적용 순서 설정 | GAN 증강이 MVP 제외이므로 부속 기능도 자동 제외 | ✅ |
| 4 | 앙상블 모드 Percentile 자동 전환 안내 | 앙상블 자체가 MVP 제외이므로 함께 제외 | ✅ |

---

**문서 버전 이력**

| 버전 | 일자 | 작성자 | 비고 |
|---|---|---|---|
| v1.0 | 2026-05-08 | ML 엔지니어링팀 | MVP PRD 초안 |
