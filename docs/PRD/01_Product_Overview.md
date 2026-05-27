# 01. Product Overview

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **버전**: v1.0
> **작성일**: 2026-05-08
> **후속 문서**: [02_User_Personas_and_Use_Cases.md](./02_User_Personas_and_Use_Cases.md)

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Detailed Specification](#b-detailed-specification)
- [C. System & Data Design](#c-system--data-design)
- [D. API Contracts](#d-api-contracts)
- [E. AI/ML Details](#e-aiml-details)
- [F. Non-Functional Requirements](#f-non-functional-requirements)
- [G. Observability](#g-observability)
- [H. QA & Validation](#h-qa--validation)
- [I. Implementation Plan](#i-implementation-plan)

---

## A. Objective & Scope

### A.1 제품 정의

이 제품은 **제조산업 품질검사를 위한 딥러닝 기반 비전검사 최적 모델 탐색 대시보드**이다.

자동차 볼트 외관 결함 탐지를 목적으로, 데이터 분석가·ML 엔지니어가 코드 작성 없이 이상 탐지 모델(EfficientAD, PatchCore)을 학습·평가·비교하고 최적 모델을 자산화할 수 있도록 단일 Streamlit 웹 애플리케이션으로 제공한다.

### A.2 해결하는 문제

| # | 문제 | 현재 상태 | 이 제품이 해결하는 방식 |
|---|------|-----------|------------------------|
| P-01 | 육안 검사의 주관성 | 검사자별 판정 기준 상이, 피로도에 따라 정확도 변동 | 딥러닝 모델로 객관적·일관된 판정 기준 확보 |
| P-02 | 실험 반복 비용 | 모델·전처리 조합 변경 시마다 코드 수정·재실행 필요 | GUI 기반 파라미터 설정으로 코드 작성 없이 실험 반복 |
| P-03 | 실험 재현성 부족 | 실험 조건이 파일·메모·기억에 산재 | `random_seed` + `configs.yaml` 단일 파일로 조건 완전 재현 |
| P-04 | 모델 비교의 수작업 | 실험별 지표를 수동으로 스프레드시트에 집계 | 실험 히스토리 DB(history.json) + 비교 차트로 한 화면 비교 |
| P-05 | 모델 자산화 부재 | 학습된 모델을 추론 애플리케이션에 연계할 표준 형식 없음 | `state_dict + configs.yaml` 고정 포맷 저장으로 추론 앱 연계 준비 |

### A.3 제품 비전

> "데이터 분석가가 하루 안에 EfficientAD와 PatchCore를 모두 실험하고, 더 나은 모델을 골라 추론 파이프라인에 바로 연결할 수 있는 워크벤치"

### A.4 MVP 범위 (v1.0)

#### IN SCOPE

| 기능 영역 | 포함 항목 |
|-----------|-----------|
| 데이터 검증 | MVTec AD 폴더 구조 검증, OK/NG 폴더 형식 자동 감지 및 80/20 자동 분할, 이미지 수 카운트, 썸네일 렌더링, Grayscale 자동 감지 |
| 전처리 | None / Homomorphic Filter / HE / CLAHE 선택, Resize+Padding 고정, 정규화 선택 |
| 모델 설정 | EfficientAD (small/medium), PatchCore (WideResNet50/ResNet18/ResNet50) 파라미터 GUI |
| 학습 실행 | 학습 루프, Progress Bar, 실시간 Loss 곡선, 학습 중지, 일시정지/재시작, 체크포인트 저장/재시작, 실험 히스토리 저장 |
| 결과 비교 | Confusion Matrix, ROC Curve, Anomaly Score 분포, 다중 실험 비교 차트 |
| 시각화 | 3분할 Anomaly Map (원본/GT/Heatmap), Threshold 슬라이더, PNG 저장 |
| 모델 저장 | state_dict + configs.yaml 고정 포맷 |
| 배포 | Docker + NVIDIA Container Toolkit, AWS EC2 g4dn.xlarge |

#### OUT OF SCOPE (v1.0 제외, 향후 확장)

| 제외 항목 | 제외 사유 |
|-----------|-----------|
| 앙상블 (EfficientAD + PatchCore 가중 평균) | 스케일 정합성·Threshold 정책 추가 설계 필요 |
| GAN 기반 이미지 증강 | 데이터셋 의존적 효과·학습 안정성 검증 필요 |
| 전처리 필터 ↔ GAN 증강 적용 순서 설정 | GAN 증강 MVP 제외에 따른 자동 제외 |
| 다중 사용자 / 권한 관리 | 단일 사용자 워크스테이션 환경 (가정 A-01) |
| 실시간 카메라 연동 | MVP 범위 초과 |
| 클라우드 모델 레지스트리 | MLflow/SageMaker 연동은 추론 앱 단계 |

### A.5 성공 지표 (측정 기준 포함)

| 지표 | 목표값 | 측정 방법 | 측정 시점 |
|------|--------|-----------|-----------|
| 모델 정확도 | AUC ≥ 0.95 (MVTec AD Screw 기준) | `history.json → metrics.auc` | 각 실험 완료 후 |
| EfficientAD 학습 시간 | 70,000 steps ≤ 20분 (g4dn.xlarge) | `experiment.duration_seconds ≤ 1200` | 실험 완료 후 |
| PatchCore 학습 시간 | coreset 10% ≤ 10분 (g4dn.xlarge) | `experiment.duration_seconds ≤ 600` | 실험 완료 후 |
| 단일 실험 사이클 | 설정→학습→평가 ≤ 30분 | 탭1~탭5 E2E 소요 시간 | 통합 테스트 |
| UI 응답성 | 탭 전환 < 1초, 학습 중 UI 블로킹 없음 | 수동 테스트 | 통합 테스트 |
| 재현성 | 동일 seed+파라미터 → 동일 AUC (소수점 4자리) | 2회 실행 결과 비교 | QA 단계 |

---

## B. Detailed Specification

### B.1 핵심 사용자 플로우 (Golden Path)

아래는 새로운 실험을 처음부터 완료하는 표준 플로우이다. 각 단계는 탭 번호와 1:1 대응한다.

```
Step 1 [탭1]  데이터 폴더 경로 입력
              → MVTec AD 구조 검증 (또는 OK/NG 형식 자동 감지)
              → 폴더 트리, 이미지 수, 썸네일 확인
              → OK/NG 형식 감지 시 80/20 자동 분할 안내 배너 표시
              → session_state.dataset_path, dataset_meta 저장

Step 2 [탭2]  전처리 방식 선택 (None / Homomorphic / HE / CLAHE)
              → 파라미터 조정
              → 적용 전·후 미리보기 확인
              → image_size, 정규화 설정
              → 모델 선택 (EfficientAD / PatchCore)
              → 공통·전용 파라미터 설정
              → Threshold 방식·값 설정
              → 디바이스 확인 (CUDA / CPU)
              → session_state.preprocessing_config, model_config, device_info 저장

Step 3 [탭3]  실험명 입력 (또는 자동 생성)
              → [학습 시작] 클릭
              → Progress Bar, Loss 곡선, 로그 실시간 확인
              → 완료 알림 + 소요 시간 확인

Step 4 [탭4]  실험 목록에서 완료된 실험 선택
              → Confusion Matrix, ROC Curve, Anomaly Score 분포 확인
              → (선택) 다른 실험과 지표 비교
              → (선택) 모델 저장

Step 5 [탭5]  선택된 실험의 테스트 이미지 목록 확인
              → 이미지 선택
              → 원본 / GT 마스크 / Heatmap 3분할 시각화 확인
              → Threshold 슬라이더로 이진화 조정
              → PNG 저장
```

### B.2 탭별 기능 요약

| 탭 | 탭명 | 핵심 입력 | 핵심 출력 | session_state Write |
|----|------|-----------|-----------|---------------------|
| 탭1 | 데이터 폴더 구조 | 로컬 경로 텍스트 | 폴더 트리, 이미지 수, 썸네일 | `dataset_path`, `dataset_meta` |
| 탭2 | 전처리 및 모델 설정 | 전처리 방식, 파라미터, 모델 종류, 하이퍼파라미터 | 전후 미리보기, 설정 요약, 디바이스 정보 | `preprocessing_config`, `model_config`, `device_info` |
| 탭3 | 학습 시작 + 학습 로그 | 실험명, 학습 시작 버튼 | Progress Bar, Loss 곡선, 로그 | `experiments[exp_id]`, `current_run_status` |
| 탭4 | 실험 히스토리 + 결과 + 저장 | 실험 선택, 저장 경로 | 지표 카드, 차트, 비교 시각화 | `selected_experiment_id` |
| 탭5 | 이상 영역 시각화 | 이미지 선택, Threshold 슬라이더 | 3분할 시각화, PNG | `anomaly_map_threshold` |

### B.3 사이드바 상시 노출 정보

```
┌─────────────────────────────┐
│ ■ 데이터셋                   │
│   경로: /app/dataset/screw  │
│   학습: 320장 / 테스트: 119장│
│                             │
│ ■ 디바이스                   │
│   CUDA (Tesla T4)           │
│   VRAM: 16.0 GB             │
│                             │
│ ■ 현재 설정                  │
│   전처리: Homomorphic        │
│   모델: EfficientAD-medium  │
└─────────────────────────────┘
```

사이드바는 `session_state.dataset_meta`, `device_info`, `preprocessing_config`, `model_config`가 None이 아닌 경우에만 해당 섹션을 렌더링한다. None이면 해당 섹션 미표시.

### B.4 탭 진입 차단 조건 (Guard)

아래 조건이 충족되지 않으면 해당 탭의 핵심 기능을 렌더링하지 않고 `utils/messages.py`의 표준 메시지를 `st.warning()`으로 표시한다.

| 탭 | 차단 조건 | 표시 메시지 키 |
|----|-----------|---------------|
| 탭2 | `session_state.dataset_path is None` | `MSG["NO_DATASET"]` |
| 탭3 | `session_state.model_config is None` | `MSG["NO_MODEL_CONFIG"]` |
| 탭4 | `len(session_state.experiments) == 0` | `MSG["NO_EXPERIMENTS"]` |
| 탭5 | `session_state.selected_experiment_id is None` | `MSG["NO_SELECTED_EXP"]` |

차단 탭에서도 탭 자체는 클릭 가능해야 한다. `st.tabs()`는 항상 **5개 탭**을 렌더링하되, 차단 조건에 해당하는 탭의 본문에서 guard 처리한다.

### B.5 Edge Cases

| # | 상황 | 처리 방식 |
|---|------|-----------|
| EC-01 | 입력 경로가 존재하지 않음 | `st.error(ERR_DATASET_NOT_FOUND)` + dataset_path = None 유지 |
| EC-02 | `train/good/`에 이미지가 0개 | `st.error()` + 탭4 진입 차단 |
| EC-03 | 지원 포맷 외 파일 포함 | `st.warning()` 안내 후 지원 포맷만 카운트·사용, 학습 차단하지 않음 |
| EC-04 | Grayscale 이미지 | `st.info(MSG["GRAYSCALE_DETECT"])` + `image_utils.py`에서 자동 RGB 변환 |
| EC-05 | `ground_truth/` 디렉토리 없음 | 해당 클래스 GT를 빈 마스크(전체 0)로 처리, 탭6 Heatmap은 정상 렌더링 |
| EC-06 | CUDA 미사용 가능 환경 | CPU fallback, 사이드바에 "CPU" 표시, 학습 진행은 허용 |
| EC-07 | 학습 중 [학습 중지] 클릭 | 즉시 중단 신호 전달, 데이터 폐기, status="중단"으로 history.json 기록 |
| EC-12 | 학습 중 [⏸ 일시정지] 클릭 | 현재 step 완료 후 체크포인트 저장, `current_run_status = "paused"`, history.json 기록 안 함 |
| EC-13 | 일시정지 중 [⏹ 학습 중지] 클릭 | `pause_event.clear()` 후 `stop_event.set()`, 학습 스레드가 stop_event 감지하여 종료, status="중단" 기록 |
| EC-14 | 체크포인트 파일 손상/호환 불가 | `st.error()` 표시, 재시작 불가 안내, 체크포인트 삭제 버튼 제공 |
| EC-15 | 체크포인트 재시작 시 experiment_id 충돌 | 새 ID 자동 생성 후 재시작, 사용자에게 `st.info()`로 안내 |
| EC-08 | configs.yaml 없는 상태에서 불러오기 | 빈 dict 반환, 현재 UI 상태 유지 (오류 발생 금지) |
| EC-09 | 동일 experiment_id 중복 | uuid4().hex[:4]의 충돌 확률 = 1/65536. 충돌 시 재생성 1회 (최대 2회 시도) |
| EC-10 | 모델 저장 중 디스크 공간 부족 | `shutil.disk_usage()`로 사전 확인, 여유 < 500MB면 `st.warning()` + 저장 중단 |
| EC-11 | 탭6에서 GT 마스크 이미지 크기 불일치 | Anomaly Map 크기 기준으로 GT 마스크를 `cv2.resize()` 후 표시 |
| EC-12 | history.json 파싱 오류 | `st.error()` + 빈 실험 목록 표시, 파일 덮어쓰기 금지 |
| EC-16 | OK/NG 형식에서 OK 또는 NG 디렉토리 중 하나만 존재 | `st.error(ERR_INVALID_FOLDER_STRUCTURE)` + dataset_path = None 유지 |
| EC-17 | OK/NG 형식에서 OK 이미지 수 < 5개 | `st.warning()` 표시 (분할 후 학습 데이터 부족 경고), 학습 진행은 허용 |

### B.6 실패 시나리오

| # | 시나리오 | 감지 방법 | 복구 방법 |
|---|----------|-----------|-----------|
| F-01 | 학습 중 GPU OOM | `torch.cuda.OutOfMemoryError` catch | `st.error()` + current_run_status="idle" + batch_size 축소 권장 메시지 |
| F-02 | Anomalib 모델 초기화 실패 | `Exception` catch in `model_factory.py` | `st.error(ERR_MODEL_INIT_FAILED)` + 로그 파일 기록 |
| F-03 | history.json 쓰기 실패 (권한·공간) | `OSError` catch | `st.error(ERR_MODEL_SAVE_FAILED)` + session_state에는 유지 (메모리 손실 방지) |
| F-04 | 학습 완료 후 메트릭 계산 실패 | `Exception` catch in `metrics.py` | status="중단" 처리, metrics=null, 에러 로그 기록 |
| F-05 | 탭6 모델 재로드 실패 | `FileNotFoundError` catch | `st.error()` + 해당 실험 model_path 확인 안내 |

---

## C. System & Data Design

### C.1 데이터 모델 참조

이 문서는 데이터 모델을 정의하지 않는다. 모든 스키마는 [00_Global_Context_Document.md 1절](./00_Global_Context_Document.md#1-core-data-model)에서 확정되었으며, 이후 파일에서 재사용한다.

| 스키마 | 정의 위치 |
|--------|-----------|
| experiment 레코드 | 00_Global_Context 1.1절 |
| metrics 오브젝트 | 00_Global_Context 1.2절 |
| preprocessing_config | 00_Global_Context 1.6절 |
| model_config | 00_Global_Context 1.7절 |
| dataset_meta | 00_Global_Context 1.5절 |
| configs.yaml 구조 | 00_Global_Context 1.9절 |

### C.2 시스템 구성 참조

시스템 아키텍처(컴포넌트 다이어그램, 디렉토리 구조, 탭별 데이터 흐름)는 [00_Global_Context_Document.md 5절](./00_Global_Context_Document.md#5-system-architecture)에 정의되어 있으며, [04_System_Architecture.md](./04_System_Architecture.md)에서 상세 확장된다.

### C.3 session_state 흐름 요약

```
탭1 Write: dataset_path, dataset_meta
  ↓
탭2 Write: preprocessing_config, model_config, device_info
  ↓
탭3 Write: experiments[exp_id], current_run_status
  ↓
탭4 Write: selected_experiment_id
  ↓
탭5 Write: anomaly_map_threshold
```

각 키의 타입·제약조건은 [00_Global_Context_Document.md 3.1절](./00_Global_Context_Document.md#31-session_state-초기화-명세)을 따른다.

---

## D. API Contracts

```
N/A - 이 시스템은 REST API 서버를 포함하지 않는다.
      Streamlit 단일 프로세스 애플리케이션이며, 외부 클라이언트 접근이 없다.
      내부 인터페이스(session_state 계약, 파일 I/O 계약)는
      00_Global_Context_Document.md 3절에 정의되어 있다.
      상세 인터페이스 명세는 06_API_Specification.md에서 다룬다.
```

---

## E. AI/ML Details

### E.1 모델 선택 근거

이 제품은 두 가지 이상 탐지 모델을 제공한다. 아래는 각 모델의 선택 근거와 사용 적합 시나리오다.

#### EfficientAD

| 항목 | 내용 |
|------|------|
| **알고리즘 분류** | Knowledge Distillation 기반 이상 탐지 |
| **학습 방식** | Student-Teacher + AutoEncoder 구조, 정상 이미지만으로 학습 |
| **선택 근거** | 소형 모델(small) ~100MB, 중형(medium) ~200MB로 T4 16GB VRAM 내 학습 가능. 70,000 steps 기준 20분 이내 목표 달성 가능 |
| **강점** | 학습 속도 빠름, 다양한 결함 유형에 강건 |
| **약점** | train_steps 수에 민감, 수만 step 이상 학습 필요 |
| **권장 사용 시나리오** | 다양한 결함 유형, 텍스처 기반 이상 탐지 |

#### PatchCore

| 항목 | 내용 |
|------|------|
| **알고리즘 분류** | Memory Bank (Coreset Sampling) 기반 이상 탐지 |
| **학습 방식** | Pretrained backbone으로 특징 추출 후 coreset 메모리 뱅크 구성, 추론 시 k-NN 거리로 점수 계산 |
| **선택 근거** | 학습 epoch 불필요(1 pass), coreset 10% 기준 10분 이내 완료. WideResNet50 backbone 기준 ImageNet pretrained 가중치 즉시 활용 |
| **강점** | 학습 데이터 적어도 높은 성능, 설정 단순 |
| **약점** | 메모리 뱅크 크기에 따라 추론 시간 증가, coreset_sampling_ratio 낮으면 recall 감소 |
| **권장 사용 시나리오** | 학습 데이터 수 적음, 빠른 프로토타이핑 필요 |

### E.2 Anomalib 연동 방침

- Anomalib **≥ 1.0.0 (v1 API)** 기준으로 구현한다 (가정 A-03).
- `model_factory.py`에서 EfficientAD Engine, PatchCore Engine을 각각 래퍼 함수로 캡슐화한다.
- Anomalib 내부 DataModule 대신 커스텀 DataLoader를 사용하여 탭2 전처리 파이프라인과 연동한다.
- 상세 구현 명세는 [08_AI_ML_Integration.md](./08_AI_ML_Integration.md)에서 다룬다.

### E.3 전처리 파이프라인 위치

전처리(Homomorphic/HE/CLAHE)는 Anomalib DataModule이 아닌 **`utils/image_utils.py`에서 구현**하여 탭2 미리보기와 탭3 학습 루프가 동일 코드를 공유한다.

```
이미지 로드 (PIL)
  → 채널 변환: Grayscale → RGB (if channels == 1)
  → 전처리 필터 적용 (None / Homomorphic / HE / CLAHE)
  → Resize + Padding (image_size × image_size, 검정 0)
  → 정규화 (ImageNet 또는 커스텀 mean/std)
  → torch.Tensor 변환
```

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체를 상속한다.

이 문서에서 추가로 명시하는 항목:

| 항목 | 요구사항 |
|------|----------|
| **브라우저 호환성** | Chrome ≥ 110, Firefox ≥ 110, Edge ≥ 110. Safari는 미지원 (Streamlit WebSocket 안정성). |
| **화면 해상도** | 최소 1280×800. 반응형 레이아웃은 Streamlit 기본 제공 범위 내. |
| **Streamlit 재실행 안전성** | 모든 탭 함수는 Streamlit rerun 시 멱등(idempotent)하게 동작한다. 부작용(파일 쓰기, 모델 초기화)은 버튼 클릭 이벤트 내부에만 위치한다. |
| **학습 스레드 안전성** | 학습 백그라운드 스레드와 메인 스레드 간 통신은 `queue.Queue`만 사용한다. `st.session_state` 직접 쓰기는 메인 스레드에서만 허용한다. |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards)을 전체 상속한다.

이 문서(Product Overview)에서 추가로 명시하는 제품 수준 관측 항목:

| 관측 항목 | 목적 | 구현 위치 |
|-----------|------|-----------|
| 실험 완료 시 AUC 기록 | 성공 지표(AUC ≥ 0.95) 달성 여부 추적 | `history.json → metrics.auc` |
| 학습 소요 시간 기록 | 성능 목표(EfficientAD ≤ 1200s, PatchCore ≤ 600s) 달성 여부 추적 | `history.json → duration_seconds` |
| 모델 타입별 실험 수 | EfficientAD vs PatchCore 사용 빈도 | `history.json` 집계 |

---

## H. QA & Validation

### H.1 제품 수준 인수 기준 (Acceptance Criteria)

아래 기준을 모두 통과하면 MVP v1.0을 완료로 판정한다.

| # | 기준 | 검증 방법 |
|---|------|-----------|
| AC-01 | 탭1~탭5 Golden Path 플로우(B.1절)를 오류 없이 완주 | E2E 수동 테스트 |
| AC-02 | EfficientAD-medium 학습 완료 시 `duration_seconds ≤ 1200` | g4dn.xlarge 실측 |
| AC-03 | PatchCore (coreset 10%) 학습 완료 시 `duration_seconds ≤ 600` | g4dn.xlarge 실측 |
| AC-04 | MVTec AD Screw 기준 AUC ≥ 0.95 달성 실험 존재 | history.json 확인 |
| AC-05 | 동일 seed+파라미터로 2회 실행 시 AUC 소수점 4자리 일치 | 재현성 테스트 |
| AC-06 | Docker 이미지 빌드 성공 + GPU 컨테이너 정상 실행 | `docker run --gpus all` |
| AC-07 | 탭 전환 응답 < 1초 (학습 중 포함) | 수동 측정 |
| AC-08 | 잘못된 폴더 경로 입력 시 탭3 진입 차단 + 경고 메시지 표시 | 수동 테스트 |
| AC-09 | [학습 중지] 클릭 후 history.json에 status="중단" 기록 확인 | 파일 내용 검증 |
| AC-10 | 모델 저장 후 `./models/{exp_id}/` 디렉토리에 `.pth`와 `configs.yaml` 존재 확인 | 파일 존재 검증 |
| AC-11 | [⏸ 일시정지] 클릭 후 `./models/checkpoints/{exp_id}_step{N}.ckpt` 파일 존재 확인 | 파일 존재 검증 |
| AC-12 | 체크포인트에서 재시작 후 중단 지점 이후 step부터 학습 로그 출력 확인 | 수동 테스트 |

### H.2 Given-When-Then 시나리오 (제품 수준)

#### TC-01: 정상 실험 완료 플로우

```
Given:  MVTec AD Screw 데이터셋이 /app/dataset/screw 에 올바른 구조로 존재한다
        CUDA 디바이스가 사용 가능하다
When:   탭1에서 /app/dataset/screw 경로 입력 → 검증 통과
        탭2에서 전처리 = CLAHE (clipLimit=2.0), image_size=256 및 EfficientAD-medium, train_steps=70000, seed=42 설정
        탭3에서 [학습 시작] 클릭 → 학습 완료 대기
        탭4에서 완료된 실험 선택
        탭5에서 테스트 이미지 선택
Then:   history.json에 status="completed" 레코드 존재
        metrics.auc >= 0.95
        duration_seconds <= 1200
        ./models/{exp_id}/model_state_dict.pth 파일 존재
        3분할 시각화 정상 렌더링
```

#### TC-02: 학습 중단 플로우

```
Given:  탭3에서 학습이 진행 중이다 (current_run_status == "running")
When:   사용자가 [학습 중지] 버튼을 클릭한다
Then:   current_run_status == "idle"로 복귀
        history.json에 status="중단", metrics=null 레코드 추가
        ./models/{exp_id}/ 디렉토리가 존재하지 않음
        st.warning(MSG["TRAIN_STOPPED"]) 표시
```

#### TC-03: 잘못된 경로 입력

```
Given:  존재하지 않는 경로 "/nonexistent/path" 를 탭1에 입력한다
When:   검증 로직이 실행된다
Then:   st.error(ERR_DATASET_NOT_FOUND) 표시
        session_state.dataset_path == None 유지
        탭2 진입 시 MSG["NO_DATASET"] 표시
        탭3 [학습 시작] 버튼 렌더링 안됨 (guard 적용)
```

---

## I. Implementation Plan

> 전체 구현 계획(WBS)은 [14_Deployment_and_Release_Plan.md](./14_Deployment_and_Release_Plan.md)에서 다룬다.
> 이 절에서는 Product Overview 관점의 의존성과 병렬화 전략만 기술한다.

### I.1 파일 생성 의존성 체인

```
00_Global_Context (완료)
  ↓
01_Product_Overview (이 문서)
  ↓
02_User_Personas_and_Use_Cases   ← 01 필요
  ↓
03_Functional_Requirements       ← 01, 02 필요
  ↓
04_System_Architecture           ← 03 필요 (병목)
  ↓
05_Data_Model  ──────────────────┐
06_API_Specification             │  ← 04 필요
07_Backend_Service_Design  ──────┤
08_AI_ML_Integration             │
09_Infrastructure_and_Cloud  ────┘
  ↓
10_Security, 11_NFR, 12_Observability  (병렬 가능)
  ↓
13_QA_and_Testing_Strategy
14_Deployment_and_Release_Plan
```

### I.2 구현 작업 분류 (2일 MVP 기준)

| Day | 시간 블록 | 담당 | 작업 |
|-----|-----------|------|------|
| Day 1 오전 1h | 09:00~10:00 | A | 프로젝트 구조 초기 세팅, requirements.txt |
| Day 1 오전 1h | 09:00~10:00 | B | session_state_init.py 구현 |
| Day 1 오전 1h | 09:00~10:00 | C | config_manager.py, configs_template.yaml |
| Day 1 오전 2h | 10:00~12:00 | A | 탭1 구현 (폴더 검증, 트리, 썸네일) |
| Day 1 오전 2h | 10:00~12:00 | B | 통합 탭2 EfficientAD 모델 설정 파트 (tab2_config.py) |
| Day 1 오전 2h | 10:00~12:00 | C | 통합 탭2 PatchCore 모델 설정 파트 (tab2_config.py) |
| Day 1 오후 | 13:00~18:00 | A | 탭2 구현 (전처리·모델 설정 통합, 미리보기) (tab2_config.py) |
| Day 1 오후 | 13:00~18:00 | B | 탭3 EfficientAD 학습 루프 |
| Day 1 오후 | 13:00~18:00 | C | 탭3 PatchCore 학습 루프 |
| Day 2 오전 | 09:00~13:00 | A | 탭5 구현 |
| Day 2 오전 | 09:00~13:00 | B | 탭4 전반 (히스토리, 상세 결과) |
| Day 2 오전 | 09:00~13:00 | C | 탭4 후반 (비교 차트, 저장) + Docker |
| Day 2 오후 | 14:00~18:00 | A+B+C | 통합 테스트, 버그 수정 |

> **역할 상세**: [역할분담표_비전검사_대시보드.md](./역할분담표_비전검사_대시보드.md) 참조

---

*다음 문서*: [02_User_Personas_and_Use_Cases.md](./02_User_Personas_and_Use_Cases.md)
