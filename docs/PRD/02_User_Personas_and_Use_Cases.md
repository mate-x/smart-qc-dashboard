# 02. User Personas and Use Cases

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [01_Product_Overview.md](./01_Product_Overview.md)
> **버전**: v2.0
> **작성일**: 2026-05-08
> **최종수정**: 2026-06-11
> **후속 문서**: [03_Functional_Requirements.md](./03_Functional_Requirements.md)

---

## 버전 이력

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-08 | 초기 작성 — Streamlit 탭 기반, 모델 탐색 페르소나(P-01·P-02) |
| v1.1 | 2026-05-26 | 비전검사 페르소나(P-03·P-04) 및 UC-13~18 추가 |
| v2.0 | 2026-06-11 | "탭N" 언급 → Explorer/Vision React 화면 이름으로 교체. `session_state` → Zustand store + FastAPI 기준으로 교체. Streamlit 내부 구현 상세는 각 UC 하단 v1.x 참고로 이동. UC-11(배포) → 3개 레포 기준으로 재작성. |

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

### A.1 이 문서의 목적

이 문서는 시스템이 누구를 위해 무엇을 해야 하는지를 구체적인 사용자 프로파일과 행동 시나리오로 정의한다. 이후 03_Functional_Requirements 이하 모든 파일에서 기능 결정의 근거로 참조한다.

### A.2 페르소나 범위

이 시스템은 **단일 사용자 환경**에서 동작한다 (가정 A-01). 동시 접속 다중 사용자 시나리오는 MVP 범위 밖이다.

| 페르소나 ID | 이름 | 역할 | 대상 앱 | 우선순위 |
|------------|------|------|---------|---------|
| P-01 | ML 엔지니어 / 데이터 분석가 | 실험 설계·실행·분석 담당 | Explorer (React) | **Primary** |
| P-02 | MLOps 엔지니어 | 배포·운영 환경 구성 담당 | Explorer + Vision + Dashboard | Secondary |
| P-03 | 현장 검사 작업자 | 부품 검사 수행, 불량 조치 | Vision (React) | **Primary** |
| P-04 | 품질 관리자 | 불량률 모니터링, 이력 분석 | Vision (React) | Secondary |

> P-01이 Explorer의 기능 결정 주 기준이다.
> P-03이 Vision의 기능 결정 주 기준이다.

---

## B. Detailed Specification

### B.1 Persona P-01: ML 엔지니어 / 데이터 분석가

#### 프로파일

| 속성 | 내용 |
|------|------|
| **직책** | 제조 AI팀 ML 엔지니어 또는 데이터 분석가 |
| **경력** | ML/데이터 분석 3~7년, 제조 도메인 이해 보유 |
| **Python 수준** | 중급 이상 (pandas, numpy, sklearn 능숙) |
| **PyTorch 수준** | 기초 이해 (모델 학습 흐름 이해, 직접 코드 작성은 선호하지 않음) |
| **이상 탐지 이해** | EfficientAD·PatchCore 개념 이해, Anomalib 사용 경험 있거나 없음 둘 다 해당 |
| **Docker 수준** | 기초 사용 가능 (`docker run` 수준, Dockerfile 작성 불필요) |
| **사용 환경** | 로컬 GPU 머신 또는 AWS EC2 g4dn.xlarge SSH 접속 후 브라우저 포워딩 |
| **주요 목표** | 다양한 전처리·모델·하이퍼파라미터 조합을 하루 안에 실험하고 최적 조합 탐색 |
| **보조 목표** | 최적 모델을 추론 애플리케이션 팀에 넘길 수 있는 형태로 저장 |

#### 페인 포인트 (현재 상태)

| # | 페인 포인트 | 이 제품의 해결 방식 |
|---|------------|-------------------|
| PP-01 | 모델·전처리 조합 변경 시마다 Python 스크립트를 수정하고 재실행해야 함 | Explorer Config 화면 GUI로 코드 작성 없이 파라미터 변경 |
| PP-02 | 실험 결과를 수동으로 Excel에 기록하다가 어떤 조합이 최선이었는지 잊어버림 | history.json + Experiments 화면 비교 차트로 자동 기록·시각화 |
| PP-03 | 학습 시작 후 완료까지 터미널만 보며 기다려야 함, 중간 진행 상황 파악 불가 | Training 화면 Progress Bar + WebSocket 실시간 Loss 곡선 |
| PP-04 | 결함 위치가 어디인지 모델이 어디를 보는지 확인할 방법이 없음 | AnomalyMap 화면 Triplet 시각화(원본/GT/Heatmap) + TP/FP/TN/FN 분류 |
| PP-05 | 전처리 필터를 적용하면 이미지가 어떻게 바뀌는지 학습 전에 확인할 수 없음 | Config 화면 전처리 전·후 실시간 미리보기 |
| PP-06 | 학습한 모델을 추론 파이프라인에 연결하려면 별도 코드 작성 필요 | `state_dict + configs.yaml` 고정 포맷 저장 → Vision 앱 직접 연계 |

#### 기술 수준별 사용 행동 예측

| 기술 수준 | 예상 행동 패턴 |
|-----------|--------------|
| **입문 (PyTorch 기초)** | 기본값 그대로 실험 1회 → 결과 확인 → 파라미터 1~2개만 변경하여 재실험 |
| **중급 (Anomalib 경험)** | 전처리·모델 파라미터 적극 조정, 고급 설정 패널 활용, Experiments 화면 다중 실험 비교 적극 사용 |
| **고급 (이상 탐지 전문)** | ae/st weight, coreset_sampling_ratio 등 모델 핵심 파라미터 조정, AUC/F2 기준으로 실험 필터링, set_id 기반 배치 비교 |

---

### B.2 Persona P-02: MLOps 엔지니어

#### 프로파일

| 속성 | 내용 |
|------|------|
| **직책** | MLOps 엔지니어 또는 DevOps 엔지니어 |
| **경력** | 인프라·배포 5년 이상, ML 워크플로우 운영 경험 |
| **Python 수준** | 중급 (운영 스크립트 작성 가능) |
| **Docker/K8s 수준** | 고급 (Dockerfile 작성, GPU 컨테이너 운영 능숙) |
| **AWS 수준** | 중급 이상 (EC2, EBS, 보안 그룹 설정 가능) |
| **Node.js 수준** | 기초 (npm run dev / npm run build 수준) |
| **사용 환경** | AWS EC2 SSH 접속, Docker CLI, Node.js 런타임 |
| **주요 목표** | Dashboard FastAPI + Explorer/Vision React 앱을 EC2에 배포·운영하고, 저장된 모델을 Vision 앱에 연결 |
| **보조 목표** | GPU 컨테이너 안정성 확인, 포트 설정(8000/5173) 및 볼륨 마운트 정상 동작 검증 |

#### 페인 포인트

| # | 페인 포인트 | 이 제품의 해결 방식 |
|---|------------|-------------------|
| PP-07 | 로컬 환경과 서버 환경의 패키지 버전 불일치로 재현성 실패 | Docker 이미지 단일화 (`nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04`) |
| PP-08 | 저장된 모델 파일에 학습 조건 정보가 없어서 Vision 앱 연결 시 설정 재확인 필요 | `configs.yaml`이 모델과 동일 경로에 함께 저장됨 |

---

### B.3 Use Cases

#### 전체 Use Case 목록

| UC ID | 제목 | 주 페르소나 | 관련 화면 | 우선순위 |
|-------|------|------------|----------|---------|
| UC-01 | 새 데이터셋 등록 및 검증 | P-01 | Dataset 화면 (Explorer) | Must |
| UC-02 | 전처리 방식 탐색 및 설정 | P-01 | Config 화면 (Explorer) | Must |
| UC-03 | EfficientAD 실험 설정 및 실행 | P-01 | Config 화면, Training 화면 (Explorer) | Must |
| UC-04 | PatchCore 실험 설정 및 실행 | P-01 | Config 화면, Training 화면 (Explorer) | Must |
| UC-05 | 실험 결과 분석 | P-01 | Experiments 화면 (Explorer) | Must |
| UC-06 | 두 실험 비교 및 최적 모델 선택 | P-01 | Experiments 화면 (Explorer) | Must |
| UC-07 | AnomalyMap으로 모델 행동 검증 | P-01 | AnomalyMap 화면 (Explorer) | Must |
| UC-08 | 최적 모델 저장 | P-01 | Experiments 화면 (Explorer) | Must |
| UC-09 | 설정 재사용 (서버 설정 불러오기) | P-01 | Config 화면 (Explorer) | Should |
| UC-10 | 실패 실험 정리 (히스토리 삭제) | P-01 | Experiments 화면 (Explorer) | Must |
| UC-11 | EC2 환경 배포 (3개 레포) | P-02 | — | Must |
| UC-12 | 결함 유형별 탐지 성능 필터링 확인 | P-01 | AnomalyMap 화면 (Explorer) | Should |
| UC-13 | 검사 모델 선택 및 검사 시작 | P-03 | Model Settings 화면, Realtime Inspection 화면 (Vision) | Must |
| UC-14 | 수동 검사 실행 및 판정 확인 | P-03 | Realtime Inspection 화면 (Vision) | Must |
| UC-15 | 자동 검사 실행 및 중지 | P-03 | Realtime Inspection 화면 (Vision) | Must |
| UC-16 | 불량 감지 팝업 확인 및 검사 재개 | P-03 | Realtime Inspection 화면 (Vision) | Must |
| UC-17 | 검사 이력 조회 및 CSV 내보내기 | P-04 | History 화면 (Vision) | Must |
| UC-18 | 모델 교체 후 새 검사 세션 시작 | P-04 | Model Settings 화면, Realtime Inspection 화면 (Vision) | Should |

---

#### UC-01: 새 데이터셋 등록 및 검증

```
액터:      P-01
전제조건:  MVTec AD 또는 OK/NG 형식 데이터셋이 로컬 경로에 존재한다
성공 종료: datasetStore.datasetPath 에 검증된 경로 저장,
           datasetStore.datasetMeta 구성 완료 (POST /api/dataset/validate 응답)
실패 종료: datasetPath = null 유지, 에러 메시지 표시

주 플로우:
  1. 사용자가 Explorer Dataset 화면의 텍스트 입력 필드에 데이터셋 루트 경로를 입력한다
  2. 사용자가 [경로 확인] 버튼을 클릭하면 POST /api/dataset/validate 호출
     2a. 경로 존재 확인 → 실패 시 ERR_DATASET_NOT_FOUND
     2b. MVTec AD 구조 우선 감지:
         - train/good/ 하위 이미지 ≥ 1개 확인
         - test/ 하위 디렉토리 ≥ 1개 확인
     2c. MVTec AD 구조 실패 시 → OK/NG 형식 감지:
         - OK 별칭 디렉토리 탐지 (ok/good/normal/pass/neg 등)
         - NG 별칭 디렉토리 탐지 (ng/bad/defect/fail/abnormal/anomaly 등)
     2d. 두 형식 모두 실패 시 → 400 ERR_INVALID_FOLDER_STRUCTURE
  3. 서버가 DatasetValidateResponse 반환 (05_Data_Model.md 2.4절 스키마)
     [MVTec AD]
     3a. 폴더별 이미지 수 카운트 (지원 포맷만)
     3b. 채널 수 감지 (첫 번째 이미지 기준)
     3c. 결함 클래스 목록 추출 (test/ 하위 디렉토리명)
     [OK/NG]
     3d. OK/NG 이미지 수 카운트
     3e. 채널 수 감지
     3f. train_ratio=0.8 기준 분할 비율 기록 (실제 분할은 학습 시 수행)
  4. datasetStore.datasetPath, datasetMeta 업데이트
  5. 화면에 폴더 트리 문자열(folder_tree), 클래스별 이미지 수 테이블 렌더링
  6. 각 클래스 대표 이미지 썸네일 렌더링
     (MVTec AD: 결함 클래스별 3열 / OK/NG: OK·NG 2열)
  7. channels == 1 감지 시 Grayscale 안내 메시지 표시
  8. OK/NG 형식 감지 시 80/20 분할 안내 배너 표시

대안 플로우:
  2a-1. 지원 포맷 외 파일 존재 (has_invalid_files = true) → 경고 표시 후 주 플로우 계속
  2b-1. ground_truth/ 없음 (MVTec AD) → 빈 마스크로 처리하며 계속 진행
  2c-1. OK/NG에서 OK 또는 NG 디렉토리 중 하나만 존재 → 에러 + datasetPath = null
  경로 변경 감지:
  - 입력 경로가 이전 datasetPath와 다르면
    configStore.preprocessingConfig, modelConfig, deviceInfo 를 null로 초기화
    (Config·Training 화면 재설정 필요)
```

---

#### UC-02: 전처리 방식 탐색 및 설정

```
액터:      P-01
전제조건:  UC-01 완료, datasetStore.datasetPath != null
성공 종료: configStore.preprocessingConfig 저장 완료 (POST /api/config)
실패 종료: preprocessingConfig = null 유지

주 플로우:
  1. 사용자가 Explorer Config 화면에 진입 (guard 통과)
  2. 사용자가 전처리 방식 선택 (None / Homomorphic / HE / CLAHE)
  3. 선택된 방식의 파라미터 UI만 렌더링 (비선택 UI 미렌더링)
  4. 사용자가 파라미터 슬라이더 조정 → 즉시 미리보기 갱신
     4a. POST /api/config/preview → PreviewImageResponse 수신
         { original_b64, processed_b64, warning }
     4b. 좌(원본) / 우(처리 후) 2컬럼 미리보기 렌더링
  5. 사용자가 배경 제거 방법 선택 (none / sam2) 🆕
  6. 사용자가 image_size 입력 (32의 배수, 32~1024)
  7. 사용자가 정규화 방식 확인 (imagenet 고정)
     [확인 필요: v1.x에서 'custom' normalization 허용했으나 TypeScript 구현은 'imagenet' 전용]
  8. 사용자가 [설정 저장] 클릭 → POST /api/config
  9. configStore.preprocessingConfig, deviceInfo 업데이트

대안 플로우:
  8-1. [현재 설정 불러오기] 클릭 → GET /api/config → 저장된 설정으로 UI 반영 (UC-09)
```

> **v1.x 참고**: 전처리 설정 저장이 `session_state.preprocessing_config` Write 방식이었으며,
> 미리보기가 슬라이더 on_change 즉시 Streamlit rerun으로 갱신됐음.
> v2.0: POST /api/config/preview 호출로 대체.

---

#### UC-03: EfficientAD 실험 설정 및 실행

```
액터:      P-01
전제조건:  UC-01 완료, datasetStore.datasetMeta != null
성공 종료: history.json에 status="completed" 레코드 저장 (FastAPI 기록)
실패 종료: history.json에 status="중단" 또는 "실패" 레코드, 에러 메시지 표시

주 플로우 — Config 화면 (설정):
  1. 사용자가 Explorer Config 화면에 진입 (guard 통과)
  2. 화면이 GET /api/config 응답으로 deviceInfo 표시 (CUDA / CPU 자동 감지)
  3. 사용자가 모델 선택에서 "EfficientAD" 선택
  4. 사용자가 기본 노출 파라미터 설정
     (model_size, train_steps, optimizer, learning_rate, weight_decay,
      out_channels, padding, ae_loss_weight)
  5. ae_loss_weight(α) 설정 — ST 비중(1-α)은 학습 루프에서 자동 적용
  6. (선택) 고급 파라미터 패널 열기 → 고급 파라미터 설정
  7. (선택) early_stopping / patience / min_delta 설정 🆕
  8. Threshold 방식·값 설정
  9. [설정 저장] 클릭 → POST /api/config
  10. configStore.modelConfig 업데이트

주 플로우 — Training 화면 (실행):
  11. 사용자가 Explorer Training 화면에 진입 (guard 통과)
  12. 실험명 입력 (빈칸이면 자동 생성: efficientad_{YYYYMMDD}_{HHMMSS}_{4자리})
  13. 사용자가 [학습 시작] 클릭 → POST /api/training/start
  14. trainingStore.status = 'running'
  15. WS /ws/training 연결 → 실시간 메시지 수신:
      - type: 'progress' → Progress Bar, 소요 시간 갱신
      - type: 'log' → 로그 텍스트 영역 갱신
      - type: 'stage' → 학습 단계 인디케이터 갱신
      - type: 'completed' → 완료 알림 + 소요 시간 표시
  16. 학습 완료 (type: 'completed' 수신):
      16a. 완료 알림 및 소요 시간 표시 ({ auc, duration_seconds, early_stopped })
      16b. history.json에 status="completed" 레코드 자동 기록 (서버)
      16c. model_state_dict.pth + configs.yaml 저장 (서버)
      16d. trainingStore.status = 'idle'

대안 플로우:
  13-1. [학습 중지] 클릭 (학습 중 표시)
        → POST /api/training/stop
        → type: 'stopped' 수신 → trainingStore.status = 'idle'
        → history.json에 status="중단" 레코드 (서버 자동 기록)
  13-2. [⏸ 일시정지] 클릭 → POST /api/training/pause
        → type: 'paused' 수신 ({ step, ckpt_path }) → trainingStore.status = 'paused'
        → 체크포인트 저장 완료
  13-3. [▶ 재개] 클릭 → POST /api/training/unpause → type: 'progress' 수신 재개

실패 플로우:
  GPU OOM → type: 'error' 수신 → Training 화면 에러 표시 + batch_size 축소 권장
           → trainingStore.status = 'idle'

배치 학습 플로우 🆕:
  - Config 화면에서 [큐에 추가] 클릭 → POST /api/queue (QueueItem 추가)
  - Training 화면에서 [배치 학습 시작] 클릭 → POST /api/training/batch/start
  - type: 'batch_item_started' → 현재 진행 항목 표시
  - type: 'batch_completed' → 완료/실패/건너뜀 카운트 표시
```

> **v1.x 참고**: v1.x에서 학습 진행상황은 `threading.Thread + queue.Queue → st.empty() + st.rerun()`
> 패턴으로 Streamlit 세션에 표시했음.
> v2.0: `WS /ws/training` 비동기 Push로 대체됨.
> v1.x `current_run_status` → v2.0 `trainingStore.status: 'idle' | 'running' | 'paused'`

---

#### UC-04: PatchCore 실험 설정 및 실행

```
액터:      P-01
전제조건:  UC-01 완료, datasetStore.datasetMeta != null
성공 종료: history.json에 status="completed" 레코드 저장

주 플로우 — Config 화면 (설정):
  1~2.  UC-03 1~2와 동일
  3.    사용자가 모델 선택에서 "PatchCore" 선택
  4.    사용자가 기본 노출 파라미터 설정
        (backbone, pretrained_source, coreset_sampling_ratio, neighbourhood_kernel_size)
  4a.   pretrained_source == "local" 선택 시 pretrained_path 입력 필드 렌더링
  5.    (선택) 고급 파라미터 패널 → max_train, knn, top_k_ratio
  6~10. UC-03 8~10과 동일

주 플로우 — Training 화면 (실행):
  11~16. UC-03 11~16과 동일
         단, type: 'stage' 메시지: "Memory Bank 구축", "Coreset 샘플링" 등 PatchCore 단계 표시
         Loss 차트는 PatchCore 학습 특성상 실시간 Loss 값이 없을 수 있음
         [확인 필요: PatchCore Loss WS 스트림 동작 여부]

비고:
  PatchCore는 train_steps 파라미터를 사용하지 않는다.
  Config 화면에서 PatchCore 선택 시 train_steps UI를 미렌더링한다.
```

---

#### UC-05: 실험 결과 분석

```
액터:      P-01
전제조건:  UC-03 또는 UC-04 완료, status="completed" 실험 ≥ 1개
성공 종료: 선택한 실험의 상세 결과 시각화 완료,
           experimentsStore.selectedExperimentId 업데이트

주 플로우:
  1. 사용자가 Explorer Experiments 화면에 진입 (guard 통과)
  2. 화면이 GET /api/experiments 호출 → 실험 목록 테이블 렌더링
     컬럼: 실험명 / 모델 / 파라미터 요약 / Accuracy / Precision / Recall / F1 / F2 / AUC / 실행 시각 / 상태
     status="중단"/"실패"/"건너뜀"인 행: 회색 텍스트 처리
  3. 사용자가 테이블에서 실험 1개 선택 (행 클릭)
  4. experimentsStore.selectedExperimentId 업데이트
  5. 상세 결과 렌더링 (같은 화면 하단 또는 우측 패널):
     5a. 성능 지표 카드 4종: Accuracy, Precision, Recall, F1
     5b. Confusion Matrix (2×2 div 구현)
     5c. ROC Curve + AUC 값 (Recharts, JS trapezoidal AUC 계산)
     5d. Anomaly Score 분포 히스토그램 (Min-Max 정규화 후 렌더링)

대안 플로우:
  3-1. status != "completed" 실험 선택 → 지표 카드·차트 미렌더링,
       "완료된 실험이 아닙니다" 안내
```

> **v1.x 참고**: `session_state.selected_experiment_id` Write 방식 →
> v2.0 `experimentsStore.selectedExperimentId` + GET /api/experiments 응답 기반 렌더링.

---

#### UC-06: 두 실험 비교 및 최적 모델 선택

```
액터:      P-01
전제조건:  status="completed" 실험 ≥ 2개
성공 종료: 비교 차트 렌더링 완료

주 플로우:
  1. UC-05 플로우 진행 (Experiments 화면 진입)
  2. 사용자가 테이블에서 체크박스로 비교 대상 다중 선택
  3. 비교 메트릭 다중 선택 (Accuracy / Precision / Recall / F1 / F2)
  4. 차트 유형 선택 (막대 차트 / 레이더 차트)
  5. 시스템이 비교 차트 렌더링 (Recharts)
  6. (선택) set_id 기준 배치 실험 비교 테이블 🆕
     - set_id 그룹화 → 정렬 지표 선택 → 그룹별 비교 테이블
  7. 사용자가 최적 모델 판단 후 UC-08 진행

대안 플로우:
  2-1. 선택 실험이 2개 미만이면 비교 차트 미렌더링, "2개 이상 선택 필요" 안내
```

---

#### UC-07: AnomalyMap으로 모델 행동 검증

```
액터:      P-01
전제조건:  UC-05 완료, experimentsStore.selectedExperimentId != null, status="completed"
성공 종료: Triplet 이미지 그리드 렌더링, CSV 또는 ZIP 내보내기 완료

주 플로우:
  1. 사용자가 Explorer AnomalyMap 화면에 진입 (guard 통과)
  2. 화면이 GET /api/anomaly-map/{expId}/status 호출 → 빌드 완료 여부 확인
  3. 빌드 미완료 시:
     3a. [Anomaly Map 생성] 버튼 표시
     3b. 사용자 클릭 → POST /api/anomaly-map/{expId}/build → { job_id } 반환
     3c. GET /api/anomaly-map/job/{jobId} polling (1초 간격) → status 확인
     3d. status="completed" 시 이미지 목록 로드
  4. 빌드 완료 시: 즉시 이미지 목록 로드
     → GET /api/anomaly-map/{expId}/images → AnomalyMapImagesResponse 수신
     (score_max, score_avg, TP/FP/TN/FN 통계 포함)
  5. Threshold 슬라이더 표시 (초기값: experiment.threshold_value 기반 자동 계산, 300ms debounce)
     범위: 0 ~ 1.2, step: 0.01
  6. 결함 유형 필터 드롭다운 (unique defect_class 추출)
  7. 이미지 그리드 렌더링 (4열, 20개/페이지):
     - 각 이미지 카드: 분류 배지 (TP: 초록 / FP: 빨강 / TN: 파랑 / FN: 주황)
     - 이미지 클릭 → GET /api/anomaly-map/{expId}/image/{path}/triplet
       → Triplet PNG (원본/GT마스크/Heatmap) 인라인 표시
  8. 통계 바: 전체 수, TP/FP/TN/FN 개수, Max/Avg Score
  9. 내보내기:
     - [CSV 다운로드] → GET /api/anomaly-map/{expId}/export/csv
     - [ZIP 다운로드] → POST /api/anomaly-map/{expId}/export/zip → job polling
       → GET /api/anomaly-map/zip/{jobId} → 파일 다운로드

대안 플로우:
  3d-1. job status="failed" → 에러 메시지 + [재시도] 버튼
```

> **v1.x 참고**: v1.x에서 Anomaly Map은 학습 완료 시 metrics.anomaly_scores에 이미 계산됐으며
> 이미지 선택 시 모델 재로드 + 재추론이 st.spinner() 동기 방식으로 실행됐음.
> v2.0: 비동기 build job 패턴 (POST → polling → 완료 후 이미지 제공)으로 전면 교체.
> v1.x `탭5 PNG 저장` → v2.0 CSV + ZIP 비동기 내보내기로 확장.

---

#### UC-08: 최적 모델 저장

```
액터:      P-01
전제조건:  UC-05 완료, 저장할 실험 선택됨 (status="completed")
성공 종료: model_state_dict.pth + configs.yaml 저장, 저장 경로·용량 출력

주 플로우:
  1. Experiments 화면에서 저장 대상 실험 선택 (UC-05 3단계)
  2. 저장 경로 입력 필드 (기본값: ./models/{exp_id}/)
  3. [모델 저장] 클릭 → POST /api/experiments/{id}/save { save_path }
  4. 서버가 디스크 여유 공간 확인 → < 500MB 시 경고 응답
  5. state_dict 저장: torch.save(model.state_dict(), path/model_state_dict.pth)
  6. configs.yaml 복사
  7. 저장 완료 메시지 표시: "저장 완료: {path} ({size_mb:.1f} MB)"
  8. experiment.model_path, configs_path 갱신

실패 플로우:
  4-1. 디스크 부족으로 저장 실패 → 에러 메시지 표시
```

---

#### UC-09: 설정 재사용 (서버 설정 불러오기)

```
액터:      P-01
전제조건:  이전 실험의 설정이 서버에 저장되어 있다
성공 종료: Config 화면 UI 필드에 설정값 자동 반영

주 플로우:
  1. Explorer Config 화면에서 [현재 설정 불러오기] 클릭
  2. GET /api/config 호출 → ConfigResponse 수신
     { preprocessing_config, model_config, device_info }
  3. 반환된 설정이 null이 아닌 경우 Config 화면 UI 값 자동 반영
  4. 반영된 값으로 미리보기 갱신

대안 플로우:
  3-1. 서버에 저장된 설정 없음 (preprocessing_config = null) →
       "저장된 설정이 없습니다" 안내, 현재 UI 상태 유지
```

> **v1.x 참고**: v1.x에서는 `configs.yaml` 파일 경로를 직접 입력하여 불러왔음 (`load_config(path)`).
> v2.0: GET /api/config로 서버 저장 설정 조회 방식으로 교체.

---

#### UC-10: 실패 실험 정리 (히스토리 삭제)

```
액터:      P-01
전제조건:  삭제 대상 실험이 history.json에 존재한다
성공 종료: history.json에서 해당 레코드 제거, Experiments 화면 목록 즉시 갱신

주 플로우:
  1. Experiments 화면 실험 목록 테이블에서 삭제 대상 선택
  2. [실험 삭제] 버튼 클릭
  3. 확인 모달 표시: 삭제 경고 + [확인] / [취소] 버튼
  4. [확인] 클릭 → DELETE /api/experiments/{id}:
     4a. history.json에서 해당 experiment_id 레코드 제거 (원자적 쓰기)
     4b. ./models/{exp_id}/ 디렉토리 삭제 (model_path != null인 경우만)
     4c. ./logs/{exp_id}.log 삭제
  5. 응답 수신 후 experimentsStore 업데이트:
     - experimentsStore.selectedExperimentId가 삭제된 ID이면 null로 초기화
  6. Experiments 화면 목록 자동 갱신 (GET /api/experiments 재호출)

비고: [취소] 클릭 시 아무 변경 없음
```

> **v1.x 참고**: v1.x에서 `session_state.experiments` dict에서 직접 키를 제거하고
> `selected_experiment_id == 삭제된 ID`이면 None으로 초기화했음.
> v2.0: DELETE /api/experiments/{id} 서버 호출 + 클라이언트 store 동기화.

---

#### UC-11: EC2 환경 배포 (3개 레포)

```
액터:      P-02
전제조건:  Docker, NVIDIA Container Toolkit 설치된 Ubuntu 22.04 EC2 인스턴스,
           Node.js 18 이상 설치됨
성공 종료: http://{ec2-ip}:8000 (API), http://{ec2-ip}:5173 (Explorer 또는 Vision) 정상 동작

주 플로우 — Dashboard (FastAPI 서버):
  1. EC2 SSH 접속
  2. git clone {smart-qc-dashboard 레포}
  3. docker compose -f docker-compose.base.yml -f docker-compose.yml build
  4. docker compose -f docker-compose.base.yml -f docker-compose.yml up -d
     (GPU 사용: --gpus all 설정 포함)
  5. http://{ec2-ip}:8000/docs 접속 → FastAPI Swagger UI 정상 확인

주 플로우 — Explorer (React 프론트엔드):
  6. git clone {smart-qc-explorer 레포}
  7. npm install
  8. npm run build
     (또는 개발 환경: npm run dev → http://localhost:5173)

주 플로우 — Vision (React 프론트엔드):
  9.  git clone {smart-qc-vision 레포}
  10. npm install
  11. npm run build
      (또는 개발 환경: npm run dev → http://localhost:5173)

포트 구성:
  - Dashboard (FastAPI): :8000
  - Explorer (React): :5173 (Explorer 실행 시)
  - Vision (React):   :5173 (Vision 실행 시)
  ※ Explorer와 Vision은 동시에 실행되지 않는 별개 앱

CPU 전용 환경:
  docker compose -f docker-compose.base.yml -f docker-compose.cpu.yml up -d

실패 플로우:
  3-1. 빌드 실패 → requirements.txt 패키지 버전 충돌 확인
  4-1. GPU 인식 실패 → nvidia-smi 명령으로 드라이버 확인
  7-1. npm install 실패 → Node.js 18 이상 버전 확인
```

> **v1.x 참고**: v1.x에서 단일 Streamlit 앱만 배포했음:
> `docker run -p 8501:8501 vision-inspection-dashboard:latest`
> v2.0: Dashboard(FastAPI) + Explorer(React) + Vision(React) 3개 서비스 별도 실행.

---

#### UC-12: 결함 유형별 탐지 성능 필터링 확인

```
액터:      P-01
전제조건:  UC-07 진입 조건과 동일 (AnomalyMap 빌드 완료)
성공 종료: 선택한 결함 유형의 이미지만 그리드에 표시

주 플로우:
  1. Explorer AnomalyMap 화면 진입
  2. 결함 유형 필터 드롭다운에서 특정 클래스 선택 (예: "crack", "scratch")
     드롭다운 옵션: ["전체"] + 로드된 이미지에서 추출한 unique defect_class
  3. 선택 클래스의 이미지만 그리드 필터링 (클라이언트 필터링, 서버 재요청 없음)
  4. UC-07 7~9 플로우와 동일

비고: "전체" 선택 시 필터 없이 전체 테스트 이미지 표시
```

---

### B.4 사용자 여정 맵 (P-01, 첫 실험 완주)

```
[인식 단계]
데이터셋 준비 → Dataset 화면: 경로 입력 → POST /api/dataset/validate → 구조 검증 통과 확인
    ↓ (소요: ~2분)

[설정 단계]
Config 화면: 전처리 방식 선택 → 실시간 미리보기 확인 → 모델 파라미터 설정 → [설정 저장]
    ↓ (소요: ~5분)

[실행 단계]
Training 화면: 실험명 입력 → [학습 시작] → WS 실시간 Progress·Loss 모니터링 → 완료 알림 수신
    ↓ (소요: EfficientAD ~20분 / PatchCore ~10분)

[분석 단계]
Experiments 화면: AUC 확인, Confusion Matrix 검토 → 필요 시 Config 화면으로 돌아가 재실험
    ↓ (소요: ~3분)

[검증 단계]
AnomalyMap 화면: [Anomaly Map 생성] → Triplet 그리드로 결함 위치 시각화 → 모델 행동 이해
    ↓ (소요: ~3분)

[저장 단계]
Experiments 화면: [모델 저장] → POST /api/experiments/{id}/save → Vision 팀에 경로 전달
    ↓ (소요: ~1분)

총 소요: 단일 실험 사이클 약 30분 이내 (성공 지표 기준)
```

---

### B.5 Edge Cases (페르소나 관점)

| # | 상황 | 영향 페르소나 | 처리 방식 |
|---|------|-------------|-----------|
| ECU-01 | Experiments 화면에서 실험 삭제 후 AnomalyMap 화면 접근 | P-01 | selectedExperimentId = null 자동 초기화 → AnomalyMap 화면 진입 차단 + 안내 |
| ECU-02 | Config 화면에서 EfficientAD → PatchCore로 모델 변경 | P-01 | modelConfig 완전 교체, 이전 EfficientAD 설정 유지 안됨 (안내 필요) |
| ECU-03 | 학습 중 WS 연결 끊김 | P-01 | trainingStore.status 즉시 반영 안 됨 → 화면 새로고침 후 GET /api/training/status 재조회 로 상태 복구 [확인 필요: 자동 WS 재연결 구현 여부] |
| ECU-04 | 동일 설정으로 실험 2회 실행 | P-01 | experiment_id에 4자리 난수 포함이므로 중복 방지됨. 두 실험이 별개 레코드로 저장됨 |
| ECU-05 | Vision 앱 실행 중 Explorer도 실행 시도 | P-02 | 동일 포트(5173) 충돌 → 두 앱은 별개 시나리오에서 실행해야 함 |

---

## C. System & Data Design

```
N/A — 이 문서는 사용자·시나리오 정의가 목적이다.
      데이터 모델과 시스템 설계는 00_Global_Context_Document.md 1~5절에 정의되어 있으며,
      05_Data_Model.md에서 상세 확장된다.
```

---

## D. API Contracts

이 문서에서 참조하는 주요 API 엔드포인트 요약 (상세 명세는 [06_API_Specification.md](./06_API_Specification.md) 참조):

| UC | 관련 엔드포인트 |
|----|----------------|
| UC-01 | POST /api/dataset/validate |
| UC-02 | GET /api/config, POST /api/config, POST /api/config/preview |
| UC-03, 04 | POST /api/training/start·stop·pause·unpause, POST /api/training/batch/start, WS /ws/training |
| UC-05, 06 | GET /api/experiments |
| UC-07, 12 | GET /api/anomaly-map/{expId}/status, POST /api/anomaly-map/{expId}/build, GET /api/anomaly-map/job/{jobId}, GET /api/anomaly-map/{expId}/images, GET /api/anomaly-map/{expId}/image/{path}/triplet, GET /api/anomaly-map/{expId}/export/csv, POST /api/anomaly-map/{expId}/export/zip |
| UC-08 | POST /api/experiments/{id}/save |
| UC-09 | GET /api/config |
| UC-10 | DELETE /api/experiments/{id} |
| UC-13, 18 | GET /api/models, POST /api/inspection/model |
| UC-14 | POST /api/inspection/run, GET /api/inspection/job/{id} |
| UC-15 | WS /ws/inspection/auto |
| UC-17 | GET /api/inspection/records, GET /api/inspection/records/csv, DELETE /api/inspection/records |

---

## E. AI/ML Details

```
N/A — 이 문서는 사용자 페르소나와 Use Case 정의가 목적이다.
      AI/ML 모델 상세 명세는 08_AI_ML_Integration.md에서 다룬다.
      페르소나 관점에서 모델 선택 근거는 01_Product_Overview.md E절 참조.
```

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체를 상속한다.

페르소나 관점에서 추가로 명시:

| 항목 | 요구사항 | 근거 페르소나 |
|------|----------|--------------|
| **인터페이스 언어** | 전체 UI 한국어, 기술 용어 한국어+영문 병기 | P-01, P-03 (한국어 사용자) |
| **학습 시작 전 확인 없음** | [학습 시작] 클릭 즉시 실행. 추가 확인 팝업 없음 | P-01 (실험 반복 빈도 높음) |
| **[실험 삭제] 확인 팝업 필수** | 데이터 삭제는 되돌릴 수 없으므로 1회 확인 | P-01 (실수 방지) |
| **파라미터 기본값 사전 설정** | 모든 파라미터는 합리적인 기본값으로 초기화됨 | P-01 입문 수준 고려 |
| **고급 설정은 접을 수 있는 패널로 숨김** | 입문 사용자에게 기본값으로 시작할 수 있는 경로 제공 | P-01 입문~중급 모두 |
| **Vision 불량 알림 즉각성** | 불량 판정 즉시 DefectPopup 표시, 화면 전환 없음 | P-03 (현장 즉각 조치 필요) |
| **Vision 버튼 단순화** | 핵심 검사 버튼 4개(수동/자동/중지/불량만) 상단 고정 | P-03 (비전문가) |

---

## G. Observability

```
N/A — 페르소나 문서에서 추가 관측 항목 없음.
      시스템 관측 기준은 00_Global_Context_Document.md 7절 및
      12_Observability_and_Operations.md에서 다룬다.
```

---

## H. QA & Validation

### H.1 페르소나 기반 인수 기준

| # | 기준 | 검증 방법 |
|---|------|-----------|
| PUA-01 | P-01이 코드 작성 없이 Explorer Dataset→Config→Training→Experiments→AnomalyMap Golden Path 완주 가능 | 실사용자 수동 테스트 |
| PUA-02 | P-01이 EfficientAD와 PatchCore 두 실험을 Experiments 화면에서 비교하여 더 높은 AUC 모델 식별 가능 | UC-06 시나리오 실행 |
| PUA-03 | P-02가 README 없이 3개 레포 실행 명령만으로 EC2 배포 완료 가능 | UC-11 시나리오 실행 |
| PUA-04 | AnomalyMap 화면에서 P-01이 Triplet(원본/GT/Heatmap) 이미지로 결함 위치를 육안 확인 가능 | 시각 검수 |
| PUA-05 | [학습 중지] 후 재실험 시 이전 중단 상태가 현재 실험에 영향 없음 | UC-03 대안 플로우 후 재실행 |
| PUA-06 | P-03이 Vision Realtime Inspection 화면에서 수동·자동 검사 모두 조작 가능 | UC-14, UC-15 시나리오 실행 |
| PUA-07 | P-04가 History 화면에서 CSV 내보내기 후 Excel 열기 가능 | UC-17 시나리오 실행 |

---

### H.2 Given-When-Then 시나리오

#### TC-UC-02: 전처리 미리보기 즉시 반응

```
Given:  Explorer Config 화면이 렌더링된 상태이고 Homomorphic이 선택되어 있다
When:   사용자가 sigma 슬라이더를 10.0에서 30.0으로 변경한다
Then:   300ms debounce 이후 POST /api/config/preview 호출
        우측 처리 후 이미지가 1초 이내에 갱신된다
        원본 이미지(좌)는 변경되지 않는다
        configStore.preprocessingConfig 는 아직 갱신되지 않는다
        (설정 저장은 [설정 저장] 버튼 클릭 시에만 발생)
```

#### TC-UC-03: ae_loss_weight 저장 확인

```
Given:  Explorer Config 화면에서 EfficientAD가 선택된 상태이다
        ae_loss_weight = 0.5
When:   사용자가 ae_loss_weight 슬라이더를 0.7로 변경하고 [설정 저장]을 클릭한다
Then:   POST /api/config 호출
        configStore.modelConfig.params.ae_loss_weight == 0.7
        ST 비중(1 - 0.7 = 0.3)은 학습 루프 내부에서 자동 적용된다
```

#### TC-UC-10: 실험 삭제 안전성

```
Given:  experimentsStore.selectedExperimentId = "efficientad_20260508_140023_7f3a"
        해당 실험이 history.json에 존재한다
        AnomalyMap 화면이 해당 실험을 참조 중이다
When:   사용자가 Experiments 화면에서 해당 실험을 삭제하고 [확인]을 클릭한다
Then:   DELETE /api/experiments/{id} 응답 성공
        history.json에 해당 레코드가 존재하지 않는다
        experimentsStore.selectedExperimentId == null
        AnomalyMap 화면 진입 시 "실험을 먼저 선택해 주세요" 안내 표시
        ./models/efficientad_20260508_140023_7f3a/ 디렉토리가 존재하지 않는다
```

#### TC-UC-16: 불량 감지 팝업 및 재개

```
Given:  Vision Realtime Inspection 화면에서 자동 검사 중 (isAutoRunning = true)
When:   WS /ws/inspection/auto 에서 type: 'defect_stopped' 메시지 수신
Then:   inspectionStore.defectStopped = true
        DefectPopup 모달 표시 (고정 오버레이, bg-black/55)
        "❌ 불량이 감지되었습니다!" 헤더 표시
        [✅ 확인 및 재개] 클릭 시: setDefectStopped(false) + start() (WS 재연결)
        [🛑 검사 종료] 클릭 시: setDefectStopped(false) (팝업 닫기, 자동 검사 중지 유지)
```

---

## I. Implementation Plan

```
N/A — 전체 구현 계획은 14_Deployment_and_Release_Plan.md에서 다룬다.
      이 문서의 Use Case 시나리오는 13_QA_and_Testing_Strategy.md의
      테스트 케이스 설계 시 Given-When-Then 입력값으로 직접 재사용된다.
```

---

---

### B.6 Persona P-03: 현장 검사 작업자

#### 프로파일

| 속성 | 내용 |
|------|------|
| **직책** | 생산 라인 검사 작업자 |
| **경력** | 제조 현장 경험, IT·ML 비전문가 |
| **주요 목표** | 빠르고 정확하게 부품의 양품/불량 판정, 불량 부품 즉시 조치 |
| **사용 환경** | 현장 PC 브라우저 (Vision 앱, localhost:5173), 마우스·키보드 기본 조작 |
| **기술 수준** | 비전문가 — 복잡한 수치나 그래프 해석 불필요, 큰 글씨와 색상으로 판정 결과 파악 |

#### 페인 포인트

| # | 페인 포인트 | 이 제품의 해결 방식 |
|---|------------|-------------------|
| PP-10 | 불량이 발생해도 인지하지 못하고 라인이 계속 진행됨 | 불량 감지 즉시 DefectPopup 모달 알림 + 자동 검사 중지 |
| PP-11 | 판정 결과가 어디에 불량이 있는지 설명 없음 | 이상 영역 오버레이 이미지 패널로 불량 위치 시각적 표시 |
| PP-12 | 다음 부품 검사를 위해 복잡한 조작이 필요함 | 버튼 4개(수동/자동/중지/불량만)로 단순화 |

---

### B.7 Persona P-04: 품질 관리자

#### 프로파일

| 속성 | 내용 |
|------|------|
| **직책** | 품질팀 관리자 또는 공정 엔지니어 |
| **주요 목표** | 불량률 추이 파악, 이력 데이터 기반 품질 보고 |
| **사용 환경** | 현장 PC 또는 사무실 PC 브라우저 (Vision 앱, localhost:5173) |

---

#### UC-13: 검사 모델 선택 및 검사 시작

```
액터:      P-03 (또는 P-04)
전제조건:  Vision 앱에 접속, FastAPI 서버 정상 실행,
           history.json에 status="completed" 실험 ≥ 1개
성공 종료: inspectionStore.activeModel 설정 완료,
           Realtime Inspection 화면에서 검사 가능 상태

주 플로우:
  1. Vision 앱 (http://localhost:5173) 접속
  2. 최초 진입 시: useActiveModel() → GET /api/inspection/model
     → 이전 세션 모델 정보 확인 (없으면 activeModel = null → NoModelGuard 표시)
  3. [설정] 탭 클릭 → Model Settings 화면 진입
  4. 화면이 GET /api/models (30초 폴링) → 완료 실험 목록 테이블 렌더링
     컬럼: (선택 radio), 실험명, 검사 제품, 모델타입, F1, AUC, 실행시각
  5. 원하는 실험 행 선택
  6. [모델 적용] 클릭 → POST /api/inspection/model { experiment_id }
  7. 응답: { active_model, gpu_warning }
     → inspectionStore.setActiveModel(active_model, gpu_warning)
     → gpu_warning 있으면 GpuWarningBanner 황색 배너 표시
     → lastResult, imageStamp, isAutoRunning, defectStopped 초기화 (서버 records 초기화)
  8. ModelStatusChip 초록으로 전환 → Realtime Inspection 화면으로 이동
```

> **v1.x 참고**: v1.x에서 사이드바 [🏭 비전검사 대시보드] 클릭 후 탭3 진입,
> `insp_active_model` session_state 직접 갱신 + `reset_inspection_state()` 호출.
> v2.0: POST /api/inspection/model → 서버 상태 갱신 + inspectionStore 동기화.

---

#### UC-14: 수동 검사 실행 및 판정 확인

```
액터:      P-03
전제조건:  UC-13 완료, inspectionStore.activeModel != null
성공 종료: InspectionResult 1건 반환,
           Realtime Inspection 화면에 판정 pill + 이미지 3패널 표시

주 플로우:
  1. Vision Realtime Inspection 화면 진입 (NoModelGuard guard 통과)
  2. [수동 검사 (1개 검사)] 클릭
  3. POST /api/inspection/run {} → { job_id } 반환
  4. GET /api/inspection/job/{job_id} polling (1초 간격, max 120초):
     → status: 'pending' | 'running' → 계속 polling (버튼 disabled)
     → status: 'completed' → result: InspectionResult 취득
     → status: 'failed' → 에러 메시지 표시
  5. inspectionStore.setLastResult(result) → imageStamp = Date.now()
  6. 헤더 행 판정 pill 갱신:
     양품: 초록 "✅ 양품 {score}"
     불량: 빨강 "❌ 불량 {score}"
  7. 이미지 3패널 갱신 (cache-bust ?t={imageStamp}):
     - 원본 이미지: GET /api/inspection/image/last?t={stamp}
     - Anomaly Map:  GET /api/inspection/anomaly-map/last?t={stamp}
     - 오버레이:     GET /api/inspection/overlay/last?t={stamp}
  8. 양품이면 버튼 활성 유지, 다음 검사 대기
  9. 불량이면 UC-16 DefectPopup 팝업 플로우 진행

대안 플로우:
  4-1. polling 120초 타임아웃 → "검사 시간이 초과됐습니다." 에러 표시
```

> **v1.x 참고**: v1.x에서 동기 추론으로 즉시 결과 반환, 4열 레이아웃(판정카드+원본+AnomalyMap 표시).
> v2.0: 비동기 job + polling 패턴으로 교체, 3열 이미지 패널 + 헤더 인라인 pill 레이아웃.

---

#### UC-15: 자동 검사 실행 및 중지

```
액터:      P-03
전제조건:  UC-13 완료
성공 종료: 자동 루프 실행 후 수동 중지 또는 불량 감지로 중지

주 플로우:
  1. [▶ 자동 검사 (3초마다 1개)] 클릭
  2. WS /ws/inspection/auto 연결 → onopen: send('start')
     → inspectionStore.isAutoRunning = true
     → AutoRunningBanner 황색 배너 표시
     → ▶ 자동 검사 / 수동 검사 / 불량만 검사 버튼 disabled
  3. WS type: 'result' 수신 시마다:
     → setLastResult(result) → imageStamp = Date.now()
     → 판정 pill + 이미지 3패널 자동 갱신
     → was_reshuffled == true → reshuffledToast 표시 (3초 자동 해제)
  4. [⏹ 자동 검사 중지] 클릭:
     → send('stop') → isAutoRunning = false
     → WS type: 'stopped' 수신 → 루프 종료

대안 플로우:
  3-1. WS type: 'defect_stopped' 수신 → isAutoRunning = false + UC-16 DefectPopup 진행
  3-2. WS type: 'error' 수신 → isAutoRunning = false, 에러 표시
  화면 이탈 (Tab1 unmount): isAutoRunning == true → send('stop') + WS close()
```

> **v1.x 참고**: v1.x에서 `insp_auto_active = True → st.rerun() → time.sleep(3) → st.rerun()` 반복.
> v2.0: WS /ws/inspection/auto Push 방식으로 전면 교체. 3초 간격은 서버 제어.

---

#### UC-16: 불량 감지 팝업 확인 및 검사 재개

```
액터:      P-03
전제조건:  추론 결과 verdict == "불량" (자동 검사 중 defect_stopped 메시지 수신)
성공 종료: DefectPopup 닫힘, 검사 재개 가능 상태 (재개 선택 시 자동 검사 즉시 재시작)

주 플로우:
  1. WS type: 'defect_stopped' 수신
     → inspectionStore.defectStopped = true
     → DefectPopup 모달 표시 (고정 오버레이, z-[1000])
       헤더: "❌ 불량이 감지되었습니다! 자동 검사가 중지되었습니다." (red-100 배경)
  2. 이미지 3패널은 계속 렌더링됨 (결과 확인 가능)
  3. 팝업 버튼 선택:
     [✅ 확인 및 재개] 클릭:
       → setDefectStopped(false)
       → start() (새 WS 연결 + send('start'))
       → isAutoRunning = true → 자동 검사 즉시 재개
     [🛑 검사 종료] 클릭:
       → setDefectStopped(false) (팝업만 닫힘)
       → isAutoRunning = false 유지 (자동 검사 중지 상태 유지)
       → 수동으로 재시작 버튼 클릭 필요
```

> **v1.x 참고**: v1.x에서 `st.dialog` 팝업으로 표시, `insp_defect_popup = False`로 닫힘.
> 확인 후 자동 재개 없이 "수동으로 자동 검사 버튼을 다시 눌러야" 하는 구조.
> v2.0: DefectPopup 모달 컴포넌트, [✅ 확인 및 재개] 클릭 시 WS 재연결로 즉시 재개 가능.

---

#### UC-17: 검사 이력 조회 및 CSV 내보내기

```
액터:      P-04
전제조건:  검사 이력 ≥ 1건
성공 종료: 이력 테이블 확인, CSV 파일 브라우저 다운로드

주 플로우:
  1. [검사 이력] 탭 클릭 → Vision History 화면 진입 (NoModelGuard guard 통과)
  2. GET /api/inspection/records → allRecords 로드 (마운트 1회)
  3. 이력 테이블 표시 (seq 내림차순):
     컬럼: 번호 / 시각 / 이미지명 / 판정결과 / Anomaly Score
     PAGE_SIZE=10, 페이지네이션
  4. 판정 필터 라디오 선택 (전체 / 양품만 / 불량만)
     → 클라이언트 필터링 (서버 재요청 없음)
  5. KPI 카드 확인 (총검사 / 양품 / 불량 / 불량률), 클라이언트 계산
  6. 통계 차트 확인:
     - 단위 선택 (20 / 40 / 100 개)
     - 시간 범위 테이블 / Score 히스토그램 / Score 산점도 3분할
  7. [CSV 내보내기] 클릭 → window.open('/api/inspection/records/csv') → 파일 다운로드

대안 플로우:
  7-1. [🗑 이력 초기화] 클릭 → ClearHistoryDialog 모달 표시
       [초기화 확인] 클릭 → DELETE /api/inspection/records → allRecords = []
```

> **v1.x 참고**: v1.x에서 이력이 `session_state.insp_records` 리스트에 저장됐으며
> 앱 재시작 시 초기화됐음. v2.0: FastAPI 서버 메모리에 저장 (모델 교체 시 초기화,
> 서버 재시작 시 초기화). GET /api/inspection/records 로 조회.

---

#### UC-18: 모델 교체 후 새 검사 세션 시작

```
액터:      P-04
전제조건:  Vision 앱 접속 중, 더 좋은 완료 실험 존재
성공 종료: 새 모델 적용, 기존 이력 서버에서 초기화, 새 검사 세션 시작 가능

주 플로우:
  1. [설정] 탭 클릭 → Model Settings 화면 진입
  2. 현재 적용 모델 (ModelStatusChip 초록 칩으로 표시)
  3. 모델 목록에서 다른 실험 행 선택
  4. "⚠️ 모델을 교체하면 현재 세션의 모든 검사 이력이 삭제됩니다." 경고 문구 표시
  5. [모델 적용] 클릭 → POST /api/inspection/model { experiment_id }
  6. 서버:
     6a. 새 모델 로드 (model_path 경로의 .pth 파일)
     6b. inspection_records 메모리 초기화
     6c. active_model 갱신
  7. 클라이언트:
     7a. inspectionStore.setActiveModel(res.active_model, res.gpu_warning)
         → lastResult, imageStamp, isAutoRunning, defectStopped 초기화
     7b. ModelStatusChip 새 모델명으로 전환
  8. Realtime Inspection 화면으로 이동 → 새 모델로 검사 시작 대기
```

---

---

# v1.x 참고 — Streamlit 탭 기반 흐름 (삭제 금지)

> 이하 내용은 v1.x Streamlit 구현 기준 용어 대조표입니다.
> v2.0 React/FastAPI UI가 공식 구현입니다.

### 탭 이름 ↔ React 화면 이름 대조표

| v1.x 탭 (Explorer) | v2.0 React 화면 | 경로 |
|--------------------|----------------|------|
| 탭1 (데이터 폴더 구조) | Dataset 화면 | `/` |
| 탭2 (전처리 및 모델 설정) | Config 화면 | `/config` |
| 탭3 (학습 시작 + 학습 로그) | Training 화면 | `/training` |
| 탭4 (실험 히스토리 + 결과 + 저장) | Experiments 화면 | `/experiments` |
| 탭5 / 탭6 (이상 영역 시각화) | AnomalyMap 화면 | `/anomaly-map` |

| v1.x 탭 (Vision) | v2.0 React 화면 | 경로 |
|------------------|----------------|------|
| 검사탭1 (실시간 검사) | Realtime Inspection 화면 | `/` |
| 검사탭2 (검사 이력 및 통계) | History 화면 | `/history` |
| 검사탭3 (딥러닝 모델 교체) | Model Settings 화면 | `/settings` |

### 상태 관리 용어 대조표

| v1.x session_state 키 | v2.0 Zustand store / FastAPI |
|-----------------------|------------------------------|
| `dataset_path` | `datasetStore.datasetPath` |
| `dataset_meta` | `datasetStore.datasetMeta` |
| `preprocessing_config` | `configStore.preprocessingConfig` |
| `model_config` | `configStore.modelConfig` |
| `device_info` | `configStore.deviceInfo` |
| `experiments[exp_id]` | `history.json` (FastAPI 서버 영속 기록) |
| `current_run_status` | `trainingStore.status` |
| `selected_experiment_id` | `experimentsStore.selectedExperimentId` |
| `anomaly_map_threshold` | `anomalyMapStore.threshold` |
| `insp_active_model` | `inspectionStore.activeModel` |
| `insp_records` | FastAPI 서버 메모리 (GET /api/inspection/records) |
| `insp_auto_active` | `inspectionStore.isAutoRunning` |
| `insp_defect_popup` | `inspectionStore.defectStopped` |
| `insp_last_result` | `inspectionStore.lastResult` |

---

*다음 문서*: [03_Functional_Requirements.md](./03_Functional_Requirements.md)
