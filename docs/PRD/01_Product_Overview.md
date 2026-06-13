# 01. Product Overview

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **버전**: v2.0
> **작성일**: 2026-05-08
> **최종수정**: 2026-06-11
> **후속 문서**: [02_User_Personas_and_Use_Cases.md](./02_User_Personas_and_Use_Cases.md)

---

## 버전 이력

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-08 | 초기 작성 — Streamlit 단독 앱, 모델 탐색 대시보드(탭1~5) |
| v1.1 | 2026-05-26 | 이중 대시보드 구조 — 비전검사 대시보드(탭1~3) 추가, 사이드바 전환 버튼 |
| v1.2 | 2026-05-29 | 탭2 대기열 UI, 탭3 학습 단계 인디케이터, 탭2 실시간 차트 추가 |
| v2.0 | 2026-06-11 | 3개 레포(Explorer/Vision/Dashboard) 구조로 전면 재작성. Golden Path·MVP 범위·성공 지표·인수 기준 React UI 기준으로 교체. Streamlit 내용은 v1.x 참고 섹션으로 이동 (삭제 금지). |

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

**Smart QC Platform**은 제조산업 품질검사를 위한 딥러닝 기반 비전검사 시스템이다.

3개 레포지토리가 역할 분리된 하나의 플랫폼을 구성한다.

| 레포 (v2.0 명칭) | 대상 사용자 | 핵심 역할 |
|------------------|------------|-----------|
| **smart-qc-explorer** (Explorer) | AI/ML 엔지니어, 데이터 분석가 | 코드 작성 없이 EfficientAD/PatchCore 학습·평가·비교·자산화 |
| **smart-qc-vision** (Vision) | 현장 작업자, 품질 관리자 | 검증된 모델로 부품 추론·판정·이력 관리 |
| **smart-qc-dashboard** (Dashboard) | 시스템 공통 | FastAPI REST API + WebSocket + Anomalib ML 레이어. Streamlit은 개발자 보조 도구 |

**포트 구성:**

| 컴포넌트 | 포트 | 실행 명령 |
|----------|------|-----------|
| Explorer (React) | 5173 | `npm run dev` (smart-qc-explorer) |
| Vision (React) | 5173 | `npm run dev` (smart-qc-vision) |
| Dashboard (FastAPI) | 8000 | `uvicorn api.main:app --port 8000` (smart-qc-dashboard) |
| Dashboard (Streamlit, 보조) | 8501 | `streamlit run app.py` |

> Explorer와 Vision은 동일 포트(5173)를 사용하며, 동시에 실행되지 않는 별개의 React 앱이다.

---

### A.2 해결하는 문제

> **v2.0 안내**: 아래 P-01~P-05는 v1.x에서 정의된 문제이며 v2.0에서도 내용이 유효하다.

| # | 문제 | 현재 상태 | 이 제품이 해결하는 방식 |
|---|------|-----------|------------------------|
| P-01 | 육안 검사의 주관성 | 검사자별 판정 기준 상이, 피로도에 따라 정확도 변동 | 딥러닝 모델로 객관적·일관된 판정 기준 확보 |
| P-02 | 실험 반복 비용 | 모델·전처리 조합 변경 시마다 코드 수정·재실행 필요 | Explorer GUI 기반 파라미터 설정으로 코드 작성 없이 실험 반복 |
| P-03 | 실험 재현성 부족 | 실험 조건이 파일·메모·기억에 산재 | `random_seed` + `configs.yaml` 단일 파일로 조건 완전 재현 |
| P-04 | 모델 비교의 수작업 | 실험별 지표를 수동으로 스프레드시트에 집계 | Explorer Experiments 화면 + 비교 차트로 한 화면 비교 |
| P-05 | 모델 자산화 부재 | 학습된 모델을 추론 애플리케이션에 연계할 표준 형식 없음 | `state_dict + configs.yaml` 고정 포맷 저장 → Vision 앱 직접 연계 |

---

### A.3 제품 비전

> "AI/ML 엔지니어가 하루 안에 EfficientAD와 PatchCore를 모두 실험하고, 더 나은 모델을 골라 현장 Vision 앱에 바로 연결할 수 있는 엔드-투-엔드 품질검사 워크벤치"

- **Explorer**: 학습·분석·모델 자산화
- **Vision**: 자산화된 모델을 즉시 현장 검사에 투입
- **Dashboard**: 두 UI를 단일 FastAPI 백엔드로 연결

---

### A.4 MVP 범위 (v2.0 — 3개 레포 구현 기준)

#### Explorer IN SCOPE

| 기능 영역 | 포함 항목 |
|-----------|-----------|
| **데이터셋 검증** | MVTec AD 및 OK/NG 이진 폴더 구조 자동 감지, 폴더 트리·클래스별 이미지 수 표시, 썸네일 그리드, Grayscale 자동 감지, `background_clean/` 존재 여부 감지 |
| **전처리 설정** | None / Homomorphic / HE / CLAHE 선택 및 파라미터 설정, 배경 제거 (none / SAM2) 선택, 이미지 크기 설정, 정규화 설정 |
| **모델 설정** | EfficientAD (small/medium, 학습 스텝·옵티마이저·LR 등), PatchCore (WideResNet50/ResNet18/ResNet50, coreset ratio 등), Threshold 방법(percentile/absolute), 배치 크기, random seed |
| **배치 학습 큐** | 여러 설정 조합을 큐에 추가 → 순차 실행, 개별 건너뜀 지원 |
| **학습 실행** | WebSocket 실시간 모니터링, Progress Bar + Loss 차트, 학습 단계 인디케이터, 소요 시간 표시, 일시정지/재개/중단, 체크포인트 저장·재시작 |
| **실험 결과** | 메트릭 카드 (Accuracy/Precision/Recall/F1/F2/AUC), Confusion Matrix, ROC Curve, Anomaly Score 분포 히스토그램 |
| **다중 실험 비교** | 체크박스 선택 후 막대/레이더 차트 비교, set_id 기준 배치 비교 테이블 |
| **모델 저장** | `state_dict + configs.yaml` 고정 포맷, 저장 경로 지정 |
| **실험 삭제** | 확인 단계 포함 삭제 |
| **Anomaly Map** | 비동기 job 생성 (캐시 있으면 즉시 완료), Threshold 슬라이더 (0~1.2, 300ms debounce), 결함 유형 필터, TP/FP/TN/FN 그리드, CSV + ZIP 내보내기 |

#### Vision IN SCOPE

| 기능 영역 | 포함 항목 |
|-----------|-----------|
| **실시간 검사** | 수동 검사(1회, POST + polling), 자동 검사(WebSocket, 불량 감지 시 자동 중지), 불량만 검사 (1회, defect_only 옵션) |
| **이미지 표시** | 원본 / Anomaly Map / 이상 영역 오버레이 3패널, `?t={stamp}` cache-bust, 동일 종횡비 동기화 |
| **판정 표시** | 헤더 행 인라인 verdict pill (양품 초록 / 불량 빨강 + score) |
| **불량 감지 알림** | DefectPopup 모달 (확인 및 재개 / 검사 종료), GpuWarningBanner (모델 적용 시), 재셔플 toast (3초), 불량없음 경고 toast (3초) |
| **검사 이력** | 이력 테이블 (5컬럼, PAGE_SIZE=10, 판정 필터링), KPI 카드 4종, 통계 차트 3분할 (단위 20/40/100), CSV 내보내기, 이력 초기화 (확인 모달) |
| **모델 설정** | 완료 실험 목록 (30초 폴링), 모델 적용 + 이력 자동 초기화, 이미지 소스 경로 설정 |
| **진입 제한** | NoModelGuard 오버레이 모달 (activeModel=null 시) |

#### Dashboard IN SCOPE

| 기능 영역 | 포함 항목 |
|-----------|-----------|
| **Explorer API** | POST /api/dataset/validate, GET /api/config, POST /api/config, /api/queue, /api/training/*, WS /ws/training, /api/experiments/*, /api/anomaly-map/* |
| **Vision API** | GET/POST /api/inspection/model, POST /api/inspection/run, GET /api/inspection/job/{id}, GET/DELETE /api/inspection/records, GET /api/inspection/records/csv, GET /api/inspection/image+anomaly-map+overlay/last, PATCH /api/inspection/source-path, WS /ws/inspection/auto, GET /api/models |
| **ML 레이어** | EfficientAD Engine, PatchCore Engine, run_inference(), Evaluator (Anomalib ≥ 1.0) |
| **영속 저장** | history.json (실험 이력), model_state_dict.pth + configs.yaml (모델 파일), 검사 이력 (서버 메모리) |
| **Streamlit (보조)** | 개발자 디버그용 — app.py (streamlit run) |

#### OUT OF SCOPE (v2.0 제외, 향후 확장)

| 제외 항목 | 제외 사유 |
|-----------|-----------|
| 앙상블 (EfficientAD + PatchCore 가중 평균) | 스케일 정합성·Threshold 정책 추가 설계 필요 |
| GAN 기반 이미지 증강 | 데이터셋 의존적 효과·학습 안정성 검증 필요 |
| 다중 사용자 / 권한 관리 | 단일 사용자 워크스테이션 환경 (가정 A-01) |
| 실시간 카메라 연동 | MVP 범위 초과. test 데이터셋 샘플링으로 대체 |
| 클라우드 모델 레지스트리 | MLflow/SageMaker 연동은 추론 앱 단계 |
| 검사 이력 영속 저장 (DB) | 서버 메모리 기반 단순화. 모델 교체 시 초기화 |
| Explorer-Vision 통합 접근 권한 분리 | 단일 로컬 사용자 환경 |

---

### A.5 성공 지표 (v2.0 — React UI 기준)

| 지표 | 목표값 | 측정 방법 | 측정 시점 |
|------|--------|-----------|-----------|
| 모델 정확도 | AUC ≥ 0.95 (MVTec AD Screw 기준) | `history.json → metrics.auc` | 각 실험 완료 후 |
| EfficientAD 학습 시간 | 70,000 steps ≤ 20분 (g4dn.xlarge) | `experiment.duration_seconds ≤ 1200` | 실험 완료 후 |
| PatchCore 학습 시간 | coreset 10% ≤ 10분 (g4dn.xlarge) | `experiment.duration_seconds ≤ 600` | 실험 완료 후 |
| Explorer 단일 실험 사이클 | Dataset→Config→Training→Experiments→AnomalyMap E2E ≤ 30분 | 화면별 소요 시간 합산 | 통합 테스트 |
| Explorer UI 응답성 | 화면 전환 < 1초, 학습 중 UI 블로킹 없음 (WebSocket 비동기) | 수동 테스트 | 통합 테스트 |
| 재현성 | 동일 seed+파라미터 → 동일 AUC (소수점 4자리) | 2회 실행 결과 비교 | QA 단계 |
| Vision 추론 지연 | 이미지 1장 수동 검사 → 판정 결과 표시 ≤ 3초 | 수동 측정 | 통합 테스트 |
| Vision 자동 검사 타이밍 | WS result 수신 간격 ≈ 3초 (오차 ≤ 0.5초) | 수동 측정 | 통합 테스트 |
| Vision 모델 적용 | 모델 적용 후 첫 검사 가능 상태까지 ≤ 5초 | 수동 측정 | 통합 테스트 |

---

## B. Detailed Specification

### B.1 핵심 사용자 플로우 (Golden Path)

#### Explorer Golden Path — 신규 실험 처음부터 완료

```
Step 1 [Dataset 화면]     데이터셋 경로 입력
                          → POST /api/dataset/validate
                          → 폴더 트리, 클래스별 이미지 수, 썸네일 확인
                          → OK/NG 형식 감지 시 80/20 분할 안내 배너
                          → datasetStore (datasetPath, productName, datasetMeta) 저장

Step 2 [Config 화면]      전처리 방법 선택 (none / homomorphic / he / clahe)
                          → 배경 제거 방법 선택 (none / sam2)
                          → 모델 선택 (EfficientAD / PatchCore)
                          → 모델별 파라미터, Threshold 방법·값 설정
                          → POST /api/config
                          → configStore (preprocessingConfig, modelConfig, deviceInfo) 저장

Step 3 [Training 화면]    실험명 입력 (또는 자동 생성)
                          → [학습 시작] 클릭 → POST /api/training/start
                          → WS /ws/training 연결 → Progress, Loss 차트, 로그 실시간 확인
                          → 완료 알림 + 소요 시간 확인
                          → history.json 기록

Step 4 [Experiments 화면] 실험 목록에서 완료 실험 선택
                          → 메트릭 카드, Confusion Matrix, ROC Curve, Score 분포 확인
                          → (선택) 다른 실험과 비교
                          → (선택) POST /api/experiments/{id}/save → 모델 저장

Step 5 [AnomalyMap 화면]  선택된 실험의 테스트셋 전체 추론
                          → POST /api/anomaly-map/{expId}/build → job 시작
                          → GET /api/anomaly-map/job/{jobId} 폴링 → 완료 대기
                          → Threshold 슬라이더 조정 (300ms debounce)
                          → TP/FP/TN/FN 이미지 그리드 확인
                          → CSV 또는 ZIP 내보내기
```

#### Vision Golden Path — 현장 검사 시작

```
Step 0 [Model Settings]   학습 완료 실험 목록 확인 (GET /api/models, 30초 폴링)
                          → 최적 모델 행 선택 → [모델 적용] 클릭
                          → POST /api/inspection/model
                          → GpuWarningBanner 표시 (gpu_warning 있는 경우)
                          → ModelStatusChip 초록으로 전환

Step 1 [Realtime]         수동 검사 또는 자동 검사 선택
                          수동: [수동 검사 (1개 검사)] 클릭
                               → POST /api/inspection/run
                               → polling GET /api/inspection/job/{id} (1초 간격)
                               → 이미지 3패널 + 판정 pill 갱신
                          자동: [▶ 자동 검사 (3초마다 1개)] 클릭
                               → WS /ws/inspection/auto 연결
                               → result 수신 시 이미지·판정 자동 갱신

Step 2 [불량 감지]        WS type: 'defect_stopped' 수신
                          → DefectPopup 모달 표시
                          → [✅ 확인 및 재개]: 새 WS 연결 + 자동 검사 재개
                          → [🛑 검사 종료]: 팝업 닫기 (자동 검사 유지 중지)

Step 3 [History]          [검사 이력] 탭 이동
                          → GET /api/inspection/records → 이력 테이블 확인
                          → KPI 카드 (총검사/양품/불량/불량률) 확인
                          → 통계 차트 (단위별 그룹화, 히스토그램·산점도)
                          → [CSV 내보내기] 클릭
```

---

### B.2 화면별 기능 요약

#### Explorer 화면 요약

| 화면 | 경로 | 핵심 입력 | 핵심 출력 | Store Write |
|------|------|-----------|-----------|-------------|
| Dataset | `/` | 로컬 경로, 제품명 | 폴더 트리, 이미지 수, 썸네일 | `datasetPath`, `productName`, `datasetMeta` |
| Config | `/config` | 전처리 방법, 배경 제거, 모델 종류, 파라미터 | 디바이스 정보, 설정 요약 | `preprocessingConfig`, `modelConfig`, `deviceInfo`, `queueItems` |
| Training | `/training` | 실험명, 학습 시작 버튼 | Progress, Loss 차트, 로그 | `trainingStatus`, `progress`, `lossHistory`, `logs` |
| Experiments | `/experiments` | 실험 선택, 비교 체크박스, 저장 경로 | 메트릭 카드, 차트, 비교 시각화 | `selectedExperimentId` |
| AnomalyMap | `/anomaly-map` | Threshold 슬라이더, 결함 유형 필터 | Triplet 이미지 그리드, TP/FP/TN/FN 통계 | `threshold` |

#### Vision 화면 요약

| 화면 | 경로 | 핵심 입력 | 핵심 출력 | Store Write |
|------|------|-----------|-----------|-------------|
| Realtime Inspection | `/` | 검사 버튼 (수동/자동/불량만/중지) | 이미지 3패널, 판정 pill, 불량 팝업 | `lastResult`, `imageStamp`, `isAutoRunning`, `defectStopped` |
| History | `/history` | 판정 필터, 단위 선택 | 이력 테이블, KPI 카드, 통계 차트 | — (read only) |
| Model Settings | `/settings` | 실험 선택, 소스 경로 | 모델 목록 테이블, 적용 버튼 | `activeModel`, `gpuWarning` |

---

### B.3 화면 진입 제한 (Guard)

#### Explorer — Store 기반 진입 차단

| 화면 | 차단 조건 | 동작 |
|------|-----------|------|
| Config | `datasetPath === null` | 안내 메시지 표시 (데이터셋 미설정) |
| Training | `preprocessingConfig === null \|\| modelConfig === null` | 안내 메시지 표시 (설정 미완료) |
| Experiments | 완료 실험 없음 | 안내 메시지 표시 |
| AnomalyMap | `selectedExperimentId === null` | 안내 메시지 표시 (실험 미선택) |

#### Vision — NoModelGuard 오버레이

| 화면 | Guard | 동작 |
|------|-------|------|
| Realtime Inspection | `activeModel === null` | children 렌더링 + 그 위에 반투명 모달 오버레이. "설정 페이지로 이동" 버튼 → navigate('/settings') |
| History | `activeModel === null` | 동일 |
| Model Settings | — | Guard 없음 (항상 접근 가능) |

> **NoModelGuard 특성 (v2.0)**: Streamlit의 렌더링 중단 방식(return)과 달리 children을 항상 렌더링 후 고정 오버레이를 그 위에 표시한다.

---

### B.4 Edge Cases

| # | 상황 | 처리 방식 |
|---|------|-----------|
| EC-01 | 입력 경로 미존재 | POST /api/dataset/validate 400 → Explorer Dataset 화면 에러 표시 + datasetPath = null 유지 |
| EC-02 | `train/good/`에 이미지 0개 | 400 응답 + 에러 메시지 → Config/Training 화면 진입 차단 |
| EC-03 | 지원 포맷 외 파일 포함 | `has_invalid_files=true` 안내 후 지원 포맷만 카운트 사용 |
| EC-04 | Grayscale 이미지 | `channels == 1` 감지 → `GRAYSCALE_DETECT` 안내, 학습 시 자동 RGB 변환 |
| EC-05 | `ground_truth/` 없음 | GT를 빈 마스크(전체 0)로 처리, AnomalyMap Triplet은 정상 렌더링 |
| EC-06 | CUDA 미사용 환경 | CPU fallback, `device_info.device = "cpu"` 표시, 학습 진행 허용 |
| EC-07 | 학습 중 [학습 중지] 클릭 | POST /api/training/stop → 즉시 중단, status="중단" 기록 |
| EC-08 | 학습 중 [⏸ 일시정지] 클릭 | POST /api/training/pause → 체크포인트 저장, WS type:"paused" 수신 |
| EC-09 | 체크포인트 재시작 시 충돌 | 새 experiment_id 자동 생성, 안내 메시지 표시 |
| EC-10 | 모델 저장 중 디스크 공간 부족 | 여유 < 500MB 시 에러 응답 + 저장 중단 |
| EC-11 | AnomalyMap GT 크기 불일치 | Anomaly Map 크기 기준으로 GT 리사이즈 후 표시 |
| EC-12 | history.json 파싱 오류 | 빈 목록 반환, 파일 덮어쓰기 금지 |
| EC-13 | Vision: activeModel = null 시 검사 클릭 | NoModelGuard 오버레이 표시 (버튼 클릭 자체는 차단되지 않지만 API 호출 전 상태로 진입 불가) |
| EC-14 | Vision: WS 연결 오류 | setAutoRunning(false), 사용자가 자동 검사 버튼 재클릭으로 재연결 (자동 재연결 없음) |
| EC-15 | Vision: 불량만 검사 시 불량 이미지 없음 | 에러 대신 amber warning toast (3초 자동 해제) |
| EC-16 | OK/NG 형식에서 단일 폴더만 존재 | 400 에러 + datasetPath = null 유지 |
| EC-17 | OK/NG OK 이미지 < 5개 | 경고 표시 (학습 데이터 부족 안내), 학습 진행 허용 |

---

### B.5 실패 시나리오

| # | 시나리오 | 감지 방법 | 복구 방법 |
|---|----------|-----------|-----------|
| F-01 | 학습 중 GPU OOM | `torch.cuda.OutOfMemoryError` → WS type:"error" 전송 | Explorer Training 화면 에러 표시 + batch_size 축소 권장 |
| F-02 | Anomalib 모델 초기화 실패 | `Exception` in model_factory.py → 500 응답 | Training 화면 에러 + 로그 파일 기록 |
| F-03 | history.json 쓰기 실패 | `OSError` 캐치 → 500 응답 | 에러 표시 (메모리 내 실험 상태는 유지) |
| F-04 | 학습 완료 후 메트릭 계산 실패 | `Exception` in metrics.py | status="중단" 처리, metrics=null, 에러 로그 |
| F-05 | Vision 모델 로드 실패 (model_path 손상) | `FileNotFoundError` → 500 응답 | Vision Model Settings에서 다른 모델 선택 안내 |
| F-06 | Vision 수동 검사 타임아웃 (120초) | polling 120s 초과 | InspectionControls 에러 텍스트 표시 |
| F-07 | AnomalyMap job 실패 | GET /api/anomaly-map/job/{id} status="failed" | BuildSection 에러 표시 + 재시도 버튼 |

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
| inspection_record | 00_Global_Context 1.10절 |
| inspectionStore 스키마 | [15_UI_UX_Design_Vision.md](./15_UI_UX_Design_Vision.md) 4절 |

---

### C.2 시스템 구성 참조

시스템 아키텍처(3개 레포 컴포넌트 다이어그램, 디렉토리 구조, 화면별 데이터 흐름)는 [00_Global_Context_Document.md 5절](./00_Global_Context_Document.md#5-system-architecture)에 정의되어 있으며, [04_System_Architecture.md](./04_System_Architecture.md)에서 상세 확장된다.

---

### C.3 상태 흐름 요약

#### Explorer Zustand Store 흐름

```
Dataset 화면  → datasetStore (datasetPath, productName, datasetMeta)
  ↓
Config 화면   → configStore (preprocessingConfig, modelConfig, deviceInfo, queueItems)
  ↓
Training 화면 → trainingStore (status, progress, lossHistory, logs)
  ↓
Experiments   → experimentsStore (selectedExperimentId)
  ↓
AnomalyMap    → anomalyMapStore (threshold)
```

#### Vision Zustand Store 흐름

```
Model Settings → inspectionStore (activeModel, gpuWarning)
  ↓
Realtime       → inspectionStore (lastResult, imageStamp, isAutoRunning, defectStopped, reshuffledToast)
  ↓
History        → inspectionStore (clearHistory) ← allRecords는 useInspectionRecords 로컬
```

각 키의 타입·제약조건은 [00_Global_Context_Document.md 3절](./00_Global_Context_Document.md#3-global-state-contract-standard)을 따른다.

---

## D. API Contracts

이 시스템은 FastAPI REST API + WebSocket 서버(smart-qc-dashboard, port 8000)를 포함한다.

전체 API 명세는 [06_API_Specification.md](./06_API_Specification.md)에서 다룬다.

**Explorer API 그룹 요약:**

| 그룹 | 주요 엔드포인트 |
|------|----------------|
| 데이터셋 | POST /api/dataset/validate, GET /api/dataset/thumbnail/{class_name} |
| 설정·큐 | GET/POST /api/config, POST /api/config/preview, GET/POST/DELETE /api/queue/{id} |
| 학습 | POST /api/training/start·resume·pause·unpause·stop, GET /api/training/checkpoints, POST /api/training/batch/start·skip |
| 학습 WS | WS /ws/training |
| 실험 | GET /api/experiments, POST /api/experiments/{id}/save, DELETE /api/experiments/{id} |
| AnomalyMap | GET /api/anomaly-map/{expId}/status, POST /api/anomaly-map/{expId}/build, GET /api/anomaly-map/job/{jobId}, GET /api/anomaly-map/{expId}/images, GET /api/anomaly-map/{expId}/image/{path}/triplet, GET /api/anomaly-map/{expId}/export/csv, POST /api/anomaly-map/{expId}/export/zip, GET /api/anomaly-map/zip/{jobId} |

**Vision API 그룹 요약:**

| 그룹 | 주요 엔드포인트 |
|------|----------------|
| 모델 관리 | GET /api/models, GET/POST /api/inspection/model, PATCH /api/inspection/source-path |
| 검사 실행 | POST /api/inspection/run, GET /api/inspection/job/{id} |
| 이미지 | GET /api/inspection/image/last?t={stamp}, GET /api/inspection/anomaly-map/last?t={stamp}, GET /api/inspection/overlay/last?t={stamp} |
| 이력 | GET/DELETE /api/inspection/records, GET /api/inspection/records/csv |
| 자동 검사 WS | WS /ws/inspection/auto |

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

---

### E.2 Anomalib 연동 방침

- Anomalib **≥ 1.0.0 (v1 API)** 기준으로 구현한다.
- `model_factory.py`에서 EfficientAD Engine, PatchCore Engine을 각각 래퍼 함수로 캡슐화한다.
- Anomalib 내부 DataModule 대신 커스텀 DataLoader를 사용하여 Config 화면 전처리 파이프라인과 연동한다.
- Vision의 `run_inference()`는 Dashboard ML 레이어를 통해 동일 Engine을 호출한다.
- 상세 구현 명세는 [08_AI_ML_Integration.md](./08_AI_ML_Integration.md)에서 다룬다.

---

### E.3 전처리 파이프라인 위치

전처리(Homomorphic/HE/CLAHE)는 Anomalib DataModule이 아닌 **`utils/image_utils.py`에서 구현**하여 Explorer Config 화면 미리보기와 Training 학습 루프가 동일 코드를 공유한다.

```
이미지 로드 (PIL)
  → 채널 변환: Grayscale → RGB (if channels == 1)
  → 배경 제거 (none / SAM2)
  → 전처리 필터 적용 (None / Homomorphic / HE / CLAHE)
  → Resize + Padding (image_size × image_size, 검정 0)
  → 정규화 (ImageNet 또는 커스텀 mean/std)
  → torch.Tensor 변환
```

> **v2.0 추가**: `background_method` 필드 (none/sam2)가 파이프라인에 추가됨. Config 화면 Tab2에서 선택.

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체를 상속한다.

이 문서에서 추가로 명시하는 항목:

| 항목 | 요구사항 |
|------|----------|
| **브라우저 호환성** | Chrome ≥ 110, Firefox ≥ 110, Edge ≥ 110. Safari는 미지원 (WebSocket 안정성). |
| **화면 해상도** | 최소 1280×800. 반응형 레이아웃은 Tailwind CSS 유틸리티 범위 내. |
| **Explorer 기술 스택** | React 19, Vite 8, TypeScript, Zustand v5, react-router-dom v7, Recharts v2, Tailwind CSS v4 |
| **Vision 기술 스택** | React, Vite, TypeScript, Zustand, react-router-dom v6, Recharts, Tailwind CSS |
| **Dashboard 기술 스택** | Python 3.12, FastAPI, Uvicorn, Anomalib ≥ 1.0, PyTorch ≥ 2.1, CUDA 12.4 |
| **API 비동기 안전성** | FastAPI 엔드포인트는 학습 백그라운드 스레드와 asyncio 이벤트 루프 간 통신 시 thread-safe Queue 사용 |
| **WebSocket 단일 연결** | Explorer WS /ws/training: 화면당 1개. Vision WS /ws/inspection/auto: Tab1 마운트 시 1개, 언마운트 시 cleanup (stop 신호 전송 후 close) |
| **이미지 cache-bust** | Vision 이미지 URL에 `?t={imageStamp}` 파라미터 필수. imageStamp = setLastResult() 호출 시 Date.now() |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards)을 전체 상속한다.

이 문서(Product Overview)에서 추가로 명시하는 제품 수준 관측 항목:

| 관측 항목 | 목적 | 구현 위치 |
|-----------|------|-----------|
| 실험 완료 시 AUC 기록 | 성공 지표(AUC ≥ 0.95) 달성 여부 추적 | `history.json → metrics.auc` |
| 학습 소요 시간 기록 | 성능 목표(EfficientAD ≤ 1200s, PatchCore ≤ 600s) 달성 여부 추적 | `history.json → duration_seconds` |
| 모델 타입별 실험 수 | EfficientAD vs PatchCore 사용 빈도 | `history.json` 집계 |
| Vision 검사 이력 | 불량률 추이, 검사 건수 | GET /api/inspection/records → KPI 카드 |
| GPU 경고 | 모델 적용 시 VRAM 부족 등 경고 | `gpu_warning` 필드 → GpuWarningBanner |

---

## H. QA & Validation

### H.1 제품 수준 인수 기준 (Acceptance Criteria)

아래 기준을 모두 통과하면 MVP v2.0을 완료로 판정한다.

| # | 기준 | 검증 방법 |
|---|------|-----------|
| AC-01 | Explorer Golden Path (B.1절) Dataset→Config→Training→Experiments→AnomalyMap 오류 없이 완주 | E2E 수동 테스트 |
| AC-02 | Vision Golden Path (B.1절) Model Settings→Realtime→History 오류 없이 완주 | E2E 수동 테스트 |
| AC-03 | EfficientAD-medium 학습 완료 시 `duration_seconds ≤ 1200` | g4dn.xlarge 실측 |
| AC-04 | PatchCore (coreset 10%) 학습 완료 시 `duration_seconds ≤ 600` | g4dn.xlarge 실측 |
| AC-05 | MVTec AD Screw 기준 AUC ≥ 0.95 달성 실험 존재 | history.json 확인 |
| AC-06 | 동일 seed+파라미터로 2회 실행 시 AUC 소수점 4자리 일치 | 재현성 테스트 |
| AC-07 | Docker 이미지 빌드 성공 + GPU 컨테이너 정상 실행 | `docker run --gpus all` |
| AC-08 | Explorer 화면 전환 응답 < 1초 (학습 중 포함) | 수동 측정 |
| AC-09 | [학습 중지] 클릭 후 history.json에 status="중단" 기록 확인 | 파일 내용 검증 |
| AC-10 | 모델 저장 후 `./models/{exp_id}/` 에 `.pth`와 `configs.yaml` 존재 | 파일 존재 검증 |
| AC-11 | [⏸ 일시정지] 클릭 후 `./models/checkpoints/{exp_id}_step{N}.ckpt` 파일 존재 | 파일 존재 검증 |
| AC-12 | 체크포인트에서 재시작 후 중단 지점 이후 step부터 학습 로그 출력 | 수동 테스트 |
| AC-13 | Vision 수동 검사: 판정 결과 표시까지 ≤ 3초 | 수동 측정 |
| AC-14 | Vision activeModel = null 상태에서 Realtime/History 화면 진입 시 NoModelGuard 오버레이 표시 | 수동 테스트 |
| AC-15 | Vision 불량 감지 시 DefectPopup 표시 + "✅ 확인 및 재개" 클릭 후 자동 검사 재개 | 수동 테스트 |
| AC-16 | Vision CSV 내보내기 버튼 클릭 시 브라우저 파일 다운로드 트리거 | 수동 테스트 |

---

### H.2 Given-When-Then 시나리오 (제품 수준)

#### TC-01: Explorer 정상 실험 완료 플로우

```
Given:  MVTec AD Screw 데이터셋이 /app/dataset/screw 에 올바른 구조로 존재한다
        FastAPI 서버(port 8000)가 실행 중이다
        CUDA 디바이스가 사용 가능하다
When:   [Dataset] /app/dataset/screw 경로 입력 → POST /api/dataset/validate → 통과
        [Config] 전처리=CLAHE(clipLimit=2.0), image_size=256, 모델=EfficientAD-medium,
                 train_steps=70000, seed=42 → POST /api/config
        [Training] [학습 시작] 클릭 → POST /api/training/start → WS 모니터링 → 완료 대기
        [Experiments] 완료 실험 선택 → 메트릭 확인
        [AnomalyMap] build → job 완료 → Threshold 슬라이더 조정
Then:   history.json에 status="completed" 레코드 존재
        metrics.auc >= 0.95
        duration_seconds <= 1200
        ./models/{exp_id}/model_state_dict.pth 파일 존재
        AnomalyMap 이미지 그리드 정상 렌더링
```

#### TC-02: Vision 자동 검사 + 불량 감지 플로우

```
Given:  FastAPI 서버가 실행 중이다
        history.json에 completed 실험이 존재한다
When:   [Model Settings] 완료 실험 선택 → [모델 적용] 클릭
        → POST /api/inspection/model → activeModel 설정
        [Realtime] [▶ 자동 검사] 클릭 → WS /ws/inspection/auto 연결
        서버에서 type: 'defect_stopped' 메시지 전송
Then:   DefectPopup 모달 표시 (bg-black/55 오버레이)
        [✅ 확인 및 재개] 클릭 → setDefectStopped(false) + 새 WS 연결
        isAutoRunning === true로 복귀
        이미지 패널 갱신 계속됨
```

#### TC-03: Explorer 잘못된 경로 입력

```
Given:  존재하지 않는 경로 "/nonexistent/path" 를 Dataset 화면에 입력한다
When:   POST /api/dataset/validate 호출
Then:   400 응답 → Dataset 화면 에러 표시
        datasetPath === null 유지
        Config 화면에서 "데이터셋 미설정" 안내 메시지
        Training 화면 진입 차단 (preprocessingConfig === null)
```

#### TC-04: Vision 모델 미선택 상태 진입

```
Given:  앱 최초 진입 (activeModel === null)
When:   사용자가 "/" 경로(Realtime Inspection)에 접근
Then:   Tab1Realtime 컴포넌트가 렌더링됨 (NoModelGuard는 차단이 아닌 오버레이)
        반투명 모달 오버레이 표시 ("모델 미선택" + "설정 페이지로 이동" 버튼)
        "설정 페이지로 이동" 클릭 → navigate('/settings')
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
15_UI_UX_Design_Explorer (완료)
15_UI_UX_Design_Vision (완료)
```

### I.2 레포별 병렬 개발 전략

```
[smart-qc-dashboard] FastAPI 라우터 + ML 레이어 구현
       │
       ├── Explorer API (POST /api/dataset/*, /api/config/*, /api/training/*, ...)
       │         ↓ 동시 가능
       └── Vision API (/api/inspection/*, /api/models, /ws/inspection/auto)
                 ↓
[smart-qc-explorer] Explorer React UI 구현
[smart-qc-vision]   Vision React UI 구현
  (Dashboard API 준비되는 화면부터 순차 연동)
```

**레포 간 의존성 핵심 계약:**

| 계약 항목 | Explorer 의존 | Vision 의존 |
|-----------|-------------|------------|
| history.json 스키마 | 쓰기 전용 (save_history) | 읽기 전용 (load_history) |
| experiment_id 형식 | 생성 주체 | 참조 주체 |
| `/api/models` 응답 | — | activeModel 적용 전 목록 조회 |
| 이미지 파일 경로 | dataset_path 설정 | source_path PATCH 후 사용 |

---

---

# v1.x 참고 (Streamlit 기반 — 삭제 금지)

> 이하 내용은 v1.x Streamlit 구현 기준입니다. v2.0 React/FastAPI UI가 공식 구현입니다.
> 설계 결정 참고, 탭 기반 플로우 이해를 위해 보존합니다.

---

### v1.x A.1 제품 정의 (Streamlit 단일 앱)

하나의 Streamlit 앱 안에 **두 개의 대시보드**를 포함한다.

| 플랫폼 | 대상 사용자 | 핵심 기능 |
|----------|------------|---------|
| **모델 탐색 플랫폼** | AI/ML 엔지니어, 데이터 분석가 | 코드 작성 없이 EfficientAD/PatchCore 학습·평가·비교·자산화 |
| **비전검사 플랫폼** | 현장 작업자, 품질 관리자 | 검증된 모델로 부품 추론·판정·이력 관리 |

사이드바의 전환 버튼으로 두 플랫폼을 전환한다. 두 플랫폼은 `session_state` 네임스페이스(`insp_` 접두사)로 격리된다.

---

### v1.x B.1 Golden Path (Streamlit 탭 기반)

**모델 탐색 플랫폼:**

```
Step 1 [탭1]  데이터 폴더 경로 입력 → MVTec AD 구조 검증
              → session_state.dataset_path, dataset_meta 저장
Step 2 [탭2]  전처리 방식 선택 → 파라미터 설정 → 전후 미리보기 확인
              → 모델 선택 → 파라미터 설정 → Threshold 방식·값 설정
              → session_state.preprocessing_config, model_config, device_info 저장
Step 3 [탭3]  실험명 입력 → [학습 시작] 클릭
              → Progress Bar, Loss 곡선, 로그 실시간 확인
Step 4 [탭4]  실험 목록에서 완료 실험 선택 → 지표 확인 → (선택) 저장
Step 5 [탭5]  이미지 선택 → 3분할 시각화 확인 → Threshold 슬라이더 조정 → PNG 저장
```

**비전검사 대시보드:**

```
Step 0 [사이드바]  [🏭 비전검사 대시보드] 클릭 → active_dashboard = "inspection"
Step 1 [탭3]       완료 실험 목록 → 최적 모델 [적용] → insp_active_model 설정
Step 2 [탭1]       [수동 검사 (1개 검사)] 클릭 → 4열 결과 확인
                   [자동 검사 (3초마다 1개)] 클릭 → 자동 루프 시작
Step 3 [팝업]      불량 감지 → st.dialog 팝업 → [확인 및 재개] 또는 [검사 종료]
Step 4 [탭2]       이력 테이블, KPI 카드 확인 → [CSV 내보내기]
```

---

### v1.x B.2 탭별 기능 요약 (Streamlit)

**모델 탐색 대시보드:**

| 탭 | 탭명 | 핵심 입력 | 핵심 출력 | session_state Write |
|----|------|-----------|-----------|---------------------|
| 탭1 | 데이터 폴더 구조 | 로컬 경로 텍스트 | 폴더 트리, 이미지 수, 썸네일 | `dataset_path`, `dataset_meta` |
| 탭2 | 전처리 및 모델 설정 | 전처리 방식, 파라미터, 모델 종류 | 전후 미리보기, 설정 요약, 디바이스 | `preprocessing_config`, `model_config`, `device_info` |
| 탭3 | 학습 시작 + 학습 로그 | 실험명, 학습 시작 버튼 | Progress Bar, Loss 곡선, 로그 | `experiments[exp_id]`, `current_run_status` |
| 탭4 | 실험 히스토리 + 결과 + 저장 | 실험 선택, 저장 경로 | 지표 카드, 차트, 비교 시각화 | `selected_experiment_id` |
| 탭5 | 이상 영역 시각화 | 이미지 선택, Threshold 슬라이더 | 3분할 시각화, PNG | `anomaly_map_threshold` |

---

### v1.x B.3 사이드바 구성 (Streamlit v1.2)

```
┌─────────────────────────────────┐
│        Smart QC Platform        │
│  [ 🔬 모델 탐색 플랫폼         ] │  ← 버튼 (active_dashboard == "explorer" 시 강조)
│  [ 🏭 비전검사 플랫폼          ] │  ← 버튼 (active_dashboard == "inspection" 시 강조)
└─────────────────────────────────┘
```

- `active_dashboard == "explorer"`: 모델 탐색 플랫폼(탭1~5) 렌더링
- `active_dashboard == "inspection"`: 비전검사 플랫폼(탭1~3) 렌더링
- 버튼 2개 외 사이드바 추가 콘텐츠 없음 (v1.1에서 데이터셋·디바이스 표시 완전 제거)

---

### v1.x B.4 탭 진입 차단 조건 (Streamlit guard)

| 탭 | 차단 조건 | 표시 메시지 키 |
|----|-----------|---------------|
| 탭2 | `session_state.dataset_path is None` | `MSG["NO_DATASET"]` |
| 탭3 | `session_state.model_config is None` | `MSG["NO_MODEL_CONFIG"]` |
| 탭4 | `len(session_state.experiments) == 0` | `MSG["NO_EXPERIMENTS"]` |
| 탭5 | `session_state.selected_experiment_id is None` | `MSG["NO_SELECTED_EXP"]` |

---

### v1.x A.4 MVP 범위 — 비전검사 대시보드 (Streamlit)

**IN SCOPE:**

| 기능 영역 | 포함 항목 |
|-----------|-----------|
| 실시간 검사 | 수동 검사(1개 검사) / 자동 검사(3초마다 1개) 버튼, 판정결과·원본·Anomaly Map 4열 표시 |
| 불량 알림 | 불량 감지 시 st.dialog 팝업, 자동 검사 자동 중지, 확인 버튼으로 팝업 해제 |
| 검사 이력 | 세션 기반 이력 테이블(5컬럼), CSV 내보내기, KPI 카드(총 검사·양품·불량·불량률), 통계 차트 3분할(단위별 그룹화) |
| 모델 교체 | history.json의 완료 실험 목록, F1 정렬, 교체 시 이력 초기화 |

**OUT OF SCOPE:**

| 제외 항목 | 제외 사유 |
|-----------|-----------|
| 이력 영속 저장 (DB/파일) | 세션 기반 단순화. 앱 재시작 시 초기화 |
| 접근 권한 분리 | 단일 로컬 사용자 환경 |
| 성능 지표 카드 (F1/Accuracy) | 현장 실제 사용 — 정답 레이블 미공개 원칙 |
| 실시간 카메라 스트리밍 | test 데이터셋 샘플링으로 대체 |

---

### v1.x A.5 성공 지표 (Streamlit 기준)

| 지표 | 목표값 | 측정 방법 | 측정 시점 |
|------|--------|-----------|-----------|
| 단일 실험 사이클 | 설정→학습→평가 ≤ 30분 | 탭1~탭5 E2E 소요 시간 | 통합 테스트 |
| UI 응답성 | 탭 전환 < 1초, 학습 중 UI 블로킹 없음 | 수동 테스트 | 통합 테스트 |
| 추론 지연 (비전검사) | 이미지 1장 추론 → 판정 결과 표시 ≤ 3초 | 수동 측정 | 통합 테스트 |
| 자동 검사 타이밍 | 3초 간격 오차 ≤ 0.5초 | 수동 측정 | 통합 테스트 |

---

*다음 문서*: [02_User_Personas_and_Use_Cases.md](./02_User_Personas_and_Use_Cases.md)
