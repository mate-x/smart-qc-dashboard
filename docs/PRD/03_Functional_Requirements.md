# 03. Functional Requirements

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [01_Product_Overview.md](./01_Product_Overview.md), [02_User_Personas_and_Use_Cases.md](./02_User_Personas_and_Use_Cases.md)
> **버전**: v2.0
> **작성일**: 2026-05-08
> **수정일**: 2026-06-11 — v2.0: React UI 공식화; 화면명 전환(탭N→React 화면); 구현 상태 태그 전면 추가; 🆕 신규 FR 추가
> **후속 문서**: [04_System_Architecture.md](./04_System_Architecture.md)

### 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|---------|
| v1.0 | 2026-05-08 | 최초 작성 — Streamlit 탭 기반 FR |
| v1.1 | 2026-05-15 | 비전검사 대시보드(INSP) 섹션 추가 |
| v1.2 | 2026-05-29 | 플랫폼 명칭 변경, 학습 단계 시각화+ETA, 실험 대기열+일괄 학습, 검사 실시간 차트 추가 |
| v2.0 | 2026-06-11 | React UI 공식 UI로 확정; "탭N" 표현 → React 화면명으로 전환; 전 FR 항목에 구현 상태 태그 추가; README 기반 🆕 신규 FR 추가; 구현 과정에서 요구사항이 구체화된 내용 반영 |

---

## 문서 사용 규칙

- **FR ID 체계**: `FR-{영역}-{순번}` — 예: `FR-CMN-01`, `FR-T1-03`
- **우선순위**: `M` = Must Have (MVP 필수) / `S` = Should Have (MVP 권장) / `N` = Nice to Have (MVP 제외)
- **구현 상태 태그** (v2.0 추가):
  - ✅ 구현 완료 — Explorer 또는 Vision React UI에서 동작 확인
  - 🔄 부분 구현 — 일부만 구현되거나 Streamlit에만 있음
  - ❌ 미구현 — PRD에 있으나 코드에 없음
  - 🆕 코드에만 존재 — PRD에 없는 신규 기능 (README 기반)
- **v1.x 참고**: Streamlit 구현 세부 내용은 `(v1.x 구현: ...)` 형식으로 표기
- **유효성 검증**: 범위·타입·제약조건을 숫자로 명시한다

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Detailed Specification](#b-detailed-specification)
  - [B.1 공통 기능 (CMN)](#b1-공통-기능-cmn)
  - [B.2 Dataset 화면 (Explorer)](#b2-dataset-화면-explorer--구-탭1)
  - [B.3 Config 화면 (Explorer)](#b3-config-화면-explorer--구-탭2)
  - [B.4 Training 화면 (Explorer)](#b4-training-화면-explorer--구-탭3)
  - [B.5 Experiments 화면 (Explorer)](#b5-experiments-화면-explorer--구-탭4)
  - [B.6 AnomalyMap 화면 (Explorer)](#b6-anomalymap-화면-explorer--구-탭5)
  - [B.7 Vision 검사 플랫폼 (INSP)](#b7-vision-검사-플랫폼-insp--v11-추가)
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

Dataset·Config·Training·Experiments·AnomalyMap(Explorer) 및 Realtime Inspection·History·Model Settings(Vision) 전체 화면 기능을 FR ID 단위로 분해하여, 각 기능의 동작 조건·UI 컴포넌트·유효성 검증·입출력을 구현 가능한 수준으로 명세한다. v2.0부터 공식 UI는 React(Explorer/Vision)이며, FastAPI REST API + WebSocket을 통해 백엔드와 통신한다.

### A.2 FR 전체 목록 요약

#### Explorer 대시보드 (smart-qc-explorer)

| 영역 | 화면명 | FR 수 (M) | FR 수 (S) | 합계 |
|------|--------|-----------|-----------|------|
| 공통 (CMN) | — | 4 | 0 | 4 |
| Dataset 화면 | `/` | 6 | 4 | 10 |
| Config 화면 | `/config` | 12 | 3 | 15 |
| Training 화면 | `/training` | 11 | 4 | 15 |
| Experiments 화면 | `/experiments` | 7 | 4 | 11 |
| AnomalyMap 화면 | `/anomaly-map` | 6 | 5 | 11 |
| **합계** | | **46** | **20** | **66** |

#### Vision 검사 플랫폼 (smart-qc-vision)

| 영역 | 화면명 | FR 수 (M) | FR 수 (S) | 합계 |
|------|--------|-----------|-----------|------|
| 검사 공통 (INSP-CMN) | — | 2 | 0 | 2 |
| Realtime Inspection 화면 | `/` | 7 | 0 | 7 |
| History 화면 | `/history` | 5 | 3 | 8 |
| Model Settings 화면 | `/models` | 4 | 0 | 4 |
| **소계** | | **18** | **3** | **21** |

---

## B. Detailed Specification

---

### B.1 공통 기능 (CMN)

---

#### FR-CMN-01 (M): 앱 진입점 초기화

> **구현 상태**: 🔄 부분 구현 — React 진입점(App.tsx + Zustand stores)으로 대체됨; Streamlit session_state 방식은 v1.x

| 항목 | 내용 |
|------|------|
| **설명** | 앱 최초 진입 시 전역 상태를 초기화하고 라우팅 구조를 구성한다 |
| **v2.0 구현** | Explorer: `main.tsx` → `BrowserRouter` + Zustand store 자동 초기화 (datasetStore, configStore, trainingStore, experimentsStore, anomalyMapStore) <br> Vision: `main.tsx` → `BrowserRouter` + inspectionStore 자동 초기화 |
| **라우팅** | Explorer: `/`(Dataset) / `/config`(Config) / `/training`(Training) / `/experiments`(Experiments) / `/anomaly-map`(AnomalyMap) <br> Vision: `/`(Realtime Inspection) / `/history`(History) / `/models`(Model Settings) |
| **v1.x 구현** | `app.py`의 `init_session_state()` 호출 + `st.set_page_config()` + `st.tabs()` 5탭 렌더링 |

---

#### FR-CMN-02 (M): 사이드바 상시 정보 렌더링

> **구현 상태**: 🔄 부분 구현 — Explorer: `Sidebar.tsx` 컴포넌트 존재 (전체 상태 요약); Vision: 사이드바 없음 [확인 필요]

| 항목 | 내용 |
|------|------|
| **설명** | 데이터셋 정보, 디바이스 정보, 현재 설정 요약을 상시 표시한다 |
| **v2.0 구현** | Explorer: `components/layout/Sidebar.tsx` — datasetStore·configStore 기반으로 정보 렌더링 |
| **표시 항목** | 데이터셋 경로, 학습/테스트 이미지 수, 디바이스(CUDA/CPU), 전처리 방식, 모델 타입 |
| **v1.x 구현** | `components/sidebar.py` — `st.sidebar.markdown()` 기반 3개 섹션 |

---

#### FR-CMN-03 (M): 화면 진입 Guard

> **구현 상태**: 🔄 부분 구현 — Explorer: configStore path 유무 기반 조건부 렌더링; Vision: `NoModelGuard` 컴포넌트 ✅

| 항목 | 내용 |
|------|------|
| **설명** | 선행 설정이 완료되지 않은 화면 진입 시 핵심 기능을 차단하고 안내 메시지를 표시한다 |
| **Explorer Guard 조건** | Config 화면: `datasetStore.datasetPath == null` → 경로 미설정 안내 <br> Training 화면: `configStore.preprocessingConfig == null` 또는 `modelConfig == null` → 설정 미완료 안내 |
| **Vision Guard** | `components/layout/NoModelGuard.tsx` — `inspectionStore.activeModel == null` 시 차단 (Realtime/History 화면) |
| **v1.x 구현** | 각 `tabs/tab{n}_*.py` 상단에 `if session_state.{key} is None: st.warning(); return` 패턴 |

---

#### FR-CMN-04 (M): 표준 안내 메시지

> **구현 상태**: 🔄 부분 구현 — React 컴포넌트 내 인라인 메시지; 중앙 MSG 딕셔너리 패턴 미확인 [확인 필요]

| 항목 | 내용 |
|------|------|
| **설명** | 모든 안내·오류 메시지를 일관된 형식으로 표시한다 |
| **v2.0 구현** | 각 컴포넌트 내 토스트/인라인 메시지로 처리; Axios 에러 인터셉터로 공통 에러 처리 |
| **v1.x 구현** | `utils/messages.py`의 `MSG` 딕셔너리 — 탭 코드에서 문자열 리터럴 직접 사용 금지 |

---

### B.2 Dataset 화면 (Explorer) — 구 탭1

---

#### FR-T1-01 (M): 데이터셋 경로 입력 및 즉시 검증

> **구현 상태**: ✅ 구현 완료 (Explorer Dataset 화면 — `POST /api/dataset/validate`)

| 항목 | 내용 |
|------|------|
| **설명** | Dataset 화면(Explorer)에서 사용자는 데이터셋 루트 경로를 입력하여 MVTec AD 또는 OK/NG 폴더 구조를 자동 감지·검증할 수 있어야 한다 |
| **API 연동** | `POST /api/dataset/validate` — `ValidateDatasetRequest { dataset_path }` → `ValidateDatasetResponse` |
| **검증 항목** | 1. 경로 존재 여부 → 없으면 `ERR_DATASET_NOT_FOUND` 에러 <br> 2. MVTec AD 형식(`train/good/` 존재) 자동 감지 <br> 3. OK/NG 형식(`detect_ok_ng_dirs()`) 자동 감지 <br> 4. 검증 통과 → datasetStore에 경로·메타 저장 |
| **출력** | 형식 감지 결과(MVTec AD / OK/NG) + 이미지 수 요약 표시 |
| **v1.x 구현** | `st.text_input()` + `st.button("경로 확인")` 조합 |

---

#### FR-T1-01b (M): 데이터셋 변경 시 하위 설정 초기화

> **구현 상태**: ✅ 구현 완료 (Explorer Dataset → Config 연동 — README: "탭1에서 경로 변경 시 configStore의 설정이 자동 초기화")

| 항목 | 내용 |
|------|------|
| **설명** | 새 경로가 기존 경로와 다를 때 configStore의 전처리/모델 설정을 자동 초기화한다 |
| **초기화 대상** | `configStore.preprocessingConfig = null`, `configStore.modelConfig = null`, `configStore.deviceInfo = null` |
| **조건** | `newPath !== datasetStore.datasetPath` 일 때만 초기화 |

---

#### FR-T1-02 (M): dataset_meta 구성

> **구현 상태**: ✅ 구현 완료 (`ValidateDatasetResponse` 스키마로 반환)

| 항목 | 내용 |
|------|------|
| **설명** | 검증 통과한 경로에서 `dataset_format`에 따라 메타 정보를 구성하여 datasetStore에 저장한다 |
| **MVTec 구성** | train/good 이미지 수, test 클래스별 이미지 수, ground_truth GT 마스크 수, 채널, 포맷 |
| **OK/NG 구성** | OK 80% → train, OK 20% → test(good), NG 전체 → test(ng) |
| **채널 감지** | 이미지 첫 번째 파일 모드 분석 → `channels: 1`(그레이스케일) or `3`(RGB) |

---

#### FR-T1-02b (M): OK/NG 형식 안내 배너

> **구현 상태**: 🔄 부분 구현 — OK/NG 형식 감지는 구현됨; 분할 비율 배너 표시 여부 [확인 필요]

| 항목 | 내용 |
|------|------|
| **설명** | OK/NG 형식으로 로드된 경우 분할 비율(80/20%) 및 사용 방식을 안내한다 |
| **표시 조건** | `datasetMeta.datasetFormat == "oking"` |

---

#### FR-T1-03 (M): 폴더 트리 시각화

> **구현 상태**: ✅ 구현 완료 (Explorer Dataset 화면 — README: "메타 정보 표시: 폴더 트리")

| 항목 | 내용 |
|------|------|
| **설명** | Dataset 화면(Explorer)에서 데이터셋 폴더 구조를 계층형으로 표시한다 |
| **MVTec 렌더링** | 최대 3단계 (`train/good/`, `test/{class}/`, `ground_truth/{class}/`) |
| **OK/NG 렌더링** | `OK/ (N장 전체)` → 학습/테스트 자동 분할 안내 / `NG/ (N장)` → 테스트(불량) |

---

#### FR-T1-04 (M): 클래스별 이미지 수 테이블

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/tab1/ClassCountTable.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Dataset 화면(Explorer)에서 학습/테스트/GT 마스크 별 클래스별 이미지 수를 테이블로 표시한다 |
| **테이블 컬럼** | 클래스명 / 학습(train) 수 / 테스트(test) 수 / GT 마스크 수 |
| **합계 행** | 각 열의 합계 |

---

#### FR-T1-05 (M): 대표 썸네일 렌더링

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/tab1/ThumbnailGrid.tsx` — README: "MVTec AD는 3열, OK/NG는 2열 썸네일 그리드")

| 항목 | 내용 |
|------|------|
| **설명** | Dataset 화면(Explorer)에서 각 클래스의 대표 이미지를 썸네일 그리드로 표시한다 |
| **API 연동** | `GET /api/dataset/thumbnail/{class_name}` |
| **MVTec 레이아웃** | 3열 그리드 |
| **OK/NG 레이아웃** | 2열 그리드 (OK / NG) |

---

#### FR-T1-06 (M): Grayscale 자동 감지 안내

> **구현 상태**: ✅ 구현 완료 (Explorer Dataset 화면 — README: "그레이스케일 감지")

| 항목 | 내용 |
|------|------|
| **설명** | `datasetMeta.channels == 1`이면 그레이스케일 감지 안내 메시지를 표시한다 |
| **표시 조건** | `datasetMeta.channels == 1` |

---

#### FR-T1-07 (S): 지원 포맷 외 파일 경고

> **구현 상태**: 🔄 부분 구현 [확인 필요] — 백엔드 검증 시 감지 가능, UI 표시 여부 미확인

| 항목 | 내용 |
|------|------|
| **설명** | 지원 포맷(jpg/jpeg/png/bmp) 외 파일이 존재하면 경고 메시지를 표시한다. 학습은 차단하지 않는다 |
| **표시 조건** | `datasetMeta.hasInvalidFiles == true` |

---

#### FR-T1-08 (S): 폴더 구조 오류 상세 안내

> **구현 상태**: 🔄 부분 구현 [확인 필요] — API 에러 메시지 반환 여부 미확인

| 항목 | 내용 |
|------|------|
| **설명** | 검증 실패 시 어떤 하위 폴더가 누락되었는지 구체적으로 표시한다 |
| **출력 예시** | "누락된 폴더: `train/good/` — MVTec AD 형식의 폴더 구조가 아닙니다." |

---

#### FR-T1-NEW-01 (M): 제품명(Product Name) 설정 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Explorer Dataset 화면 — `datasetStore.productName`)

| 항목 | 내용 |
|------|------|
| **설명** | Dataset 화면(Explorer)에서 사용자는 실험 기록에 사용할 검사 제품명을 입력할 수 있어야 한다 |
| **저장 위치** | `datasetStore.productName` → 학습 시 실험명 자동 생성 및 history.json에 반영 |
| **출처** | Explorer README "제품명 설정: 실험 기록에 사용할 검사 제품명 지정" |

---

#### FR-T1-NEW-02 (S): 배경 분리 이미지 존재 여부 표시 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Explorer Dataset 화면)

| 항목 | 내용 |
|------|------|
| **설명** | Dataset 화면(Explorer)에서 `background_clean/` 폴더 존재 여부를 메타 정보로 표시한다 |
| **표시 조건** | 데이터셋 경로 내 `background_clean/` 디렉토리 존재 시 배지로 표시 |
| **출처** | Explorer README "배경 분리 이미지(background_clean/) 존재 여부" |

---

### B.3 Config 화면 (Explorer) — 구 탭2

---

#### FR-T2-01 (M): 전처리 방식 선택

> **구현 상태**: ✅ 구현 완료 (Explorer Config 화면 — `components/config/PreprocessingForm.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 사용자는 이미지 전처리 방식을 선택할 수 있어야 하며, 선택된 방식은 학습과 추론 과정에 동일하게 적용되어야 한다 |
| **선택지** | `none` / `homomorphic` / `he` / `clahe` |
| **조건부 렌더링** | 선택된 방식의 파라미터 UI만 렌더링 (비선택 방식 DOM 미노출) |
| **v1.x 구현** | `st.radio("전처리 방식", options=["없음", "Homomorphic", "HE", "CLAHE"], horizontal=True)` |

---

#### FR-T2-02 (M): 전처리 파라미터 UI

> **구현 상태**: ✅ 구현 완료 (Explorer — `PreprocessingForm.tsx`)

| 방식 | 파라미터 | 범위 | 기본값 |
|------|----------|------|--------|
| **Homomorphic** | sigma | 0.1 ~ 50.0 | 10.0 |
| | gamma_H | 1.0 ~ 3.0 | 1.5 |
| | gamma_L | 0.1 ~ 1.0 | 0.5 |
| | normalize | bool | true |
| **HE** | (없음) | — | — |
| **CLAHE** | clip_limit | 0.1 ~ 40.0 | 2.0 |

---

#### FR-T2-03 (M): Resize + 정규화 설정

> **구현 상태**: ✅ 구현 완료 (Explorer — `PreprocessingForm.tsx`)

| 항목 | 내용 |
|------|------|
| **image_size** | 32의 배수, 32~1024, 기본값 256 |
| **resize_mode** | 고정값 "padding" (UI 노출 없음) |
| **정규화 방식** | ImageNet 고정값 또는 커스텀 mean/std 3채널 입력 |
| **커스텀 mean/std** | 커스텀 선택 시만 렌더링; 길이 != 3 시 에러 |

---

#### FR-T2-04 (M): 전처리 적용 전후 미리보기

> **구현 상태**: 🔄 부분 구현 — `POST /api/config/preview-image` 엔드포인트 존재; 전/후 비교 UI 구현 여부 [확인 필요]

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 파라미터 변경 시 샘플 이미지에 전처리를 적용한 전후 비교 이미지를 표시한다 |
| **API 연동** | `POST /api/config/preview-image` — 원본·처리 후 이미지 반환 |
| **샘플 이미지** | `train/good/`의 첫 번째 파일 |
| **v1.x 구현** | `st.columns(2)` 원본/처리 후 — 파라미터 변경 시 자동 rerun으로 갱신 |

---

#### FR-T2-05 (M): 설정 저장 및 YAML 관리

> **구현 상태**: ✅ 구현 완료 (Explorer Config 화면 — `POST /api/config`, `POST /api/config/yaml/save`, `POST /api/config/yaml/load`)

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 전처리 및 모델 설정을 저장하고 YAML 파일로 내보내거나 불러올 수 있어야 한다 |
| **설정 저장** | `POST /api/config` — 현재 전처리·모델 설정 저장 → configStore 갱신 |
| **YAML 저장** | `POST /api/config/yaml/save` — `configs.yaml`로 파일시스템 저장 |
| **YAML 불러오기** | `POST /api/config/yaml/load` — `configs.yaml`에서 설정 복원 |

---

#### FR-T2-06 (S): 비선택 파라미터 UI 완전 숨김

> **구현 상태**: ✅ 구현 완료 (React 조건부 렌더링으로 자연스럽게 구현)

| 항목 | 내용 |
|------|------|
| **설명** | 선택되지 않은 전처리 방식 및 모델 타입의 파라미터 UI를 DOM에서 완전히 제거한다 |
| **구현 방식** | React 조건부 렌더링 (`{method === "homomorphic" && <HomomorphicParams />}`) |

---

#### FR-T2-07 (S): Grayscale 자동 RGB 변환

> **구현 상태**: 🔄 부분 구현 [확인 필요] — 백엔드 전처리 파이프라인에서 처리될 것으로 추정

| 항목 | 내용 |
|------|------|
| **설명** | `channels == 1` 이면 전처리 파이프라인에서 자동으로 RGB 변환을 수행한다 |

---

#### FR-T2-08 (M): 모델 종류 선택

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/config/ModelConfigForm.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 사용자는 EfficientAD 또는 PatchCore 모델을 선택할 수 있어야 한다 |
| **선택지** | `efficientad` / `patchcore` |
| **조건부 렌더링** | 선택 모델 전용 파라미터 컴포넌트만 렌더링 |

---

#### FR-T2-09 (M): 공통 파라미터 UI

> **구현 상태**: ✅ 구현 완료 (Explorer — `ModelConfigForm.tsx`)

| 파라미터 | 범위 | 기본값 |
|----------|------|--------|
| batch_size | 1~128 | 16 |
| random_seed | 0~2147483647 | 42 |

---

#### FR-T2-10 (M): EfficientAD 전용 파라미터 UI

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/config/EfficientAdParams.tsx`)

| 파라미터 | 범위 | 기본값 |
|----------|------|--------|
| model_size | small / medium | medium |
| train_steps | 1~200000 | 70000 |
| optimizer | adam / adamw / sgd | adam |
| learning_rate | 1e-6~1e-1 | 0.0001 |
| weight_decay | 0.0~0.1 | 0.0001 |
| out_channels | 128 / 256 / 384 / 512 | 384 |
| padding | bool | false |
| ae_loss_weight (α) | 0.0~1.0 | 0.5 |

`ae_loss_weight(α)`: 학습 루프 내 `total = α * loss_ae + (1-α) * loss_st + loss_stae` 공식 적용

---

#### FR-T2-11 (M): EfficientAD 고급 설정

> **구현 상태**: 🔄 부분 구현 [확인 필요] — 고급 설정 expander 존재 여부 미확인

| 파라미터 | 범위 | 기본값 |
|----------|------|--------|
| autoencoder_lr | 1e-6~1e-1 | 0.0001 |
| autoencoder_weight_decay | 0.0~0.1 | 0.00001 |
| lr_decay_epochs | 1000~200000 | 50000 |
| lr_decay_factor | 0.01~1.0 | 0.1 |
| scheduler | StepLR / CosineAnnealingLR | StepLR |
| use_imagenet_penalty | bool | false |
| penalty_batch_size | 1~64 | 8 |

---

#### FR-T2-12 (M): PatchCore 전용 파라미터 UI

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/config/PatchCoreParams.tsx`)

| 파라미터 | 범위 | 기본값 |
|----------|------|--------|
| backbone | wide_resnet50_2 / resnet18 / resnet50 | wide_resnet50_2 |
| pretrained_source | torchvision / 로컬 경로 | torchvision |
| coreset_sampling_ratio | 0.01~1.0 | 0.1 |
| neighbourhood_kernel_size | 1 / 3 / 5 / 7 / 9 | 3 |

---

#### FR-T2-13 (M): PatchCore 고급 설정

> **구현 상태**: 🔄 부분 구현 [확인 필요]

| 파라미터 | 범위 | 기본값 |
|----------|------|--------|
| max_train | 100~10000 | 1000 |
| knn | 1~50 | 9 |
| top_k_ratio | 0.0~1.0 | 0.1 |

---

#### FR-T2-14 (M): Threshold 설정 + 디바이스 감지

> **구현 상태**: ✅ 구현 완료 (Explorer Config 화면 — `GET /api/config` 디바이스 정보 반환)

| 항목 | 내용 |
|------|------|
| **Threshold 방식** | `percentile`(백분위 0~100) / `absolute`(절대값 0~1) |
| **Percentile 기본값** | 95.0 |
| **Absolute 기본값** | 0.5 |
| **디바이스 감지** | 화면 최초 진입 시 `GET /api/config` 호출 → `deviceInfo` 반환 → configStore 저장 |

---

#### FR-T2-15 (S): 정상/결함 비율 미리보기

> **구현 상태**: ✅ 구현 완료 (Explorer Config 화면 — `POST /api/config/preview`)

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 현재 threshold 설정 기준으로 정상/결함 판정 비율 예상치를 표시한다 |
| **API 연동** | `POST /api/config/preview` — `PreviewThresholdRequest` → `PreviewThresholdResponse { normal_ratio, defect_ratio }` |
| **표시 조건** | `datasetMeta != null` AND `preprocessingConfig != null` |

---

#### FR-T2-16 (M): 대기열 추가 버튼 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Config 화면 — `components/config/QueueSection.tsx`, `POST /api/queue`)

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 현재 설정 스냅샷을 배치 학습 대기열에 추가할 수 있어야 한다 |
| **API 연동** | `POST /api/queue` — `AddQueueRequest { name, preprocessingConfig, modelConfig }` → `AddQueueResponse { id, name }` |
| **비활성화 조건** | `preprocessingConfig == null` 또는 `modelConfig == null` |
| **v1.x 구현** | `st.button("📋 대기열에 추가")` → `experiment_queue.append()` |

---

#### FR-T2-17 (M): 대기열 테이블 + 순서 조정 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Config 화면 — `QueueSection.tsx`, `DELETE /api/queue/{id}`, `PATCH /api/queue/reorder`)

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 대기열 항목 목록을 표시하고, 순서 조정 및 삭제를 제공한다 |
| **API 연동** | `GET /api/queue` (목록 조회) / `DELETE /api/queue/{id}` (삭제) / `PATCH /api/queue/reorder` (순서 변경) |
| **테이블 컬럼** | 순번 / 실험명 / 모델 타입 / 상태 |
| **상태 색상** | 대기중: 회색 / 진행중: 파란색 / 완료: 초록색 / 실패: 빨간색 / 건너뜀: 주황색 |
| **삭제 제약** | "대기중" 상태 항목만 삭제 가능 |

---

#### FR-T2-18 (M): 대기열 항목 상세 보기 — v1.2 추가

> **구현 상태**: 🔄 부분 구현 [확인 필요]

| 항목 | 내용 |
|------|------|
| **설명** | Config 화면(Explorer)에서 대기열 항목 선택 시 해당 전처리·모델 파라미터 전체를 표시한다 |
| **표시 조건** | 대기열 테이블에서 항목이 선택된 경우 |
| **표시 내용** | 전처리 설정 + 모델 파라미터 전체 |

---

### B.4 Training 화면 (Explorer) — 구 탭3

---

#### FR-T3-01 (M): 실험명 입력 및 자동 생성

> **구현 상태**: ✅ 구현 완료 (Explorer Training 화면 — `POST /api/training/start` `experiment_name` 필드)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 사용자는 실험명을 지정하거나 자동 생성할 수 있어야 한다 |
| **자동 생성 규칙** | `{model_type}_{YYYYMMDD}_{HHMMSS}_{uuid4().hex[:4]}` — 예: `efficientad_20260508_140023_7f3a` |
| **유효성 검증** | 1~64자, 영문·숫자·한글·하이픈·언더스코어만 허용 |

---

#### FR-T3-02 (M): 학습 전 설정 요약 표시

> **구현 상태**: 🔄 부분 구현 [확인 필요] — 학습 시작 전 설정 요약 표시 여부 미확인

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 학습 시작 전 현재 설정 요약을 표시하여 사용자가 확인할 수 있어야 한다 |
| **표시 항목** | 데이터셋 경로, 전처리 방식, 모델 타입, 주요 파라미터 3~5개 |

---

#### FR-T3-03 (M): 학습 실행 제어 버튼

> **구현 상태**: ✅ 구현 완료 (Explorer Training 화면 — `components/training/ProgressSection.tsx`, `POST /api/training/{pause|unpause|stop}`)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 사용자는 학습 시작·일시정지·재개·중단을 제어할 수 있어야 한다 |
| **[학습 시작]** | `POST /api/training/start` — 학습 중 비활성화 |
| **[⏸ 일시정지]** | `POST /api/training/pause` — running 상태에서만 활성화 |
| **[▶ 재개]** | `POST /api/training/unpause` — paused 상태에서만 활성화 |
| **[⏹ 중단]** | `POST /api/training/stop` — running/paused 상태에서 활성화 |
| **상태 표시** | WebSocket `/ws/training` 연결 — 실시간 상태 수신 |

---

#### FR-T3-04 (M): 진행률 + 실시간 차트 갱신

> **구현 상태**: ✅ 구현 완료 (Explorer Training 화면 — `hooks/useTrainingWs.ts`; WebSocket `/ws/training`; Recharts LineChart)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 학습 진행 상황(진행률, Loss, 예상 완료 시간)을 실시간으로 수신하여 표시한다 |
| **WebSocket** | `/ws/training` — 서버 Push 방식; 연결 즉시 `snapshot` 메시지 수신(재연결 복구) |
| **Progress Bar** | 현재 Step / 전체 Step + 퍼센트 표시 |
| **학습 단계 인디케이터** | FR-T3-11 / FR-T3-12 참조 |
| **ETA 표시** | FR-T3-13 참조 |
| **Loss 차트** | Recharts LineChart — 실시간 갱신 |
| **로그** | 최신 100줄 유지 |

---

#### FR-T3-05 (M): 학습 완료 처리

> **구현 상태**: ✅ 구현 완료 (백엔드 자동 처리 — `GET /api/training/status`로 확인)

| 항목 | 내용 |
|------|------|
| **설명** | 학습 루프 정상 종료 시 메트릭 계산, 히스토리 저장, 모델 저장을 수행한다 |
| **WebSocket 알림** | WS `{"type": "completed"}` 수신 시 UI에 완료 표시 |
| **저장 파일** | `./models/{exp_id}/model_state_dict.pth` + `./models/{exp_id}/configs.yaml` |
| **history.json** | `status="completed"` 레코드 자동 저장 |

---

#### FR-T3-06 (M): 학습 중단 처리

> **구현 상태**: ✅ 구현 완료 (`POST /api/training/stop`)

| 항목 | 내용 |
|------|------|
| **설명** | 학습 중단 시 history.json에 중단 상태로 기록한다 |
| **동작** | `POST /api/training/stop` → 백엔드 스레드 stop_event.set() → history.json `status="중단"` 저장 |

---

#### FR-T3-07 (S): 실험명 자동 생성 미리보기

> **구현 상태**: 🔄 부분 구현 [확인 필요]

| 항목 | 내용 |
|------|------|
| **설명** | 실험명 입력란이 비어있으면 자동 생성될 이름을 미리보기로 표시한다 |

---

#### FR-T3-08 (S): 중단 실험 히스토리 기록

> **구현 상태**: ✅ 구현 완료 (history.json `status="중단"` 레코드 저장)

| 항목 | 내용 |
|------|------|
| **설명** | 중단 시 Experiments 화면에서 중단 상태 레코드로 표시된다 |

---

#### FR-T3-09 (M): 일시정지 및 체크포인트 저장

> **구현 상태**: ✅ 구현 완료 (`POST /api/training/pause` → 체크포인트 자동 저장)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 일시정지 시 현재 학습 상태를 체크포인트 파일로 저장하고 학습을 대기 상태로 전환한다 |
| **API 연동** | `POST /api/training/pause` → 백엔드 `./models/checkpoints/{exp_id}_step{N}.ckpt` 저장 |
| **WebSocket 알림** | WS `{"type": "paused", "last_ckpt_path": str}` 수신 |

---

#### FR-T3-10 (M): 체크포인트에서 재시작

> **구현 상태**: ✅ 구현 완료 (`GET /api/training/checkpoints` + `POST /api/training/resume`)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 저장된 체크포인트 목록을 표시하고 선택한 체크포인트부터 학습을 재개할 수 있어야 한다 |
| **체크포인트 목록** | `GET /api/training/checkpoints` — CheckpointsResponse |
| **재시작** | `POST /api/training/resume` — `ResumeTrainingRequest { checkpoint_path, experiment_name? }` |
| **삭제** | `DELETE /api/training/checkpoints/{name}` |

---

#### FR-T3-11 (M): EfficientAD 학습 단계 인디케이터 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Training — `components/training/StageIndicator.tsx`; WS `"stage"` 메시지 타입)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 EfficientAD 학습 중 현재 단계를 순서형 인디케이터로 표시한다 |
| **단계** | ① 데이터 로딩 → ② 모델 초기화 → ③ 학습 루프 → ④ 테스트 추론 → ⑤ 완료 |
| **WebSocket** | WS `{"type": "stage", "stage_idx": int, "stage_name": str}` 수신 |
| **표시 예시** | `✅ 데이터 로딩  ✅ 모델 초기화  🔵 학습 루프  ○ 테스트 추론  ○ 완료` |

---

#### FR-T3-12 (M): PatchCore 학습 단계 인디케이터 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Training — `StageIndicator.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 PatchCore 학습 중 현재 단계를 순서형 인디케이터로 표시한다 |
| **단계** | ① 데이터 로딩 → ② 모델 초기화 → ③ 특징 추출 → ④ Coreset 구성 → ⑤ Memory Bank → ⑥ 테스트 추론 → ⑦ 완료 |

---

#### FR-T3-13 (M): 학습 예상 완료 시간 (ETA) 표시 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Training — `ProgressSection.tsx`; WS 메시지의 진행률 데이터 기반 계산)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 학습 루프 단계의 예상 완료 시간(ETA)을 실시간으로 표시한다 |
| **EfficientAD ETA** | `elapsed / step * (total_steps - step)` — 100 step 이상 진행 후부터 표시 |
| **PatchCore ETA** | `elapsed / batch_idx * (total_batches - batch_idx)` |
| **표시 형식** | `Step {step}/{total} ({pct:.1f}%) | Loss: {loss:.4f} | 경과: {elapsed:.0f}s | ETA: {eta:.0f}s` |

---

#### FR-T3-14 (M): Training 화면 상단 대기열 표시 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Training — `components/training/QueuePanel.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer) 상단에 배치 학습 대기열 현황을 표시한다 |
| **표시 조건** | 대기열에 항목이 있는 경우에만 렌더링 |
| **데이터 소스** | `GET /api/queue` 또는 trainingStore의 배치 학습 상태 |

---

#### FR-T3-15 (M): 일괄 학습 시작 + 제어 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Explorer Training — `POST /api/training/batch/start`, `batch/skip`, `batch/stop`)

| 항목 | 내용 |
|------|------|
| **설명** | Training 화면(Explorer)에서 대기열에 등록된 설정을 순서대로 자동으로 학습할 수 있어야 한다 |
| **일괄 학습 시작** | `POST /api/training/batch/start` — 대기열 순서대로 학습 자동 실행 |
| **건너뜀** | `POST /api/training/batch/skip` — 현재 학습 건너뜀 후 다음 항목 자동 시작 |
| **전체 중단** | `POST /api/training/batch/stop` — 전체 일괄 학습 종료 |
| **진행 표시** | `일괄 학습 진행 중: {완료수}/{전체수}` 배너 + 기존 모니터링 UI |

---

### B.5 Experiments 화면 (Explorer) — 구 탭4

---

#### FR-T4-01 (M): 실험 목록 테이블 렌더링

> **구현 상태**: ✅ 구현 완료 (Explorer Experiments 화면 — `GET /api/experiments`)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer)에서 전체 실험 목록을 메트릭과 함께 테이블로 표시한다 |
| **API 연동** | `GET /api/experiments` — Experiment 배열 반환 |
| **테이블 컬럼** | 실험명 / 모델 / 파라미터 요약 / Accuracy / Precision / Recall / F1 / F2 / AUC / 실행 시각 / 상태 |
| **파라미터 요약 형식** | EfficientAD: `medium/70k/adam` / PatchCore: `wrn50/0.1` |
| **중단 실험 표시** | 지표 컬럼에 "—" 표시 |
| **정렬** | 기본: created_at 내림차순 |

---

#### FR-T4-02 (M): 실험 상세 결과 표시

> **구현 상태**: ✅ 구현 완료 (Explorer — `ConfusionMatrixChart.tsx`, `RocCurveChart.tsx`, `ScoreDistChart.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer)에서 선택된 completed 실험의 상세 결과를 시각화한다 |
| **지표 카드** | Accuracy / Precision / Recall / F1 — 4개 메트릭 카드 |
| **혼동 행렬** | 2×2 Confusion Matrix (`ConfusionMatrixChart.tsx` — div 구현) |
| **ROC 곡선** | FPR(x) vs TPR(y), JS trapezoidal AUC 계산 (`RocCurveChart.tsx`) |
| **Anomaly Score 분포** | Min-Max 정규화(0~1) 히스토그램, threshold 수직선 표시 (`ScoreDistChart.tsx`) |

---

#### FR-T4-03 (M): 실험 삭제

> **구현 상태**: ✅ 구현 완료 (`DELETE /api/experiments/{id}`)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer)에서 선택된 실험을 히스토리와 파일시스템에서 삭제할 수 있어야 한다 |
| **API 연동** | `DELETE /api/experiments/{id}` — `DeleteExperimentResponse` |
| **확인 절차** | 삭제 확인 단계 포함 |
| **비활성화 조건** | 선택된 실험이 없을 때 |

---

#### FR-T4-04 (M): 모델 저장

> **구현 상태**: ✅ 구현 완료 (`POST /api/experiments/{id}/save`)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer)에서 선택된 실험의 모델을 지정 경로로 저장할 수 있어야 한다 |
| **API 연동** | `POST /api/experiments/{id}/save` — `SaveModelRequest { destination_path }` → `SaveModelResponse { saved_path, size_bytes }` |
| **표시 조건** | 선택 실험 `status == "completed"` |

---

#### FR-T4-05 (M): 실험 없음 Guard

> **구현 상태**: ✅ 구현 완료

| 항목 | 내용 |
|------|------|
| **설명** | 실험이 없으면 안내 메시지를 표시한다 |
| **조건** | `experiments.length == 0` |

---

#### FR-T4-06 (M): 화면 진입 시 실험 목록 재로드

> **구현 상태**: ✅ 구현 완료 (React Router 화면 진입 시 `GET /api/experiments` 재호출)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer) 진입 시마다 최신 실험 목록을 로드하여 Training 완료 직후의 실험도 즉시 반영한다 |

---

#### FR-T4-07 (S): 다중 실험 비교 차트

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/tab4/ComparisonSection.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer)에서 다중 실험을 선택하여 메트릭 비교 차트를 표시한다 |
| **차트 유형** | 막대 차트 / 레이더 차트 선택 |
| **최대 선택** | 10개 |
| **비교 메트릭** | Accuracy / Precision / Recall / F1 / F2 |

---

#### FR-T4-08 (S): 저장 완료 정보 출력

> **구현 상태**: ✅ 구현 완료 (저장 경로·파일명·용량 표시)

| 항목 | 내용 |
|------|------|
| **설명** | 모델 저장 완료 시 경로·파일명·용량을 명확히 출력한다 |

---

#### FR-T4-09 (S): 중단 실험 시각적 구분

> **구현 상태**: 🔄 부분 구현 [확인 필요] — 테이블 상태 컬럼 존재; 행 색상 구분 여부 미확인

| 항목 | 내용 |
|------|------|
| **설명** | 테이블에서 status="중단" 행을 시각적으로 구분한다 |

---

#### FR-T4-NEW-01 (S): 배치 실험 비교 테이블 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Explorer — `components/tab4/BatchComparisonSection.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Experiments 화면(Explorer)에서 같은 set_id로 묶인 배치 실험들을 그룹화하여 비교 테이블로 표시한다 |
| **그룹화 기준** | 실험의 `set_id` 필드 |
| **정렬 지표** | 사용자가 비교 지표 선택 가능 |
| **출처** | Explorer README "배치 실험 비교: set_id 기준 그룹화 테이블, 정렬 지표 선택" |

---

### B.6 AnomalyMap 화면 (Explorer) — 구 탭5

---

#### FR-T5-01 (M): 실험 미선택 Guard

> **구현 상태**: ✅ 구현 완료

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 실험이 선택되지 않으면 핵심 기능을 차단하고 안내 메시지를 표시한다 |
| **조건** | `experimentsStore.selectedExperimentId == null` |

---

#### FR-T5-02 (M): 이미지 그리드 표시

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/tab5/ImageGrid.tsx` — 4열 20개/페이지, TP/FP/TN/FN 분류 배지)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 테스트 이미지를 그리드로 표시하며 각 이미지의 분류(TP/FP/TN/FN)를 배지로 표시한다 |
| **API 연동** | `GET /api/anomaly-map/{expId}/images?threshold=&defect_class=` |
| **그리드 레이아웃** | 4열 20개/페이지 |
| **분류 배지 색상** | TP: 초록 / FP: 빨강 / TN: 파랑 / FN: 주황 |
| **Triplet 이미지** | 각 이미지 클릭 시 원본/GT마스크/Heatmap 인라인 표시 (`GET .../triplet`) |

---

#### FR-T5-03 (M): Threshold 슬라이더 실시간 갱신

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/tab5/ControlBar.tsx` — 0~1.2 범위, step 0.01, 300ms debounce)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 Threshold 슬라이더 조정 시 이미지 분류(TP/FP/TN/FN)가 실시간으로 갱신된다 |
| **슬라이더 범위** | 0 ~ 1.2, step 0.01 |
| **초기값** | `threshold_method` / `threshold_value` 기반 Min-Max 정규화 자동 계산 |
| **debounce** | 300ms |
| **Write** | `anomalyMapStore.threshold` |

---

#### FR-T5-04 (M): 3분할(Triplet) 시각화

> **구현 상태**: ✅ 구현 완료 (Explorer — `GET /api/anomaly-map/{expId}/image/{path}/triplet`)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 선택된 이미지에 대해 원본 / GT 마스크 / Anomaly Map Heatmap을 3분할로 표시한다 |
| **원본** | `GET .../original` |
| **GT 마스크** | `GET .../gt_mask` — 파일 없으면 404 (빈 마스크 표시) |
| **Heatmap** | `GET .../heatmap` |
| **Triplet** | `GET .../triplet` — 3개를 합성한 단일 PNG |

---

#### FR-T5-05 (M): 결과 내보내기

> **구현 상태**: ✅ 구현 완료 (Explorer — `components/tab5/ExportSection.tsx` — CSV 직접 다운로드, ZIP 비동기 생성)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 분석 결과를 CSV 또는 ZIP으로 내보낼 수 있어야 한다 |
| **CSV** | `GET /api/anomaly-map/{expId}/export/csv?threshold=&defect_class=` — 직접 다운로드 |
| **ZIP** | `POST /api/anomaly-map/{expId}/export/zip` → job_id → 1초 폴링 → `GET /api/anomaly-map/zip/{jobId}` 다운로드 |
| **v1.x 구현** | `st.download_button()` PNG 단일 이미지 저장만 지원 |

---

#### FR-T5-06 (S): 결함 유형 필터 드롭다운

> **구현 상태**: ✅ 구현 완료 (Explorer — `ControlBar.tsx` — unique `defect_class` 추출 드롭다운)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 결함 유형별로 이미지 목록을 필터링할 수 있어야 한다 |
| **"전체" 선택** | 필터 없이 전체 이미지 표시 |
| **특정 클래스 선택** | 해당 클래스 이미지만 표시 |

---

#### FR-T5-07 (S): 이미지별 통계 바 표시

> **구현 상태**: ✅ 구현 완료 (Explorer — 이미지 그리드 상단 통계 바: 전체 수, TP/FP/TN/FN, Max/Avg Score)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 현재 표시 중인 이미지들의 통계(수, 분류, Score)를 표시한다 |
| **표시 항목** | 전체 수 / TP / FP / TN / FN 개수 / Max Anomaly Score / Avg Anomaly Score |

---

#### FR-T5-08 (S): TP/FP/TN/FN 개수 표시

> **구현 상태**: ✅ 구현 완료 (통계 바에 포함)

| 항목 | 내용 |
|------|------|
| **설명** | 현재 threshold 기준 TP/FP/TN/FN 수를 표시한다 |
| **갱신 조건** | Threshold 슬라이더 변경 시 자동 갱신 |

---

#### FR-T5-NEW-01 (M): Anomaly Map 비동기 빌드 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Explorer — `components/tab5/BuildSection.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 선택된 실험의 테스트셋 전체를 재추론하여 Anomaly Map을 비동기로 생성할 수 있어야 한다 |
| **빌드 상태 확인** | `GET /api/anomaly-map/{expId}/status` — 캐시 존재 시 즉시 완료 반환 |
| **빌드 시작** | `POST /api/anomaly-map/{expId}/build` → job_id 반환 |
| **진행 폴링** | `GET /api/anomaly-map/job/{jobId}` — 1초 간격 폴링 |
| **출처** | Explorer README "Anomaly Map 생성: 선택된 실험의 테스트셋 전체 재추론 (비동기 job + 1초 폴링)" |

---

#### FR-T5-NEW-02 (S): 이미지 그리드 페이지네이션 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Explorer — `ImageGrid.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | AnomalyMap 화면(Explorer)에서 이미지 그리드를 20개 단위로 페이지 분할하여 표시한다 |
| **페이지 크기** | 20개/페이지 |
| **그리드** | 4열 레이아웃 |
| **출처** | Explorer README "이미지 그리드: 4열 20개/페이지" |

---

### B.7 Vision 검사 플랫폼 (INSP) — v1.1 추가

> 이 절의 모든 FR은 smart-qc-vision 레포(`http://localhost:5173`)에서 실행된다.
> 상태 관리: Zustand `inspectionStore`
> 공유 데이터 계약: `history.json` (Explorer 쓰기 / Vision 읽기 전용)

---

### B.7.1 검사 공통 기능 (INSP-CMN)

---

#### FR-INSP-CMN-01 (M): 모델 미선택 Guard

> **구현 상태**: ✅ 구현 완료 (Vision — `components/layout/NoModelGuard.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Vision 검사 플랫폼에서 적용 모델이 없으면 Realtime Inspection·History 화면의 핵심 기능을 차단하고 안내 메시지를 표시한다 |
| **차단 조건** | `inspectionStore.activeModel == null` |
| **예외** | Model Settings 화면은 Guard 없이 항상 렌더링 |
| **v1.x 구현** | `FR-INSP-CMN-03` — `insp_active_model is None`인 경우 `st.info(); return` |

---

#### FR-INSP-CMN-02 (M): 검사 세션 초기화

> **구현 상태**: 🔄 부분 구현 — Zustand inspectionStore 자동 초기화로 대체됨 (v1.x session_state 패턴과 다름)

| 항목 | 내용 |
|------|------|
| **설명** | 앱 최초 진입 시 검사 플랫폼용 전역 상태를 초기화한다 |
| **v2.0 구현** | `store/inspectionStore.ts` Zustand store 자동 초기화 |
| **초기 상태** | activeModel: null, records: [], autoActive: false |
| **v1.x 구현** | `utils/session_state_init.py`의 `init_session_state()` — `INSPECTION_SESSION_SCHEMA` 전체 초기화 |

---

#### FR-INSP-CMN-OLD-01 (참고): 대시보드 전환 사이드바 — v1.x 전용

> **구현 상태**: ❌ 미구현 — v2.0에서 별도 React 앱으로 분리됨; 대시보드 전환 UI 불필요

| 항목 | 내용 |
|------|------|
| **설명 (v1.x)** | Streamlit 단일 앱에서 모델 탐색 ↔ 비전검사 대시보드 전환 버튼을 사이드바에 렌더링했다 |
| **v2.0 전환** | smart-qc-explorer / smart-qc-vision 별도 앱으로 분리 → 사이드바 전환 패턴 불필요 |

---

### B.7.2 Realtime Inspection 화면 (Vision) — 구 검사 탭1

---

#### FR-INSP-T1-01 (M): 수동 검사 실행

> **구현 상태**: ✅ 구현 완료 (Vision — `hooks/useManualInspection.ts`, `POST /api/inspection/run`)

| 항목 | 내용 |
|------|------|
| **설명** | Realtime Inspection 화면(Vision)에서 버튼 클릭 시 이미지 1개를 검사하고 결과를 저장한다 |
| **API 연동** | `POST /api/inspection/run` — async job 반환 → `GET /api/inspection/job/{job_id}` 폴링 |
| **비활성화 조건** | 자동 검사 진행 중 수동 검사 비활성화 |
| **Guard** | `NoModelGuard` 통과 필요 |
| **v1.x 구현** | `st.button("🔍 수동 검사 (1개 검사)")` → `run_inference()` 직접 호출 |

---

#### FR-INSP-T1-02 (M): 자동 검사 실행 및 중지

> **구현 상태**: ✅ 구현 완료 (Vision — `hooks/useAutoInspection.ts`, WebSocket `/ws/inspection/auto`)

| 항목 | 내용 |
|------|------|
| **설명** | Realtime Inspection 화면(Vision)에서 자동 검사를 시작하면 3초마다 검사가 반복되고, 불량 감지 또는 중지 시 중단된다 |
| **WebSocket** | `/ws/inspection/auto` — 클라이언트 → 서버: `"start"` / `"stop"` 텍스트 전송 |
| **서버 → 클라이언트 메시지** | `{type: "result", ...}` / `{type: "defect_stopped"}` / `{type: "stopped"}` / `{type: "error", message}` |
| **검사 간격** | 3.0초 (서버 설정 `_INSPECTION_INTERVAL = 3.0`) |
| **불량 감지 시 자동 중지** | 서버가 `{type: "defect_stopped"}` 전송 → 클라이언트 팝업 표시 |
| **v1.x 구현** | `time.sleep(3)` + `st.rerun()` 루프 — Streamlit 단일 스레드 방식 |

---

#### FR-INSP-T1-03 (M): 검사 결과 이미지 패널 + 판정 카드

> **구현 상태**: ✅ 구현 완료 (Vision — `components/tab1/ImagePanel.tsx`, `components/tab1/VerdictCard.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Realtime Inspection 화면(Vision)에서 검사 결과를 판정 카드 + 3종 이미지 패널로 표시한다 |
| **판정 카드** | 양품/불량 판정 결과 + Anomaly Score (`VerdictCard.tsx`) |
| **이미지 패널 1** | 원본 이미지 (`GET /api/inspection/image/last`) |
| **이미지 패널 2** | Anomaly Map (`GET /api/inspection/anomaly-map/last`) |
| **이미지 패널 3** | 이상 영역 오버레이 (`GET /api/inspection/overlay/last`) |
| **초기 상태** | 결과 없으면 검사 안내 메시지 표시 |
| **v1.x 구현** | 3열: 판정결과/원본이미지/Anomaly Map (오버레이 없음) |

---

#### FR-INSP-T1-04 (M): 검사 결과 유지

> **구현 상태**: ✅ 구현 완료 (inspectionStore에 마지막 결과 유지)

| 항목 | 내용 |
|------|------|
| **설명** | 이전 검사 결과는 새 검사가 완료될 때까지 패널에 유지된다 |
| **구현 방식** | 이미지 URL에 cache-bust 파라미터 추가 (`hooks/useInspectionImages.ts`) |
| **초기화 조건** | 모델 교체 시에만 리셋 |

---

#### FR-INSP-T1-05 (M): 불량 감지 팝업

> **구현 상태**: ✅ 구현 완료 (Vision — WebSocket `defect_stopped` 수신 시 팝업 표시)

| 항목 | 내용 |
|------|------|
| **설명** | 불량 감지 시 팝업을 표시하고, 사용자가 확인하면 자동 검사를 재개하거나 종료할 수 있다 |
| **팝업 트리거** | WS `{type: "defect_stopped"}` 수신 |
| **[확인 및 재개]** | 팝업 닫기 + WS `"start"` 재전송 |
| **[검사 종료]** | 팝업 닫기 + 자동 검사 완전 종료 |

---

#### FR-INSP-T1-06 (M): test_pool 관리

> **구현 상태**: ✅ 구현 완료 (백엔드 `inspection_service.py`에서 관리)

| 항목 | 내용 |
|------|------|
| **설명** | 검사에 사용할 이미지 풀을 모델 데이터셋에서 구성하고 순서를 관리한다 |
| **pool 빌드** | `apply_model()` 7단계 초기화 중 `build test pool` 단계 |
| **샘플링** | 순서대로 샘플링 후 풀 소진 시 shuffle |
| **was_reshuffled** | InspectionResult의 `was_reshuffled` 필드로 shuffle 여부 반환 |

---

#### FR-INSP-T1-NEW-01 (M): 오버레이 이미지 표시 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Vision — `GET /api/inspection/overlay/last`)

| 항목 | 내용 |
|------|------|
| **설명** | Realtime Inspection 화면(Vision)에서 원본 이미지 위에 이상 영역을 오버레이한 합성 이미지를 표시한다 |
| **API 연동** | `GET /api/inspection/overlay/last` |
| **출처** | Vision README "이미지 패널: 원본 이미지 / Anomaly Map / 이상 영역 오버레이 3종 실시간 표시" |

---

### B.7.3 History 화면 (Vision) — 구 검사 탭2

---

#### FR-INSP-T2-01 (M): 검사 이력 테이블

> **구현 상태**: ✅ 구현 완료 (Vision — `components/tab2/RecordsTable.tsx`, `GET /api/inspection/records`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 검사 이력을 5열 테이블로 표시한다 |
| **API 연동** | `GET /api/inspection/records?verdict=` |
| **테이블 컬럼** | 번호(seq) / 시각(inspected_at) / 이미지명(image_name) / 판정결과(verdict) / Anomaly Score |
| **정렬** | seq 내림차순 (최신 상단) |
| **행 색상** | "불량" → 빨간색 / "양품" → 초록색 |

---

#### FR-INSP-T2-02 (M): KPI 카드

> **구현 상태**: ✅ 구현 완료 (Vision — `components/tab2/KpiCards.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 검사 통계 KPI 카드 4개를 표시한다 |
| **KPI 항목** | 총 검사 / 양품 수 / 불량 수 / 불량률 |
| **불량률 계산** | `defect_count / total if total > 0 else 0.0` |

---

#### FR-INSP-T2-03 (M): 검사 이력 없음 Guard

> **구현 상태**: ✅ 구현 완료

| 항목 | 내용 |
|------|------|
| **설명** | 검사 기록이 없으면 안내 메시지 표시 + KPI 카드는 0 값으로 표시 |
| **조건** | `records.length == 0` |

---

#### FR-INSP-T2-04 (S): 검사 이력 CSV 내보내기

> **구현 상태**: ✅ 구현 완료 (Vision — `hooks/useInspectionRecords.ts`, `GET /api/inspection/records/csv`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 검사 이력을 CSV 파일로 내보낸다 |
| **API 연동** | `GET /api/inspection/records/csv` — UTF-8 BOM 인코딩 (Excel 한글 지원) |
| **CSV 컬럼** | 번호 / 시각 / 이미지명 / 판정결과 / Anomaly Score |

---

#### FR-INSP-T2-05 (M): 검사 단위 선택 + 시간 범위 테이블 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Vision — `hooks/useStatCharts.ts`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 N개 단위(20/40/100)로 검사 이력을 그룹화하여 시간 범위 테이블을 표시한다 |
| **단위 선택** | 20개 / 40개 / 100개 (기본 20) |
| **시간 계산** | 검사 간격 3초 기준 그룹별 시간 범위 계산 |
| **행 선택** | 그룹 선택 시 중앙/오른쪽 차트 갱신 |

---

#### FR-INSP-T2-06 (M): Anomaly Score 히스토그램 — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Vision — `components/tab2/ScoreHistogram.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 선택된 그룹의 Anomaly Score 분포를 히스토그램으로 표시한다 |
| **x축** | 0~1 고정 (Anomaly Score) |
| **막대 색상** | 파란색: score < threshold (정상) / 빨간색: score ≥ threshold (불량) |
| **threshold 선** | 수직선으로 표시 |

---

#### FR-INSP-T2-07 (M): Anomaly Score 산점도 (Control Chart) — v1.2 추가

> **구현 상태**: ✅ 구현 완료 (Vision — `components/tab2/ScoreScatter.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 선택된 그룹의 Anomaly Score 순차 변화를 산점도로 표시한다 |
| **x축** | 1~N 고정 (그룹 내 검사 순번) |
| **y축** | 0~1 고정 (Anomaly Score) |
| **Threshold 선** | 빨간 점선 수평선 |
| **모드** | lines+markers (control chart 스타일) |

---

#### FR-INSP-T2-NEW-01 (M): 검사 이력 초기화 버튼 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Vision — `hooks/useInspectionRecords.ts`, `DELETE /api/inspection/records`)

| 항목 | 내용 |
|------|------|
| **설명** | History 화면(Vision)에서 현재 세션의 검사 이력 전체를 초기화할 수 있어야 한다 |
| **API 연동** | `DELETE /api/inspection/records` — `ClearRecordsResponse { cleared_count }` |
| **확인 절차** | 삭제 확인 후 실행 |
| **출처** | Vision README "CSV 내보내기 및 이력 초기화" |

---

### B.7.4 Model Settings 화면 (Vision) — 구 검사 탭3

---

#### FR-INSP-T3-01 (M): 완료된 실험 목록 표시

> **구현 상태**: ✅ 구현 완료 (Vision — `components/tab3/ModelTable.tsx`, `GET /api/models`, 30초 폴링)

| 항목 | 내용 |
|------|------|
| **설명** | Model Settings 화면(Vision)에서 학습 완료된 실험 목록을 표시한다 |
| **API 연동** | `GET /api/models` — `status == "completed"` 필터링 반환 |
| **폴링** | 30초 간격 자동 갱신 (`hooks/useModels.ts`) |
| **테이블 컬럼** | 실험명 / 모델 타입 / F1 Score / AUC / 실행 시각 |
| **현재 적용 모델** | 적용 중인 모델 행에 배지 표시 |
| **v1.x 구현** | `load_history()` 호출 후 `status == "completed"` 필터 — 폴링 없음 |

---

#### FR-INSP-T3-02 (M): 모델 적용 + 세션 초기화

> **구현 상태**: ✅ 구현 완료 (Vision — `hooks/useApplyModel.ts`, `POST /api/inspection/model`)

| 항목 | 내용 |
|------|------|
| **설명** | Model Settings 화면(Vision)에서 선택된 실험의 모델을 검사 모델로 적용하고 검사 세션을 초기화한다 |
| **API 연동** | `POST /api/inspection/model` — `ApplyModelRequest { experiment_id }` → `ApplyModelResponse { active_model }` |
| **백엔드 초기화** | 7단계: GPU 경고 → 실험 조회 → 모델 캐시 클리어 → 상태 리셋 → 활성 모델 설정 → test pool 빌드 → 모델 프리로드 |
| **클라이언트 초기화** | `inspectionStore` 리셋: records=[], autoActive=false, lastResult=null |
| **경고** | "모델을 교체하면 현재 세션의 모든 검사 이력이 삭제됩니다" 표시 |
| **비활성화** | 선택된 실험이 없을 때 |

---

#### FR-INSP-T3-03 (M): 완료된 실험 없음 Guard

> **구현 상태**: ✅ 구현 완료

| 항목 | 내용 |
|------|------|
| **설명** | 사용 가능한 완료된 실험이 없으면 안내 메시지를 표시한다 |
| **조건** | `GET /api/models` 결과가 빈 배열 |

---

#### FR-INSP-T3-NEW-01 (M): 현재 적용 모델 배지 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Vision — `components/layout/ModelStatusChip.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Vision 검사 플랫폼의 모든 화면에서 현재 적용된 모델 이름을 헤더 배지로 상시 표시한다 |
| **API 연동** | `GET /api/inspection/model` — 앱 최초 진입 시 1회 조회 (`hooks/useActiveModel.ts`) |
| **표시 형식** | 모델명 + 제품명 배지 (미적용 시 "모델 없음" 표시) |
| **출처** | Vision README `components/layout/` 목록의 `ModelStatusChip` |

---

#### FR-INSP-T3-NEW-02 (S): GPU 경고 배너 — 🆕 v2.0 추가

> **구현 상태**: 🆕 코드에만 존재 (Vision — `components/layout/GpuWarningBanner.tsx`)

| 항목 | 내용 |
|------|------|
| **설명** | Vision 검사 플랫폼에서 GPU 미사용(CPU 추론) 상태를 배너로 경고한다 |
| **표시 조건** | activeModel.device == "cpu" (GPU 없이 CPU로 추론 중) |
| **출처** | Vision README `components/layout/` 목록의 `GpuWarningBanner` |

---

## C. System & Data Design

모든 데이터 스키마는 [00_Global_Context_Document.md 1절](./00_Global_Context_Document.md#1-core-data-model)에서 확정된 것을 그대로 사용한다. 이 문서에서 재정의하지 않는다.

### C.1 FR별 상태 의존성 매핑 (v2.0)

#### Explorer 대시보드

| FR ID | Zustand Store Read | Zustand Store Write |
|-------|-------------------|---------------------|
| FR-T1-01~FR-T1-NEW-02 | — | `datasetStore` (datasetPath, productName, datasetMeta) |
| FR-T2-01~FR-T2-18 | `datasetStore` | `configStore` (preprocessingConfig, modelConfig, deviceInfo) |
| FR-T3-01~FR-T3-15 | `datasetStore`, `configStore` | `trainingStore` (status, progress, lossHistory, logs, stageIdx) |
| FR-T4-01~FR-T4-NEW-01 | — | `experimentsStore` (selectedExperimentId) |
| FR-T5-01~FR-T5-NEW-02 | `experimentsStore` | `anomalyMapStore` (threshold) |

#### Vision 검사 플랫폼

| FR ID | inspectionStore Read | inspectionStore Write |
|-------|---------------------|----------------------|
| FR-INSP-CMN-01 | activeModel | — |
| FR-INSP-CMN-02 | — | 전체 inspectionStore 초기화 |
| FR-INSP-T1-01~T1-06 | activeModel | records, lastResult, autoActive |
| FR-INSP-T1-02 | autoActive | autoActive, defectPopup |
| FR-INSP-T2-01~T2-NEW-01 | records | — |
| FR-INSP-T3-01~T3-NEW-02 | activeModel | activeModel, records (리셋) |

### C.2 파일 I/O 의존성 매핑

| FR ID | 읽기 파일 | 쓰기 파일 |
|-------|-----------|-----------|
| FR-T2-05 | `./configs.yaml` (선택적) | `./configs.yaml` |
| FR-T3-05 | — | `./experiments/history.json`, `./models/{exp_id}/model_state_dict.pth`, `./models/{exp_id}/configs.yaml`, `./logs/{exp_id}.log` |
| FR-T3-06 | — | `./experiments/history.json` |
| FR-T4-01 | `./experiments/history.json` | — |
| FR-T4-03 | — | `./experiments/history.json` (삭제) |
| FR-T4-04 | `./models/{exp_id}/model_state_dict.pth` | 지정 경로 |
| FR-INSP-T3-01 | `./experiments/history.json` (읽기 전용) | — |

---

## D. API Contracts

v2.0부터 전체 API 명세는 [06_API_Specification.md](./06_API_Specification.md)에서 관리한다.

| 그룹 | 엔드포인트 접두사 | 관련 FR |
|------|-----------------|---------|
| Dataset | `POST /api/dataset/validate`, `GET /api/dataset/thumbnail/{class_name}` | FR-T1-01~FR-T1-05 |
| Config·Queue | `GET/POST /api/config`, `POST /api/config/preview*`, `GET/POST/DELETE/PATCH /api/queue` | FR-T2-05~FR-T2-18 |
| Training | `POST /api/training/{start|resume|pause|unpause|stop}`, `GET /api/training/status`, `batch/*` | FR-T3-03~FR-T3-15 |
| Experiments | `GET /api/experiments`, `DELETE /{id}`, `POST /{id}/save` | FR-T4-01~FR-T4-04 |
| Anomaly Map | `GET/POST /api/anomaly-map/{expId}/*` | FR-T5-01~FR-T5-NEW-02 |
| Vision | `GET /api/models`, `POST/GET /api/inspection/model`, `POST /api/inspection/run`, `GET/DELETE /api/inspection/records*` | FR-INSP-T1~T3 |
| WebSocket | `/ws/training` (Explorer), `/ws/inspection/auto` (Vision) | FR-T3-04, FR-INSP-T1-02 |

---

## E. AI/ML Details

```
N/A — 이 문서는 기능 명세 범위이다.
      학습 루프, 모델 초기화, 메트릭 계산 알고리즘의 상세 구현은
      08_AI_ML_Integration.md에서 다룬다.
      여기서는 각 화면에서 ML 기능을 트리거하는 UI·API 계약만 정의한다.
```

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체 상속.

이 문서에서 추가로 명시하는 항목:

| 항목 | 요구사항 | 관련 FR |
|------|----------|---------|
| **Threshold debounce** | AnomalyMap 슬라이더 조작 후 300ms debounce 후 API 호출 | FR-T5-03 |
| **폴링 간격** | Vision Model Settings 화면 실험 목록 30초 폴링 | FR-INSP-T3-01 |
| **WebSocket 재연결 복구** | WS 재연결 시 snapshot 메시지로 상태 복원 | FR-T3-04 |
| **비동기 job 폴링** | Anomaly Map 빌드 1초 폴링 | FR-T5-NEW-01 |
| **Threshold 슬라이더 응답** | 슬라이더 조작 후 테이블 갱신 ≤ 0.5초 (재추론 없이 기존 scores 재계산) | FR-T5-03 |
| **configs.yaml 쓰기 안전성** | R-ATOMIC-01 (임시 파일 → rename) | FR-T2-05 |
| **history.json 쓰기 안전성** | R-ATOMIC-01 | FR-T3-05, FR-T3-06, FR-T4-03 |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards) 전체 상속.

FR별 로그 이벤트 매핑:

#### Explorer 대시보드

| FR ID | 이벤트명 | 레벨 |
|-------|---------|------|
| FR-T1-01 (성공) | `dataset_validated` | INFO |
| FR-T1-01 (실패) | `dataset_validation_failed` | ERROR |
| FR-T2-05 | `config_saved` | INFO |
| FR-T3-03 (시작) | `training_started` | INFO |
| FR-T3-05 (완료) | `training_completed` | INFO |
| FR-T3-06 (중단) | `training_stopped` | WARNING |
| FR-T3-05 (실패) | `training_failed` | ERROR |
| FR-T3-09 (일시정지) | `training_paused` | INFO |
| FR-T3-10 (재시작) | `training_resumed` | INFO |
| FR-T4-04 (저장) | `model_saved` | INFO |
| FR-T4-03 (삭제) | `experiment_deleted` | WARNING |

#### Vision 검사 플랫폼

| FR ID | 이벤트명 | 레벨 |
|-------|---------|------|
| FR-INSP-T3-02 | `insp_model_applied` | INFO |
| FR-INSP-T1-01 | `insp_inspection_started_manual` | INFO |
| FR-INSP-T1-02 (시작) | `insp_inspection_started_auto` | INFO |
| FR-INSP-T1-02 (중지) | `insp_inspection_stopped_auto` | INFO |
| FR-INSP-T1-05 (감지) | `insp_defect_detected` | WARNING |
| FR-INSP-T1-06 (소진) | `insp_pool_reshuffled` | INFO |

---

## H. QA & Validation

### H.1 FR 완료 기준 체크리스트

#### Dataset 화면 (Explorer)

- [ ] FR-T1-01: 존재하지 않는 경로 입력 시 에러 표시
- [ ] FR-T1-01: 올바른 MVTec AD 경로 입력 시 검증 통과 + datasetMeta 구성
- [ ] FR-T1-01b: 경로 변경 시 configStore 자동 초기화
- [ ] FR-T1-03: 폴더 트리 최대 3단계 렌더링
- [ ] FR-T1-04: 클래스별 이미지 수 테이블 + 합계 행
- [ ] FR-T1-05: MVTec 3열 / OK/NG 2열 썸네일 그리드
- [ ] FR-T1-06: channels == 1 시 그레이스케일 감지 메시지 표시
- [ ] FR-T1-NEW-01: 제품명 입력 → datasetStore.productName 저장

#### Config 화면 (Explorer)

- [ ] FR-T2-01: 비선택 전처리 방식의 파라미터 DOM 미존재
- [ ] FR-T2-02: 모든 슬라이더 범위 내 값만 허용
- [ ] FR-T2-03: image_size 32의 배수 아닌 값 입력 시 에러
- [ ] FR-T2-05: [설정 저장] 후 configStore 갱신
- [ ] FR-T2-08: PatchCore 선택 시 EfficientAD 파라미터 DOM 미존재
- [ ] FR-T2-15: POST /api/config/preview 정상 응답
- [ ] FR-T2-16: [대기열 추가] → POST /api/queue 성공
- [ ] FR-T2-17: 순서 조정 → PATCH /api/queue/reorder 성공
- [ ] FR-T2-17: 삭제 → DELETE /api/queue/{id} 성공

#### Training 화면 (Explorer)

- [ ] FR-T3-03: 학습 중 [학습 시작] 비활성화
- [ ] FR-T3-04: WS 연결 즉시 snapshot 수신 → UI 상태 복원
- [ ] FR-T3-04: Loss 차트 실시간 갱신
- [ ] FR-T3-04: 로그 최신 100줄 유지
- [ ] FR-T3-05: 완료 후 `./models/{exp_id}/` 에 .pth + configs.yaml 존재
- [ ] FR-T3-06: 중단 후 history.json에 status="중단" 레코드 존재
- [ ] FR-T3-09: 일시정지 후 `.ckpt` 파일 생성 확인
- [ ] FR-T3-10: 체크포인트 목록 정상 표시 (idle 상태)
- [ ] FR-T3-10: 체크포인트에서 재시작 → POST /api/training/resume 성공
- [ ] FR-T3-11: EfficientAD 단계 인디케이터 표시 확인
- [ ] FR-T3-15: 일괄 학습 → POST /api/training/batch/start 성공

#### Experiments 화면 (Explorer)

- [ ] FR-T4-01: 화면 진입 시 최신 실험 목록 반영
- [ ] FR-T4-02: ROC Curve AUC 값 범례 표시
- [ ] FR-T4-02: Anomaly Score 분포 threshold 수직선 표시
- [ ] FR-T4-03: 삭제 확인 후 삭제 + `./models/{exp_id}/` 제거
- [ ] FR-T4-04: 저장 후 경로·파일명·용량 출력
- [ ] FR-T4-07: 다중 선택 막대/레이더 차트 표시

#### AnomalyMap 화면 (Explorer)

- [ ] FR-T5-NEW-01: POST /api/anomaly-map/{id}/build → job 폴링 → 완료
- [ ] FR-T5-02: 이미지 그리드 TP/FP/TN/FN 분류 배지 표시
- [ ] FR-T5-03: 슬라이더 변경 후 300ms debounce + 분류 갱신
- [ ] FR-T5-05: CSV 다운로드 정상 동작
- [ ] FR-T5-05: ZIP 비동기 생성 + 폴링 + 다운로드

#### Vision 검사 플랫폼

- [ ] FR-INSP-CMN-01: activeModel == null 시 Realtime/History 화면 Guard 표시
- [ ] FR-INSP-T1-01: 수동 검사 → POST /api/inspection/run + job 폴링 + 결과 표시
- [ ] FR-INSP-T1-02: 자동 검사 시작 → WS "start" 전송 → result 수신
- [ ] FR-INSP-T1-02: 불량 감지 → WS defect_stopped 수신 → 팝업 표시
- [ ] FR-INSP-T1-03: 원본/Anomaly Map/오버레이 3종 이미지 표시
- [ ] FR-INSP-T2-01: GET /api/inspection/records → 테이블 표시
- [ ] FR-INSP-T2-02: KPI 카드 4개 정상 계산
- [ ] FR-INSP-T2-04: GET /api/inspection/records/csv 다운로드
- [ ] FR-INSP-T2-NEW-01: DELETE /api/inspection/records → 이력 초기화
- [ ] FR-INSP-T3-01: GET /api/models → 30초 폴링 → 목록 갱신
- [ ] FR-INSP-T3-02: POST /api/inspection/model → activeModel 설정 + 이력 초기화

### H.2 Given-When-Then 시나리오

#### TC-FR-T3-05: 학습 완료 후 파일 검증

```
Given:  datasetPath, preprocessingConfig, modelConfig 모두 설정됨
        model_type = "patchcore", coreset_sampling_ratio = 0.1
When:   [학습 시작] 클릭 → WS completed 메시지 수신
Then:   history.json 에 status="completed" 레코드 1개 추가됨
        exp_id = "patchcore_{YYYYMMDD}_{HHMMSS}_{4자리}" 형식
        ./models/{exp_id}/model_state_dict.pth 파일 존재
        ./models/{exp_id}/configs.yaml 파일 존재
        ./logs/{exp_id}.log 파일 존재
```

#### TC-FR-T5-03: Threshold 슬라이더 실시간 갱신

```
Given:  AnomalyMap 화면에서 실험 선택됨, 이미지 그리드 표시 중
        initial_threshold = 0.5
        첫 번째 이미지 score = 0.6 → 초기 판정 "NG"
When:   Threshold 슬라이더를 0.7로 변경한다
Then:   300ms debounce 후 API 재호출
        score=0.6 < threshold=0.7 → 해당 이미지 판정 "OK"로 변경
        GT label=1(결함) AND 판정="OK" → 오분류 "FN"으로 변경
        anomalyMapStore.threshold == 0.7
```

#### TC-FR-INSP-T1-02: 불량 감지 시 자동 검사 중지

```
Given:  자동 검사 중 (WS /ws/inspection/auto 연결됨)
        activeModel 설정됨
When:   서버에서 verdict == "불량" 결과 → WS {type: "defect_stopped"} 수신
Then:   autoActive == false
        팝업 메시지 표시
        자동 검사 루프 중단
        [확인 및 재개] 클릭 시 WS "start" 재전송
```

#### TC-FR-INSP-T3-02: 모델 교체 후 이력 초기화

```
Given:  inspectionStore.records에 검사 기록 5개 존재
        activeModel = "exp_A"
When:   Model Settings 화면에서 "exp_B" 선택 후 [이 모델로 검사 시작] 클릭
Then:   POST /api/inspection/model { experiment_id: "exp_B" } → 성공
        inspectionStore.records == []
        inspectionStore.activeModel.experimentId == "exp_B"
        inspectionStore.lastResult == null
```

#### TC-FR-INSP-T2-02: KPI 카드 계산 정확성

```
Given:  records = [
          {verdict: "양품"}, {verdict: "불량"}, {verdict: "양품"},
          {verdict: "불량"}, {verdict: "불량"}
        ]
When:   History 화면 렌더링
Then:   총 검사 = 5
        양품 = 2
        불량 = 3
        불량률 = "60.0%"
```

---

## I. Implementation Plan

```
N/A — 전체 구현 순서·WBS·소요 시간은 14_Deployment_and_Release_Plan.md에서 다룬다.
      이 문서의 FR ID(FR-T1-01 ~ FR-INSP-T3-NEW-02)는
      14_Deployment_and_Release_Plan.md의 WBS 작업 항목으로 직접 매핑된다.
```

---

*다음 문서*: [04_System_Architecture.md](./04_System_Architecture.md)
