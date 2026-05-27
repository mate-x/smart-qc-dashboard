# 03. Functional Requirements

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [01_Product_Overview.md](./01_Product_Overview.md), [02_User_Personas_and_Use_Cases.md](./02_User_Personas_and_Use_Cases.md)
> **버전**: v1.0
> **작성일**: 2026-05-08
> **후속 문서**: [04_System_Architecture.md](./04_System_Architecture.md)

---

## 문서 사용 규칙

- **FR ID 체계**: `FR-{영역}-{순번}` — 예: `FR-CMN-01`, `FR-T1-03`
- **우선순위**: `M` = Must Have (MVP 필수) / `S` = Should Have (MVP 권장) / `N` = Nice to Have (MVP 제외)
- **Streamlit 컴포넌트**: 실제 함수명과 주요 인자를 명시한다
- **유효성 검증**: 범위·타입·제약조건을 숫자로 명시한다. "적절한 값" 같은 표현 금지

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Detailed Specification](#b-detailed-specification)
  - [B.1 공통 기능 (CMN)](#b1-공통-기능-cmn)
  - [B.2 탭1 — 데이터 폴더 구조](#b2-탭1--데이터-폴더-구조)
  - [B.3 탭2 — 전처리 및 모델 설정](#b3-탭2--전처리-및-모델-설정)
  - [B.4 탭3 — 학습 시작 + 학습 로그](#b4-탭3--학습-시작--학습-로그)
  - [B.5 탭4 — 실험 히스토리 + 결과 상세 + 모델 저장](#b5-탭4--실험-히스토리--결과-상세--모델-저장)
  - [B.6 탭5 — 이상 영역 시각화](#b6-탭5--이상-영역-시각화)
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

탭1~탭5 전체 기능을 FR ID 단위로 분해하여, 각 기능의 동작 조건·UI 컴포넌트·유효성 검증·입출력을 구현 가능한 수준으로 명세한다. 이 문서에 정의된 FR이 구현 완료 기준이 된다.

### A.2 FR 전체 목록 요약

| 영역 | FR 수 (M) | FR 수 (S) | 합계 |
|------|-----------|-----------|------|
| 공통 (CMN) | 4 | 0 | 4 |
| 탭1 | 6 | 2 | 8 |
| 탭2 (전처리 및 모델 설정 통합) | 12 | 3 | 15 |
| 탭3 (구 탭4) | 6 | 2 | 8 |
| 탭4 (구 탭5) | 6 | 3 | 9 |
| 탭5 (구 탭6) | 5 | 3 | 8 |
| **합계** | **39** | **13** | **52** |

---

## B. Detailed Specification

---

### B.1 공통 기능 (CMN)

---

#### FR-CMN-01 (M): 앱 진입점 초기화

| 항목 | 내용 |
|------|------|
| **설명** | 앱 최초 실행 시 session_state를 초기화하고 전체 레이아웃을 구성한다 |
| **트리거** | `app.py` 실행 (Streamlit 프로세스 시작) |
| **구현 위치** | `app.py` |
| **동작** | 1. `utils/session_state_init.py`의 `init_session_state()` 호출 (00_Global_Context 3.1절 스키마 기준) <br> 2. `st.set_page_config(page_title="비전검사 대시보드", layout="wide")` <br> 3. `components/sidebar.py`의 사이드바 렌더링 <br> 4. `st.tabs(["📁 데이터", "⚙️ 전처리 및 모델 설정", "🚀 학습", "📊 히스토리", "🔍 Anomaly Map"])` 로 5탭 렌더링 <br> 5. 각 탭 내부에서 해당 `tabs/tab{n}_*.py` 함수 호출 |
| **멱등성** | Streamlit rerun 발생 시 `init_session_state()`는 이미 설정된 키를 덮어쓰지 않는다 |

---

#### FR-CMN-02 (M): 사이드바 상시 정보 렌더링

| 항목 | 내용 |
|------|------|
| **설명** | 사이드바에 데이터셋 정보, 디바이스 정보, 현재 설정 요약을 상시 표시한다 |
| **구현 위치** | `components/sidebar.py` |
| **렌더링 조건** | 각 섹션은 해당 session_state 키가 None이 아닌 경우에만 렌더링한다 |
| **섹션 1 — 데이터셋** | `dataset_meta` != None 시 렌더링 <br> · 경로: `dataset_path` (전체 경로 표시) <br> · 학습 이미지: `dataset_meta.train_good_count`장 <br> · 테스트 이미지: `dataset_meta.total_test_count`장 |
| **섹션 2 — 디바이스** | `device_info` != None 시 렌더링 <br> · device == "cuda": "CUDA ({gpu_name}), VRAM: {vram_gb:.1f} GB" <br> · device == "cpu": "CPU" |
| **섹션 3 — 현재 설정** | `preprocessing_config` != None 시 전처리 방식 표시 <br> `model_config` != None 시 모델 타입 + model_size 또는 backbone 표시 |
| **Streamlit 컴포넌트** | `st.sidebar.markdown()`, `st.sidebar.divider()` |

---

#### FR-CMN-03 (M): 탭 진입 Guard

| 항목 | 내용 |
|------|------|
| **설명** | 선행 설정이 완료되지 않은 탭에 진입 시 핵심 기능을 차단하고 안내 메시지를 표시한다 |
| **구현 위치** | 각 `tabs/tab{n}_*.py` 파일 상단 |
| **구현 패턴** | `if session_state.{key} is None: st.warning(MSG["{key}"]); return` |
| **차단 조건 및 메시지** | 01_Product_Overview.md B.4절 표와 동일 |
| **Guard 체인** | 탭2 guard: `dataset_meta is None` → `st.warning(MSG["NO_DATASET"]); return` <br> 탭3 guard: `dataset_path is None or preprocessing_config is None or model_config is None` → `st.warning(MSG["{key}"]); return` <br> (구 탭3의 독립 guard `preprocessing_config is None` 제거 — 탭2·탭3 통합으로 소멸) |
| **비고** | `st.tabs()`는 항상 5개 탭을 렌더링한다. `return`으로 함수를 조기 종료하여 이후 코드를 실행하지 않는다 |

---

#### FR-CMN-04 (M): 표준 안내 메시지 상수

| 항목 | 내용 |
|------|------|
| **설명** | 모든 안내·오류 메시지는 `utils/messages.py`의 `MSG` 딕셔너리에서 참조한다. 탭 코드에서 문자열 리터럴 직접 사용 금지 |
| **구현 위치** | `utils/messages.py` |
| **상수 목록** | 00_Global_Context 3.4절 `MSG` 딕셔너리 그대로 사용 |

---

### B.2 탭1 — 데이터 폴더 구조

---

#### FR-T1-01 (M): 데이터셋 경로 입력 및 즉시 검증

| 항목 | 내용 |
|------|------|
| **설명** | 텍스트 입력으로 데이터셋 루트 경로를 받아 MVTec AD 구조를 검증한다 |
| **Streamlit 컴포넌트** | `st.text_input(label="데이터셋 경로 (Dataset Path)", key="input_dataset_path")` |
| **트리거** | 입력값 변경 시 (`on_change` 또는 Enter) |
| **검증 순서** | 1. `Path(input).exists()` → False면 `st.error(ERR_DATASET_NOT_FOUND)`, `dataset_path = None`, 종료 <br> 2. `Path(input, "train", "good").is_dir()` → False면 `st.error(ERR_INVALID_FOLDER_STRUCTURE)`, 종료 <br> 3. `Path(input, "test").is_dir()` → False면 `st.error(ERR_INVALID_FOLDER_STRUCTURE)`, 종료 <br> 4. `train/good/` 하위 지원 포맷 이미지 수 == 0 → `st.error("train/good/ 에 유효한 이미지가 없습니다.")`, 종료 <br> 5. 검증 통과 → `dataset_path = input`, `dataset_meta` 구성 후 Write |
| **지원 포맷 판별** | `suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}` |
| **출력** | 검증 성공 시 `st.success("데이터셋 구조 검증 완료.")` |

---

#### FR-T1-02 (M): dataset_meta 구성

| 항목 | 내용 |
|------|------|
| **설명** | 검증 통과한 경로에서 dataset_meta를 구성하여 session_state에 저장한다 |
| **구현 위치** | `tabs/tab1_data_folder.py` 내 `build_dataset_meta(path: str) -> dict` 함수 |
| **구성 항목** | 00_Global_Context 1.5절 `dataset_meta` 스키마 전체 |
| **채널 감지** | `PIL.Image.open(첫_번째_이미지).mode` <br> · mode == "L" → `channels = 1` <br> · mode in {"RGB", "RGBA"} → `channels = 3` <br> · 그 외 → `channels = 3` (기본값) |
| **결함 클래스 추출** | `test/` 하위 디렉토리명 목록. `good` 포함. |
| **Write** | `st.session_state.dataset_meta = build_dataset_meta(path)` |

---

#### FR-T1-03 (M): 폴더 트리 시각화

| 항목 | 내용 |
|------|------|
| **설명** | 데이터셋 폴더 구조를 들여쓰기 텍스트로 렌더링한다 |
| **Streamlit 컴포넌트** | `st.code(tree_text, language=None)` |
| **렌더링 깊이** | 최대 3단계 (`dataset_root/train/good/`, `dataset_root/test/{class}/`, `dataset_root/ground_truth/{class}/`) |
| **출력 형식 예시** | `📂 screw/` <br>&nbsp;&nbsp;`📂 train/` <br>&nbsp;&nbsp;&nbsp;&nbsp;`📂 good/ (320장)` <br>&nbsp;&nbsp;`📂 test/` <br>&nbsp;&nbsp;&nbsp;&nbsp;`📂 good/ (41장)` <br>&nbsp;&nbsp;&nbsp;&nbsp;`📂 thread_side/ (15장)` |

---

#### FR-T1-04 (M): 클래스별 이미지 수 테이블

| 항목 | 내용 |
|------|------|
| **설명** | train/test/ground_truth 별 클래스별 이미지 수를 테이블로 표시한다 |
| **Streamlit 컴포넌트** | `st.dataframe(df, use_container_width=True)` |
| **테이블 컬럼** | 클래스명 / 학습(train) 수 / 테스트(test) 수 / GT 마스크 수 |
| **train 행** | `good` 클래스만 존재, `train_good_count` 값 |
| **test 행** | `test_counts` 딕셔너리의 모든 클래스 |
| **합계 행** | 각 열의 합계 |

---

#### FR-T1-05 (M): 대표 썸네일 렌더링

| 항목 | 내용 |
|------|------|
| **설명** | 각 결함 클래스의 대표 이미지(첫 번째 파일)를 썸네일로 표시한다 |
| **Streamlit 컴포넌트** | `st.image(image, caption="{class_name}", width=150)` |
| **레이아웃** | `st.columns(min(len(defect_classes), 4))` — 최대 4열, 초과 시 다음 행으로 wrap |
| **이미지 처리** | `PIL.Image.open(path).resize((150, 150), Image.LANCZOS)` |
| **Grayscale 처리** | channels == 1이면 `image.convert("RGB")` 후 표시 |

---

#### FR-T1-06 (M): Grayscale 자동 감지 안내

| 항목 | 내용 |
|------|------|
| **설명** | dataset_meta.channels == 1이면 안내 메시지를 표시한다 |
| **Streamlit 컴포넌트** | `st.info(MSG["GRAYSCALE_DETECT"])` |
| **표시 조건** | `dataset_meta.channels == 1` |
| **표시 위치** | FR-T1-02 완료 직후, FR-T1-03 위 |

---

#### FR-T1-07 (S): 지원 포맷 외 파일 경고

| 항목 | 내용 |
|------|------|
| **설명** | 지원 포맷 외 파일이 존재하면 경고 메시지를 표시한다. 학습은 차단하지 않는다 |
| **Streamlit 컴포넌트** | `st.warning(f"지원하지 않는 파일 {N}개가 발견되었습니다. 학습에서 제외됩니다.")` |
| **표시 조건** | `dataset_meta.has_invalid_files == True` |

---

#### FR-T1-08 (S): 폴더 구조 오류 상세 안내

| 항목 | 내용 |
|------|------|
| **설명** | 검증 실패 시 어떤 하위 폴더가 누락되었는지 구체적으로 표시한다 |
| **출력 예시** | "누락된 폴더: `train/good/` — MVTec AD 형식의 폴더 구조가 아닙니다." |

---

### B.3 탭2 — 전처리 및 모델 설정

---

#### FR-T2-01 (M): 전처리 방식 라디오 선택

| 항목 | 내용 |
|------|------|
| **설명** | 전처리 방식을 라디오 버튼으로 선택한다. 선택에 따라 파라미터 UI를 조건부 렌더링한다 |
| **Streamlit 컴포넌트** | `st.radio("전처리 방식 (Preprocessing Method)", options=["없음", "Homomorphic", "HE", "CLAHE"], horizontal=True)` |
| **내부 값 매핑** | "없음" → `"none"`, "Homomorphic" → `"homomorphic"`, "HE" → `"he"`, "CLAHE" → `"clahe"` |
| **파라미터 UI 조건** | `if method == "homomorphic":` → Homomorphic 파라미터 렌더링 <br> `elif method == "clahe":` → CLAHE 파라미터 렌더링 <br> `elif method == "he":` → 안내 텍스트만 렌더링 <br> `else:` → 파라미터 UI 없음 <br> **비선택 방식의 UI는 DOM에 렌더링하지 않는다 (R-UI-02)** |

---

#### FR-T2-02 (M): 전처리 파라미터 UI

| 방식 | 파라미터 | Streamlit 컴포넌트 | 범위 | 기본값 |
|------|----------|-------------------|------|--------|
| **Homomorphic** | sigma | `st.slider("sigma", 0.1, 50.0, 10.0, 0.1)` | 0.1 ~ 50.0 | 10.0 |
| | gamma_H | `st.slider("gamma_H", 1.0, 3.0, 1.5, 0.05)` | 1.0 ~ 3.0 | 1.5 |
| | gamma_L | `st.slider("gamma_L", 0.1, 1.0, 0.5, 0.05)` | 0.1 ~ 1.0 | 0.5 |
| | normalize | `st.checkbox("정규화 적용 (normalize)", value=True)` | bool | True |
| **HE** | (없음) | `st.info("히스토그램 평탄화(HE)는 파라미터가 없습니다.")` | - | - |
| **CLAHE** | clip_limit | `st.slider("클립 한계 (clipLimit)", 0.1, 40.0, 2.0, 0.1)` | 0.1 ~ 40.0 | 2.0 |

---

#### FR-T2-03 (M): Resize + 정규화 설정

| 항목 | 내용 |
|------|------|
| **image_size 입력** | `st.number_input("이미지 크기 (image_size)", min_value=32, max_value=1024, value=256, step=32)` <br> · 32의 배수 강제: `if value % 32 != 0: st.error("32의 배수만 입력 가능합니다.")` |
| **resize_mode** | 고정값 "padding". UI 노출 없음 (변경 불가) |
| **정규화 방식** | `st.radio("정규화 방식 (Normalization)", ["ImageNet", "커스텀"], horizontal=True)` |
| **커스텀 mean/std** | "커스텀" 선택 시만 렌더링 (R-UI-02) <br> `st.text_input("mean (쉼표 구분, 예: 0.5,0.5,0.5)", value="0.5,0.5,0.5")` <br> `st.text_input("std (쉼표 구분, 예: 0.5,0.5,0.5)", value="0.5,0.5,0.5")` <br> 파싱: `[float(x.strip()) for x in value.split(",")]` — 길이 != 3 시 `st.error()` |
| **ImageNet 고정값** | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

---

#### FR-T2-04 (M): 전처리 적용 전후 미리보기

| 항목 | 내용 |
|------|------|
| **설명** | 파라미터 변경 시 샘플 이미지에 전처리를 즉시 적용하여 미리보기를 갱신한다 |
| **샘플 이미지** | `train/good/` 의 첫 번째 파일 (알파벳 순 정렬) |
| **레이아웃** | `col1, col2 = st.columns(2)` <br> col1: 원본 이미지 (`st.image(orig, caption="원본", use_container_width=True)`) <br> col2: 처리 후 이미지 (`st.image(processed, caption="{method} 적용 후", use_container_width=True)`) |
| **처리 파이프라인** | 전처리 필터 → Resize+Padding → (정규화는 미리보기에서 역정규화 후 표시) |
| **갱신 트리거** | 파라미터 슬라이더, 라디오, 텍스트 입력 변경 시 (Streamlit 자동 rerun) |
| **Grayscale 처리** | channels == 1이면 `convert("RGB")` 후 전처리 적용 |

---

#### FR-T2-05 (M): 설정 저장

| 항목 | 내용 |
|------|------|
| **설명** | 전처리 영역 및 모델 영역의 현재 UI 설정을 session_state에 동시 저장한다 |
| **Streamlit 컴포넌트** | `st.button("설정 저장", type="primary")` |
| **동작** | 1. 현재 UI 값으로 preprocessing_config dict 구성 (00_Global_Context 1.6절 스키마) <br> 2. 현재 UI 값으로 model_config dict 구성 (00_Global_Context 1.7절 스키마) <br> 3. `st.session_state.preprocessing_config = preprocessing_config` <br> 4. `st.session_state.model_config = model_config` <br> 5. `st.success("설정이 저장되었습니다.")` |
| **추가 버튼** | `st.button("configs.yaml 저장")` → `save_config_section("preprocessing", preprocessing_config)` + `save_config_section("model", model_config)` (preprocessing + model 섹션 동시 저장) |
| **추가 버튼** | `st.button("configs.yaml 불러오기")` → UC-09 플로우 실행 (preprocessing + model 섹션 모두 로드) |

---

#### FR-T2-06 (S): 비선택 파라미터 UI 완전 숨김

| 항목 | 내용 |
|------|------|
| **설명** | 선택되지 않은 전처리 방식의 파라미터 UI를 DOM에서 완전히 제거한다 |
| **구현 방식** | `if method == "homomorphic":` 조건부 렌더링. `disabled=True` 사용 금지 (R-UI-02) |
| **검증** | 개발자 도구에서 비선택 방식의 슬라이더 DOM 요소가 존재하지 않아야 함 |

---

#### FR-T2-07 (S): Grayscale 자동 RGB 변환

| 항목 | 내용 |
|------|------|
| **설명** | dataset_meta.channels == 1이면 전처리 파이프라인에서 자동으로 RGB 변환을 수행한다 |
| **구현 위치** | `utils/image_utils.py`의 `apply_preprocessing()` 함수 |
| **동작** | `if image.mode == "L": image = image.convert("RGB")` (전처리 필터 적용 전) |

---

#### FR-T2-08 (M): 모델 종류 라디오 선택

| 항목 | 내용 |
|------|------|
| **설명** | EfficientAD 또는 PatchCore를 선택한다. 선택에 따라 전용 파라미터 UI를 조건부 렌더링한다 |
| **Streamlit 컴포넌트** | `st.radio("모델 선택 (Model Type)", ["EfficientAD", "PatchCore"], horizontal=True)` |
| **내부 값 매핑** | "EfficientAD" → `"efficientad"`, "PatchCore" → `"patchcore"` |
| **비선택 파라미터 처리** | 선택되지 않은 모델의 파라미터 UI는 렌더링하지 않는다 (R-UI-02) |

---

#### FR-T2-09 (M): 공통 파라미터 UI

공통 설정은 batch_size + random_seed 2개로 구성된다. image_size는 전처리 영역(FR-T2-03) 단독 소유이며 이 섹션에서 중복 렌더링하지 않는다.

| 파라미터 | Streamlit 컴포넌트 | 범위 | 기본값 | 비고 |
|----------|-------------------|------|--------|------|
| batch_size | `st.number_input("배치 크기 (batch_size)", 1, 128, 16, 1)` | 1~128 | 16 | |
| random_seed | `st.number_input("랜덤 시드 (random_seed)", 0, 2147483647, 42, 1)` | 0~2147483647 | 42 | |

---

#### FR-T2-10 (M): EfficientAD 전용 파라미터 UI — 기본 노출

| 파라미터 | Streamlit 컴포넌트 | 범위 | 기본값 |
|----------|-------------------|------|--------|
| model_size | `st.radio("모델 크기 (model_size)", ["small", "medium"], horizontal=True)` | enum | "medium" |
| train_steps | `st.number_input("학습 단계 수 (train_steps)", 1000, 200000, 70000, 1000)` | 1000~200000 | 70000 |
| optimizer | `st.selectbox("옵티마이저 (optimizer)", ["adam", "adamw", "sgd"])` | enum | "adam" |
| learning_rate | `st.number_input("학습률 (learning_rate)", 1e-6, 1e-1, 1e-4, format="%.6f")` | 1e-6~1e-1 | 0.0001 |
| weight_decay | `st.number_input("가중치 감쇠 (weight_decay)", 0.0, 0.1, 1e-4, format="%.6f")` | 0.0~0.1 | 0.0001 |
| out_channels | `st.selectbox("출력 채널 수 (out_channels)", [128, 256, 384, 512])` | enum | 384 |
| padding | `st.checkbox("패딩 사용 (padding)", value=False)` | bool | False |
| ae_loss_weight (α) | `st.slider("AE Loss 비중 (ae_loss_weight)", 0.0, 1.0, 0.5, 0.01)` | 0.0~1.0 | 0.5 |

ae_loss_weight(α)는 학습 루프 내에서 `total = α * loss_ae + (1-α) * loss_st + loss_stae` 공식으로 사용된다. UI에서는 ae_loss_weight 슬라이더만 표시한다.

---

#### FR-T2-11 (M): EfficientAD 고급 설정

| 항목 | 내용 |
|------|------|
| **Streamlit 컴포넌트** | `with st.expander("고급 설정 (Advanced Settings)"):` |
| **파라미터 목록** | |

| 파라미터 | 컴포넌트 | 범위 | 기본값 |
|----------|----------|------|--------|
| autoencoder_lr | `st.number_input(...)` | 1e-6~1e-1 | 0.0001 |
| autoencoder_weight_decay | `st.number_input(...)` | 0.0~0.1 | 0.00001 |
| lr_decay_epochs | `st.number_input(...)` | 1000~200000 | 50000 |
| lr_decay_factor | `st.slider(...)` | 0.01~1.0 | 0.1 |
| scheduler | `st.selectbox(...)` | ["StepLR", "CosineAnnealingLR"] | "StepLR" |
| use_imagenet_penalty | `st.checkbox("ImageNet Penalty 사용 (use_imagenet_penalty)", value=False)` | bool | False |
| penalty_batch_size | `st.number_input(...)` | 1~64 | 8 |

---

#### FR-T2-12 (M): PatchCore 전용 파라미터 UI — 기본 노출

| 파라미터 | Streamlit 컴포넌트 | 범위 | 기본값 |
|----------|-------------------|------|--------|
| backbone | `st.selectbox("백본 (backbone)", ["wide_resnet50_2", "resnet18", "resnet50"])` | enum | "wide_resnet50_2" |
| pretrained_source | `st.radio("사전학습 가중치 출처 (pretrained_source)", ["torchvision", "로컬 경로"], horizontal=True)` | enum | "torchvision" |
| pretrained_path | `st.text_input("로컬 가중치 경로 (pretrained_path)")` — `pretrained_source=="로컬 경로"`일 때만 렌더링 | str | null |
| coreset_sampling_ratio | `st.slider("코어셋 비율 (coreset_sampling_ratio)", 0.01, 1.0, 0.1, 0.01)` | 0.01~1.0 | 0.1 |
| neighbourhood_kernel_size | `st.select_slider("이웃 커널 크기 (neighbourhood_kernel_size)", [1,3,5,7,9])` | 홀수 1~9 | 3 |

---

#### FR-T2-13 (M): PatchCore 고급 설정

| 항목 | 내용 |
|------|------|
| **Streamlit 컴포넌트** | `with st.expander("고급 설정 (Advanced Settings)"):` |

| 파라미터 | 컴포넌트 | 범위 | 기본값 |
|----------|----------|------|--------|
| max_train | `st.number_input("최대 학습 샘플 수 (max_train)", 100, 10000, 1000, 100)` | 100~10000 | 1000 |
| knn | `st.number_input("k-NN 이웃 수 (knn)", 1, 50, 9, 1)` | 1~50 | 9 |
| top_k_ratio | `st.slider("Top-k 비율 (top_k_ratio)", 0.0, 1.0, 0.1, 0.01)` | 0.0~1.0 | 0.1 |

---

#### FR-T2-14 (M): Threshold 설정 + 디바이스 감지

| 항목 | 내용 |
|------|------|
| **Threshold 방식** | `st.radio("Threshold 방식", ["Percentile (백분위)", "Absolute (절대값)"], horizontal=True)` <br> "Percentile" → `"percentile"`, "Absolute" → `"absolute"` |
| **Threshold 값 — percentile** | `st.slider("백분위 값", 0.0, 100.0, 95.0, 0.5)` |
| **Threshold 값 — absolute** | `st.slider("절대값", 0.0, 1.0, 0.5, 0.01)` |
| **디바이스 감지** | 탭2 최초 렌더링 시 `torch.cuda.is_available()` 실행 <br> → `device_info` 구성 후 Write <br> → `st.info(f"현재 디바이스: {'CUDA (' + gpu_name + ')' if device=='cuda' else 'CPU'}")` |
| **디바이스 감지 실행 조건** | `device_info is None`인 경우에만 실행 (rerun 시 재감지 방지) |

---

#### FR-T2-15 (S): 정상/결함 비율 실시간 표시

| 항목 | 내용 |
|------|------|
| **설명** | 현재 threshold 설정 기준으로 정상/결함 판정 비율 예상치를 표시한다 |
| **표시 조건** | `dataset_meta != None` AND `preprocessing_config != None` |
| **계산 방식** | threshold_method == "percentile": `정상 비율 = threshold_value / 100` (근사치 표시) <br> threshold_method == "absolute": 계산 불가 → "학습 후 확인 가능" 표시 |
| **Streamlit 컴포넌트** | `st.metric("예상 정상 판정 비율", f"{normal_ratio:.1%}")`, `st.metric("예상 결함 판정 비율", f"{defect_ratio:.1%}")` |

---

### B.4 탭3 — 학습 시작 + 학습 로그

---

#### FR-T3-01 (M): 실험명 입력 및 자동 생성

| 항목 | 내용 |
|------|------|
| **설명** | 실험명을 입력받는다. 빈칸이면 자동 생성한다 |
| **Streamlit 컴포넌트** | `st.text_input("실험 이름 (비워두면 자동 생성)", max_chars=64, key="exp_name_input")` |
| **자동 생성 규칙** | `{model_type}_{YYYYMMDD}_{HHMMSS}_{uuid4().hex[:4]}` (R-NAMING-03) <br> 예: `efficientad_20260508_140023_7f3a` |
| **유효성 검증** | 입력된 경우: 길이 1~64자, 영문·숫자·한글·하이픈·언더스코어만 허용 <br> 허용하지 않는 문자 포함 시 `st.error()` + 저장 차단 |

---

#### FR-T3-02 (M): 학습 전 설정 요약 표시

| 항목 | 내용 |
|------|------|
| **설명** | [학습 시작] 버튼 위에 현재 설정 요약을 표시하여 사용자가 확인할 수 있게 한다 |
| **표시 항목** | 데이터셋 경로, 전처리 방식, 모델 타입, 주요 파라미터 3~5개 (model_size 또는 backbone, image_size, batch_size, random_seed) |
| **Streamlit 컴포넌트** | `st.json(summary_dict, expanded=False)` |

---

#### FR-T3-03 (M): 학습 실행 제어 버튼

| 항목 | 내용 |
|------|------|
| **[학습 시작] 버튼** | `st.button("🚀 학습 시작", type="primary", disabled=(current_run_status == "running"))` |
| **[학습 중지] 버튼** | `st.button("⏹ 학습 중지", type="secondary", disabled=(current_run_status != "running"))` |
| **상태 표시** | `current_run_status == "running"` → `st.warning("학습 진행 중입니다...")` |
| **[학습 시작] 동작** | 1. `current_run_status = "running"` <br> 2. `threading.Thread(target=training_worker, args=(queue, config), daemon=True).start()` <br> 3. 메인 루프 진입 (FR-T3-04) |
| **[학습 중지] 동작** | 1. `stop_event.set()` (threading.Event) <br> 2. 백그라운드 스레드가 stop_event 감지 후 종료 <br> 3. FR-T3-06 중단 처리 실행 |

---

#### FR-T3-04 (M): 진행률 + 실시간 차트 갱신

| 항목 | 내용 |
|------|------|
| **설명** | 학습 중 Progress Bar, Loss 곡선, 로그 텍스트를 주기적으로 갱신한다 |
| **갱신 메커니즘** | `st.empty()` 컨테이너 + `time.sleep(0.3)` + `st.rerun()` 루프 |
| **Progress Bar** | `st.progress(current_step / total_steps, text=f"진행: {current_step}/{total_steps} ({pct:.1f}%)")` |
| **EfficientAD Loss 갱신 주기** | 매 500 step마다 queue에 `{"step": int, "loss": float}` 전송 (가정 A-08) |
| **PatchCore Loss 갱신 주기** | 에포크 단위 (PatchCore는 단일 에포크 학습) |
| **Loss 곡선 차트** | `st.line_chart(data={"step": [...], "loss": [...]})` 또는 `plotly` 라인 차트 |
| **로그 텍스트** | `st.text_area("학습 로그", value="\n".join(log_lines[-100:]), height=200, disabled=True)` <br> 최신 100줄 유지 (가정 A-09) <br> 로그 형식: `[Step {n}/{total}] Loss: {loss:.4f} | 경과: {elapsed:.1f}s` |
| **EfficientAD 로그 파일 저장** | `./logs/{exp_id}.log` 에 전량 저장 (00_Global_Context 7.3절 형식) |

---

#### FR-T3-05 (M): 학습 완료 처리

| 항목 | 내용 |
|------|------|
| **설명** | 학습 루프 정상 종료 시 메트릭 계산, 히스토리 저장, 모델 저장을 수행한다 |
| **메트릭 계산** | `utils/metrics.py`의 `compute_metrics(y_true, anomaly_scores, threshold) -> dict` <br> 반환: accuracy, precision, recall, f1_score, f2_score, auc, confusion_matrix, anomaly_scores 배열 |
| **experiment 레코드 구성** | 00_Global_Context 1.1절 스키마 전체 필드 |
| **history.json 저장** | `save_history(records)` — R-ATOMIC-01 원자적 쓰기 |
| **model_state_dict 저장** | `torch.save(model.state_dict(), f"./models/{exp_id}/model_state_dict.pth")` |
| **configs.yaml 복사** | 현재 `./configs.yaml` → `./models/{exp_id}/configs.yaml` |
| **완료 알림** | `st.success(f"학습이 완료되었습니다. 소요 시간: {m}분 {s}초")` |
| **상태 초기화** | `current_run_status = "idle"` |

---

#### FR-T3-06 (M): 학습 중단 처리

| 항목 | 내용 |
|------|------|
| **설명** | [학습 중지] 클릭 시 학습 결과를 폐기하고 히스토리에 중단 상태로 기록한다 |
| **동작** | 1. `stop_event.set()` → 백그라운드 스레드 종료 대기 (timeout=5s) <br> 2. `./models/{exp_id}/` 디렉토리 미생성 (model_path = null) <br> 3. experiment 레코드: `status="중단"`, `metrics=null`, `model_path=null`, `configs_path=null` <br> 4. `save_history(records)` <br> 5. `st.warning(MSG["TRAIN_STOPPED"])` <br> 6. `current_run_status = "idle"` |

---

#### FR-T3-07 (S): 실험명 자동 생성 표시

| 항목 | 내용 |
|------|------|
| **설명** | 실험명 입력란이 비어있으면 자동 생성될 이름을 미리보기로 표시한다 |
| **Streamlit 컴포넌트** | `st.caption(f"자동 생성될 이름: {auto_name}")` — 입력란 아래에 표시 |

---

#### FR-T3-08 (S): 중단 실험 히스토리 기록

| 항목 | 내용 |
|------|------|
| **설명** | 중단 시 히스토리에 "중단" 상태 레코드를 남긴다 (FR-T3-06과 동일, Should 항목으로 명시) |
| **표시** | 탭4 테이블에서 status="중단" 행은 회색 텍스트로 렌더링 |

---

### B.5 탭4 — 실험 히스토리 + 결과 상세 + 모델 저장

---

#### FR-T4-01 (M): 실험 목록 테이블 렌더링

| 항목 | 내용 |
|------|------|
| **설명** | history.json을 로드하여 실험 목록을 테이블로 표시한다 |
| **데이터 로드** | `load_history()` 호출 (탭4 진입 시마다 재로드하여 최신 상태 반영) |
| **테이블 컬럼** | 실험명 / 모델 / 파라미터 요약 / Accuracy / Precision / Recall / F1 / F2 / AUC / 실행 시각 / 상태 |
| **파라미터 요약 형식** | EfficientAD: `medium/70k/adam` / PatchCore: `wrn50/0.1` |
| **중단 실험 표시** | status="중단" 행: 지표 컬럼에 "—" 표시 |
| **정렬** | 기본: created_at 내림차순 (최신 실험 상단) |
| **Streamlit 컴포넌트** | `st.dataframe(df, use_container_width=True, selection_mode="single-row", on_select="rerun")` |
| **선택 반영** | 선택된 행의 experiment_id → `selected_experiment_id` Write |

---

#### FR-T4-02 (M): 실험 상세 결과 표시

| 항목 | 내용 |
|------|------|
| **설명** | selected_experiment_id의 상세 결과를 시각화한다 |
| **표시 조건** | `selected_experiment_id != None` AND `status == "completed"` |
| **지표 카드** | `st.columns(4)` 에 Accuracy, Precision, Recall, F1 각 `st.metric()` |
| **Confusion Matrix** | Plotly heatmap: 행=실제(정상/결함), 열=예측(정상/결함) <br> 셀 값: TP/FP/TN/FN 수치 표시 |
| **ROC Curve** | Plotly line chart: FPR(x) vs TPR(y), AUC 값 범례에 표시 <br> 데이터: `metrics.anomaly_scores` + `metrics.image_labels` → `sklearn.metrics.roc_curve` |
| **Anomaly Score 분포** | Plotly histogram: 정상(image_label=0)과 결함(image_label=1) 겹쳐서 표시 <br> x축: Anomaly Score, y축: 이미지 수 <br> 현재 threshold 값을 수직선으로 표시 (`add_vline`) |

---

#### FR-T4-03 (M): 실험 삭제

| 항목 | 내용 |
|------|------|
| **설명** | 선택된 실험을 히스토리와 파일시스템에서 삭제한다 |
| **Streamlit 컴포넌트** | `st.button("🗑 실험 삭제", type="secondary")` |
| **확인 절차** | `st.warning("삭제 후 복구할 수 없습니다. 계속하시겠습니까?")` + `[확인]` / `[취소]` 버튼 |
| **[확인] 동작** | UC-10 주 플로우 4번 항목 전체 실행 |
| **비활성화 조건** | `selected_experiment_id is None` 시 버튼 `disabled=True` |

---

#### FR-T4-04 (M): 모델 저장

| 항목 | 내용 |
|------|------|
| **설명** | 선택된 실험의 모델을 state_dict + configs.yaml 형식으로 저장한다 |
| **표시 조건** | `selected_experiment_id != None` AND `status == "completed"` |
| **저장 경로 입력** | `st.text_input("저장 경로", value=f"./models/{selected_experiment_id}/")` |
| **[모델 저장] 버튼** | `st.button("💾 모델 저장", type="primary")` |
| **저장 전 검사** | `shutil.disk_usage(path).free < 500 * 1024**2` → `st.warning("디스크 여유 공간 부족")` |
| **저장 동작** | UC-08 주 플로우 5~7 실행 |
| **완료 출력** | `st.success(f"저장 완료\n경로: {path}\n파일: model_state_dict.pth, configs.yaml\n용량: {size_mb:.1f} MB")` |

---

#### FR-T4-05 (M): 탭4 Guard

| 항목 | 내용 |
|------|------|
| **설명** | 실험이 없으면 탭4 핵심 기능을 차단한다 |
| **조건** | `len(load_history()) == 0` |
| **처리** | `st.warning(MSG["NO_EXPERIMENTS"]); return` |

---

#### FR-T4-06 (M): 탭4 진입 시 history.json 재로드

| 항목 | 내용 |
|------|------|
| **설명** | 탭4 진입 시마다 history.json을 재로드하여 탭3 완료 직후의 실험도 즉시 반영한다 |
| **구현** | 탭4 함수 최상단에서 `experiments = load_history()` 호출 후 `session_state.experiments` 갱신 |

---

#### FR-T4-07 (S): 다중 실험 비교 차트

| 항목 | 내용 |
|------|------|
| **설명** | 다중 실험을 선택하여 메트릭 비교 차트를 렌더링한다 |
| **다중 선택** | 테이블 `selection_mode="multi-row"` 또는 별도 체크박스 (구현자 선택) |
| **최대 선택 수** | 10개 (가정 A-13). 초과 시 `st.warning()` |
| **비교 메트릭 선택** | `st.multiselect("비교 메트릭", ["Accuracy", "Precision", "Recall", "F1", "F2"])` |
| **차트 유형 선택** | `st.radio("차트 유형", ["막대 차트", "레이더 차트"], horizontal=True)` |
| **차트 라이브러리** | Plotly |
| **표시 조건** | 선택 실험 수 ≥ 2 AND 비교 메트릭 ≥ 1 |

---

#### FR-T4-08 (S): 저장 완료 정보 출력

| 항목 | 내용 |
|------|------|
| **설명** | 모델 저장 완료 시 경로·파일명·용량을 명확히 출력한다 (FR-T4-04 완료 메시지와 동일) |

---

#### FR-T4-09 (S): 중단 실험 시각적 구분

| 항목 | 내용 |
|------|------|
| **설명** | 테이블에서 status="중단" 행을 시각적으로 구분한다 |
| **구현** | Pandas DataFrame에 스타일 적용: `df.style.apply(lambda row: ["color: gray"] * len(row) if row["상태"] == "중단" else [""] * len(row), axis=1)` |

---

### B.6 탭5 — 이상 영역 시각화

---

#### FR-T5-01 (M): 탭5 Guard

| 항목 | 내용 |
|------|------|
| **설명** | selected_experiment_id가 None이면 탭5 핵심 기능을 차단한다 |
| **조건** | `selected_experiment_id is None` |
| **처리** | `st.warning(MSG["NO_SELECTED_EXP"]); return` |

---

#### FR-T5-02 (M): 테스트 이미지 목록 테이블

| 항목 | 내용 |
|------|------|
| **설명** | 선택된 실험의 테스트 이미지 목록을 테이블로 표시한다 |
| **데이터 소스** | `experiment.metrics.anomaly_scores`, `experiment.metrics.image_labels`, `dataset_path/test/` |
| **테이블 컬럼** | 이미지명 / Anomaly Score / 판정(OK/NG) / GT 일치 여부 / 오분류(FP/FN/TN/TP) |
| **판정 계산** | `score >= threshold → "NG"`, `score < threshold → "OK"` |
| **GT 일치** | 판정 결과 == image_label (0=정상=OK, 1=결함=NG) 이면 "✓", 아니면 "✗" |
| **오분류 분류** | label=0, pred=NG → FP / label=1, pred=OK → FN / label=0, pred=OK → TN / label=1, pred=NG → TP |
| **Streamlit 컴포넌트** | `st.dataframe(df, use_container_width=True, selection_mode="single-row", on_select="rerun")` |

---

#### FR-T5-03 (M): Threshold 슬라이더 실시간 갱신

| 항목 | 내용 |
|------|------|
| **설명** | Threshold 슬라이더 조정 시 테이블 판정 결과와 3분할 시각화를 실시간 갱신한다 |
| **Streamlit 컴포넌트** | `st.slider("Threshold", 0.0, float(max(anomaly_scores))*1.2, threshold_value, 0.001, format="%.4f")` |
| **초기값** | `anomaly_map_threshold if anomaly_map_threshold is not None else experiment.threshold_value` |
| **갱신 대상** | 1. 테이블의 판정/GT 일치/오분류 컬럼 <br> 2. Anomaly Map 이진화 마스크 (이미지 선택 시) |
| **Write** | `st.session_state.anomaly_map_threshold = current_threshold` |

---

#### FR-T5-04 (M): 3분할 시각화

| 항목 | 내용 |
|------|------|
| **설명** | 선택된 이미지에 대해 원본 / GT 마스크 / Anomaly Map Heatmap을 3분할로 표시한다 |
| **표시 조건** | 테이블에서 이미지 선택 시 |
| **레이아웃** | `col1, col2, col3 = st.columns(3)` |
| **col1 — 원본** | 선택된 테스트 이미지 원본 (`st.image(orig, caption="원본 이미지")`) |
| **col2 — GT 마스크** | `ground_truth/{class}/{image_name}` 로드 → 이진 마스크 표시 <br> 파일 없으면 빈 마스크(전체 검정) 표시 (가정 A-10) |
| **col3 — Heatmap** | Anomaly Map을 jet colormap으로 변환 → `st.image(heatmap, caption="Anomaly Map (Heatmap)")` <br> Anomaly Map 위에 threshold 기준 이진화 경계선 오버레이 |
| **이미지 크기 통일** | 세 이미지 모두 원본 이미지 크기 기준으로 표시 |

---

#### FR-T5-05 (M): PNG 저장

| 항목 | 내용 |
|------|------|
| **설명** | 3분할 이미지를 단일 PNG 파일로 저장한다 |
| **Streamlit 컴포넌트** | `st.download_button("📥 PNG 저장", data=png_bytes, file_name=f"{exp_id}_{image_name}_anomaly.png", mime="image/png")` |
| **PNG 구성** | 원본 / GT / Heatmap을 가로로 이어 붙인 단일 이미지 (각 패널 동일 크기) |
| **라이브러리** | `PIL.Image`, `numpy`, `matplotlib.cm.jet` |

---

#### FR-T5-06 (S): 결함 유형 필터 드롭다운

| 항목 | 내용 |
|------|------|
| **설명** | 결함 유형별로 테스트 이미지 목록을 필터링한다 |
| **Streamlit 컴포넌트** | `st.selectbox("결함 유형 필터", ["전체"] + dataset_meta.defect_classes)` |
| **"전체" 선택** | 필터 없이 전체 이미지 표시 |
| **특정 클래스 선택** | 해당 `test/{class}/` 경로의 이미지만 테이블에 표시 |

---

#### FR-T5-07 (S): 이미지별 Anomaly Score 요약 표시

| 항목 | 내용 |
|------|------|
| **설명** | 테이블 상단에 현재 표시 중인 이미지들의 최대·평균 Anomaly Score를 표시한다 |
| **Streamlit 컴포넌트** | `st.columns(2)` 에 `st.metric("최대 Anomaly Score", f"{max_score:.4f}")`, `st.metric("평균 Anomaly Score", f"{mean_score:.4f}")` |

---

#### FR-T5-08 (S): FP/FN 표시

| 항목 | 내용 |
|------|------|
| **설명** | 현재 threshold 기준 FP/FN 수를 요약 표시한다 |
| **Streamlit 컴포넌트** | `st.columns(4)` 에 TP/FP/TN/FN 수 `st.metric()` 표시 |
| **갱신 조건** | Threshold 슬라이더 변경 시 자동 갱신 |

---

## C. System & Data Design

모든 데이터 스키마는 [00_Global_Context_Document.md 1절](./00_Global_Context_Document.md#1-core-data-model)에서 확정된 것을 그대로 사용한다. 이 문서에서 재정의하지 않는다.

### C.1 FR별 session_state 의존성 매핑

| FR ID | Read 키 | Write 키 |
|-------|---------|----------|
| FR-T1-01~06 | - | `dataset_path`, `dataset_meta` |
| FR-T2-01~15 | `dataset_path`, `dataset_meta` | `preprocessing_config`, `model_config`, `device_info` |
| FR-T3-01~08 | `dataset_path`, `preprocessing_config`, `model_config`, `device_info` | `experiments[exp_id]`, `current_run_status`, `current_exp_id` |
| FR-T4-01~09 | `experiments` | `selected_experiment_id` |
| FR-T5-01~08 | `selected_experiment_id`, `experiments`, `dataset_meta` | `anomaly_map_threshold` |

### C.2 파일 I/O 의존성 매핑

| FR ID | 읽기 파일 | 쓰기 파일 |
|-------|-----------|-----------|
| FR-T2-05 | `./configs.yaml` (선택적) | `./configs.yaml` (preprocessing 섹션 + model 섹션) |
| FR-T3-05 | - | `./experiments/history.json`, `./models/{exp_id}/model_state_dict.pth`, `./models/{exp_id}/configs.yaml`, `./logs/{exp_id}.log` |
| FR-T3-06 | - | `./experiments/history.json` |
| FR-T4-01 | `./experiments/history.json` | - |
| FR-T4-03 | - | `./experiments/history.json` (삭제) |
| FR-T4-04 | `./models/{exp_id}/model_state_dict.pth` | 지정 경로 |

---

## D. API Contracts

```
N/A — REST API 없음. 내부 인터페이스는 00_Global_Context_Document.md 3절 참조.
```

---

## E. AI/ML Details

```
N/A — 이 문서는 기능 명세 범위이다.
      학습 루프, 모델 초기화, 메트릭 계산 알고리즘의 상세 구현은
      08_AI_ML_Integration.md에서 다룬다.
      여기서는 FR-T3 탭에서 ML 기능을 트리거하는 UI 계약만 정의한다.
```

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체 상속.

이 문서에서 추가로 명시하는 항목:

| 항목 | 요구사항 | 관련 FR |
|------|----------|---------|
| **미리보기 갱신 속도** | 탭2 파라미터 변경 후 미리보기 갱신 ≤ 2초 | FR-T2-04 |
| **테이블 로드 속도** | 실험 ≤ 100개 기준 탭4 테이블 초기 렌더링 ≤ 1초 | FR-T4-01 |
| **Threshold 슬라이더 응답** | 슬라이더 조작 후 테이블 갱신 ≤ 0.5초 (재추론 없이 기존 scores 재계산만) | FR-T5-03 |
| **configs.yaml 쓰기 안전성** | R-ATOMIC-01 (임시 파일 → rename) | FR-T2-05 |
| **history.json 쓰기 안전성** | R-ATOMIC-01 | FR-T3-05, FR-T3-06, FR-T4-03 |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards) 전체 상속.

FR별 로그 이벤트 매핑:

| FR ID | 이벤트명 | 레벨 |
|-------|---------|------|
| FR-T1-01 (성공) | `dataset_validated` | INFO |
| FR-T1-01 (실패) | `dataset_validation_failed` | ERROR |
| FR-T2-05 | `preprocessing_config_saved` | INFO |
| FR-T2-05 | `model_config_saved` | INFO |
| FR-T3-03 (시작) | `training_started` | INFO |
| FR-T3-04 (step) | `training_step` | INFO |
| FR-T3-05 (완료) | `training_completed` | INFO |
| FR-T3-06 (중단) | `training_stopped` | WARNING |
| FR-T3-05 (실패) | `training_failed` | ERROR |
| FR-T4-04 (저장) | `model_saved` | INFO |
| FR-T4-03 (삭제) | `experiment_deleted` | WARNING |

---

## H. QA & Validation

### H.1 FR 완료 기준 체크리스트

모든 M 항목(39개)을 구현하고 아래 체크리스트를 통과해야 MVP 완료로 판정한다. (탭2+탭3 통합으로 전체 FR 수 52개, M 39개 유지)

#### 탭1

- [ ] FR-T1-01: 존재하지 않는 경로 입력 시 ERR_DATASET_NOT_FOUND 표시
- [ ] FR-T1-01: 올바른 MVTec AD 경로 입력 시 검증 통과 + dataset_meta 구성
- [ ] FR-T1-03: 폴더 트리 최대 3단계 렌더링
- [ ] FR-T1-04: 클래스별 이미지 수 테이블 + 합계 행
- [ ] FR-T1-05: 결함 클래스 수 ≤ 4 시 1행, > 4 시 다음 행으로 wrap
- [ ] FR-T1-06: channels == 1 시 MSG["GRAYSCALE_DETECT"] 표시

#### 탭2 (전처리 영역)

- [ ] FR-T2-01: HE 선택 시 슬라이더 DOM 미존재
- [ ] FR-T2-02: 모든 슬라이더 범위 내 값만 허용
- [ ] FR-T2-03: image_size 32의 배수 아닌 값 입력 시 st.error()
- [ ] FR-T2-04: 슬라이더 변경 후 2초 이내 미리보기 갱신
- [ ] FR-T2-05: [설정 저장] 후 session_state.preprocessing_config 및 session_state.model_config 갱신

#### 탭2 (모델 영역)

- [ ] FR-T2-08: PatchCore 선택 시 EfficientAD 파라미터 DOM 미존재
- [ ] FR-T2-10: ae_loss_weight 슬라이더 0.0~1.0 범위 정상 동작
- [ ] FR-T2-14: torch.cuda.is_available() 결과 사이드바에 반영

#### 탭3

- [ ] FR-T3-03: 학습 중 [학습 시작] disabled
- [ ] FR-T3-04: 500 step마다 Loss 차트 갱신
- [ ] FR-T3-04: 로그 텍스트 최신 100줄 유지
- [ ] FR-T3-05: 완료 후 ./models/{exp_id}/ 에 .pth + configs.yaml 존재
- [ ] FR-T3-06: 중지 후 history.json에 status="중단" 레코드 존재

#### 탭4

- [ ] FR-T4-01: 탭4 진입 시 history.json 재로드 반영
- [ ] FR-T4-02: ROC Curve AUC 값 범례에 표시
- [ ] FR-T4-02: Anomaly Score 분포에 threshold 수직선 표시
- [ ] FR-T4-03: 삭제 확인 팝업 후 삭제, ./models/{exp_id}/ 제거
- [ ] FR-T4-04: 저장 후 경로·파일명·용량 출력

#### 탭5

- [ ] FR-T5-02: 테이블에 FP/FN/TP/TN 분류 표시
- [ ] FR-T5-03: 슬라이더 변경 후 0.5초 이내 테이블 갱신
- [ ] FR-T5-04: GT 없는 이미지에 빈 마스크 표시
- [ ] FR-T5-05: PNG 다운로드 버튼 정상 동작

### H.2 Given-When-Then 시나리오

#### TC-FR-T2-10: ae_loss_weight 저장 검증

```
Given:  탭2 모델 영역에서 EfficientAD 선택됨
        ae_loss_weight=0.5 초기 상태
When:   ae_loss_weight 슬라이더를 0.73으로 변경 후 [설정 저장] 클릭
Then:   st.session_state.model_config["params"]["ae_loss_weight"] == 0.73
        학습 루프에서 ST 비중은 (1 - 0.73) = 0.27 이 자동 적용됨
```

#### TC-FR-T3-05: 학습 완료 후 파일 검증

```
Given:  dataset_path, preprocessing_config, model_config 모두 설정됨
        model_type = "patchcore", coreset_sampling_ratio = 0.1
When:   [학습 시작] 클릭 → 학습 완료 대기
Then:   history.json 에 status="completed" 레코드 1개 추가됨
        exp_id = "patchcore_{YYYYMMDD}_{HHMMSS}_{4자리}" 형식
        ./models/{exp_id}/model_state_dict.pth 파일 존재
        ./models/{exp_id}/configs.yaml 파일 존재
        ./logs/{exp_id}.log 파일 존재
        experiment.duration_seconds <= 600 (g4dn.xlarge 기준)
        experiment.metrics.auc >= 0.0 (값 존재 확인)
```

#### TC-FR-T5-03: Threshold 슬라이더 실시간 갱신

```
Given:  탭5에서 실험 선택됨, 이미지 목록 테이블 표시 중
        initial_threshold = 0.5
        테이블 첫 번째 이미지 score = 0.6 → 초기 판정 "NG"
When:   Threshold 슬라이더를 0.7로 변경한다
Then:   0.5초 이내 테이블 재렌더링
        score=0.6 < threshold=0.7 → 해당 이미지 판정 "OK"로 변경
        GT label=1(결함) AND 판정="OK" → 오분류 "FN"으로 변경
        session_state.anomaly_map_threshold == 0.7
```

#### TC-FR-T4-03: 실험 삭제 안전성

```
Given:  history.json 에 실험 2개 존재 (exp_A, exp_B)
        selected_experiment_id = "exp_A"
        ./models/exp_A/ 디렉토리 존재
When:   탭4에서 exp_A 선택 후 [실험 삭제] → [확인] 클릭
Then:   history.json 에 exp_A 레코드 없음, exp_B 레코드 유지
        session_state.selected_experiment_id == None
        ./models/exp_A/ 디렉토리 없음
        ./logs/exp_A.log 없음
        탭5 진입 시 MSG["NO_SELECTED_EXP"] 표시
```

---

## I. Implementation Plan

```
N/A — 전체 구현 순서·WBS·소요 시간은 14_Deployment_and_Release_Plan.md에서 다룬다.
      이 문서의 FR ID(FR-T1-01 ~ FR-T5-08)는
      14_Deployment_and_Release_Plan.md의 WBS 작업 항목으로 직접 매핑된다.
```

---

*다음 문서*: [04_System_Architecture.md](./04_System_Architecture.md)
