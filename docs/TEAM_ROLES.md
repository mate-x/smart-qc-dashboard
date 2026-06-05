# 비전검사 대시보드 — 3인 팀 역할 분담표

> 프로젝트: 제조산업 품질검사를 위한 딥러닝 기반 비전검사 최적 모델 탐색 대시보드 v1.0 (MVP)
> 기간: 2일 (Day 1 / Day 2, 하루 8시간 풀타임)
> 범위: Must Have(M) + Should Have(S) 기능만 포함

---

## 1. 팀 구성 및 역할 개요

| 팀원 | 사전 학습 배경 | 주 담당 영역 | 보조 담당 영역 |
|------|--------------|------------|--------------|
| **A** | 이미지 전처리 (Homomorphic, HE, CLAHE) | 탭2 전처리 파라미터 설정 | 탭1 데이터 폴더 구조, 탭6 Anomaly Map 시각화, 공통 인프라 |
| **B** | EfficientAD 모델 구조 및 학습 | 탭3 EfficientAD 파라미터, 탭4 EfficientAD 학습 | 탭5 실험 히스토리 (전반부), 공통 인프라 |
| **C** | PatchCore 모델 구조 및 학습 | 탭3 PatchCore 파라미터, 탭4 PatchCore 학습 | 탭5 모델 저장·비교 (후반부), 공통 인프라 |

---

## 2. Day 1 / Day 2 타임라인

### Day 1

| 시간 블록 | 팀원 A | 팀원 B | 팀원 C |
|----------|--------|--------|--------|
| **오전 1h**  (09:00–10:00) | 공통: configs.yaml 스키마 설계·문서화 | 공통: session_state 키 명세 확정·문서화 | 공통: 프로젝트 구조·Git 초기 세팅, requirements.txt 초안 |
| **오전 2h** (10:00–12:00) | 탭1 구현: MVTec AD 폴더 구조 검증, 트리 시각화, 이미지 수 카운트, 썸네일 렌더링 | 탭3 EfficientAD 파트: 공통 파라미터 UI, EfficientAD 전용 파라미터(기본 노출 항목), ae/st weight 슬라이더 | 탭3 PatchCore 파트: PatchCore 전용 파라미터(backbone, coreset ratio 등), Pretrained Weights 방식 선택 |
| **오후 1h** (13:00–14:00) | 탭1 보완: Grayscale 자동 감지·RGB 변환 안내, 지원 포맷 필터링, 경고 메시지 | 탭3 EfficientAD 보완: 고급 설정 expander, Threshold UI, 디바이스 자동 감지 표시 | 탭3 PatchCore 보완: 고급 설정 expander, Threshold UI, image_size 탭2 연동 반영 |
| **오후 3h** (14:00–17:00) | 탭2 구현: 전처리 라디오 선택, Homomorphic·HE·CLAHE 파라미터 UI, 적용 전·후 미리보기, Resize+Padding, 정규화 선택, configs.yaml 저장/불러오기 | 탭4 EfficientAD 파트: Anomalib 기반 EfficientAD 학습 루프, Progress Bar, 실시간 Loss 곡선, 로그 텍스트 박스 | 탭4 PatchCore 파트: Anomalib 기반 PatchCore 학습 루프, Progress Bar, 실시간 Loss 곡선, 로그 텍스트 박스 |
| **오후 버퍼** (17:00–18:00) | 탭2 엣지케이스 처리, Claude Code로 코드 리뷰·리팩터링 | 탭4 EfficientAD: 학습 중지 처리(데이터 폐기, "중단" 상태 기록), 실험명 자동 생성 | 탭4 PatchCore: 학습 중지 처리(동일 규격), 완료 알림 및 소요 시간 표시 |

### Day 2

| 시간 블록 | 팀원 A | 팀원 B | 팀원 C |
|----------|--------|--------|--------|
| **오전 1h** (09:00–10:00) | 탭3·탭4 통합 지원: EfficientAD·PatchCore 파트 session_state 연결 확인, 탭 간 흐름 검증 | 탭4 마무리: session_state.experiments[exp_id] 갱신, history.json 파일 저장 구조 완성 | 탭4 마무리: EfficientAD 파트와 PatchCore 파트 통합, 단일 탭4 UI로 병합 |
| **오전 3h** (10:00–13:00) | 탭6 구현: 탭5 실험 선택 연동, 테스트 이미지 목록 테이블, 결함 유형 필터 드롭다운, 3분할 시각화(원본/GT/Heatmap), Threshold 슬라이더 실시간 갱신, PNG 저장 | 탭5 전반부: 실험 목록 테이블 렌더링, 실험 선택 시 상세 결과(Confusion Matrix, ROC, Anomaly Score 분포), 실험 삭제 기능 | 탭5 후반부: 다중 실험 선택 비교 차트(Accuracy/Precision/Recall/F1/F2), 모델 저장(state_dict + configs.yaml), 저장 완료 경로·용량 출력 |
| **오후 1h** (13:00–14:00) | 탭6 보완: 이미지별 최대/평균 Anomaly Score 표시, FP/FN 표시, 다중 이미지 PNG 내보내기(S 항목) | 탭5 UI 통합: 전반부·후반부 연결, selected_experiment_id 공유 확인 | Dockerfile 작성, docker-compose.yml, .env 구성, GPU 실행 테스트 |
| **오후 2h** (14:00–16:00) | **통합 테스트** — 탭1→탭2→탭3→탭4→탭5→탭6 전체 플로우 E2E 검증 | **통합 테스트** — EfficientAD 전체 실험 사이클(설정→학습→평가→저장) 검증, session_state 흐름 확인 | **통합 테스트** — PatchCore 전체 실험 사이클 검증, Docker 빌드·실행 최종 확인 |
| **오후 버퍼** (16:00–18:00) | 버그 수정, 한국어 UI 일관성 검토, 안내 메시지 표준 문구 적용 | 버그 수정, 재현성(random_seed) 확인, 로그 파일 저장(`./logs/{exp_id}.log`) 확인 | 버그 수정, 성능 목표(EfficientAD 20분·PatchCore 10분 이내) 측정 기록 |

---

## 3. 탭별 상세 역할 분담표

| 탭 | 기능명 | 담당자 | 주요 구현 항목 (M/S) | 산출물 | 비고 |
|----|--------|--------|----------------------|--------|------|
| **탭1** | 데이터 폴더 구조 | **A** | MVTec AD 폴더 구조 검증, 트리 시각화, 폴더별 이미지 수 표시, 대표 샘플 썸네일, Grayscale 자동 감지 → RGB 변환 안내 (M) / 폴더 구조 오류 경고, 지원 포맷 외 파일 필터링 (S) | `session_state.dataset_path`, `session_state.dataset_meta` | 사이드바 상시 노출 경로 표시 포함 |
| **탭2** | 전처리 파라미터 설정 | **A** | 전처리 라디오 선택(None/Homomorphic/HE/CLAHE), 적용 전·후 미리보기, Resize+Padding 고정, resize 크기 입력, 정규화 방식 선택, configs.yaml 저장/불러오기 (M) / 비선택 파라미터 UI 완전 숨김, Grayscale 자동 RGB 변환 (S) | `session_state.preprocessing_config`, `preprocessing` 섹션 configs.yaml | GAN 증강·적용 순서 설정은 MVP 제외 |
| **탭3 (EfficientAD)** | 모델 파라미터 설정 — EfficientAD | **B** | EfficientAD 라디오 선택, 공통 파라미터, EfficientAD 전용 파라미터(기본 노출), ae/st weight 합산 1.0 슬라이더, 고급 설정 expander, Threshold 방식 선택, 디바이스 자동 감지 표시, configs.yaml 저장/불러오기 (M) / image_size 탭2 연동, 정상/결함 비율 실시간 표시 (S) | `session_state.model_config` (EfficientAD 파트), `model` 섹션 configs.yaml | C와 탭3 UI 구조 합의 후 단일 탭으로 통합 |
| **탭3 (PatchCore)** | 모델 파라미터 설정 — PatchCore | **C** | PatchCore 라디오 선택, 공통 파라미터, PatchCore 전용 파라미터(기본 노출), Pretrained Weights 방식, 고급 설정 expander, Threshold 방식 선택, 디바이스 자동 감지 표시, configs.yaml 저장/불러오기 (M) / image_size 탭2 연동, 정상/결함 비율 실시간 표시 (S) | `session_state.model_config` (PatchCore 파트), `model` 섹션 configs.yaml | B와 탭3 UI 구조 합의 필수 |
| **탭4 (EfficientAD)** | 학습 시작 + 학습 로그 — EfficientAD | **B** | EfficientAD 학습 루프(Anomalib), 진행률 Progress Bar, 실시간 Loss 곡선, 학습 로그 텍스트 박스, [학습 중지] 버튼(데이터 폐기), 완료 알림·소요 시간, 실험명 자동 생성 (M) / 중단 시 "중단" 상태 히스토리 기록 (S) | `session_state.experiments[exp_id]`, `./experiments/history.json`, `./logs/{exp_id}.log` | C 파트와 통합 시 모델 선택 분기 처리 |
| **탭4 (PatchCore)** | 학습 시작 + 학습 로그 — PatchCore | **C** | PatchCore 학습 루프(Anomalib), 진행률 Progress Bar, 실시간 Loss 곡선, 학습 로그 텍스트 박스, [학습 중지] 버튼(데이터 폐기), 완료 알림·소요 시간, 실험명 입력란 (M) / 중단 시 "중단" 상태 히스토리 기록 (S) | `session_state.experiments[exp_id]`, `./experiments/history.json`, `./logs/{exp_id}.log` | B 파트와 Day 2 오전에 단일 탭4로 병합 |
| **탭5 (전반)** | 실험 히스토리 + 결과 상세 | **B** | 실험 목록 테이블(전체 컬럼), 실험 선택 시 상세 결과(Confusion Matrix, ROC, Anomaly Score 분포 히스토그램), 실험 삭제 (M) | `session_state.selected_experiment_id`, 테이블·차트 렌더링 | C 파트와 selected_experiment_id 공유 확인 |
| **탭5 (후반)** | 모델 저장 + 비교 차트 | **C** | 다중 실험 선택 비교 차트(막대/레이더), 모델 저장(state_dict + configs.yaml 고정), 저장 완료 시 경로·파일명·용량 출력 (M/S) | `./models/{exp_id}/model_state_dict.pth`, `./models/{exp_id}/configs.yaml` | 저장 방식 옵션 UI 없음(고정) |
| **탭6** | 이상 영역 시각화 (Anomaly Map) | **A** | 탭5 실험 선택 연동, 테스트 이미지 목록 테이블, 3분할 시각화(원본/GT/Heatmap), Threshold 슬라이더 실시간 갱신, PNG 저장 (M) / 결함 유형 필터 드롭다운, 이미지별 최대/평균 Anomaly Score 표시, FP/FN 표시 (S) | `session_state.anomaly_map_threshold`, PNG 다운로드 파일 | 다중 이미지 일괄 PNG 내보내기는 S 범위 내 여유 시 구현 |

---

## 4. 공통/인프라 작업 분담

| 작업 항목 | 담당자 | 일정 | 주요 내용 | 산출물 |
|----------|--------|------|----------|--------|
| 프로젝트 초기 세팅 | **C** | Day 1 오전 1h | Git 저장소 초기화, 디렉토리 구조 생성(`app/`, `tabs/`, `utils/`, `models/`, `experiments/`, `logs/`), requirements.txt 초안 작성, `.gitignore` | 프로젝트 뼈대 코드, requirements.txt |
| session_state 키 명세 확정 | **B** | Day 1 오전 1h | PRD 7.2절 기준 키 목록 문서화, 초기화 코드 작성(`session_state_init.py`), 탭 간 의존성 확인 | `utils/session_state_init.py`, 명세 문서 |
| configs.yaml 스키마 설계 | **A** | Day 1 오전 1h | PRD 9절 기준 전체 YAML 스키마 확정, 저장/불러오기 유틸 함수 작성(`utils/config_manager.py`) | `utils/config_manager.py`, `configs_template.yaml` |
| 공통 UI 컴포넌트 | **A** | Day 1 오전 (탭1 구현 중 병행) | 사이드바 레이아웃(데이터셋 경로·디바이스 정보), 표준 안내 메시지 상수 정의(`utils/messages.py`), 데이터 없음 가드 함수 | `utils/messages.py`, `components/sidebar.py` |
| Anomalib 연동 유틸 | **B, C** | Day 1 오후 (탭4 구현 전) | Anomalib 버전 확인, EfficientAD·PatchCore 모델 초기화 래퍼 함수, 평가 메트릭 계산 유틸(`utils/metrics.py`) | `utils/model_factory.py`, `utils/metrics.py` |
| Docker 구성 | **C** | Day 2 오후 1h | Dockerfile 작성(PRD 11.1절 기준), `docker-compose.yml`, GPU 볼륨 마운트, 8501/tcp 포트 설정, 빌드·실행 테스트 | `Dockerfile`, `docker-compose.yml` |
| 통합 테스트 | **A, B, C** | Day 2 오후 마지막 2h | 탭1→6 E2E 전체 플로우 검증, EfficientAD·PatchCore 실험 사이클 검증, Docker 빌드 최종 확인, 한국어 UI 일관성 검토, random_seed 재현성 확인, 성능 목표 측정 | 버그 수정 완료 코드, 최종 Docker 이미지 |

---

## 5. 탭 간 의존성 및 인터페이스 합의 항목

> PRD 7.2절 session_state 키 기준. **Write**: 해당 키를 생성·갱신하는 탭/담당자. **Read**: 해당 키를 소비하는 탭/담당자.

| session_state 키 | Write (생성·갱신) | Read (소비) | 합의 필요 사항 |
|-----------------|-----------------|------------|--------------|
| `dataset_path` | 탭1 — **A** | 탭2 — **A**, 탭4 — **B/C** | 경로 검증 실패 시 `None` 유지, 탭4 진입 전 `None` 체크 가드 필요 |
| `dataset_meta` (이미지 수, 채널) | 탭1 — **A** | 탭2 — **A** | dict 구조: `{"train_count": int, "test_count": int, "channels": int}` 형식 A·B·C 합의 |
| `preprocessing_config` | 탭2 — **A** | 탭3 — **B/C** (image_size 연동), 탭4 — **B/C** | `image_size` 키 명 및 타입(int) 사전 합의. 탭3에서 탭2 값 자동 반영 로직은 B/C가 구현 |
| `model_config` | 탭3 — **B** (EfficientAD), **C** (PatchCore) | 탭4 — **B/C** | 단일 dict 구조로 통합: `{"type": "efficientad"/"patchcore", "common": {...}, "model_params": {...}, "threshold": {...}}`. B·C 사전 합의 필수 |
| `device_info` | 탭3 — **B** (디바이스 감지 후 저장) | 탭4 — **B/C** | `{"device": "cuda"/"cpu", "gpu_name": str/"N/A"}` 형식 합의 |
| `experiments[exp_id]` | 탭4 — **B** (EfficientAD), **C** (PatchCore) | 탭5 — **B/C**, 탭6 — **A** | `exp_id` 생성 규칙(`{모델명}_{날짜}_{시간}` 자동 생성), 중단 시 `status: "중단"`, `metrics: None` 처리 방식 B·C 합의 |
| `current_run_status` | 탭4 — **B/C** | 탭4 UI (진행률 갱신) | `"idle"/"running"/"stopped"/"completed"` 상태 값 B·C 합의 |
| `selected_experiment_id` | 탭5 — **B** (목록 선택 시 갱신) | 탭6 — **A** | 탭5 전반(B)이 write, 탭6(A)이 read. `None` 상태에서 탭6 진입 시 안내 메시지 처리 A·B 합의 |
| `anomaly_map_threshold` | 탭6 — **A** | 탭6 내부 (실시간 이진화 갱신) | float 타입, 초기값은 `model_config.threshold.value` 기준. A가 단독 관리 |

### 인터페이스 합의 체크리스트 (Day 1 오전 완료 목표)

- [ ] `dataset_meta` dict 구조 합의 (A 주도)
- [ ] `preprocessing_config` 전체 키 목록 및 타입 합의 (A 주도, B·C 확인)
- [ ] `model_config` 통합 dict 구조 합의 (B·C 공동, A 확인)
- [ ] `experiments[exp_id]` JSON 스키마 합의 (PRD 8.2절 기준, B·C 공동)
- [ ] `current_run_status` 상태 값 열거 합의 (B·C 공동)
- [ ] configs.yaml 로드/세이브 API 확정 (C 주도, A·B 확인)

---

## 6. 완료 조건 체크리스트 (PRD 12절 기준, 담당자별 재분류)

### 팀원 A 담당

#### 탭1 — 데이터 폴더 구조
- [ ] MVTec AD 폴더 구조 검증 동작 (`train/good`, `test/`, `ground_truth/`)
- [ ] 폴더별 이미지 수 정확 표시
- [ ] 대표 샘플 썸네일 렌더링
- [ ] Grayscale 자동 감지 및 RGB 변환 안내
- [ ] 잘못된 구조에 대한 경고 메시지 출력
- [ ] `.jpg`/`.png`/`.bmp` 외 파일 필터링

#### 탭2 — 전처리 파라미터 설정
- [ ] 라디오 선택에 따라 비선택 모델 파라미터 UI 완전 숨김 (disabled 처리 금지)
- [ ] Homomorphic / HE / CLAHE 적용 전·후 미리보기 정상 동작
- [ ] Resize + Padding (검정 0) 고정 적용
- [ ] ImageNet/커스텀 정규화 선택 가능
- [ ] configs.yaml 전처리 섹션 저장/불러오기 동작

#### 탭6 — 이상 영역 시각화
- [ ] 탭5 실험 선택 연동 동작
- [ ] 테스트 이미지 목록 테이블 렌더링
- [ ] 결함 유형별 필터링 동작
- [ ] 3분할 시각화 (원본/GT/Heatmap) 정상 표시
- [ ] Threshold 슬라이더 실시간 이진화 갱신
- [ ] 3분할 PNG 저장 동작

#### 비기능
- [ ] 한국어 UI 일관성 검토 완료 (전체 탭 표준 안내 메시지 적용 확인)

---

### 팀원 B 담당

#### 탭3 (EfficientAD) — 모델 파라미터 설정
- [ ] EfficientAD / PatchCore 라디오 선택 동작 (탭3 전체 UI 통합 책임)
- [ ] image_size 탭2 연동 자동 반영
- [ ] EfficientAD ae/st loss weight 합산 1.0 슬라이더 동작
- [ ] 고급 설정 expander 동작 (EfficientAD 파트)
- [ ] Threshold 방식 선택 및 정상/결함 비율 실시간 표시
- [ ] 디바이스 자동 감지 표시 (`torch.cuda.is_available()`)
- [ ] configs.yaml 모델 섹션 저장/불러오기 동작

#### 탭4 (EfficientAD) — 학습 시작 + 학습 로그
- [ ] 진행률 Progress Bar 실시간 갱신
- [ ] EfficientAD Loss 곡선 실시간 표시
- [ ] 학습 로그 텍스트 박스 스트리밍
- [ ] 학습 중지 시 데이터 폐기·"중단" 상태 기록
- [ ] 완료 시 알림 + 총 소요 시간 표시
- [ ] 실험명 자동 생성 또는 입력란 동작

#### 탭5 (전반) — 실험 히스토리 + 결과 상세
- [ ] 실험 목록 테이블 렌더링 및 정렬 (전체 컬럼: 실험명/모델/파라미터 요약/지표/시각/상태)
- [ ] 실험 선택 시 상세 결과 (Confusion Matrix, ROC Curve + AUC, Anomaly Score 분포) 표시
- [ ] 실험 삭제 동작

#### 비기능
- [ ] 학습 중 UI 블로킹 없음 확인 (EfficientAD 파트)
- [ ] g4dn.xlarge에서 EfficientAD-medium 70k steps **20분 이내** 완료 확인
- [ ] 동일 random_seed 재현성 확인 (EfficientAD)
- [ ] `./logs/{exp_id}.log` 로그 파일 저장 확인

---

### 팀원 C 담당

#### 탭3 (PatchCore) — 모델 파라미터 설정
- [ ] PatchCore 고급 설정 expander 동작
- [ ] PatchCore backbone 선택 및 Pretrained Weights 방식(torchvision/로컬) 동작
- [ ] configs.yaml PatchCore 모델 섹션 저장/불러오기 동작

#### 탭4 (PatchCore) — 학습 시작 + 학습 로그
- [ ] 진행률 Progress Bar 실시간 갱신 (PatchCore 파트)
- [ ] PatchCore Loss 곡선 실시간 표시
- [ ] 학습 로그 텍스트 박스 스트리밍 (PatchCore 파트)
- [ ] 학습 중지 시 데이터 폐기·"중단" 상태 기록 (PatchCore 파트)
- [ ] 완료 시 알림 + 총 소요 시간 표시 (PatchCore 파트)

#### 탭5 (후반) — 비교 차트 + 모델 저장
- [ ] 다중 실험 선택 비교 차트 동작 (Accuracy/Precision/Recall/F1/F2, 막대 또는 레이더)
- [ ] state_dict + configs.yaml 저장 동작 (저장 옵션 UI 없음, 고정 방식)
- [ ] 저장 완료 시 경로·파일명·용량 출력

#### 인프라
- [ ] Docker 이미지 빌드 및 GPU 컨테이너 정상 실행 (`--gpus all`)
- [ ] 볼륨 마운트 동작 확인 (dataset / models / experiments)

#### 비기능
- [ ] 학습 중 UI 블로킹 없음 확인 (PatchCore 파트)
- [ ] g4dn.xlarge에서 PatchCore(coreset 10%) **10분 이내** 완료 확인
- [ ] 동일 random_seed 재현성 확인 (PatchCore)

---

## 부록 — 작업량 균형 요약

| 팀원 | 주요 담당 탭 수 | 공통 작업 비중 | 예상 작업 부하 |
|------|--------------|-------------|-------------|
| **A** | 탭1, 탭2, 탭6 (3개 탭) | 프로젝트 초기 세팅, 공통 UI 컴포넌트 | ★★★ (균등) |
| **B** | 탭3 EfficientAD, 탭4 EfficientAD, 탭5 전반 (3개 파트) | session_state 명세, Anomalib EfficientAD 연동 | ★★★ (균등) |
| **C** | 탭3 PatchCore, 탭4 PatchCore, 탭5 후반 (3개 파트) | configs.yaml 스키마, Anomalib PatchCore 연동, Docker | ★★★ (균등) |

> **Note**: 탭3·탭4는 EfficientAD(B)와 PatchCore(C) 파트로 분리 구현 후 Day 2 오전에 단일 탭으로 통합합니다. 통합 작업은 B·C 공동 진행하며, A는 탭6 구현을 병행합니다.

---

## 7. Claude Code 프롬프트 가이드

> 공통 프롬프트 템플릿: [docs/CLAUDE_PROMPTS.md](./CLAUDE_PROMPTS.md)

### 시간블록별 태스크 조회표

| 일자 | 시간 블록 | 팀원 | 태스크명 | 참조 PRD (00 항상 포함) |
|------|----------|------|---------|----------------------|
| Day 1 | 오전 1h (09–10) | A | configs.yaml 스키마 설계 | 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 09_Infrastructure_and_Cloud.md |
| Day 1 | 오전 1h (09–10) | B | session_state 키 명세 확정 | 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 07_Backend_Service_Design.md |
| Day 1 | 오전 1h (09–10) | C | 프로젝트 구조·Git 세팅 | 04_System_Architecture.md, 09_Infrastructure_and_Cloud.md, 10_Security_and_Compliance.md, 13_QA_and_Testing_Strategy.md, 14_Deployment_and_Release_Plan.md |
| Day 1 | 오전 2h (10–12) | A | 탭1 구현 | 03_Functional_Requirements.md, 06_API_Specification.md, 10_Security_and_Compliance.md |
| Day 1 | 오전 2h (10–12) | B | 탭3 EfficientAD 파트 구현 | 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 06_API_Specification.md, 08_AI_ML_Integration.md |
| Day 1 | 오전 2h (10–12) | C | 탭3 PatchCore 파트 구현 | 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 06_API_Specification.md, 08_AI_ML_Integration.md |
| Day 1 | 오후 1h (13–14) | A | 탭1 보완 (Grayscale·포맷 필터링) | 03_Functional_Requirements.md, 06_API_Specification.md, 10_Security_and_Compliance.md |
| Day 1 | 오후 1h (13–14) | B | 탭3 EfficientAD 보완 (고급 설정·Threshold) | 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 06_API_Specification.md, 08_AI_ML_Integration.md |
| Day 1 | 오후 1h (13–14) | C | 탭3 PatchCore 보완 (고급 설정·Threshold) | 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 06_API_Specification.md, 08_AI_ML_Integration.md |
| Day 1 | 오후 3h (14–17) | A | 탭2 구현 | 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 06_API_Specification.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md |
| Day 1 | 오후 3h (14–17) | B | 탭4 EfficientAD 학습 루프 | 01_Product_Overview.md, 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 06_API_Specification.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md, 11_Non_Functional_Requirements.md, 12_Observability_and_Operations.md |
| Day 1 | 오후 3h (14–17) | C | 탭4 PatchCore 학습 루프 | 01_Product_Overview.md, 03_Functional_Requirements.md, 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 06_API_Specification.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md, 11_Non_Functional_Requirements.md, 12_Observability_and_Operations.md |
| Day 1 | 버퍼 (17–18) | A | 탭2 엣지케이스·리팩터링 | 03_Functional_Requirements.md |
| Day 1 | 버퍼 (17–18) | B | 탭4 EfficientAD 중지 처리 | 01_Product_Overview.md, 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 06_API_Specification.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md, 12_Observability_and_Operations.md |
| Day 1 | 버퍼 (17–18) | C | 탭4 PatchCore 중지 처리 | 01_Product_Overview.md, 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 06_API_Specification.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md, 12_Observability_and_Operations.md |
| Day 2 | 오전 1h (09–10) | A | 탭3·탭4 session_state 통합 지원 | 08_AI_ML_Integration.md |
| Day 2 | 오전 1h (09–10) | B | 탭4 마무리 (history.json 저장 구조) | 05_Data_Model_and_Storage_Strategy.md, 08_AI_ML_Integration.md |
| Day 2 | 오전 1h (09–10) | C | 탭4 단일 탭 병합 | 07_Backend_Service_Design.md, 08_AI_ML_Integration.md |
| Day 2 | 오전 3h (10–13) | A | 탭6 구현 | 03_Functional_Requirements.md, 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 06_API_Specification.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md |
| Day 2 | 오전 3h (10–13) | B | 탭5 전반 (실험 목록·결과 상세) | 02_User_Personas_and_Use_Cases.md, 03_Functional_Requirements.md, 05_Data_Model_and_Storage_Strategy.md, 11_Non_Functional_Requirements.md, 12_Observability_and_Operations.md |
| Day 2 | 오전 3h (10–13) | C | 탭5 후반 (비교 차트·모델 저장) | 03_Functional_Requirements.md, 05_Data_Model_and_Storage_Strategy.md, 11_Non_Functional_Requirements.md |
| Day 2 | 오후 1h (13–14) | A | 탭6 보완 (Score·FP/FN·다중 내보내기) | 03_Functional_Requirements.md, 07_Backend_Service_Design.md |
| Day 2 | 오후 1h (13–14) | B | 탭5 UI 통합 | 03_Functional_Requirements.md |
| Day 2 | 오후 1h (13–14) | C | Docker 구성 | 09_Infrastructure_and_Cloud.md, 10_Security_and_Compliance.md, 14_Deployment_and_Release_Plan.md |
| Day 2 | 오후 2h (14–16) | A·B·C | 통합 테스트 | 01_Product_Overview.md, 04_System_Architecture.md, 08_AI_ML_Integration.md, 13_QA_and_Testing_Strategy.md |
| Day 2 | 버퍼 (16–18) | A | 버그 수정·한국어 UI | 03_Functional_Requirements.md |
| Day 2 | 버퍼 (16–18) | B | 버그 수정·재현성·로그 확인 | 11_Non_Functional_Requirements.md, 12_Observability_and_Operations.md |
| Day 2 | 버퍼 (16–18) | C | 버그 수정·성능 측정 | 11_Non_Functional_Requirements.md |

---

## 8. 팀원 D — v1.2 추가 기능 구현 계획 (Day 3–5)

> v1.2 수정사항 #01~#04 구현. 기존 코드베이스에 추가 작업.
> 참조 기준: `00_Global_Context_Document.md` 항상 포함.

| 일자 | 시간 블록 | 팀원 | 태스크명 | 참조 PRD (00 항상 포함) |
|------|----------|------|---------|----------------------|
| Day 3 | 오전 1h (09–10) | D | #01 — sidebar.py 타이틀·버튼 레이블 변경 + app.py page_title 변경 | 01_Product_Overview.md, 03_Functional_Requirements.md, 15_UI_UX_Design.md |
| Day 3 | 오전 2h (10–12) | D | #02 — TrainingWorker "stage" 메시지 타입 추가 (EfficientAD 5단계, PatchCore 7단계) | 03_Functional_Requirements.md, 07_Backend_Service_Design.md, 08_AI_ML_Integration.md |
| Day 3 | 오후 2h (13–15) | D | #02 — tab3_training.py 단계 인디케이터 UI 구현 (진행률 바 위 가로 인디케이터) | 03_Functional_Requirements.md, 15_UI_UX_Design.md, 07_Backend_Service_Design.md |
| Day 3 | 오후 2h (15–17) | D | #02 — ETA 계산 로직 + progress 레이블 통합 + 단계별 경과 시간 표시 | 03_Functional_Requirements.md, 11_Non_Functional_Requirements.md |
| Day 3 | 버퍼 1h (17–18) | D | #02 — 통합 테스트 (EfficientAD/PatchCore 각 단계 전환 시 UI 정상 작동 확인) | 13_QA_and_Testing_Strategy.md |
| Day 4 | 오전 1h (09–10) | D | #03 — session_state_init.py에 `experiment_queue`, `_batch_queue_mode` 키 추가 | 00_Global_Context_Document.md, 05_Data_Model_and_Storage_Strategy.md |
| Day 4 | 오전 2h (10–12) | D | #03 — tab2_config.py 하단 2분할 대기열 테이블 + 상세 보기 UI (FR-T2-16~18) | 03_Functional_Requirements.md, 15_UI_UX_Design.md |
| Day 4 | 오후 1h (13–14) | D | #03 — tab3_training.py 상단 대기열 테이블 표시 (FR-T3-14) + 일괄 학습 시작 버튼 | 03_Functional_Requirements.md, 15_UI_UX_Design.md, 07_Backend_Service_Design.md |
| Day 4 | 오후 3h (14–17) | D | #03 — 일괄 학습 순차 실행 로직 (완료→자동 다음, 실패→기록+계속, 건너뛰기→기록+다음) + 제어 버튼 3개 | 03_Functional_Requirements.md, 04_System_Architecture.md, 05_Data_Model_and_Storage_Strategy.md, 07_Backend_Service_Design.md |
| Day 4 | 버퍼 1h (17–18) | D | #03 — 통합 테스트 (대기열 3개 모델 순차 학습, 건너뛰기, 실패 시뮬레이션) | 13_QA_and_Testing_Strategy.md |
| Day 5 | 오전 1h (09–10) | D | #04 — 시뮬레이션 시간 계산 유틸 작성 (고정 시작 2026-06-24 14:00, 3초 간격, 그룹 시간 범위 레이블) | 03_Functional_Requirements.md, 15_UI_UX_Design.md |
| Day 5 | 오전 2h (10–12) | D | #04 — insp_tab2_history.py 하단 3분할 레이아웃 기반 구조 + 단위 선택 버튼 + 시간 범위 테이블 (FR-INSP-T2-05) | 03_Functional_Requirements.md, 15_UI_UX_Design.md, 06_API_Specification.md |
| Day 5 | 오후 2h (13–15) | D | #04 — Anomaly Score 히스토그램 구현 (그룹별 분리, 단위 변경 재계산, 정상/불량 색상) (FR-INSP-T2-06) | 03_Functional_Requirements.md, 15_UI_UX_Design.md, 08_AI_ML_Integration.md |
| Day 5 | 오후 2h (15–17) | D | #04 — Anomaly Score 산점도 구현 (x축 1~N 고정, threshold 점선, 점+선 연결) (FR-INSP-T2-07) | 03_Functional_Requirements.md, 15_UI_UX_Design.md |
| Day 5 | 버퍼 1h (17–18) | D | #04 — 통합 테스트 (자동 검사 20건 진행 후 그룹 확정, 단위 전환 재계산, 그룹 선택 차트 갱신 확인) | 01_Product_Overview.md, 13_QA_and_Testing_Strategy.md |
