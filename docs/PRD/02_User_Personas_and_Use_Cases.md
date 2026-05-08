# 02. User Personas and Use Cases

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [01_Product_Overview.md](./01_Product_Overview.md)
> **버전**: v1.0
> **작성일**: 2026-05-08
> **후속 문서**: [03_Functional_Requirements.md](./03_Functional_Requirements.md)

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

| 페르소나 ID | 이름 | 역할 | 우선순위 |
|------------|------|------|---------|
| P-01 | ML 엔지니어 / 데이터 분석가 | 실험 설계·실행·분석 담당 | **Primary** |
| P-02 | MLOps 엔지니어 | 배포·운영 환경 구성 담당 | Secondary |

> P-01이 이 제품의 모든 기능 결정의 주 기준이다. P-02는 배포·인프라 관련 결정에만 참조한다.

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
| PP-01 | 모델·전처리 조합 변경 시마다 Python 스크립트를 수정하고 재실행해야 함 | 탭2·탭3 GUI로 코드 작성 없이 파라미터 변경 |
| PP-02 | 실험 결과를 수동으로 Excel에 기록하다가 어떤 조합이 최선이었는지 잊어버림 | history.json + 탭5 비교 차트로 자동 기록·시각화 |
| PP-03 | 학습 시작 후 완료까지 터미널만 보며 기다려야 함, 중간 진행 상황 파악 불가 | 탭4 Progress Bar + 실시간 Loss 곡선 |
| PP-04 | 결함 위치가 어디인지 모델이 어디를 보는지 확인할 방법이 없음 | 탭6 Anomaly Map 3분할 시각화 |
| PP-05 | 전처리 필터를 적용하면 이미지가 어떻게 바뀌는지 학습 전에 확인할 수 없음 | 탭2 적용 전·후 실시간 미리보기 |
| PP-06 | 학습한 모델을 추론 파이프라인에 연결하려면 별도 코드 작성 필요 | state_dict + configs.yaml 고정 포맷 저장 |

#### 기술 수준별 사용 행동 예측

| 기술 수준 | 예상 행동 패턴 |
|-----------|--------------|
| **입문 (PyTorch 기초)** | 기본값 그대로 실험 1회 → 결과 확인 → 파라미터 1~2개만 변경하여 재실험 |
| **중급 (Anomalib 경험)** | 전처리·모델 파라미터 적극 조정, 고급 설정 expander 활용, 다중 실험 비교 차트 적극 사용 |
| **고급 (이상 탐지 전문)** | ae/st weight, coreset_sampling_ratio 등 모델 핵심 파라미터 조정, AUC/F2 기준으로 실험 필터링 |

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
| **사용 환경** | AWS EC2 SSH 접속, Docker CLI |
| **주요 목표** | 대시보드를 EC2에 배포·운영하고, 저장된 모델을 추론 파이프라인에 연결 |
| **보조 목표** | GPU 컨테이너 안정성 확인, 볼륨 마운트 정상 동작 검증 |

#### 페인 포인트

| # | 페인 포인트 | 이 제품의 해결 방식 |
|---|------------|-------------------|
| PP-07 | 로컬 환경과 서버 환경의 패키지 버전 불일치로 재현성 실패 | Docker 이미지 단일화 (`nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04`) |
| PP-08 | 저장된 모델 파일에 학습 조건 정보가 없어서 추론 앱 연결 시 설정 재확인 필요 | `configs.yaml`이 모델과 동일 경로에 함께 저장됨 |

---

### B.3 Use Cases

#### 전체 Use Case 목록

| UC ID | 제목 | 주 페르소나 | 관련 탭 | 우선순위 |
|-------|------|------------|---------|---------|
| UC-01 | 새 데이터셋 등록 및 검증 | P-01 | 탭1 | Must |
| UC-02 | 전처리 방식 탐색 및 설정 | P-01 | 탭2 | Must |
| UC-03 | EfficientAD 실험 설정 및 실행 | P-01 | 탭3, 탭4 | Must |
| UC-04 | PatchCore 실험 설정 및 실행 | P-01 | 탭3, 탭4 | Must |
| UC-05 | 실험 결과 분석 | P-01 | 탭5 | Must |
| UC-06 | 두 실험 비교 및 최적 모델 선택 | P-01 | 탭5 | Must |
| UC-07 | Anomaly Map으로 모델 행동 검증 | P-01 | 탭6 | Must |
| UC-08 | 최적 모델 저장 | P-01 | 탭5 | Must |
| UC-09 | 설정 재사용 (configs.yaml 불러오기) | P-01 | 탭2, 탭3 | Should |
| UC-10 | 실패 실험 정리 (히스토리 삭제) | P-01 | 탭5 | Must |
| UC-11 | EC2 환경 배포 | P-02 | - | Must |
| UC-12 | 결함 유형별 탐지 성능 필터링 확인 | P-01 | 탭6 | Should |

---

#### UC-01: 새 데이터셋 등록 및 검증

```
액터:      P-01
전제조건:  MVTec AD 스타일 데이터셋이 로컬 경로에 존재한다
성공 종료: session_state.dataset_path 에 검증된 경로 저장, dataset_meta 구성 완료
실패 종료: dataset_path = None 유지, 오류 메시지 표시

주 플로우:
  1. 사용자가 탭1의 텍스트 입력 필드에 데이터셋 루트 경로를 입력한다
  2. 시스템이 입력 즉시(on_change) 아래 검증을 수행한다
     2a. 경로 존재 확인 → 실패 시 ERR_DATASET_NOT_FOUND
     2b. train/good/ 하위 이미지 ≥ 1개 확인 → 실패 시 ERR_INVALID_FOLDER_STRUCTURE
     2c. test/ 하위 디렉토리 ≥ 1개 확인 → 실패 시 ERR_INVALID_FOLDER_STRUCTURE
  3. 시스템이 dataset_meta를 구성한다 (1.5절 스키마)
     3a. 폴더별 이미지 수 카운트 (지원 포맷만)
     3b. 채널 수 감지 (첫 번째 이미지 기준)
     3c. 결함 클래스 목록 추출 (test/ 하위 디렉토리명)
  4. 시스템이 폴더 트리를 텍스트로 렌더링한다
  5. 시스템이 클래스별 이미지 수 테이블을 렌더링한다
  6. 시스템이 각 클래스 대표 이미지 썸네일을 렌더링한다
  7. Grayscale 감지 시 st.info(MSG["GRAYSCALE_DETECT"]) 표시

대안 플로우:
  2a-1. 지원 포맷 외 파일 존재 → st.warning() 표시 후 주 플로우 계속 진행
  2b-1. ground_truth/ 없음 → 가정 A-10 적용, 빈 마스크로 처리하며 계속 진행

사전 오류:
  경로 입력 후 2초 이내 반응 없으면 → 로딩 스피너 표시 (st.spinner)
```

---

#### UC-02: 전처리 방식 탐색 및 설정

```
액터:      P-01
전제조건:  UC-01 완료, session_state.dataset_path != None
성공 종료: session_state.preprocessing_config 저장 완료
실패 종료: preprocessing_config = None 유지

주 플로우:
  1. 사용자가 탭2 진입 (guard 통과)
  2. 사용자가 전처리 방식 라디오 선택 (None / Homomorphic / HE / CLAHE)
  3. 선택된 방식의 파라미터 UI만 렌더링 (비선택 UI 미렌더링, R-UI-02 규칙)
  4. 사용자가 파라미터 슬라이더 조정 → 즉시 미리보기 갱신
     4a. train/good/ 에서 첫 번째 이미지를 샘플로 로드
     4b. 전처리 → Resize+Padding → 정규화 적용
     4c. 좌(원본) / 우(처리 후) 2컬럼 미리보기 렌더링
  5. 사용자가 image_size 입력 (32의 배수, 32~1024)
  6. 사용자가 정규화 방식 선택 (ImageNet / 커스텀)
     6a. 커스텀 선택 시 mean/std 입력 필드 렌더링
  7. 사용자가 [전처리 설정 저장] 클릭
  8. session_state.preprocessing_config Write
  9. (선택) [configs.yaml 저장] 클릭 → preprocessing 섹션만 업데이트

대안 플로우:
  7-1. [configs.yaml 불러오기] 클릭
       → load_config() 호출
       → preprocessing 섹션 로드 후 UI 값 자동 반영
       → UC-09 플로우와 동일
```

---

#### UC-03: EfficientAD 실험 설정 및 실행

```
액터:      P-01
전제조건:  UC-02 완료, session_state.preprocessing_config != None
성공 종료: experiments[exp_id] 에 status="completed" 레코드 저장, history.json 갱신
실패 종료: experiments[exp_id] 에 status="중단" 또는 미기록, 오류 메시지 표시

주 플로우 — 탭3 (설정):
  1. 사용자가 탭3 진입 (guard 통과)
  2. 시스템이 torch.cuda.is_available() 실행 → device_info Write
  3. 사용자가 모델 라디오에서 "EfficientAD" 선택
  4. image_size 필드에 preprocessing_config.image_size 자동 반영 (편집 가능)
  5. 사용자가 기본 노출 파라미터 설정
     (model_size, train_steps, optimizer, learning_rate, weight_decay,
      out_channels, padding, ae_loss_weight ↔ st_loss_weight)
  6. ae_loss_weight 변경 시 st_loss_weight = 1.0 - ae_loss_weight 자동 보정 (R-03)
  7. (선택) [고급 설정] expander 열기 → 고급 파라미터 설정
  8. Threshold 방식·값 설정
  9. [모델 설정 저장] 클릭 → session_state.model_config Write
  10. (선택) [configs.yaml 저장] 클릭

주 플로우 — 탭4 (실행):
  11. 사용자가 탭4 진입 (guard 통과)
  12. 실험명 입력 (빈칸이면 자동 생성: efficientad_{YYYYMMDD}_{HHMMSS}_{4자리})
  13. 사용자가 [학습 시작] 클릭
  14. current_run_status = "running"
  15. 백그라운드 스레드 시작 (threading.Thread + queue.Queue)
  16. 메인 스레드: st.empty() 컨테이너에 Progress Bar, Loss 곡선, 로그 텍스트 갱신
      (500 step마다 queue에서 데이터 수신 후 st.rerun())
  17. 학습 완료:
      17a. metrics 계산 (accuracy, precision, recall, f1, f2, auc)
      17b. experiments[exp_id] Write (status="completed")
      17c. history.json append (R-ATOMIC-01 원자적 쓰기)
      17d. model_state_dict.pth 저장
      17e. configs.yaml 모델 경로에 복사
      17f. current_run_status = "completed" → "idle"
      17g. st.success("학습이 완료되었습니다. 소요 시간: {N}분 {M}초")

대안 플로우:
  13-1. [학습 중지] 클릭 (학습 중 표시)
        → 중단 신호 → 백그라운드 스레드 종료
        → experiments[exp_id] Write (status="중단", metrics=null)
        → history.json append
        → current_run_status = "idle"
        → st.warning(MSG["TRAIN_STOPPED"])

실패 플로우:
  GPU OOM → torch.cuda.OutOfMemoryError catch
           → st.error() + batch_size 축소 권장 메시지
           → current_run_status = "idle"
```

---

#### UC-04: PatchCore 실험 설정 및 실행

```
액터:      P-01
전제조건:  UC-02 완료
성공 종료: experiments[exp_id] 에 status="completed" 레코드 저장

주 플로우 — 탭3 (설정):
  1~2.  UC-03 1~2와 동일
  3.    사용자가 모델 라디오에서 "PatchCore" 선택
  4.    image_size 자동 반영 (UC-03 4와 동일)
  5.    사용자가 기본 노출 파라미터 설정
        (backbone, pretrained_source, coreset_sampling_ratio, neighbourhood_kernel_size)
  5a.   pretrained_source == "local" 선택 시 pretrained_path 입력 필드 렌더링
  6.    (선택) [고급 설정] expander → max_train, knn, top_k_ratio
  7~10. UC-03 8~10과 동일

주 플로우 — 탭4 (실행):
  11~12. UC-03 11~12와 동일 (실험명 자동 생성: patchcore_{YYYYMMDD}_{HHMMSS}_{4자리})
  13.    사용자가 [학습 시작] 클릭
  14~17. UC-03 14~17과 동일
         단, Loss 곡선은 에포크 단위 갱신 (PatchCore는 1 pass 학습)

비고:
  PatchCore는 train_steps 파라미터를 사용하지 않는다.
  탭3에서 PatchCore 선택 시 train_steps UI를 미렌더링한다 (R-UI-02).
```

---

#### UC-05: 실험 결과 분석

```
액터:      P-01
전제조건:  UC-03 또는 UC-04 완료, status="completed" 실험 ≥ 1개
성공 종료: 선택한 실험의 상세 결과 시각화 완료, selected_experiment_id Write

주 플로우:
  1. 사용자가 탭5 진입 (guard 통과)
  2. 시스템이 history.json 로드 → 실험 목록 테이블 렌더링
     컬럼: 실험명 / 모델 / 주요 파라미터 요약 / Accuracy / Precision / Recall / F1 / F2 / AUC / 실행 시각 / 상태
     상태="중단"인 행: 회색 텍스트 처리
  3. 사용자가 테이블에서 실험 1개 선택
  4. session_state.selected_experiment_id Write
  5. 상세 결과 렌더링:
     5a. 성능 지표 카드 (st.metric): Accuracy, Precision, Recall, F1
     5b. Confusion Matrix heatmap (Plotly)
     5c. ROC Curve + AUC 값 (Plotly)
     5d. Anomaly Score 분포 히스토그램 (정상 vs 결함 중첩, Plotly)

대안 플로우:
  3-1. 상태="중단" 실험 선택 → 지표 카드·차트 미렌더링, "중단된 실험입니다" 안내
```

---

#### UC-06: 두 실험 비교 및 최적 모델 선택

```
액터:      P-01
전제조건:  status="completed" 실험 ≥ 2개
성공 종료: 비교 차트 렌더링 완료

주 플로우:
  1. UC-05 플로우 진행 (탭5 진입)
  2. 사용자가 테이블에서 체크박스로 비교 대상 다중 선택 (최대 10개, 가정 A-13)
  3. 비교 메트릭 다중 선택 (Accuracy / Precision / Recall / F1 / F2)
  4. 차트 유형 선택 (막대 차트 / 레이더 차트)
  5. 시스템이 비교 차트 렌더링 (Plotly)
  6. 사용자가 최적 모델 판단 후 UC-08 진행

대안 플로우:
  2-1. 10개 초과 선택 시 st.warning("최대 10개까지 비교 가능합니다.")
  4-1. 선택 실험이 2개 미만이면 비교 차트 미렌더링, "2개 이상 선택 필요" 안내
```

---

#### UC-07: Anomaly Map으로 모델 행동 검증

```
액터:      P-01
전제조건:  UC-05 완료, selected_experiment_id != None, status="completed"
성공 종료: 3분할 시각화 렌더링, PNG 저장 완료

주 플로우:
  1. 사용자가 탭6 진입 (guard 통과)
  2. 시스템이 selected_experiment의 테스트 이미지 목록 테이블 렌더링
     컬럼: 이미지명 / Anomaly Score / OK·NG 판정 / GT 일치 여부 / 오분류(FP/FN)
  3. (선택) 결함 유형 드롭다운으로 필터링
  4. 사용자가 이미지 1개 선택
  5. 시스템이 해당 이미지에 대한 Anomaly Map 로드
     (학습 완료 시 metrics.anomaly_scores에 이미 계산됨, 가정 A-05)
  6. Threshold 슬라이더 표시 (초기값: experiment.threshold_value)
  7. 3분할 시각화 렌더링:
     7a. 좌: 원본 이미지
     7b. 중: GT 마스크 (없으면 빈 마스크)
     7c. 우: Anomaly Map (jet colormap heatmap)
  8. Threshold 슬라이더 조정 → 이진화 결과 실시간 갱신
     → anomaly_map_threshold Write
  9. 사용자가 [PNG 저장] 클릭 → 3분할 이미지 PNG 다운로드

대안 플로우:
  5-1. 모델 재로드 필요 시 (캐시 미스):
       → st.spinner() 표시 후 model_state_dict.pth 로드
       → 전체 테스트셋 재추론 (이전 scores 덮어쓰기 금지, 읽기 전용 사용)
```

---

#### UC-08: 최적 모델 저장

```
액터:      P-01
전제조건:  UC-05 완료, 저장할 실험 선택됨
성공 종료: model_state_dict.pth + configs.yaml 저장, 경로·용량 출력

주 플로우:
  1. 탭5에서 저장 대상 실험 선택 (UC-05 3단계)
  2. 저장 경로 입력 필드 (기본값: ./models/{exp_id}/)
  3. [모델 저장] 클릭
  4. 시스템이 디스크 여유 공간 확인 → < 500MB 시 st.warning()
  5. 시스템이 state_dict 저장: torch.save(model.state_dict(), path/model_state_dict.pth)
  6. 시스템이 configs.yaml 복사: experiment.configs_path → 저장 경로
  7. st.success(f"저장 완료: {path} ({size_mb:.1f} MB)")
  8. experiment.model_path, configs_path 갱신 (이미 저장된 경우 덮어쓰기)

실패 플로우:
  4-1. 디스크 부족으로 저장 실패 → st.error(ERR_MODEL_SAVE_FAILED)
```

---

#### UC-09: 설정 재사용 (configs.yaml 불러오기)

```
액터:      P-01
전제조건:  이전 실험의 configs.yaml이 존재한다
성공 종료: UI 필드에 설정값 자동 반영

주 플로우:
  1. 탭2 또는 탭3에서 [configs.yaml 불러오기] 클릭
  2. 파일 경로 입력 (st.text_input)
  3. load_config(path) 호출
  4. 탭2: preprocessing 섹션 → UI 값 자동 반영
     탭3: model 섹션 → UI 값 자동 반영
  5. 반영된 값으로 미리보기 즉시 갱신 (탭2 경우)

실패 플로우:
  3-1. 파일 없음 → st.error("파일을 찾을 수 없습니다: {path}")
  3-2. YAML 파싱 실패 → st.error(ERR_CONFIG_LOAD_FAILED) + 현재 UI 상태 유지
```

---

#### UC-10: 실패 실험 정리 (히스토리 삭제)

```
액터:      P-01
전제조건:  삭제 대상 실험이 history.json에 존재한다
성공 종료: history.json에서 해당 레코드 제거, 테이블 즉시 갱신

주 플로우:
  1. 탭5 실험 목록 테이블에서 삭제 대상 선택
  2. [실험 삭제] 버튼 클릭
  3. 확인 다이얼로그: st.warning() + [확인] / [취소] 버튼
  4. [확인] 클릭:
     4a. history.json에서 해당 experiment_id 레코드 제거 (R-ATOMIC-01)
     4b. session_state.experiments 에서 해당 키 제거
     4c. selected_experiment_id == 삭제된 ID이면 → None으로 초기화
     4d. ./models/{exp_id}/ 디렉토리 삭제 (model_path != null인 경우만)
     4e. ./logs/{exp_id}.log 삭제
     4f. 테이블 즉시 갱신 → event: experiment_deleted 로그 기록

비고: [취소] 클릭 시 아무 변경 없음
```

---

#### UC-11: EC2 환경 배포

```
액터:      P-02
전제조건:  Docker, NVIDIA Container Toolkit 설치된 Ubuntu 22.04 EC2 인스턴스
성공 종료: http://{ec2-ip}:8501 접속 및 탭1 정상 렌더링

주 플로우:
  1. EC2 SSH 접속
  2. git clone {repository_url}
  3. docker build -t vision-inspection-dashboard:latest .
  4. docker run -d \
       --name vision-dashboard \
       --gpus all \
       -p 8501:8501 \
       -v /home/ubuntu/dataset:/app/dataset \
       -v /home/ubuntu/models:/app/models \
       -v /home/ubuntu/experiments:/app/experiments \
       --restart unless-stopped \
       vision-inspection-dashboard:latest
  5. 브라우저에서 http://{ec2-ip}:8501 접속

실패 플로우:
  3-1. 빌드 실패 → requirements.txt 패키지 버전 충돌 확인
  4-1. GPU 인식 실패 → nvidia-smi 명령으로 드라이버 확인
```

---

#### UC-12: 결함 유형별 탐지 성능 필터링 확인

```
액터:      P-01
전제조건:  UC-07 진입 조건과 동일
성공 종료: 선택한 결함 유형의 이미지만 목록에 표시

주 플로우:
  1. 탭6 진입
  2. 결함 유형 드롭다운에서 특정 클래스 선택 (예: "crack", "scratch")
     드롭다운 옵션: ["전체"] + dataset_meta.defect_classes
  3. 선택 클래스의 이미지만 테이블 필터링
  4. UC-07 4~9 플로우와 동일

비고: "전체" 선택 시 필터 없이 전체 테스트 이미지 표시
```

---

### B.4 사용자 여정 맵 (P-01, 첫 실험 완주)

```
[인식 단계]
데이터셋 준비 → 탭1 경로 입력 → 구조 검증 통과 확인
    ↓ (소요: ~2분)

[설정 단계]
탭2: 전처리 방식 선택, 미리보기 확인 → 탭3: 모델 선택, 파라미터 조정
    ↓ (소요: ~5분)

[실행 단계]
탭4: 학습 시작 → Progress Bar, Loss 곡선 모니터링 → 완료 알림 수신
    ↓ (소요: EfficientAD ~20분 / PatchCore ~10분)

[분석 단계]
탭5: AUC 확인, Confusion Matrix 검토 → 필요 시 파라미터 조정 후 재실험
    ↓ (소요: ~3분)

[검증 단계]
탭6: Anomaly Map으로 결함 위치 시각화 → 모델 행동 이해
    ↓ (소요: ~3분)

[저장 단계]
탭5: 최적 모델 저장 → 추론 팀에 전달
    ↓ (소요: ~1분)

총 소요: 단일 실험 사이클 약 30분 이내 (성공 지표 기준)
```

---

### B.5 Edge Cases (페르소나 관점)

| # | 상황 | 영향 페르소나 | 처리 방식 |
|---|------|-------------|-----------|
| ECU-01 | 탭5에서 실험 삭제 후 탭6 진입 | P-01 | selected_experiment_id = None으로 자동 초기화, 탭6 guard 발동 |
| ECU-02 | 탭3에서 EfficientAD → PatchCore로 모델 변경 | P-01 | model_config 완전 교체, 이전 EfficientAD 설정 유지 안됨 (사용자 안내 필요) |
| ECU-03 | 학습 중 브라우저 새로고침 | P-01 | session_state 초기화 → 백그라운드 스레드는 계속 실행 → 미아 스레드 발생. 탭4에 "학습 중 새로고침 시 학습 상태를 확인할 수 없습니다" 안내 텍스트 표시 |
| ECU-04 | 탭2에서 image_size 변경 후 탭3 재진입 | P-01 | model_config.image_size 불일치 발생. 탭3 진입 시 preprocessing_config.image_size와 비교하여 불일치 감지 → st.warning() + 자동 동기화 제안 |
| ECU-05 | 동일 설정으로 실험 2회 실행 | P-01 | experiment_id에 4자리 난수 포함이므로 중복 방지됨. 두 실험이 별개 레코드로 저장됨 |

---

## C. System & Data Design

```
N/A — 이 문서는 사용자·시나리오 정의가 목적이다.
      데이터 모델과 시스템 설계는 00_Global_Context_Document.md 1~5절에 정의되어 있으며,
      05_Data_Model_and_Storage_Strategy.md에서 상세 확장된다.
```

---

## D. API Contracts

```
N/A — 이 시스템은 REST API 서버를 포함하지 않는다.
      내부 인터페이스 계약은 00_Global_Context_Document.md 3절 참조.
```

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
| **인터페이스 언어** | 전체 UI 한국어, 기술 용어 한국어+영문 병기 | P-01 (한국어 사용자) |
| **학습 시작 전 확인 없음** | [학습 시작] 클릭 즉시 실행. 추가 확인 팝업 없음 | P-01 (실험 반복 빈도 높음) |
| **[실험 삭제] 클릭 시 확인 팝업 필수** | 데이터 삭제는 되돌릴 수 없으므로 1회 확인 | P-01 (실수 방지) |
| **파라미터 기본값 사전 설정** | 모든 파라미터는 합리적인 기본값으로 초기화됨 (1.4절) | P-01 입문 수준 고려 |
| **고급 설정은 expander로 숨김** | 입문 사용자에게 기본값으로 시작할 수 있는 경로 제공 | P-01 입문~중급 모두 |

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
| PUA-01 | P-01이 코드 작성 없이 탭1~탭6 Golden Path 완주 가능 | 실사용자 수동 테스트 |
| PUA-02 | P-01이 EfficientAD와 PatchCore 두 실험을 비교하여 더 높은 AUC 모델 식별 가능 | UC-06 시나리오 실행 |
| PUA-03 | P-02가 README 없이 docker run 명령만으로 EC2 배포 완료 가능 | UC-11 시나리오 실행 |
| PUA-04 | 탭6 Anomaly Map에서 사용자가 결함 위치를 육안으로 확인 가능 | 시각 검수 |
| PUA-05 | [학습 중지] 후 재실험 시 이전 중단 상태가 현재 실험에 영향 없음 | UC-03 대안 플로우 후 재실행 |

### H.2 Given-When-Then 시나리오

#### TC-UC-02: 전처리 미리보기 즉시 반응

```
Given:  탭2가 렌더링된 상태이고 Homomorphic이 선택되어 있다
When:   사용자가 sigma 슬라이더를 10.0에서 30.0으로 변경한다
Then:   우측 미리보기 이미지가 1초 이내에 갱신된다
        원본 이미지(좌)는 변경되지 않는다
        session_state.preprocessing_config 는 아직 갱신되지 않는다
        (설정 저장은 [전처리 설정 저장] 버튼 클릭 시에만 발생)
```

#### TC-UC-03: ae_loss_weight 자동 보정

```
Given:  탭3에서 EfficientAD가 선택된 상태이다
        ae_loss_weight = 0.5, st_loss_weight = 0.5
When:   사용자가 ae_loss_weight 슬라이더를 0.7로 변경한다
Then:   st_loss_weight 슬라이더가 자동으로 0.3으로 갱신된다
        ae_loss_weight + st_loss_weight == 1.0 (부동소수점 오차 허용: abs(sum-1.0) < 1e-6)
```

#### TC-UC-10: 실험 삭제 안전성

```
Given:  selected_experiment_id = "efficientad_20260508_140023_7f3a"
        해당 실험이 history.json에 존재한다
        탭6이 해당 실험을 참조 중이다
When:   사용자가 탭5에서 해당 실험을 삭제하고 [확인]을 클릭한다
Then:   history.json에 해당 레코드가 존재하지 않는다
        session_state.selected_experiment_id == None
        탭6 진입 시 MSG["NO_SELECTED_EXP"] 표시
        ./models/efficientad_20260508_140023_7f3a/ 디렉토리가 존재하지 않는다
```

---

## I. Implementation Plan

```
N/A — 전체 구현 계획은 14_Deployment_and_Release_Plan.md에서 다룬다.
      이 문서의 Use Case 시나리오는 13_QA_and_Testing_Strategy.md의
      테스트 케이스 설계 시 Given-When-Then 입력값으로 직접 재사용된다.
```

---

*다음 문서*: [03_Functional_Requirements.md](./03_Functional_Requirements.md)
