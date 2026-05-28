# 14. Deployment and Release Plan

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md), [13_QA_and_Testing_Strategy.md](./13_QA_and_Testing_Strategy.md)
> **버전**: v1.1
> **작성일**: 2026-05-09
> **수정일**: 2026-05-26
> **중요**: MVP v1.0 배포는 단일 환경(로컬 또는 AWS EC2 g4dn.xlarge) 대상이다. CI/CD 파이프라인은 GitHub Actions 기준으로 설계하나, 실행 환경이 없는 경우 수동 배포 절차로 대체한다.

---

## 목차

- [A. Release Scope](#a-release-scope)
- [B. Pre-Deployment Checklist](#b-pre-deployment-checklist)
- [C. Local Deployment Procedure](#c-local-deployment-procedure)
- [D. AWS EC2 Deployment Procedure](#d-aws-ec2-deployment-procedure)
- [E. CI/CD Pipeline (GitHub Actions)](#e-cicd-pipeline-github-actions)
- [F. Rollback Plan](#f-rollback-plan)
- [G. Release Versioning](#g-release-versioning)

---

## A. Release Scope

### A.1 MVP v1.0 포함 항목

| 기능 | 포함 여부 |
|------|---------|
| 탭1: 데이터 폴더 구조 확인 | ✅ |
| 탭2: 전처리 파라미터 설정 + 미리보기 | ✅ |
| 탭3: 모델 파라미터 설정 (EfficientAD / PatchCore) | ✅ |
| 탭4: 학습 실행 + 실시간 로그 | ✅ |
| 탭5: 실험 히스토리 + 결과 상세 + 모델 저장 | ✅ |
| 탭6: 이상 영역 시각화 + Threshold 조정 | ✅ |
| Docker 패키징 | ✅ (선택적) |

### A.2 MVP v1.0 제외 항목 (차기 버전)

| 항목 | 이유 |
|------|------|
| 다중 사용자 지원 | 아키텍처 변경 필요 |
| 클라우드 스토리지 (S3) 연동 | 스코프 외 |
| REST API 외부 제공 | 스코프 외 |
| 실시간 스트리밍 알림 (Slack, Email) | 스코프 외 |

---

## B. Pre-Deployment Checklist

배포 전 아래 항목을 순서대로 확인한다.

### B.1 코드 품질

```
□ pytest tests/ -m "not slow" — 전체 유닛/통합 테스트 PASS
□ pytest --cov=utils --cov-fail-under=80 — 커버리지 ≥ 80%
□ pip-audit -r requirements.txt — HIGH 이상 CVE 없음
□ 코드 내 eval(), exec(), yaml.load(), pickle.load() 없음
□ .gitignore에 dataset/, models/, experiments/, *.pth 포함 확인
```

### B.2 환경 설정

```
□ python scripts/preflight_check.py — 모든 항목 ✅
□ .streamlit/config.toml — enableXsrfProtection = true
□ requirements.txt 버전 고정 확인
□ 필수 디렉터리 존재: experiments/, models/, logs/, results/
```

### B.3 데이터 준비

```
□ 대상 데이터셋이 MVTec AD 형식인지 확인
□ EfficientAD 사용 시: ./dataset/imagenet_penalty/ 내 이미지 존재 확인
□ .env 파일 존재 확인 (MYSQL_ROOT_PASSWORD, MYSQL_DATABASE 설정)
□ DB 컨테이너(smart-qc-db) 기동 및 healthcheck 통과 확인
```

### B.4 최종 기능 검증 (수동)

```
□ 탭1: 데이터셋 경로 입력 → 스캔 성공
□ 탭2: 전처리 설정 저장 → 미리보기 표시
□ 탭3: PatchCore (resnet18, train_steps 제한) 설정 저장
□ 탭4: 학습 시작 → Progress bar 갱신 → 완료 메시지
□ 탭5: 실험 목록 표시 → 선택 → 결과 상세 확인
□ 탭6: 추론 실행 → Anomaly Map 표시 → PNG 저장
□ [학습 중지] 버튼 동작 확인
```

---

## C. Local Deployment Procedure

### C.1 최초 설치 (신규 머신)

```bash
# 1. 저장소 클론
git clone <repo-url> smart-qc-dashboard
cd smart-qc-dashboard

# 2. conda 환경 생성 (09절 §D.1)
conda create -n smart-qc python=3.12 -y
conda activate smart-qc

# 3. PyTorch 설치 (CUDA 12.4)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
    -c pytorch -c nvidia -y

# 4. 나머지 의존성
pip install -r requirements.txt

# 5. 환경 검증
python scripts/preflight_check.py

# 6. ImageNet penalty 데이터 배치 (EfficientAD 사용 시)
# ./dataset/imagenet_penalty/ 에 이미지 파일 배치

# 7. 실행
streamlit run app.py
```

### C.2 업데이트 배포 (기존 설치)

```bash
conda activate smart-qc

# 1. 코드 업데이트
git pull origin main

# 2. 의존성 변경 확인
pip install -r requirements.txt

# 3. 마이그레이션 스크립트 실행 (있는 경우)
# python scripts/migrate_history.py  ← 버전별 적용 여부 확인

# 4. 재시작
streamlit run app.py
```

---

## D. AWS EC2 Deployment Procedure

### D.1 인스턴스 스펙 (00절 §10.5)

| 항목 | 값 |
|------|-----|
| 인스턴스 타입 | g4dn.xlarge |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| vCPU | 4 |
| RAM | 16 GB |
| 스토리지 | 100 GB gp3 EBS |
| OS | Ubuntu 22.04 LTS |
| 포트 | 8501/tcp (Streamlit), 22/tcp (SSH) |

### D.2 EC2 초기 설정

```bash
# Ubuntu 22.04 기준

# 1. NVIDIA 드라이버 설치
sudo apt-get update
sudo apt-get install -y nvidia-driver-550

# 2. CUDA 12.4 설치 (nvidia-cuda-toolkit)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get install -y cuda-12-4

# 3. NVIDIA Container Toolkit (Docker 사용 시)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get install -y nvidia-container-toolkit

# 4. Docker 설치
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu

# 5. conda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc
```

### D.3 애플리케이션 배포

```bash
# 저장소 클론
git clone <repo-url> smart-qc-dashboard
cd smart-qc-dashboard

# 환경 구성 (09절 §D.1과 동일)
conda create -n smart-qc python=3.12 -y
conda activate smart-qc
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt

# 데이터셋 마운트 (EBS 또는 S3 sync)
# aws s3 sync s3://your-bucket/imagenet_penalty ./dataset/imagenet_penalty/

# 백그라운드 실행 (tmux 권장)
tmux new -s smartqc
streamlit run app.py --server.headless true --server.address 0.0.0.0

# 접속: http://<EC2-PUBLIC-IP>:8501
# 보안그룹에서 8501/tcp 개방 필요
```

### D.4 Docker 기반 배포 (EC2)

```bash
# 빌드 및 실행 (09절 §H 참조)
docker compose -f docker-compose.base.yml -f docker-compose.yml build
docker compose -f docker-compose.base.yml -f docker-compose.yml up -d

# 로그 확인
docker compose -f docker-compose.base.yml -f docker-compose.yml logs -f smart-qc

# 중단
docker compose -f docker-compose.base.yml -f docker-compose.yml down
```

### D.5 보안그룹 설정

| 규칙 | 포트 | 소스 |
|------|------|------|
| SSH 접속 | 22/tcp | 작업자 IP만 |
| Streamlit 접속 | 8501/tcp | 필요한 IP 범위만 (전체 공개 금지) |

---

## E. CI/CD Pipeline (GitHub Actions)

### E.1 워크플로우 트리거

```yaml
# .github/workflows/ci.yml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

### E.2 CI 파이프라인 (빠른 테스트)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install pytest pytest-cov pip-audit

      - name: Security audit
        run: pip-audit -r requirements.txt

      - name: Run unit and integration tests (excluding slow)
        run: pytest tests/ -m "not slow" --cov=utils --cov-report=term-missing --cov-fail-under=80 -v

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage
```

### E.3 릴리즈 워크플로우

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - "v*.*.*"

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t smart-qc-dashboard:${{ github.ref_name }} .

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          body: |
            ## Smart QC Dashboard ${{ github.ref_name }}
            
            ### 변경 사항
            CHANGELOG.md 참조
          files: |
            requirements.txt
```

### E.4 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 안정 릴리즈 버전 |
| `develop` | 통합 개발 브랜치 |
| `feature/*` | 기능 개발 |
| `fix/*` | 버그 수정 |
| `docs/*` | 문서 작업 |

**PR 규칙**:
- `feature/*` → `develop`: 리뷰 1명 이상
- `develop` → `main`: 전체 테스트 통과 + 리뷰 1명 이상

---

## F. Rollback Plan

### F.1 로컬 환경 롤백

```bash
# 이전 커밋으로 롤백
git log --oneline -10   # 대상 커밋 확인
git checkout <commit-hash>

# 또는 이전 태그로
git checkout v0.9.0

# conda 환경은 재설치 (requirements.txt 버전 변경된 경우)
pip install -r requirements.txt
```

### F.2 데이터 롤백

학습 결과 파일은 불변이므로 코드 롤백 후 별도 조치 불필요.  
`history.json` 이 손상된 경우:

```python
# scripts/repair_history.py
# models/ 디렉터리를 스캔하여 history.json 재구성

from pathlib import Path
import json

models_dir = Path("./models")
records = []
for exp_dir in models_dir.iterdir():
    if exp_dir.is_dir():
        configs_path = exp_dir / "configs.yaml"
        pth_path = exp_dir / "model_state_dict.pth"
        if configs_path.exists() and pth_path.exists():
            records.append({
                "experiment_id": exp_dir.name,
                "status": "completed",
                "model_path": str(exp_dir),
                # 나머지 필드는 configs.yaml에서 복원
            })

# history.json 재기록 (incomplete — configs.yaml 파싱 후 보완)
print(f"복구 가능한 실험: {len(records)}개")
```

### F.3 EC2 인스턴스 롤백

```bash
# Docker 기반인 경우 — 이전 이미지로 전환
docker compose -f docker-compose.base.yml -f docker-compose.yml down
docker tag smart-qc-dashboard:v1.0 smart-qc-dashboard:current
docker compose -f docker-compose.base.yml -f docker-compose.yml up -d

# EBS 스냅샷 복구 (심각한 경우)
# AWS 콘솔에서 스냅샷 → 새 볼륨 생성 → 인스턴스에 연결
```

---

## G. Release Versioning

### G.1 버전 스키마

`MAJOR.MINOR.PATCH` (Semantic Versioning)

| 버전 구분 | 조건 |
|----------|------|
| MAJOR | 하위 호환 불가 변경 (history.json 스키마 변경, 탭 구조 변경) |
| MINOR | 신기능 추가 (하위 호환 유지) |
| PATCH | 버그 수정 |

### G.2 현재 릴리즈 계획

| 버전 | 내용 |
|------|------|
| **v1.0.0** | MVP — 6탭 전체 기능 (이 PRD 대상) |
| v1.1.0 | (차기) PatchCore 배치 최적화, 추론 속도 개선 |
| v2.0.0 | (장기) 클라우드 스토리지 연동, 다중 사용자 지원 |

### G.3 history.json 스키마 마이그레이션

버전 업그레이드로 `history.json` 스키마가 변경되는 경우:

```python
# scripts/migrate_history.py — 버전별로 작성
# 예: v1.0 → v1.1 마이그레이션

from utils.storage import load_history, save_history

def migrate_v1_0_to_v1_1():
    records = load_history()
    for r in records:
        # 새 필드 추가 (기본값)
        if "new_field" not in r:
            r["new_field"] = None
    save_history(records)
    print(f"마이그레이션 완료: {len(records)}개 레코드")

if __name__ == "__main__":
    migrate_v1_0_to_v1_1()
```

---

---

## H. 비전검사 대시보드 배포 체크리스트 (v1.1)

### H.1 신규 파일 생성 확인

배포 전 아래 파일이 소스 저장소에 존재하는지 확인한다.

```
□ inspection/__init__.py (또는 inspection/ 디렉터리 존재)
□ inspection/inspection_app.py
□ inspection/tabs/__init__.py
□ inspection/tabs/insp_tab1_realtime.py
□ inspection/tabs/insp_tab2_history.py
□ inspection/tabs/insp_tab3_model.py
□ inspection/utils/__init__.py
□ inspection/utils/insp_session_init.py
□ inspection/utils/test_sampler.py
```

### H.2 insp_ 세션 상태 초기화 검증

```
□ app.py 사이드바에서 🏭 비전검사 버튼 클릭 시 insp_session_init.init_insp_session_state() 호출 확인
□ 앱 첫 진입 시 insp_seq == 0, insp_records == [] 확인
□ 페이지 새로고침(앱 재시작) 시 모든 insp_ 키 초기화 확인
□ 모델 교체 시 insp_records, insp_seq 초기화 확인 (TC-INSP-03)
```

### H.3 _load_insp_model() 캐시 검증

```
□ 동일 experiment_id로 두 번 호출 시 모델 재로드 없음 확인 (st.cache_resource 또는 session_state 캐시)
□ 모델 교체 후 이전 캐시 무효화 확인
□ 추론 시간 ≤ 3초 확인 (NFR: 추론 ≤ 3초)
□ 자동 검사 타이밍 오차 ±0.5초 이내 확인
□ 불량 팝업 표시 지연 ≤ 0.5초 확인
```

### H.4 WBS — 비전검사 기능 요구사항

배포 전 아래 기능 요구사항 구현이 완료되어야 한다.

| 요구사항 ID | 설명 | 구현 파일 |
|------------|------|---------|
| FR-INSP-CMN-01 | 사이드바 대시보드 전환 버튼 (🔬/🏭) | `app.py` |
| FR-INSP-CMN-02 | `insp_` 네임스페이스 session_state 초기화 | `insp_session_init.py` |
| FR-INSP-CMN-03 | `inspection_record` 스키마 준수 (seq/inspected_at/image_name/image_path/verdict/anomaly_score) | `insp_tab1_realtime.py` |
| FR-INSP-T1-01 | 수동 검사 버튼 | `insp_tab1_realtime.py` |
| FR-INSP-T1-02 | 자동 검사 버튼 (3초 간격) | `insp_tab1_realtime.py` |
| FR-INSP-T1-03 | 3열 레이아웃: 판정결과 / 원본이미지 / Anomaly Map | `insp_tab1_realtime.py` |
| FR-INSP-T1-04 | 불량 감지 팝업 표시 및 자동 검사 중지 | `insp_tab1_realtime.py` |
| FR-INSP-T1-05 | 테스트 풀 소진 시 재섞기 | `test_sampler.py` |
| FR-INSP-T1-06 | 모델 미적용 상태 가드 (`INSP_MSG["NO_MODEL"]`) | `insp_tab1_realtime.py` |
| FR-INSP-T2-01 | 5열 이력 테이블 (번호/시각/이미지명/판정결과/Anomaly Score) | `insp_tab2_history.py` |
| FR-INSP-T2-02 | KPI 카드 4개 (총검사/양품/불량/불량률) | `insp_tab2_history.py` |
| FR-INSP-T2-03 | 이력 없음 상태 가드 | `insp_tab2_history.py` |
| FR-INSP-T2-04 | 히스토그램/차트 없음 (명세 준수) | `insp_tab2_history.py` |
| FR-INSP-T3-01 | 완료된 실험 목록 F1 기준 정렬 | `insp_tab3_model.py` |
| FR-INSP-T3-02 | 모델 [적용] 버튼 → `insp_model` 설정 | `insp_tab3_model.py` |
| FR-INSP-T3-03 | 모델 교체 시 모든 `insp_` 이력 초기화 | `insp_tab3_model.py` |

### H.5 배포 전 수동 검증 체크리스트 추가 항목

```
□ 사이드바: 🔬 모델 탐색 / 🏭 비전검사 버튼 표시 확인
□ 🏭 비전검사 클릭 시 비전검사 대시보드로 전환 확인
□ 탭1: 수동 검사 버튼 클릭 → 3열 결과 표시 확인
□ 탭1: 자동 검사 시작 → 3초 간격 동작 확인
□ 탭1: 불량 이미지 검사 → 팝업 표시 및 자동 중지 확인
□ 탭2: 검사 후 이력 테이블 및 KPI 카드 표시 확인
□ 탭3: 완료 실험 목록 F1 기준 정렬 확인
□ 탭3: 모델 교체 후 탭2 이력이 초기화됨을 확인
□ 앱 재시작 후 비전검사 이력 없음(세션 전용 저장소) 확인
```

---

*[14_Deployment_and_Release_Plan.md 끝 — PRD Phase 2 전체 완료]*
