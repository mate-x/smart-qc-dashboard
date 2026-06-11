# 14. Deployment and Release Plan

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md), [13_QA_and_Testing_Strategy.md](./13_QA_and_Testing_Strategy.md)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **수정일**: 2026-06-11
> **중요**: MVP v1.0 배포는 단일 환경(로컬 또는 AWS EC2 g4dn.xlarge) 대상이다. CI/CD 파이프라인은 GitHub Actions 기준으로 설계하나, 실행 환경이 없는 경우 수동 배포 절차로 대체한다.

---

## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 최초 작성 |
| v1.1 | 2026-05-26 | 비전검사 대시보드 배포 체크리스트 추가 (H절) |
| v2.0 | 2026-06-11 | 3-레포 구조 반영: 실행 명령을 `uvicorn`(백엔드) + `npm run dev`(Explorer/Vision)로 교체; `streamlit run` 제거; B.2 `.streamlit/config.toml` → Node.js 체크; D.1 포트 8501→8000; E.2 CI 커버리지 `--cov=api` 추가; H절 Streamlit session_state / `insp_tab*.py` → FastAPI endpoint + Vision React 기준으로 전면 교체 |

---

## 목차

- [A. Release Scope](#a-release-scope)
- [B. Pre-Deployment Checklist](#b-pre-deployment-checklist)
- [C. Local Deployment Procedure](#c-local-deployment-procedure)
- [D. AWS EC2 Deployment Procedure](#d-aws-ec2-deployment-procedure)
- [E. CI/CD Pipeline (GitHub Actions)](#e-cicd-pipeline-github-actions)
- [F. Rollback Plan](#f-rollback-plan)
- [G. Release Versioning](#g-release-versioning)
- [H. 비전검사 대시보드 배포 체크리스트](#h-비전검사-대시보드-배포-체크리스트-v20)

---

## A. Release Scope

### A.1 MVP v1.0 포함 항목

| 기능 | 레포 | 포함 여부 |
|------|------|---------|
| Explorer Dataset 화면 (`/`): 데이터 폴더 구조 확인 | smart-qc-explorer | ✅ |
| Explorer Config 화면 (`/config`): 전처리 및 모델 설정 + 미리보기 | smart-qc-explorer | ✅ |
| Explorer Training 화면 (`/training`): 학습 실행 + WS 실시간 로그 | smart-qc-explorer | ✅ |
| Explorer Experiments 화면 (`/experiments`): 실험 히스토리 + 결과 상세 + 모델 저장 | smart-qc-explorer | ✅ |
| Explorer AnomalyMap 화면 (`/anomaly-map`): 이상 영역 시각화 + Threshold 조정 | smart-qc-explorer | ✅ |
| Vision Realtime 화면 (`/`): 수동/자동 비전 검사 + WS 실시간 결과 | smart-qc-vision | ✅ |
| Vision History 화면 (`/history`): KPI 카드, 이력 테이블, 통계 차트 | smart-qc-vision | ✅ |
| Vision Models 화면 (`/models`): 모델 교체 | smart-qc-vision | ✅ |
| FastAPI 백엔드 (`smart-qc-dashboard`) | smart-qc-dashboard | ✅ |
| Docker 패키징 | — | ✅ (선택적) |

### A.2 MVP v1.0 제외 항목 (차기 버전)

| 항목 | 이유 |
|------|------|
| 다중 사용자 지원 | 아키텍처 변경 필요 |
| 클라우드 스토리지 (S3) 연동 | 스코프 외 |
| 외부 시스템용 REST API 공개 | 스코프 외 (내부 FastAPI는 포함) |
| 실시간 스트리밍 알림 (Slack, Email) | 스코프 외 |

---

## B. Pre-Deployment Checklist

배포 전 아래 항목을 순서대로 확인한다.

### B.1 코드 품질

```
□ pytest tests/ -m "not slow" — 전체 유닛/통합 테스트 PASS
□ pytest --cov=utils --cov=api --cov-fail-under=80 — 커버리지 ≥ 80%
□ pip-audit -r requirements.txt — HIGH 이상 CVE 없음
□ 코드 내 eval(), exec(), yaml.load(), pickle.load() 없음
□ .gitignore에 dataset/, models/, experiments/, *.pth 포함 확인
```

### B.2 환경 설정

```
□ python scripts/preflight_check.py — 모든 항목 ✅
□ requirements.txt 버전 고정 확인
□ 필수 디렉터리 존재: experiments/, models/, logs/, results/
□ Node.js 18 이상 설치 확인 (Explorer / Vision npm 빌드용)
□ smart-qc-explorer: npm install 완료, npm run build 오류 없음
□ smart-qc-vision: npm install 완료, npm run build 오류 없음
□ FastAPI CORS: ALLOWED_ORIGINS에 Explorer(:5173)/Vision(:5174) 포함 확인
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
# FastAPI 백엔드
□ uvicorn api.main:app --reload --port 8000 — 정상 기동 확인
□ GET http://localhost:8000/health — 200 OK 반환

# Explorer (smart-qc-explorer)
□ Dataset 화면: 데이터셋 경로 입력 → POST /api/dataset/validate 성공
□ Config 화면: 전처리·모델 설정 저장 (PatchCore, resnet18, train_steps 제한 포함)
□ Training 화면: 학습 시작 → WS /ws/training 연결 → Progress bar 갱신 → 완료 메시지
□ Training 화면: [학습 중단] 버튼 → stopped 메시지 수신
□ Experiments 화면: 실험 목록 표시 → 선택 → 결과 상세 확인
□ AnomalyMap 화면: Anomaly Map 빌드 → 이미지 그리드 + Threshold 슬라이더 표시

# Vision (smart-qc-vision)
□ Realtime 화면: 모델 적용 후 수동 검사 버튼 클릭 → 3열 결과 표시
□ Realtime 화면: 자동 검사 시작 → WS /ws/inspection/auto 연결 확인
□ History 화면: 검사 후 이력 테이블 및 KPI 카드 표시 확인
□ Models 화면: 완료 실험 목록 F1 기준 정렬 확인, 모델 교체 후 이력 초기화 확인
```

---

## C. Local Deployment Procedure

### C.1 최초 설치 (신규 머신)

3개 레포를 각각 설치한다. 실행 순서: **백엔드 먼저 → 프론트엔드**.

```bash
# ──────────────────────────────────────────────
# [1단계] 백엔드: smart-qc-dashboard (FastAPI)
# ──────────────────────────────────────────────

git clone <repo-url> smart-qc-dashboard
cd smart-qc-dashboard

# conda 환경 생성 (09절 §D.1)
conda create -n smart-qc python=3.12 -y
conda activate smart-qc

# PyTorch 설치 (CUDA 12.4)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
    -c pytorch -c nvidia -y

# 나머지 의존성
pip install -r requirements.txt

# 환경 검증
python scripts/preflight_check.py

# ImageNet penalty 데이터 배치 (EfficientAD 사용 시)
# ./dataset/imagenet_penalty/ 에 이미지 파일 배치

# FastAPI 서버 실행 (개발 모드)
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000  (다른 터미널에서 프론트엔드 실행)


# ──────────────────────────────────────────────
# [2단계] Explorer: smart-qc-explorer (React)
# ──────────────────────────────────────────────

cd ../smart-qc-explorer

# 의존성 설치 (Node.js 18+ 필요)
npm install

# 개발 서버 실행
npm run dev
# → http://localhost:5173


# ──────────────────────────────────────────────
# [3단계] Vision: smart-qc-vision (React)
# ──────────────────────────────────────────────

cd ../smart-qc-vision

npm install

# Explorer와 동시 실행 시 포트 충돌 — 5174 사용
npx vite --port 5174
# → http://localhost:5174

# Explorer 미실행 시 기본 포트 사용 가능
# npm run dev  → http://localhost:5173
```

### C.2 업데이트 배포 (기존 설치)

```bash
# 백엔드 업데이트
cd smart-qc-dashboard
conda activate smart-qc
git pull origin main
pip install -r requirements.txt
# 마이그레이션 스크립트 실행 (있는 경우)
# python scripts/migrate_history.py

# FastAPI 서버 재시작
uvicorn api.main:app --reload --port 8000


# Explorer 업데이트
cd ../smart-qc-explorer
git pull origin main
npm install
npm run dev


# Vision 업데이트
cd ../smart-qc-vision
git pull origin main
npm install
npx vite --port 5174
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
| 포트 | 8000/tcp (FastAPI), 22/tcp (SSH) |

> **v2.0 변경**: v1.x의 8501/tcp (Streamlit) 제거. React 프론트엔드는 `npm run build` 후 정적 파일을 FastAPI 또는 Nginx로 서빙하거나, EC2 내부 포트(`5173/5174`)는 방화벽으로 차단하고 리버스 프록시(Nginx)를 통해 80/443으로만 외부 노출한다. [확인 필요: EC2 환경에서 React 정적 파일 서빙 방식 — FastAPI StaticFiles vs. Nginx]

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

# 6. Node.js 18 설치 (React 빌드용)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### D.3 애플리케이션 배포

```bash
# ── 백엔드 (smart-qc-dashboard) ──────────────
git clone <repo-url> smart-qc-dashboard
cd smart-qc-dashboard

conda create -n smart-qc python=3.12 -y
conda activate smart-qc
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt

# 데이터셋 마운트 (EBS 또는 S3 sync)
# aws s3 sync s3://your-bucket/imagenet_penalty ./dataset/imagenet_penalty/

# 백그라운드 실행 (tmux 권장)
tmux new -s smartqc-api
uvicorn api.main:app --host 127.0.0.1 --port 8000
# 보안: 127.0.0.1 바인딩 — Nginx 리버스 프록시를 통해 외부 노출


# ── Explorer (smart-qc-explorer) ─────────────
cd ../smart-qc-explorer
npm install
npm run build
# dist/ 디렉터리 생성 → Nginx 또는 FastAPI StaticFiles로 서빙
# [확인 필요: 실제 정적 파일 서빙 경로 설정]


# ── Vision (smart-qc-vision) ─────────────────
cd ../smart-qc-vision
npm install
npm run build
# dist/ 디렉터리 생성 → Nginx 서빙 (별도 경로 또는 서브도메인)
# [확인 필요: Explorer와 Vision의 Nginx 경로 분리 방법]
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
| FastAPI (Nginx 리버스 프록시) | 80/tcp, 443/tcp | 필요한 IP 범위만 (전체 공개 금지) |
| FastAPI 직접 접근 | 8000/tcp | 차단 (127.0.0.1 바인딩, Nginx 경유) |

> **v2.0 변경**: v1.x의 Streamlit 8501/tcp 규칙 제거. 외부 노출은 Nginx(80/443) 단일 포트를 통해 FastAPI API와 React 정적 파일 모두 서빙한다.

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
  # ── Python 백엔드 (smart-qc-dashboard) ──────
  test-backend:
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
        run: |
          pytest tests/ -m "not slow" \
            --cov=utils --cov=api \
            --cov-report=term-missing \
            --cov-fail-under=80 -v

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage

  # ── Explorer TypeScript 빌드 (smart-qc-explorer) ──
  build-explorer:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ../smart-qc-explorer  # [확인 필요: 모노레포 vs. 별도 레포 구조]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Node.js 18
        uses: actions/setup-node@v4
        with:
          node-version: "18"
      - name: Install and build
        run: |
          npm install
          npm run build

  # ── Vision TypeScript 빌드 (smart-qc-vision) ──────
  build-vision:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ../smart-qc-vision  # [확인 필요: 모노레포 vs. 별도 레포 구조]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Node.js 18
        uses: actions/setup-node@v4
        with:
          node-version: "18"
      - name: Install and build
        run: |
          npm install
          npm run build
```

> **[확인 필요]**: Explorer/Vision이 별도 레포인 경우 각 레포에 CI 워크플로우를 독립적으로 배치하고 `working-directory` 대신 해당 레포 루트를 기준으로 설정한다.

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
            ## Smart QC ${{ github.ref_name }}
            
            ### 포함 레포
            - smart-qc-dashboard (FastAPI :8000)
            - smart-qc-explorer (React Explorer)
            - smart-qc-vision (React Vision)
            
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

# Explorer / Vision npm 의존성 롤백
cd smart-qc-explorer && npm install
cd ../smart-qc-vision && npm install
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
| MAJOR | 하위 호환 불가 변경 (history.json 스키마 변경, API 엔드포인트 구조 변경, 화면 구조 변경) |
| MINOR | 신기능 추가 (하위 호환 유지) |
| PATCH | 버그 수정 |

### G.2 현재 릴리즈 계획

| 버전 | 내용 |
|------|------|
| **v1.0.0** | MVP — Explorer 5화면 + Vision 3화면 + FastAPI 백엔드 전체 기능 (이 PRD 대상) |
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

## H. 비전검사 대시보드 배포 체크리스트 (v2.0)

> **v2.0 변경**: v1.1의 Streamlit `inspection/tabs/insp_tab*.py` 파일 체크 및 session_state 초기화 검증이  
> FastAPI endpoint (`api/routers/inspection.py`, `api/ws/inspection_ws.py`) + Vision React 파일 체크로 교체됐다.

### H.1 신규 파일 생성 확인

배포 전 아래 파일이 소스 저장소에 존재하는지 확인한다.

```
# ── FastAPI 백엔드 (smart-qc-dashboard) ──────────────
□ api/routers/inspection.py
□ api/ws/inspection_ws.py
□ inspection/utils/__init__.py
□ inspection/utils/test_sampler.py

# ── Vision React (smart-qc-vision) ────────────────────
□ src/main.tsx
□ src/App.tsx
□ src/pages/Tab1Realtime.tsx
□ src/pages/Tab2History.tsx
□ src/pages/Tab3Setting.tsx
□ src/hooks/useAutoInspection.ts
□ src/hooks/useManualInspection.ts
□ src/hooks/useApplyModel.ts
□ src/hooks/useInspectionRecords.ts
□ src/store/inspectionStore.ts
□ src/components/layout/NoModelGuard.tsx
```

### H.2 검사 상태 초기화 검증

```
□ FastAPI 서버 최초 기동 시 GET /api/inspection/model 응답 null (미적용 모델)
□ POST /api/inspection/model 호출 후 GET /api/inspection/records 응답 [] (초기화 확인)
□ 모델 교체(POST /api/inspection/model — 새 모델) 시 GET /api/inspection/records 빈 배열 반환
□ Vision Zustand inspectionStore: 모델 교체 후 records 초기화 확인
□ [확인 필요: FastAPI 서버 재시작 시 검사 이력 in-memory 초기화 vs. DB 영속 여부]
```

### H.3 모델 캐시 검증

```
□ POST /api/inspection/model 동일 experiment_id로 2회 호출 시 2번째 호출에서 모델 재로드 없음
  (서버 측 캐시 — st.cache_resource 제거됨, FastAPI 서버 메모리 캐시 사용)
□ 모델 교체 후 이전 모델 캐시 무효화 확인
□ 추론 응답 시간 ≤ 3초 확인 (11절 NFR-P-05)
□ WS 자동 검사 — 서버 측 asyncio.sleep(3) 타이밍 오차 ±0.5초 이내 확인
□ 불량 팝업 표시 지연 ≤ 0.5초 확인 (11절 B.8 WS NFR)
```

### H.4 WBS — 비전검사 기능 요구사항

배포 전 아래 기능 요구사항 구현이 완료되어야 한다.

| 요구사항 ID | 설명 | 구현 파일 (v2.0) |
|------------|------|----------------|
| FR-INSP-CMN-01 | Explorer ↔ Vision 화면 전환 (브라우저 탭 또는 URL 전환) | Vision: `src/App.tsx` TabBar |
| FR-INSP-CMN-02 | 서버 측 검사 상태 초기화 (모델 적용 시) | FastAPI: `api/routers/inspection.py` `POST /api/inspection/model` |
| FR-INSP-CMN-03 | `inspection_record` 스키마 준수 (seq/inspected_at/image_name/image_path/verdict/anomaly_score) | FastAPI: `api/routers/inspection.py` |
| FR-INSP-T1-01 | 수동 검사 버튼 | Vision: `src/components/tab1/InspectionControls.tsx` + `useManualInspection.ts` |
| FR-INSP-T1-02 | 자동 검사 (서버 측 asyncio 3초 간격, WS 스트림) | Vision: `src/hooks/useAutoInspection.ts` / FastAPI: `api/ws/inspection_ws.py` |
| FR-INSP-T1-03 | 3열 레이아웃: 판정결과 / 원본이미지 / Anomaly Map | Vision: `src/components/tab1/ImagePanel.tsx`, `VerdictCard.tsx` |
| FR-INSP-T1-04 | 불량 감지 팝업 표시 및 자동 검사 중지 | Vision: `useAutoInspection.ts` (`inspectionStore.showDefectPopup`), `ws.close()` |
| FR-INSP-T1-05 | 테스트 풀 소진 시 재섞기 | FastAPI: `inspection/utils/test_sampler.py` |
| FR-INSP-T1-06 | 모델 미적용 상태 가드 | Vision: `src/components/layout/NoModelGuard.tsx` |
| FR-INSP-T2-01 | 5열 이력 테이블 (번호/시각/이미지명/판정결과/Anomaly Score) | Vision: `src/components/tab2/RecordsTable.tsx` |
| FR-INSP-T2-02 | KPI 카드 4개 (총검사/양품/불량/불량률) | Vision: `src/components/tab2/KpiCards.tsx` |
| FR-INSP-T2-03 | 이력 없음 상태 가드 | Vision: `src/pages/Tab2History.tsx` |
| FR-INSP-T2-04 | 통계 차트 (Anomaly Score 히스토그램 + 산점도, N개 단위 그룹) | Vision: `src/components/tab2/ScoreHistogram.tsx`, `ScoreScatter.tsx` |
| FR-INSP-T2-05 | CSV 내보내기 및 이력 초기화 | Vision: `src/hooks/useInspectionRecords.ts` / FastAPI: `GET /api/inspection/records/csv`, `DELETE /api/inspection/records` |
| FR-INSP-T3-01 | 완료된 실험 목록 F1 기준 정렬 | Vision: `src/components/tab3/ModelTable.tsx` |
| FR-INSP-T3-02 | 모델 [적용] 버튼 → 서버에 `POST /api/inspection/model` | Vision: `src/components/tab3/ApplyModelButton.tsx` + `useApplyModel.ts` |
| FR-INSP-T3-03 | 모델 교체 시 모든 검사 이력 초기화 | Vision: `useApplyModel.ts` (store 초기화) / FastAPI: `POST /api/inspection/model` (서버 이력 초기화) |

### H.5 배포 전 수동 검증 체크리스트 추가 항목

```
# Vision 접속 확인
□ 브라우저에서 http://localhost:5174 (또는 :5173) 접속 → Vision 화면 정상 표시
□ Vision 최초 접속 시 모델 미적용 상태 — NoModelGuard 표시 확인

# Vision Realtime 화면 (/)
□ Models 화면에서 모델 적용 후 Realtime 화면으로 이동 → NoModelGuard 해제 확인
□ 수동 검사 버튼 클릭 → POST /api/inspection/run → 3열 결과(판정/원본/Anomaly Map) 표시 확인
□ 자동 검사 시작 → WS /ws/inspection/auto 연결 → 3초 간격 자동 검사 동작 확인
□ 불량 이미지 검사 → 팝업 표시 및 자동 검사 중지 확인 (ws.close() 트리거)

# Vision History 화면 (/history)
□ 검사 후 이력 테이블 및 KPI 카드 표시 확인
□ CSV 내보내기 버튼 동작 확인
□ 이력 초기화 버튼 → GET /api/inspection/records 응답 빈 배열 확인

# Vision Models 화면 (/models)
□ 완료 실험 목록 F1 기준 정렬 확인
□ 모델 교체 후 History 화면 이력이 초기화됨을 확인 (TC-INSP-03)
□ [확인 필요: 모델 30초 폴링 동작 확인 — useModels.ts]
```

---

*[14_Deployment_and_Release_Plan.md 끝]*
