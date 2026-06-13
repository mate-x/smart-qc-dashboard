# 10. Security and Compliance

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md)
> **버전**: v2.0
> **작성일**: 2026-05-09
> **수정일**: 2026-06-11
> **중요**: 이 시스템은 단일 사용자 로컬 실행 플랫폼이다. FastAPI(:8000)와 React 프론트엔드(Explorer/Vision :5173)의 3개 레포 구조에서 보안 요구사항의 범위는 **로컬 파일시스템 보호**, **FastAPI CORS 및 API 엔드포인트 보안**, **의존성 안전성**에 한정된다.

---

## 버전 히스토리

| 버전 | 날짜 | 변경 요약 |
|------|------|-----------|
| v1.0 | 2026-05-09 | 최초 작성 — Streamlit 단독 앱 보안 설정 |
| v1.1 | 2026-05-26 | 비전검사 대시보드 보안 고려사항 추가 (H절) |
| v2.0 | 2026-06-11 | C절 "Streamlit 보안" → "FastAPI 보안"으로 전면 교체; CORS 설정 및 API 엔드포인트 노출 범위 추가; 위협 모델 업데이트 (T-05/T-06/T-07); H절 React/FastAPI 아키텍처 기준으로 교체; MVP 미구현 항목 ❌ 표기 |

---

## 목차

- [A. Scope and Threat Model](#a-scope-and-threat-model)
- [B. File System Security](#b-file-system-security)
- [C. FastAPI Security Configuration](#c-fastapi-security-configuration)
- [D. Dependency Security](#d-dependency-security)
- [E. Sensitive Data Handling](#e-sensitive-data-handling)
- [F. Docker Security (Optional)](#f-docker-security-optional)
- [G. Security Checklist](#g-security-checklist)
- [H. React 프론트엔드 보안](#h-react-프론트엔드-보안-v20)

---

## A. Scope and Threat Model

### A.1 시스템 특성

| 항목 | 값 |
|------|-----|
| 배포 환경 | 단일 워크스테이션 (로컬) |
| 동시 사용자 | 1명 (00절 §9 A-01) |
| 외부 네트워크 연결 | 없음 (모델 다운로드 등 초기 설정 제외) |
| 인증/인가 | ❌ 없음 (로컬 단일 사용자, A-15 원칙) |
| 데이터 민감도 | 내부 제조 이미지 (공개 금지 가능) |
| API 서버 | FastAPI :8000 (로컬 바인딩 기본) |
| UI 클라이언트 | React Explorer :5173, Vision :5173/:5174 |

### A.2 위협 범위

이 문서에서 다루는 위협:

| 위협 ID | 위협 | 관련 자산 | v2.0 변경 |
|---------|------|---------|-----------|
| T-01 | 경로 탈출 (Path Traversal) | 데이터셋 경로 입력 (`POST /api/dataset/validate`) | 적용 위치: FastAPI 레이어 |
| T-02 | 디스크 과부하 (Disk Exhaustion) | 모델 저장, 로그 축적 | 변경 없음 |
| T-03 | 민감 파일 노출 | 데이터셋 이미지, 실험 설정 | 변경 없음 |
| T-04 | 악의적 YAML/JSON 로드 | `configs.yaml`, `history.json` | 변경 없음 |
| T-05 | 서버 포트 외부 노출 | FastAPI :8000, React dev :5173 | Streamlit → FastAPI/React로 갱신 |
| T-06 | CORS 오설정 | FastAPI → 외부 오리진 허용 | **v2.0 신규** |
| T-07 | WebSocket 무단 연결 | `/ws/training`, `/ws/inspection/auto` | **v2.0 신규** |

### A.3 스코프 외 위협

| 위협 | 이유 |
|------|------|
| SQL Injection | 파일시스템 기반 설계 — DB 직접 쿼리 없음. DB 연결 코드 추가 시 parameterized query 필수 |
| XSS | React는 JSX 렌더링으로 XSS 대부분 방어. 로컬 단일 사용자 환경에서 외부 입력 없음 |
| CSRF | CORS를 통해 오리진 제한이 적용됨 (C.1 참조). 인증 없는 로컬 환경 |
| 인증 우회 | 인증 없음 (로컬 단일 사용자, A-15) |
| 공급망 공격 (Supply Chain) | requirements.txt 하한 버전 명시로 완화 |
| DDoS | 로컬 환경 적용 불가 |
| HTTPS / TLS | ❌ MVP 미구현 — 로컬 HTTP 전용. 외부 노출 시 nginx reverse proxy + TLS 적용 권장 |

---

## B. File System Security

### B.1 경로 탈출 방지 (T-01)

**위협**: `POST /api/dataset/validate` 요청 바디의 `path` 필드에 `../../../../etc/passwd` 같은 경로를 입력하여 시스템 파일 접근.

**대응**:

```python
# utils/path_validator.py
from pathlib import Path

def validate_dataset_path(user_input: str) -> Path:
    """
    1. 빈 문자열 차단
    2. 절대 경로로 resolve (symlink 포함)
    3. 실제 디렉터리인지 확인
    Returns: resolved Path
    Raises: ValueError on invalid input
    """
    if not user_input or not user_input.strip():
        raise ValueError("경로를 입력해 주세요.")

    try:
        p = Path(user_input.strip()).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"유효하지 않은 경로입니다: {e}")

    if not p.exists():
        raise ValueError(f"경로가 존재하지 않습니다: {p}")

    if not p.is_dir():
        raise ValueError(f"디렉터리가 아닙니다: {p}")

    return p
```

> **로컬 환경 적용**: 연구 목적 로컬 환경에서 `ALLOWED_BASE_DIR` 고정 제한은 편의성을 해친다.  
> 대신 symlink resolve + 존재·디렉터리 여부 확인으로 기본 탈출을 방지한다.  
> 프로덕션 배포 시에는 허용 기반 경로(`/data/` 등)로 강화한다.

**적용 위치**: `api/routers/dataset.py` — `POST /api/dataset/validate` 핸들러 진입 직후 호출.

---

### B.2 모델 디렉터리 격리

**위협**: 모델 저장 경로가 시스템 디렉터리를 덮어쓰는 경우.

**대응**: 모델 저장 경로는 코드에 하드코딩된 상대 경로만 사용.

```python
# utils/storage.py
MODELS_DIR = Path("./models")      # 변경 불가
LOGS_DIR = Path("./logs")          # 변경 불가
HISTORY_FILE = Path("./experiments/history.json")  # 변경 불가
```

사용자가 지정하는 유일한 경로는 `dataset_path` (읽기 전용) 뿐이며, `validate_dataset_path()`로 검증된다.

---

### B.3 디스크 용량 보호 (T-02)

**위협**: 학습 로그 무제한 축적 또는 대용량 모델 반복 저장으로 디스크 고갈.

**대응**:

1. **저장 전 여유 공간 확인**:
```python
import shutil

def check_disk_space(min_free_mb: int = 500) -> bool:
    usage = shutil.disk_usage("./models")
    return usage.free >= min_free_mb * 1024 * 1024
```

2. **로그 파일 크기 제한**: 단일 실험 로그는 최대 약 5 MB (70k step 기준 추정). 파일 크기 상한을 강제하지 않되, 실험 삭제(`DELETE /api/experiments/{id}`) 시 로그 파일도 함께 삭제.

3. **사용자 경고**: 여유 공간 < 500 MB 시 API 응답에 `"disk_warning": true` 포함 (차단 아닌 경고).

---

### B.4 YAML / JSON 파싱 안전성 (T-04)

**위협**: 외부에서 수정된 `configs.yaml` 또는 `history.json` 이 악의적인 YAML 태그를 포함.

**대응**:

```python
# utils/config_manager.py
import yaml

def load_config(path: str = "./configs.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        # yaml.safe_load() — !!python/object 등 임의 코드 실행 태그 차단
        return yaml.safe_load(f) or {}
```

- `yaml.load(f, Loader=yaml.FullLoader)` 또는 `yaml.load(f)` 사용 **금지**
- `yaml.safe_load()` 만 허용 (임의 Python 객체 역직렬화 차단)
- `json.load()` 는 기본적으로 안전 (임의 코드 실행 불가)

---

## C. FastAPI Security Configuration

> v2.0: Streamlit Security Configuration 절이 FastAPI Security Configuration으로 전면 교체됨.

### C.1 CORS 설정 (T-06 대응)

**위협**: CORS 와일드카드(`*`) 또는 과도한 오리진 허용으로 외부 오리진에서 API 호출 가능.

**대응**:

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 허용 오리진 — React 개발 서버만 명시
ALLOWED_ORIGINS = [
    "http://localhost:5173",    # Explorer 기본 포트
    "http://localhost:5174",    # Vision (Explorer 동시 실행 시 대체 포트)
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,    # 인증 없는 로컬 환경
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)
```

> **`allow_origins=["*"]` 사용 금지**: 로컬 환경이라도 와일드카드는 악성 웹페이지에서  
> 로컬 API를 호출할 수 있게 한다 (T-06). 허용 오리진은 위 목록으로 명시 고정.

프로덕션 빌드 서빙 시 (`npm run build` → nginx 정적 서빙): CORS는 nginx가 동일 도메인에서 서빙할 경우 불필요할 수 있다. [확인 필요: 프로덕션 배포 방식 확정 후 CORS 설정 재검토]

### C.2 API 엔드포인트 노출 범위

FastAPI는 기본적으로 `--host 127.0.0.1`에 바인딩되어 로컬에서만 접근 가능하다.  
아래는 노출되는 엔드포인트와 인증 요구사항이다 (MVP 기준 — 인증 없음).

**Explorer 연동 엔드포인트**:

| 메서드 | 경로 | 설명 | 인증 |
|--------|------|------|------|
| POST | `/api/dataset/validate` | 데이터셋 경로 검증 | ❌ 없음 |
| GET | `/api/dataset/thumbnail/{class_name}` | 클래스 대표 이미지 | ❌ 없음 |
| GET | `/api/config` | 현재 설정 조회 | ❌ 없음 |
| POST | `/api/config` | 설정 저장 | ❌ 없음 |
| POST | `/api/config/preview` | Threshold 미리보기 | ❌ 없음 |
| GET/POST/DELETE | `/api/queue` | 학습 큐 관리 | ❌ 없음 |
| POST | `/api/training/start` | 학습 시작 | ❌ 없음 |
| POST | `/api/training/stop` | 학습 중단 | ❌ 없음 |
| POST | `/api/training/batch/*` | 배치 학습 | ❌ 없음 |
| WS | `/ws/training` | 학습 진행 WebSocket | ❌ 없음 |
| GET | `/api/experiments` | 실험 목록 | ❌ 없음 |
| POST | `/api/experiments/{id}/save` | 모델 저장 | ❌ 없음 |
| DELETE | `/api/experiments/{id}` | 실험 삭제 | ❌ 없음 |
| POST | `/api/anomaly-map/{expId}/build` | Anomaly Map 생성 | ❌ 없음 |
| GET | `/api/anomaly-map/{expId}/images` | 이미지 목록 | ❌ 없음 |
| GET | `/api/anomaly-map/{expId}/export/csv` | CSV 다운로드 | ❌ 없음 |

**Vision 연동 엔드포인트**:

| 메서드 | 경로 | 설명 | 인증 |
|--------|------|------|------|
| GET | `/api/models` | 실험 목록 | ❌ 없음 |
| GET/POST | `/api/inspection/model` | 모델 조회/적용 | ❌ 없음 |
| POST | `/api/inspection/run` | 수동 검사 실행 | ❌ 없음 |
| GET/DELETE | `/api/inspection/records` | 이력 조회/초기화 | ❌ 없음 |
| GET | `/api/inspection/records/csv` | CSV 다운로드 | ❌ 없음 |
| GET | `/api/inspection/image/last` | 마지막 원본 이미지 | ❌ 없음 |
| GET | `/api/inspection/anomaly-map/last` | Anomaly Map 이미지 | ❌ 없음 |
| GET | `/api/inspection/overlay/last` | 오버레이 이미지 | ❌ 없음 |
| WS | `/ws/inspection/auto` | 자동 검사 WebSocket | ❌ 없음 |

> **MVP 인증 없음**: 모든 엔드포인트에 인증이 없다. 이는 A-15 원칙(단일 사용자 로컬 환경)에 따른 의도적 설계다. 다중 사용자 환경이나 외부 노출 시에는 API Key 또는 JWT 기반 인증을 추가해야 한다.

### C.3 FastAPI 서버 바인딩 주소 (T-05 대응)

**기본 실행** — localhost 전용:

```bash
# 권장: localhost만 허용
uvicorn api.main:app --host 127.0.0.1 --port 8000

# 개발 편의 (reload 포함)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**외부 노출이 필요한 경우** (실험 환경, 신뢰 네트워크에서만):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# 위 설정 시 같은 네트워크의 다른 기기에서 접근 가능
# 인증 없으므로 신뢰할 수 있는 격리 네트워크에서만 사용
```

**운영 환경**: `--host 0.0.0.0` 사용 금지. `127.0.0.1` (기본값) 유지.

### C.4 FastAPI 요청 크기 제한

이미지 파일 업로드는 없으나 (경로 문자열만 전달), JSON 페이로드 크기를 제한한다.

```python
# api/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 최대 요청 바디 크기: 10 MB (학습 설정 JSON, 큐 항목 등)
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"요청 크기가 너무 큽니다 (최대 {MAX_BODY_SIZE // 1024 // 1024} MB)"},
            )
    return await call_next(request)
```

### C.5 WebSocket 보안 (T-07 대응)

**위협**: 허가되지 않은 오리진에서 WebSocket 연결 시도.

**대응**: FastAPI의 `CORSMiddleware`는 WebSocket에 직접 적용되지 않는다. WebSocket 핸들러에서 `Origin` 헤더를 명시적으로 검증한다.

```python
# api/routers/training_ws.py
from fastapi import WebSocket, WebSocketDisconnect, HTTPException

ALLOWED_WS_ORIGINS = {
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
}

@app.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    origin = websocket.headers.get("origin", "")
    if origin and origin not in ALLOWED_WS_ORIGINS:
        await websocket.close(code=1008)  # Policy Violation
        return
    await websocket.accept()
    # ... 학습 이벤트 스트림
```

> [확인 필요: 실제 api/routers/ 코드에서 WebSocket origin 검증 구현 여부 확인]  
> 구현되지 않았다면 로컬 환경에서는 낮은 위험이나, 외부 노출 시 필수 적용.

### C.6 FastAPI Swagger UI 접근 제어

기본적으로 FastAPI는 `/docs` (Swagger UI)와 `/redoc`을 자동 생성한다.

**개발 환경**: `/docs` 유지 (개발·테스트 편의)

**프로덕션 배포 시** (외부 노출 가능성):
```python
# api/main.py — 프로덕션 모드에서 Swagger 비활성화
import os

app = FastAPI(
    docs_url="/docs" if os.getenv("ENVIRONMENT", "development") == "development" else None,
    redoc_url=None,
)
```

---

## D. Dependency Security

### D.1 버전 고정

보안 취약점이 있는 구버전 사용 방지:

```
# requirements.txt — 하한 버전 명시 (현재 알려진 취약 버전 이상)
torch>=2.7.0
anomalib>=1.0.0
Pillow>=10.0.0          # CVE-2023-44271 이전 버전 취약
PyYAML>=6.0             # CVE-2020-14343 이전 버전 취약 (yaml.load 사용 시)
fastapi>=0.111.0        # 최신 보안 패치 포함
uvicorn[standard]>=0.29.0
```

### D.2 의존성 감사

```bash
# pip-audit 으로 알려진 CVE 확인 (설치 후 실행)
pip install pip-audit
pip-audit -r requirements.txt
```

정기적으로 (새 패키지 추가 시) 실행. 심각도 HIGH 이상 CVE는 즉시 패치.

**Node.js 의존성 감사** (Explorer / Vision):

```bash
# npm audit — 알려진 취약점 확인
cd smart-qc-explorer && npm audit
cd smart-qc-vision && npm audit

# 자동 수정 가능한 취약점 처리
npm audit fix
```

### D.3 사용 금지 패턴

| 패턴 | 이유 |
|------|------|
| `yaml.load(f)` | Arbitrary code execution 가능 |
| `pickle.load()` | Arbitrary code execution 가능 |
| `eval()`, `exec()` | Code injection 가능 |
| `subprocess.run(shell=True, input=user_input)` | Command injection 가능 |
| CORS `allow_origins=["*"]` | 외부 오리진 API 호출 허용 |

`torch.load()` 사용 시: `weights_only=True` 옵션 필수 (PyTorch 2.x 기준, pickle 실행 제한).

```python
# utils/storage.py — 모델 로드 시
model_state = torch.load(model_path, map_location=device, weights_only=True)
```

---

## E. Sensitive Data Handling

### E.1 민감 파일 범위

이 시스템이 다루는 민감 데이터:

| 자산 | 민감도 | 위치 |
|------|--------|------|
| 제조 공정 이미지 | 기업 기밀 가능 | 사용자 지정 dataset_path |
| 학습된 모델 가중치 | 독점 기술 가능 | `./models/{exp_id}/` |
| 실험 설정 | 내부 파라미터 | `./experiments/history.json`, `./models/{exp_id}/configs.yaml` |

### E.2 Git 추적 제외

```gitignore
# .gitignore — 데이터·모델 파일 추적 제외
dataset/
experiments/
models/
logs/
results/
*.pth
configs.yaml
.env
```

실수로 커밋되지 않도록 위 항목을 `.gitignore` 에 포함.

### E.3 Docker 이미지에 민감 파일 포함 금지

```
# .dockerignore
dataset/
experiments/
models/
logs/
results/
*.pth
configs.yaml
.env
```

Docker 볼륨 마운트로 런타임에 연결 (09절 §H.2).

### E.4 환경 변수 / 비밀 관리

현재 MVP 버전에서 비밀 정보(API 키, 패스워드 등)는 없다.  
향후 클라우드 스토리지 연동 또는 사용자 인증 추가 시:
- 비밀은 `.env` 파일에 저장, `.gitignore` 에 포함
- `python-dotenv` 로 로드 — 코드에 하드코딩 금지
- React 환경변수는 `VITE_` 접두사 사용. `VITE_` 없는 변수는 클라이언트 번들에 포함되지 않음

---

## F. Docker Security (Optional)

### F.1 비루트 사용자 실행

```dockerfile
# smart-qc-dashboard Dockerfile — 보안 강화 시
FROM nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04

# 비루트 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --chown=appuser:appuser . .

RUN pip install --no-cache-dir -r requirements.txt

USER appuser

EXPOSE 8000
ENTRYPOINT ["uvicorn", "api.main:app", \
            "--host", "0.0.0.0", \
            "--port", "8000"]
```

### F.2 컨테이너 파일시스템 읽기 전용 (선택)

```yaml
# docker-compose.yml
services:
  backend:
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - ./dataset:/app/dataset:ro    # 데이터셋: 읽기 전용
      - ./experiments:/app/experiments:rw
      - ./models:/app/models:rw
      - ./logs:/app/logs:rw
      - ./results:/app/results:rw
```

### F.3 네트워크 격리

```yaml
# docker-compose.yml
services:
  backend:
    networks:
      - internal
  explorer:
    networks:
      - internal
  vision:
    networks:
      - internal

networks:
  internal:
    driver: bridge
    internal: false   # 외부 인터넷 차단 시 true로 변경 (모델 다운로드 불가)
```

---

## G. Security Checklist

### G.1 구현 체크리스트

```
□ validate_dataset_path() — POST /api/dataset/validate 핸들러에서 호출 확인
□ yaml.safe_load() — 모든 YAML 로드에 사용 확인
□ torch.load(weights_only=True) — 모델 로드 시 적용 확인
□ .gitignore — dataset/, models/, experiments/, *.pth, .env 포함 확인
□ .dockerignore — 동일 항목 포함 확인
□ CORS allow_origins — 와일드카드(*) 없이 localhost:5173/5174 명시 확인
□ uvicorn --host 127.0.0.1 — 기본 바인딩 localhost 유지 확인
□ check_disk_space() — 모델 저장 전 호출 확인
```

### G.2 배포 전 체크리스트

```
□ pip-audit -r requirements.txt — HIGH 이상 CVE 없음
□ npm audit (Explorer, Vision) — HIGH 이상 취약점 없음
□ FastAPI --host 127.0.0.1 바인딩 확인 (0.0.0.0 사용 시 이유 명시)
□ eval(), exec(), yaml.load(), pickle.load() 코드에 없음
□ 하드코딩된 패스워드/API 키 없음 (grep -r "password\|api_key\|secret")
□ CORS 허용 오리진이 와일드카드가 아닌 명시 목록인지 확인
□ WebSocket origin 검증 구현 여부 확인 [확인 필요]
□ /docs (Swagger UI) 외부 노출 시 비활성화
```

---

## H. React 프론트엔드 보안 (v2.0)

> **v2.0 변경**: v1.1의 "Streamlit 비전검사 대시보드 보안 고려사항"이 v2.0에서 "React 프론트엔드 보안"으로 교체됐다.  
> Streamlit session_state 기반 설계는 더 이상 적용되지 않으며, Explorer/Vision React 앱 기준으로 작성됐다.

### H.1 접근 제어 (A-15 유지)

00절 §9 A-15 원칙에 따라 Explorer와 Vision 사이에 별도 접근 제어가 없다.  
두 React 앱은 독립된 브라우저 탭에서 동작하며, 인증·인가를 구현하지 않는다.

| 항목 | 내용 |
|------|------|
| Explorer ↔ Vision 접근 제어 | 없음 (A-15) |
| FastAPI 엔드포인트 인증 | ❌ 없음 (MVP, A-15) |
| 이유 | 단일 사용자 로컬 환경 — 멀티테넌시 불필요 |

### H.2 React 앱의 파일시스템 직접 접근 없음

v1.1 Streamlit 아키텍처에서는 Python 서버 코드가 파일시스템에 직접 접근했다.  
v2.0 React 아키텍처에서는 **브라우저(React)가 파일시스템에 직접 접근하지 않는다**.  
모든 파일 I/O는 FastAPI 서버 레이어에서만 발생한다.

| 작업 | v1.1 (Streamlit) | v2.0 (React/FastAPI) |
|------|-----------------|----------------------|
| 데이터셋 경로 접근 | Python 직접 read | POST /api/dataset/validate → FastAPI가 검증 |
| 모델 저장 | Python 직접 write | POST /api/experiments/{id}/save → FastAPI 처리 |
| 이미지 렌더링 | st.image() 직접 | GET /api/inspection/image/last → blob URL |
| 검사 기록 | session_state 메모리 | FastAPI 메모리 + GET /api/inspection/records |

이 구조로 인해 React 앱 자체가 T-01(경로 탈출), T-02(디스크 과부하) 위협의 직접 대상이 되지 않는다.

### H.3 API 통신 보안

Explorer와 Vision은 `axios` 인스턴스를 통해 FastAPI와 통신한다.

```typescript
// src/api/client.ts (Explorer / Vision 공통 패턴)
import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000',
  timeout: 30_000,
});

export default client;
```

| 항목 | 내용 |
|------|------|
| baseURL | 환경변수(`VITE_API_BASE_URL`)로 주입, 기본값 localhost:8000 |
| credentials | 포함하지 않음 (인증 없는 환경) |
| timeout | 30초 (긴 추론 요청 고려) |
| HTTPS | ❌ MVP 미구현 — 로컬 HTTP 전용 |

### H.4 Vision React WebSocket 보안

Vision의 자동 검사는 `/ws/inspection/auto` WebSocket으로 동작한다.

```typescript
// src/hooks/useAutoInspection.ts (Vision)
const ws = new WebSocket(`ws://localhost:8000/ws/inspection/auto`);
```

- 브라우저는 WebSocket 연결 시 `Origin` 헤더를 자동으로 포함한다
- FastAPI 서버 측에서 C.5에 명시된 origin 검증을 수행한다 [확인 필요]
- 자동 검사 중단은 `ws.close()` 호출로 처리됨 (서버 측에서 연결 해제 감지)

### H.5 React 번들 내 민감 정보 금지

React 앱은 빌드 시 JavaScript 번들로 패키징된다. 번들은 브라우저에서 열람 가능하므로:

- API 키, 패스워드, 내부 경로를 소스코드에 하드코딩 금지
- `VITE_` 접두사가 있는 환경변수만 번들에 포함됨 (Vite 규칙)
- 비밀 값은 절대 `VITE_SECRET_KEY=...` 형태로 React 환경변수에 넣지 않음

| 환경변수 | 번들 포함 | 사용 예 |
|---------|---------|--------|
| `VITE_API_BASE_URL` | ✅ (노출 가능) | API 서버 주소 (민감 정보 아님) |
| `API_SECRET_KEY` (prefix 없음) | ❌ (번들 미포함) | Python 서버 측에서만 사용 |

### H.6 보안 체크리스트 추가 항목 (v2.0)

```
□ CORS 허용 오리진에 Explorer(:5173)/Vision(:5174) 포함 확인
□ VITE_ 접두사 환경변수에 민감 정보 없음 확인
□ React axios 인스턴스 baseURL이 환경변수로 주입되는지 확인
□ Vision WebSocket 자동 검사 중단 시 ws.close() 호출 확인
□ FastAPI WebSocket origin 검증 구현 확인 [확인 필요]
□ npm audit — HIGH 이상 취약점 없음 (Explorer, Vision 각각)
□ torch.load(weights_only=True) FastAPI 모델 로드 코드에 적용 확인
```

---

*다음 문서*: [13_QA_and_Testing_Strategy.md](./13_QA_and_Testing_Strategy.md)
