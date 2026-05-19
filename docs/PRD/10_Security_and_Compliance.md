# 10. Security and Compliance

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [09_Infrastructure_and_Cloud.md](./09_Infrastructure_and_Cloud.md)
> **버전**: v1.0
> **작성일**: 2026-05-09
> **중요**: 이 시스템은 단일 사용자 로컬 실행 대시보드다. 클라우드·멀티테넌시·외부 API 연동이 없으므로 보안 요구사항의 범위는 **로컬 파일시스템 보호**와 **Streamlit 기본 보안 설정**에 한정된다.

---

## 목차

- [A. Scope and Threat Model](#a-scope-and-threat-model)
- [B. File System Security](#b-file-system-security)
- [C. Streamlit Security Configuration](#c-streamlit-security-configuration)
- [D. Dependency Security](#d-dependency-security)
- [E. Sensitive Data Handling](#e-sensitive-data-handling)
- [F. Docker Security (Optional)](#f-docker-security-optional)
- [G. Security Checklist](#g-security-checklist)

---

## A. Scope and Threat Model

### A.1 시스템 특성

| 항목 | 값 |
|------|-----|
| 배포 환경 | 단일 워크스테이션 (로컬) |
| 동시 사용자 | 1명 (00절 §9 A-01) |
| 외부 네트워크 연결 | 없음 (모델 다운로드 등 초기 설정 제외) |
| 인증/인가 | 없음 (로컬 단일 사용자) |
| 데이터 민감도 | 내부 제조 이미지 (공개 금지 가능) |

### A.2 위협 범위

이 문서에서 다루는 위협:

| 위협 ID | 위협 | 관련 자산 |
|---------|------|---------|
| T-01 | 경로 탈출 (Path Traversal) | 데이터셋 경로 입력 |
| T-02 | 디스크 과부하 (Disk Exhaustion) | 모델 저장, 로그 축적 |
| T-03 | 민감 파일 노출 | 데이터셋 이미지, 실험 설정 |
| T-04 | 악의적 YAML/JSON 로드 | `configs.yaml`, `history.json` |
| T-05 | 브라우저 외부 노출 | Streamlit 서버 포트 |

### A.3 스코프 외 위협

| 위협 | 이유 |
|------|------|
| SQL Injection | 현재 앱 코드에서 DB 직접 쿼리 없음 — DB 연결 코드 추가 시 parameterized query 필수 |
| XSS / CSRF | 단일 사용자 로컬, 외부 입력 없음 |
| 인증 우회 | 인증 없음 (로컬 단일 사용자) |
| 공급망 공격 (Supply Chain) | requirements.txt 고정 버전으로 완화 |
| DDoS | 로컬 환경 적용 불가 |

---

## B. File System Security

### B.1 경로 탈출 방지 (T-01)

**위협**: 탭1 데이터셋 경로 입력에서 `../../../../etc/passwd` 같은 경로를 입력하여 시스템 파일 접근.

**대응**:

```python
# tab1 경로 검증 — utils/path_validator.py
from pathlib import Path

ALLOWED_BASE_DIR = Path("/app/dataset")  # `/app/dataset` 하위만 허용

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

    if not str(p).startswith(str(ALLOWED_BASE_DIR)):
        raise ValueError(f"/app/dataset 하위 경로만 허용됩니다: {p}")

    if not p.exists():
        raise ValueError(f"경로가 존재하지 않습니다: {p}")

    if not p.is_dir():
        raise ValueError(f"디렉터리가 아닙니다: {p}")

    return p
```

**적용 위치**: `tab1._handle_path_input()` — 사용자 입력 수신 직후 호출.

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

1. **저장 전 여유 공간 확인** (11절 §D.4):
```python
import shutil
def check_disk_space(min_free_mb: int = 500) -> bool:
    usage = shutil.disk_usage("./models")
    return usage.free >= min_free_mb * 1024 * 1024
```

2. **로그 파일 크기 제한**: 단일 실험 로그는 최대 약 5 MB (70k step 기준 추정). 파일 크기 상한을 강제하지 않되, 탭5 [실험 삭제] 시 로그 파일도 함께 삭제 (12절 §C.3).

3. **사용자 경고**: 여유 공간 < 500 MB 시 저장 전 `st.warning()` 표시 (차단 아닌 경고).

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

## C. Streamlit Security Configuration

### C.1 XSRF 보호 활성화

```toml
# .streamlit/config.toml
[server]
enableXsrfProtection = true   # 기본값 true — 명시적으로 유지
```

`enableXsrfProtection = false` 설정 **금지**.

### C.2 외부 노출 차단 (T-05)

**기본 실행** (`streamlit run app.py`): `localhost:8501` — 로컬만 접근 가능.

**외부 노출이 필요한 경우** (실험 환경 제한적으로만):
```bash
streamlit run app.py --server.address 0.0.0.0
# 위 설정 시 같은 네트워크의 다른 기기에서 접근 가능
# 인증 없으므로 신뢰할 수 있는 격리 네트워크에서만 사용
```

**운영 환경**: `--server.address 0.0.0.0` 사용 금지. `localhost` (기본값) 유지.

### C.3 파일 업로드 크기 제한

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200   # MB
```

탭1은 `st.text_input()` (경로 입력)을 사용하므로 파일 직접 업로드는 없다.  
`maxUploadSize` 는 향후 기능 추가 시 기본 제한으로 유지.

---

## D. Dependency Security

### D.1 버전 고정

보안 취약점이 있는 구버전 사용 방지:

```
# requirements.txt — 하한 버전 명시 (현재 알려진 취약 버전 이상)
torch>=2.1.0
anomalib>=1.0.0
Pillow>=10.0.0          # CVE-2023-44271 이전 버전 취약
PyYAML>=6.0             # CVE-2020-14343 이전 버전 취약 (yaml.load 사용 시)
```

### D.2 의존성 감사

```bash
# pip-audit 으로 알려진 CVE 확인 (설치 후 실행)
pip install pip-audit
pip-audit -r requirements.txt
```

정기적으로 (새 패키지 추가 시) 실행. 심각도 HIGH 이상 CVE는 즉시 패치.

### D.3 사용 금지 패턴

| 패턴 | 이유 |
|------|------|
| `yaml.load(f)` | Arbitrary code execution 가능 |
| `pickle.load()` | Arbitrary code execution 가능 |
| `eval()`, `exec()` | Code injection 가능 |
| `subprocess.run(shell=True, input=user_input)` | Command injection 가능 |

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
```

실수로 커밋되지 않도록 위 항목을 `.gitignore` 에 포함.

### E.3 Docker 이미지에 민감 파일 포함 금지

```dockerfile
# Dockerfile
# .dockerignore 파일로 빌드 컨텍스트에서 제외
```

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
향후 클라우드 스토리지 연동 등 추가 시:
- 비밀은 `.env` 파일에 저장, `.gitignore` 에 포함
- `python-dotenv` 로 로드 — 코드에 하드코딩 금지

---

## F. Docker Security (Optional)

### F.1 비루트 사용자 실행

```dockerfile
# Dockerfile — 보안 강화 시
FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 비루트 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --chown=appuser:appuser . .

RUN pip install --no-cache-dir -r requirements.txt

USER appuser

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]
```

### F.2 컨테이너 파일시스템 읽기 전용 (선택)

```yaml
# docker-compose.yml
services:
  smart-qc:
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

---

## G. Security Checklist

### G.1 구현 전 체크리스트

```
□ validate_dataset_path() — 경로 탈출 방지 구현
□ yaml.safe_load() — 모든 YAML 로드에 사용
□ torch.load(weights_only=True) — 모델 로드 시 적용
□ .gitignore — dataset/, models/, experiments/, *.pth 포함
□ .dockerignore — 동일 항목 포함
□ .streamlit/config.toml — enableXsrfProtection = true
□ check_disk_space() — 모델 저장 전 호출
```

### G.2 배포 전 체크리스트

```
□ pip-audit -r requirements.txt — HIGH 이상 CVE 없음
□ Streamlit 포트가 localhost 바인딩인지 확인
□ eval(), exec(), yaml.load(), pickle.load() 코드에 없음
□ 하드코딩된 패스워드/API 키 없음 (grep -r "password\|api_key\|secret")
```

---

*다음 문서*: [13_QA_and_Testing_Strategy.md](./13_QA_and_Testing_Strategy.md)
