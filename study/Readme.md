# 🔩 EfficientAD 나사 이상 감지 Streamlit 앱

MVTec Screw 데이터셋을 사용한 **비지도 이상 감지** 시스템입니다.  
EfficientAD의 Teacher-Student + AutoEncoder 구조를 PyTorch로 구현했습니다.

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 앱 실행
streamlit run app.py
```

---

## 앱 사용법

### 탭 1: 모델 학습
사이드바에서 파라미터 조정 후 **[학습 시작]** 클릭  
→ `screw/train/good/` 폴더의 정상 이미지로 Teacher 특징 통계 계산

### 탭 2: 이상 감지
- **데이터셋에서 선택**: test 폴더의 이미지를 유형별로 선택
- **직접 업로드**: 나사 이미지 파일 업로드  
→ 원본 / 히트맵 / 오버레이 3종 시각화 + 판정 결과

### 탭 3: 배치 평가
여러 결함 유형을 한꺼번에 평가해 정확도·F1·점수 분포를 확인

---

## 조정 가능한 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| PDN 모드 | `small`(경량) / `medium`(고성능) | `small` |
| 출력 채널 수 | Teacher/Student 특징 차원 (128~512) | 384 |
| 입력 이미지 크기 | 128~320px | 256 |
| 학습 이미지 수 | train/good 사용 수량 | 80 |
| AE 가중치 | Teacher-Student vs AutoEncoder 비중 | 0.5 |
| 이상 임계값 | 이상 판정 기준 점수 | 0.5 |
| 히트맵 투명도 | 오버레이 알파값 | 0.5 |

---

## 모델 구조

```
입력 이미지
    ├── Teacher (PDN, 고정 특징 추출기)
    │       └── 정규화 통계 (μ, σ)
    │
    ├── Student (PDN × 2채널)
    │       ├── 앞절반: Teacher 모사
    │       └── 뒷절반: AE 출력 모사
    │
    └── AutoEncoder (Encoder → Bottleneck → Decoder)

이상 점수 = (1-w) × ||T - S_t||² + w × ||AE - S_ae||²
```

---

## 데이터셋 구조

```
screw/
├── train/
│   └── good/          # 정상 이미지 (학습용)
└── test/
    ├── good/          # 정상 (평가용)
    ├── manipulated_front/
    ├── scratch_head/
    ├── scratch_neck/
    ├── thread_side/
    └── thread_top/
```