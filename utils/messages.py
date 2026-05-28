MSG: dict[str, str] = {
    "NO_DATASET":       "먼저 탭1에서 데이터 폴더를 설정해 주세요.",
    "NO_PREPROCESSING": "먼저 탭2에서 전처리 설정을 완료해 주세요.",
    "NO_MODEL_CONFIG":  "먼저 탭3에서 모델 파라미터를 설정해 주세요.",
    "NO_EXPERIMENTS":   "아직 실행된 실험이 없습니다. 탭4에서 학습을 먼저 실행해 주세요.",
    "NO_SELECTED_EXP":  "탭5에서 분석할 실험을 먼저 선택해 주세요.",
    "GRAYSCALE_DETECT": "Grayscale 이미지가 감지되었습니다. 모델 입력을 위해 RGB 3채널로 자동 변환됩니다.",
    "INVALID_FOLDER":   "MVTec AD 형식의 폴더 구조가 아닙니다. (필수: train/good/, test/, ground_truth/)",
    "TRAIN_STOPPED":    "학습이 중단되었습니다. 해당 실험은 '중단' 상태로 히스토리에 기록되었습니다.",
}

ERR: dict[str, str] = {
    "ERR_DATASET_NOT_FOUND":           "지정 경로가 존재하지 않습니다.",
    "ERR_INVALID_FOLDER_STRUCTURE":    "MVTec AD 폴더 구조가 충족되지 않았습니다.",
    "ERR_NO_VALID_IMAGES":             "지원 포맷 이미지가 없습니다.",
    "ERR_PREPROCESSING_CONFIG_MISSING":"전처리 설정이 없습니다.",
    "ERR_MODEL_CONFIG_MISSING":        "모델 설정이 없습니다.",
    "ERR_MODEL_INIT_FAILED":           "모델 초기화에 실패했습니다.",
    "ERR_TRAINING_INTERRUPTED":        "학습이 강제 중단되었습니다.",
    "ERR_CONFIG_LOAD_FAILED":          "YAML 파싱에 실패했습니다.",
    "ERR_MODEL_SAVE_FAILED":           "모델 저장에 실패했습니다.",
    "ERR_EXPERIMENT_NOT_FOUND":        "해당 실험 ID가 존재하지 않습니다.",
    "ERR_INVALID_PARAM_RANGE":         "파라미터 값이 허용 범위를 벗어났습니다.",
    # 비전검사 대시보드 에러 코드 (00_Global §3.5)
    "ERR_INSP_NO_MODEL":              "선택된 모델이 없습니다.",
    "ERR_INSP_MODEL_LOAD_FAILED":     "모델 로드에 실패했습니다.",
    "ERR_INSP_TEST_POOL_EMPTY":       "테스트 이미지가 없습니다.",
    "ERR_INSP_INFERENCE_FAILED":      "추론에 실패했습니다.",
    "ERR_INSP_MODEL_NOT_COMPLETED":   "완료되지 않은 실험입니다.",
}

# 비전검사 대시보드 안내 메시지 (00_Global §3.4)
INSP_MSG: dict[str, str] = {
    "NO_MODEL":         "선택된 모델이 없습니다. 탭3 [딥러닝 모델 교체]에서 모델을 선택한 후 검사를 시작해 주세요.",
    "NO_COMPLETED_EXP": "적용 가능한 완료 실험이 없습니다. 모델 탐색 대시보드에서 학습을 완료해 주세요.",
    "DEFECT_DETECTED":  "불량이 감지되었습니다! 자동 검사가 중지되었습니다.",
    "MODEL_REPLACED":   "모델이 교체되었습니다. 검사 이력이 초기화되었습니다.",
    "HISTORY_CLEARED":  "검사 이력이 초기화되었습니다.",
    "AUTO_STOPPED":     "불량 감지로 자동 검사가 중지되었습니다. 확인 후 검사를 재시작해 주세요.",
    "POOL_RESHUFFLED":  "테스트 이미지 풀을 모두 소진하여 재구성했습니다.",
    "POOL_EMPTY":       "테스트 이미지를 찾을 수 없습니다. 데이터셋 경로를 확인하거나 탭3에서 모델을 재선택해 주세요.",
}
