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
    "ERR_WEIGHT_SUM_INVALID":          "ae_loss_weight + st_loss_weight != 1.0 입니다.",
}
