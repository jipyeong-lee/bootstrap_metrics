import os

# 데이터 경로
DATA_DIR = 'data'  # 데이터가 저장된 디렉토리
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')  # 학습 데이터 경로
VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid.csv')  # 검증 데이터 경로
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')    # 테스트 데이터 경로

# 예측 파일 경로
PRED_DIR = 'models/predictions'  # 예측 파일이 저장된 디렉토리

# 성능 지표 이름
METRIC_NAMES = [
    'AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Precision', 
    'F1', 'Accuracy', 'Brier', 'ICI', 'Calibration Slope', 
    'Calibration Intercept', 'Unreliability p-value'
]

# 대상 변수 및 입력 변수
OUTCOME_VAR = 'target'  # 예측 대상 변수
INPUT_VARS = [          # 입력 변수 목록
    'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
    'feature6', 'feature7', 'feature8', 'feature9', 'feature10'
]