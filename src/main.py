import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from config import *
from utils import load_data, load_predictions, process_multiple_files, calculate_format_averages
from metrics import *

# 데이터 로드
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(TRAIN_DATA_PATH, VALID_DATA_PATH, TEST_DATA_PATH, OUTCOME_VAR, INPUT_VARS)

# 파일 경로 리스트
file_paths = [
    {
        'name': 'model1-sentence',
        'valid': os.path.join(PRED_DIR, 'pred_valid_sentence.pkl'), 
        'test': os.path.join(PRED_DIR, 'pred_test_sentence.pkl')
    },
    {
        'name': 'model1-json',
        'valid': os.path.join(PRED_DIR, 'pred_valid_json.npy'), 
        'test': os.path.join(PRED_DIR, 'pred_test_json.npy')
    },
    {
        'name': 'model2-sentence',
        'valid': os.path.join(PRED_DIR, 'pred_valid_sentence_v2.pkl'), 
        'test': os.path.join(PRED_DIR, 'pred_test_sentence_v2.pkl')
    },
    {
        'name': 'model2-json',
        'valid': os.path.join(PRED_DIR, 'pred_valid_json_v2.npy'), 
        'test': os.path.join(PRED_DIR, 'pred_test_json_v2.npy')
    },
]

# 여러 파일에 대해 성능 지표 계산 및 데이터프레임 생성
results_df = process_multiple_files(file_paths, METRIC_NAMES)
results_df = calculate_format_averages(results_df, METRIC_NAMES)

# 결과 출력
results_df.to_csv('results/results.csv', index=False)
print("Results saved to 'results/results.csv'")