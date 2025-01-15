import pandas as pd
from src.data.load_data import load_data
from src.evaluation.evaluate import process_multiple_files

# 데이터 로드
train_path = '/home/jplee/narrow_or_general/data/inspire/train_data.csv'
valid_path = '/home/jplee/narrow_or_general/data/inspire/valid_data.csv'
test_path = '/home/jplee/narrow_or_general/data/inspire/test_data.csv'
outcome_var = 'inhosp_death_30day'
input_vars = [
    'age', 'sex', 'emop', 'bmi', 'andur', 
    'preop_hb', 'preop_platelet', 'preop_wbc', 'preop_aptt', 'preop_ptinr', 'preop_glucose',
    'preop_bun', 'preop_albumin', 'preop_ast', 'preop_alt', 'preop_creatinine', 'preop_sodium', 'preop_potassium'
]

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(train_path, valid_path, test_path, outcome_var, input_vars)

# 파일 경로 리스트
file_paths = [
    {
        'name': '2-shot-false-sentence',
        'valid': f'/home/jplee/narrow_or_general/pred/openbiollm/inspire/inhosp_death_30day/BF16/sentence/pred_2shot_false_valid_inspire.pkl', 
        'test': f'/home/jplee/narrow_or_general/pred/openbiollm/inspire/inhosp_death_30day/BF16/sentence/pred_2shot_false_test_inspire.pkl'
    },
    {
        'name': '2-shot-false-json',
        'valid': f'/home/jplee/narrow_or_general/pred/openbiollm/inspire/inhosp_death_30day/BF16/json/pred_2shot_false_valid_inspire.pkl', 
        'test': f'/home/jplee/narrow_or_general/pred/openbiollm/inspire/inhosp_death_30day/BF16/json/pred_2shot_false_test_inspire.pkl'
    },
]

# 계산할 성능 지표 이름들
metric_names = ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Accuracy', 'Brier', 'ICI', 'Calibration Slope', 'Calibration Intercept', 'Unreliability p-value']

# 여러 파일에 대해 성능 지표 계산 및 데이터프레임 생성
results_df = process_multiple_files(file_paths, metric_names, y_valid, y_test)

# 결과 출력
results_df.to_csv('tmp_result_with_averages.csv', index=False)