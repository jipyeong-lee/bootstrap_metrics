# utils.py
import pandas as pd
import numpy as np
import pickle
import os
import re

def load_data(train_path, valid_path, test_path, outcome_var, input_vars):
    """
    데이터를 로드하는 함수
    """
    train_set = pd.read_csv(train_path)
    valid_set = pd.read_csv(valid_path)
    test_set = pd.read_csv(test_path)

    y_train = train_set.loc[:, outcome_var]
    x_train = train_set.loc[:, input_vars].astype(float)
    y_valid = valid_set.loc[:, outcome_var]
    x_valid = valid_set.loc[:, input_vars].astype(float)
    y_test = test_set.loc[:, outcome_var]
    x_test = test_set.loc[:, input_vars].astype(float)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_file(file_path):
    """
    파일 형식에 관계없이 파일을 로드하는 함수
    - .pkl 파일: pickle로 로드
    - .npy 파일: numpy로 로드
    - 그 외: ValueError 발생
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pkl':
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    elif file_extension == '.npy':
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .pkl and .npy are supported.")

def load_predictions(valid_path, test_path):
    """
    예측 파일을 로드하는 함수
    """
    y_prob_medi = load_file(valid_path)
    y_prob_medi_test = load_file(test_path)
    return np.array(y_prob_medi), np.array(y_prob_medi_test)

def extract_mean(value):
    """
    문자열에서 숫자 값을 추출하는 함수 (예: "0.123 (0.111-0.135)" -> 0.123)
    """
    match = re.search(r'(\d+\.\d*|\d*\.\d+)', str(value))
    if match:
        return float(match.group(1))
    return np.nan

def add_confidence_intervals_to_df(metric_name, bootstrap_samples, df):
    """
    부트스트랩 결과를 데이터프레임에 추가하는 함수
    """
    mean = np.mean(bootstrap_samples)
    confidence_interval = np.percentile(bootstrap_samples, [2.5, 97.5])
    df[metric_name] = f'{mean:.3f} ({confidence_interval[0]:.3f}-{confidence_interval[1]:.3f})'

def process_multiple_files(file_paths, metric_names):
    """
    여러 파일에 대해 성능 지표를 계산하고 데이터프레임으로 정리하는 함수
    """
    results_df = pd.DataFrame(columns=['File'] + metric_names)

    for file_path in file_paths:
        # 예측 확률 읽기
        y_prob_medi, y_prob_medi_test = load_predictions(file_path['valid'], file_path['test'])

        # 최적 임계값 찾기
        optimal_threshold = find_optimal_threshold(y_valid, y_prob_medi)
        print(f'Optimal Threshold for {file_path["name"]}: {round(optimal_threshold, 4)}')

        # 부트스트랩 성능 지표 계산
        bootstrap_results = bootstrap_metrics(y_test, y_prob_medi_test, optimal_threshold)

        # 각 성능 지표에 대해 신뢰구간을 계산하고 데이터프레임에 추가
        row_data = {'File': file_path['name']}
        for metric_name, metric_values in zip(metric_names, bootstrap_results):
            add_confidence_intervals_to_df(metric_name, metric_values, row_data)

        row_df = pd.DataFrame([row_data])
        results_df = pd.concat([results_df, row_df], ignore_index=True)

    # 각 성능 지표의 평균 계산
    averages = results_df[metric_names].apply(lambda x: x.str.extract(r'(\d+\.\d+)').astype(float).mean()).round(3)

    # average 행 추가 (평균 값을 가진 새로운 행 생성)
    average_row = pd.DataFrame([['average'] + averages.squeeze().tolist()], columns=['File'] + metric_names)
    results_df = pd.concat([results_df, average_row], ignore_index=True)

    return results_df

def calculate_format_averages(results_df, metric_names):
    """
    파일 이름 형식(sentence, json 등)별 평균 성능을 계산하고 결과를 데이터프레임에 추가하는 함수
    """
    formats = ['sentence', 'json']  # 파일 이름 형식
    format_averages = {}

    for fmt in formats:
        # NaN 값 제거 및 해당 포맷에 맞는 행만 필터링
        format_df = results_df.dropna(subset=['File'])
        format_df = format_df[format_df['File'].str.contains(fmt, na=False)]
        
        # 각 성능 지표에서 평균 숫자만 추출하여 계산
        averages = format_df[metric_names].applymap(extract_mean).mean().round(3)
        format_averages[fmt] = averages

    # 포맷 별 평균을 가진 새로운 행 추가
    for fmt, averages in format_averages.items():
        format_row = pd.DataFrame([{'File': f'{fmt}_average', **averages}], columns=['File'] + metric_names)
        results_df = pd.concat([results_df, format_row], ignore_index=True)

    return results_df