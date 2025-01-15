import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Dict, List
from src.metrics.classification import calculate_metrics
from src.metrics.calibration import ici_error, calculate_calibration_slope_intercept, calculate_hosmer_lemeshow_p_value
from src.metrics.utils import add_confidence_intervals_to_df

def bootstrap_metrics(y_test: pd.Series, y_prob: np.ndarray, threshold: float, n_bootstraps: int = 2000, subsample_size: int = None, base_seed: int = 42) -> Dict[str, np.ndarray]:
    if subsample_size is None:
        subsample_size = len(y_test)
    
    def single_bootstrap(seed: int) -> List[float]:
        np.random.seed(seed)
        indices = np.random.choice(len(y_test), size=subsample_size, replace=True)
        y_test_sample = y_test.iloc[indices]
        y_prob_sample = y_prob[indices]
        metrics = calculate_metrics(y_test_sample, y_prob_sample, threshold)
        metrics['ICI'] = ici_error(y_test_sample, y_prob_sample)
        metrics['Calibration Slope'], metrics['Calibration Intercept'] = calculate_calibration_slope_intercept(y_test_sample, y_prob_sample)
        metrics['Unreliability p-value'] = calculate_hosmer_lemeshow_p_value(y_test_sample, y_prob_sample)
        return list(metrics.values())
    
    seeds = np.random.RandomState(base_seed).randint(0, 100000, size=n_bootstraps)
    results = Parallel(n_jobs=-1)(delayed(single_bootstrap)(seed) for seed in seeds)
    return {metric: np.array([result[i] for result in results]) for i, metric in enumerate(calculate_metrics(y_test, y_prob, threshold).keys())}

def process_multiple_files(file_paths: List[Dict[str, str]], metric_names: List[str], y_valid: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    results_df = pd.DataFrame(columns=['File'] + metric_names)
    
    for file_path in file_paths:
        y_prob_medi, y_prob_medi_test = load_predictions(file_path['valid'], file_path['test'])
        optimal_threshold = find_optimal_threshold(y_valid, y_prob_medi)
        print(f'Optimal Threshold for {file_path["name"]}: {round(optimal_threshold, 4)}')
        
        bootstrap_results = bootstrap_metrics(y_test, y_prob_medi_test, optimal_threshold)
        row_data = {'File': file_path['name']}
        for metric_name, metric_values in bootstrap_results.items():
            add_confidence_intervals_to_df(metric_name, metric_values, row_data)
        
        row_df = pd.DataFrame([row_data])
        results_df = pd.concat([results_df, row_df], ignore_index=True)
    
    averages = results_df[metric_names].apply(lambda x: x.str.extract(r'(\d+\.\d+)').astype(float).mean()).round(3)
    average_row = pd.DataFrame([['average'] + averages.squeeze().tolist()], columns=['File'] + metric_names)
    results_df = pd.concat([results_df, average_row], ignore_index=True)
    return results_df