import numpy as np
import pandas as pd
from typing import Dict, List

def add_confidence_intervals_to_df(metric_name: str, bootstrap_samples: np.ndarray, df: pd.DataFrame) -> None:
    mean = np.mean(bootstrap_samples)
    confidence_interval = np.percentile(bootstrap_samples, [2.5, 97.5])
    df[metric_name] = f'{mean:.3f} ({confidence_interval[0]:.3f}-{confidence_interval[1]:.3f})'

def extract_mean(value: str) -> float:
    match = re.search(r'(\d+\.\d*|\d*\.\d+)', str(value))
    if match:
        return float(match.group(1))
    return np.nan