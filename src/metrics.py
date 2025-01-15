import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from typing import Callable, Tuple

def calculate_brier_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier 점수를 계산하는 함수
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError("The shape of predictions and labels must be the same.")
    brier_score = np.mean((predictions - labels) ** 2)
    return brier_score

def ici_error(y_true: np.ndarray, y_pred: np.ndarray, agg: Callable = np.mean, lowess_kwargs: dict = None) -> float:
    """
    Integrated Calibration Index (ICI)를 계산하는 함수
    """
    if lowess_kwargs is None:
        lowess_kwargs = {}
    lowess_result = sm.nonparametric.lowess(endog=y_true, exog=y_pred, **lowess_kwargs)
    y_pred_sorted = lowess_result[:, 0]
    y_true_smooth = lowess_result[:, 1]
    diff = np.abs(y_true_smooth - y_pred_sorted)
    score = agg(diff)
    return score

def calculate_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calibration slope와 intercept를 계산하는 함수
    """
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "y_prob must be between 0 and 1"
    log_reg = LogisticRegression(penalty='l2', C=1e10, solver='lbfgs', max_iter=1000)
    log_reg.fit(y_prob.reshape(-1, 1), y_true)
    slope = log_reg.coef_[0][0]
    intercept = log_reg.intercept_[0]
    return slope, intercept

def calculate_hosmer_lemeshow_p_value(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """
    Hosmer-Lemeshow 검정을 수행하는 함수
    """
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    data['group'] = pd.qcut(data['y_prob'], q=bins, duplicates='drop')
    obs = data.groupby('group', observed=False)['y_true'].sum()
    exp = data.groupby('group', observed=False)['y_prob'].sum()
    n = data.groupby('group', observed=False).size()
    hl_stat = ((obs - exp)**2 / (exp * (n - exp) / n)).sum()
    p_value = 1 - stats.chi2.cdf(hl_stat, bins - 2)
    return p_value

def calculate_performance_metrics(y_valid, y_prob_medi):
    """
    AUROC 및 AUPRC를 계산하는 함수
    """
    auroc = roc_auc_score(y_valid, y_prob_medi)
    prc, rec, _ = precision_recall_curve(y_valid, y_prob_medi)
    auprc = auc(rec, prc)
    return auroc, auprc

def find_optimal_threshold(y_valid, y_prob_medi):
    """
    최적 임계값을 찾는 함수
    """
    fpr, tpr, thresholds = roc_curve(y_valid, y_prob_medi)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def calculate_classification_metrics(y_true, y_pred):
    """
    분류 성능 지표를 계산하는 함수
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return sensitivity, specificity, precision, f1, accuracy