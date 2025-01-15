import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from typing import Tuple

def ici_error(y_true: np.ndarray, y_pred: np.ndarray, agg: Callable = np.mean, lowess_kwargs: dict = None) -> float:
    if lowess_kwargs is None:
        lowess_kwargs = {}
    lowess_result = sm.nonparametric.lowess(endog=y_true, exog=y_pred, **lowess_kwargs)
    y_pred_sorted = lowess_result[:, 0]
    y_true_smooth = lowess_result[:, 1]
    diff = np.abs(y_true_smooth - y_pred_sorted)
    return agg(diff)

def calculate_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "y_prob must be between 0 and 1"
    log_reg = LogisticRegression(penalty='l2', C=1e10, solver='lbfgs', max_iter=1000)
    log_reg.fit(y_prob.reshape(-1, 1), y_true)
    return log_reg.coef_[0][0], log_reg.intercept_[0]

def calculate_hosmer_lemeshow_p_value(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    data['group'] = pd.qcut(data['y_prob'], q=bins, duplicates='drop')
    obs = data.groupby('group', observed=False)['y_true'].sum()
    exp = data.groupby('group', observed=False)['y_prob'].sum()
    n = data.groupby('group', observed=False).size()
    hl_stat = ((obs - exp)**2 / (exp * (n - exp) / n)).sum()
    return 1 - stats.chi2.cdf(hl_stat, bins - 2)