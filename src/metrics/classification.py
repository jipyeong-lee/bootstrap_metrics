import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def calculate_brier_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError("The shape of predictions and labels must be the same.")
    return np.mean((predictions - labels) ** 2)

def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob > threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    prc, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rec, prc)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    brier = calculate_brier_score(y_prob, y_true)
    return {
        'AUROC': auroc, 'AUPRC': auprc, 'Sensitivity': sensitivity, 'Specificity': specificity,
        'Precision': precision, 'F1': f1, 'Accuracy': accuracy, 'Brier': brier
    }