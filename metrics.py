import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(labels, preds):
    """
    Computes classification metrics (accuracy, precision, recall, and F1-score) based on the given ground truth labels and predictions.
    
    Args:
        labels (numpy.ndarray or list): Ground truth labels.
        preds (numpy.ndarray or list): Predicted labels (assumed to be already argmaxed if applicable).
    
    Returns:
        dict: Dictionary containing the computed metrics: "accuracy", "precision", "recall", and "f1_score".
    """
    labels = np.array(labels)
    preds = np.array(preds)

    # Compute metrics with handling for division-by-zero cases
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0) 

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
