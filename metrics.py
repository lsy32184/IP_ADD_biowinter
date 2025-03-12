import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(labels, preds):
    """
    주어진 정답(labels)과 예측값(preds)으로 주요 분류 지표(accuracy, precision, recall, f1-score) 계산

    Args:
        labels (numpy.ndarray or list): 정답 레이블
        preds (numpy.ndarray or list): 모델의 예측값 (argmax 적용된 상태)

    Returns: 
        dict: {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}
    """
    labels = np.array(labels)
    preds = np.array(preds)

    #분모가 0이 되는 경우를 처리하는 방법: 분모가 0이면 해당 연산 결과를 0으로 처리
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
