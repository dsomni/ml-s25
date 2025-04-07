from sklearn.metrics import f1_score, r2_score, recall_score, roc_auc_score


def compute_metrics(predictions, truths):
    """
    ROC AUC SCORE
    """

    assert len(predictions) == len(truths)
    rounded = [round(x) for x in predictions]
    score = roc_auc_score(truths, rounded)
    r2 = r2_score(truths, rounded)
    f1 = f1_score(truths, rounded)
    recall = recall_score(truths, rounded)

    to_return = {
        "lb": round(score, 4),
        "r2": round(r2, 4),
        "f1": round(f1, 4),
        "recall": round(recall, 4),
    }

    return to_return
