from sklearn.metrics import accuracy_score, f1_score
from .du_lieu import IDS

def diem(y, pred):
    scores = f1_score(y, pred, labels=IDS, average=None, zero_division=0)
    return {
        "macro_f1": float(scores.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_tieu_cuc": float(scores[0]),
        "f1_trung_tinh": float(scores[1]),
        "f1_tich_cuc": float(scores[2]),
    }
