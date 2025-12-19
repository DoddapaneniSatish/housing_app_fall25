from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


@dataclass
class EvalResult:
    f1: float
    report: Dict[str, Any]
    confusion_matrix: list


def evaluate_binary(y_true, y_pred) -> EvalResult:
    f1 = float(f1_score(y_true, y_pred))
    rep = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return EvalResult(f1=f1, report=rep, confusion_matrix=cm)
