"""
Classification Models (Logistic Regression & Baselines) on Document Representations.
Evaluates both:
1. 15 Issue Areas (Broad Domains)
2. 279 Issue Codes (Fine-Grained Legal Issues)
"""

import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.data_loader import ISSUE_AREA_NAMES


class LegalClassifier:
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = None,
        random_state: int = 42
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.classes_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Logistic Regression classifier.
        """
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        print(f"Trained Logistic Regression on {len(X_train)} samples across {len(self.classes_)} classes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_name: str = "Issue Area"
    ) -> Dict[str, Any]:
        """
        Computes Accuracy, Macro F1, Weighted F1, and detailed per-class metrics.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Precision, recall, F1
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # Per-class metrics
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0, labels=sorted(list(set(y_test) | set(y_pred)))
        )

        all_unique_labels = sorted(list(set(y_test) | set(y_pred)))
        per_class_table = []
        for i, lbl in enumerate(all_unique_labels):
            name = ISSUE_AREA_NAMES.get(lbl, str(lbl)) if isinstance(lbl, (int, np.integer)) else str(lbl)
            per_class_table.append({
                "label_id": lbl,
                "label_name": f"{lbl} - {name}" if isinstance(lbl, (int, np.integer)) and lbl in ISSUE_AREA_NAMES else str(lbl),
                "precision": round(float(p_class[i]), 2),
                "recall": round(float(r_class[i]), 2),
                "f1_score": round(float(f1_class[i]), 2),
                "support": int(support[i])
            })

        print(f"\n[{target_name}] Evaluation Results:")
        print(f"  Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro F1:         {f1_macro:.4f}")
        print(f"  Weighted F1:      {f1_weighted:.4f}")

        return {
            "accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f1_weighted),
            "per_class": per_class_table,
            "y_pred": y_pred,
            "y_test": y_test
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved classifier to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "LegalClassifier":
        return joblib.load(filepath)
