"""
Evaluation, Visualization, and Metrics Reporting.
Reproduces Table 1, Table 2, and Confusion Matrices from Iyer (2020).
"""

import os
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.data_loader import ISSUE_AREA_NAMES


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str
):
    """
    Renders and saves a normalized confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = sorted(list(set(y_true) | set(y_pred)))
    label_names = [f"{lbl}: {ISSUE_AREA_NAMES.get(lbl, str(lbl))[:12]}" for lbl in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    ax.set_title(f"Normalized Confusion Matrix: {model_name} (15 Issue Areas)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Legal Domain", fontsize=12)
    ax.set_ylabel("True Legal Domain", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix for {model_name} to {output_path}")


def generate_table1_comparison(
    results_dict: Dict[str, Dict[str, float]],
    output_path: str = "outputs/table1_comparison.csv"
) -> pd.DataFrame:
    """
    Reproduces Table 1: Model Classification Accuracy comparison across 15-Labels and 279-Labels.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Model": model_name,
            "15-Labels (Issue Area)": f"{metrics.get('acc_15', 0.0):.4f}",
            "279-Labels (Issue Code)": f"{metrics.get('acc_279', 0.0):.4f}",
            "Macro F1 (15-Labels)": f"{metrics.get('f1_macro_15', 0.0):.4f}",
            "Weighted F1 (15-Labels)": f"{metrics.get('f1_weighted_15', 0.0):.4f}"
        })

    df_table1 = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_table1.to_csv(output_path, index=False)
    print(f"\n=== Table 1: Classification Accuracy Comparison ===")
    print(df_table1.to_string(index=False))
    return df_table1


def generate_table2_breakdown(
    per_class_results: List[Dict[str, Any]],
    output_path: str = "outputs/table2_breakdown.csv"
) -> pd.DataFrame:
    """
    Reproduces Table 2: Detailed Classification Results by (manually curated) Category of Legal Opinions.
    """
    rows = []
    for item in per_class_results:
        lbl = item['label_id']
        name = ISSUE_AREA_NAMES.get(lbl, str(lbl))
        rows.append({
            "Label": f"{lbl} - {name}",
            "Precision": f"{item['precision']:.2f}",
            "Recall": f"{item['recall']:.2f}",
            "F-1": f"{item['f1_score']:.2f}",
            "# Docs": item['support']
        })

    df_table2 = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_table2.to_csv(output_path, index=False)
    print(f"\n=== Table 2: Detailed Results by Category (Doc2Vec + LR) ===")
    print(df_table2.to_string(index=False))
    return df_table2
