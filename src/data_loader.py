"""
Data Loader and Preprocessing for US Supreme Court Legal Opinions Dataset.
Corresponds to the dataset described in:
'Classification of Legal Text' - Krithika Iyer
Source: Supreme Court Database (SCDB) / Washington University via textacy.
"""

import gzip
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# SCDB Issue Area Category Names (0 to 14)
ISSUE_AREA_NAMES = {
    0: "None",
    1: "Criminal Procedure",
    2: "Civil Rights",
    3: "First Amendment",
    4: "Due Process",
    5: "Privacy",
    6: "Attorneys",
    7: "Unions",
    8: "Economic Activity",
    9: "Judicial Power",
    10: "Federalism",
    11: "Interstate Relations",
    12: "Federal Taxation",
    13: "Miscellaneous",
    14: "Private Action"
}


def simple_tokenize(text: str, min_len: int = 3, max_len: int = 25) -> List[str]:
    """
    Tokenizes raw legal text: lowercases, removes non-alphanumeric characters,
    and returns word tokens within specified length constraints.
    """
    # Lowercase and extract alphanumeric word tokens
    words = re.findall(r'[a-zA-Z]{3,25}', text.lower())
    return words


def load_supreme_court_data(
    file_path: Optional[str] = None,
    min_text_len: int = 100,
    filter_unknown_area: bool = True
) -> pd.DataFrame:
    """
    Loads US Supreme Court legal decisions from the compressed JSON Lines file.
    
    Returns a pandas DataFrame containing cleaned and filtered records.
    """
    if file_path is None:
        candidates = [
            "supreme-court-py3.json.gz",
            os.path.join("guidelines", "supreme-court-py3.json.gz"),
            os.path.join(os.path.dirname(__file__), "..", "supreme-court-py3.json.gz"),
            os.path.join(os.path.dirname(__file__), "..", "guidelines", "supreme-court-py3.json.gz"),
            os.path.abspath("supreme-court-py3.json.gz"),
            os.path.abspath(os.path.join("guidelines", "supreme-court-py3.json.gz"))
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = os.path.abspath(c)
                break

    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    records = []
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            try:
                record = json.loads(line)
            except Exception:
                continue

            text = record.get("text", "")
            if not text or len(text.strip()) < min_text_len:
                continue

            issue_area = record.get("issue_area")
            # In SCDB, -1 or None indicates unidentified/missing category
            if filter_unknown_area and (issue_area is None or issue_area < 0):
                continue

            case_name = record.get("case_name") or "Unknown Case"
            us_cite = record.get("us_cite_id") or "Unknown Cite"
            issue_code = str(record.get("issue") or "unknown")

            records.append({
                "doc_id": idx,
                "case_name": case_name,
                "us_cite_id": us_cite,
                "decision_date": record.get("decision_date"),
                "issue_area": int(issue_area) if issue_area is not None else -1,
                "issue_area_name": ISSUE_AREA_NAMES.get(issue_area, "Unknown"),
                "issue_code": issue_code,
                "n_maj_votes": record.get("n_maj_votes"),
                "n_min_votes": record.get("n_min_votes"),
                "decision_direction": record.get("decision_direction"),
                "text": text,
                "char_length": len(text),
                "word_count": len(text.split())
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} legal opinions successfully from {file_path}.")
    return df


def get_stratified_train_test_split(
    df: pd.DataFrame,
    target_col: str = "issue_area",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a train/test split. For rare classes with fewer than 2 instances,
    they are retained in the training set to preserve valid stratified splitting.
    """
    # Check class counts
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    
    stratified_df = df[df[target_col].isin(valid_classes)].copy()
    singleton_df = df[~df[target_col].isin(valid_classes)].copy()

    train_df, test_df = train_test_split(
        stratified_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratified_df[target_col]
    )

    if len(singleton_df) > 0:
        train_df = pd.concat([train_df, singleton_df], ignore_index=True)

    print(f"Split data into Train: {len(train_df)} samples, Test: {len(test_df)} samples.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
