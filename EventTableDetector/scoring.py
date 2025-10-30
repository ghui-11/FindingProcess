"""
Scoring and candidate generation for mandatory event log columns.
Implements baseline and enhanced baseline (mean distance, logistic LR) scoring.
Supports model-based methods: logistic, tree, rf, gb.
Uses separate binary models for case and activity columns only.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

from .core import ensure_long_format, filter_datetime_columns
from .utils import (
    detect_timestamp_columns_general,
    field_screening,
    field_scoring,
    get_column_features_for_classifier,
)
from .model import (
    load_case_act_classifiers,
    train_case_act_classifiers,
)
from .feature_extraction import get_feature_means_from_train

DEFAULT_BLACK_LIST = [
    "comment", "note", "description", "remark", "frequency"
]

def suggest_mandatory_column_candidates(
    df: pd.DataFrame,
    black_list: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    method: str = "baseline",
    verbose: bool = False,
) -> List[Tuple[str, str, str]]:
    """
    Suggest all possible (timestamp, case, activity) column combinations.

    Args:
        df (pd.DataFrame): Input DataFrame.
        black_list (list or None): Column blacklist.
        params (dict or None): Additional params, including "baseline_mode".
        method (str): 'baseline', 'logistic', 'tree', 'rf', or 'gb'.
        verbose (bool): If True, print detailed info.

    Returns:
        List[Tuple[str, str, str]]: Candidate (timestamp, case, activity) tuples.
    """
    df = df.head(1000)
    black_list = black_list or DEFAULT_BLACK_LIST
    try:
        long_df = ensure_long_format(df, verbose=verbose)
    except ValueError as e:
        if verbose:
            print(f"Error: {e}")
        return []
    timestamp_cols = detect_timestamp_columns_general(long_df)
    candidates = field_screening(long_df, attribute_blacklist=black_list, timestamp_cols=timestamp_cols)
    candidates_no_datetime = filter_datetime_columns(long_df, candidates)
    candidates_screened = candidates_no_datetime
    if len(candidates_screened) < 2:
        if verbose:
            print("Not enough candidate fields after screening. Not a valid event log.")
        return []
    if verbose:
        print(f"Candidate fields after screening: {candidates_screened}")

    train_dir = params["train_dir"] if params and "train_dir" in params else "./Train"

    if method in ("logistic", "tree", "rf", "gb", 'svm'):
        try:
            case_model, act_model, scaler, feature_names = load_case_act_classifiers(method=method)
            if verbose:
                print(f"Loaded {method} case and activity classifiers (manual features) from saved models.")
        except Exception as e:
            if verbose:
                print(f"Could not load saved {method} case/act models (manual features), training new ones... ({e})")
            nrows = params["nrows"] if params and "nrows" in params else 1000
            max_word_threshold = params["max_word_threshold"] if params and "max_word_threshold" in params else 6
            case_model, act_model, scaler, feature_names = train_case_act_classifiers(
                train_dir=train_dir, nrows=nrows, max_word_threshold=max_word_threshold,
                method=method, params=params, verbose=verbose, save_model=True
            )
        features, fields = [], []
        for col in candidates_screened:
            feat_vec = get_column_features_for_classifier(long_df[col]) # 默认返回 5特征
            features.append(feat_vec)
            fields.append(col)
        X = pd.DataFrame(features, columns=feature_names, index=fields)
        X_scale = scaler.transform(X)
        case_probs = case_model.predict_proba(X_scale)[:, 1]
        act_probs = act_model.predict_proba(X_scale)[:, 1]
        case_indices = np.argsort(case_probs)[-2:]
        act_indices = np.argsort(act_probs)[-2:]
        case_fields = [fields[i] for i in case_indices]
        act_fields = [fields[i] for i in act_indices]
        if verbose:
            print(f"{method} manual pipeline predicted case_fields: {case_fields}, act_fields: {act_fields}")

    # --- Baseline logic ---
    else:
        if method == "baseline":
            scoring_params = params if params else None
            scored, case_fields, act_fields = field_scoring(long_df, candidates_screened, scoring_params=scoring_params)
        else:
            print('Given method not recognized.')
            return []
        if verbose:
            print("Top case fields:", case_fields)
            print("Top activity fields:", act_fields)
    # Timestamp selection logic unchanged
    if timestamp_cols:
        nan_counts = [long_df[col].isna().sum() for col in timestamp_cols]
        min_nan = min(nan_counts)
        first_idx = nan_counts.index(min_nan)
        chosen_timestamp_col = timestamp_cols[first_idx]
        timestamp_cols_sorted = [chosen_timestamp_col]
    else:
        timestamp_cols_sorted = []
    combinations = []
    for t in timestamp_cols_sorted:
        for c in case_fields:
            for a in act_fields:
                if c != a and t not in [c, a]:
                    combinations.append((t, c, a))
    combinations = [tpl for tpl in combinations if tpl[1] != tpl[2]]
    if verbose:
        print("All possible (timestamp, case, activity) combinations (case ≠ activity):", combinations)
    return combinations