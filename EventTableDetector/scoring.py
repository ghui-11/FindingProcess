"""
Scoring and candidate generation for mandatory event log columns.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

from .utils import (
    detect_true_event_timestamp_columns,
    field_screening,
    field_scoring,
    get_column_features_for_classifier,
    filter_datetime_columns
)
from .model import (
    load_case_act_classifiers,
    train_case_act_classifiers,
)

DEFAULT_BLACK_LIST = [
    "comment", "note", "description", "remark", "frequency"
]

def suggest_mandatory_column_candidates(
        df: pd.DataFrame,
        black_list: Optional[List[str]] = None,
        params: Optional[Dict] = None,
        method: str = "baseline",
        verbose: bool = True,
        case_k: int = 2,
        act_k: int = 2,
) -> List[Tuple[str, str, str]]:
    """
    Suggest (timestamp, case, activity) column combinations.

    Uses *detect_true_event_timestamp_columns* for strict timestamp detection.
    Screens candidate fields strictly (excludes timestamp/duration columns).
    """

    df_sample = df.head(1000)
    black_list = black_list or DEFAULT_BLACK_LIST

    # 1. Find timestamp columns with modern primary detection
    timestamp_cols = detect_true_event_timestamp_columns(df_sample, verbose=verbose)
    if verbose:
        print(f"Detected timestamp columns: {timestamp_cols}")

    if len(timestamp_cols) == 0:
        if verbose:
            print("No timestamp columns detected")
        return []

    # 2. Field screening, exclude true timestamps and duration columns
    candidates = field_screening(
        df_sample,
        attribute_blacklist=black_list,
        timestamp_cols=timestamp_cols,
    )
    # Remove string columns that look like datetime (extra safety)
    candidates_screened = filter_datetime_columns(df_sample, candidates, timestamp_cols=timestamp_cols)

    if len(candidates_screened) < 2:
        if verbose:
            print("Not enough candidate fields after screening")
        return []

    if verbose:
        print(f"Candidate fields (original columns): {candidates_screened}")

    train_dir = params.get("train_dir", "./Train") if params else "./Train"

    # 3. Case/activity classifier or scoring
    if method in ("logistic", "tree", "rf", "gb", "svm"):
        try:
            case_model, act_model, scaler, feature_names = load_case_act_classifiers(method=method)
            if verbose:
                print(f"Loaded {method} classifiers")
        except Exception as e:
            if verbose:
                print(f"Could not load {method} classifiers, training new ones: {e}")
            nrows = params.get("nrows", 1000) if params else 1000
            max_word_threshold = params.get("max_word_threshold", 6) if params else 6
            case_model, act_model, scaler, feature_names = train_case_act_classifiers(
                train_dir=train_dir, nrows=nrows, max_word_threshold=max_word_threshold,
                method=method, params=params, verbose=verbose, save_model=True
            )

        features, fields = [], []
        for col in candidates_screened:
            feat_vec = get_column_features_for_classifier(df_sample[col])
            features.append(feat_vec)
            fields.append(col)

        X = pd.DataFrame(features, columns=feature_names, index=fields)
        X_scale = scaler.transform(X)
        case_probs = case_model.predict_proba(X_scale)[:, 1]
        act_probs = act_model.predict_proba(X_scale)[:, 1]
        case_indices = np.argsort(case_probs)[-case_k:]
        act_indices = np.argsort(act_probs)[-act_k:]
        case_fields = [fields[i] for i in case_indices]
        act_fields = [fields[i] for i in act_indices]

        if verbose:
            print(f"Predicted case fields (top-{case_k}): {case_fields}")
            print(f"Predicted activity fields (top-{act_k}): {act_fields}")

    else:
        if method == "baseline":
            scoring_params = params if params else None
            scored, case_fields, act_fields = field_scoring(df_sample, candidates_screened,
                                                            scoring_params=scoring_params)
        else:
            if verbose:
                print(f"Unknown method: {method}")
            return []

        if verbose:
            print(f"Case fields: {case_fields}")
            print(f"Activity fields: {act_fields}")

    # 4. Primary timestamp selection (keep API consistent with core.py)
    # If 3+, pick by logic, else pick best by missing
    if len(timestamp_cols) >= 3:
        primary_ts = timestamp_cols[0]
    elif timestamp_cols:
        nan_counts = [df_sample[col].isna().sum() for col in timestamp_cols]
        min_nan = min(nan_counts)
        first_idx = nan_counts.index(min_nan)
        primary_ts = timestamp_cols[first_idx]
    else:
        primary_ts = None

    timestamp_cols_sorted = [primary_ts] if primary_ts else []

    # 5. Assemble combinations output
    combinations = []
    for t in timestamp_cols_sorted:
        for c in case_fields:
            for a in act_fields:
                if c != a and t not in [c, a]:
                    combinations.append((t, c, a))

    combinations = [tpl for tpl in combinations if tpl[1] != tpl[2]]

    if verbose:
        print(f"Generated {len(combinations)} combinations")
        for comb in combinations:
            print(f"  Timestamp: {comb[0]}, Case: {comb[1]}, Activity: {comb[2]}")

    return combinations