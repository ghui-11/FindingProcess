"""
Scoring and candidate generation for mandatory event log columns.
Implements baseline and enhanced baseline (mean distance, logistic LR) scoring.
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
    load_column_classifier,
    train_column_classifier,
    train_baseline_lr,
    load_baseline_lr,
)
from .feature_extraction import get_feature_means_from_train

DEFAULT_BLACK_LIST = [
    "comment", "irrelevant", "note", "description", "remark", "frequency"
]

def baseline_mean_distance(df, candidates, train_dir):
    """
    Enhanced baseline: score using mean feature vectors from training set (distance).
    """
    case_mean, act_mean = get_feature_means_from_train(train_dir)
    feats = [get_column_features_for_classifier(df[col]) for col in candidates]
    case_scores = [-np.linalg.norm(f-case_mean) for f in feats]
    act_scores = [-np.linalg.norm(f-act_mean) for f in feats]
    case_fields = [candidates[i] for i in np.argsort(case_scores)[-2:]]
    act_fields = [candidates[i] for i in np.argsort(act_scores)[-2:]]
    scored = pd.DataFrame({
        "field": candidates,
        "case_score": case_scores,
        "act_score": act_scores
    })
    return scored, case_fields, act_fields

def baseline_logistic_lr(df, candidates, train_dir):
    """
    Enhanced baseline: use a trained logistic regression for case/activity scoring.
    """
    lr = load_baseline_lr(train_dir)
    feats = [get_column_features_for_classifier(df[col]) for col in candidates]
    proba = lr.predict_proba(feats)
    case_indices = np.argsort(proba[:, 0])[-2:]
    act_indices = np.argsort(proba[:, 1])[-2:]
    case_fields = [candidates[i] for i in case_indices]
    act_fields = [candidates[i] for i in act_indices]
    scored = pd.DataFrame({
        "field": candidates,
        "case_prob": proba[:, 0],
        "act_prob": proba[:, 1]
    })
    return scored, case_fields, act_fields

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
        method (str): 'baseline', 'logistic', or 'tree'.
        verbose (bool): If True, print detailed info.

    Returns:
        List[Tuple[str, str, str]]: Candidate (timestamp, case, activity) tuples.
    """
    df = df.head(1000)
    black_list = black_list or DEFAULT_BLACK_LIST
    long_df = ensure_long_format(df, verbose=verbose)
    timestamp_cols = detect_timestamp_columns_general(long_df)
    if verbose:
        print(f"Field screening with blacklist: {black_list}")
    candidates = field_screening(long_df, attribute_blacklist=black_list, timestamp_cols=timestamp_cols)
    candidates_no_datetime = filter_datetime_columns(long_df, candidates)
    candidates_screened = field_screening(long_df[candidates_no_datetime], attribute_blacklist=None, timestamp_cols=None)
    if verbose:
        print(f"Candidate fields after screening (non-blacklist, non-timestamp, non-datetime): {candidates_screened}")

    baseline_mode = params["baseline_mode"] if params and "baseline_mode" in params else "default"
    train_dir = params["train_dir"] if params and "train_dir" in params else "./Train"

    if method in ("logistic", "tree"):
        try:
            clf = load_column_classifier(method=method)
            feature_names = ["n_unique_ratio_norm", "max_freq_norm", "entropy_norm", "sim_norm"]
            if verbose:
                print(f"Loaded {method} classifier from saved model.")
        except Exception as e:
            if verbose:
                print(f"Could not load saved {method} model, training a new one... ({e})")
            nrows = params["nrows"] if params and "nrows" in params else 1000
            max_word_threshold = params["max_word_threshold"] if params and "max_word_threshold" in params else 6
            clf, feature_names = train_column_classifier(
                train_dir=train_dir, nrows=nrows, max_word_threshold=max_word_threshold,
                method=method, params=params, verbose=verbose, save_model=True
            )
        features, fields = [], []
        for col in candidates_screened:
            feat_vec = get_column_features_for_classifier(long_df[col])
            features.append(feat_vec)
            fields.append(col)
        X = pd.DataFrame(features, columns=feature_names, index=fields)
        preds = clf.predict(X)
        case_fields = [field for field, pred in zip(fields, preds) if pred == "case"]
        act_fields = [field for field, pred in zip(fields, preds) if pred == "activity"]
        if verbose:
            print(f"{method} classifier predicted case_fields: {case_fields}, act_fields: {act_fields}")
    else:
        if baseline_mode == "mean_distance":
            scored, case_fields, act_fields = baseline_mean_distance(long_df, candidates_screened, train_dir)
            if verbose:
                print("Baseline mean-distance scored fields\n", scored)
        elif baseline_mode == "logistic_lr":
            scored, case_fields, act_fields = baseline_logistic_lr(long_df, candidates_screened, train_dir)
            if verbose:
                print("Baseline logistic-lr scored fields\n", scored)
        else:
            scoring_params = params if params else None
            scored, case_fields, act_fields = field_scoring(long_df, candidates_screened, scoring_params=scoring_params)
            if verbose:
                print("Scored fields:\n", scored)
        if verbose:
            print("Top case fields:", case_fields)
            print("Top activity fields:", act_fields)
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