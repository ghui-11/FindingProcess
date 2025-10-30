"""
General utility functions for type detection, feature calculation, string similarity, normalization, etc.
"""

import pandas as pd
import numpy as np
import difflib
from dateutil.parser import parse
import warnings
from typing import Optional, Dict, List, Tuple
import random

def entropy(arr):
    """
    Calculate entropy for a numpy array.

    Args:
        arr (np.ndarray): Data array.

    Returns:
        float: Entropy value.
    """
    arr_str = np.array([str(x) for x in arr])
    vals, counts = np.unique(arr_str, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob + 1e-9))

def normalize(series):
    """
    Normalize a pandas Series to [0, 1].

    Args:
        series (pd.Series): Series to normalize.

    Returns:
        pd.Series: Normalized series.
    """
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

def avg_levenshtein(strings):
    """
    Compute average Levenshtein ratio for a set of strings.
    If unique strings > 100, randomly sample 100 strings.

    Args:
        strings (iterable): Input strings.

    Returns:
        float: Average pairwise Levenshtein ratio.
    """
    strings = [str(s) for s in set(strings) if pd.notnull(s)]
    if len(strings) < 2:
        return 1.0
    if len(strings) > 100:
        strings = random.sample(strings, 100)
    total, count = 0, 0
    for i, s1 in enumerate(strings):
        for s2 in strings[i + 1:]:
            total += difflib.SequenceMatcher(None, s1, s2).ratio()
            count += 1
    return total / count if count > 0 else 0

def std_char_count(col_vals):
    return np.std([len(str(v)) for v in col_vals])


def get_column_features_for_classifier(series: pd.Series) -> list:
    """
    Extract normalized feature vector for a column.

    Args:
        series (pd.Series): Input column.

    Returns:
        list: [n_unique_ratio_norm, max_freq_norm, entropy_norm, sim_norm]
    """
    N = len(series)
    n_unique = series.nunique(dropna=True)
    n_unique_ratio = n_unique / N
    vc = series.value_counts(dropna=True)
    max_freq = vc.iloc[0] / N if len(vc) > 0 else 0.0
    ent = entropy(series.dropna().values)
    sim = avg_levenshtein(series.dropna().unique())
    std_length = std_char_count(series.dropna().values)
    return [n_unique_ratio, max_freq, ent, sim, std_length]

def detect_mixed_types(series, sample_size=50):
    """
    Detect if a pandas Series contains mixed types.

    Args:
        series (pd.Series): Input series.
        sample_size (int): Number of samples to check.

    Returns:
        bool: True if mixed types detected, False otherwise.
    """
    samples = series.dropna().astype(str).head(sample_size)
    types = set()
    for v in samples:
        try:
            fv = float(v)
            types.add('int' if fv.is_integer() else 'float')
        except:
            types.add('str')
    return len(types) > 1

def field_screening(df, max_word_threshold=6, attribute_blacklist=None, timestamp_cols=None):
    """
    Screen DataFrame columns for valid candidate fields.

    Args:
        df (pd.DataFrame): DataFrame to screen.
        max_word_threshold (int): Max word count for string columns.
        attribute_blacklist (list): Blacklist of columns or substrings.
        timestamp_cols (list): Timestamp column names to exclude.

    Returns:
        list: List of candidate field names.
    """
    attribute_blacklist = attribute_blacklist or []
    timestamp_cols = timestamp_cols or []
    candidates, N = [], len(df)
    ts_set = set([c.lower() for c in timestamp_cols])
    for col in df.columns:
        col_lower = col.lower()
        if col in attribute_blacklist or col_lower in ts_set:
            continue
        if any(key in col_lower for key in attribute_blacklist):
            continue
        dtype = df[col].dtype
        is_string = dtype == object or pd.api.types.is_string_dtype(dtype)
        is_int = pd.api.types.is_integer_dtype(dtype)
        is_category = isinstance(dtype, pd.CategoricalDtype)
        if not (is_string or is_int or is_category):
            continue
        n_notna = df[col].notna().sum()
        if N == 0 or n_notna / N < 0.8:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 2:
            continue
        if detect_mixed_types(df[col]):
            continue
        if is_string:
            n_long = (df[col].astype(str).apply(lambda x: len(x.split())) > max_word_threshold).sum()
            if n_long / N >= 0.1:
                continue
        candidates.append(col)
    return candidates

def field_scoring(df, candidates, scoring_params: Optional[Dict] = None):
    """
    Score candidate fields as case or activity columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        candidates (list): Candidate field names.
        scoring_params (dict): Custom scoring parameters.

    Returns:
        pd.DataFrame: DataFrame of scores.
        list: Top case fields.
        list: Top activity fields.
    """
    rows, N = [], len(df)
    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        n_unique_ratio = n_unique / N
        vc = df[col].value_counts(dropna=True)
        max_freq = vc.iloc[0] / N if len(vc) > 0 else 0.0
        ent = entropy(df[col].dropna().values)
        sim = avg_levenshtein(df[col].dropna().unique())
        rows.append({
            "field": col,
            "n_unique_ratio": n_unique_ratio,
            "max_freq": max_freq,
            "entropy": ent,
            "sim": sim
        })
    if not rows:
        return pd.DataFrame([]), [], []
    scored = pd.DataFrame(rows)
    scored["n_unique_ratio_norm"] = normalize(scored["n_unique_ratio"])
    scored["max_freq_norm"] = normalize(scored["max_freq"])
    scored["entropy_norm"] = normalize(scored["entropy"])
    scored["sim_norm"] = normalize(scored["sim"])
    # Use scoring_params if provided
    case_weights = {"n_unique_ratio_norm": 0.4, "max_freq_norm": -0.2, "entropy_norm": 0.2, "sim_norm": -0.2}
    act_weights = {"n_unique_ratio_norm": 0.3, "max_freq_norm": 0.2, "entropy_norm": -0.2, "sim_norm": 0.3}
    if scoring_params:
        case_weights = scoring_params.get("case_weights", case_weights)
        act_weights = scoring_params.get("act_weights", act_weights)
    scored["case_score"] = (
        case_weights.get("n_unique_ratio_norm", 0) * scored["n_unique_ratio_norm"] +
        case_weights.get("max_freq_norm", 0) * (1 - scored["max_freq_norm"]) +
        case_weights.get("entropy_norm", 0) * scored["entropy_norm"] +
        case_weights.get("sim_norm", 0) * (1 - scored["sim_norm"])
    )
    scored["act_score"] = (
        act_weights.get("n_unique_ratio_norm", 0) * (1 - abs(scored["n_unique_ratio_norm"] - 0.1)) +
        act_weights.get("max_freq_norm", 0) * scored["max_freq_norm"] +
        act_weights.get("entropy_norm", 0) * (1 - scored["entropy_norm"]) +
        act_weights.get("sim_norm", 0) * scored["sim_norm"]
    )
    case_top = scored.sort_values("case_score", ascending=False).head(2)
    act_top = scored.sort_values("act_score", ascending=False).head(2)
    case_fields = list(case_top["field"])
    act_fields = list(act_top["field"])
    return scored, case_fields, act_fields

def is_datetime_like_general(value):
    """
    Check if a value looks like a datetime.

    Args:
        value: Value to check.

    Returns:
        bool: True if value is datetime-like, False otherwise.
    """
    if pd.isna(value):
        return False
    if not isinstance(value, str):
        value = str(value)
    v = value.strip()
    if v == "" or v.lower() == "nan":
        return False
    if v.isdigit():
        if len(v) == 8:
            try:
                dt = pd.to_datetime(v, format="%Y%m%d", errors='raise')
                return 1970 <= dt.year <= 2100
            except Exception:
                return False
        if len(v) in (10, 13):
            try:
                ts = int(v)
                if len(v) == 10:
                    dt = pd.to_datetime(ts, unit='s')
                else:
                    dt = pd.to_datetime(ts // 1000, unit='s')
                return 1970 <= dt.year <= 2100
            except Exception:
                return False
    if len(v) < 6:
        return False
    date_sep_count = sum([sep in v for sep in ['-', '/', ':', '.', 'T', ' ']])
    if date_sep_count < 2 and not v.isdigit():
        return False
    dt_formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y.%m.%d", "%Y%m%d",
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z", "%y/%m/%d %H.%M", "%d/%m/%y %H.%M", "%y-%m-%d %H.%M",
        "%d-%m-%Y %H.%M"
    ]
    for fmt in dt_formats:
        try:
            _ = pd.to_datetime(v, format=fmt)
            return True
        except Exception:
            continue
    try:
        _ = pd.to_datetime(v, errors='raise')
        return True
    except Exception:
        pass
    try:
        _ = parse(v)
        return True
    except Exception:
        return False

def detect_timestamp_columns_general(df, sample_size=20, threshold=0.5):
    """
    Detect timestamp columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        sample_size (int): Number of samples per column.
        threshold (float): Proportion threshold.

    Returns:
        list: List of detected timestamp columns.
    """
    timestamp_cols = []
    for col in df.columns:
        samples = df[col].dropna().astype(str).head(sample_size)
        if len(samples) == 0:
            continue
        valid_count = sum(is_datetime_like_general(v) for v in samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                converted = pd.to_datetime(samples, errors='raise')
                if converted.notna().mean() >= threshold:
                    timestamp_cols.append(col)
                    continue
            except Exception:
                pass
        if valid_count / len(samples) >= threshold:
            timestamp_cols.append(col)
    return timestamp_cols


def select_primary_timestamp(timestamp_cols: List[str], df: pd.DataFrame, verbose: bool = False) -> Tuple[str, float]:
    """
    Select the primary timestamp column from multiple timestamp columns.
    
    Criteria:
        - Maximum number of unique values (most diverse)
        - Minimum number of NaN values (least missing)
    
    Scoring: score = n_unique - weight * n_missing
    
    Args:
        timestamp_cols (List[str]): List of timestamp column names.
        df (pd.DataFrame): The DataFrame.
        verbose (bool): If True, print debug info.
    
    Returns:
        Tuple[str, float]: (best_timestamp_col, score)
    
    Raises:
        ValueError: If no timestamp columns provided.
    """
    if not timestamp_cols:
        raise ValueError("No timestamp columns provided.")
    
    if len(timestamp_cols) == 1:
        if verbose:
            print(f"Only one timestamp column: {timestamp_cols[0]}")
        return timestamp_cols[0], 0.0
    
    best_col = None
    best_score = -np.inf
    scores = {}
    
    for col in timestamp_cols:
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        total_rows = len(df)
        
        # Scoring: unique values are positive, missing values are negative (weighted)
        weight = 0.5  # Adjust weight if needed
        score = n_unique - weight * n_missing
        scores[col] = {
            'n_unique': n_unique,
            'n_missing': n_missing,
            'score': score
        }
        
        if score > best_score:
            best_score = score
            best_col = col
    
    if verbose:
        print(f"Timestamp column scores: {scores}")
        print(f"Selected primary timestamp: {best_col} (score: {best_score:.2f})")
    
    return best_col, best_score