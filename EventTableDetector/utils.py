import pandas as pd
import numpy as np
import difflib
import random
import re
import warnings
from typing import List, Optional, Dict, Tuple
from dateutil.parser import parse  # For robust datetime string parsing

def entropy(arr):
    """Calculate entropy for an array-like."""
    arr_str = np.array([str(x) for x in arr])
    vals, counts = np.unique(arr_str, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob + 1e-9))

def normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

def avg_levenshtein(strings) -> float:
    strings = [str(s) for s in set(strings) if pd.notnull(s)]
    if len(strings) < 2: return 1.0
    if len(strings) > 100: strings = random.sample(strings, 100)
    total, count = 0.0, 0
    for i, s1 in enumerate(strings):
        for s2 in strings[i + 1:]:
            total += difflib.SequenceMatcher(None, s1, s2).ratio()
            count += 1
    return total / count if count > 0 else 0.0

def std_char_count(col_vals) -> float:
    return np.std([len(str(v)) for v in col_vals])

def get_column_features_for_classifier(series: pd.Series) -> List[float]:
    N = len(series)
    n_unique = series.nunique(dropna=True)
    n_unique_ratio = n_unique / N if N > 0 else 0.0
    vc = series.value_counts(dropna=True)
    max_freq = vc.iloc[0] / N if len(vc) > 0 and N > 0 else 0.0
    ent = entropy(series.dropna().values) if N > 0 else 0.0
    sim = avg_levenshtein(series.dropna().unique()) if N > 0 else 0.0
    std_length = std_char_count(series.dropna().values) if N > 0 else 0.0
    return [n_unique_ratio, max_freq, ent, sim, std_length]

def detect_mixed_types(series: pd.Series, sample_size: int = 50) -> bool:
    samples = series.dropna().astype(str).head(sample_size)
    types = set()
    for v in samples:
        try: fv = float(v); types.add('int' if fv.is_integer() else 'float')
        except Exception: types.add('str')
    return len(types) > 1

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen = set()
    cols = []
    for col in df.columns:
        values = tuple(df[col].fillna("-9999").astype(str).values)
        if values not in seen:
            seen.add(values)
            cols.append(col)
    return df[cols]

def has_timedelta_format(val: str) -> bool:
    v = str(val)
    return (re.match(r'^-?\d+\s+days?\s+\d{2}:\d{2}:\d{2}', v)
            or re.match(r'^\d{2}:\d{2}:\d{2}$', v)
            or re.match(r'^-?\d+\s+days?\s+\d{2}:\d{2}:\d{2}\.\d+', v))

def is_duration_column_by_keywords(series: pd.Series, sample_size: int = 50) -> bool:
    n = 0
    samples = series.dropna().astype(str).head(sample_size)
    for val in samples:
        if has_timedelta_format(val) or any(kw in val.lower() for kw in ('day','hour','minute','second','week','month','timedelta')):
            n += 1
    return n / max(1, len(samples)) > 0.3


def is_datetime_like_general(val: str, col_name: str = "", require_keyword: bool = False) -> bool:
    """
    Primary datetime-like checker, strict with unix timestamps:
    - Only accept pure numbers if col_name includes 'unix/time/timestamp/epoch'
    - Only accept unix timestamps (10/13 digits) if year between 2000~2050
    """
    if pd.isna(val): return False
    v = str(val).strip()
    if v == "" or v.lower() == "nan": return False
    # --- UNIX timestamp strict handling ---
    if v.isdigit() and len(v) in (10, 13):
        # Only accept numeric unix timestamps if column name contains clue keyword!
        if not any(kw in col_name.lower() for kw in ['unix', 'time', 'timestamp', 'epoch']):
            return False
        try:
            ts = int(v)
            if len(v) == 10:
                dt = pd.to_datetime(ts, unit='s')
            else:
                dt = pd.to_datetime(ts // 1000, unit='s')
            # Only accept years between 2000 and 2050 inclusive
            return 2000 <= dt.year <= 2050
        except Exception:
            return False
    if len(v) < 6:
        return False

    # ===== CHANGED: More lenient separator check =====
    date_sep_count = sum([sep in v for sep in ['-', '/', ':', '.', 'T', ' ']])
    if date_sep_count < 1 and not v.isdigit():  # CHANGED: < 1 instead of < 2
        return False

    dt_formats = [
        # Date only formats (YYYY first)
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S%z",

        # Date + Time formats (YYYY first)
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S+00:00",

        # Ambiguous date formats (try DD/MM first, then MM/DD)
        "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%m-%d-%Y",
        "%d.%m.%Y", "%m.%d.%Y",

        # With time
        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S",

        # 2-digit year formats
        "%d/%m/%y", "%m/%d/%y",
        "%d-%m-%y", "%m-%d-%y",
        "%d.%m.%y", "%m.%d.%y",
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

def detect_true_event_timestamp_columns(df: pd.DataFrame, sample_size=30, verbose=False) -> list:
    """
    Detect all time-format columns, filtering out duration, group formats for >2 cols.
    Uses is_datetime_like_general for robust detection, and unix logic for numeric columns.
    """
    ts_candidates = []
    for col in df.columns:
        ser = df[col]
        # Step 1: Exclude duration columns
        if is_duration_column_by_keywords(ser, sample_size=sample_size):
            if verbose: print(f"[ts_detect] Skip duration column: {col}")
            continue
        ts_candidates.append(col)
    # Step2: Format string time detection (and robust numeric unix detection)
    result = []
    for col in ts_candidates:
        ser = df[col]
        samples = ser.dropna().astype(str).head(sample_size)
        # Accept column if majority samples match is_datetime_like_general with correct col_name passed
        valid_count = sum(is_datetime_like_general(s, col) for s in samples)
        if valid_count / max(1, len(samples)) >= 0.6:
            result.append(col)
        if verbose: print(f"[ts_detect] {col}: {valid_count}/{len(samples)} detected")
    # Step3: Format grouping if >=3 cols
    if len(result) >= 3:
        format_signatures = {}
        for col in result:
            ser = df[col]
            vals = ser.dropna().astype(str).head(sample_size)
            signs = (len(vals.iloc[0]), sum('-' in v for v in vals), sum(' ' in v for v in vals))
            format_signatures[col] = signs
        groups = {}
        for col, sig in format_signatures.items():
            groups.setdefault(sig, []).append(col)
        largest_group = max(groups.values(), key=len)
        if verbose: print(f"[ts_detect] Groups: {groups}")
        result = largest_group
    return result

def filter_datetime_columns(df: pd.DataFrame, columns: list, timestamp_cols: list = None, sample_size: int = 20) -> list:
    """
    Filter out columns detected as datetime-like from a provided list.
    - Skips numeric columns
    - Skips already-detected timestamp columns
    - Only filters string columns that look like timestamps
    Returns a list of columns NOT likely to be datetime.
    """
    filtered_cols = []
    timestamp_cols = timestamp_cols or []
    ts_set = set([c.lower() for c in timestamp_cols])
    for col in columns:
        col_lower = col.lower()
        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            filtered_cols.append(col)
            continue
        # Skip already-detected timestamp columns
        if col_lower in ts_set or col in timestamp_cols:
            continue
        series_sample = df[col].dropna().astype(str).head(sample_size)
        if len(series_sample) == 0:
            filtered_cols.append(col)
            continue
        # Only filter out string columns that are datetime-like
        is_datetime_col = all(is_datetime_like_general(v, col, require_keyword=False) for v in series_sample)
        if not is_datetime_col:
            filtered_cols.append(col)
    return filtered_cols

def field_screening(
    df: pd.DataFrame,
    attribute_blacklist: Optional[List[str]] = None,
    timestamp_cols: Optional[List[str]] = None,
    min_unique: int = 3,
    max_word_threshold: int = 6,
    max_str_len: int = 100,
    remove_duplicate: bool = True
) -> List[str]:
    """
    Candidate column screening for case/activity:
    - removes duplicate columns (by value)
    - excludes timestamps/duration columns (timestamp_cols from detect_true_event_timestamp_columns)
    - unique values < min_unique are skipped
    - numeric columns (all number) are skipped unless id-like
    - string columns with long text are skipped
    - columns with mixed types are skipped
    """
    attribute_blacklist = attribute_blacklist or []
    timestamp_cols = timestamp_cols or []
    filtered_df = remove_duplicate_columns(df) if remove_duplicate else df
    candidates, N = [], len(filtered_df)
    ts_set = set([c.lower() for c in timestamp_cols])
    for col in filtered_df.columns:
        col_lower = col.lower()
        if col in attribute_blacklist or col_lower in ts_set:
            continue
        if any(key in col_lower for key in attribute_blacklist):
            continue
        if is_duration_column_by_keywords(filtered_df[col], sample_size=30): continue
        dtype = filtered_df[col].dtype
        is_string = dtype == object or pd.api.types.is_string_dtype(dtype)
        is_int = pd.api.types.is_integer_dtype(dtype)
        is_float = pd.api.types.is_float_dtype(dtype)
        is_category = isinstance(dtype, pd.CategoricalDtype)
        if not (is_string or is_int or is_float or is_category):
            continue
        n_notna = filtered_df[col].notna().sum()
        if N == 0 or n_notna / N < 0.8:
            continue
        n_unique = filtered_df[col].nunique(dropna=True)
        if n_unique < min_unique:
            continue
        if detect_mixed_types(filtered_df[col]):
            continue
        if is_string:
            n_long = (filtered_df[col].astype(str).apply(lambda x: len(x.split())) > max_word_threshold).sum()
            if n_long / N >= 0.1: continue
            if filtered_df[col].astype(str).apply(len).max() >= max_str_len: continue
        candidates.append(col)
    return candidates

def field_scoring(df: pd.DataFrame, candidates: List[str], scoring_params: Optional[Dict] = None):
    """
    Score candidate fields and return top case/activity fields.
    """
    rows, N = [], len(df)
    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        n_unique_ratio = n_unique / N if N > 0 else 0
        vc = df[col].value_counts(dropna=True)
        max_freq = vc.iloc[0] / N if len(vc) > 0 and N > 0 else 0.0
        ent = entropy(df[col].dropna().values) if N > 0 else 0.0
        sim = avg_levenshtein(df[col].dropna().unique()) if N > 0 else 0.0
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