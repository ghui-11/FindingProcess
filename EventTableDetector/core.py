"""
Core table structure transformation and detection utilities for eventlog processing.
"""

import pandas as pd
from typing import List, Optional

from .utils import (
    detect_timestamp_columns_general,
    is_datetime_like_general,
)

def is_caseid_candidate(series: pd.Series, threshold: float = 0.98) -> bool:
    """
    Determine if a pandas Series is a candidate for a case id column.

    Args:
        series (pd.Series): The column to check.
        threshold (float): Minimum unique ratio to qualify as case id.

    Returns:
        bool: True if candidate, False otherwise.
    """
    if series.isna().sum() > 0:
        return False
    return series.nunique(dropna=True) / len(series) >= threshold

def detect_table_type(df: pd.DataFrame, verbose: bool = False) -> str:
    """
    Detect whether a DataFrame is in 'long' or 'wide' format.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        verbose (bool): If True, print debug info.

    Returns:
        str: 'long' or 'wide'
    """
    timestamp_cols = detect_timestamp_columns_general(df)
    N = len(timestamp_cols)
    if verbose:
        print(f"Detected timestamp columns: {timestamp_cols} (count: {N})")
    if N <= 2:
        if verbose:
            print("Table type judged as long (<=2 timestamp columns).")
        return 'long'
    for col in df.columns:
        if col not in timestamp_cols and is_caseid_candidate(df[col], threshold=0.98):
            if df[col].is_unique:
                if verbose:
                    print(f"Table type judged as wide (column '{col}' is a valid case id).")
                return 'wide'
    if N >= 3:
        non_nan_counts = [df[c].notna().sum() for c in timestamp_cols]
        total_rows = len(df)
        if verbose:
            print(f"Timestamp columns non-NaN counts: {non_nan_counts} (total rows: {total_rows})")
        if all(cnt > 0.7 * total_rows for cnt in non_nan_counts):
            std_ratio = pd.Series(non_nan_counts).std() / (total_rows + 1e-9)
            if verbose:
                print(f"Std ratio of timestamp non-NaN counts: {std_ratio}")
            if std_ratio < 0.3:
                if verbose:
                    print("Table type judged as wide (timestamp columns are evenly filled).")
                return 'wide'
    if verbose:
        print("Table type judged as long.")
    return 'long'

def wide_to_long(df: pd.DataFrame, timestamp_cols: List[str], case_col: str) -> pd.DataFrame:
    """
    Convert a DataFrame from wide format to long format.

    Args:
        df (pd.DataFrame): The wide-format DataFrame.
        timestamp_cols (List[str]): Timestamp columns.
        case_col (str): Case id column.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns 'case', 'activity', 'timestamp'.
    """
    long_df = (
        df.melt(id_vars=[case_col], value_vars=timestamp_cols,
                var_name='activity', value_name='timestamp')
        .dropna(subset=['timestamp'])
        .rename(columns={case_col: 'case'})
    )
    return long_df

def ensure_long_format(df: pd.DataFrame, case_candidates=None, verbose: bool = False) -> pd.DataFrame:
    """
    Ensure the input DataFrame is in long format.

    Args:
        df (pd.DataFrame): The DataFrame to check/convert.
        case_candidates (list or None): Optional candidate case columns.
        verbose (bool): If True, print debug info.

    Returns:
        pd.DataFrame: Long-format DataFrame.
    """
    table_type = detect_table_type(df, verbose=verbose)
    if verbose:
        print(f"ensure_long_format: detected table type is '{table_type}'")
    if table_type == 'wide':
        timestamp_cols = detect_timestamp_columns_general(df)
        if not case_candidates:
            case_candidates = [col for col in df.columns if df[col].nunique() > 1 and col not in timestamp_cols]
        case_col = case_candidates[0]
        if verbose:
            print(f"Converting wide to long using case column: {case_col} and timestamp columns: {timestamp_cols}")
        df_long = wide_to_long(df, timestamp_cols, case_col)
        return df_long
    return df

def filter_datetime_columns(df: pd.DataFrame, columns: list, sample_size: int = 20) -> list:
    """
    Filter out columns detected as datetime-like from a provided list.

    Args:
        df (pd.DataFrame): The DataFrame.
        columns (list): Candidate columns.
        sample_size (int): Number of samples to check.

    Returns:
        list: Columns that are NOT detected as datetime-like.
    """
    filtered_cols = []
    for col in columns:
        series_sample = df[col].dropna().astype(str).head(sample_size)
        if len(series_sample) == 0:
            filtered_cols.append(col)
            continue
        is_datetime_col = all(is_datetime_like_general(v) for v in series_sample)
        if not is_datetime_col:
            filtered_cols.append(col)
    return filtered_cols

def convert_to_event_log(df: pd.DataFrame, case: str, activity: str, timestamp: str, verbose: bool = False) -> pd.DataFrame:
    """
    Convert a DataFrame to a standardized event log with columns 'case', 'activity', 'timestamp'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        case (str): The case column name.
        activity (str): The activity column name.
        timestamp (str): The timestamp column name.
        verbose (bool): If True, print debug info.

    Returns:
        pd.DataFrame: Standardized event log DataFrame.
    """
    if verbose:
        print(f"Converting to event log using columns: case={case}, activity={activity}, timestamp={timestamp}")
    long_df = ensure_long_format(df, [case], verbose=verbose)
    event_log = long_df[[case, activity, timestamp]].rename(
        columns={case: 'case', activity: 'activity', timestamp: 'timestamp'}
    )
    if verbose:
        print("Event log preview:\n", event_log.head())
    return event_log