import pandas as pd
import re
from typing import List, Optional, Tuple

from .utils import (
    detect_timestamp_columns_general,
    is_datetime_like_general,
    select_primary_timestamp,
)

def is_caseid_candidate(series: pd.Series, threshold: float = 0.98) -> bool:
    """
    Determine if a pandas Series is a candidate for a case id column.

    Criteria:
        - No missing values.
        - High uniqueness ratio.
        - Not purely numeric (including no sign or decimal point).
        - All values' character length < 100.

    Args:
        series (pd.Series): The column to check.
        threshold (float): Minimum unique ratio to qualify as case id.

    Returns:
        bool: True if candidate, False otherwise.
    """
    if series.isna().sum() > 0:
        return False
    n_unique_ratio = series.nunique(dropna=True) / len(series)
    if n_unique_ratio < threshold:
        return False
    vals = series.astype(str)
    # Exclude if all values are purely numeric (with optional sign/decimal)
    is_numeric_like = vals.apply(lambda x: bool(re.fullmatch(r"[\+\-]?\d+(\.\d+)?", x)))
    if is_numeric_like.all():
        return False
    # Exclude if any value is too long
    if (vals.apply(len) >= 100).any():
        return False
    return True

def detect_table_type(df: pd.DataFrame, verbose: bool = False) -> str:
    """
    Detect whether a DataFrame is in 'long' or 'wide' format.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        verbose (bool): If True, print debug info.

    Returns:
        str: 'long' or 'wide' or 'No' (if no timestamp columns).
    """
    timestamp_cols = detect_timestamp_columns_general(df)
    N = len(timestamp_cols)
    if verbose:
        print(f"Detected timestamp columns: {timestamp_cols}")
    if N == 0:
        if verbose:
            print("No timestamp columns detected. Not event log.")
        return 'No'
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
        if all(cnt > 0.7 * total_rows for cnt in non_nan_counts):
            std_ratio = pd.Series(non_nan_counts).std() / (total_rows + 1e-9)
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
    long_df = long_df.drop_duplicates()
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
    Raises:
        ValueError: If no suitable case id column found in wide mode.
    """
    table_type = detect_table_type(df, verbose=verbose)
    if table_type == 'No':
        raise ValueError("No timestamp columns detected. Event log validation FAILED.")
    if table_type == 'wide':
        timestamp_cols = detect_timestamp_columns_general(df)
        if not case_candidates:
            candidates = [
                col for col in df.columns
                if col not in timestamp_cols and is_caseid_candidate(df[col])
            ]
            if candidates:
                case_col = max(candidates, key=lambda c: df[c].nunique())
            else:
                id_like = [col for col in df.columns if 'id' in col.lower()]
                if id_like:
                    case_col = max(id_like, key=lambda c: df[c].nunique() / len(df))
                else:
                    raise ValueError("No suitable case id column found for wide table, cannot convert to long format.")
        else:
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


def melt_case_level_table(
    df: pd.DataFrame,
    timestamp_cols: List[str],
    verbose: bool = False
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Convert a wide-format DataFrame (case-level) to long format by melting all timestamp columns.
    
    This function:
    1. Identifies a case_id column
    2. Melts all timestamp columns into (case_id, activity, timestamp) format
    3. Returns the melted DataFrame and column names
    
    Args:
        df (pd.DataFrame): The wide-format DataFrame.
        timestamp_cols (List[str]): All timestamp column names.
        verbose (bool): If True, print debug info.
    
    Returns:
        Tuple[pd.DataFrame, str, str, str]: (melted_df, case_col, activity_col, timestamp_col)
    
    Raises:
        ValueError: If no suitable case_id column found.
    """
    # Step 1: Identify case_id column
    case_col = None
    
    # Try to find a high-uniqueness non-timestamp column
    for col in df.columns:
        if col not in timestamp_cols and is_caseid_candidate(df[col], threshold=0.98):
            case_col = col
            break
    
    # If not found, try columns with 'id' in the name
    if case_col is None:
        id_like = [col for col in df.columns if 'id' in col.lower() and col not in timestamp_cols]
        if id_like:
            case_col = max(id_like, key=lambda c: df[c].nunique())
    
    # If still not found, use the index
    if case_col is None:
        if verbose:
            print("No suitable case_id column found. Using index as case_id.")
        df_copy = df.copy()
        df_copy['case_id'] = df.index
        case_col = 'case_id'
    
    if verbose:
        print(f"Case-level melt: identified case_id column: {case_col}")
    
    # Step 2: Melt operation
    melted_df = df.melt(
        id_vars=[case_col],
        value_vars=timestamp_cols,
        var_name='activity',
        value_name='timestamp'
    )
    
    # Step 3: Clean melted data
    melted_df = melted_df.dropna(subset=['timestamp'])
    melted_df = melted_df.drop_duplicates()
    
    # Step 4: Rename columns to standardized names
    melted_df = melted_df.rename(columns={
        case_col: 'case_id'
    })
    
    if verbose:
        print(f"Case-level melt completed. Shape: {melted_df.shape}")
        print(f"Melted DataFrame preview:\n{melted_df.head()}")
    
    return melted_df, 'case_id', 'activity', 'timestamp'


def preprocess_and_generate_candidates(
    df: pd.DataFrame,
    verbose: bool = False
) -> List[dict]:
    """
    Preprocess input DataFrame and generate event table candidates based on timestamp column count.
    
    Logic:
    - If <=2 timestamp columns: Generate 1 event-level candidate (requires candidate generation)
    - If >=3 timestamp columns: Generate 2 candidates
        1. event-level: select primary timestamp, requires candidate generation
        2. case-level (melt): melt all timestamps, no candidate generation needed
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        verbose (bool): If True, print debug info.
    
    Returns:
        List[dict]: List of event table candidates. Each candidate is a dict with keys:
            - 'df': DataFrame (the candidate event table)
            - 'timestamp_col': str (primary timestamp column name)
            - 'path_type': str ('event-level', 'event-level-multi', 'case-level-melt')
            - 'requires_candidate_generation': bool
            - 'case_col': str (optional, for case-level-melt)
            - 'activity_col': str (optional, for case-level-melt)
    
    Raises:
        ValueError: If no timestamp columns detected.
    """
    # Step 1: Detect timestamp columns
    timestamp_cols = detect_timestamp_columns_general(df)
    
    if verbose:
        print(f"Detected timestamp columns: {timestamp_cols}")
    
    if len(timestamp_cols) == 0:
        raise ValueError("No timestamp columns detected. Cannot generate event table candidates.")
    
    candidates_list = []
    
    # Step 2: Branch based on timestamp column count
    if len(timestamp_cols) <= 2:
        # Path A: Event-level processing (<=2 timestamp columns)
        if verbose:
            print(f"Path A triggered: {len(timestamp_cols)} timestamp columns (<=2)")
        
        primary_ts, score = select_primary_timestamp(timestamp_cols, df, verbose=verbose)
        
        candidate = {
            'df': df.copy(),
            'timestamp_col': primary_ts,
            'path_type': 'event-level',
            'requires_candidate_generation': True
        }
        candidates_list.append(candidate)
        
        if verbose:
            print(f"Generated 1 event-level candidate")
    
    else:  # len(timestamp_cols) >= 3
        # Path B-1: Event-level processing (>=3 timestamp columns)
        if verbose:
            print(f"Path B-1 triggered: {len(timestamp_cols)} timestamp columns (>=3, event-level)")
        
        primary_ts, score = select_primary_timestamp(timestamp_cols, df, verbose=verbose)
        
        candidate_b1 = {
            'df': df.copy(),
            'timestamp_col': primary_ts,
            'path_type': 'event-level-multi',
            'requires_candidate_generation': True
        }
        candidates_list.append(candidate_b1)
        
        if verbose:
            print(f"Generated event-level-multi candidate with primary timestamp: {primary_ts}")
        
        # Path B-2: Case-level (melt) processing (>=3 timestamp columns)
        if verbose:
            print(f"Path B-2 triggered: {len(timestamp_cols)} timestamp columns (>=3, case-level melt)")
        
        try:
            melted_df, case_col, activity_col, timestamp_col = melt_case_level_table(
                df, timestamp_cols, verbose=verbose
            )
            
            candidate_b2 = {
                'df': melted_df,
                'case_col': case_col,
                'activity_col': activity_col,
                'timestamp_col': timestamp_col,
                'path_type': 'case-level-melt',
                'requires_candidate_generation': False  # KEY: No candidate generation needed
            }
            candidates_list.append(candidate_b2)
            
            if verbose:
                print(f"Generated case-level-melt candidate")
        
        except Exception as e:
            if verbose:
                print(f"Case-level melt failed: {e}. Skipping this candidate.")
    
    if verbose:
        print(f"Total candidates generated: {len(candidates_list)}")
    
    return candidates_list