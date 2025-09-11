"""
Feature extraction routines for columns and event logs.
Handles extraction for both training and inference.
"""

import os
import pandas as pd
import numpy as np

from .utils import (
    field_screening,
    is_datetime_like_general,
    get_column_features_for_classifier,
)

def extract_train_features(
        train_dir=None,
        output_name='train_features.csv',
        display_result=True,
        max_rows=None
):
    """
    Extract global event log statistical features from each CSV in the training directory.

    Args:
        train_dir (str, optional): Path to directory with training CSV files.
        output_name (str): Output CSV file name.
        display_result (bool): Whether to print the extracted DataFrame.
        max_rows (int, optional): If set, use only this many rows per file.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    if train_dir is None:
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train'))
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix'))
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]

    feature_list = []

    for file in files:
        filepath = os.path.join(train_dir, file)
        df = pd.read_csv(filepath, nrows=max_rows)
        case_candidates = ['case:concept:name', 'case_id', 'Case_ID']
        act_candidates = ['concept:name', 'activity']
        time_candidates = ['time:timestamp']

        def find_column(columns, candidates):
            for c in candidates:
                if c in columns:
                    return c
            return None

        case_col = find_column(df.columns, case_candidates)
        act_col = find_column(df.columns, act_candidates)
        time_col = find_column(df.columns, time_candidates)
        if not case_col or not act_col:
            print(f"File {file} missing required columns. Found columns: {list(df.columns)}")
            continue

        df = df[[case_col, act_col] + ([time_col] if time_col else [])]

        traces = df.groupby(case_col)[act_col].apply(list)
        valid_traces = traces[traces.apply(lambda x: len(x) > 1 and len(set(x)) > 1)]
        valid_trace_percentage = len(valid_traces) / len(traces) if len(traces) > 0 else 0

        valid_cases = valid_traces.index
        df_filtered = df[df[case_col].isin(valid_cases)]

        total_event_count = len(df_filtered)
        unique_case_count = df_filtered[case_col].nunique()
        unique_activity_count = df_filtered[act_col].nunique()
        avg_events_per_case = df_filtered.groupby(case_col).size().mean() if unique_case_count > 0 else 0
        variants = df_filtered.groupby(case_col)[act_col].apply(tuple)
        variant_count = variants.nunique() if unique_case_count > 0 else 0
        avg_unique_acts_per_trace = variants.apply(lambda x: len(set(x))).mean() if unique_case_count > 0 else 0

        feature_list.append({
            'file': file,
            'total_event_count': total_event_count,
            'unique_case_count': unique_case_count,
            'valid_trace_percentage': valid_trace_percentage,
            'unique_activity_count': unique_activity_count,
            'avg_events_per_case': avg_events_per_case,
            'variant_count': variant_count,
            'avg_unique_acts_per_trace': avg_unique_acts_per_trace,
        })

    features_df = pd.DataFrame(feature_list)
    output_path = os.path.join(output_folder, output_name)
    features_df.to_csv(output_path, index=False)
    if display_result:
        print(features_df)
    return features_df

def extract_column_features_and_labels_from_dir(
    train_dir: str,
    nrows: int = 1000,
    max_word_threshold: int = 6,
    verbose: bool = True
):
    """
    Extract normalized features and labels for each valid column in all CSVs in a directory.

    Args:
        train_dir (str): Directory containing training CSV files.
        nrows (int): Max rows to use per file.
        max_word_threshold (int): Max word threshold for field screening.
        verbose (bool): If True, print the number of columns extracted.

    Returns:
        (pd.DataFrame, pd.Series): Features DataFrame and label Series.
    """
    data = []
    files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
    for file in files:
        df = pd.read_csv(os.path.join(train_dir, file), nrows=nrows)
        dt_cols = [col for col in df.columns if any(is_datetime_like_general(str(v)) for v in df[col].dropna().head(10))]
        df_no_dt = df.drop(columns=dt_cols)
        candidates = field_screening(df_no_dt, max_word_threshold=max_word_threshold, attribute_blacklist=None, timestamp_cols=None)
        for col in candidates:
            feats = get_column_features_for_classifier(df_no_dt[col])
            if col == "case:concept:name":
                label = "case"
            elif col == "concept:name":
                label = "activity"
            else:
                label = "irrelevant"
            data.append(feats + [label])
    X = pd.DataFrame([row[:-1] for row in data],
                     columns=["n_unique_ratio_norm", "max_freq_norm", "entropy_norm", "sim_norm"])
    y = pd.Series([row[-1] for row in data], name="label")
    if verbose:
        print(f"Extracted features for {len(y)} columns from {len(files)} files.")
    return X, y

def get_feature_means_from_train(train_dir: str):
    """
    Compute the mean feature vector for case columns and activity columns in the training data.

    Args:
        train_dir (str): Path to the training directory.

    Returns:
        (np.ndarray, np.ndarray): Mean feature vector for case columns, activity columns.
    """
    from .utils import get_column_features_for_classifier
    case_feats, act_feats = [], []
    for file in os.listdir(train_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(train_dir, file))
            for col in df.columns:
                feats = get_column_features_for_classifier(df[col])
                if col == "case:concept:name":
                    case_feats.append(feats)
                elif col == "concept:name":
                    act_feats.append(feats)
    case_mean = np.mean(case_feats, axis=0) if case_feats else np.zeros(4)
    act_mean = np.mean(act_feats, axis=0) if act_feats else np.zeros(4)
    return case_mean, act_mean