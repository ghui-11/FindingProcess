"""
Batch Event Table Detection Script

This script scans a folder of CSV files, automatically detects event table structure,
screens fields, and evaluates case/activity combinations using EventTableDetector package.

Usage:
    python scripts/main.py --data-dir Positiv_pass --train-features Train/train_features.csv

If no arguments are given, defaults are used.

Requires EventTableDetector package in the project root directory.
"""

import os
import argparse
from EventTableDetector.data_utils import simple_read_csv, detect_timestamp_columns_general
from EventTableDetector.field_utils import field_screening, field_scoring
from EventTableDetector.event_structures import wide_to_long, event_combinations
from EventTableDetector.train_feature_extraction import MAX_ROWS

def process_file_auto_wide_long(file_path, train_features_path, max_word_threshold=6):
    """
    Automatically processes a CSV file for event table detection and CA evaluation.
    Args:
        file_path (str): Path to CSV file.
        train_features_path (str): Path to training features CSV.
        max_word_threshold (int): Maximum word count threshold for field screening.
    """
    df = simple_read_csv(file_path, nrows=MAX_ROWS)
    timestamp_cols = detect_timestamp_columns_general(df)
    n_ts, N = len(timestamp_cols), len(df)
    print(f"Detected timestamp columns: {timestamp_cols}")
    unique_ratios = {col: df[col].nunique(dropna=True) / N if N > 0 else 0 for col in df.columns}
    case_candidates = [col for col, ratio in unique_ratios.items() if ratio > 0.95 and col not in timestamp_cols]
    low_unique_cols = [col for col, ratio in unique_ratios.items() if ratio < 0.5 and col not in timestamp_cols]

    if n_ts >= 3 and case_candidates:
        print(f"Identified as WIDE event log, timestamp columns: {timestamp_cols}, case candidates: {case_candidates}")
        for case_col in case_candidates:
            long_df = wide_to_long(df, timestamp_cols, case_col)
            print(f"\nWide table converted to long format, current case field: {case_col}")
            print("Running CA detection for case/act (WIDE structure):")
            event_combinations(long_df, ['act'], ['case'], train_features_path)
    elif n_ts == 1:
        print("Identified as LONG event log structure (1 timestamp column), field screening and CA detection flow:")
        attribute_blacklist = timestamp_cols
        candidates = field_screening(df, max_word_threshold=max_word_threshold, attribute_blacklist=attribute_blacklist)
        scored, final_case, final_act = field_scoring(df, candidates)
        if not final_case or not final_act:
            print("No valid fields for case/activity detection."); return
        event_combinations(df, final_act, final_case, train_features_path)
    elif n_ts == 2 and low_unique_cols:
        print("Identified as LONG event log structure (2 timestamp columns and low-uniqueness fields), field screening and CA detection flow:")
        attribute_blacklist = timestamp_cols
        candidates = field_screening(df, max_word_threshold=max_word_threshold, attribute_blacklist=attribute_blacklist)
        scored, final_case, final_act = field_scoring(df, candidates)
        if not final_case or not final_act:
            print("No valid fields for case/activity detection."); return
        event_combinations(df, final_act, final_case, train_features_path)
    else:
        print("Not identified as WIDE or LONG event structure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Table Detector Batch Runner")
    parser.add_argument("--data-dir", type=str, default="Positiv_pass", help="CSV data directory")
    parser.add_argument("--train-features", type=str, default="Train/train_features.csv", help="Training features csv")
    args = parser.parse_args()

    folder_path = args.data_dir
    train_features_path = args.train_features
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            print(f"\n==== Checking file: {filename} ====")
            file_path = os.path.join(folder_path, filename)
            try:
                process_file_auto_wide_long(file_path, train_features_path)
            except Exception as e:
                print(f"File read or analysis failed: {e}")