"""
Demo for EventTableDetector package usage.

This script demonstrates how to:
    - Load a CSV file
    - Automatically detect timestamp columns
    - Screen candidate case/activity fields
    - Evaluate case/activity combinations using event_structures

To use:
    1. Run from the project root directory:
        python scripts/demo.py
    2. Ensure EventTableDetector is in the same root directory.
    3. Update data_path and train_features_path as needed.
"""

import os
from EventTableDetector.data_utils import simple_read_csv, detect_timestamp_columns_general
from EventTableDetector.field_utils import field_screening, field_scoring
from EventTableDetector.event_structures import wide_to_long, event_combinations

if __name__ == "__main__":
    # Update these paths to your local data files
    data_path = "Positiv_pass/BPIC_2012_W_test.csv"
    train_features_path = "Train/train_features.csv"

    # Load data
    df = simple_read_csv(data_path, nrows=1000)
    timestamp_cols = detect_timestamp_columns_general(df)
    print("Detected timestamp columns:", timestamp_cols)

    # Candidate field screening (excluding timestamps)
    candidates = field_screening(df, attribute_blacklist=timestamp_cols)
    print("Candidate fields:", candidates)

    # Field scoring to select best case/activity fields
    scored, case_fields, act_fields = field_scoring(df, candidates)
    print("Top case fields:", case_fields)
    print("Top activity fields:", act_fields)

    # Run CA detection and evaluation
    result = event_combinations(df, act_fields, case_fields, train_features_path)
    if result is not None:
        print("Best CA combination found:", result)
    else:
        print("No valid CA combination detected.")