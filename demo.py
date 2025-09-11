import pandas as pd
from EventTableDetector.api import (
    suggest_mandatory_column_candidates,
    validate_event_log,
    get_event_log_statistic,
    get_event_log_quality,
    convert_to_event_log
)

# Set pandas display options for better visibility
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

csv_path = "./BPIC15_1.csv"

# 1. Load the CSV file
df = pd.read_csv(csv_path, on_bad_lines='skip')

# 2. Suggest all possible (timestamp, case, activity) combinations using baseline logistic_lr
params = {
    "baseline_mode": "logistic_lr",  # Use enhanced baseline with logistic regression
    "train_dir": "./Train"
}
candidates = suggest_mandatory_column_candidates(
    df,
    method="baseline",   # Use baseline (not ML model for this demo)
    params=params,
    verbose=True
)
print("Candidate (timestamp, case, activity) columns (baseline logistic_lr):", candidates)

# 3. Validate the event log structure using these candidates
is_log, col_mapping = validate_event_log(df, candidates=candidates, verbose=True)
print("Is input a valid event log?", is_log)
print("Best column mapping:", col_mapping)

# 4. If valid, get statistics and quality, convert to standard event log and preview
if is_log:
    stats = get_event_log_statistic(df, col_mapping['case'], col_mapping['activity'])
    print("Event log statistics:", stats)

    quality = get_event_log_quality(df, col_mapping['case'], col_mapping['activity'])
    print("Event log quality:", quality)

    event_log_df = convert_to_event_log(df, col_mapping['case'], col_mapping['activity'], col_mapping['timestamp'])
    print("Standard event log preview:")
    print(event_log_df.head())
else:
    print("Input data cannot be automatically recognized as an event log.")