import pandas as pd
from EventTableDetector.api import (
    suggest_mandatory_column_candidates,
    validate_event_log,
    get_event_log_statistic,
    get_event_log_quality,
    convert_to_event_log
)
from EventTableDetector.feature_extraction import extract_train_features


# extract_train_features(
#     train_dir="C:/programmierung/Python/FindingProcess/Train",  # 你的训练集目录
#     output_name="train_features.csv",  # 输出文件名
#     display_result=True               # 是否打印结果
# )


csv_path = "./Positive__/BPIC15_1.csv"
 #"./Positive__/bank_transactions_data_2.csv"
#"C:/programmierung/Python/FindingProcess/positive_fialed/Incident_Management_CSV.csv"
# ./Positive__/BPIC15_1.csv
# 1. Load the CSV file
df = pd.read_csv(csv_path, on_bad_lines='skip',sep=',')


# # 2. Suggest all possible (timestamp, case, activity) combinations using baseline logistic_lr
# params = {
#     "baseline_mode": "logistic_lr",  # Use enhanced baseline with logistic regression
#     "train_dir": "./Train"
# }
candidates = suggest_mandatory_column_candidates(
    df,
    method="rf",   # Use baseline (not ML model for this demo)
    params=None,
    verbose=True
)
print("Candidate (timestamp, case, activity) columns (baseline logistic_lr):", candidates)

# 3. Validate the event log structure using these candidates
#

is_log, col_mapping = validate_event_log(df, test_types=['lof', 'isoforest','svdd','dbscan'],verbose=True)
print("Is input a valid event log?", is_log)


# # 4. If valid, get statistics and quality, convert to standard event log and preview
# if is_log:
#     stats = get_event_log_statistic(df, col_mapping['case'], col_mapping['activity'])
#     print("Event log statistics:", stats)
#
#     quality = get_event_log_quality(df, col_mapping['case'], col_mapping['activity'])
#     print("Event log quality:", quality)
#
#     event_log_df = convert_to_event_log(df, col_mapping['case'], col_mapping['activity'], col_mapping['timestamp'])
#     print("Standard event log preview:")
#     print(event_log_df.head())
# else:
#     print("Input data cannot be automatically recognized as an event log.")