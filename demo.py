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

wide_data = {
    'case_id': ['case_1', 'case_2', 'case_3', 'case_4'],
    'timestamp_1': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
    'timestamp_2': ['2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
    'timestamp_3': ['2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12'],
    'other_col': [100, 200, 300, 400]
}
df_wide = pd.DataFrame(wide_data)

# 自动分支会处理3个时间戳列
# Path B-1: 选择最优时间戳，进入candidate generation
# Path B-2: melt所有时间戳，直接进入validation
is_valid, result = validate_event_log(
    df_wide,
    test_types=['lof', 'isoforest','svdd','dbscan'],
    verbose=True,
    auto_branch=True
)

print(f"\nIs valid event log: {is_valid}")
print(f"Result: {result}")


csv_path = './Positive__/BPIC15_1.csv'
#"./Positive__/assists.csv"
#  #"./Positive__/bank_transactions_data_2.csv"
# #"C:/programmierung/Python/FindingProcess/positive_fialed/Incident_Management_CSV.csv"
# # ./Positive__/BPIC15_1.csv
# 1. Load the CSV file
df = pd.read_csv(csv_path, on_bad_lines='skip',sep=',')


# # 2. Suggest all possible (timestamp, case, activity) combinations using baseline logistic_lr
# params = {
#     "baseline_mode": "logistic_lr",  # Use enhanced baseline with logistic regression
#     "train_dir": "./Train"
# }
# candidates = suggest_mandatory_column_candidates(
#     df,
#     method="rf",   # Use baseline (not ML model for this demo)
#     params=None,
#     verbose=True
# )
# print("Candidate (timestamp, case, activity) columns (baseline logistic_lr):", candidates)
#
# 3. Validate the event log structure using these candidates
#
candidates = [
    ('dateFinished', 'monitoringResource', 'action_code'),
    ('dateFinished', 'case:concept:name', 'concept:name')
]
is_log, col_mapping = validate_event_log(df, candidates=candidates,test_types=['lof', 'isoforest','svdd','dbscan'],verbose=True)
print("Is input a valid event log?", is_log)


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