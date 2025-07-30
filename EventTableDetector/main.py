import os
import pandas as pd

from data_utils import simple_read_csv, detect_timestamp_columns_general
from field_utils import field_screening, field_scoring
from event_structures import wide_to_long, event_combinations

def process_file_auto_wide_long_v3(file_path, train_features_path, max_word_threshold=6):
    df = simple_read_csv(file_path, nrows=5000)
    timestamp_cols = detect_timestamp_columns_general(df)
    n_ts, N = len(timestamp_cols), len(df)
    print(f"检测到时间戳列: {timestamp_cols}")
    unique_ratios = {col: df[col].nunique(dropna=True) / N if N > 0 else 0 for col in df.columns}
    case_candidates = [col for col, ratio in unique_ratios.items() if ratio > 0.95 and col not in timestamp_cols]
    low_unique_cols = [col for col, ratio in unique_ratios.items() if ratio < 0.5 and col not in timestamp_cols]

    if n_ts >= 3 and case_candidates:
        print(f"判为WIDE事件日志，时间戳列: {timestamp_cols}，case候选: {case_candidates}")
        for case_col in case_candidates:
            long_df = wide_to_long(df, timestamp_cols, case_col)
            print(f"\n宽表已展开为长表，当前case字段: {case_col}")
            print("直接以case/act做CA检验（WIDE结构）：")
            event_combinations(long_df, ['act'], ['case'], train_features_path)
    elif n_ts == 1:
        print("判为LONG事件日志结构（1个时间戳列），字段筛选和CA检验流程：")
        attribute_blacklist = timestamp_cols
        candidates = field_screening(df, max_word_threshold=max_word_threshold, attribute_blacklist=attribute_blacklist)
        scored, final_case, final_act = field_scoring(df, candidates)
        if not final_case or not final_act:
            print("No valid fields for case/act detection"); return
        event_combinations(df, final_act, final_case, train_features_path)
    elif n_ts == 2 and low_unique_cols:
        print("判为LONG事件日志结构（2个时间戳列且存在低唯一性字段），字段筛选和CA检验流程：")
        attribute_blacklist = timestamp_cols
        candidates = field_screening(df, max_word_threshold=max_word_threshold, attribute_blacklist=attribute_blacklist)
        scored, final_case, final_act = field_scoring(df, candidates)
        if not final_case or not final_act:
            print("No valid fields for case/act detection"); return
        event_combinations(df, final_act, final_case, train_features_path)
    else:
        print("不属于WIDE或LONG结构。")

if __name__ == "__main__":
    folder_path = "../Positiv_pass"
    train_features_path = os.path.join(os.getcwd(), "../Train", "train_features.csv")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            print(f"\n==== 检查文件: {filename} ====")
            file_path = os.path.join(folder_path, filename)
            try:
                process_file_auto_wide_long_v3(file_path, train_features_path)
            except Exception as e:
                print(f"文件读取或分析失败: {e}")