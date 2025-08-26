import os
import pandas as pd
import csv

# Maximum number of rows/events to read from each file
MAX_ROWS = 1000  # Change this constant to adjust the global row/event limit

def extract_train_features(train_dir=None, output_name='train_features.csv', display_result=True, max_rows=MAX_ROWS):
    """
    Extracts statistical features from up to max_rows events from each .csv file in the specified Train directory.
    For each file, computes 'valid_trace_percentage' (traces with length >1 and unique activity count >1).
    All other metrics are computed only on valid traces/events.
    The features are saved as a CSV file in the same directory.

    New features:
        - direct_following_score: 1 - |DF_relation_set| / act^2
        - change_rate: total number of activity changes in traces / (total_event_count - case_count)

    Parameters:
        train_dir (str): Path to the Train directory. Defaults to sibling 'Train' directory.
        output_name (str): Name of the CSV file to save the extracted features.
        display_result (bool): If True, prints the feature DataFrame after extraction.
        max_rows (int): Maximum number of rows/events to read from each file.

    Returns:
        features_df (pd.DataFrame): DataFrame containing extracted features for each file.
    """
    if train_dir is None:
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train'))
    files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]

    feature_list = []

    for file in files:
        filepath = os.path.join(train_dir, file)
        # Read up to max_rows from CSV file, auto-detect delimiter
        with open(filepath, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[',', ':', ';', '\t'])
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
        df = pd.read_csv(filepath, sep=delimiter, nrows=max_rows)
        # Try to locate case/activity columns from candidate names
        case_candidates = ['case:concept:name', 'case_id', 'Case_ID']
        act_candidates = ['concept:name','activity', 'activity']
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

        # Compute traces and valid traces
        traces = df.groupby(case_col)[act_col].apply(list)
        valid_traces = traces[traces.apply(lambda x: len(x) > 1 and len(set(x)) > 1)]
        valid_trace_percentage = len(valid_traces) / len(traces) if len(traces) > 0 else 0

        # Filter original dataframe to only those events in valid traces
        valid_cases = valid_traces.index
        df_filtered = df[df[case_col].isin(valid_cases)]

        # Feature extraction using filtered events
        total_event_count = len(df_filtered)  # Total number of events in valid traces
        unique_case_count = df_filtered[case_col].nunique()  # Number of unique valid cases
        unique_activity_count = df_filtered[act_col].nunique()  # Number of unique activities
        avg_events_per_case = df_filtered.groupby(case_col).size().mean() if unique_case_count > 0 else 0
        variants = df_filtered.groupby(case_col)[act_col].apply(tuple)  # Activity sequence for each valid case
        variant_count = variants.nunique() if unique_case_count > 0 else 0
        avg_unique_acts_per_trace = variants.apply(lambda x: len(set(x))).mean() if unique_case_count > 0 else 0

        # ----- New Feature 1: Direct Following Relation -----
        # Get activity sequences for all cases
        all_traces = df_filtered.groupby(case_col)[act_col].apply(list)
        # Count all direct following pairs
        df_relations = set()
        for trace in all_traces:
            df_relations.update(zip(trace[:-1], trace[1:]))
        direct_following_score = len(df_relations) / (unique_activity_count ** 2) if unique_activity_count > 0 else 0

        # ----- New Feature 2: Change Rate -----
        change_count = 0
        for trace in all_traces:
            change_count += sum(1 for i in range(1, len(trace)) if trace[i] != trace[i-1])
        denominator = total_event_count - unique_case_count
        change_rate = change_count / denominator if denominator > 0 else 0

        feature_list.append({
            'file': file,
            'total_event_count': total_event_count,
            'unique_case_count': unique_case_count,
            'valid_trace_percentage': valid_trace_percentage,
            'unique_activity_count': unique_activity_count,
            'avg_events_per_case': avg_events_per_case,
            'variant_count': variant_count,
            'avg_unique_acts_per_trace': avg_unique_acts_per_trace,
            'direct_following_score': direct_following_score,
            'change_rate': change_rate
        })

    features_df = pd.DataFrame(feature_list)
    output_path = os.path.join(train_dir, output_name)
    features_df.to_csv(output_path, index=False)
    if display_result:
        print(features_df)
    return features_df

if __name__ == "__main__":
    extract_train_features()