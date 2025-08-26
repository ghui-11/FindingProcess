import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp
from sklearn.utils import resample
import matplotlib.pyplot as plt

def wide_to_long(df, timestamp_cols, case_col):
    """
    Converts a wide-format event table to a long-format event log.
    Args:
        df (pd.DataFrame): Input dataframe.
        timestamp_cols (list): Timestamp column names.
        case_col (str): Case column name.
    Returns:
        pd.DataFrame: Melted dataframe in long format.
    """
    long_df = (
        df.melt(id_vars=[case_col], value_vars=timestamp_cols,
                var_name='act', value_name='timestamp')
        .dropna(subset=['timestamp'])
        .rename(columns={case_col: 'case'})
    )
    return long_df

def extract_ca_features(df, case_col, act_col):
    """
    Extracts statistical features from valid CA traces.
    Args:
        df (pd.DataFrame): Input dataframe.
        case_col (str): Case column name.
        act_col (str): Activity column name.
    Returns:
        list: [unique_activity_count, avg_events_per_case, variant_count, avg_unique_acts_per_trace]
    """
    unique_activity_count = df[act_col].nunique()
    avg_events_per_case = df.groupby(case_col).size().mean()
    variants = df.groupby(case_col)[act_col].apply(tuple)
    variant_count = variants.nunique()
    avg_unique_acts_per_trace = variants.apply(lambda x: len(set(x))).mean()
    return [
        unique_activity_count,
        avg_events_per_case,
        variant_count,
        avg_unique_acts_per_trace
    ]

def read_train_features(train_path):
    """
    Reads training features from a CSV file.
    Args:
        train_path (str): CSV file path.
    Returns:
        np.ndarray: Feature matrix.
    """
    df = pd.read_csv(train_path)
    needed_cols = [
        'valid_trace_percentage',
        'unique_activity_count',
        'avg_events_per_case',
        'variant_count',
        'avg_unique_acts_per_trace'
    ]
    feature_matrix = df.loc[:, needed_cols].values
    return feature_matrix

def ks_test(new_vector, train_matrix, alpha=0.1, feature_names=None):
    """
    Performs KS test on each feature.
    Args:
        new_vector (list): Feature vector for new sample.
        train_matrix (np.ndarray): Training feature matrix.
        alpha (float): p-value threshold.
        feature_names (list): Feature names.
    Returns:
        (bool, list): Pass flag, details.
    """
    results = []
    print("\nKS test p-values for each feature:")
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix[:, i]
        new_val = new_vector[i]
        stat, pval = ks_2samp(train_col, [new_val])
        results.append((stat, pval))
        fname = feature_names[i] if feature_names else f"Feature {i+1}"
        print(f"{fname} KS p-value: {pval:.4f}")
    ks_pass = all(pval > alpha for stat, pval in results)
    return ks_pass, results

def bootstrap_test(new_vector, train_matrix, n_bootstrap=1000, alpha=0.05):
    """
    Performs a bootstrap confidence interval test on features.
    Args:
        new_vector (list): Feature vector for new sample.
        train_matrix (np.ndarray): Training feature matrix.
        n_bootstrap (int): Number of bootstrap samples.
        alpha (float): p-value threshold.
    Returns:
        (bool, list): Pass flag, details.
    """
    if train_matrix.shape[0] < 10:
        print(f"Warning: Only {train_matrix.shape[0]} training samples, bootstrap may be unreliable!")
    results = []
    np.random.seed(42)
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix[:, i]
        boot_means = [np.mean(resample(train_col)) for _ in range(n_bootstrap)]
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        new_val = new_vector[i]
        pass_test = ci_lower <= new_val <= ci_upper
        results.append((ci_lower, ci_upper, new_val, pass_test))
    boot_pass = all(pass_test for ci_low, ci_up, new_val, pass_test in results)
    return boot_pass, results

def event_combinations(df, a_candidates, c_candidates, train_features_path):
    """
    Main function for evaluating case/activity combinations.
    Args:
        df (pd.DataFrame): Input dataframe.
        a_candidates (list): Activity candidates.
        c_candidates (list): Case candidates.
        train_features_path (str): Training features CSV path.
    Returns:
        dict or None: Best CA combination, or None if not found.
    """
    results = []
    train_matrix = read_train_features(train_features_path)
    feature_names = [
        "valid_trace_percentage",
        "unique_activity_count", "avg_events_per_case",
        "variant_count", "avg_unique_acts_per_trace"
    ]
    print("\nAll CA combination candidates:")
    for c in c_candidates:
        for a in a_candidates:
            if a == c:
                continue
            print(f"Case field: {c}, Activity field: {a}")
    for c in c_candidates:
        for a in a_candidates:
            if a == c:
                continue
            traces = df.groupby(c)[a].apply(list)
            valid_traces = traces[traces.apply(lambda x: len(x) > 1 and len(set(x)) > 1)]
            p = len(valid_traces) / len(traces) if len(traces) > 0 else 0
            if p < 0.5:
                print('Over 50% of traces are invalid. Skipping this combination.')
                continue
            valid_cases = valid_traces.index
            sub_df = df[df[c].isin(valid_cases)][[c, a]].rename(columns={c: 'case', a: 'act'})
            features = [p] + extract_ca_features(sub_df, 'case', 'act')
            print(f"\nEvaluating CA combination: case={c}, act={a}, valid_trace_percentage={p:.4f}")
            ks_ok, ks_detail = ks_test(features, train_matrix, feature_names=feature_names)
            boot_ok, boot_detail = bootstrap_test(features, train_matrix)
            print(f"Bootstrap test pass: {boot_ok}")
            print("Bootstrap confidence intervals and new values:")
            for idx, (ci_low, ci_up, new_val, pass_test) in enumerate(boot_detail):
                print(f"{feature_names[idx]}: [{ci_low:.2f}, {ci_up:.2f}], new={new_val:.2f}, pass={pass_test}")
            results.append({
                "case": c,
                "act": a,
                "valid_trace_percentage": p,
                "features": features,
                "ks_ok": ks_ok,
                "ks_detail": ks_detail,
                "boot_ok": boot_ok,
                "boot_detail": boot_detail,
            })
    for res in results:
        print(f"\nCase: {res['case']} | Act: {res['act']} | ValidTrace%: {res['valid_trace_percentage']:.2f}")
        print(f"  Features: {res['features']}")
        print(f"  KS test pass: {res['ks_ok']}")
        print("-"*40)
    for res in results:
        if res['ks_ok']:
            print(f"\nValid CA combination found: case={res['case']}, act={res['act']}, features={res['features']}")
            return res
    print("\nNo valid CA combination found (KS test failed or no frequent sequence).")
    return None