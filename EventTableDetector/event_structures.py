import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp
from sklearn.utils import resample


def wide_to_long(df, timestamp_cols, case_col):
    """
    Convert a wide-format event table to a long-format event log.
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
    Extract statistical features from valid case/activity traces.
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
    Read training features from a CSV file.
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
    Perform KS test for each feature.
    Args:
        new_vector (list): Feature values for new sample.
        train_matrix (np.ndarray): Training feature matrix.
        alpha (float): p-value threshold.
        feature_names (list): Feature names.
    Returns:
        (bool, list): Whether all features pass, details per feature.
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
    Perform bootstrap confidence interval test for features.
    Args:
        new_vector (list): Feature vector for new sample.
        train_matrix (np.ndarray): Training feature matrix.
        n_bootstrap (int): Number of bootstrap samples.
        alpha (float): p-value threshold.
    Returns:
        (bool, list): Whether all features pass, details per feature.
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

def generate_ca_combinations(c_candidates, a_candidates):
    """
    Generate all valid case/activity field combinations.
    Args:
        c_candidates (list): Case field candidates.
        a_candidates (list): Activity field candidates.
    Returns:
        list of tuples: [(case_field, activity_field), ...]
    """
    combinations = []
    for c in c_candidates:
        for a in a_candidates:
            if a != c:
                combinations.append((c, a))
    return combinations

def compute_ca_features(df, case_field, act_field):
    """
    Compute CA features for a given case/activity field pair.
    Args:
        df (pd.DataFrame): Input dataframe.
        case_field (str): Case field name.
        act_field (str): Activity field name.
    Returns:
        tuple: (features, valid_trace_percentage, valid_cases)
    """
    traces = df.groupby(case_field)[act_field].apply(list)
    valid_traces = traces[traces.apply(lambda x: len(x) > 1 and len(set(x)) > 1)]
    p = len(valid_traces) / len(traces) if len(traces) > 0 else 0
    if p < 0.5:
        return None, None, None
    valid_cases = valid_traces.index
    sub_df = df[df[case_field].isin(valid_cases)][[case_field, act_field]].rename(columns={case_field: 'case', act_field: 'act'})
    features = [p] + extract_ca_features(sub_df, 'case', 'act')
    return features, p, valid_cases

def run_feature_tests(features, train_matrix, test_types=['ks'], test_options=None, feature_names=None):
    """
    Run one or more statistical tests on CA features.
    Args:
        features (list): Feature vector for CA combination.
        train_matrix (np.ndarray): Training feature matrix.
        test_types (list): List of test types, e.g. ['ks'], ['bootstrap'].
        test_options (dict): Additional test options.
        feature_names (list): Feature names.
    Returns:
        dict: {test_type: {'ok': bool, 'detail': ...}}
    """
    result = {}
    test_options = test_options or {}
    if 'ks' in test_types:
        ks_alpha = test_options.get('ks_alpha', 0.1)
        ks_ok, ks_detail = ks_test(features, train_matrix, alpha=ks_alpha, feature_names=feature_names)
        result['ks'] = {'ok': ks_ok, 'detail': ks_detail}
    if 'bootstrap' in test_types:
        boot_alpha = test_options.get('bootstrap_alpha', 0.05)
        n_bootstrap = test_options.get('n_bootstrap', 1000)
        boot_ok, boot_detail = bootstrap_test(features, train_matrix, n_bootstrap=n_bootstrap, alpha=boot_alpha)
        result['bootstrap'] = {'ok': boot_ok, 'detail': boot_detail}
    # Extension: Add other test types here if needed
    return result

def event_combinations(
    df, a_candidates, c_candidates, train_features_path,
    test_types=['ks'],  # e.g. ['ks'], ['bootstrap'], ['ks','bootstrap']
    test_options=None,
    verbose=True
):
    """
    Main function for evaluating case/activity combinations.
    Args:
        df (pd.DataFrame): Input dataframe.
        a_candidates (list): Activity candidates.
        c_candidates (list): Case candidates.
        train_features_path (str): Training features CSV path.
        test_types (list): List of test types for validation.
        test_options (dict): Additional options for tests.
        verbose (bool): Whether to print details.
    Returns:
        dict or None: Best CA combination, or None if not found.
    """
    train_matrix = read_train_features(train_features_path)
    feature_names = [
        "valid_trace_percentage",
        "unique_activity_count", "avg_events_per_case",
        "variant_count", "avg_unique_acts_per_trace"
    ]
    combinations = generate_ca_combinations(df, c_candidates, a_candidates)
    results = []
    for c, a in combinations:
        features, p, valid_cases = compute_ca_features(df, c, a)
        if features is None:
            if verbose:
                print(f"Over 50% of traces are invalid for case={c}, act={a}. Skipping.")
            continue
        if verbose:
            print(f"\nEvaluating CA combination: case={c}, act={a}, valid_trace_percentage={p:.4f}")
        test_result = run_feature_tests(features, train_matrix, test_types, test_options, feature_names)
        if verbose:
            for test_type, res in test_result.items():
                print(f"{test_type} test pass: {res['ok']}")
                if test_type == 'bootstrap':
                    print("Bootstrap confidence intervals and new values:")
                    for idx, (ci_low, ci_up, new_val, pass_test) in enumerate(res['detail']):
                        print(f"{feature_names[idx]}: [{ci_low:.2f}, {ci_up:.2f}], new={new_val:.2f}, pass={pass_test}")
        results.append({
            "case": c,
            "act": a,
            "valid_trace_percentage": p,
            "features": features,
            "test_result": test_result
        })
    # Select valid CA combination(s)
    for res in results:
        pass_test = True
        for test_type in test_types:
            pass_test = pass_test and res['test_result'][test_type]['ok']
        if pass_test:
            if verbose:
                print(f"\nValid CA combination found: case={res['case']}, act={res['act']}, features={res['features']}")
            return res
    print("\nNo valid CA combination found.")
    return None