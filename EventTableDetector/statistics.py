import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.utils import resample
from typing import Tuple, List, Dict, Optional
from .core import ensure_long_format
from .scoring import suggest_mandatory_column_candidates

def get_event_log_statistic(df: pd.DataFrame, case: str, activity: str) -> dict:
    """
    Compute basic statistics for an event log.

    Parameters:
        df (pd.DataFrame): Event log DataFrame.
        case (str): Case ID column name.
        activity (str): Activity column name.

    Returns:
        dict: Statistics such as number of cases, activities, variants, and average trace length.
    """
    num_cases = df[case].nunique()
    num_activities = df[activity].nunique()
    variants = df.groupby(case)[activity].apply(tuple)
    num_variants = variants.nunique()
    avg_trace_length = df.groupby(case).size().mean()
    return {
        'num_cases': num_cases,
        'num_activities': num_activities,
        'num_variants': num_variants,
        'avg_trace_length': avg_trace_length
    }

def get_event_log_quality(df: pd.DataFrame, case: str, activity: str) -> dict:
    """
    Compute quality metrics for an event log.

    Parameters:
        df (pd.DataFrame): Event log DataFrame.
        case (str): Case ID column name.
        activity (str): Activity column name.

    Returns:
        dict: Dictionary with quality metrics, e.g., "Uniqueness".
    """
    uniq = df[activity].nunique() / len(df)
    return {'Uniqueness': uniq}

def read_train_features(train_path=None):
    """
    Read training features from a CSV file.

    Parameters:
        train_path (str): Path to training features CSV.

    Returns:
        np.ndarray: Feature matrix (n_samples, n_features).
    """
    if train_path is None:
        train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix', 'train_features.csv'))
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

def ks_test(new_vector, train_matrix, alpha=0.1):
    """
    Perform Kolmogorov-Smirnov test for each feature.

    Parameters:
        new_vector (list): Feature values for new sample.
        train_matrix (np.ndarray): Training feature matrix.
        alpha (float): p-value threshold.

    Returns:
        (bool, list): Whether all features pass, details per feature.
    """
    results = []
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix[:, i]
        new_val = new_vector[i]
        stat, pval = ks_2samp(train_col, [new_val])
        results.append((stat, pval))
    ks_pass = all(pval > alpha for stat, pval in results)
    return ks_pass, results

def bootstrap_test(new_vector, train_matrix, n_bootstrap=1000, alpha=0.05):
    """
    Perform bootstrap confidence interval test for features.

    Parameters:
        new_vector (list): Feature vector for new sample.
        train_matrix (np.ndarray): Training feature matrix.
        n_bootstrap (int): Number of bootstrap samples.
        alpha (float): p-value threshold.

    Returns:
        (bool, list): Whether all features pass, details per feature.
    """
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

    Parameters:
        c_candidates (list): Candidate case columns.
        a_candidates (list): Candidate activity columns.

    Returns:
        list: List of (case, activity) tuples.
    """
    combinations = []
    for c in c_candidates:
        for a in a_candidates:
            if a != c:
                combinations.append((c, a))
    return combinations

def extract_ca_features(df, case_col, act_col):
    """
    Extract statistical features from valid case/activity traces.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
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

def compute_ca_features(df, case_field, act_field):
    """
    Compute CA features for a given case/activity field pair.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        case_field (str): Case column name.
        act_field (str): Activity column name.

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

def run_feature_tests(features, train_matrix, test_types=['ks'], test_options=None):
    """
    Run one or more statistical tests on CA features.

    Parameters:
        features (list): Feature vector for CA combination.
        train_matrix (np.ndarray): Training feature matrix.
        test_types (list): List of test types, e.g. ['ks'], ['bootstrap'].
        test_options (dict): Additional test options.

    Returns:
        dict: {test_type: {'ok': bool, 'detail': ...}}
    """
    result = {}
    test_options = test_options or {}
    if 'ks' in test_types:
        ks_alpha = test_options.get('ks_alpha', 0.1)
        ks_ok, ks_detail = ks_test(features, train_matrix, alpha=ks_alpha)
        result['ks'] = {'ok': ks_ok, 'detail': ks_detail}
    if 'bootstrap' in test_types:
        boot_alpha = test_options.get('bootstrap_alpha', 0.05)
        n_bootstrap = test_options.get('n_bootstrap', 1000)
        boot_ok, boot_detail = bootstrap_test(features, train_matrix, n_bootstrap=n_bootstrap, alpha=boot_alpha)
        result['bootstrap'] = {'ok': boot_ok, 'detail': boot_detail}
    return result

def event_combinations(
    df, a_candidates, c_candidates, train_features_path,
    test_types=['ks'], 
    test_options=None,
    verbose=True
):
    """
    Main function for evaluating case/activity combinations.

    Parameters:
        df (pd.DataFrame): Input table (long format).
        a_candidates (list): Activity candidate columns.
        c_candidates (list): Case candidate columns.
        train_features_path (str): Path to training features.
        test_types (list): List of test types.
        test_options (dict): Test options.
        verbose (bool): If True, print details.

    Returns:
        dict or None: The best valid CA combination, or None if not found.
    """
    train_matrix = read_train_features(train_features_path)
    combinations = generate_ca_combinations(c_candidates, a_candidates)
    results = []
    for c, a in combinations:
        features, p, valid_cases = compute_ca_features(df, c, a)
        if features is None:
            continue
        test_result = run_feature_tests(features, train_matrix, test_types, test_options)
        results.append({
            "case": c,
            "act": a,
            "valid_trace_percentage": p,
            "features": features,
            "test_result": test_result
        })
    for res in results:
        pass_test = True
        for test_type in test_types:
            pass_test = pass_test and res['test_result'][test_type]['ok']
        if pass_test:
            return res
    return None


def validate_event_log(
    df: pd.DataFrame,
    candidates: Optional[List[Tuple[str, str, str]]] = None,
    test_types=['ks'],
    test_options=None,
    vote='majority',
    train_features_path=None,
    verbose: bool = False
) -> Tuple[bool, Dict[str, str]]:
    """
    Validate candidate (timestamp, case, activity) tuples as event log structure.
    If candidates is None, automatically call suggest_mandatory_column_candidates to generate candidates.
    """
    long_df = ensure_long_format(df, verbose=verbose)
    results = []
    # Automatically generate candidates if not provided
    if candidates is None:
        if verbose:
            print("No candidates provided, calling suggest_mandatory_column_candidates...")
        candidates = suggest_mandatory_column_candidates(df, verbose=verbose)
        if verbose:
            print(f"Generated candidates: {candidates}")
    if train_features_path is None:
        train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix', 'train_features.csv'))
    for t, c, a in candidates:
        if verbose:
            print(f"Validating combination: timestamp={t}, case={c}, activity={a}")
        res = event_combinations(
            long_df, [a], [c], train_features_path,
            test_types=test_types, test_options=test_options, verbose=verbose
        )
        if res:
            if verbose:
                print(f"Combination PASSED: {t}, {c}, {a}")
            results.append((True, {'timestamp': t, 'case': c, 'activity': a}))
        else:
            if verbose:
                print(f"Combination FAILED: {t}, {c}, {a}")
    if vote == 'majority':
        is_log = len(results) >= (len(candidates)//2 + 1)
    elif vote == 'all':
        is_log = len(results) == len(candidates)
    else:
        is_log = len(results) > 0
    if is_log and results:
        if verbose:
            print("Event log validation PASSED.")
        return True, results[0][1]
    else:
        if verbose:
            print("Event log validation FAILED.")
        if candidates:
            t, c, a = candidates[0]
            return False, {'timestamp': t, 'case': c, 'activity': a}
        else:
            return False, {}