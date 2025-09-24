import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.utils import resample
from typing import Tuple, List, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

from .core import ensure_long_format
from .scoring import suggest_mandatory_column_candidates

def get_event_log_statistic(df: pd.DataFrame, case: str, activity: str, timestamp: Optional[str] = None) -> dict:
    num_cases = df[case].nunique()
    num_activities = df[activity].nunique()
    variants = df.groupby(case)[activity].apply(tuple)
    num_variants = variants.nunique()
    avg_trace_length = df.groupby(case).size().mean()
    variant_ratio = num_variants / num_cases if num_cases > 0 else 0
    avg_events_per_case_ratio = avg_trace_length / num_activities if num_activities > 0 else 0
    dfg_density = None
    if timestamp is not None and timestamp in df.columns:
        try:
            df_sorted = df.sort_values([case, timestamp])
            pairs = df_sorted.groupby(case)[activity].apply(lambda x: list(zip(x[:-1], x[1:]))).explode()
            dfg_pattern_count = pairs.nunique()
            dfg_density = dfg_pattern_count / (num_activities ** 2) if num_activities > 0 else 0
        except Exception as e:
            print(f"DFG density calculation error: {e}")
            dfg_density = None

    return {
        'num_cases': num_cases,
        'num_activities': num_activities,
        'num_variants': num_variants,
        'avg_trace_length': avg_trace_length,
        'variant_ratio': variant_ratio,
        'avg_events_per_case_ratio': avg_events_per_case_ratio,
        'dfg_density': dfg_density,
    }

def get_event_log_quality(df: pd.DataFrame, case: str, activity: str) -> dict:
    uniq = df[activity].nunique() / len(df)
    return {'Uniqueness': uniq}

def read_train_features(train_path=None):
    if train_path is None:
        train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix', 'train_features.csv'))
    df = pd.read_csv(train_path)
    needed_cols = [
        'unique_case_count',
        'unique_activity_count',
        'avg_events_per_case',
        'variant_count',
        'avg_unique_acts_per_trace',
        'dfg_density',
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan
    feature_matrix = df.loc[:, needed_cols].values
    return feature_matrix

def ks_test(new_vector, train_matrix, alpha=0.1):
    results = []
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix[:, i]
        new_val = new_vector[i]
        if np.isnan(new_val) or np.isnan(train_col).all():
            results.append((np.nan, np.nan))
            continue
        stat, pval = ks_2samp(train_col[~np.isnan(train_col)], [new_val])
        results.append((stat, pval))
    ks_pass = all((pval is np.nan or pval > alpha) for stat, pval in results)
    return ks_pass, results

def bootstrap_test(new_vector, train_matrix, n_bootstrap=1000, alpha=0.05):
    results = []
    np.random.seed(42)
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix[:, i]
        new_val = new_vector[i]
        if np.isnan(new_val) or np.isnan(train_col).all():
            results.append((np.nan, np.nan, np.nan, True))
            continue
        boot_means = [np.mean(resample(train_col[~np.isnan(train_col)])) for _ in range(n_bootstrap)]
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        pass_test = ci_lower <= new_val <= ci_upper
        results.append((ci_lower, ci_upper, new_val, pass_test))
    boot_pass = all(r[-1] for r in results)
    print(results)
    return boot_pass, results

def log1p_transform(train_matrix, new_vector):
    return np.log1p(train_matrix), np.log1p(new_vector)

def bootstrap_test_logmean(new_vector, train_matrix, n_bootstrap=1000, alpha=0.05, verbose=True):
    train_matrix_log, new_vector_log = log1p_transform(train_matrix, new_vector)
    results = []
    np.random.seed(42)
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix_log[:, i]
        new_val = new_vector_log[i]
        if np.isnan(new_val) or np.isnan(train_col).all():
            results.append((np.nan, np.nan, np.nan, True))
            continue
        boot_means = [np.mean(resample(train_col[~np.isnan(train_col)])) for _ in range(n_bootstrap)]
        ci_lower, ci_upper = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
        pass_test = ci_lower <= new_val <= ci_upper
        if verbose:
            print(f"Feature {i} (log1p): sample={new_val:.3f}, 95%CI=({ci_lower:.3f},{ci_upper:.3f}), pass={pass_test}")
        results.append((ci_lower, ci_upper, new_val, pass_test))
    boot_pass = all(r[-1] for r in results)
    return boot_pass, results

def generate_ca_combinations(c_candidates, a_candidates):
    combinations = []
    for c in c_candidates:
        for a in a_candidates:
            if a != c:
                combinations.append((c, a))
    return combinations

def extract_ca_features(df, case_col, act_col, timestamp_col: Optional[str]=None):
    unique_activity_count = df[act_col].nunique()
    avg_events_per_case = df.groupby(case_col).size().mean()
    variants = df.groupby(case_col)[act_col].apply(tuple)
    variant_count = variants.nunique()
    avg_unique_acts_per_trace = variants.apply(lambda x: len(set(x))).mean()
    unique_case_count = df[case_col].nunique()
    dfg_density = None
    if timestamp_col is not None and timestamp_col in df.columns:
        try:
            df_sorted = df.sort_values([case_col, timestamp_col])
            pairs = df_sorted.groupby(case_col)[act_col].apply(lambda x: list(zip(x[:-1], x[1:]))).explode()
            dfg_pattern_count = pairs.nunique()
            dfg_density = dfg_pattern_count / (unique_activity_count ** 2) if unique_activity_count > 0 else 0
        except Exception as e:
            print(f"DFG density calculation error: {e}")
            dfg_density = None
    return [
        unique_case_count,
        unique_activity_count,
        avg_events_per_case,
        variant_count,
        avg_unique_acts_per_trace,
        dfg_density
    ]

def compute_ca_features(df, case_field, act_field, timestamp_field: Optional[str]=None):
    traces = df.groupby(case_field)[act_field].apply(list)
    valid_traces = traces[traces.apply(lambda x: len(x) > 1 and len(set(x)) > 1)]
    p = len(valid_traces) / len(traces) if len(traces) > 0 else 0
    if p < 0.6:
        print('High proportion of single-activity or single-event traces.')
        return None, None, None
    valid_cases = valid_traces.index
    sub_df = df[df[case_field].isin(valid_cases)][[case_field, act_field] + ([timestamp_field] if timestamp_field and timestamp_field in df.columns else [])].rename(columns={case_field: 'case', act_field: 'act', **({timestamp_field: 'time'} if timestamp_field and timestamp_field in df.columns else {})})
    features = extract_ca_features(sub_df, 'case', 'act', 'time' if timestamp_field and timestamp_field in df.columns else None)
    return features, p, valid_cases

def detect_lof(train_matrix, new_vector, verbose=False):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    n_neighbors = min(6, len(train_matrix)-1) if len(train_matrix) > 1 else 1
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_scaled)
    lof_score = lof.decision_function(new_scaled)[0]
    lof_pred = lof.predict(new_scaled)[0]
    if verbose:
        print(f"LOF score: {lof_score:.4f}, LOF prediction: {'Outlier' if lof_pred == -1 else 'Inlier'}")
    return {"ok": lof_pred != -1, "detail": {"lof_score": lof_score, "lof_pred": int(lof_pred)}}

def detect_isolation_forest(train_matrix, new_vector, verbose=False):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso.fit(train_scaled)
    iso_score = iso.decision_function(new_scaled)[0]
    iso_pred = iso.predict(new_scaled)[0]
    if verbose:
        print(f"Isolation Forest score: {iso_score:.4f}, IF prediction: {'Outlier' if iso_pred == -1 else 'Inlier'}")
    return {"ok": iso_pred != -1, "detail": {"iso_score": iso_score, "iso_pred": int(iso_pred)}}

def detect_svdd(train_matrix, new_vector, verbose=False):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    svdd = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    svdd.fit(train_scaled)
    svdd_score = svdd.decision_function(new_scaled)[0]
    svdd_pred = svdd.predict(new_scaled)[0]
    if verbose:
        print(f"SVDD score: {svdd_score:.4f}, SVDD prediction: {'Outlier' if svdd_pred == -1 else 'Inlier'}")
    return {"ok": svdd_pred != -1, "detail": {"svdd_score": svdd_score, "svdd_pred": int(svdd_pred)}}

def detect_dbscan(train_matrix, new_vector, verbose=False):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    dbscan = DBSCAN(eps=0.6, min_samples=3)
    dbscan.fit(train_scaled)
    core_labels = dbscan.labels_
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_scaled)
    dist, idx = nn.kneighbors(new_scaled)
    dbscan_label = core_labels[idx[0][0]]
    dbscan_pred = -1 if dbscan_label == -1 else 1
    if verbose:
        print(f"DBSCAN prediction: {'Outlier' if dbscan_pred == -1 else 'Inlier'} ")
    return {"ok": dbscan_pred != -1, "detail": {"dbscan_pred": int(dbscan_pred)}}

def run_feature_tests(features, train_matrix, test_types=['ks'], test_options=None, verbose=False):
    result = {}
    test_options = test_options or {}
    for test_type in test_types:
        if test_type == 'ks':
            ks_alpha = test_options.get('ks_alpha', 0.1)
            ks_ok, ks_detail = ks_test(features, train_matrix, alpha=ks_alpha)
            result['ks'] = {'ok': ks_ok, 'detail': ks_detail}
        elif test_type == 'bootstrap':
            boot_alpha = test_options.get('bootstrap_alpha', 0.05)
            n_bootstrap = test_options.get('n_bootstrap', 1000)
            boot_ok, boot_detail = bootstrap_test_logmean(features, train_matrix, n_bootstrap=n_bootstrap, alpha=boot_alpha)
            result['bootstrap'] = {'ok': boot_ok, 'detail': boot_detail}
        elif test_type == 'lof':
            result['lof'] = detect_lof(train_matrix, features, verbose=verbose)
        elif test_type == 'isoforest':
            result['isoforest'] = detect_isolation_forest(train_matrix, features, verbose=verbose)
        elif test_type == 'svdd':
            result['svdd'] = detect_svdd(train_matrix, features, verbose=verbose)
        elif test_type == 'dbscan':
            result['dbscan'] = detect_dbscan(train_matrix, features, verbose=verbose)
    return result

def voting(results, test_types, vote='majority'):
    ok_list = [results[t]['ok'] for t in test_types if t in results]
    if not ok_list:
        return False
    if vote == 'majority':
        return sum(ok_list) >= (len(ok_list) // 2 + 1)
    elif vote == 'all':
        return all(ok_list)
    elif vote == 'any':
        return any(ok_list)
    elif vote == 'half':
        return sum(ok_list) >= (len(ok_list) // 2)
    else:
        return False

def event_combinations(
    df, a_candidates, c_candidates, train_features_path,
    test_types=['ks'],
    test_options=None,
    verbose=True,
    timestamp_candidates: Optional[List[str]] = None,
    vote='majority'
):
    train_matrix = read_train_features(train_features_path)
    combinations = generate_ca_combinations(c_candidates, a_candidates)
    results = []
    if timestamp_candidates is None:
        timestamp_candidates = []
    for c, a in combinations:
        ts = timestamp_candidates[0] if len(timestamp_candidates) > 0 else None
        features, p, valid_cases = compute_ca_features(df, c, a, ts)
        if features is None:
            continue
        test_result = run_feature_tests(features, train_matrix, test_types, test_options, verbose=verbose)
        pass_test = voting(test_result, test_types, vote)
        res = {
            "case": c,
            "act": a,
            "timestamp": ts,
            "valid_trace_percentage": p,
            "features": features,
            "test_result": test_result,
            "pass": pass_test
        }
        results.append(res)
    return results

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
    Validate whether the given DataFrame is an event log, based on candidate (timestamp, case, activity) combinations.

    Args:
        df (pd.DataFrame): Input DataFrame to validate.
        candidates (list, optional): List of (timestamp, case, activity) column combinations. If None, will be auto-generated.
        test_types (list): List of test types to apply for validation.
        test_options: Additional options for the tests.
        vote (str): Voting strategy ('majority', 'all', 'any') to determine validation pass within EACH combination.
        train_features_path: Path to training features CSV.
        verbose (bool): If True, print detailed debug info.

    Returns:
        Tuple[bool, Dict[str, str]]: (is_event_log, mapping_dict). If true, mapping_dict contains the validated column mapping.
    """
    # Ensure DataFrame is in long format
    long_df = ensure_long_format(df, verbose=False)
    # If no candidates provided, generate them
    if candidates is None:
        if verbose:
            print("No candidates provided, calling suggest_mandatory_column_candidates...")
        candidates = suggest_mandatory_column_candidates(df, method="rf", verbose=verbose)
    # If no valid candidates
    if not candidates:
        if verbose:
            print("No valid candidates. Event log validation FAILED.")
        return False, {}
    # Default train features path
    if train_features_path is None:
        train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix', 'train_features.csv'))

    # Iterate through all combinations
    for t, c, a in candidates:
        if verbose:
            print(f"Validating combination: timestamp={t}, case={c}, activity={a}")
        res_list = event_combinations(
            long_df, [a], [c], train_features_path,
            test_types=test_types, test_options=test_options, verbose=verbose,
            timestamp_candidates=[t] if t else [],
            vote=vote
        )
        # Voting is applied to the test results INSIDE EACH combination.
        # If any combination passes, return immediately as valid.
        if res_list:
            res = res_list[0]
            if res["pass"]:  # This combination passed the vote among its tests
                if verbose:
                    print(f"Combination {t}, {c}, {a} PASSED all tests. Event log validation PASSED.")
                return True, {'timestamp': t, 'case': c, 'activity': a, "test_result": res.get("test_result", {})}
            else:
                if verbose:
                    print(f"Combination {t}, {c}, {a} FAILED one or more tests.")
        else:
            if verbose:
                print(f"Combination FAILED: {t}, {c}, {a}")
    # If none of the combinations passed, return False
    if verbose:
        print("Event log validation FAILED. No combination satisfied the requirement.")
    if candidates:
        t, c, a = candidates[0]
        return False, {'timestamp': t, 'case': c, 'activity': a}
    else:
        return False, {}