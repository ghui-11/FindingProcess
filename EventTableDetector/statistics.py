import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

from .core import preprocess_and_generate_candidates
from .scoring import suggest_mandatory_column_candidates


def get_event_log_statistic(df: pd.DataFrame, case: str, activity: str, timestamp: str = None) -> dict:
    """
    Extract all event log statistical and quality metrics in one dictionary.

    Args:
        df: Input dataframe
        case: Column name for case ID
        activity: Column name for activity
        timestamp: Optional column name for timestamp

    Returns:
        Dictionary with all statistics and quality metrics
    """
    # ====== Statistical Features ======
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

    # ====== Quality Metric ======
    uniq = df[activity].nunique() / len(df) if len(df) > 0 else np.nan

    # ====== Combine All Metrics ======
    metrics = {
        'num_cases': num_cases,
        'num_activities': num_activities,
        'num_variants': num_variants,
        'avg_trace_length': avg_trace_length,
        'variant_ratio': variant_ratio,
        'avg_events_per_case_ratio': avg_events_per_case_ratio,
        'dfg_density': dfg_density,
        'Uniqueness': uniq,
    }
    return metrics


def read_train_features(train_path=None):
    """
    Read training feature matrix from file.

    Args:
        train_path: Path to training features CSV file

    Returns:
        Feature matrix array
    """
    if train_path is None:
        train_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix', 'train_features.csv'))
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


def generate_ca_combinations(c_candidates, a_candidates):
    """
    Generate all valid case-activity combinations.

    Args:
        c_candidates: List of case column candidates
        a_candidates: List of activity column candidates

    Returns:
        List of (case, activity) tuples
    """
    combinations = []
    for c in c_candidates:
        for a in a_candidates:
            if a != c:
                combinations.append((c, a))
    return combinations


def extract_ca_features(df, case_col, act_col, timestamp_col: Optional[str] = None):
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

def compute_ca_features(df, case_field, act_field, timestamp_field: Optional[str] = None):
    traces = df.groupby(case_field)[act_field].apply(list)
    validation_info = {
        'passed': True,
        'messages': []
    }

    if timestamp_field and timestamp_field in df.columns:
        df = df.sort_values([case_field, timestamp_field])
        df = df.drop_duplicates(subset=[case_field, timestamp_field], keep='first')

    all_single_events = all(len(trace) == 1 for trace in traces)
    if all_single_events:
        validation_message = 'Validation failed: All traces are single-event traces. Skipping this combination.'
        validation_info['passed'] = False
        validation_info['messages'].append(validation_message)
        return None, None, None, validation_info

    all_identical_values_within_traces = all(len(set(trace)) == 1 for trace in traces)
    if all_identical_values_within_traces:
        validation_message = 'Validation failed: All traces have identical activity values ' \
                             'within each trace. Skipping this combination.'
        validation_info['passed'] = False
        validation_info['messages'].append(validation_message)
        return None, None, None, validation_info

    valid_traces = traces[traces.apply(lambda x: len(x) > 1 and len(set(x)) > 1)]
    p = len(valid_traces) / len(traces) if len(traces) > 0 else 0
    valid_cases = valid_traces.index

    if p < 0.3:
        validation_message = f'Validation failed: Valid trace ratio {p:.2%} is below 30% threshold. ' \
                             f'{len(valid_traces)}/{len(traces)} traces are valid. Skipping this combination.'
        validation_info['passed'] = False
        validation_info['messages'].append(validation_message)

    validation_message = f'Validation passed: {len(valid_traces)}/{len(traces)} traces are valid. Valid ratio: {p:.2%}'
    validation_info['messages'].append(validation_message)

    sub_df = df[df[case_field].isin(valid_cases)][[case_field, act_field] + (
        [timestamp_field] if timestamp_field and timestamp_field in df.columns else [])].rename(
        columns={case_field: 'case', act_field: 'act',
                 **({timestamp_field: 'time'} if timestamp_field and timestamp_field in df.columns else {})})

    features = extract_ca_features(sub_df, 'case', 'act',
                                   'time' if timestamp_field and timestamp_field in df.columns else None)
    return features, p, valid_cases, validation_info

def voting(results, test_types, vote='majority'):
    """
    Aggregate test results using voting strategy.

    Returns True if anomaly is detected (outlier detected by majority/all/any tests)
    - ok=False means outlier detected (anomaly)
    - ok=True means normal (inlier)
    """
    anomaly_list = [not results[t]['ok'] for t in test_types if t in results]
    if not anomaly_list:
        return False

    if vote == 'majority':
        return sum(anomaly_list) >= (len(anomaly_list) // 2 + 1)
    elif vote == 'all':
        return all(anomaly_list)
    elif vote == 'any':
        return any(anomaly_list)
    elif vote == 'half':
        return sum(anomaly_list) >= (len(anomaly_list) // 2)
    else:
        return False


def detect_lof(train_matrix, new_vector, verbose=False):
    """
    Detect anomalies using Local Outlier Factor.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    n_neighbors = min(4, len(train_matrix) - 1) if len(train_matrix) > 1 else 1
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_scaled)
    lof_score = lof.decision_function(new_scaled)[0]
    lof_pred = lof.predict(new_scaled)[0]

    if verbose:
        print(f"LOF score: {lof_score:.4f}, anomaly_prob: {anomaly_probability:.4f}, "
              f"prediction: {'Outlier' if lof_pred == -1 else 'Inlier'}")

    return {
        "ok": lof_pred != -1,
        "detail": {
            "lof_score": float(lof_score),
            "lof_pred": int(lof_pred),
            "is_anomaly": lof_pred == -1,
            "prediction_binary": 0 if lof_pred == -1 else 1
        }
    }


def detect_isolation_forest(train_matrix, new_vector, verbose=False):
    """
    Detect anomalies using Isolation Forest.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso.fit(train_scaled)
    iso_score = iso.decision_function(new_scaled)[0]
    iso_pred = iso.predict(new_scaled)[0]

    if verbose:
        print(f"Isolation Forest score: {iso_score:.4f}, anomaly_prob: {anomaly_probability:.4f}, "
              f"prediction: {'Outlier' if iso_pred == -1 else 'Inlier'}")

    return {
        "ok": iso_pred != -1,
        "detail": {
            "iso_score": float(iso_score),
            "iso_pred": int(iso_pred),
            "is_anomaly": iso_pred == -1,
            "prediction_binary": 0 if iso_pred == -1 else 1
        }
    }


def detect_ocsvm(train_matrix, new_vector, verbose=False):
    """
    Detect anomalies using One-Class SVM.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    svdd = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    svdd.fit(train_scaled)
    svdd_score = svdd.decision_function(new_scaled)[0]
    svdd_pred = svdd.predict(new_scaled)[0]

    if verbose:
        print(f"OCSVM score: {svdd_score:.4f}, anomaly_prob: {anomaly_probability:.4f}, "
              f"prediction: {'Outlier' if svdd_pred == -1 else 'Inlier'}")

    return {
        "ok": svdd_pred != -1,
        "detail": {
            "ocsvm_score": float(svdd_score),
            "ocsvm_pred": int(svdd_pred),
            "is_anomaly": svdd_pred == -1,
            "prediction_binary": 0 if svdd_pred == -1 else 1
        }
    }


def detect_dbscan(train_matrix, new_vector, verbose=False):
    """
    Detect anomalies using DBSCAN.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    new_scaled = scaler.transform(np.array(new_vector).reshape(1, -1))
    dbscan = DBSCAN(eps=0.6, min_samples=4)
    dbscan.fit(train_scaled)
    core_labels = dbscan.labels_
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_scaled)
    dist, idx = nn.kneighbors(new_scaled)
    dbscan_label = core_labels[idx[0][0]]
    distance = dist[0][0]

    dbscan_pred = -1 if distance > 0.6 else 1

    if verbose:
        print(f"DBSCAN distance: {distance:.4f}, anomaly_prob: {anomaly_probability:.4f}, "
              f"prediction: {'Outlier' if dbscan_pred == -1 else 'Inlier'}")

    return {
        "ok": dbscan_pred != -1,
        "detail": {
            "dbscan_distance": float(distance),
            "dbscan_pred": int(dbscan_pred),
            "is_anomaly": dbscan_pred == -1,
            "prediction_binary": 0 if dbscan_pred == -1 else 1
        }
    }


def run_feature_tests(features, train_matrix, test_types=['lof'], test_options=None, verbose=False):
    """
    Run multiple feature tests and return results with anomaly scores.

    Args:
        features: Feature vector to test
        train_matrix: Training feature matrix
        test_types: List of test types to run (lof, isoforest, ocsvm, dbscan)
        test_options: Dictionary with test-specific options
        verbose: Whether to print detailed results

    Returns:
        Dictionary with test results
    """
    result = {}
    test_options = test_options or {}
    for test_type in test_types:
        if test_type == 'lof':
            result['lof'] = detect_lof(train_matrix, features, verbose=verbose)
        elif test_type == 'isoforest':
            result['isoforest'] = detect_isolation_forest(train_matrix, features, verbose=verbose)
        elif test_type == 'ocsvm':
            result['ocsvm'] = detect_ocsvm(train_matrix, features, verbose=verbose)
        elif test_type == 'dbscan':
            result['dbscan'] = detect_dbscan(train_matrix, features, verbose=verbose)
    return result


def event_combinations(
        df, a_candidates, c_candidates, train_features_path,
        test_types=['lof'],
        test_options=None,
        verbose=True,
        timestamp_candidates: Optional[List[str]] = None,
        vote='majority'
):
    """
    Run event combinations with enhanced logging.

    Args:
        df: Input dataframe
        a_candidates: List of activity column candidates
        c_candidates: List of case column candidates
        train_features_path: Path to training features
        test_types: List of test types to run
        test_options: Dictionary with test-specific options
        verbose: Whether to print detailed results
        timestamp_candidates: List of timestamp column candidates
        vote: Voting strategy

    Returns:
        List of result dictionaries with validation info
    """
    train_matrix = read_train_features(train_features_path)
    combinations = generate_ca_combinations(c_candidates, a_candidates)
    results = []
    if timestamp_candidates is None:
        timestamp_candidates = []

    for c, a in combinations:
        ts = timestamp_candidates[0] if len(timestamp_candidates) > 0 else None
        features, p, valid_cases, validation_info = compute_ca_features(df, c, a, ts)

        if features is None:
            res = {
                "case": c,
                "act": a,
                "timestamp": ts,
                "valid_trace_percentage": p,
                "features": None,
                "test_result": {},
                "pass": False,
                "validation_info": validation_info
            }
            results.append(res)
            if verbose:
                print(f"[{c}, {a}] Validation failed: {validation_info['messages']}")
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
            "pass": pass_test,
            "validation_info": validation_info
        }
        results.append(res)

    return results


def validate_event_log(
        df: pd.DataFrame,
        candidates: Optional[List[Tuple[str, str, str]]] = None,
        test_types=['lof', 'isoforest', 'ocsvm', 'dbscan'],
        test_options=None,
        vote='half',
        train_features_path=None,
        params=None,
        verbose: bool = False,
        auto_branch: bool = True,
) -> Tuple[bool, List[Dict]]:
    """
    Validate event log and detect anomalies.
    Returns (success: bool, all valid combinations as [{'timestamp':..., 'case':..., 'activity':...}, ...])
    """
    if train_features_path is None:
        train_features_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix', 'train_features.csv'))

    valid_combinations = []
    has_valid = False

    if auto_branch and candidates is None:
        try:
            event_table_candidates = preprocess_and_generate_candidates(df, verbose=verbose)
        except ValueError as e:
            if verbose:
                print(f"Branching preprocessing failed: {e}")
            return False, []

        if not event_table_candidates:
            if verbose:
                print("No event table candidates generated.")
            return False, []

        for candidate_idx, candidate in enumerate(event_table_candidates, 1):
            candidate_df = candidate['df']
            requires_gen = candidate['requires_candidate_generation']

            if requires_gen:
                try:
                    combinations = suggest_mandatory_column_candidates(
                        candidate_df, method="rf", params=params, verbose=verbose
                    )
                except Exception as e:
                    if verbose:
                        print(f"  Suggest failed: {e}")
                    combinations = []
                for (t, c, a) in combinations:
                    if t in candidate_df.columns and c in candidate_df.columns and a in candidate_df.columns:
                        res_list = event_combinations(
                            candidate_df, [a], [c], train_features_path,
                            test_types=test_types, test_options=test_options, verbose=False,
                            timestamp_candidates=[t] if t else [],
                            vote=vote
                        )
                        if res_list and res_list[0].get('validation_info', {}).get('passed', False):
                            valid_combinations.append({'timestamp': t, 'case': c, 'activity': a})
                            has_valid = True
            else:
                case_col = candidate.get('case_col')
                activity_col = candidate.get('activity_col')
                timestamp_col = candidate.get('timestamp_col')
                if not all([case_col, activity_col, timestamp_col]):
                    continue
                if case_col not in candidate_df.columns or activity_col not in candidate_df.columns or timestamp_col not in candidate_df.columns:
                    continue
                res_list = event_combinations(
                    candidate_df, [activity_col], [case_col], train_features_path,
                    test_types=test_types, test_options=test_options, verbose=False,
                    timestamp_candidates=[timestamp_col] if timestamp_col else [],
                    vote=vote
                )
                if res_list and res_list[0].get('validation_info', {}).get('passed', False):
                    valid_combinations.append({'timestamp': timestamp_col, 'case': case_col, 'activity': activity_col})
                    has_valid = True

        return has_valid, valid_combinations

    return False, []