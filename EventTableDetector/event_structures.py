import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp
from sklearn.utils import resample
import matplotlib.pyplot as plt

def wide_to_long(df, timestamp_cols, case_col):
    long_df = (
        df.melt(id_vars=[case_col], value_vars=timestamp_cols,
                var_name='act', value_name='timestamp')
        .dropna(subset=['timestamp'])
        .rename(columns={case_col: 'case'})
    )
    return long_df


def extract_ca_features(df, case_col, act_col):
    total_event_count = len(df)
    unique_case_count = df[case_col].nunique()
    unique_activity_count = df[act_col].nunique()
    avg_events_per_case = df.groupby(case_col).size().mean()
    variants = df.groupby(case_col)[act_col].apply(tuple)
    variant_count = variants.nunique()
    avg_unique_acts_per_trace = variants.apply(lambda x: len(set(x))).mean()
    return [
        total_event_count,
        unique_case_count,
        unique_activity_count,
        avg_events_per_case,
        variant_count,
        avg_unique_acts_per_trace
    ]

def read_train_features(train_path):
    df = pd.read_csv(train_path)
    feature_matrix = df.iloc[:, 1:].values
    return feature_matrix

def ks_test(new_vector, train_matrix, alpha=0.05, feature_names=None):
    results = []
    print("\nKS检验各特征p值：")
    for i in range(train_matrix.shape[1]):
        train_col = train_matrix[:, i]
        new_val = new_vector[i]
        stat, pval = ks_2samp(train_col, [new_val])
        results.append((stat, pval))
        fname = feature_names[i] if feature_names else f"Feature {i+1}"
        print(f"{fname} KS p值: {pval:.4f}")
    ks_pass = all(pval > alpha for stat, pval in results)
    return ks_pass, results

def plot_ks_distributions(train_matrix, new_vector, feature_names=None):
    n_features = train_matrix.shape[1]
    fig, axs = plt.subplots(1, n_features, figsize=(4*n_features, 3))
    if n_features == 1:
        axs = [axs]
    for i in range(n_features):
        train_col = train_matrix[:, i]
        axs[i].hist(train_col, bins=10, alpha=0.7, label='Train', color='blue')
        axs[i].axvline(new_vector[i], color='red', linestyle='--', label='New sample')
        title = feature_names[i] if feature_names else f"Feature {i+1}"
        axs[i].set_title(title)
        axs[i].legend()
    plt.tight_layout()
    plt.show()

def bootstrap_test(new_vector, train_matrix, n_bootstrap=1000, alpha=0.05):
    # 警告训练集太小
    if train_matrix.shape[0] < 10:
        print(f"警告：训练集样本只有{train_matrix.shape[0]}，bootstrap不可靠！")
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
    results = []
    train_matrix = read_train_features(train_features_path)
    feature_names = [
        "total_event_count", "unique_case_count", "unique_activity_count",
        "avg_events_per_case", "variant_count", "avg_unique_acts_per_trace"
    ]
    for c in c_candidates:
        for a in a_candidates:
            if a == c:
                continue
            traces = df.groupby(c)[a].apply(list)
            valid_traces = traces[traces.apply(lambda x: len(x) > 2 and len(set(x)) > 1)]
            p = len(valid_traces) / len(traces) if len(traces) > 0 else 0
            if p < 0.5:
                continue
            sub_df = df[[c, a]].rename(columns={c: 'case', a: 'act'})
            features = extract_ca_features(sub_df, 'case', 'act')
            ks_ok, ks_detail = ks_test(features, train_matrix, feature_names=feature_names)
            boot_ok, boot_detail = bootstrap_test(features, train_matrix)
            print(f"Bootstrap检验通过: {boot_ok}")
            print("Bootstrap各特征区间和新值：")
            for idx, (ci_low, ci_up, new_val, pass_test) in enumerate(boot_detail):
                print(f"{feature_names[idx]}: [{ci_low:.2f}, {ci_up:.2f}], new={new_val:.2f}, pass={pass_test}")
            plot_ks_distributions(train_matrix, features, feature_names)

            mean_trace_len = valid_traces.apply(len).mean() if len(valid_traces) else 0
            mean_act_unique = valid_traces.apply(lambda x: len(set(x))).mean() if len(valid_traces) else 0
            seq_tuples = valid_traces.apply(tuple)
            nunique_seq = seq_tuples.nunique()
            subseq_counter = Counter()
            for seq in valid_traces:
                if len(seq) < 2: continue
                for l in range(2, min(5, len(seq)+1)):
                    for i in range(len(seq)-l+1):
                        subseq = tuple(seq[i:i+l])
                        subseq_counter[subseq] += 1
            top_subsequences = subseq_counter.most_common(5)
            has_frequent_subsequence = any(count >= 0.5 * nunique_seq for _, count in top_subsequences) if nunique_seq > 0 else False
            results.append({
                "case": c,
                "act": a,
                "valid_trace_percentage": p,
                "features": features,
                "ks_ok": ks_ok,
                "ks_detail": ks_detail,
                "boot_ok": boot_ok,
                "boot_detail": boot_detail,
                "mean_trace_length": mean_trace_len,
                "mean_unique_activity_per_trace": mean_act_unique,
                "unique_sequence_count": nunique_seq,
                "top_subsequences": top_subsequences,
                "has_frequent_subsequence": has_frequent_subsequence
            })
    # 输出结果
    for res in results:
        print(f"\nCase: {res['case']} | Act: {res['act']} | ValidTrace%: {res['valid_trace_percentage']:.2f}")
        print(f"  Features: {res['features']}")
        print(f"  KS检验通过: {res['ks_ok']} | bootstrap检验通过: {res['boot_ok']}")
        print(f"  Top Subsequences: {res['top_subsequences']}")
        print("-"*40)
    for res in results:
        if res['ks_ok'] and res['boot_ok'] and res['has_frequent_subsequence']:
            print(f"\n可用CA组合: case={res['case']}, act={res['act']}, features={res['features']}")
            return res
    print("\n无可行CA组合（KS/bootstrap未通过或无频繁序列）")
    return None