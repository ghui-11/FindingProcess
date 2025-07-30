import pandas as pd
import numpy as np
import difflib

def entropy(arr):
    vals, counts = np.unique(arr, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob + 1e-9))

def normalize(series):
    if series.max() == series.min(): return series * 0
    return (series - series.min()) / (series.max() - series.min())

def avg_levenshtein(strings):
    strings = [str(s) for s in set(strings) if pd.notnull(s)]
    if len(strings) < 2: return 1.0
    total, count = 0, 0
    for i, s1 in enumerate(strings):
        for s2 in strings[i+1:]:
            total += difflib.SequenceMatcher(None, s1, s2).ratio()
            count += 1
    return total / count if count > 0 else 0

def detect_mixed_types(series, sample_size=50):
    samples = series.dropna().astype(str).head(sample_size)
    types = set()
    for v in samples:
        try:
            fv = float(v)
            types.add('int' if fv.is_integer() else 'float')
        except: types.add('str')
    return len(types) > 1

def field_screening(df, max_word_threshold=6, attribute_blacklist=None):
    attribute_blacklist = attribute_blacklist or []
    candidates, N = [], len(df)
    for col in df.columns:
        if col in attribute_blacklist: continue
        dtype = df[col].dtype
        is_string = dtype == object or pd.api.types.is_string_dtype(dtype)
        is_int = pd.api.types.is_integer_dtype(dtype)
        is_category = isinstance(dtype, pd.CategoricalDtype)
        if not (is_string or is_int or is_category): continue
        n_notna = df[col].notna().sum()
        if N == 0 or n_notna / N < 0.8: continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 2: continue
        if detect_mixed_types(df[col]): continue
        if is_string:
            n_long = (df[col].astype(str).apply(lambda x: len(x.split())) > max_word_threshold).sum()
            if n_long / N >= 0.1: continue
        candidates.append(col)
    return candidates

def field_scoring(df, candidates):
    rows, N = [], len(df)
    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        n_unique_ratio = n_unique / N
        vc = df[col].value_counts(dropna=True)
        max_freq = vc.iloc[0] / N
        ent = entropy(df[col].dropna().values)
        sim = avg_levenshtein(df[col].dropna().unique())
        rows.append({
            "field": col,
            "n_unique_ratio": n_unique_ratio,
            "max_freq": max_freq,
            "entropy": ent,
            "sim": sim
        })
    if not rows: return pd.DataFrame([]), [], []
    scored = pd.DataFrame(rows)
    scored["n_unique_ratio_norm"] = normalize(scored["n_unique_ratio"])
    scored["max_freq_norm"] = normalize(scored["max_freq"])
    scored["entropy_norm"] = normalize(scored["entropy"])
    scored["sim_norm"] = normalize(scored["sim"])
    scored["case_score"] = (
        0.4 * scored["n_unique_ratio_norm"]
        + 0.2 * (1 - scored["max_freq_norm"])
        + 0.2 * scored["entropy_norm"]
        + 0.2 * (1 - scored["sim_norm"])
    )
    scored["act_score"] = (
        0.3 * (1 - abs(scored["n_unique_ratio_norm"] - 0.1))
        + 0.2 * scored["max_freq_norm"]
        + 0.2 * (1 - scored["entropy_norm"])
        + 0.3 * scored["sim_norm"]
    )
    case_top = scored.sort_values("case_score", ascending=False).head(2)
    act_top = scored.sort_values("act_score", ascending=False).head(2)
    case_fields = list(case_top["field"])
    act_fields = list(act_top["field"])
    return scored, case_fields, act_fields