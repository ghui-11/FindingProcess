import pandas as pd
from typing import List, Optional, Tuple
from .utils import detect_true_event_timestamp_columns
import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

def select_primary_timestamp(timestamp_cols: List[str], df: pd.DataFrame, verbose: bool = False) -> Tuple[str, float]:
    """Pick best timestamp column by uniqueness/missing."""
    if not timestamp_cols:
        raise ValueError("No timestamp columns provided.")
    if len(timestamp_cols) == 1:
        if verbose: print(f"[primary_ts] Only one ts col: {timestamp_cols[0]}")
        return timestamp_cols[0], 0.0
    best_col, best_score = None, -float("inf")
    for col in timestamp_cols:
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        score = n_unique - 0.5 * n_missing
        if verbose: print(f"[primary_ts] {col}: unique={n_unique}, missing={n_missing}, score={score:.2f}")
        if score > best_score:
            best_col = col
            best_score = score
    if verbose: print(f"[primary_ts] Picked: {best_col}")
    return best_col, best_score

def detect_table_type(df: pd.DataFrame, verbose: bool = False) -> str:
    """Judge table format: long/wide/none."""
    timestamp_cols = detect_true_event_timestamp_columns(df, verbose=verbose)
    N = len(timestamp_cols)
    if verbose: print(f"[type] Timestamp cols: {timestamp_cols}")
    if N == 0:
        if verbose: print("[type] No timestamp columns found.")
        return 'No'
    if N <= 2:
        if verbose: print("[type] Judged as long table.")
        return 'long'
    for col in df.columns:
        if col not in timestamp_cols and is_caseid_candidate(df[col]):
            if df[col].is_unique:
                if verbose: print(f"[type] Wide (unique case id: {col})")
                return 'wide'
    if N >= 3:
        non_nan = [df[c].notna().sum() for c in timestamp_cols]
        total = len(df)
        std_ratio = pd.Series(non_nan).std() / (total + 1e-9)
        if all(cnt > 0.7 * total for cnt in non_nan) and std_ratio < 0.3:
            if verbose: print("[type] Wide (timestamps evenly filled).")
            return 'wide'
    if verbose: print("[type] Long table.")
    return 'long'

def wide_to_long(df: pd.DataFrame, timestamp_cols: List[str], case_col: str, verbose: bool = False) -> pd.DataFrame:
    """Wide->long; dedup by case/activity/timestamp."""
    long_df = df.melt(id_vars=[case_col], value_vars=timestamp_cols, var_name='activity', value_name='timestamp')
    long_df = long_df.dropna(subset=['timestamp'])
    long_df = long_df.rename(columns={case_col: 'case'})
    long_df = long_df.drop_duplicates(['case', 'activity', 'timestamp'])
    if verbose: print(f"[wide2long] Rows: {long_df.shape[0]}")
    return long_df

def ensure_long_format(df: pd.DataFrame, case_candidates=None, verbose: bool = False) -> pd.DataFrame:
    """Ensure long format; wide->long if needed."""
    table_type = detect_table_type(df, verbose=verbose)
    if table_type == 'No': raise ValueError("No ts columns found.")
    if table_type == 'wide':
        timestamp_cols = detect_true_event_timestamp_columns(df, verbose=verbose)
        candidates = ([col for col in df.columns if col not in timestamp_cols and is_caseid_candidate(df[col])]
                      if not case_candidates else case_candidates)
        if candidates:
            case_col = max(candidates, key=lambda c: df[c].nunique())
        else:
            id_like = [col for col in df.columns if 'id' in col.lower()]
            if id_like:
                case_col = max(id_like, key=lambda c: df[c].nunique() / len(df))
            else:
                raise ValueError("No case id for wide table.")
        long_df = wide_to_long(df, timestamp_cols, case_col, verbose=verbose)
        return long_df
    return df

def convert_to_event_log(df: pd.DataFrame, case: str, activity: str, timestamp: str, verbose: bool = False) -> pd.DataFrame:
    """
    Convert to standardized event log.
    """

    wanted_cols = [case, activity, timestamp]
    # All columns must exist and must be unique
    cols_exist = all(col in df.columns for col in wanted_cols)
    if cols_exist:
        long_df = df.copy()
        log = long_df[[case, activity, timestamp]].rename(
            columns={case: 'case:concept:name', activity: 'concept:name', timestamp: 'time:timestamp'}
        )
        if verbose:
            print(f"[eventlog] Used direct columns: {wanted_cols}, rows: {len(log)}")
        return log
    else:
        error_msg = f"[eventlog] Assignment is wrong: not all of case/activity/timestamp columns exist in table."
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)

def check_case_level_candidate_by_intervals(df, candidate_ts_cols, interval_threshold_seconds=120,
                                            identical_ratio_threshold=0.85, verbose=False) -> bool:
    """Check if timestamp columns can be case-melted (vectorized for performance, no overlap check)."""
    sample_size = min(500, len(df))
    df_sample = df.iloc[:sample_size].copy()

    interval_stats = []
    identical_count = 0

    for col in candidate_ts_cols:
        col_dt = pd.to_datetime(df_sample[col], errors='coerce')
        df_sample[f'_{col}_dt'] = col_dt

    for idx, row in df_sample.iterrows():
        times = [row[f'_{col}_dt'] for col in candidate_ts_cols if pd.notnull(row[f'_{col}_dt'])]
        if len(times) >= 2:
            times_sorted = sorted(times)
            diffs = [(times_sorted[i] - times_sorted[i - 1]).total_seconds() for i in range(1, len(times_sorted))]
            interval_stats.extend(diffs)
            if len(set(times_sorted)) == 1:
                identical_count += 1

    if not interval_stats:
        if verbose: print("[caselevel] No valid intervals found.")
        return False

    pct_short = sum(1 for d in interval_stats if d < interval_threshold_seconds) / len(interval_stats)
    pct_identical = identical_count / len(df_sample) if len(df_sample) > 0 else 0
    if verbose: print(
        f"[caselevel] Short ratio (<{interval_threshold_seconds}s): {pct_short:.2f}, Identical: {pct_identical:.2f}")
    return pct_short <= 0.8 and pct_identical <= identical_ratio_threshold


def melt_case_level_table(df: pd.DataFrame, timestamp_cols: list, verbose: bool = False):
    """
    For 3+ timestamp columns (same format), melt into event log:
    - case: input table index
    - activity: column name of timestamp
    - timestamp: value
    Remove duplicate events (case,activity,timestamp).
    """
    # Step 1: Add case column as index
    df_copy = df.copy()
    df_copy['case'] = df.index

    # Step 2: Melt wide to long format
    melted_df = df_copy.melt(
        id_vars=['case'], value_vars=timestamp_cols,
        var_name='activity', value_name='timestamp'
    )

    # Step 3: Drop invalid events
    melted_df = melted_df.dropna(subset=['timestamp'])

    # Step 4: Remove duplicates (case, activity, timestamp)
    melted_df = melted_df.drop_duplicates(['case', 'activity', 'timestamp'])

    if verbose:
        print(f"[melt] Melted rows: {len(melted_df)}, Unique cases: {melted_df['case'].nunique()}, Timestamp cols: {timestamp_cols}")

    # Standardize column order for downstream
    return melted_df[['case','activity','timestamp']], 'case', 'activity', 'timestamp'


def preprocess_and_generate_candidates(df: pd.DataFrame, verbose: bool = False) -> List[dict]:
    """Generate event table candidates. Overlap ratio check removed."""
    timestamp_cols = detect_true_event_timestamp_columns(df, verbose=verbose)
    if not timestamp_cols: raise ValueError("No ts columns detected.")
    candidates_list = []
    num_ts = len(timestamp_cols)
    if verbose: print(f"[preprocess] Timestamp cols: {timestamp_cols}")
    if num_ts <= 2:
        long_df = ensure_long_format(df, verbose=verbose)
        primary_ts, _ = select_primary_timestamp(timestamp_cols, long_df, verbose=verbose)
        candidates_list.append({
            'df': long_df,
            'timestamp_col': primary_ts,
            'path_type': 'event-level',
            'requires_candidate_generation': True
        })
        if verbose: print("[preprocess] Added event-level candidate")
    else:
        primary_ts, _ = select_primary_timestamp(timestamp_cols, df, verbose=verbose)
        candidates_list.append({
            'df': df,
            'timestamp_col': primary_ts,
            'path_type': 'event-level-multi',
            'requires_candidate_generation': True
        })
        interval_ok = check_case_level_candidate_by_intervals(df, timestamp_cols, verbose=verbose)
        if interval_ok:
            melted_df, case_col, activity_col, timestamp_col = melt_case_level_table(df, timestamp_cols, verbose=verbose)
            candidates_list.append({
                'df': melted_df,
                'case_col': case_col,
                'activity_col': activity_col,
                'timestamp_col': timestamp_col,
                'path_type': 'case-level-melt',
                'requires_candidate_generation': False
            })
            if verbose: print("[preprocess] Added case-level-melt candidate")
        else:
            if verbose:
                msg = "[preprocess] Case-level-melt skipped: "
                if not interval_ok:
                    print(msg + "interval diversity insufficient.")
    if verbose: print(f"[preprocess] Finished. Total candidates: {len(candidates_list)}")
    return candidates_list

def sort_event_log_by_timestamp(df, case_col='case', activity_col='activity', timestamp_col='timestamp', verbose=False) -> pd.DataFrame:
    """Sort and clean event log by timestamp."""
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[timestamp_col])
    df_sorted = df_copy.sort_values([case_col, timestamp_col])
    if verbose: print(f"[sortlog] Sorted: {len(df_sorted)} rows. Date range: {df_sorted[timestamp_col].min()} ~ {df_sorted[timestamp_col].max()}")
    return df_sorted


def search_data_platform(search_term, platform="kaggle", download_dir="datasets", topn=10):
    """
    Search and download CSV datasets from Kaggle or Zenodo.

    Args:
        search_term (str): Search keyword.
        platform (str): 'kaggle' or 'zenodo' (default: 'kaggle').
        download_dir (str): Directory to store downloaded files.
        topn (int): Maximum number of datasets to fetch.

    Returns:
        datasets (list): List of datasets.
    """
    os.makedirs(download_dir, exist_ok=True)

    if platform.lower() == "kaggle":
        return _search_kaggle(search_term, download_dir, topn)
    elif platform.lower() == "zenodo":
        return _search_zenodo(search_term, download_dir, topn)
    else:
        raise ValueError(f"Unsupported platform: {platform}. Use 'kaggle' or 'zenodo'.")


def _search_kaggle(search_term, download_dir, topn):
    """Search and download from Kaggle"""
    api = KaggleApi()
    api.authenticate()
    results = api.dataset_list(search=search_term, file_type="csv", sort_by="hottest")
    datasets = []

    for ds in results:
        ref = ds.ref
        files = api.dataset_list_files(ref).files
        csv_files = [f for f in files if f.name.lower().endswith('.csv')]

        if len(csv_files) != 1:
            continue

        csv_file = csv_files[0]
        ds_dir = os.path.join(download_dir, ref.replace('/', '__'))
        os.makedirs(ds_dir, exist_ok=True)

        try:
            api.dataset_download_file(ref, csv_file.name, path=ds_dir, force=True, quiet=True)
        except Exception as e:
            print(f"Failed to download {csv_file.name} from {ref}: {e}")
            continue

        csv_path = os.path.join(ds_dir, csv_file.name)

        datasets.append({
            "ref": ref,
            "csv_path": csv_path,
            "title": ds.title,
        })

        if len(datasets) >= topn:
            break

    return datasets


def _search_zenodo(search_term, download_dir, topn):
    """Search and download from Zenodo"""
    url = "https://zenodo.org/api/records"
    params = {
        'q': search_term,
        'sort': 'newest',
        'size': topn * 3,
        'type': 'dataset'
    }

    datasets = []
    count = 0

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"Zenodo search failed with status {r.status_code}")
            return datasets

        data = r.json()
        hits = data.get('hits', {}).get('hits', [])

        for hit in hits:
            if count >= topn:
                break

            record_id = hit['id']
            title = hit['metadata']['title']
            files = hit.get('files', [])

            # Filter CSV files
            csv_files = [f for f in files if f['key'].lower().endswith('.csv')]
            if len(csv_files) != 1:
                continue

            csv_file = csv_files[0]
            ds_dir = os.path.join(download_dir, f"zenodo__{record_id}")
            os.makedirs(ds_dir, exist_ok=True)

            download_url = csv_file['links']['self']
            local_path = os.path.join(ds_dir, csv_file['key'])

            try:
                print(f"Downloading:  {csv_file['key']} from Zenodo record {record_id}")
                r = requests.get(download_url, stream=True, timeout=120)
                if r.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    datasets.append({
                        "ref": f"zenodo__{record_id}",
                        "csv_path": local_path,
                        "title": title,
                    })
                    count += 1
                else:
                    print(f"Failed to download {csv_file['key']}:  HTTP {r.status_code}")
            except Exception as e:
                print(f"Error downloading {csv_file['key']}: {e}")

    except Exception as e:
        print(f"Zenodo search error: {e}")

    return datasets