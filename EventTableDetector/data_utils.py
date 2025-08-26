import pandas as pd
from dateutil.parser import parse as date_parse

def simple_read_csv(path, sep=',', nrows=None, **kwargs):
    """
    Reads a CSV file with basic error tolerance.
    Args:
        path (str): Path to CSV file.
        sep (str): Separator, default ','.
        nrows (int, optional): Number of rows to read.
        **kwargs: Other pandas read_csv arguments.
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_csv(path, sep=sep, nrows=nrows, on_bad_lines='skip', engine='python', **kwargs)

def is_unix_timestamp(s):
    """
    Checks if a string is a valid unix timestamp.
    Args:
        s (str): Input string.
    Returns:
        bool: True if valid unix timestamp, else False.
    """
    if not isinstance(s, str): s = str(s)
    if not s.isdigit() or len(s) not in (10, 13): return False
    ts = int(s)
    if len(s) == 10:  # seconds
        return 0 <= ts <= 4102444800
    elif len(s) == 13:  # milliseconds
        return 0 <= ts // 1000 <= 4102444800
    return False

def is_datetime_like_general(value):
    """
    Determines if a value is a general datetime string.
    Args:
        value (str): Input value.
    Returns:
        bool: True if value is datetime-like.
    """
    import pandas as pd
    from dateutil.parser import parse as date_parse
    if not isinstance(value, str): value = str(value)
    v = value.strip()
    if v == "" or v.lower() == "nan": return False
    if v.isdigit() and len(v) in (10,13):
        try:
            ts = int(v)
            if len(v)==10: dt = pd.to_datetime(ts, unit='s')
            else: dt = pd.to_datetime(ts // 1000, unit='s')
            return 1970 <= dt.year <= 2100
        except: return False
    if len(v)<6: return False
    date_sep_count = sum([sep in v for sep in ['-', '/', ':', '.', 'T', ' ']])
    if date_sep_count < 2: return False
    dt_formats = [
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y.%m.%d", "%Y%m%d",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z",
        "%y/%m/%d %H.%M", "%d/%m/%y %H.%M", "%y-%m-%d %H.%M", "%d-%m-%Y %H.%M"
    ]
    for fmt in dt_formats:
        try:
            _ = pd.to_datetime(v, format=fmt)
            return True
        except: continue
    try:
        _ = date_parse(v)
        return True
    except: return False

def detect_timestamp_columns_general(df, sample_size=20, threshold=0.5):
    """
    Detects timestamp columns in a dataframe by checking sample values.
    Args:
        df (pd.DataFrame): Input dataframe.
        sample_size (int): Number of samples per column.
        threshold (float): Ratio threshold to decide timestamp column.
    Returns:
        list: List of detected timestamp columns.
    """
    timestamp_cols = []
    for col in df.columns:
        samples = df[col].dropna().astype(str).head(sample_size)
        if len(samples) == 0:
            continue
        valid_count = sum(is_datetime_like_general(v) for v in samples)
        if valid_count / len(samples) >= threshold:
            timestamp_cols.append(col)
    return timestamp_cols