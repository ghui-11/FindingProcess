import pandas as pd
import os
import re
from typing import Optional
from dateutil import parser as date_parser

DEFAULT_SAMPLE_SIZE = 5000

class EventTableDetector:
    def __init__(self, filepath: str, sample_size: int = DEFAULT_SAMPLE_SIZE):
        self.filepath = filepath
        self.sample_size = sample_size
        self.df: Optional[pd.DataFrame] = None
        self.timestamp_columns = []
        self.valid_type_columns = []

    def load_csv(self) -> bool:
        try:
            self.df = pd.read_csv(self.filepath, nrows=self.sample_size)
            print(f"[INFO] Loaded {self.filepath} with shape {self.df.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load {self.filepath}: {str(e)}")
            return False
        return True

    def is_numeric_string(self, s: str) -> bool:
        pattern = r'^-?\d+(\.\d+)?$'
        return re.match(pattern, s) is not None

    def is_datetime_like(self, value: str) -> bool:
        try:
            if self.is_numeric_string(value):
                return False
            _ = date_parser.parse(value)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def detect_timestamp_column(self) -> bool:
        if self.df is None:
            return False

        self.timestamp_columns = []
        for col in self.df.columns:
            # Get the first 100 non-empty rows
            sample_values = self.df[col].dropna().astype(str).head(100)
            if len(sample_values) == 0:
                continue
            # Check which values can be recognized as timestamps)
            valid_count = sum(self.is_datetime_like(val) for val in sample_values)

            # If more than 90% timestamp format, the column is considered a timestamp column
            if valid_count >= 0.9 * len(sample_values):
                self.timestamp_columns.append(col)

        if not self.timestamp_columns:
            print(f"[FAIL] {os.path.basename(self.filepath)} - No timestamp column detected")
            return False

        print(f"[INFO] Detected timestamp columns: {self.timestamp_columns}")
        return True

    def detect_case_or_activity_column(self) -> bool:
        if self.df is None:
            return False

        self.valid_type_columns = []

        # Exclude timestamp columns
        timestamp_cols = set(self.timestamp_columns) if hasattr(self, 'timestamp_columns') else set()

        for col in self.df.columns:
            if col in timestamp_cols:
                continue

            dtype = self.df[col].dtype
            if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
                self.valid_type_columns.append(col)

        if not self.valid_type_columns:
            print(f"[FAIL] {os.path.basename(self.filepath)} - No string or integer column detected (no case/activity)")
            return False

        print(f"[INFO] Candidate case/activity columns: {self.valid_type_columns}")
        return True

    def is_potential_event_table(self) -> bool:
        if not self.load_csv():
            return False
        if not self.detect_timestamp_column():
            return False
        if not self.detect_case_or_activity_column():
            return False
        print(f"[PASS] {os.path.basename(self.filepath)} - Potential event table detected ✅")
        return True



