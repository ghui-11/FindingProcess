import os
import re
from scipy.stats import entropy
from .structure_detector import TableStructureType

class TableStructureAnalyzer:
    """
    Analyze a single table (LONG/WIDE) for key statistics and event-like features.
    """

    def __init__(self, detector):
        self.detector = detector
        self.results = {}

    @staticmethod
    def _value_pattern(series):
        """
        Infer a simple pattern for the column values (int/float/date/str).
        """
        patterns = set()
        for v in series.dropna().astype(str).unique():
            if re.match(r'^\d+$', v):
                patterns.add('int')
            elif re.match(r'^\d+\.\d+$', v):
                patterns.add('float')
            elif re.match(r'^\d{4}-\d{2}-\d{2}', v):
                patterns.add('date')
            else:
                patterns.add('str')
        return "|".join(sorted(patterns))

    @staticmethod
    def _calc_entropy(series):
        vc = series.value_counts(normalize=True)
        return entropy(vc, base=2)

    def analyze_long(self):
        df = self.detector.df
        stats = []
        for col in self.detector.valid_type_columns:
            ser = df[col].dropna()
            unique_count = ser.nunique()
            pat = self._value_pattern(ser)
            col_entropy = self._calc_entropy(ser)
            stats.append({
                'column': col,
                'unique_count': unique_count,
                'pattern': pat,
                'entropy': col_entropy
            })
        self.results['type'] = 'LONG'
        self.results['file'] = os.path.basename(self.detector.filepath)
        # self.results['stats'] = stats
        return self.results

    def analyze_wide(self):
        df = self.detector.df
        stats = []
        case_candidates = []
        for col in self.detector.valid_type_columns:
            ser = df[col].dropna()
            unique_ratio = ser.nunique() / len(ser) if len(ser) > 0 else 0
            pat = self._value_pattern(ser)
            stats.append({
                'column': col,
                'unique_ratio': unique_ratio,
                'pattern': pat
            })
            # Identify case candidate columns (unique_ratio >= 0.99 and single type)
            if unique_ratio >= 0.99 and len(pat.split('|')) == 1:
                case_candidates.append(col)
        event_candidate = len(case_candidates) > 0
        self.results['type'] = 'WIDE'
        self.results['file'] = os.path.basename(self.detector.filepath)
        # self.results['stats'] = stats
        self.results['is_event_candidate'] = event_candidate
        self.results['case_candidates'] = case_candidates

        # Print according to your requirements
        table_name = os.path.basename(self.detector.filepath)
        print(f"Table: {table_name}")
        print(f"Structure type: WIDE")
        print(f"Timestamp columns: {self.detector.timestamp_columns}")
        if case_candidates:
            print(f"Case candidate columns: {case_candidates}")
        else:
            print("No case candidate column detected (no column with unique_ratio >= 0.99 and single type)")

        return self.results

    @classmethod
    def analyze_single(cls, detector, structure_type):
        analyzer = cls(detector)
        if structure_type == TableStructureType.LONG:
            return analyzer.analyze_long()
        elif structure_type == TableStructureType.WIDE:
            return analyzer.analyze_wide()
        else:
            return {"type": "UNKNOWN", "file": os.path.basename(detector.filepath)}
