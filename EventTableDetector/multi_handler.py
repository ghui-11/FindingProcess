from .single_handler import TableStructureAnalyzer
from .structure_detector import TableStructureType

class MultiTableHandler:
    """
    Analyze multi-table (MULTI) structures, including shared columns, join candidates,
    and event-like evaluation across tables.
    """

    def __init__(self, detectors):
        self.detectors = detectors

    def find_shared_columns(self):
        """Return a list of columns shared by all tables."""
        shared = set(self.detectors[0].df.columns)
        for d in self.detectors[1:]:
            shared &= set(d.df.columns)
        return list(shared)

    def analyze(self):
        shared_columns = self.find_shared_columns()
        result = {
            "shared_columns": shared_columns,
            "join_candidate": bool(shared_columns),
            "subtables": []
        }
        if shared_columns:
            print(f"[MULTI] Can join tables on these columns: {shared_columns}")
            result["join_strategy"] = f"Join on {shared_columns}"
            # TODO: Extend with actual join/event logic if needed
        else:
            print("[MULTI] No shared columns. Analyze each table as WIDE or LONG.")
            for d in self.detectors:
                # Simple heuristic: more than 2 timestamp columns means wide, else long
                if len(d.timestamp_columns) > 2:
                    res = TableStructureAnalyzer.analyze_single(d, TableStructureType.WIDE)
                else:
                    res = TableStructureAnalyzer.analyze_single(d, TableStructureType.LONG)
                result["subtables"].append(res)
        return result