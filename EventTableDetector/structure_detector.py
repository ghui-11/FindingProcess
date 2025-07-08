import os
from enum import Enum
from typing import List, Dict
from .event_table_detector import EventTableDetector


class TableStructureType(Enum):
    LONG = "long"
    WIDE = "wide"
    MULTI = "multi"
    UNKNOWN = "unknown"


class SubfolderTableGroup:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.files = self._scan_csv_files()
        self.event_tables: Dict[str, EventTableDetector] = {}
        self.structure_type = TableStructureType.UNKNOWN

    def _scan_csv_files(self) -> List[str]:
        return [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]

    def detect_event_tables(self):
        temp_event_tables = {}

        # First check all files and temporarily save qualified ones
        for fpath in self.files:
            detector = EventTableDetector(fpath)
            if detector.is_potential_event_table():
                temp_event_tables[fpath] = detector

        # Discard unqualified files and keep qualified files
        self.event_tables = temp_event_tables

        # Determine the structure type based on the number of remaining files
        count = len(self.event_tables)
        if count == 0:
            self.structure_type = TableStructureType.UNKNOWN
        elif count > 1:
            self.structure_type = TableStructureType.MULTI
        else:
            # There is only one qualified file, determine wide (more than 2 timestamp columns) or long (otherwise) format
            detector = next(iter(self.event_tables.values()))
            if len(detector.timestamp_columns) > 2:
                self.structure_type = TableStructureType.WIDE
            else:
                self.structure_type = TableStructureType.LONG

