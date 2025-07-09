import os
from EventTableDetector.structure_detector import SubfolderTableGroup, TableStructureType
from EventTableDetector.single_handler import TableStructureAnalyzer
from EventTableDetector.multi_handler import MultiTableHandler

def process_table_by_structure(folder_path: str):
    group = SubfolderTableGroup(folder_path)
    group.detect_event_tables()
    if group.structure_type == TableStructureType.LONG:
        detector = next(iter(group.event_tables.values()))
        result = TableStructureAnalyzer.analyze_single(detector, TableStructureType.LONG)
        print(result)
    elif group.structure_type == TableStructureType.WIDE:
        detector = next(iter(group.event_tables.values()))
        result = TableStructureAnalyzer.analyze_single(detector, TableStructureType.WIDE)
        print(result)
    elif group.structure_type == TableStructureType.MULTI:
        detectors = list(group.event_tables.values())
        result = MultiTableHandler(detectors).analyze()
        print(result)
    else:
        print(f"[UNKNOWN] No valid event data detected in {folder_path}")

if __name__ == "__main__":
    dataset_root = os.path.join(os.path.dirname(__file__), 'Dataset')
    for entry in os.scandir(dataset_root):
        if entry.is_dir():
            print(f"\n[INFO] Scanning folder: {entry.name}")
            process_table_by_structure(entry.path)