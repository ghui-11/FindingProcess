import os
from EventTableDetector.structure_detector import SubfolderTableGroup, TableStructureType

def scan_dataset_root(root_path: str):
    valid_groups = {}
    for entry in os.scandir(root_path):
        if entry.is_dir():
            group = SubfolderTableGroup(entry.path)
            group.detect_event_tables()
            if group.structure_type != TableStructureType.UNKNOWN:
                valid_groups[entry.name] = group
            else:
                print(f"[INFO] Folder {entry.name} discarded, no event tables detected")
    return valid_groups

if __name__ == "__main__":
    dataset_root = os.path.join(os.path.dirname(__file__), 'Dataset')
    detected_groups = scan_dataset_root(dataset_root)

    print(f"Detected {len(detected_groups)} valid event table groups:")
    for folder_name, group in detected_groups.items():
        print(f"- Folder: {folder_name}, Structure: {group.structure_type}, Tables: {[os.path.basename(f) for f in group.event_tables.keys()]}")