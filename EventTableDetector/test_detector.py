import os
from glob import glob
from event_table_detector import EventTableDetector

def test_event_table_detector_auto():
    # Step up from 'detector/' to root, then into 'dataset/'
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))
    csv_files = glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print(f"[WARNING] No CSV files found in {data_dir}")
        print(f"[DEBUG] Files in dataset/: {os.listdir(data_dir)}")
        return

    for file in csv_files:
        print("\n--- Checking:", file)
        detector = EventTableDetector(file)
        detector.load_csv()
        detector.is_potential_event_table()
if __name__ == "__main__":
    test_event_table_detector_auto()