# FindingProcess

FindingProcess is a method for automatically identifying event log from a table.
It helps to identify case id, activity and timestamp column and validate event logs. 

---

## Repository Structure

- `EventTableDetector/`  
  Main package. Contains detection, scoring, formatting, validation, and all API logic.
- `Train/TrainMatrix/`  
  Stores pre-trained column classifier models, statistical feature data (`train_features.csv`), 
- and learned classifiers used by the core package at runtime. 
---

## Dependencies

Make sure you have the following dependencies installed in your environment before using FindingProcess:

- **Python 3.7+**
- **pandas**
- **numpy**
- **scikit-learn**
- **joblib**
- **imbalanced-learn**
- **python-dateutil**
- **requests**
---
## How to Use
An example using the unified API:

```python
import pandas as pd
from EventTableDetector import api

# Search csv
api.search_data_platform('car',platform="kaggle", download_dir="datasets", topn=1)

# Read your raw data
df = pd.read_csv('your_input.csv')  

# Suggest main event log columns (timestamp, case, activity)
candidates = api.suggest_mandatory_column_candidates(df)
print(candidates)

# Validate the event log
valid, attributes = api.validate_event_log(df)
print("Result:", "Valid" if valid else "Invalid", "| Core attributes:", attributes)

# Try the first valid attribute assignment
t, c, a = attributes[0]['timestamp'], attributes[0]['case'], attributes[0]['activity']
metrics = api.get_event_log_statistic(df, case=c, activity=a, timestamp=t)
for k, v in metrics.items():
    print(f'{k}: {v}')

eventlog_df = api.convert_to_event_log(df, case=c, activity=a, timestamp=t)

```
