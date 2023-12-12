"""
Instructions:

Fill in the methods of the DataCleaner class to produce the same printed results
as in the comments below. Good luck, and have fun!
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame and create a history list for tracking changes."""
        self.current = df.copy()
        self.history = [("Initial df", self.current.copy())]

    def adjust_dtype(self, types: Dict[str, Any]) -> None:
        """Adjust the data types of DataFrame columns as specified in the 'types' dictionary."""
        for col, dtype in types.items():
            if dtype == np.datetime64:
                # Convert to datetime if np.datetime64 is specified
                self.current[col] = pd.to_datetime(self.current[col])
            else:
                # Adjust data type for other columns
                self.current[col] = self.current[col].astype(dtype)
        self.history.append((f"Adjusted dtypes using {types}", self.current.copy()))

    def impute_missing(self, columns: List[str]) -> None:
        """Impute missing values in specified columns with their mean value."""
        for col in columns:
            if self.current[col].isnull().any():
                mean_val = self.current[col].mean()
                self.current[col].fillna(mean_val, inplace=True)
        self.history.append((f"Imputed missing in {columns}", self.current.copy()))

    def revert(self, steps_back: int = 1) -> None:
        """Revert the DataFrame to a previous state by the specified number of steps."""
        if steps_back < len(self.history):
            self.current = self.history[-(steps_back + 1)][1].copy()
            self.history = self.history[:-(steps_back)]

    def save(self, path: str) -> None:
        """Serialize the DataCleaner object with pickle.dump and save it to a file."""
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> DataCleaner:
        """Deserialize a DataCleaner object with pickle.load from a file."""
        with open(path, 'rb') as file:
            return pickle.load(file)

# Usage example with a sample DataFrame
transactions = pd.DataFrame({
    "customer_id": [10, 10, 13, 10, 11, 11, 10],
    "amount": [1.00, 1.31, 20.5, 0.5, 0.2, 0.2, np.nan],
    "timestamp": ["2020-10-08 11:32:01", "2020-10-08 13:45:00", "2020-10-07 05:10:30",
                  "2020-10-08 12:30:00", "2020-10-07 01:29:33", "2020-10-08 13:45:00", 
                  "2020-10-09 02:05:21"]
})


transactions_dc = DataCleaner(transactions)

print(f"Current dataframe:\n{transactions_dc.current}")

# Current dataframe:
#    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21

print(f"Current dtypes:\n{transactions_dc.current.dtypes}")

# Initial dtypes:
# customer_id      int64
# amount         float64
# timestamp       object
# dtype: object

transactions_dc.adjust_dtype({"timestamp": np.datetime64})

print(f"Changed dtypes to:\n{transactions_dc.current.dtypes}")

# Changed dtypes to:
# customer_id             int64
# amount                float64
# timestamp      datetime64[ns]

transactions_dc.impute_missing(columns=["amount"])

print(f"Imputed missing as overall mean:\n{transactions_dc.current}")

# Imputed missing as mean:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

print(f"History of changes:\n{transactions_dc.history}")

# ** Any coherent structure with history of changes **
# E.g., here's one possibility

# History of changes:
# [('Initial df',    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21), ("Adjusted dtypes using {'timestamp': <class 'numpy.datetime64'>}",    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21), ("Imputed missing in ['amount']",    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21)]

transactions_dc.save("transactions")
loaded_dc = DataCleaner.load("transactions")
print(f"Loaded DataCleaner current df:\n{loaded_dc.current}")

# Loaded DataCleaner current df:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

transactions_dc.revert()
print(f"Reverting missing value imputation:\n{transactions_dc.current}")

# Reverting missing value imputation:
#    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21
