import numpy as np
import pandas as pd
import os
from data_cleaner import DataCleaner  # Import the DataCleaner class from main file

# Test for __init__ method
def test_init():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    dc = DataCleaner(df)
    assert dc.current.equals(df), "Initialization failed: DataFrame not set correctly."
    assert len(dc.history) == 1, "Initialization failed: History not initialized correctly."

# Test for adjust_dtype method
def test_adjust_dtype():
    df = pd.DataFrame({'a': ['1', '2']})
    dc = DataCleaner(df)
    dc.adjust_dtype({'a': np.int64})
    assert dc.current['a'].dtype == np.int64, "adjust_dtype failed: Data type not adjusted correctly."

# Test for impute_missing method
def test_impute_missing():
    df = pd.DataFrame({'a': [1, np.nan, 3]})
    dc = DataCleaner(df)
    dc.impute_missing(['a'])
    assert dc.current['a'].isnull().sum() == 0, "impute_missing failed: Missing values not imputed correctly."

# Test for revert method
def test_revert():
    df = pd.DataFrame({'a': [1, 2]})
    dc = DataCleaner(df)
    dc.adjust_dtype({'a': np.float64})
    dc.revert()
    assert dc.current['a'].dtype != np.float64, "revert failed: DataFrame not reverted correctly."

# Test for save and load methods
def test_save_load():
    df = pd.DataFrame({'a': [1, 2]})
    dc = DataCleaner(df)
    filepath = "test_datacleaner.pkl"
    dc.save(filepath)
    assert os.path.exists(filepath), "save failed: File not saved correctly."

    loaded_dc = DataCleaner.load(filepath)
    assert loaded_dc.current.equals(dc.current), "load failed: Loaded DataFrame not matching."
    
    if os.path.exists(filepath):
        os.remove(filepath)

# Run all tests
if __name__ == '__main__':
    test_init()
    test_adjust_dtype()
    test_impute_missing()
    test_revert()
    test_save_load()
    print("All tests passed!")
