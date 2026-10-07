import pandas as pd
from src.data.loader import load_csv
from src.analysis.profiler import get_dataset_overview

df = load_csv("..\AI_DataScientist\data\Test data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Converted:", df.attrs["converted_columns"])
print("TotalCharges dtype:", df["TotalCharges"].dtype)
print("TotalCharges missing:", df["TotalCharges"].isna().sum())

overview = get_dataset_overview(df)
print("TotalCharges ->", overview["column_types"]["TotalCharges"])
print("customerID   ->", overview["column_types"]["customerID"])

df = pd.DataFrame({
    "id": range(1, 31),                               # 1..30, all unique integers
    "age": [20 + i % 10 for i in range(30)],          # 10 distinct values
    "city": ["Delhi", "Mumbai", "Pune"] * 10,         # 3 distinct values
    "salary": [30000.5 + i * 100 for i in range(30)], # unique decimals
    "joined": pd.date_range("2024-01-01", periods=30) # dates
})

overview = get_dataset_overview(df)
for key, value in overview.items():
    print(key, "->", value)