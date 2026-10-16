import json
import pandas as pd
from src.ai.tools import call_tool

df = pd.DataFrame({
    "contract": ["Monthly"] * 4 + ["Yearly"] * 4,
    "churn":    ["Yes", "Yes", "Yes", "No", "No", "No", "No", "Yes"],
    "charges":  [70, 80, 90, 100, 20, 30, 40, 50],
    "tenure":   [1, 2, 3, 4, 40, 50, 60, 70],
    "visits":   [5, 6, 5, 7, 6, 5, 6, 60],
})

# 1. summary
r = call_tool(df, "get_dataset_summary")
print(1, r["rows"], r["columns"], r["numerical_columns"], r["duplicate_rows"])

# 2. churn split per contract
r = call_tool(df, "group_by_column", {"group_column": "contract", "value_column": "churn"})
for row in r["results"]:
    print(2, row)

# 3. mean charges per contract
r = call_tool(df, "group_by_column",
              {"group_column": "contract", "value_column": "charges", "aggregation": "mean"})
for row in r["results"]:
    print(3, row)
print(3, "highest:", r["highest_group"])

# 4. correlation
r = call_tool(df, "calculate_correlation", {"column1": "charges", "column2": "tenure"})
print(4, r["correlation"], r["strength"], r["direction"])

# 5. outliers
r = call_tool(df, "detect_outliers", {"column": "visits"})
print(5, r["lower_fence"], r["upper_fence"], r["outliers_found"], r["most_extreme_examples"])

# 6. the safety nets
print(6, call_tool(df, "group_by_column", {"group_column": "nope", "value_column": "churn"}))
print(6, call_tool(df, "delete_everything", {}))

# 7. results must be JSON-safe
json.dumps(call_tool(df, "get_column_statistics", {"column": "charges"}))
print(7, "JSON ok")