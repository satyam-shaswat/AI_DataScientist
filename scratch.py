import pandas as pd
from src.analysis.quality import run_quality_checks

df = pd.DataFrame({
    "id": range(100),
    "value": [10 + i % 5 for i in range(99)] + [500],
    "code": [f"C{i % 60}" for i in range(100)],
})

for f in run_quality_checks(df):
    print(f"[{f['severity']}] {f['message']}")