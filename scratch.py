import io
from src.data.loader import load_csv

good =io.StringIO("Name, Age\nAlice,30\nBob,25\n")
data = load_csv(good)
print (data)
print ("columns: ", list(data.columns))

try:
    load_csv(good)
except ValueError as e:
    print("Empty file->", e)
    

try:
    load_csv(io.StringIO("Name,Age\n"))
except ValueError as e:
    print("No rows ->", e)