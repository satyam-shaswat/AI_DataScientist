import pandas as pd
from plotly import data


def load_csv(file)-> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    `file` can be a file path or an uploaded file object.
    Raises ValueError with a clear message if the file cannot be used.
    """
    try:
        data=pd.read_csv(file)
    except UnicodeDecodeError:
        if hasattr(file, "seek"):
            file.seek(0)
        try:
            data = pd.read_csv(file, encoding="latin-1")
        except Exception:
            raise ValueError("The file's text encoding could not be read.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file could not be parsed as a CSV.")

    if data.empty:
        raise ValueError("The file has column headers but no data rows.")

    data.columns = data.columns.str.strip()

    data, converted = convert_numeric_text_columns(data)
    data.attrs["converted_columns"] = converted

    return data


def convert_numeric_text_columns(data: pd.DataFrame):
    """Convert text columns that actually hold numbers (e.g. '29.85', ' ').

    A column is converted only if every non-blank value is a valid number.
    Blank values become missing values (NaN).
    Returns the new DataFrame and the list of converted column names.
    """
    data1 = data.copy()
    converted = []

    for col in data.columns:
        if data[col].dtype != "object":
            continue

        cleaned = data[col].str.strip()
        cleaned = cleaned.where(cleaned != "")        # blank -> missing
        numeric = pd.to_numeric(cleaned, errors="coerce")
        non_blank = cleaned.notna()

        if non_blank.any() and numeric[non_blank].notna().all():
            data1[col] = numeric
            converted.append(col)

    return data1, converted