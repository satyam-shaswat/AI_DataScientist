import pandas as pd


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

    data.columns = data.columns.str.strip()  # Strip whitespace from column names
    return data
