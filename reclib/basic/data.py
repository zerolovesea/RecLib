import pandas as pd

def get_column_data(data: dict | pd.DataFrame, name: str):
    if isinstance(data, dict):
        return data[name] if name in data else None
    elif isinstance(data, pd.DataFrame):
        if name not in data.columns:
            return None
        return data[name].values
    else:
        if hasattr(data, name):
            return getattr(data, name)
        raise KeyError(f"Unsupported data type for extracting column {name}")