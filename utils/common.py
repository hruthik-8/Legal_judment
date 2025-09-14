import os
import pandas as pd

def load_csv(path_env_key: str, default_path: str):
    path = os.getenv(path_env_key, default_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path} (set {path_env_key})")
    return pd.read_csv(path)
