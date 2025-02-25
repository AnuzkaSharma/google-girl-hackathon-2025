import pandas as pd

def load_data(file_path):
    """
    Loads a dataset from a CSV file.

    Parameters:
        file_path (str): Path to the dataset CSV.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded dataset from {file_path} with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
