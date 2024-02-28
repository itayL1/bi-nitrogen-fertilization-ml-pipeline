from pathlib import Path

import pandas as pd


def load_Nitrogen_with_Era5_and_NDVI_dataset() -> pd.DataFrame:
    current_file_path = Path(__file__)
    dataset_csv_file_path = current_file_path.parent / '../assets/Nitrogen_with_Era5_and_NDVI.xlsx - Sheet1.csv'
    return pd.read_csv(dataset_csv_file_path, low_memory=False)
