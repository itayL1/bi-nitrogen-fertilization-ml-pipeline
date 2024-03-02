import pandas as pd


def concat_dataframes_horizontally(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    num_rows = dataframes[0].shape[0]
    if not all(df.shape[0] == num_rows for df in dataframes):
        raise ValueError("all dataframes must have the same number of rows for horizontal concatenation")

    result_df = pd.concat(dataframes, axis=1)
    return result_df


def validate_dataframe_has_column(df: pd.DataFrame, column: str) -> None:
    assert column in df, f"the column '{column}' is missing in the input dataframe"
