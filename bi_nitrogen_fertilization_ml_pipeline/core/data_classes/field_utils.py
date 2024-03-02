from typing import Optional

import pandas as pd
from pydantic import constr

not_empty_str = constr(min_length=1)


def validate_dataframe_has_2_dimensions(df: Optional[pd.DataFrame]) -> None:
    if df is not None and df.ndim != 2:
        raise ValueError("the dataframe must have exactly two dimensions")


def validate_percentage_str(value: str, validate_between_0_and_100: bool) -> None:
    if not value.endswith('%'):
        raise ValueError(f"the percentage string '{value}' must end with '%'")

    actual_percentage_value = float(value.rstrip('%'))
    if validate_between_0_and_100:
        if not 0 <= actual_percentage_value <= 100:
            raise ValueError(f"the percentage string '{value}' must be between 0 and 100")


def validate_percentage_distribution_dict(percentage_distribution: dict[str, str]) -> None:
    for perc_val in percentage_distribution.values():
        validate_percentage_str(perc_val, validate_between_0_and_100=True)
