import numpy as np
import pandas as pd
import pandera.pandas as pa

def pandera_template(df: pd.DataFrame) -> dict:
    """
    Generate a pandera schema template for the given dataframe.
    """
    return pandera_template(df)

def pandera_validate(df: pd.DataFrame, schema: dict) -> bool:
    """
    Validate the given dataframe against the given schema.
    """
    return pandera_validate(df, schema)