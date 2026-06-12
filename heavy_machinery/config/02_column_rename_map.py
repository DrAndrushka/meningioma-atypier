"""Step 02 — rename columns before schema inference.

``list_cols`` prints a copy-paste ``COLUMN_RENAME_MAP`` skeleton for the notebook.
``apply_rename`` applies the map to the raw dataframe.
"""
from __future__ import annotations

import json

import pandas as pd


def list_cols(df_raw: pd.DataFrame) -> None:
    print("COLUMN_RENAME_MAP = {")
    for col in df_raw.columns:
        print(f"    {json.dumps(col, ensure_ascii=False)}: \"\",")
    print("}")


def apply_rename(df_raw: pd.DataFrame, column_rename_map: dict) -> pd.DataFrame:
    return df_raw.rename(columns=column_rename_map)
