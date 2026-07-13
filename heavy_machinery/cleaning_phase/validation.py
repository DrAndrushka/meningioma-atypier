"""Pandera schema validation for cleaned and imputed cohort frames.

Builds serializable checks from ``ColSpec`` (categories, dtypes, ranges) and validates
every handoff parquet — unimputed cohort after cleaning and each MICE draw before
multivariable modelling. Artifacts → ``output/cleaning/schema_validation.json``.
"""
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pandera.extensions as extensions
from pathlib import Path


#🟧🟧🟧 HELPERS
@extensions.register_check_method(statistics=["expected", "ordered"])
def validate_category(series: pd.Series, *, expected: tuple, ordered: bool = False) -> bool:
    """
    Validate categorical values and, when ordered=True, level order.
    Registered with Pandera so statistics serialize to JSON (no lambdas).
    """
    expected = tuple(expected)
    if not series.dropna().isin(expected).all():
        return False
    if ordered:
        return series.cat.ordered and tuple(series.cat.categories) == expected
    return True


def category_validation(expected: tuple, ordered: bool = False) -> tuple:
    """
    Build serializable Pandera checks for a categorical column.
    """
    return (pa.Check.validate_category(expected=tuple(expected), ordered=ordered),)


DEFAULT_SCHEMA_VALIDATION_PATH = Path("output/cleaning/schema_validation.json")


def load_schema_validation(
    path: str | Path = DEFAULT_SCHEMA_VALIDATION_PATH,
    ) -> pa.DataFrameSchema:
    """
    Load a Pandera schema saved by pandera_validate.

    Importing this module registers custom checks (e.g. validate_category)
    required for from_json round-trip.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Schema validation JSON not found: {path}. "
            "Run meningioma-cleaning.ipynb §14 first."
        )
    return pa.DataFrameSchema.from_json(path)


def validate_imputed_frames(
    schema_validation: pa.DataFrameSchema,
    imputed_frames: list[pd.DataFrame],
    ) -> None:
    """Pandera-validate every imputed draw before inferential modelling."""
    for frame in imputed_frames:
        schema_validation(frame, lazy=True)
    if len(imputed_frames) == 1:
        print("✅ Pandera validated imputed modelling cohort")
    else:
        print(f"✅ Pandera validated {len(imputed_frames)} MICE imputed draws")


def pandera_template(df: pd.DataFrame) -> dict:
    """
    Generate a pandera schema template for the given dataframe.
    """
    lines = ["schema_validation = pa.DataFrameSchema("]
    lines.append("    {")
    for col in df.columns:
        extras = []
        if pd.api.types.is_bool_dtype(df[col]):
            extras.append(f"dtype=\"boolean\"")
            extras.append(f"nullable=True,")
        elif pd.api.types.is_categorical_dtype(df[col]):
            extras.append(f"dtype=\"category\"")
            extras.append(f"nullable=True,")
            categories = tuple(df[col].cat.categories)
            extras.append(f"checks=[*category_validation({categories!r})],")
        elif pd.api.types.is_numeric_dtype(df[col]):
            extras.append(f"dtype=\"Float64\"")
            extras.append(f"nullable=True,")
            extras.append(f"checks=[\n                pa.Check.ge(-np.inf),\n                pa.Check.le(np.inf)\n                ],")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            extras.append(f"dtype=\"datetime64\"")
            extras.append(f"nullable=True,")
            extras.append(f"checks=[\n                pa.Check.ge(pd.Timestamp(\"2017-01-01\")),\n                pa.Check.le(pd.Timestamp.today())\n                ],")
        else:
            extras.append("dtype=str",)
            extras.append(f"nullable=True,")
            extras.append(f"unique=True,")

        extras_str = ("\n            " + "\n            ".join(extras)) if extras else ""
        lines.append(f'        {col!r}: pa.Column({extras_str}\n            ),')
    lines.append("  },")
    lines.append("  strict=True,")
    lines.append(")")
    lines.append("schema_validation(df, lazy=True).head()")    
    
    # Print the lines as a string (multiline)
    print("\n".join(lines))


def pandera_validate(
    schema_validation: pa.DataFrameSchema, 
    df: pd.DataFrame, 
    output_path: str = "output/cleaning/schema_validation.json"
    ) -> None:
    """
    Validate df against schema.
    If valid → save full schema to JSON file.
    If invalid → show all errors and stop.
    """
    # Zero-dep ANSI colors (works on your MacBook + most terminals)
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    try:
        schema_validation(df, lazy=True)
        
        # Validation passed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        schema_validation.to_json(output_path, indent=4)
        
        print(f"{GREEN}{BOLD}✅ VALIDATION PASSED — CLEAN DATA DETECTED{RESET}")
        print(f"{CYAN}   Your DataFrame conforms to the schema. Zero violations.{RESET}")
        print(f"{CYAN}   Schema artifact saved → {output_path}{RESET}")
        print(f"{GREEN}   Pipeline can proceed safely. Good shit, doc.{RESET}\n")
        
    except pa.errors.SchemaErrors as exc:
        print(f"{RED}{BOLD}❌ VALIDATION FAILED — DATA QUALITY ISSUES FOUND{RESET}")
        print(f"{YELLOW}   Pandera rejected the DataFrame. {len(exc.failure_cases)} failure case(s) detected.{RESET}")
        print(f"{RED}Error details:{RESET}")
        print(exc)

