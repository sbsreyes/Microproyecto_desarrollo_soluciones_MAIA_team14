from typing import Optional, Tuple
import pandas as pd
from model.config.core import config

REQUIRED_COLUMNS = [
    "Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases",
    "Gender", "Category", "Location", "Size", "Color", "Season",
    "Shipping Type", "Discount Applied", "Payment Method",
    "Frequency of Purchases",
]

def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    return input_data.copy()

def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    errors = None
    missing = [c for c in REQUIRED_COLUMNS if c not in input_data.columns]
    if missing:
        errors = f"Columnas faltantes: {missing}"
        return input_data, errors
    validated = drop_na_inputs(input_data=input_data)
    if validated.empty:
        errors = "El DataFrame está vacío."
    return validated, errors