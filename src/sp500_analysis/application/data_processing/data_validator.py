from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, NamedTuple
from sp500_analysis.config.settings import settings


class ValidationResult(NamedTuple):
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataValidator:
    """Data validator implementing step_7_0_train_models.py validation logic."""

    def __init__(self, date_col: str = 'date', min_samples: int = 100) -> None:
        self.date_col = date_col
        self.min_samples = min_samples

    def validate_temporal_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate temporal data following step_7_0_train_models.py logic.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []

        # Basic structure validation
        if df is None or df.empty:
            errors.append("DataFrame is None or empty")
            return ValidationResult(False, errors, warnings)

        # Minimum samples check
        if len(df) < self.min_samples:
            errors.append(f"Insufficient data: {len(df)} samples (minimum: {self.min_samples})")

        # Target column validation (last column)
        target_col = df.columns[-1]
        
        # Check for target validity
        if 'Target' not in target_col:
            warnings.append(f"Target column '{target_col}' doesn't contain 'Target' in name")

        # Check target values
        target_values = df[target_col].dropna()
        if len(target_values) == 0:
            errors.append("No valid target values found")
        elif target_values.nunique() <= 1:
            errors.append("All target values are equal (no variability)")
        elif target_values.nunique() < 10:
            warnings.append(f"Low target variability: only {target_values.nunique()} unique values")

        # Date column validation
        if self.date_col in df.columns:
            try:
                date_series = pd.to_datetime(df[self.date_col])
                if date_series.isnull().any():
                    warnings.append("Some date values could not be parsed")
            except Exception as e:
                warnings.append(f"Date column validation failed: {e}")
        else:
            warnings.append(f"Date column '{self.date_col}' not found")

        # Feature validation
        feature_cols = [col for col in df.columns if col != target_col and col != self.date_col]
        if len(feature_cols) == 0:
            errors.append("No feature columns found")
        
        # Check for excessive NaN values
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if nan_ratio > 0.5:
            warnings.append(f"High proportion of NaN values: {nan_ratio:.1%}")

        # Infinite values check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                warnings.append(f"Column '{col}' contains infinite values")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings) 