from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sp500_analysis.config.settings import settings


class TemporalDataSplitter:
    """Temporal data splitter implementing step_7_0_train_models.py zone logic."""

    def __init__(self, date_col: str = 'date') -> None:
        self.date_col = date_col

    def split_temporal_zones(
        self, 
        df: pd.DataFrame,
        zone_a_ratio: float = 0.6,
        zone_b_ratio: float = 0.2,
        zone_c_ratio: float = 0.2
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into 3 temporal zones (A, B, C) following step_7_0_train_models.py logic.
        
        Args:
            df: Input dataframe with date column and target
            zone_a_ratio: Proportion for Zone A (training)
            zone_b_ratio: Proportion for Zone B (validation)
            zone_c_ratio: Proportion for Zone C (holdout)
            
        Returns:
            Dictionary with (X, y) tuples for each zone
        """
        # Validate ratios
        if abs(zone_a_ratio + zone_b_ratio + zone_c_ratio - 1.0) > 1e-6:
            raise ValueError("Zone ratios must sum to 1.0")

        # Ensure data is sorted by date
        df_sorted = df.copy()
        if self.date_col in df_sorted.columns:
            df_sorted = df_sorted.sort_values(self.date_col)
            logging.info(f"Data sorted by {self.date_col}")
        else:
            logging.warning(f"Date column '{self.date_col}' not found, assuming data is already sorted")

        # Get target column (last column)
        target_col = df_sorted.columns[-1]
        logging.info(f"Target column identified: {target_col}")

        # Remove rows with NaN targets
        initial_size = len(df_sorted)
        df_clean = df_sorted.dropna(subset=[target_col])
        final_size = len(df_clean)
        
        if final_size < initial_size:
            logging.info(f"Removed {initial_size - final_size} rows with NaN targets")

        if final_size == 0:
            raise ValueError("No valid data after removing NaN targets")

        # Calculate split indices
        n_total = len(df_clean)
        n_zone_a = int(n_total * zone_a_ratio)
        n_zone_b = int(n_total * zone_b_ratio)
        n_zone_c = n_total - n_zone_a - n_zone_b  # Ensure exact split

        logging.info(f"Total samples: {n_total}")
        logging.info(f"Zone A (training): {n_zone_a} samples ({zone_a_ratio:.1%})")
        logging.info(f"Zone B (validation): {n_zone_b} samples ({zone_b_ratio:.1%})")
        logging.info(f"Zone C (holdout): {n_zone_c} samples ({zone_c_ratio:.1%})")

        # Split data
        zone_a_data = df_clean.iloc[:n_zone_a]
        zone_b_data = df_clean.iloc[n_zone_a:n_zone_a + n_zone_b]
        zone_c_data = df_clean.iloc[n_zone_a + n_zone_b:]

        # Prepare features and targets for each zone
        feature_cols = [col for col in df_clean.columns if col != target_col]
        
        # Zone A
        X_zone_a = zone_a_data[feature_cols].copy()
        y_zone_a = zone_a_data[target_col].copy()
        
        # Zone B
        X_zone_b = zone_b_data[feature_cols].copy()
        y_zone_b = zone_b_data[target_col].copy()
        
        # Zone C
        X_zone_c = zone_c_data[feature_cols].copy()
        y_zone_c = zone_c_data[target_col].copy()

        # Log date ranges for each zone
        if self.date_col in df_clean.columns:
            self._log_zone_date_ranges(zone_a_data, zone_b_data, zone_c_data)

        # Validate zones
        self._validate_zones(X_zone_a, y_zone_a, X_zone_b, y_zone_b, X_zone_c, y_zone_c)

        return {
            'zone_A': (X_zone_a, y_zone_a),
            'zone_B': (X_zone_b, y_zone_b),
            'zone_C': (X_zone_c, y_zone_c)
        }

    def _log_zone_date_ranges(
        self, 
        zone_a_data: pd.DataFrame, 
        zone_b_data: pd.DataFrame, 
        zone_c_data: pd.DataFrame
    ) -> None:
        """Log date ranges for each zone."""
        try:
            # Zone A dates
            zone_a_start = zone_a_data[self.date_col].min()
            zone_a_end = zone_a_data[self.date_col].max()
            
            # Zone B dates
            zone_b_start = zone_b_data[self.date_col].min()
            zone_b_end = zone_b_data[self.date_col].max()
            
            # Zone C dates
            zone_c_start = zone_c_data[self.date_col].min()
            zone_c_end = zone_c_data[self.date_col].max()
            
            logging.info("📅 Zone date ranges:")
            logging.info(f"   Zone A: {zone_a_start} to {zone_a_end}")
            logging.info(f"   Zone B: {zone_b_start} to {zone_b_end}")
            logging.info(f"   Zone C: {zone_c_start} to {zone_c_end}")
            
        except Exception as e:
            logging.warning(f"Could not log date ranges: {e}")

    def _validate_zones(
        self, 
        X_zone_a: pd.DataFrame, y_zone_a: pd.Series,
        X_zone_b: pd.DataFrame, y_zone_b: pd.Series,
        X_zone_c: pd.DataFrame, y_zone_c: pd.Series
    ) -> None:
        """Validate that zones are properly formed."""
        
        # Check that all zones have data
        zones = [
            ("A", X_zone_a, y_zone_a),
            ("B", X_zone_b, y_zone_b),
            ("C", X_zone_c, y_zone_c)
        ]
        
        for zone_name, X_zone, y_zone in zones:
            if len(X_zone) == 0 or len(y_zone) == 0:
                raise ValueError(f"Zone {zone_name} is empty")
            
            if len(X_zone) != len(y_zone):
                raise ValueError(f"Zone {zone_name} feature/target length mismatch")
            
            # Check for target variability
            if y_zone.nunique() <= 1:
                logging.warning(f"⚠️ Zone {zone_name} has low target variability ({y_zone.nunique()} unique values)")
            
            # Check for NaN values
            nan_features = X_zone.isnull().sum().sum()
            nan_targets = y_zone.isnull().sum()
            
            if nan_features > 0:
                logging.warning(f"⚠️ Zone {zone_name} has {nan_features} NaN values in features")
            
            if nan_targets > 0:
                logging.warning(f"⚠️ Zone {zone_name} has {nan_targets} NaN values in target")

        # Check feature consistency across zones
        feature_cols_a = set(X_zone_a.columns)
        feature_cols_b = set(X_zone_b.columns)
        feature_cols_c = set(X_zone_c.columns)
        
        if not (feature_cols_a == feature_cols_b == feature_cols_c):
            raise ValueError("Feature columns are not consistent across zones")
        
        logging.info("✅ All zones validated successfully")

    def get_combined_training_data(
        self, 
        zones: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Combine Zone A and Zone B for final model training.
        
        Args:
            zones: Dictionary of zone data
            
        Returns:
            Combined training features and target
        """
        X_zone_a, y_zone_a = zones['zone_A']
        X_zone_b, y_zone_b = zones['zone_B']
        
        # Combine zones A and B
        X_combined = pd.concat([X_zone_a, X_zone_b], axis=0, ignore_index=True)
        y_combined = pd.concat([y_zone_a, y_zone_b], axis=0, ignore_index=True)
        
        logging.info(f"Combined training data: {X_combined.shape}")
        return X_combined, y_combined 