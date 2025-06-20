#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Processor with Business Days Index and Date Verification
Loads raw data from multiple sources, validates date ranges, and combines into a single dataset.

Usage:
    python ts_processor_with_date_verification.py --inventory path/to/inventory.xlsx --outdir path/to/output
"""

import os
import sys
import logging
from datetime import datetime
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BDay

# Constants
DEFAULT_INVENTORY = Path("../pipelines/full_consolidated_inventory.xlsx")
DEFAULT_OUTDIR = Path("../data/1_preprocess_ts")
IS_NOTEBOOK = 'ipykernel' in sys.modules

# Configure logging
log_file = os.path.join(settings.log_dir, f"arima_step0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
configurar_logging(log_file)
log = logging.getLogger(__name__)

# Utility function for exception handling
def catch_exceptions(func):
    """Decorator to catch and log exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Extract information
            context = ""
            if args and hasattr(args[0], '__class__'):
                context = f" in {args[0].__class__.__name__}"
            log.error(f"Error in {func.__name__}{context}: {str(e)}", exc_info=True)
            return None
    return wrapper

@dataclass
class ColumnConfig:
    """Configuration for a dataset column."""
    name: str
    source_path: Path
    source_column: str

def is_future_date(date, threshold_days=30):
    """Check if a date is too far in the future (beyond reasonable expectations)."""
    if pd.isnull(date):
        return False
    
    today = pd.Timestamp.now()
    threshold = today + pd.Timedelta(days=threshold_days)
    
    return date > threshold

def detect_frequency(dates: pd.DatetimeIndex) -> str:
    """
    Detects the frequency of a time series based on date differences.
    
    Args:
        dates: A pandas DatetimeIndex
        
    Returns:
        String indicating the detected frequency
    """
    if len(dates) < 2:
        return "desconocida (insuficientes datos)"
    
    # Calculate differences between consecutive dates in days
    diff_days = dates.to_series().diff().dt.days.dropna()
    
    # Count the occurrences of each difference
    diff_counts = Counter(diff_days)
    
    # Get the most common difference and its count
    if diff_counts:
        most_common_diff, most_common_count = diff_counts.most_common(1)[0]
        
        # Calculate percentage of this difference
        percentage = most_common_count / len(diff_days) * 100
        
        # Determine frequency based on most common difference
        if 0 <= most_common_diff <= 1:
            return f"diaria ({percentage:.1f}% de diferencias = {most_common_diff} día(s))"
        elif 12 <= most_common_diff <= 16:
            return f"quincenal ({percentage:.1f}% de diferencias = {most_common_diff} días)"
        elif 28 <= most_common_diff <= 31:
            return f"mensual ({percentage:.1f}% de diferencias = {most_common_diff} días)"
        elif 59 <= most_common_diff <= 62:
            return f"bimensual ({percentage:.1f}% de diferencias = {most_common_diff} días)"
        elif 88 <= most_common_diff <= 93:
            return f"trimestral ({percentage:.1f}% de diferencias = {most_common_diff} días)"
        else:
            # Calculate average difference for irregular series
            avg_diff = diff_days.mean()
            return f"irregular (diferencia media = {avg_diff:.1f} días)"
    else:
        return "desconocida (no se pueden calcular diferencias)"

def create_business_day_index(start_date, end_date, instrument):
    """Create business day index suitable for the instrument."""
    try:
        # Verify dates first
        current_date = pd.Timestamp.now().normalize()
        
        # Check if end date is too far in the future
        if end_date > current_date + pd.Timedelta(days=30):
            log.warning(f"⚠️ END DATE ({end_date.strftime('%Y-%m-%d')}) FOR {instrument} IS MORE THAN 30 DAYS IN THE FUTURE!")
            log.warning(f"⚠️ Adjusting end date from {end_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
            end_date = current_date
        
        # Add a buffer of a few days on each end
        buffer_days = 5
        start_with_buffer = start_date - pd.Timedelta(days=buffer_days)
        end_with_buffer = end_date + pd.Timedelta(days=buffer_days)
        
        # Different approach based on instrument type
        if instrument in ['SP500', 'SPY', 'NDX', 'DJIA']:
            # Use NYSE calendar for stock indices
            calendar = mcal.get_calendar('NYSE')
            schedule = calendar.schedule(start_date=start_with_buffer, end_date=end_with_buffer)
            business_days = pd.DatetimeIndex(schedule.index)
            log.info(f"Created business day index from NYSE calendar with {len(business_days)} trading days")
        elif any(fx in instrument for fx in ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD']):
            # For forex, use standard business days (Monday-Friday)
            # Create daily range
            all_days = pd.date_range(start=start_with_buffer, end=end_with_buffer, freq='D')
            # Filter to only weekdays (Monday=0, Friday=4)
            business_days = all_days[all_days.weekday.isin([0, 1, 2, 3, 4])]
            log.info(f"Created business day index for FOREX (weekdays) with {len(business_days)} trading days")
        else:
            # Default to standard business days
            all_days = pd.date_range(start=start_with_buffer, end=end_with_buffer, freq='B')
            business_days = all_days
            log.info(f"Created standard business day index with {len(business_days)} trading days")
        
        return business_days
        
    except Exception as e:
        log.warning(f"Error creating business day index: {str(e)}. Using standard business days.")
        # Fallback to standard business days
        try:
            all_days = pd.date_range(start=start_date, end=end_date, freq='B')
            log.info(f"Created fallback business day index with {len(all_days)} days")
            return all_days
        except Exception as fallback_error:
            log.error(f"Failed creating fallback index: {str(fallback_error)}. Proceeding with original dates.")
            return None

def parse_dates_with_specific_formats(df, date_column, source_file):
    """
    Parse dates with specific formats based on the source file.
    Handles problematic files with ambiguous date formats.
    """
    original_dates = df[date_column].copy()
    today = pd.Timestamp.now()
    max_reasonable_date = today + pd.DateOffset(months=3)
    
    log.info(f"Parsing dates for file: {source_file.name}")
    
    # Mostrar ejemplos de fechas para diagnóstico
    if len(original_dates) > 0:
        log.info(f"Sample date values from {source_file.name}:")
        for i, val in enumerate(original_dates.head(3).values):
            log.info(f"  Sample {i+1}: '{val}'")
    
    # Lógica específica para archivos con fechas en formato "Mes Día, Año (Período de referencia)"
    if 'US_ISM_Manufacturing.xlsx' in str(source_file) or 'Eurozone_Unemployment_Rate.xlsx' in str(source_file):
        log.info(f"Detected special date format in {source_file.name} with reference period in parentheses")
        
        # Función para extraer la fecha de publicación (antes del paréntesis)
        def extract_main_date(date_str):
            if pd.isna(date_str) or not isinstance(date_str, str):
                return None
            
            # Extraer la parte antes del paréntesis
            parts = date_str.split('(')
            if len(parts) > 1:
                main_date = parts[0].strip()
                return main_date
            return date_str
        
        # Aplicar extracción de la fecha principal
        main_dates = df[date_column].apply(extract_main_date)
        
        # Convertir a datetime usando el formato MMM DD, YYYY
        try:
            df[date_column] = pd.to_datetime(main_dates, format='%b %d, %Y', errors='coerce')
            log.info(f"Successfully parsed dates from {source_file.name} using format '%b %d, %Y'")
        except Exception:
            try:
                # Intentar con nombres de mes completos
                df[date_column] = pd.to_datetime(main_dates, format='%B %d, %Y', errors='coerce')
                log.info(f"Successfully parsed dates from {source_file.name} using format '%B %d, %Y'")
            except Exception as e:
                log.warning(f"Failed to parse dates from {source_file.name}: {str(e)}")
                # Último intento con detección automática
                df[date_column] = pd.to_datetime(main_dates, errors='coerce')
    
    # Lógica específica por archivo basada en los ejemplos proporcionados
    elif 'US_10Y_Treasury.csv' in str(source_file):
        log.info(f"Detected US_10Y_Treasury.csv - Using ISO format (YYYY-MM-DD)")
        df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d', errors='coerce')
        
    elif 'ECB_Deposit_Rate.csv' in str(source_file):
        log.info(f"Detected ECB_Deposit_Rate.csv - Using ISO format (YYYY-MM-DD)")
        df[date_column] = pd.to_datetime(df[date_column], format='%Y-%m-%d', errors='coerce')
        
    elif 'BOJ_Policy_Rate.xlsx' in str(source_file):
        log.info(f"Detected BOJ_Policy_Rate.xlsx - Using text month format (MMM DD, YYYY)")
        # Este formato maneja "Mar 18, 2025" correctamente
        df[date_column] = pd.to_datetime(df[date_column], format='%b %d, %Y', errors='coerce')
        
        # Si falla, intentar con mes completo
        if df[date_column].isna().mean() > 0.3:
            log.info("Trying alternative format with full month name")
            df[date_column] = pd.to_datetime(df[date_column], format='%B %d, %Y', errors='coerce')
    else:
        # Para otros archivos, usar detección automática
        formats_to_try = [
            ('%Y-%m-%d', 'ISO (YYYY-MM-DD)'),   # ISO format
            ('%m/%d/%Y', 'American (MM/DD/YYYY)'),  # American format
            ('%d/%m/%Y', 'European (DD/MM/YYYY)'),  # European format
            ('%d.%m.%Y', 'European (DD.MM.YYYY)'),  # European format with dots
            ('%b %d, %Y', 'Month name (MMM DD, YYYY)')  # Text month format
        ]
        
        for fmt, fmt_name in formats_to_try:
            try:
                tmp = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
                # Si la conversión fue exitosa (menos de 30% NaTs), usar este formato
                if tmp.isna().mean() < 0.3:
                    df[date_column] = tmp
                    log.info(f"Auto-detected format for {source_file.name}: {fmt_name}")
                    break
            except Exception:
                continue
        
        # Si ningún formato específico funciona, intentar con dayfirst=True
        if df[date_column].isna().mean() > 0.3:
            tmp = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
            if tmp.isna().mean() < 0.3:
                df[date_column] = tmp
                log.info(f"Auto-detected European format (DD/MM/YYYY) for {source_file.name}")
                
        # Como último recurso, usar inferencia automática
        if df[date_column].isna().mean() > 0.3:
            log.info(f"Trying automatic date inference for {source_file.name}")
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', infer_datetime_format=True)
    
    # Verificar si la conversión fue exitosa
    success_rate = 1 - df[date_column].isna().mean() if len(df) > 0 else 0
    log.info(f"Date conversion success rate for {source_file.name}: {success_rate:.2%}")
    
    if success_rate < 0.7 and len(df) > 0:
        log.warning(f"⚠️ Low success rate for date conversion in {source_file.name}")
        failed_dates = df[df[date_column].isna()]
        if not failed_dates.empty:
            log.warning(f"First few date values that could not be converted:")
            for i, (idx, val) in enumerate(original_dates[df[date_column].isna()].head(3).items()):
                log.warning(f"  Could not convert: '{val}'")
    
    # CORRECCIÓN CRÍTICA: Asegurarnos de que solo comparamos con valores datetime, no strings
    try:
        if len(df) > 0 and success_rate > 0:
            # Forzar la conversión a datetime
            if not pd.api.types.is_datetime64_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                
            # Ahora es seguro filtrar porque estamos seguros que todos los valores son timestamps
            mask = ~df[date_column].isna() & (df[date_column] > max_reasonable_date)
            future_dates = df[mask]
            
            if not future_dates.empty:
                future_pct = len(future_dates) / len(df[~df[date_column].isna()]) * 100
                log.warning(f"⚠️ Found {len(future_dates)} dates ({future_pct:.1f}%) beyond {max_reasonable_date.strftime('%Y-%m-%d')} in {source_file.name}")
                
                # Mostrar ejemplos de fechas problemáticas
                log.warning(f"Sample future dates (original → converted):")
                for i, (idx, row) in enumerate(future_dates.head(3).iterrows()):
                    if i < len(original_dates):
                        original = original_dates.iloc[idx] if idx < len(original_dates) else "unknown"
                        log.warning(f"  {original} → {row[date_column].strftime('%Y-%m-%d')}")
    except Exception as e:
        log.warning(f"Error checking future dates: {str(e)}")
    
    # Final check for remaining issues
    if len(df) > 0 and df[date_column].isna().mean() > 0.3:
        log.warning(f"⚠️ Failed to parse more than 30% of dates in {source_file.name}. Check the date format.")
    
    return df

class FileReader:
    """Reads various file formats and finds columns with flexible matching."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data = None
        self._date_patterns = [
            "date", "fecha", "time", "timestamp", "observation_date", 
            "release_date", "day", "mes", "año", "year", "month"
        ]
    
    @catch_exceptions
    def read(self):
        """Reads the file with automatic format detection."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        log.info(f"Reading file: {self.file_path}")
        
        suffix = self.file_path.suffix.lower()
        
        if suffix == ".csv":
            # Try different encodings and separators
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for sep in [',', ';', '\t', '|']:
                    try:
                        self.data = pd.read_csv(self.file_path, encoding=encoding, 
                                              sep=sep, dtype=str)  # Read as string first
                        
                        # If we have at least 2 columns and rows, probably correct format
                        if self.data.shape[1] >= 2 and self.data.shape[0] > 10:
                            log.info(f"CSV read successfully with encoding={encoding}, sep='{sep}'")
                            break
                    except Exception:
                        continue
                
                if self.data is not None:
                    break
            
            # If still None, try the default method
            if self.data is None:
                self.data = pd.read_csv(self.file_path)
                
        elif suffix in (".xls", ".xlsx"):
            try:
                # Tratamiento específico para archivos con formatos de fecha complejos
                if 'US_ISM_Manufacturing.xlsx' in str(self.file_path) or 'Eurozone_Unemployment_Rate.xlsx' in str(self.file_path):
                    log.info(f"Using specialized Excel reader for {self.file_path.name}")
                    try:
                        # Convertir todas las columnas a string para un procesamiento posterior más fácil
                        self.data = pd.read_excel(
                            self.file_path, 
                            engine="openpyxl", 
                            dtype=str,  # Todo como string para evitar conversiones automáticas
                            na_filter=False  # No filtrar NA automáticamente
                        )
                    except Exception as e:
                        log.warning(f"Specialized read failed: {str(e)}, trying standard method")
                        self.data = pd.read_excel(self.file_path, engine="openpyxl")
                else:
                    # Para otros archivos Excel, usar método estándar
                    self.data = pd.read_excel(self.file_path, engine="openpyxl")
            except Exception:
                try:
                    self.data = pd.read_excel(self.file_path, engine="xlrd")
                except Exception:
                    self.data = pd.read_excel(self.file_path, engine="openpyxl", sheet_name=0)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        log.info(f"File read successfully: {self.file_path.name} "
                f"({self.data.shape[0]} rows x {self.data.shape[1]} columns)")
        
        return self.data
    
    def find_column(self, target_col: str, patterns=None) -> Optional[str]:
        """Find matching column with flexible name matching."""
        if self.data is None:
            self.read()
            
        # Exact match
        if target_col in self.data.columns:
            return target_col
            
        # Case-insensitive match
        for col in self.data.columns:
            if col.lower() == target_col.lower():
                log.info(f"Column '{target_col}' found as '{col}' (case-insensitive)")
                return col
                
        # Pattern-based matching
        if patterns:
            for pattern in patterns:
                for col in self.data.columns:
                    if pattern.lower() in col.lower():
                        log.info(f"Similar column to '{target_col}' found as '{col}' (pattern: {pattern})")
                        return col
        
        # Partial matching
        normalized_target = target_col.lower()
        for col in self.data.columns:
            normalized_col = col.lower()
            if normalized_target in normalized_col or normalized_col in normalized_target:
                log.info(f"Similar column to '{target_col}' found as '{col}' (partial match)")
                return col
        
        log.warning(f"Column '{target_col}' not found. Available columns: {list(self.data.columns)}")
        return None
    
    def find_date_column(self) -> Optional[str]:
        """Find the date column in the DataFrame."""
        return self.find_column("date", self._date_patterns)
    
    def get_column(self, target_col: str, rename_to=None) -> pd.DataFrame:
        """Get a specific column with flexible detection."""
        if self.data is None:
            self.read()
            
        col_name = self.find_column(target_col)
        if col_name is None:
            # Try to use the date column as a fallback
            if target_col.lower() in self._date_patterns:
                col_name = self.find_date_column()
                if col_name:
                    log.info(f"Using '{col_name}' as fallback for date column")
                    
            if col_name is None:
                raise ValueError(f"Could not find column similar to '{target_col}'")
        
        # Get the column and make a copy
        result = self.data[[col_name]].copy()
        
        # Convert to datetime if it's the date column
        if col_name == self.find_date_column():
            # Use our specialized date parsing function to handle problematic formats
            result = parse_dates_with_specific_formats(result, col_name, self.file_path)
        # For non-date columns, try to convert numeric
        elif not col_name.lower().startswith('id'):
            try:
                # First try direct conversion
                numeric_result = pd.to_numeric(result[col_name], errors='coerce')
                # If less than 30% nulls, use the numeric version
                if numeric_result.isna().mean() < 0.3:
                    result[col_name] = numeric_result
                    
            except Exception as e:
                log.warning(f"Error converting to numeric: {str(e)}")
                    
        # Rename if needed
        if rename_to and rename_to != col_name:
            result = result.rename(columns={col_name: rename_to})
            
        return result

class SimpleDatasetProcessor:
    """Processes and builds a single combined dataset from raw data."""
    
    def __init__(self, outdir: Path):
        """Initialize the processor."""
        self.outdir = outdir
        
    @catch_exceptions
    def load_column(self, config: ColumnConfig) -> Tuple[pd.DataFrame, str]:
        """Load a column from a file with the given configuration."""
        if not config.source_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(".") / config.source_path.name,
                Path("..") / config.source_path.name,
                Path("../data") / config.source_path.name,
                Path("./data") / config.source_path.name
            ]
            
            for path in alt_paths:
                if path.exists():
                    log.info(f"Using alternative path for {config.name}: {path}")
                    config.source_path = path
                    break
            
            if not config.source_path.exists():
                raise FileNotFoundError(f"Could not find file: {config.source_path}")
        
        # Read the file and get the requested column
        reader = FileReader(config.source_path)
        df = reader.get_column(config.source_column, config.name)
        
        # If it's a date column, make sure it's a proper datetime
        date_col = None
        if config.name.lower() in ['date', 'fecha', 'timestamp']:
            if not pd.api.types.is_datetime64_dtype(df[config.name]):
                df[config.name] = pd.to_datetime(df[config.name], errors='coerce')
            date_col = config.name
        elif 'date' not in df.columns:
            # Try to get date from the original file
            file_date_col = reader.find_date_column()
            if file_date_col:
                date_df = reader.get_column(file_date_col, 'date')
                if not pd.api.types.is_datetime64_dtype(date_df['date']):
                    date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')
                df = pd.concat([date_df, df], axis=1)
                date_col = 'date'
        
        return df, date_col
    
    def print_date_summary(self, file_date_ranges, instrument):
        """Print a summary of date ranges for all sources."""
        today = pd.Timestamp.now().normalize()
        log.info(f"\n{'='*80}")
        log.info(f"📊 DATE RANGE SUMMARY FOR {instrument}")
        log.info(f"{'='*80}")
        log.info(f"{'SOURCE':<30} | {'MIN DATE':<12} | {'MAX DATE':<12} | {'FUTURE?':<7} | {'ROWS':<7}")
        log.info(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}")
        
        future_files = []
        for name, info in file_date_ranges.items():
            min_date_str = info['min_date'].strftime('%Y-%m-%d') if not pd.isnull(info['min_date']) else 'N/A'
            max_date_str = info['max_date'].strftime('%Y-%m-%d') if not pd.isnull(info['max_date']) else 'N/A'
            
            # Check if max date is in the future
            is_future = not pd.isnull(info['max_date']) and info['max_date'] > today
            future_mark = "⚠️ YES" if is_future else "NO"
            
            if is_future:
                future_files.append((name, info['max_date'], info['source_file']))
            
            log.info(f"{name:<30} | {min_date_str:<12} | {max_date_str:<12} | {future_mark:<7} | {info['count']:<7}")
        
        log.info(f"{'='*80}")
        
        # Warn about future dates
        if future_files:
            log.warning(f"\n⚠️ DETECTED {len(future_files)} FILES WITH FUTURE DATES:")
            for name, date, source_file in future_files:
                days_in_future = (date - today).days
                log.warning(f"  - {name} ({source_file}): {date.strftime('%Y-%m-%d')} ({days_in_future} days in the future)")
            log.warning("  These future dates may be incorrect or represent projections/forecasts.")
            log.warning("  Consider truncating the dataset to current date only.")
        
        return future_files
    
    @catch_exceptions
    def process_instrument(self, inventory_rows: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """Process all columns for a specific instrument."""
        log.info(f"Processing data for instrument: {instrument}")
        
        # Create column configs
        configs = []
        for _, row in inventory_rows.iterrows():
            # Skip derived/lag columns and volume columns
            column_name = row.get('canonical_name', '')
            if pd.isna(column_name):
                continue
            
            # Skip volume columns
            if any(vol_term in column_name.lower() for vol_term in ['volume', 'vol', 'volumen']):
                log.info(f"Skipping volume column: {column_name}")
                continue
                
            # Skip lag columns
            if 'lag' in column_name.lower():
                continue
                
            # Get source path
            source_path = row.get('absolute_path', '')
            if pd.isna(source_path) or not isinstance(source_path, (str, os.PathLike)):
                continue
                
            configs.append(ColumnConfig(
                name=column_name,
                source_path=Path(source_path),
                source_column=row.get('raw_column', '')
            ))
        
        log.info(f"Found {len(configs)} raw data columns for {instrument} (excluding volume columns)")
        
        # Process each column
        dfs = []
        date_cols = []
        
        # Dictionary to store min/max dates for each source file
        file_date_ranges = {}
        
        for config in configs:
            try:
                df, date_col = self.load_column(config)
                if df is not None:
                    # Log min/max dates for this file
                    if date_col and date_col in df.columns:
                        min_date = df[date_col].min()
                        max_date = df[date_col].max()
                        
                        # Store date range
                        file_date_ranges[config.name] = {
                            'min_date': min_date,
                            'max_date': max_date,
                            'count': len(df),
                            'source_file': config.source_path.name
                        }
                    
                    dfs.append(df)
                    if date_col:
                        date_cols.append(date_col)
            except Exception as e:
                log.error(f"Error loading column {config.name}: {str(e)}")
        
        if not dfs:
            log.error(f"No data could be loaded for {instrument}")
            return None
        
        # Print summary of date ranges
        future_files = self.print_date_summary(file_date_ranges, instrument)
        
        # Merge all dataframes
        log.info(f"Merging {len(dfs)} dataframes for {instrument}")
        
        # First find and prepare dataframes with date column
        merged = None
        
        try:
            # Find dataframes with date columns
            date_dfs = []
            for df in dfs:
                if any(col in df.columns for col in date_cols):
                    # Get the date column
                    date_col = next(col for col in df.columns if col in date_cols)
                    
                    # Set date as index
                    df_temp = df.copy()
                    df_temp = df_temp.set_index(date_col)
                    
                    # Remove duplicates in the index
                    if df_temp.index.duplicated().any():
                        df_temp = df_temp[~df_temp.index.duplicated(keep='first')]
                    
                    # Sort by date
                    df_temp = df_temp.sort_index()
                    date_dfs.append(df_temp)
            
            # Combine all date dataframes
            if date_dfs:
                merged = pd.concat(date_dfs, axis=1, join='outer')
                log.info(f"Initial merged dataframe: {merged.shape[0]} rows x {merged.shape[1]} columns")
            else:
                log.warning("No dataframes with valid date columns found")
                return None
                
        except Exception as e:
            log.error(f"Error merging dataframes: {str(e)}")
            return None
        
        # Log original min/max dates
        original_min_date = merged.index.min()
        original_max_date = merged.index.max()
        log.info(f"Original date range for {instrument}: {original_min_date} to {original_max_date}")
        
        # If we have future dates, trim to current date
        today = pd.Timestamp.now().normalize()
        if original_max_date > today:
            log.warning(f"⚠️ Trimming dataset to current date: {today.strftime('%Y-%m-%d')}")
            # Keep only dates up to today
            merged = merged[merged.index <= today]
            log.info(f"New date range after trimming future dates: {merged.index.min()} to {merged.index.max()}")
        
        # Create business day index
        business_days = create_business_day_index(merged.index.min(), merged.index.max(), instrument)
        
        if business_days is not None:
            # Create DataFrame with business day index
            full_index_df = pd.DataFrame(index=business_days)
            
            # Perform outer join to keep all dates
            merged = merged.join(full_index_df, how='outer')
            merged = merged.sort_index()
            
            # Log reconstructed date range
            new_min_date = merged.index.min()
            new_max_date = merged.index.max()
            new_days_count = len(merged)
            
            log.info(f"Reconstructed date range after business days merge: {new_min_date} to {new_max_date}")
            log.info(f"Total days in merged dataset: {new_days_count} days")
            
            # SOLUCIÓN: Recortar el dataframe a la fecha real máxima
            # Identificar la última fecha con datos reales (donde no todos son NaN)
            original_max_date = merged[~merged.isna().all(axis=1)].index.max()
            log.info(f"Last date with real data: {original_max_date}")
            
            # Truncar el dataframe a esa fecha para eliminar filas más allá de los datos reales
            merged = merged[merged.index <= original_max_date]
            log.info(f"Truncated dataframe to real data max date: {len(merged)} rows")
        else:
            log.warning(f"Using original dates for {instrument} without business day alignment.")
        
        # Fill missing values using only forward fill
        log.info("Filling missing values using ONLY forward fill (no backward fill)")
        
        for col in merged.columns:
            null_count = merged[col].isna().sum()
            if null_count > 0:
                null_pct = null_count / len(merged) * 100
                log.info(f"Filling {null_count} nulls ({null_pct:.1f}%) in '{col}' with FFILL only")
                
                # Apply only forward fill
                merged[col] = merged[col].ffill()  # Using ffill() instead of fillna(method='ffill')
                
                # Check remaining nulls
                remaining_nulls = merged[col].isna().sum()
                if remaining_nulls > 0:
                    log.warning(f"{remaining_nulls} null values remain in '{col}' after forward fill")
        
        # Make sure the index is a proper datetime index
        if not isinstance(merged.index, pd.DatetimeIndex):
            try:
                merged.index = pd.to_datetime(merged.index)
                merged = merged.sort_index()
            except Exception as e:
                log.warning(f"Error converting index to datetime: {str(e)}")
        
        log.info(f"Final dataset for {instrument}: {merged.shape[0]} rows, {merged.shape[1]} columns")
        return merged
    
    @catch_exceptions
    def save_dataset(self, df: pd.DataFrame, instrument: str) -> Path:
        """Save the dataset to a CSV file."""
        # Create output directory
        outdir = self.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Full path to output file
        outfile = outdir / f"{instrument}_business_days.csv"
        
        try:
            # Reset index to have date as a column
            df_reset = df.reset_index()
            
            # Make sure date column is properly formatted
            if 'date' in df_reset.columns and pd.api.types.is_datetime64_dtype(df_reset['date']):
                df_reset['date'] = df_reset['date'].dt.strftime('%Y-%m-%d')
            elif df_reset.index.name and df_reset.index.name.lower() in ['date', 'fecha', 'timestamp']:
                date_col = df_reset.index.name
                df_reset[date_col] = df_reset[date_col].dt.strftime('%Y-%m-%d')
            
            # Put date column first
            if 'date' in df_reset.columns:
                cols = ['date'] + [c for c in df_reset.columns if c != 'date']
                df_reset = df_reset[cols]
            
            # Save to CSV
            df_reset.to_csv(outfile, index=False)
            log.info(f"Dataset saved: {outfile} ({len(df):,} rows, {len(df.columns)} columns)")
            
            return outfile
        except Exception as e:
            log.error(f"Error saving dataset: {str(e)}")
            raise

def process_inventory(inventory_path: Path, outdir: Path) -> Dict:
    """Process the inventory and create datasets."""
    # Load inventory
    log.info(f"Loading inventory: {inventory_path}")
    
    try:
        # Try different engines
        try:
            inv = pd.read_excel(inventory_path, engine="openpyxl")
        except Exception:
            try:
                inv = pd.read_excel(inventory_path, engine="xlrd")
            except Exception:
                inv = pd.read_excel(inventory_path, engine="openpyxl", sheet_name=0)
    except Exception as e:
        log.error(f"Error loading inventory: {str(e)}")
        return {"status": "error", "message": f"Could not load inventory: {str(e)}"}
    
    log.info(f"Inventory loaded: {len(inv)} rows")
    
    # Create processor
    processor = SimpleDatasetProcessor(outdir)
    
    # Process each instrument
    instruments = inv["instrument"].unique()
    log.info(f"Processing {len(instruments)} instruments")
    
    results = {
        "processed": [],
        "failed": [],
        "total_instruments": len(instruments)
    }
    
    for instrument in instruments:
        try:
            # Get rows for this instrument
            instrument_rows = inv[inv["instrument"] == instrument]
            
            # Process the instrument
            df = processor.process_instrument(instrument_rows, instrument)
            
            if df is not None:
                # Save the dataset
                outfile = processor.save_dataset(df, instrument)
                
                results["processed"].append({
                    "instrument": instrument,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "date_range": {
                        "min_date": df.index.min().strftime('%Y-%m-%d'),
                        "max_date": df.index.max().strftime('%Y-%m-%d')
                    },
                    "file": str(outfile)
                })
            else:
                results["failed"].append({
                    "instrument": instrument,
                    "reason": "Processing returned None"
                })
        except Exception as e:
            log.error(f"Error processing instrument {instrument}: {str(e)}")
            results["failed"].append({
                "instrument": instrument,
                "reason": str(e)
            })
    
    # Save summary
    summary = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inventory": str(inventory_path),
        "output_dir": str(outdir),
        "total_instruments": len(instruments),
        "successful": len(results["processed"]),
        "failed": len(results["failed"])
    }
    
    summary_file = outdir / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"Summary saved to {summary_file}")
    log.info(f"Processing complete: {len(results['processed'])} successful, {len(results['failed'])} failed")
    
    return results

def main():
    """Main function."""
    log.info("Time Series Processor with Date Verification")
    
    # If running from notebook, use default paths
    if IS_NOTEBOOK:
        log.info("Running from notebook with default configuration")
        inventory_path = DEFAULT_INVENTORY
        outdir = DEFAULT_OUTDIR
    else:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Time Series Processor with Date Verification")
        parser.add_argument("--inventory", type=Path, required=True,
                          help="Path to inventory Excel file")
        parser.add_argument("--outdir", type=Path, default=Path("../data/1_preprocess_ts"),
                          help="Directory to save processed datasets")
        args = parser.parse_args()
        
        inventory_path = args.inventory
        outdir = args.outdir
    
    # Process the inventory
    process_inventory(inventory_path, outdir)

if __name__ == "__main__":
    main()