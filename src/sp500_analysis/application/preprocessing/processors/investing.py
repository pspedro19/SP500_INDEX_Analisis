from __future__ import annotations

import os
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from sp500_analysis.application.preprocessing.base import BaseProcessor
from sp500_analysis.shared.logging.logger import configurar_logging
from sp500_analysis.utils import PathManager


class InvestingProcessor(BaseProcessor):
    """Processor for economic data exported from Investing."""

    def __init__(
        self, config_file: str, data_root: str | Path = "data/0_raw", log_file: str = "myinvestingreportcp.log"
    ) -> None:
        """Create a new processor using *config_file* as source configuration."""
        super().__init__(data_root=data_root, log_file=log_file)
        self.config_file = config_file
        self.config_data: pd.DataFrame | None = None
        self.global_min_date: pd.Timestamp | None = None
        self.global_max_date: pd.Timestamp | None = None
        self.daily_index: pd.DataFrame | None = None
        self.processed_data: dict[str, pd.DataFrame | None] = {}
        self.final_df: pd.DataFrame | None = None
        self.stats: dict[str, dict] = {}
        self.date_cache: dict[str, bool] = {}

        # Initialize PathManager for efficient file discovery
        self.path_manager = PathManager(data_root)

        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PROCESO: InvestingProcessor")
        self.logger.info(f"Archivo de configuración: {config_file}")
        self.logger.info(f"Directorio raíz de datos: {data_root}")
        self.logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def _validate_input(self, data: object) -> bool:
        """Always return ``True`` as no validation is required."""
        return True

    def read_config(self) -> pd.DataFrame | None:
        """Load and filter the Excel configuration file."""
        try:
            self.logger.info("Leyendo archivo de configuración...")
            df_config = pd.read_excel(self.config_file)
            self.config_data = df_config[
                (df_config["Fuente"] == "Investing Data")
                & (df_config["Tipo de Preprocesamiento Según la Fuente"] == "Copiar y Pegar")
            ].copy()
            self.logger.info(f"Se encontraron {len(self.config_data)} configuraciones para procesar")
            return self.config_data
        except Exception as e:  # pragma: no cover - runtime behaviour
            self.logger.error(f"Error al leer configuración: {e}")
            return None

    def robust_parse_date(self, date_str: str, preferred_dayfirst: bool | None = None) -> pd.Timestamp | None:
        """Parse *date_str* trying several strategies."""
        if not isinstance(date_str, str):
            return None
        date_str = date_str.strip()
        if not date_str:
            return None

        m = re.search(r'([A-Za-z]+\s+\d{1,2},\s+\d{4})', date_str)
        if m:
            candidate = m.group(1)
            try:
                parsed = pd.to_datetime(candidate, errors='coerce')
                if parsed is not None:
                    return parsed
            except Exception as e:
                self.logger.warning(f"Error al parsear patrón en '{date_str}': {e}")

        if preferred_dayfirst is not None:
            try:
                parsed = pd.to_datetime(date_str, dayfirst=preferred_dayfirst, errors='coerce')
                threshold = pd.Timestamp.today() + pd.Timedelta(days=30)
                if parsed and parsed <= threshold:
                    return parsed
            except Exception as e:
                self.logger.warning(f"Error con dayfirst={preferred_dayfirst} en '{date_str}': {e}")

        try:
            parsed_true = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
            parsed_false = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
            threshold = pd.Timestamp.today() + pd.Timedelta(days=30)
            valid_true = parsed_true and parsed_true <= threshold
            valid_false = parsed_false and parsed_false <= threshold
            if valid_true and not valid_false:
                return parsed_true
            elif valid_false and not valid_true:
                return parsed_false
            elif valid_true and valid_false:
                return parsed_true  # Por defecto dayfirst=True
            else:
                return parsed_true if pd.notnull(parsed_true) else parsed_false
        except Exception as e:
            self.logger.warning(f"Error en robust_parse_date para '{date_str}': {e}")
            return None

    def _parse_dates(self, df: pd.DataFrame, ruta: str) -> pd.DataFrame:
        """Add a ``fecha`` column parsed from ``Release Date``."""
        if ruta not in self.date_cache:
            sample = df["Release Date"].dropna().head(10)
            count_true, count_false = 0, 0
            threshold = pd.Timestamp.today() + pd.Timedelta(days=30)
            for val in sample:
                dt_true = pd.to_datetime(val, dayfirst=True, errors="coerce")
                dt_false = pd.to_datetime(val, dayfirst=False, errors="coerce")
                if pd.notnull(dt_true) and dt_true <= threshold:
                    count_true += 1
                if pd.notnull(dt_false) and dt_false <= threshold:
                    count_false += 1
            preferred = count_true >= count_false
            self.date_cache[ruta] = preferred
            self.logger.info(f"Preferencia de dayfirst para {ruta}: {preferred}")
        else:
            preferred = self.date_cache[ruta]

        df["fecha"] = df["Release Date"].apply(lambda x: self.robust_parse_date(x, preferred_dayfirst=preferred))
        df = df.dropna(subset=["fecha"]).sort_values("fecha")
        return df

    def _convert_values(self, series: pd.Series) -> pd.Series:
        """Convert numeric strings with suffixes and symbols to floats."""

        def convertir_valor_robusto(val: object) -> float | None:
            if pd.isna(val) or val == "":
                return None

            val_str = str(val).strip()

            multiplicador = 1.0
            if val_str.endswith("B") or val_str.endswith("b"):
                multiplicador = 1e9
                val_str = val_str[:-1].strip()
            elif val_str.endswith("M") or val_str.endswith("m"):
                multiplicador = 1e6
                val_str = val_str[:-1].strip()
            elif val_str.endswith("K") or val_str.endswith("k"):
                multiplicador = 1e3
                val_str = val_str[:-1].strip()

            if val_str.endswith("%"):
                val_str = val_str[:-1].strip()
                multiplicador *= 0.01

            val_str = val_str.replace(",", ".")
            try:
                return float(val_str) * multiplicador
            except ValueError:
                return None

        return series.apply(convertir_valor_robusto)

    def _merge_daily(self, base: pd.DataFrame, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Merge ``df`` into ``base`` using an asof join and forward fill."""
        df = df.sort_values("fecha")
        daily = pd.merge_asof(base, df, on="fecha", direction="backward")
        daily[col_name] = daily[col_name].ffill()
        return base.merge(daily[["fecha", col_name]], on="fecha", how="left")

    def process_file(self, config_row: pd.Series) -> tuple[str, pd.DataFrame | None]:
        """Process a single Excel file defined in ``config_row``."""
        variable = config_row["Variable"]
        macro_type = config_row["Tipo Macro"]
        target_col = config_row["TARGET"]

        # Use PathManager for efficient file discovery
        filename = f"{variable}.xlsx"
        ruta = self.path_manager.find_file(filename)
        
        if ruta is None:
            self.logger.error(f"Archivo no encontrado: {filename}")
            
            # Suggest similar files
            suggestions = self.path_manager.suggest_similar_files(filename, max_suggestions=3)
            if suggestions:
                self.logger.info(f"Archivos similares encontrados: {', '.join(suggestions)}")
            return variable, None

        self.logger.info(f"\nProcesando: {variable} ({macro_type})")
        self.logger.info(f"- Archivo: {filename}")
        self.logger.info(f"- Columna TARGET: {target_col}")
        self.logger.info(f"- Ruta encontrada: {ruta}")
        
        # Get file info for additional logging
        file_info = self.path_manager.get_file_info(filename)
        if file_info:
            self.logger.debug(f"- Categoría: {file_info.category}")
            self.logger.debug(f"- Tamaño: {file_info.size_bytes / 1024:.2f} KB")

        try:
            if variable == "US_Leading_EconIndex":
                self.logger.info("Utilizando estrategia especial para US_Leading_EconIndex (header=2)")
                df = pd.read_excel(ruta, header=2, engine="openpyxl")
                df.columns = df.columns.str.strip()
                self.logger.info(f"Columnas leídas: {df.columns.tolist()}")
            else:
                df = pd.read_excel(ruta, engine="openpyxl")
        except Exception as e:  # pragma: no cover - runtime behaviour
            self.logger.error(f"Error al leer {ruta}: {e}")
            return variable, None

        self.logger.info(f"- Filas encontradas: {len(df)}")
        if 'Release Date' not in df.columns:
            self.logger.error(f"No se encontró la columna 'Release Date' en {ruta}")
            return variable, None

        df = self._parse_dates(df, ruta)

        # Si el target especificado no está, intenta buscar una alternativa
        if target_col not in df.columns:
            for col in df.columns:
                if col.strip().lower() == target_col.strip().lower():
                    target_col = col
                    self.logger.warning(f"No se encontró '{config_row['TARGET']}', se usará '{target_col}'")
                    break
        if target_col not in df.columns:
            self.logger.error(f"No se encontró columna TARGET ni alternativa en {ruta}")
            return variable, None

        df["valor"] = self._convert_values(df[target_col])
        df = df.dropna(subset=['valor'])

        if df.empty:
            self.logger.error(f"No se encontraron valores válidos para '{target_col}' en {ruta}")
            return variable, None

        # Actualizar rango global de fechas
        current_min = df['fecha'].min()
        current_max = df['fecha'].max()
        if self.global_min_date is None or current_min < self.global_min_date:
            self.global_min_date = current_min
        if self.global_max_date is None or current_max > self.global_max_date:
            self.global_max_date = current_max

        # Calcular cobertura (puedes ajustar la fórmula si lo deseas)
        cobertura = (len(df) / len(df)) * 100

        # RENOMBRAR LA COLUMNA: Crear un nombre único
        nuevo_nombre = f"{target_col}_{variable}_{macro_type}"
        df.rename(columns={'valor': nuevo_nombre}, inplace=True)
        self.stats[variable] = {
            'macro_type': macro_type,
            'target_column': target_col,
            'total_rows': len(df),
            'valid_values': len(df),
            'coverage': cobertura,
            'date_min': current_min,
            'date_max': current_max,
            'nuevo_nombre': nuevo_nombre,
        }
        self.logger.info(f"- Valores no nulos en TARGET: {len(df)}")
        self.logger.info(f"- Periodo: {current_min.strftime('%Y-%m-%d')} a {current_max.strftime('%Y-%m-%d')}")
        self.logger.info(f"- Cobertura: {cobertura:.2f}%")
        return variable, df[['fecha', nuevo_nombre]].copy()

    def generate_daily_index(self) -> pd.DataFrame | None:
        """Create the daily index covering the global date range."""
        if self.global_min_date is None or self.global_max_date is None:
            self.logger.error("No se pudieron determinar las fechas globales")
            return None
        self.daily_index = pd.DataFrame(
            {'fecha': pd.date_range(start=self.global_min_date, end=self.global_max_date, freq='D')}
        )
        self.logger.info(
            f"Índice diario generado: {len(self.daily_index)} días desde {self.global_min_date.strftime('%Y-%m-%d')} hasta {self.global_max_date.strftime('%Y-%m-%d')}"
        )
        return self.daily_index

    def combine_data(self) -> pd.DataFrame | None:
        """Merge all processed series into the daily index."""
        if self.daily_index is None:
            self.logger.error("El índice diario no ha sido generado")
            return None

        combined = self.daily_index.copy()
        for variable, df in self.processed_data.items():
            if df is None or df.empty:
                self.logger.warning(f"Omitiendo {variable} por falta de datos")
                continue
            col_name = self.stats[variable]["nuevo_nombre"]
            combined = self._merge_daily(combined, df, col_name)
        self.final_df = combined
        self.logger.info(
            f"DataFrame final combinado: {len(self.final_df)} filas, {len(self.final_df.columns)} columnas"
        )
        return self.final_df

    def analyze_coverage(self) -> None:
        """Log coverage statistics for each indicator."""
        total_days = len(self.daily_index)
        self.logger.info("\nResumen de Cobertura:")
        for variable, stats in self.stats.items():
            self.logger.info(
                f"- {variable}: {stats['coverage']:.2f}% desde {stats['date_min'].strftime('%Y-%m-%d')} a {stats['date_max'].strftime('%Y-%m-%d')}"
            )

    def save_results(self, output_file: str = "datos_economicos_procesados.xlsx") -> bool:
        """Save the final combined DataFrame and stats to an Excel file."""
        if self.final_df is None:
            self.logger.error("No hay datos combinados para guardar")
            return False
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.final_df.to_excel(writer, sheet_name='Datos Diarios', index=False)
                df_stats = pd.DataFrame(self.stats).T
                df_stats.to_excel(writer, sheet_name='Estadisticas')
                meta = {
                    'Proceso': ['InvestingProcessor'],
                    'Fecha de proceso': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total indicadores': [len(self.stats)],
                    'Periodo': [
                        f"{self.global_min_date.strftime('%Y-%m-%d')} a {self.global_max_date.strftime('%Y-%m-%d')}"
                    ],
                    'Total días': [len(self.daily_index)],
                }
                pd.DataFrame(meta).to_excel(writer, sheet_name='Metadatos', index=False)
            self.logger.info(f"Archivo guardado exitosamente: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {e}")
            return False

    def _transform(self, data: object) -> pd.DataFrame | None:
        """Execute the full processing pipeline."""
        start_time = time.time()
        
        # Load configuration first
        if self.read_config() is None:
            return None

        # Validate required files before processing
        self.logger.info("Validating required files...")
        if not self.validate_required_files():
            self.logger.error("No se encontraron archivos válidos para procesar")
            return None

        # Log PathManager cache statistics
        cache_stats = self.path_manager.get_cache_stats()
        self.logger.info(f"PathManager stats: {cache_stats['total_files']} archivos en {cache_stats['total_categories']} categorías")

        # Process each file
        successful_processes = 0
        for _, config_row in self.config_data.iterrows():
            var, df_processed = self.process_file(config_row)
            self.processed_data[var] = df_processed
            if df_processed is not None:
                successful_processes += 1

        if successful_processes == 0:
            self.logger.error("No se procesó ningún archivo correctamente")
            return None

        self.logger.info(f"Archivos procesados exitosamente: {successful_processes}/{len(self.config_data)}")

        self.generate_daily_index()
        self.combine_data()
        self.analyze_coverage()
        
        end_time = time.time()
        self.logger.info("\nResumen de Ejecución:")
        self.logger.info(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        self.logger.info(f"Archivos configurados: {len(self.config_data)}")
        self.logger.info(f"Archivos procesados exitosamente: {successful_processes}")
        self.logger.info(f"Tasa de éxito: {successful_processes/len(self.config_data)*100:.1f}%")
        
        return self.final_df

    def save(self, data: pd.DataFrame | None, output_file: str) -> bool:
        self.final_df = data
        return self.save_results(output_file)

    def run(self, output_file: str = "datos_economicos_procesados.xlsx") -> bool:
        """Run the processor end-to-end."""
        return super().run(None, output_file)

    def validate_required_files(self) -> bool:
        """Validate that all required files are available before processing.
        
        Returns
        -------
        bool
            True if all files are available or processing can continue, False otherwise
        """
        if self.config_data is None:
            self.logger.error("Configuration data not loaded. Call read_config() first.")
            return False
        
        # Extract required files from configuration
        required_files = [f"{row['Variable']}.xlsx" for _, row in self.config_data.iterrows()]
        
        # Validate files using PathManager
        validation_results = self.path_manager.validate_required_files(required_files)
        
        # Count missing files
        missing_files = [filename for filename, available in validation_results.items() if not available]
        found_files = [filename for filename, available in validation_results.items() if available]
        
        self.logger.info(f"File validation completed:")
        self.logger.info(f"  - Total required: {len(required_files)}")
        self.logger.info(f"  - Found: {len(found_files)}")
        self.logger.info(f"  - Missing: {len(missing_files)}")
        
        if missing_files:
            self.logger.warning("Missing files:")
            for missing_file in missing_files:
                self.logger.warning(f"  - {missing_file}")
                
                # Try to suggest similar files
                suggestions = self.path_manager.suggest_similar_files(missing_file, max_suggestions=3)
                if suggestions:
                    self.logger.info(f"    Similar files found: {', '.join(suggestions)}")
        
        # Log available categories for reference
        categories = self.path_manager.get_available_categories()
        self.logger.info(f"Available data categories: {categories}")
        
        # Return True if at least some files are available
        return len(found_files) > 0
