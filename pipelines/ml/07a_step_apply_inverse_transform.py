#!/usr/bin/env python
# coding: utf-8

"""
Script ACTUALIZADO para extraer PRICE_S&P500_Index_index_pricing_Target 
de datos_economicos_1month_SP500_TRAINING y últimos 20 días de INFERENCE
Solo mapeo de fechas para cada modelo, sin calcular valores reales
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from sp500_analysis.config.settings import settings
from sp500_analysis.shared.logging.logger import configurar_logging

# Configurar logging
log_file = os.path.join(settings.log_dir, f"apply_inverse_transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
configurar_logging(log_file)


class SP500TargetExtractor:
    """
    Clase para extraer PRICE_S&P500_Index_index_pricing_Target de archivos específicos
    y mapear fechas para cada modelo
    """

    def __init__(self) -> None:
        self.target_mapping = {}  # Mapeo fecha -> precio target
        self.date_mapping = {}  # Mapeo FechaKey -> fecha real
        self.df_predictions = None
        self.modelo_mapping = {}  # Mapeo ModeloKey -> NombreModelo

    def load_predictions_file(self, predictions_file: str) -> bool:
        """
        Carga el archivo de predicciones (hechos_predicciones_fields.csv)
        """
        logging.info(f"📂 Cargando archivo de predicciones: {predictions_file}")

        try:
            self.df_predictions = pd.read_csv(predictions_file)
            logging.info(f"✅ Predicciones cargadas: {len(self.df_predictions):,} filas")
            return True
        except Exception as e:
            logging.error(f"❌ Error cargando predicciones: {e}")
            return False

    def load_sp500_target_values(self) -> bool:
        """
        Carga PRICE_S&P500_Index_index_pricing_Target de:
        1. datos_economicos_1month_SP500_TRAINING.xlsx 
        2. Últimos 20 días de datos_economicos_1month_SP500_INFERENCE.xlsx
        """
        logging.info("Cargando valores TARGET del S&P500...")

        try:
            # Archivo 1: Training
            training_file = settings.processed_dir / "datos_economicos_1month_SP500_TRAINING.xlsx"
            logging.info(f"   Cargando TRAINING: {training_file}")
            
            df_training = pd.read_excel(training_file)
            df_training['date'] = pd.to_datetime(df_training['date'])
            
            # Archivo 2: Inference (últimos 20 días)
            inference_file = settings.processed_dir / "datos_economicos_1month_SP500_INFERENCE.xlsx"
            logging.info(f"   Cargando INFERENCE: {inference_file}")
            
            df_inference = pd.read_excel(inference_file)
            df_inference['date'] = pd.to_datetime(df_inference['date'])
            
            # Tomar solo los últimos 20 días de inference
            df_inference_last20 = df_inference.tail(20).copy()
            
            logging.info(f"Datos cargados:")
            logging.info(f"   - Training: {len(df_training)} filas")
            logging.info(f"   - Inference (total): {len(df_inference)} filas")
            logging.info(f"   - Inference (últimos 20): {len(df_inference_last20)} filas")

            # Combinar ambos datasets
            df_combined = pd.concat([df_training, df_inference_last20], ignore_index=True)
            logging.info(f"   - Combined: {len(df_combined)} filas")

            # Buscar la columna TARGET
            target_col = 'PRICE_S&P500_Index_index_pricing_Target'
            
            if target_col not in df_combined.columns:
                logging.error(f"Columna {target_col} no encontrada")
                logging.info(f"Columnas disponibles: {list(df_combined.columns)}")
                return False

            logging.info(f"Columna encontrada: {target_col}")

            # Crear mapeo de fecha a valores TARGET
            self.target_mapping = {}
            valid_rows = 0

            for idx, row in df_combined.iterrows():
                date = row['date']
                target = row[target_col]

                if pd.notna(target) and pd.notna(date):
                    self.target_mapping[date] = target
                    valid_rows += 1

            logging.info("Mapeo TARGET creado:")
            logging.info(f"   - Fechas con valor TARGET: {len(self.target_mapping)}")
            logging.info(f"   - Filas válidas: {valid_rows}")

            # Mostrar algunos ejemplos
            dates_sample = sorted(list(self.target_mapping.keys()))
            if len(dates_sample) >= 5:
                logging.info("Muestra de valores TARGET:")
                for i, date in enumerate(dates_sample[:5]):
                    target = self.target_mapping[date]
                    logging.info(f"   {date.strftime('%Y-%m-%d')}: ${target:.2f}")
                
                logging.info("   ...")
                for i, date in enumerate(dates_sample[-3:]):
                    target = self.target_mapping[date]
                    logging.info(f"   {date.strftime('%Y-%m-%d')}: ${target:.2f}")

            return True

        except Exception as e:
            logging.error(f"Error cargando valores TARGET del S&P500: {e}")
            return False

    def load_date_mapping(self, all_predictions_file: str) -> bool:
        """
        Carga el mapeo de FechaKey a fechas reales
        """
        logging.info("Cargando mapeo de fechas...")

        try:
            df_all = pd.read_csv(all_predictions_file)
            df_all['date'] = pd.to_datetime(df_all['date'])

            # Crear mapeo
            fechas_unicas = sorted(df_all['date'].unique())
            self.date_mapping = {idx + 1: fecha for idx, fecha in enumerate(fechas_unicas)}

            logging.info(f"Mapeo de fechas creado: {len(self.date_mapping)} fechas únicas")
            return True

        except Exception as e:
            logging.error(f"Error cargando mapeo de fechas: {e}")
            return False

    def load_modelo_mapping(self) -> bool:
        """
        Carga el mapeo de ModeloKey a NombreModelo desde dim_modelo.csv
        """
        logging.info("Cargando mapeo de modelos...")

        try:
            modelo_file = settings.results_dir / "dim_modelo.csv"
            df_modelos = pd.read_csv(modelo_file)

            # Crear mapeo ModeloKey -> NombreModelo
            self.modelo_mapping = dict(zip(df_modelos['ModeloKey'], df_modelos['NombreModelo']))

            logging.info(f"Mapeo de modelos creado: {len(self.modelo_mapping)} modelos")
            for key, nombre in self.modelo_mapping.items():
                logging.info(f"   {key}: {nombre}")
            
            return True

        except Exception as e:
            logging.error(f"Error cargando mapeo de modelos: {e}")
            return False

    def map_target_values_by_model(self) -> None:
        """
        Mapea los valores TARGET del S&P500 por modelo y fecha
        NO calcula valores reales, solo mapea fechas
        """
        logging.info("Mapeando valores TARGET por modelo y fecha...")

        # Inicializar columnas
        valor_target_sp500 = []
        fecha_real = []
        modelo_info = []

        # Estadísticas
        stats = {
            'total': 0, 
            'con_target': 0, 
            'sin_target': 0, 
            'forecast_future': 0,
            'por_modelo': {}
        }

        # Procesar cada fila
        for idx, row in self.df_predictions.iterrows():
            stats['total'] += 1

            try:
                # Obtener información del modelo
                modelo_key = row.get('ModeloKey', 0)
                modelo = self.modelo_mapping.get(modelo_key, f'Unknown_{modelo_key}')
                tipo_periodo = row.get('TipoPeriodo', 'Unknown')
                
                # Estadísticas por modelo
                if modelo not in stats['por_modelo']:
                    stats['por_modelo'][modelo] = {'total': 0, 'con_target': 0}
                stats['por_modelo'][modelo]['total'] += 1

                # Obtener FechaKey y mapear a fecha real
                fecha_key = row.get('FechaKey', 0)
                
                if fecha_key in self.date_mapping:
                    # Mapear fecha real
                    fecha_real_mapped = self.date_mapping[fecha_key]
                    fecha_real.append(fecha_real_mapped)
                    
                    # Información del modelo
                    modelo_info.append(f"{modelo}_{tipo_periodo}")

                    # Buscar valor TARGET en el mapeo
                    if fecha_real_mapped in self.target_mapping:
                        target_price = self.target_mapping[fecha_real_mapped]
                        valor_target_sp500.append(f"{target_price:.2f}")
                        stats['con_target'] += 1
                        stats['por_modelo'][modelo]['con_target'] += 1
                    else:
                        valor_target_sp500.append("")
                        stats['sin_target'] += 1
                        
                        # Contar predicciones futuras sin TARGET
                        if tipo_periodo == 'Forecast_Future':
                            stats['forecast_future'] += 1
                else:
                    # FechaKey no encontrada en mapeo
                    fecha_real.append("")
                    modelo_info.append(f"{modelo}_{tipo_periodo}")
                    valor_target_sp500.append("")
                    stats['sin_target'] += 1

            except Exception as e:
                logging.warning(f"Error procesando fila {idx}: {e}")
                fecha_real.append("")
                modelo_info.append("Error")
                valor_target_sp500.append("")
                stats['sin_target'] += 1

        # Agregar columnas al DataFrame
        self.df_predictions['FechaReal'] = fecha_real
        self.df_predictions['ModeloInfo'] = modelo_info
        self.df_predictions['ValorReal_SP500'] = valor_target_sp500

        # Logging de estadísticas
        logging.info("Mapeo completado:")
        logging.info(f"   - Total filas procesadas: {stats['total']:,}")
        logging.info(f"   - Con valor TARGET: {stats['con_target']:,}")
        logging.info(f"   - Sin valor TARGET: {stats['sin_target']:,}")
        logging.info(f"   - Forecast Future: {stats['forecast_future']:,}")
        
        logging.info("Estadísticas por modelo:")
        for modelo, data in stats['por_modelo'].items():
            porcentaje = (data['con_target'] / data['total'] * 100) if data['total'] > 0 else 0
            logging.info(f"   {modelo}: {data['con_target']:,}/{data['total']:,} ({porcentaje:.1f}%)")

    def save_enhanced_file(self, output_file: str) -> bool:
        """
        Guarda el archivo con los valores TARGET mapeados
        """
        logging.info(f"Guardando archivo mejorado: {output_file}")

        try:
            self.df_predictions.to_csv(output_file, index=False, sep=';')
            logging.info(f"Archivo guardado exitosamente")
            logging.info(f"   - Filas: {len(self.df_predictions):,}")
            logging.info(f"   - Columnas: {len(self.df_predictions.columns)}")
            
            # Verificar las nuevas columnas
            nuevas_cols = ['FechaReal', 'ModeloInfo', 'ValorReal_SP500']
            for col in nuevas_cols:
                if col in self.df_predictions.columns:
                    if col == 'ValorReal_SP500':
                        # Para ValorReal_SP500, contar valores que no sean strings vacíos
                        valores_validos = (self.df_predictions[col] != "").sum()
                    else:
                        valores_validos = self.df_predictions[col].notna().sum()
                    logging.info(f"   - {col}: {valores_validos:,} valores válidos")

            return True

        except Exception as e:
            logging.error(f"Error guardando archivo: {e}")
            return False

    def generate_validation_report(self) -> None:
        """
        Genera un reporte de validación
        """
        logging.info("Generando reporte de validación...")

        try:
            # Crear columna temporal con nombres de modelos para el reporte
            self.df_predictions['ModeloNombre'] = self.df_predictions['ModeloKey'].map(self.modelo_mapping)
            
            # Resumen por modelo
            resumen = self.df_predictions.groupby('ModeloNombre').agg({
                'ValorReal_SP500': ['count', lambda x: (x != "").sum()],
                'FechaReal': lambda x: x.notna().sum()
            }).round(2)

            logging.info("Resumen por modelo:")
            for modelo in resumen.index:
                total = resumen.loc[modelo, ('ValorReal_SP500', 'count')]
                validos = resumen.loc[modelo, ('ValorReal_SP500', '<lambda>')]
                fechas = resumen.loc[modelo, ('FechaReal', '<lambda>')]
                logging.info(f"   {modelo}: {validos}/{total} valores reales válidos, {fechas} fechas mapeadas")

        except Exception as e:
            logging.warning(f"Error generando reporte: {e}")


def main() -> None:
    """
    Función principal para extraer TARGET values y mapear fechas por modelo
    """
    logging.info("Iniciando extracción de valores TARGET del S&P500")
    logging.info("=" * 70)
    
    try:
        # Rutas de archivos
        predictions_file = settings.results_dir / "hechos_predicciones_fields.csv"
        all_predictions_file = settings.results_dir / "all_models_predictions.csv"
        output_file = settings.results_dir / "hechos_predicciones_fields_con_sp500.csv"

        # Verificar archivos
        if not predictions_file.exists():
            logging.error(f"❌ Archivo no encontrado: {predictions_file}")
            return

        if not all_predictions_file.exists():
            logging.error(f"❌ Archivo no encontrado: {all_predictions_file}")
            return

        # Crear extractor
        extractor = SP500TargetExtractor()

        # Cargar archivos
        if not extractor.load_predictions_file(str(predictions_file)):
            return

        if not extractor.load_sp500_target_values():
            return

        if not extractor.load_date_mapping(str(all_predictions_file)):
            return

        if not extractor.load_modelo_mapping():
            return

        # Mapear valores TARGET por modelo
        extractor.map_target_values_by_model()

        # Guardar archivo
        if not extractor.save_enhanced_file(str(output_file)):
            return

        # Generar reporte
        extractor.generate_validation_report()

        logging.info("Proceso completado exitosamente")
        logging.info("=" * 70)
        logging.info("Archivos generados:")
        logging.info(f"  {output_file}")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"❌ Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
