#!/usr/bin/env python
# coding: utf-8

"""
Script CORREGIDO para mapear valores reales del S&P500 a hechos_predicciones_fields.csv
Usa los valores originales del archivo de entrenamiento en lugar de calcular transformada inversa
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

class SP500RealValueMapper:
    """
    Clase para mapear los valores reales del S&P500 desde el archivo de entrenamiento
    """
    
    def __init__(self):
        self.price_mapping = {}      # Mapeo fecha -> precio original
        self.target_mapping = {}     # Mapeo fecha -> precio target
        self.date_mapping = {}       # Mapeo FechaKey -> fecha real
        
    def load_predictions_file(self, predictions_file):
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
    
    def load_sp500_values(self, original_data_file):
        """
        Carga los valores REALES del S&P500 del archivo de entrenamiento
        """
        logging.info(f"📂 Cargando valores reales del S&P500: {original_data_file}")
        
        try:
            # Cargar datos
            df_original = pd.read_excel(original_data_file)
            df_original['date'] = pd.to_datetime(df_original['date'])
            
            logging.info(f"✅ Datos cargados: {len(df_original)} filas x {len(df_original.columns)} columnas")
            
            # Buscar las columnas del S&P500
            price_col = None
            target_col = None
            return_col = None
            
            for col in df_original.columns:
                if col == 'PRICE_S&P500_Index_index_pricing':
                    price_col = col
                elif col == 'PRICE_S&P500_Index_index_pricing_Target':
                    target_col = col
                elif col == 'PRICE_S&P500_Index_index_pricing_Return_Target':
                    return_col = col
            
            if not all([price_col, target_col, return_col]):
                logging.error("❌ No se encontraron todas las columnas necesarias del S&P500")
                return False
            
            logging.info(f"✅ Columnas encontradas:")
            logging.info(f"   - Precio original: {price_col}")
            logging.info(f"   - Precio target: {target_col}")
            logging.info(f"   - Return: {return_col}")
            
            # Crear mapeos de fecha a valores
            self.price_mapping = {}
            self.target_mapping = {}
            self.return_mapping = {}
            
            valid_rows = 0
            for idx, row in df_original.iterrows():
                date = row['date']
                price = row[price_col]
                target = row[target_col]
                return_val = row[return_col]
                
                if pd.notna(price):
                    self.price_mapping[date] = price
                
                if pd.notna(target):
                    self.target_mapping[date] = target
                    self.return_mapping[date] = return_val
                    valid_rows += 1
            
            logging.info(f"✅ Mapeos creados:")
            logging.info(f"   - Fechas con precio original: {len(self.price_mapping)}")
            logging.info(f"   - Fechas con precio target: {len(self.target_mapping)}")
            logging.info(f"   - Filas válidas (con target): {valid_rows}")
            
            # Mostrar algunos ejemplos
            dates_sample = sorted(list(self.target_mapping.keys()))[:5]
            logging.info(f"\n📊 Muestra de valores:")
            for date in dates_sample:
                if date in self.price_mapping and date in self.target_mapping:
                    price = self.price_mapping[date]
                    target = self.target_mapping[date]
                    ret = self.return_mapping.get(date, 0)
                    logging.info(f"   {date.strftime('%Y-%m-%d')}: ${price:.2f} → ${target:.2f} (Return: {ret:.4f})")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error cargando valores del S&P500: {e}")
            return False
    
    def load_date_mapping(self, all_predictions_file):
        """
        Carga el mapeo de FechaKey a fechas reales
        """
        logging.info("📂 Cargando mapeo de fechas...")
        
        try:
            df_all = pd.read_csv(all_predictions_file)
            df_all['date'] = pd.to_datetime(df_all['date'])
            
            # Crear mapeo
            fechas_unicas = sorted(df_all['date'].unique())
            self.date_mapping = {idx + 1: fecha for idx, fecha in enumerate(fechas_unicas)}
            
            logging.info(f"✅ Mapeo de fechas creado: {len(self.date_mapping)} fechas únicas")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error cargando mapeo de fechas: {e}")
            return False
    
    def map_real_values(self):
        """
        Mapea los valores reales del S&P500 a las predicciones
        """
        logging.info("🔄 Mapeando valores reales del S&P500...")
        
        # Inicializar columnas
        valor_real_sp500 = []
        valor_predicho_sp500 = []
        
        # Estadísticas
        stats = {
            'total': 0,
            'con_valores': 0,
            'sin_valores': 0,
            'forecast_future': 0
        }
        
        # Procesar cada fila
        for idx, row in self.df_predictions.iterrows():
            stats['total'] += 1
            
            try:
                # Obtener fecha real
                fecha_key = row['FechaKey']
                if fecha_key not in self.date_mapping:
                    valor_real_sp500.append(np.nan)
                    valor_predicho_sp500.append(np.nan)
                    stats['sin_valores'] += 1
                    continue
                
                fecha_real = self.date_mapping[fecha_key]
                
                # Para valores reales (no forecast futuro)
                if row['TipoPeriodo'] != 'Forecast_Future':
                    # Buscar el valor TARGET (que es el precio en t+20)
                    if fecha_real in self.target_mapping:
                        # El valor real es el precio target
                        real_price = self.target_mapping[fecha_real]
                        
                        # Calcular el precio predicho usando el return predicho
                        # Necesitamos el precio base (20 días antes)
                        if fecha_real in self.price_mapping:
                            base_price = self.price_mapping[fecha_real]
                            pred_return = row['ValorPredicho']
                            pred_price = base_price * (1 + pred_return)
                        else:
                            pred_price = np.nan
                        
                        valor_real_sp500.append(real_price)
                        valor_predicho_sp500.append(pred_price)
                        stats['con_valores'] += 1
                        
                        # Log para primeros casos
                        if idx < 5:
                            logging.info(f"   Fila {idx}: Fecha {fecha_real.strftime('%Y-%m-%d')}")
                            logging.info(f"      Precio real (target): ${real_price:.2f}")
                            logging.info(f"      Return predicho: {pred_return:.4f}")
                            logging.info(f"      Precio predicho: ${pred_price:.2f}")
                    else:
                        valor_real_sp500.append(np.nan)
                        valor_predicho_sp500.append(np.nan)
                        stats['sin_valores'] += 1
                else:
                    # Para Forecast_Future solo tenemos predicciones
                    if fecha_real in self.price_mapping:
                        base_price = self.price_mapping[fecha_real]
                        pred_return = row['ValorPredicho']
                        pred_price = base_price * (1 + pred_return) if not pd.isna(pred_return) else np.nan
                    else:
                        pred_price = np.nan
                    
                    valor_real_sp500.append(np.nan)  # No hay valor real para futuro
                    valor_predicho_sp500.append(pred_price)
                    stats['forecast_future'] += 1
                    
            except Exception as e:
                if idx < 10:
                    logging.warning(f"Error en fila {idx}: {e}")
                valor_real_sp500.append(np.nan)
                valor_predicho_sp500.append(np.nan)
                stats['sin_valores'] += 1
        
        # Agregar columnas al DataFrame
        self.df_predictions['ValorReal_SP500'] = valor_real_sp500
        self.df_predictions['ValorPredicho_SP500'] = valor_predicho_sp500
        
        # Estadísticas finales
        logging.info(f"\n📊 ESTADÍSTICAS DE MAPEO:")
        logging.info(f"   Total filas: {stats['total']:,}")
        logging.info(f"   Con valores mapeados: {stats['con_valores']:,} ({stats['con_valores']/stats['total']*100:.1f}%)")
        logging.info(f"   Sin valores: {stats['sin_valores']:,}")
        logging.info(f"   Forecast futuro: {stats['forecast_future']:,}")
        
        # Rangos de valores
        valid_real = self.df_predictions['ValorReal_SP500'].dropna()
        valid_pred = self.df_predictions['ValorPredicho_SP500'].dropna()
        
        if len(valid_real) > 0:
            logging.info(f"\n💰 VALORES REALES DEL S&P500:")
            logging.info(f"   Mínimo: ${valid_real.min():.2f}")
            logging.info(f"   Máximo: ${valid_real.max():.2f}")
            logging.info(f"   Promedio: ${valid_real.mean():.2f}")
            
        if len(valid_pred) > 0:
            logging.info(f"\n📈 VALORES PREDICHOS DEL S&P500:")
            logging.info(f"   Mínimo: ${valid_pred.min():.2f}")
            logging.info(f"   Máximo: ${valid_pred.max():.2f}")
            logging.info(f"   Promedio: ${valid_pred.mean():.2f}")
    
    def save_enhanced_file(self, output_file):
        """
        Guarda el archivo con los valores reales del S&P500
        """
        logging.info(f"\n💾 Guardando archivo con valores del S&P500: {output_file}")
        
        try:
            # Ordenar columnas
            columns_order = [
                'PrediccionKey', 'FechaKey', 'ModeloKey', 'MercadoKey',
                'ValorReal', 'ValorPredicho', 'ErrorAbsoluto', 'ErrorPorcentual',
                'TipoPeriodo', 'ZonaEntrenamiento', 'EsPrediccionFutura',
                'ValorReal_SP500', 'ValorPredicho_SP500'
            ]
            
            columns_to_save = [col for col in columns_order if col in self.df_predictions.columns]
            
            # Guardar
            self.df_predictions[columns_to_save].to_csv(output_file, index=False, float_format='%.6f')
            
            logging.info(f"✅ Archivo guardado exitosamente")
            logging.info(f"   Nuevas columnas: ValorReal_SP500, ValorPredicho_SP500")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error guardando archivo: {e}")
            return False
    
    def generate_validation_report(self):
        """
        Genera reporte de validación comparando returns vs precios
        """
        logging.info("\n🔍 VALIDACIÓN DE VALORES:")
        
        # Filtrar filas con valores completos
        df_valid = self.df_predictions.dropna(subset=['ValorReal', 'ValorReal_SP500', 'ValorPredicho_SP500'])
        
        if len(df_valid) > 0:
            logging.info(f"\nComparando {len(df_valid)} filas con valores completos:")
            
            # Muestra de validación
            sample = df_valid.head(10)
            for idx, row in sample.iterrows():
                return_real = row['ValorReal']
                precio_real = row['ValorReal_SP500']
                return_pred = row['ValorPredicho']
                precio_pred = row['ValorPredicho_SP500']
                
                logging.info(f"\n   Registro {idx}:")
                logging.info(f"      Return real: {return_real:.4f} → Precio real: ${precio_real:.2f}")
                logging.info(f"      Return pred: {return_pred:.4f} → Precio pred: ${precio_pred:.2f}")


def main():
    """
    Función principal
    """
    print("\n" + "="*70)
    print("🔄 MAPEADOR DE VALORES REALES DEL S&P500")
    print("="*70)
    
    # Configuración de rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    # Archivos
    predictions_file = os.path.join(base_dir, 'SP500_INDEX_Analisis', 'Data', '4_results', 
                                   'hechos_predicciones_fields.csv')
    
    original_data_file = os.path.join(base_dir, 'SP500_INDEX_Analisis', 'Data', '2_processed', 
                                     'datos_economicos_1month_SP500_TRAINING.xlsx')
    
    all_predictions_file = os.path.join(base_dir, 'SP500_INDEX_Analisis', 'Data', '4_results', 
                                       'all_models_predictions.csv')
    
    output_file = os.path.join(base_dir, 'SP500_INDEX_Analisis', 'Data', '4_results', 
                              'hechos_predicciones_fields_con_sp500.csv')
    
    # Crear mapeador
    mapper = SP500RealValueMapper()
    
    # 1. Cargar predicciones
    if not mapper.load_predictions_file(predictions_file):
        return
    
    # 2. Cargar mapeo de fechas
    if not mapper.load_date_mapping(all_predictions_file):
        return
    
    # 3. Cargar valores reales del S&P500
    if not mapper.load_sp500_values(original_data_file):
        return
    
    # 4. Mapear valores reales
    mapper.map_real_values()
    
    # 5. Generar reporte de validación
    mapper.generate_validation_report()
    
    # 6. Guardar archivo
    if mapper.save_enhanced_file(output_file):
        print(f"\n✅ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"   Archivo generado: {output_file}")
        print(f"   Columnas agregadas:")
        print(f"   - ValorReal_SP500: Precio real del S&P500 (del archivo original)")
        print(f"   - ValorPredicho_SP500: Precio predicho calculado desde returns")
    else:
        print("❌ Esrror guardando archivo")


if __name__ == "__main__":
    main()