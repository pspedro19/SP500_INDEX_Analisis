"""
Financial Time Series Exploratory Data Analysis

Este módulo proporciona funciones para el análisis exploratorio de datos (EDA)
de series temporales financieras almacenadas en archivos Excel. El formato
esperado es "ancho", donde la columna 'fecha' contiene las fechas y cada
otra columna representa una variable financiera.

Autor: Data Scientist Senior
Fecha: Mayo 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import warnings

# Configuración global para visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
warnings.filterwarnings('ignore')
#Ruta por defecto
filepath = "C:/Users/natus/Documents/Trabajo/PEDRO_PEREZ/Proyecto_Mercado_de_Valores/SP500_INDEX_Analisis/Data/1_preprocess/MERGEDEXCELS.xlsx"

def load_excel_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos desde un archivo Excel y convierte la columna de fecha al formato datetime.
    
    Args:
        filepath (str): Ruta al archivo Excel (.xlsx)
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados y la columna fecha convertida
    """
    try:
        # Cargar el archivo Excel
        df = pd.read_excel(filepath)
        
        # Verificar si existe la columna 'fecha'
        if 'fecha' not in df.columns:
            raise ValueError("El archivo Excel debe contener una columna llamada 'fecha'")
        
        # Convertir la columna 'fecha' a datetime
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Establecer la columna fecha como índice
        df = df.set_index('fecha')
        
        print(f"Datos cargados correctamente. Dimensiones: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return pd.DataFrame()

def select_variable(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    """
    Selecciona una variable específica del DataFrame y crea un nuevo DataFrame
    con la fecha como índice y la variable seleccionada.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        variable_name (str): Nombre de la variable a seleccionar
        
    Returns:
        pd.DataFrame: DataFrame con la fecha como índice y la variable seleccionada
    """
    if variable_name not in df.columns:
        raise ValueError(f"La variable '{variable_name}' no existe en el DataFrame")
    
    # Crear un nuevo DataFrame con la variable seleccionada
    selected_df = df[[variable_name]].copy()
    
    print(f"Variable '{variable_name}' seleccionada. Filas disponibles: {len(selected_df)}")
    return selected_df

def clean_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza de datos para una variable financiera,
    eliminando valores nulos y duplicados.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Guardar dimensiones originales
    original_shape = df.shape
    
    # Eliminar filas duplicadas
    df = df[~df.index.duplicated(keep='first')]
    
    # Reporte sobre valores nulos
    null_count = df.isna().sum().item()
    if null_count > 0:
        print(f"Se detectaron {null_count} valores nulos")
        
        # Para series temporales, podemos optar por eliminar o interpolar
        # En este caso usaremos interpolación lineal
        df = df.interpolate(method='linear')
        
        # Verificar si quedan valores nulos al principio o al final
        if df.isna().any().any():
            # Si quedan nulos al principio o al final, los eliminamos
            df = df.dropna()
            print(f"Se eliminaron filas con valores nulos al inicio o fin de la serie temporal")
    
    # Ordenar índice cronológicamente
    df = df.sort_index()
    
    # Resumen de la limpieza
    print(f"Limpieza completada:")
    print(f"  - Filas originales: {original_shape[0]}")
    print(f"  - Filas después de limpieza: {df.shape[0]}")
    print(f"  - Valores nulos restantes: {df.isna().sum().item()}")
    
    return df

def _detect_outliers(data: pd.Series, method: str = "IQR") -> pd.Series:
    """
    Detecta outliers en una serie temporal usando el método especificado.
    
    Args:
        data (pd.Series): Serie temporal de datos
        method (str): Método para detectar outliers ('IQR' o 'Z-score')
        
    Returns:
        pd.Series: Serie booleana con True para outliers
    """
    if method == "IQR":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == "Z-score":
        z_scores = (data - data.mean()) / data.std()
        return abs(z_scores) > 3
    
    else:
        raise ValueError("Método no válido. Use 'IQR' o 'Z-score'")

def describe_variable(df: pd.DataFrame) -> None:
    """
    Muestra estadísticas descriptivas detalladas para la variable financiera.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    variable_name = df.columns[0]
    data = df[variable_name]
    
    # Estadísticas básicas
    stats = data.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    
    # Estadísticas adicionales
    stats_additional = pd.Series({
        'kurtosis': data.kurtosis(),
        'skewness': data.skew(),
        'iqr': stats.loc['75%'] - stats.loc['25%'],
        'range': stats.loc['max'] - stats.loc['min'],
        'missing_values': data.isna().sum(),
        'missing_pct': data.isna().mean() * 100,
        'n_observations': len(data)
    })
    
    # Detección de outliers
    outliers_iqr = _detect_outliers(data, method="IQR")
    outliers_zscore = _detect_outliers(data, method="Z-score")
    
    stats_outliers = pd.Series({
        'outliers_iqr_count': outliers_iqr.sum(),
        'outliers_iqr_pct': outliers_iqr.mean() * 100,
        'outliers_zscore_count': outliers_zscore.sum(),
        'outliers_zscore_pct': outliers_zscore.mean() * 100
    })
    
    # Imprimir en formato agradable
    print(f"\n{'='*60}")
    print(f" Estadísticas Descriptivas para: {variable_name}")
    print(f"{'='*60}")
    
    print("\n--- Estadísticas Básicas ---")
    print(stats.to_string())
    
    print("\n--- Estadísticas Adicionales ---")
    print(stats_additional.to_string())
    
    print("\n--- Detección de Outliers ---")
    print(stats_outliers.to_string())
    
    print(f"\n{'='*60}\n")

def plot_time_series(df: pd.DataFrame, variable_name: str) -> None:
    """
    Genera un gráfico de serie temporal para la variable financiera.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Verificar que el nombre de variable coincida con la columna
    col_name = df.columns[0]
    if col_name != variable_name:
        print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
    
    plt.figure(figsize=(12, 6))
    
    # Graficar la serie temporal
    plt.plot(df.index, df[col_name], linewidth=1.5, color='#1f77b4')
    
    # Establecer título y etiquetas
    plt.title(f'Serie Temporal: {variable_name}', fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def plot_histogram(df: pd.DataFrame, variable_name: str) -> None:
    """
    Genera un histograma para la variable financiera.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Verificar que el nombre de variable coincida con la columna
    col_name = df.columns[0]
    if col_name != variable_name:
        print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
    
    plt.figure(figsize=(10, 6))
    
    # Calcular el número de bins usando la regla de Scott
    data = df[col_name].dropna()
    bin_width = 3.5 * data.std() / (len(data) ** (1/3))
    num_bins = int((data.max() - data.min()) / bin_width) if len(data) > 0 else 20
    num_bins = min(max(10, num_bins), 50)  # Limitar entre 10 y 50 bins
    
    # Graficar el histograma
    sns.histplot(data, bins=num_bins, kde=True, color='#1f77b4')
    
    # Añadir líneas verticales para estadísticas clave
    plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Media: {data.mean():.4f}')
    plt.axvline(data.median(), color='green', linestyle='dashed', linewidth=1.5, label=f'Mediana: {data.median():.4f}')
    
    # Establecer título y etiquetas
    plt.title(f'Histograma: {variable_name}', fontsize=14)
    plt.xlabel('Valor', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def plot_boxplot(df: pd.DataFrame, variable_name: str) -> None:
    """
    Genera un boxplot para la variable financiera.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Verificar que el nombre de variable coincida con la columna
    col_name = df.columns[0]
    if col_name != variable_name:
        print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
    
    plt.figure(figsize=(10, 6))
    
    # Graficar el boxplot
    sns.boxplot(x=df[col_name].dropna(), color='#1f77b4')
    
    # Añadir stripplot para visualizar la distribución
    sns.stripplot(x=df[col_name].dropna(), color='black', alpha=0.5, size=3)
    
    # Establecer título y etiquetas
    plt.title(f'Boxplot: {variable_name}', fontsize=14)
    plt.xlabel('Valor', fontsize=12)
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def plot_rolling_mean(df: pd.DataFrame, variable_name: str, window: int = 30) -> None:
    """
    Genera un gráfico de la serie temporal con su media móvil.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
        window (int): Tamaño de la ventana para la media móvil
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Verificar que el nombre de variable coincida con la columna
    col_name = df.columns[0]
    if col_name != variable_name:
        print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
    
    # Calcular la media móvil
    rolling_mean = df[col_name].rolling(window=window).mean()
    
    plt.figure(figsize=(12, 6))
    
    # Graficar la serie temporal y la media móvil
    plt.plot(df.index, df[col_name], linewidth=1, alpha=0.6, color='#1f77b4', label='Serie Original')
    plt.plot(df.index, rolling_mean, linewidth=2, color='red', label=f'Media Móvil ({window} días)')
    
    # Establecer título y etiquetas
    plt.title(f'Serie Temporal con Media Móvil: {variable_name}', fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def plot_outliers(df: pd.DataFrame, variable_name: str, method: str = "IQR") -> None:
    """
    Genera un gráfico de la serie temporal destacando los outliers.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
        method (str): Método para detectar outliers ('IQR' o 'Z-score')
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Verificar que el nombre de variable coincida con la columna
    col_name = df.columns[0]
    if col_name != variable_name:
        print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
    
    # Detectar outliers
    data = df[col_name]
    outliers = _detect_outliers(data, method=method)
    
    plt.figure(figsize=(12, 6))
    
    # Graficar la serie temporal
    plt.plot(df.index, data, linewidth=1, alpha=0.6, color='#1f77b4', label='Serie Original')
    
    # Destacar los outliers
    outlier_dates = df.index[outliers]
    outlier_values = data[outliers]
    
    plt.scatter(outlier_dates, outlier_values, color='red', s=30, label=f'Outliers ({outliers.sum()} puntos)')
    
    # Establecer título y etiquetas
    plt.title(f'Detección de Outliers ({method}): {variable_name}', fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Información adicional sobre los outliers
    if outliers.sum() > 0:
        print(f"Se detectaron {outliers.sum()} outliers usando el método {method}")
        print(f"Porcentaje de outliers: {outliers.mean() * 100:.2f}%")
        
        # Mostrar los 5 outliers más extremos
        if outliers.sum() > 5:
            sorted_outliers = data[outliers].sort_values(ascending=False)
            print("\nTop 5 outliers más altos:")
            for date, value in sorted_outliers.head().items():
                print(f"  - {date.date()}: {value:.4f}")
                
            print("\nTop 5 outliers más bajos:")
            for date, value in sorted_outliers.tail().items():
                print(f"  - {date.date()}: {value:.4f}")
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def analyze_seasonality(df: pd.DataFrame, variable_name: str) -> None:
    """
    Analiza la estacionalidad de la serie temporal.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    # Verificar que el nombre de variable coincida con la columna
    col_name = df.columns[0]
    if col_name != variable_name:
        print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
    
    # Asegurarse de que el índice es un DatetimeIndex ordenado
    data = df.sort_index()
    
    # Crear una figura con 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Patrones mensuales
    monthly_avg = data.groupby(data.index.month)[col_name].mean()
    monthly_std = data.groupby(data.index.month)[col_name].std()
    
    axes[0].errorbar(
        range(1, 13), 
        monthly_avg.values, 
        yerr=monthly_std.values,
        fmt='o-', 
        capsize=5, 
        linewidth=2, 
        elinewidth=1
    )
    axes[0].set_title(f'Patrón Mensual: {variable_name}', fontsize=14)
    axes[0].set_xlabel('Mes', fontsize=12)
    axes[0].set_ylabel('Valor Promedio', fontsize=12)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    axes[0].grid(True, alpha=0.3)
    
    # 2. Patrones diarios de la semana
    if len(data) >= 30:  # Solo si hay suficientes datos
        weekday_avg = data.groupby(data.index.dayofweek)[col_name].mean()
        weekday_std = data.groupby(data.index.dayofweek)[col_name].std()
        
        axes[1].errorbar(
            range(0, 7), 
            weekday_avg.values, 
            yerr=weekday_std.values,
            fmt='o-', 
            capsize=5, 
            linewidth=2, 
            elinewidth=1
        )
        axes[1].set_title(f'Patrón por Día de la Semana: {variable_name}', fontsize=14)
        axes[1].set_xlabel('Día de la Semana', fontsize=12)
        axes[1].set_ylabel('Valor Promedio', fontsize=12)
        axes[1].set_xticks(range(0, 7))
        axes[1].set_xticklabels(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
        axes[1].grid(True, alpha=0.3)
    
    # 3. Descomposición de tendencia y estacionalidad (solo si hay suficientes datos)
    if len(data) >= 365:
        # Resample a frecuencia diaria para asegurar una serie continua
        resampled_data = data.resample('D').mean().interpolate()
        
        try:
            from statsmodels.tsa.seasonal import STL
            
            # Descomposición STL
            stl = STL(resampled_data, seasonal=365)
            result = stl.fit()
            
            # Graficar la descomposición
            axes[2].plot(result.trend, label='Tendencia', linewidth=2)
            axes[2].plot(result.seasonal, label='Estacionalidad', linewidth=1)
            axes[2].plot(result.resid, label='Residuos', alpha=0.5)
            axes[2].set_title(f'Descomposición STL: {variable_name}', fontsize=14)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[2].text(0.5, 0.5, f"No se pudo realizar la descomposición: {e}", 
                         horizontalalignment='center', verticalalignment='center')
    else:
        axes[2].text(0.5, 0.5, "Se requieren al menos 365 días de datos para la descomposición estacional", 
                     horizontalalignment='center', verticalalignment='center')
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(df: pd.DataFrame, variable_name: str, lags: int = 30) -> None:
    """
    Genera gráficos de autocorrelación y autocorrelación parcial.
    
    Args:
        df (pd.DataFrame): DataFrame con la fecha como índice y una única variable
        variable_name (str): Nombre de la variable para el título
        lags (int): Número de rezagos a mostrar
    """
    if df.shape[1] != 1:
        raise ValueError("El DataFrame debe contener solo una variable")
    
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        # Verificar que el nombre de variable coincida con la columna
        col_name = df.columns[0]
        if col_name != variable_name:
            print(f"Advertencia: El nombre de columna ({col_name}) no coincide con el nombre proporcionado ({variable_name})")
        
        data = df[col_name].dropna()
        
        # Crear figura con dos subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Graficar ACF
        plot_acf(data, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'Función de Autocorrelación: {variable_name}', fontsize=14)
        
        # Graficar PACF
        plot_pacf(data, lags=lags, ax=axes[1], alpha=0.05, method='ols')
        axes[1].set_title(f'Función de Autocorrelación Parcial: {variable_name}', fontsize=14)
        
        # Ajustar diseño
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error al generar gráficos de autocorrelación: {e}")
        print("Asegúrese de tener instalada la librería statsmodels")

def run_complete_eda(filepath: str, variable_name: str) -> pd.DataFrame:
    """
    Ejecuta un análisis exploratorio de datos completo para una variable financiera.
    
    Args:
        filepath (str): Ruta al archivo Excel (.xlsx)
        variable_name (str): Nombre de la variable a analizar
        
    Returns:
        pd.DataFrame: DataFrame procesado con la variable seleccionada
    """
    print(f"\n{'='*80}")
    print(f" ANÁLISIS EXPLORATORIO DE DATOS: {variable_name}")
    print(f"{'='*80}\n")
    
    # Cargar datos
    print("\n1. CARGA DE DATOS")
    df = load_excel_data(filepath)
    
    # Seleccionar variable
    print("\n2. SELECCIÓN DE VARIABLE")
    variable_df = select_variable(df, variable_name)
    
    # Limpiar datos
    print("\n3. LIMPIEZA DE DATOS")
    clean_df = clean_variable(variable_df)
    
    # Estadísticas descriptivas
    print("\n4. ESTADÍSTICAS DESCRIPTIVAS")
    describe_variable(clean_df)
    
    # Visualizaciones
    print("\n5. VISUALIZACIONES")
    
    print("\n5.1. Serie Temporal")
    plot_time_series(clean_df, variable_name)
    
    print("\n5.2. Histograma")
    plot_histogram(clean_df, variable_name)
    
    print("\n5.3. Boxplot")
    plot_boxplot(clean_df, variable_name)
    
    print("\n5.4. Media Móvil")
    plot_rolling_mean(clean_df, variable_name)
    
    print("\n5.5. Detección de Outliers")
    plot_outliers(clean_df, variable_name)
    
    print("\n5.6. Análisis de Estacionalidad")
    analyze_seasonality(clean_df, variable_name)
    
    print("\n5.7. Autocorrelación")
    plot_autocorrelation(clean_df, variable_name)
    
    print(f"\n{'='*80}")
    print(f" ANÁLISIS COMPLETADO: {variable_name}")
    print(f"{'='*80}\n")
    
    return clean_df

# Ejemplo de uso
if __name__ == "__main__":
    # Este código se ejecutaría si corres el script directamente
    variable_name = "PRICE_Australia_10Y_Bond_bond"
    # Ejecutar el análisis completo
    df = run_complete_eda(filepath, variable_name)