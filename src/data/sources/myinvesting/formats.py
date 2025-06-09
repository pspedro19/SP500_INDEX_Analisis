import pandas as pd
from src.core.utils import detectar_formato_fecha_inteligente

class FormatosFechas:
    """Gestiona y cachea la configuración de conversión de fechas por archivo."""
    def __init__(self):
        self.formatos_cache = {}  # {variable: configuracion}
    
    def detectar_formato(self, df, col_fecha, variable=None):
        configuracion = detectar_formato_fecha_inteligente(df, col_fecha)
        if variable:
            self.formatos_cache[variable] = configuracion
        return configuracion
        
    def obtener_formato(self, variable):
        return self.formatos_cache.get(variable)

