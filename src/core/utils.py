import logging
import pandas as pd
import re
from datetime import datetime

def configure_logging(log_file: str, logger_name: str = __name__):
    """Set up a basic logger writing both to a file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(logger_name)


def convertir_valor(valor, variable=None, formatos_conocidos=None):
    """Convert any numeric representation to ``float`` if possible."""
    if isinstance(valor, (int, float)):
        return float(valor)

    if not isinstance(valor, str) or valor is None:
        return None

    valor_limpio = valor.strip()
    if not valor_limpio:
        return None

    if variable and formatos_conocidos and variable in formatos_conocidos:
        formato = formatos_conocidos[variable]
        if formato == 'europeo':
            valor_limpio = valor_limpio.replace('.', '')
            valor_limpio = valor_limpio.replace(',', '.')

    multiplicadores = {'%': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    multiplicador = 1
    for sufijo, mult in multiplicadores.items():
        if valor_limpio.endswith(sufijo):
            valor_limpio = valor_limpio.replace(sufijo, '')
            multiplicador = mult
            break

    if ',' in valor_limpio and '.' in valor_limpio:
        if valor_limpio.rfind(',') > valor_limpio.rfind('.'):
            valor_limpio = valor_limpio.replace('.', '')
            valor_limpio = valor_limpio.replace(',', '.')
        else:
            valor_limpio = valor_limpio.replace(',', '')
    elif ',' in valor_limpio:
        partes = valor_limpio.split(',')
        if len(partes) == 2 and len(partes[1]) <= 2:
            valor_limpio = valor_limpio.replace(',', '.')
        else:
            valor_limpio = valor_limpio.replace(',', '')

    try:
        return float(valor_limpio) * multiplicador
    except (ValueError, TypeError):
        return None


def detectar_formato_fecha_inteligente(df, col_fecha, muestra_registros=10):
    """Return a dict with the likely dayfirst option for ``pd.to_datetime``."""
    fecha_actual = pd.Timestamp(datetime.now().date())
    muestras = df[col_fecha].dropna().astype(str).head(muestra_registros).tolist()

    resultados = {
        'dayfirst': {'validas': 0, 'invalidas': 0, 'futuras': 0},
        'no_dayfirst': {'validas': 0, 'invalidas': 0, 'futuras': 0}
    }

    for fecha_str in muestras:
        fecha_str = fecha_str.strip()
        ambigua = False
        if re.match(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}$', fecha_str):
            separador = re.findall(r'[\/\-\.]', fecha_str)[0]
            partes = fecha_str.split(separador)
            if len(partes) == 3:
                try:
                    p1, p2 = int(partes[0]), int(partes[1])
                    if p1 <= 12 and p2 <= 12:
                        ambigua = True
                except Exception:
                    pass

        for modo, dayfirst in [('dayfirst', True), ('no_dayfirst', False)]:
            try:
                fecha = pd.to_datetime(fecha_str, dayfirst=dayfirst)
                if fecha > fecha_actual + pd.Timedelta(days=30):
                    resultados[modo]['futuras'] += 1
                else:
                    resultados[modo]['validas'] += 1
            except Exception:
                resultados[modo]['invalidas'] += 1

    score_dayfirst = resultados['dayfirst']['validas'] - (
        resultados['dayfirst']['invalidas'] * 0.5) - (
        resultados['dayfirst']['futuras'] * 2)
    score_no_dayfirst = resultados['no_dayfirst']['validas'] - (
        resultados['no_dayfirst']['invalidas'] * 0.5) - (
        resultados['no_dayfirst']['futuras'] * 2)
    usar_dayfirst = score_dayfirst > score_no_dayfirst
    confianza = abs(score_dayfirst - score_no_dayfirst) / (muestra_registros * 2)
    return {'dayfirst': usar_dayfirst, 'confianza': confianza}


def convertir_fecha_adaptativo(fecha_str, configuracion_archivo=None):
    """Convert dates using the detected format configuration."""
    if isinstance(fecha_str, (pd.Timestamp, datetime)):
        return pd.Timestamp(fecha_str)
    if pd.isna(fecha_str):
        return None
    fecha_str = str(fecha_str).strip()
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', fecha_str):
        try:
            fecha = pd.to_datetime(fecha_str, format='%d.%m.%Y', dayfirst=True)
            return fecha
        except Exception:
            pass

    if configuracion_archivo is not None:
        try:
            fecha = pd.to_datetime(fecha_str, dayfirst=configuracion_archivo['dayfirst'])
            return fecha
        except Exception:
            pass

    try:
        fecha = pd.to_datetime(fecha_str)
        return fecha
    except Exception:
        return None
