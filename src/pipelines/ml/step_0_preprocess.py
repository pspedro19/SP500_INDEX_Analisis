import os
from pathlib import Path

from src.data.sources.myinvesting.copy_paste import EconomicDataProcessor
from src.data.sources.myinvesting.normal import MyinvestingreportNormal
from src.data.sources.fred.processor import FredDataProcessor
from src.data.sources.other.processor import OtherDataProcessor
from src.data.sources.bancorepublica.processor import BancoRepublicaProcessor
from src.data.sources.dane.exportaciones import DANEExportacionesProcessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def run_economic_data_processor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_procesados_cp.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/myinvestingreportcp.log')
):
    processor = EconomicDataProcessor(config_file, data_root, log_file)
    return processor.run(output_file)


def ejecutar_myinvestingreportnormal(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_normales_procesados.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/myinvestingreportnormal.log')
):
    procesador = MyinvestingreportNormal(config_file, data_root, log_file)
    return procesador.ejecutar_proceso_completo(output_file)


def run_fred_data_processor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_procesados_Fred.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/freddataprocessor.log')
):
    processor = FredDataProcessor(config_file, data_root, log_file)
    return processor.run(output_file)


def ejecutar_otherdataprocessor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_economicos_other_procesados.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/otherdataprocessor.log')
):
    procesador = OtherDataProcessor(config_file, data_root, log_file)
    return procesador.ejecutar_proceso_completo(output_file)


def ejecutar_banco_republica_processor(
    config_file=os.path.join(PROJECT_ROOT, 'pipelines/Data Engineering.xlsx'),
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_banco_republica_procesados.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/banco_republica.log')
):
    processor = BancoRepublicaProcessor(data_root, log_file)
    return processor.run(output_file)


def ejecutar_dane_exportaciones_processor(
    output_file=os.path.join(PROJECT_ROOT, 'data/0_raw/datos_dane_exportaciones_procesadas.xlsx'),
    data_root=os.path.join(PROJECT_ROOT, 'data/0_raw'),
    log_file=os.path.join(PROJECT_ROOT, 'logs/dane_exportaciones.log')
):
    processor = DANEExportacionesProcessor(data_root, log_file)
    return processor.run(output_file)


def ejecutar_todos_los_procesadores():
    return {
        'MyInvesting CP': run_economic_data_processor(),
        'MyInvesting Normal': ejecutar_myinvestingreportnormal(),
        'FRED': run_fred_data_processor(),
        'Other': ejecutar_otherdataprocessor(),
        'Banco República': ejecutar_banco_republica_processor(),
        'DANE Exportaciones': ejecutar_dane_exportaciones_processor(),
    }

if __name__ == '__main__':
    resultados = ejecutar_todos_los_procesadores()
    for nombre, exito in resultados.items():
        estado = '✅' if exito else '❌'
        print(f"{estado} {nombre}")
