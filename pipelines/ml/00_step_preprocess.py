from __future__ import annotations

import os
import sys
from pathlib import Path

from sp500_analysis.application.preprocessing.factory import ProcessorFactory
from sp500_analysis.config.settings import settings

# Configurar las rutas del proyecto correctamente
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "0_raw"
CONFIG_FILE = PROJECT_ROOT / "data" / "Data Engineering.xlsx"
LOG_DIR = PROJECT_ROOT / "logs"

# Asegurar que exista el directorio de logs
LOG_DIR.mkdir(exist_ok=True)

# Agregar el proyecto al path para importar el step_0_preprocess.py
sys.path.insert(0, str(PROJECT_ROOT))

# Importar tu función completa
try:
    from step_0_preprocess import ejecutar_todos_los_procesadores_v2
    print("✅ Usando tu código completo step_0_preprocess.py")
except ImportError as e:
    print(f"❌ Error importando step_0_preprocess.py: {e}")
    sys.exit(1)


def _run_processor(name: str, output_file: Path, config_file: Path | None = None) -> bool:
    """Execute a configured processor and store its output.

    Parameters
    ----------
    name:
        Identifier of the processor to run.
    output_file:
        Path where the processed file will be stored.
    config_file:
        Optional path to a configuration file. When ``None`` the default
        ``CONFIG_FILE`` is used.

    Returns
    -------
    bool
        ``True`` if the processor completed successfully, ``False`` otherwise.
    """

    log_file = LOG_DIR / f"{name}.log"
    processor = ProcessorFactory.get_processor(
        name,
        str(config_file or CONFIG_FILE),
        str(DATA_ROOT),
        str(log_file),
    )
    return processor.run(str(output_file))


def run_investing_cp() -> bool:
    """Process Investing data using the CP configuration."""
    return _run_processor(
        "investing",
        DATA_ROOT / "datos_economicos_procesados_cp.xlsx",
    )


def run_investing_normal() -> bool:
    """Process Investing data using the normal configuration."""
    return _run_processor(
        "investing",
        DATA_ROOT / "datos_economicos_normales_procesados.xlsx",
    )


def run_fred_data_processor() -> bool:
    """Run the FRED data processor."""
    return _run_processor("fred", DATA_ROOT / "datos_economicos_procesados_Fred.xlsx")


def run_otherdataprocessor() -> bool:
    """Run processor for the EOE dataset."""
    return _run_processor("eoe", DATA_ROOT / "datos_economicos_other_procesados.xlsx")


def run_banco_republica_processor() -> bool:
    """Run the Banco República processor."""
    return _run_processor(
        "banco_republica",
        DATA_ROOT / "datos_banco_republica_procesados.xlsx",
    )


def run_dane_exportaciones_processor() -> bool:
    """Run the DANE exportaciones processor."""
    return _run_processor(
        "dane",
        DATA_ROOT / "datos_dane_exportaciones_procesados.xlsx",
    )


def ejecutar_todos_los_procesadores() -> bool:
    """Execute all preprocessing steps using your complete step_0_preprocess.py with correct paths"""
    
    print("🚀 INICIANDO STEP 0: PREPROCESS - USANDO TU CÓDIGO COMPLETO")
    print("=" * 60)
    print(f"📂 PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"📂 CONFIG_FILE: {CONFIG_FILE}")
    print(f"📂 DATA_ROOT: {DATA_ROOT}")
    print(f"📂 LOG_DIR: {LOG_DIR}")
    print("=" * 60)
    
    # Verificar que los archivos necesarios existen
    if not CONFIG_FILE.exists():
        print(f"❌ ERROR: No se encuentra {CONFIG_FILE}")
        return False
        
    if not DATA_ROOT.exists():
        print(f"❌ ERROR: No se encuentra {DATA_ROOT}")
        return False
    
    try:
        # Llamar tu función pero con parámetros corregidos para las rutas del proyecto
        # Tu función usa rutas relativas, así que necesitamos ejecutarla desde la raíz del proyecto
        current_dir = os.getcwd()
        
        try:
            # Cambiar al directorio raíz del proyecto
            os.chdir(str(PROJECT_ROOT))
            
            # Ejecutar tu función completa
            resultado = ejecutar_todos_los_procesadores_v2()
            
            if resultado:
                print("🎉 TODOS LOS PROCESADORES COMPLETADOS EXITOSAMENTE")
            else:
                print("⚠️  ALGUNOS PROCESADORES FALLARON")
                
            return resultado
            
        finally:
            # Volver al directorio original
            os.chdir(current_dir)
        
    except Exception as e:
        print(f"❌ ERROR ejecutando tu código completo: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> bool:
    """Entry point for manual execution."""
    return ejecutar_todos_los_procesadores()


if __name__ == "__main__":
    if not main():
        sys.exit(1)
