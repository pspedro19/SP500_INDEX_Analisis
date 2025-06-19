from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib.util
import logging

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

def load_legacy_preprocessing_step():
    """
    Carga dinámicamente el código legacy de step_0_preprocess.py aplicando correcciones automáticas de rutas.
    
    Returns:
        tuple: (module, success_flag)
    """
    # Determinar rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    legacy_file_path = os.path.join(project_root, "src", "sp500_analysis", "application", "preprocessing", "legacy_step_0.py")
    
    if not os.path.exists(legacy_file_path):
        print(f"❌ Error: No se encuentra el archivo legacy en {legacy_file_path}")
        return None, False
    
    try:
        # Función para interceptar os.path.join y corregir rutas automáticamente
        original_join = os.path.join
        original_dirname = os.path.dirname
        original_abspath = os.path.abspath
        
        def patched_join(*args):
            # Convertir argumentos a strings y verificar si necesitan corrección
            str_args = [str(arg) for arg in args]
            
            # Buscar y corregir patrones problemáticos
            for i, arg in enumerate(str_args):
                if 'pipelines/Data Engineering.xlsx' in arg:
                    str_args[i] = arg.replace('pipelines/Data Engineering.xlsx', 'data/Data Engineering.xlsx')
                    print(f"🔧 Corrigiendo ruta: {arg} → {str_args[i]}")
            
            # Aplicar la función original con argumentos corregidos
            result = original_join(*str_args)
            return result
        
        def patched_dirname(path):
            # Si es el archivo legacy_step_0.py, devolver el directorio del proyecto
            if 'legacy_step_0.py' in str(path):
                return project_root
            return original_dirname(path)
            
        def patched_abspath(path):
            # Si es el dirname del legacy file, devolver project root
            if hasattr(patched_abspath, '_in_legacy_context') and patched_abspath._in_legacy_context:
                if str(path) == original_dirname(legacy_file_path):
                    return project_root
            return original_abspath(path)
        
        # Aplicar los patches
        os.path.join = patched_join
        os.path.dirname = patched_dirname
        os.path.abspath = patched_abspath
        patched_abspath._in_legacy_context = True
        
        # Cargar el módulo legacy
        spec = importlib.util.spec_from_file_location("legacy_step_0", legacy_file_path)
        legacy_module = importlib.util.module_from_spec(spec)
        
        # Ejecutar el módulo
        spec.loader.exec_module(legacy_module)
        
        # IMPORTANTE: Después de cargar el módulo, corregir manualmente el PROJECT_ROOT
        # que ya fue calculado incorrectamente
        if hasattr(legacy_module, 'PROJECT_ROOT'):
            legacy_module.PROJECT_ROOT = project_root
            print(f"🔧 PROJECT_ROOT corregido manualmente: {project_root}")
        
        # Restaurar funciones originales
        os.path.join = original_join
        os.path.dirname = original_dirname
        os.path.abspath = original_abspath
        
        print("✅ Código completo de legacy_step_0.py cargado con rutas corregidas")
        print(f"📁 PROJECT_ROOT final: {legacy_module.PROJECT_ROOT if hasattr(legacy_module, 'PROJECT_ROOT') else 'No definido'}")
        
        return legacy_module, True
        
    except Exception as e:
        # Restaurar funciones originales en caso de error
        os.path.join = original_join
        os.path.dirname = original_dirname  
        os.path.abspath = original_abspath
        print(f"❌ Error cargando legacy_step_0.py: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

# Importar tu función completa
try:
    legacy_module, success = load_legacy_preprocessing_step()
    if success:
        ejecutar_todos_los_procesadores_v2 = legacy_module.ejecutar_todos_los_procesadores_v2
        print("✅ Usando tu código completo step_0_preprocess.py")
    else:
        print("❌ Error importando step_0_preprocess.py")
        sys.exit(1)
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
