"""
Script para configurar el entorno de notebook y hacer que funcionen las importaciones originales.
Ejecutar desde notebooks/ con: python setup_notebook.py
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_project_for_notebook():
    """Configura el proyecto para que funcione en notebooks"""
    
    # Obtener rutas
    current_dir = Path.cwd()
    project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
    
    print(f"📂 Directorio actual: {current_dir}")
    print(f"🎯 Raíz del proyecto: {project_root}")
    
    # Cambiar al directorio raíz
    os.chdir(project_root)
    
    # Verificar si existe pyproject.toml o setup.py
    has_pyproject = (project_root / "pyproject.toml").exists()
    has_setup = (project_root / "setup.py").exists()
    
    if has_pyproject or has_setup:
        print("📦 Instalando proyecto en modo desarrollo...")
        try:
            # Instalar en modo desarrollo
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                         check=True, capture_output=True, text=True)
            print("✅ Proyecto instalado en modo desarrollo")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando proyecto: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
    
    # Fallback: agregar rutas manualmente
    print("🔧 Usando método manual para agregar rutas...")
    
    # Crear archivo de configuración para notebooks
    notebook_init = project_root / "notebooks" / "__init__.py"
    notebook_init.write_text(f'''
import sys
from pathlib import Path

# Auto-configuración de rutas
project_root = Path(__file__).parent.parent
src_path = project_root / "src" 
pipelines_path = project_root / "pipelines"

paths = [str(project_root), str(src_path), str(pipelines_path)]
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Cambiar directorio de trabajo
import os
os.chdir(project_root)
''')
    
    print(f"✅ Creado {notebook_init}")
    
    # Verificar importaciones
    try:
        sys.path.insert(0, str(project_root / "src"))
        from sp500_analysis.config.settings import settings
        print("🎉 ¡Importaciones funcionando!")
        return True
    except ImportError as e:
        print(f"⚠️ Aún hay problemas con importaciones: {e}")
        return False

if __name__ == "__main__":
    setup_project_for_notebook() 