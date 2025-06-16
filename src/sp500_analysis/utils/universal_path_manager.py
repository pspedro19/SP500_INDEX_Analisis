"""Universal path manager que funciona en cualquier sistema y ubicación."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import platform
import getpass


class UniversalPathManager:
    """Gestor de rutas universal que detecta automáticamente la ubicación del proyecto."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.project_root = self._find_project_root()
        self.paths = self._initialize_paths()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema actual."""
        return {
            'os': platform.system(),
            'username': getpass.getuser(),
            'home_dir': Path.home(),
            'cwd': Path.cwd(),
            'python_executable': sys.executable
        }
    
    def _find_project_root(self) -> Path:
        """Busca la raíz del proyecto de forma inteligente."""
        print("🔍 Buscando raíz del proyecto...")
        
        # Estrategia 1: Buscar desde el directorio actual hacia arriba
        current = Path.cwd()
        for _ in range(10):  # Máximo 10 niveles hacia arriba
            if self._is_project_root(current):
                print(f"✅ Proyecto encontrado: {current}")
                return current
            current = current.parent
            
        # Estrategia 2: Buscar en ubicaciones comunes
        search_locations = [
            self.system_info['home_dir'] / "Desktop",
            self.system_info['home_dir'] / "Documents", 
            self.system_info['home_dir'] / "Documentos",
            Path("C:/") if platform.system() == "Windows" else Path("/"),
            Path("/Users") if platform.system() == "Darwin" else None,
            Path("/home") if platform.system() == "Linux" else None
        ]
        
        for location in search_locations:
            if location and location.exists():
                found = self._search_in_location(location)
                if found:
                    print(f"✅ Proyecto encontrado: {found}")
                    return found
                    
        # Estrategia 3: Usar directorio actual como fallback
        print(f"⚠️ Usando directorio actual como fallback: {Path.cwd()}")
        return Path.cwd()
    
    def _is_project_root(self, path: Path) -> bool:
        """Verifica si un directorio es la raíz del proyecto SP500."""
        required_indicators = [
            "src/sp500_analysis",
            "pipelines",
            "data",
            "notebooks",
            "pyproject.toml"
        ]
        
        for indicator in required_indicators:
            if not (path / indicator).exists():
                return False
        return True
    
    def _search_in_location(self, location: Path) -> Optional[Path]:
        """Busca el proyecto en una ubicación específica."""
        try:
            # Buscar directorios que contengan "SP500" o "Pipeline"
            for item in location.iterdir():
                if item.is_dir():
                    name_lower = item.name.lower()
                    if any(keyword in name_lower for keyword in ['sp500', 'pipeline']):
                        if self._is_project_root(item):
                            return item
                        # Buscar un nivel más profundo
                        for subitem in item.iterdir():
                            if subitem.is_dir() and self._is_project_root(subitem):
                                return subitem
        except (PermissionError, OSError):
            pass
        return None
    
    def _initialize_paths(self) -> Dict[str, Path]:
        """Inicializa todas las rutas del proyecto."""
        root = self.project_root
        
        return {
            'project_root': root,
            'src': root / "src",
            'pipelines': root / "pipelines", 
            'notebooks': root / "notebooks",
            'data': root / "data",
            'data_raw': root / "data" / "raw" / "0_raw",
            'data_preprocess': root / "data" / "1_preprocess",
            'data_processed': root / "data" / "2_processed",
            'data_training': root / "data" / "3_trainingdata",
            'data_results': root / "data" / "4_results",
            'data_metrics': root / "data" / "5_metrics",
            'logs': root / "logs",
            'reports': root / "reports",
            'models': root / "models",
            'config_file': root / "data" / "Data Engineering.xlsx"
        }
    
    def setup_python_path(self) -> List[str]:
        """Configura el Python path con las rutas necesarias."""
        paths_to_add = [
            str(self.paths['project_root']),
            str(self.paths['src']),
            str(self.paths['pipelines'])
        ]
        
        added_paths = []
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
                added_paths.append(path)
                
        return added_paths
    
    def change_to_project_root(self) -> Path:
        """Cambia el directorio de trabajo a la raíz del proyecto."""
        os.chdir(self.project_root)
        return self.project_root
    
    def verify_setup(self) -> Dict[str, bool]:
        """Verifica que todo esté configurado correctamente."""
        verifications = {}
        
        # Verificar existencia de directorios clave
        key_dirs = ['src', 'pipelines', 'data', 'notebooks']
        for dir_name in key_dirs:
            verifications[f'{dir_name}_exists'] = self.paths[dir_name].exists()
        
        # Verificar importaciones
        try:
            from sp500_analysis.config.settings import settings
            verifications['settings_import'] = True
        except ImportError:
            verifications['settings_import'] = False
            
        # Verificar archivo de configuración
        verifications['config_file_exists'] = self.paths['config_file'].exists()
        
        return verifications
    
    def get_path(self, key: str) -> Path:
        """Obtiene una ruta específica por su clave."""
        return self.paths.get(key, self.project_root)
    
    def print_status(self) -> None:
        """Imprime el estado actual del gestor de rutas."""
        print("=" * 60)
        print("🔧 UNIVERSAL PATH MANAGER - ESTADO")
        print("=" * 60)
        print(f"🖥️ Sistema: {self.system_info['os']}")
        print(f"👤 Usuario: {self.system_info['username']}")
        print(f"🏠 Home: {self.system_info['home_dir']}")
        print(f"📂 Directorio actual: {self.system_info['cwd']}")
        print(f"🎯 Raíz del proyecto: {self.project_root}")
        print()
        
        print("📁 RUTAS PRINCIPALES:")
        for key, path in self.paths.items():
            status = "✅" if path.exists() else "❌"
            print(f"   {status} {key}: {path}")
        print()
        
        verifications = self.verify_setup()
        print("🧪 VERIFICACIONES:")
        for check, result in verifications.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check}")
        print("=" * 60)


# Instancia global para uso fácil
path_manager = UniversalPathManager()


def setup_universal_environment() -> UniversalPathManager:
    """Configura el entorno de forma universal y retorna el path manager."""
    print("🚀 Configurando entorno universal...")
    
    # Crear instancia del gestor
    manager = UniversalPathManager()
    
    # Configurar Python path
    added_paths = manager.setup_python_path()
    print(f"✅ Agregado al Python path: {len(added_paths)} rutas")
    
    # Cambiar al directorio del proyecto
    manager.change_to_project_root()
    print(f"✅ Directorio cambiado a: {Path.cwd()}")
    
    # Mostrar estado
    manager.print_status()
    
    return manager


def get_project_root() -> Path:
    """Función helper para obtener la raíz del proyecto."""
    return path_manager.project_root


def get_data_path(subpath: str = "") -> Path:
    """Función helper para obtener rutas de datos."""
    if subpath:
        return path_manager.paths['data'] / subpath
    return path_manager.paths['data']


def ensure_directories() -> None:
    """Crea todos los directorios necesarios."""
    dirs_to_create = [
        'data_raw', 'data_preprocess', 'data_processed', 
        'data_training', 'data_results', 'data_metrics',
        'logs', 'reports', 'models'
    ]
    
    for dir_key in dirs_to_create:
        path = path_manager.paths[dir_key]
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {path}") 