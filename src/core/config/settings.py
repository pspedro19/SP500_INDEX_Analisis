from __future__ import annotations

"""Compatibilidad con la configuración antigua."""

from src.config.base import ProjectConfig


class Settings(ProjectConfig):
    """Alias para mantener la interfaz previa."""
    pass


settings = Settings.from_env()
