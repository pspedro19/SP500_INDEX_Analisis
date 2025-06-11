from __future__ import annotations

from typing import Dict, Type, Iterator


class ModelRegistry:
    """Registry of available model wrappers."""

    def __init__(self) -> None:
        self._models: Dict[str, Type] = {}
        self.discover_models()

    def register(self, name: str, cls: Type) -> None:
        self._models[name] = cls

    def get(self, name: str) -> Type:
        return self._models[name]

    def items(self) -> Iterator[tuple[str, Type]]:
        return self._models.items()

    def discover_models(self) -> None:
        """Auto-register wrappers found in the wrappers package."""
        import inspect
        import importlib.util
        import pkgutil
        from pathlib import Path

        wrappers_dir = Path(__file__).resolve().parent / "wrappers"

        for module_info in pkgutil.iter_modules([str(wrappers_dir)]):
            module_path = wrappers_dir / f"{module_info.name}.py"
            spec = importlib.util.spec_from_file_location(module_info.name, module_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception:
                # Ignore wrappers with missing optional dependencies
                continue
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                if callable(getattr(cls, "fit", None)) and callable(getattr(cls, "predict", None)):
                    model_name = name[:-7] if name.endswith("Wrapper") else name
                    self.register(model_name, cls)


model_registry = ModelRegistry()

__all__ = ["ModelRegistry", "model_registry"]
