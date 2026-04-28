"""Load user plugin modules / files so their @register_* decorators run."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterable


def load_plugins(targets: Iterable[str]) -> list[str]:
    """Import each target so its registries populate.

    Targets can be:
      • a module path:  "my_strategies"  or  "my_strategies.rsi_gate"
      • a file path:    "./my_strategies/rsi_gate.py"
      • a directory:    "./my_strategies"  (every .py inside is imported)
    """
    loaded: list[str] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            sys.path.insert(0, str(p.parent.resolve()))
            for f in sorted(p.glob("*.py")):
                if f.stem == "__init__":
                    continue
                loaded.append(_load_file(f))
        elif p.is_file() and p.suffix == ".py":
            loaded.append(_load_file(p))
        else:
            importlib.import_module(t)
            loaded.append(t)
    return loaded


def _load_file(path: Path) -> str:
    name = f"_taautoml_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plugin file {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return name
