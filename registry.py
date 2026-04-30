"""Model registry — merges core stdlib models with optional plugin models.

A model entry is a dict: {"embed": Callable[[str], list[float]], "owned_by": str}.
Plugins live in plugins/*.py and expose a `register() -> dict[str, entry]` function;
if their backing library isn't installed, they return {} and are silently skipped.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable, Dict, List

from embeddings import MODELS as _CORE_FNS, count_tokens  # re-export

EmbedFn = Callable[[str], List[float]]
ModelEntry = Dict[str, object]  # {"embed": EmbedFn, "owned_by": str}

CORE_OWNER = "simple-embedding-mock"

MODELS: Dict[str, ModelEntry] = {
    name: {"embed": fn, "owned_by": CORE_OWNER} for name, fn in _CORE_FNS.items()
}

PLUGINS_LOADED: List[tuple] = []   # [(plugin_name, [model_ids])]
PLUGINS_SKIPPED: List[tuple] = []  # [(plugin_name, reason)]

_PLUGIN_MODULES = (
    "plugins.sentence_transformers_plugin",
    "plugins.sklearn_plugin",
)


def _load_plugins() -> None:
    for module_name in _PLUGIN_MODULES:
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            PLUGINS_SKIPPED.append((module_name, f"plugin import failed: {exc}"))
            continue
        register = getattr(mod, "register", None)
        if not callable(register):
            PLUGINS_SKIPPED.append((module_name, "no register() function"))
            continue
        try:
            added = register()
        except Exception as exc:  # plugin-side failure shouldn't crash server
            PLUGINS_SKIPPED.append((module_name, f"register() raised: {exc}"))
            continue
        if not added:
            PLUGINS_SKIPPED.append((module_name, "library not installed or no models"))
            continue
        for mid, entry in added.items():
            if mid in MODELS:
                sys.stderr.write(f"warning: plugin {module_name} model id '{mid}' collides with existing — skipping\n")
                continue
            MODELS[mid] = entry
        PLUGINS_LOADED.append((module_name, list(added.keys())))


_load_plugins()


__all__ = ["MODELS", "count_tokens", "PLUGINS_LOADED", "PLUGINS_SKIPPED", "CORE_OWNER"]
