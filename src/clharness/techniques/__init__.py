"""Technique plugins. One module per technique in this package; each registers a hook class.

    from clharness.techniques import register
    from clharness.train.hooks import BaseHook

    @register("replay")
    class Replay(BaseHook):
        name = "replay"
        def __init__(self, share: float = 0.05, source: str = "general"): ...

A run spec turns a technique on by name and passes its `params` as keyword arguments:

    {"name": "replay", "params": {"share": 0.05}}

Every module in this package is imported the first time a technique is looked up, so adding a
file is enough; nothing else needs editing.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

REGISTRY: dict[str, Callable[..., Any]] = {}
_loaded = False


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in REGISTRY and REGISTRY[name] is not factory:
            raise ValueError(f"technique {name!r} is registered twice")
        REGISTRY[name] = factory
        return factory

    return deco


def _load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    _loaded = True


def available() -> list[str]:
    _load_plugins()
    return sorted(REGISTRY)


def build_hooks(techniques: list[Any]) -> list[Any]:
    """One hook per `TechniqueSpec`, in the order given."""
    _load_plugins()
    unknown = [t.name for t in techniques if t.name not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown technique(s) {unknown}; available: {sorted(REGISTRY)}")
    return [REGISTRY[t.name](**t.params) for t in techniques]
