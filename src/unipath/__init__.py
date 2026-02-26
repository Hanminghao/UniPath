"""Top-level lazy exports for blip3o models."""

from importlib import import_module
from typing import Any

__all__ = [
    "blip3oConfig",
    "blip3oLlamaForCausalLM",
    "blip3oQwenConfig",
    "blip3oQwenForCausalLM",
    "blip3oQwenForInferenceLM",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    model_module = import_module("blip3o.model")
    value = getattr(model_module, name)
    globals()[name] = value
    return value
