"""Lazy exports for language model classes."""

from importlib import import_module
from typing import Any

__all__ = [
    "blip3oConfig",
    "blip3oLlamaForCausalLM",
    "blip3oQwenConfig",
    "blip3oQwenForCausalLM",
    "blip3oQwenForInferenceLM",
]

_MODULE_BY_SYMBOL = {
    "blip3oConfig": "blip3o.model.language_model.blip3o_llama",
    "blip3oLlamaForCausalLM": "blip3o.model.language_model.blip3o_llama",
    "blip3oQwenConfig": "blip3o.model.language_model.blip3o_qwen_inference",
    "blip3oQwenForCausalLM": "blip3o.model.language_model.blip3o_qwen",
    "blip3oQwenForInferenceLM": "blip3o.model.language_model.blip3o_qwen_inference",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_SYMBOL.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value

