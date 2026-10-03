"""Model-agnostic exact-weight MoE execution for supported Transformers models.

Keep package import light so metadata-only checkpoint inspection and hardware
planning do not import Torch/Transformers until those APIs are called.
"""

__all__ = ["NeuralBackend", "inspect_checkpoint"]


def __getattr__(name):
    if name == "NeuralBackend":
        from .backend import NeuralBackend
        return NeuralBackend
    if name == "inspect_checkpoint":
        from .adapters import inspect_checkpoint
        return inspect_checkpoint
    raise AttributeError(name)
