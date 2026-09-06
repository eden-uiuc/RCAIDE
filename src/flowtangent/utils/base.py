# src/eden_trace/utils/base.py
from typing import Any, dataclass_transform

import equinox as eqx
import jax.numpy as jnp


def null_step(*args):
    """A generic no-op function that returns its inputs unchanged."""
    return args

def field(initializer: Any, as_value: bool = False, **kwargs):
    """Smart wrapper for eqx.field that auto-routes default vs default_factory."""
    if as_value:
        return eqx.field(default=initializer, **kwargs)
    if callable(initializer):
        return eqx.field(default_factory=initializer, **kwargs)
    if isinstance(initializer, (list, dict, set)):
        raise ValueError(
            f"Mutable instance {initializer} passed to init_field. "
            "Pass the uninstantiated class (e.g., list) or a lambda instead."
        )
    return eqx.field(default=initializer, **kwargs)

def static_field(*args, **kwargs):
    return field(*args, static=True, **kwargs)

def method_field(*args, **kwargs):
    return field(*args, as_value=True, static=True, **kwargs)

def empty_array(shape: tuple | int = 0, dtype: Any = float, **kwargs):
    """Syntactic sugar for an empty JAX array in an Equinox module."""
    return field(lambda: jnp.empty(shape, dtype=dtype), **kwargs)

# Instruct IDEs to treat our custom base class as a dataclass generator
@dataclass_transform(field_specifiers=(eqx.field, field, static_field, method_field))
class Module(eqx.Module):
    """Base class for all FlowTangent modules to preserve IDE autocompletion."""
    name: str

    def __init_subclass__(cls, **kwargs) -> None:
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

        super().__init_subclass__(**kwargs)

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def field_name(self):
        return self.name.replace(" ", "_").lower()

# Metaclass logic from earlier
class StateDataMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace):
        annotations = namespace.get('__annotations__', {})
        for key, hint in annotations.items():
            if key.startswith("__"):
                continue

            hint_str = str(hint)
            if ("ndarray" in hint_str or "Array" in hint_str) and key not in namespace:
                namespace[key] = empty_array()

        return super().__new__(mcs, name, bases, namespace)

class StateData(Module, metaclass=StateDataMeta):
    pass
