# src/flowtangent/utils/base.py
import inspect
from typing import Any, dataclass_transform, get_args

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped

from .typing import _Placeholder


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


FLOWTANGENT_REGISTRY = {}


@dataclass_transform(field_specifiers=(eqx.field, field, static_field, method_field))
class Module(eqx.Module):
    """Base class for all FlowTangent modules to preserve IDE autocompletion."""

    name: str

    def __init_subclass__(cls, **kwargs) -> None:
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

        if cls.__name__ in FLOWTANGENT_REGISTRY:
            raise ValueError(
                f"Class '{cls.__name__}' is already registered. Ensure all FlowTangent module class names are unique."
            )
        FLOWTANGENT_REGISTRY[cls.__name__] = cls

        # Auto-apply jaxtyped to all standard methods that have type annotations
        for attr_name, attr_value in cls.__dict__.items():
            # Skip dunder methods (__init__, __call__, etc.) to avoid breaking Equinox
            if inspect.isfunction(attr_value) and not attr_name.startswith("__"):
                annotations = getattr(attr_value, "__annotations__", {})
                # If the method has any annotations (return or args), wrap it
                if annotations:
                    wrapped_method = jaxtyped(typechecker=beartype)(attr_value)
                    setattr(cls, attr_name, wrapped_method)

        super().__init_subclass__(**kwargs)

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def field_name(self):
        return self.name.replace(" ", "_").lower()


class StateDataMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace):
        annotations = namespace.get("__annotations__", {})
        for key, hint in annotations.items():
            if key.startswith("__"):
                continue

            args = get_args(hint)
            hint_str = str(hint) + "".join(str(a) for a in args)

            # Check if it has NO default OR if the user used the Ellipsis placeholder
            val = namespace.get(key)
            if key not in namespace or isinstance(val, _Placeholder):
                if "ndarray" in hint_str or "Array" in hint_str:
                    # Deduce the correct placeholder shape directly from the type hint!
                    shape = (0,)
                    if "time 1" in hint_str:
                        shape = (0, 1)
                    elif "time 3" in hint_str:
                        shape = (0, 3)

                    namespace[key] = empty_array(shape)

        return super().__new__(mcs, name, bases, namespace)
