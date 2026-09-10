# flowtangent/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""Flowtangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from .utils.backend import initialize_jax_cache

initialize_jax_cache(
    cache_dir="~/.flowtangent/jax_cache",
    max_size_gb=2.0,
    max_age_days=30,
)

# 1. Early Boot (Must happen first)
from .utils.backend import numerical_environment, initialize_jax_cache

numerical_environment()

# Framework Hoists
from .core._state import State
from .core._systems import System, Aircraft
from .core._component import Component
from .core._processes import Process, ProcessStep, array_barrier

# Utility Hoists
from .utils import (
    update,
    TreePath,
    field,
    static_field,
    method_field,
    null_step,
    Module,
)

from .data import units

from .solve import (
    BatchedAnalysis,
    ImplicitAnalysis,
    PACTAnalysis,
)

# 4. Short-Name Namespace Routing
from . import functional as F  # noqa: N812
from . import components as comp
from . import data
from . import solve as solve
from . import sim as sim
from . import utils as utils
from . import plots as plots

from .core._settings import Settings

__all__ = [
    # FlowTangent Classes
    "Module",
    "State",
    "System",
    "Aircraft",
    "Settings",
    "Component",
    "Process",
    "ProcessStep",
    "TreePath",
    # Key utilities
    "update",
    "field",
    "static_field",
    "method_field",
    "null_step",
    "units",
    "array_barrier",
    # Analyses
    "BatchedAnalysis",
    "ImplicitAnalysis",
    "PACTAnalysis",
    # Submodules
    "F",
    "comp",
    "data",
    "solve",
    "sim",
    "utils",
    "plots",
]
