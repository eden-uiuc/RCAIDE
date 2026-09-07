# flowtangent/Library/Methods/Aerodynamics/induced_drag.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING

import jax

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from flowtangent.core._settings import Settings
    from flowtangent.core._state import State
    from flowtangent.core._systems import System

from flowtangent.utils import inputs, outputs, update

# ----------------------------------------------------------------------------------------------------------------------
#  Viscous Induced Drag
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------
# 1. PURE LIBRARY FUNCTION (Math Only)
# ---------------------------------------------------------
@jax.jit
def func_viscous_induced_drag(
    CL: float | jax.Array,
    parasite_drag: float | jax.Array,
    viscous_lift_factor: float | jax.Array = 0.38,
):
    """Evaluates viscous induced drag based on parasite drag and drag factor"""

    return viscous_lift_factor * parasite_drag * (CL**2)


# ---------------------------------------------------------
# 2. STATEFUL FRAMEWORK ROUTER
# ---------------------------------------------------------
@inputs(
    "state.aerodynamics.coefficients.lift.total",
    "state.aerodynamics.coefficients.drag.parasite.total",
    "settings.analysis.aerodynamics.correction.viscous_lift_drag",
    "state.aerodynamics.coefficients.drag.induced.inviscid.total",
    "state.aerodynamics.coefficients.drag.parasite.wings",
)
@outputs(
    "state.aerodynamics.coefficients.drag.induced.total",
    "state.aerodynamics.coefficients.drag.induced.viscous.total",
)
def compute_viscous_induced_drag(state: "State", system: "System", settings: "Settings"):
    """Computes system and wing viscous induced drag"""

    # Total System
    CL_all = state.aerodynamics.coefficients.lift.total
    CDp_all = state.aerodynamics.coefficients.drag.parasite.total
    K = settings.analysis.aerodynamics.correction.viscous_lift_drag

    CDiv_all = func_viscous_induced_drag(CL_all, CDp_all, K)
    total_induced_drag = state.aerodynamics.coefficients.drag.induced.inviscid.total + CDiv_all

    updated_induced_drag = update(
        state.aerodynamics.coefficients.drag.induced,
        (("total", total_induced_drag), ("viscous.total", CDiv_all)),
    )

    updated_state = update(state, "aerodynamics.coefficients.drag.induced", updated_induced_drag)

    return updated_state, system, settings
