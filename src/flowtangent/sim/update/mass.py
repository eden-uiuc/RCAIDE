# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Settings, State, System

# package imports
import jax.numpy as jnp

from ...utils import update

# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


def update_mass_and_weight(
    state: State,
    system: System,
    settings: Settings,
):
    """
    Updates the current mass of the system
    """

    mdot = state.mass.rate_of_change
    I = state.time.dimensional.integrate
    g = state.freestream.gravity

    # Integrate mdot
    integrated_mass = state.initials.mass.total + jnp.dot(I, mdot)
    integrated_weight = integrated_mass * g

    # Update State
    updated_state = update(
        state,
        (
            ("mass.total", integrated_mass),
            ("frames.inertial.gravity_force_vector", integrated_weight[:, 0], (slice(None, None), slice(None, 2))),
        ),
    )

    return updated_state, system, settings
