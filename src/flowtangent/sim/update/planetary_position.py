# flowtangent/Framework/Missions/Update/planetary_position.py
# (c) Copyright 2026 Aerospace Research Community LLC
#
# Created: Mar 2026, J. Smart
# Modified: Mar 2026, J. Smart

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import TYPE_CHECKING

import jax.numpy as jnp

# --- Framework Imports (Strictly for Type Hinting to avoid Circular Imports) ---
if TYPE_CHECKING:
    from flowtangent.core._settings import Settings
    from flowtangent.core._state import State
    from flowtangent.core._systems import System

from ...data import units
from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
#  Update Planetary Position
# ----------------------------------------------------------------------------------------------------------------------


def update_planetary_position(state: "State", system: "System", settings: "Settings"):

    # Unpack state
    v = state.frames.inertial.velocity_vector[:, 0]  # Velocity over ground along true course
    alt = state.freestream.altitude

    theta = state.frames.body.inertial_rotations[:, 1]
    psi = state.frames.planet.true_course
    Re = state.freestream.planet.mean_radius

    alpha = state.aerodynamics.angles.alpha

    I = state.time.dimensional.integrate

    #  Calculate flight path and radius
    gamma = theta - alpha
    R = alt + Re

    # Find local velocities and integrate position
    lamdadot = (v / R) * jnp.cos(gamma) * jnp.cos(psi)
    lamda = jnp.dot(I, lamdadot) / units.deg  # Latitude

    mudot = (v / R) * jnp.cos(gamma) * jnp.sin(psi) / jnp.cos(lamda)
    mu = jnp.dot(I, mudot) / units.deg  # Longitude

    lat_0 = state.frames.planet.latitude[0, 0]
    lon_0 = state.frames.planet.longitude[0, 0]

    updated_state = update(
        state,
        (
            ("frames.planet.latitude", lat_0 + lamda),
            ("frames.planet.longitude", lon_0 + mu),
        ),
    )

    return updated_state, system, settings
