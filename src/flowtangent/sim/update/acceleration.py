# flowtangent/Framework/Missions/Update/acceleration.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
# Flowtangent imports
from __future__ import annotations

import jax.numpy as jnp

from ... import Settings, State, System
from ...utils import update

# ----------------------------------------------------------------------------------------------------------------------
#  acceleration
# ----------------------------------------------------------------------------------------------------------------------


def update_acceleration(state: State, system: System, settings: Settings):

    v = state.frames.inertial.velocity_vector
    D = state.numerics.time.differentiate

    state = update(state, "frames.inertial.acceleration_vector", jnp.dot(D, v))

    return state, system, settings
