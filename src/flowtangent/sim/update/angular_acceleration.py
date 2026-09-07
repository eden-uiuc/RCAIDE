# flowtangent/Framework/Missions/Update/angular_acceleration.py
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
#  Update Angular Acceleration
# ----------------------------------------------------------------------------------------------------------------------


def update_angular_acceleration(
    state: State,
    system: System,
    settings: Settings,
):

    w = state.frames.inertial.angular_velocity_vector
    D = state.numerics.time.differentiate

    state = update(state, "frames.inertial.angular_acceleration_vector", jnp.dot(D, w))

    return state, system, settings
