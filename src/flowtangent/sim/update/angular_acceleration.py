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

import equinox as eqx
import jax.numpy as jnp

from ... import Settings, State, System

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

    state = eqx.tree_at(lambda s: s.frames.inertial.angular_acceleration_vector, state, jnp.dot(D, w))

    return state, system, settings
