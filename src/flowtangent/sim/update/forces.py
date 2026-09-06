# flowtangent/Framework/Missions/Update/forces.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug, 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
# Flowtangent imports
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ... import Settings, State, System

# ----------------------------------------------------------------------------------------------------------------------
# Update Forces
# ----------------------------------------------------------------------------------------------------------------------


def update_forces(
    state: State,
    system: System,
    settings: Settings,
):

    wind = state.frames.wind.total_force_vector
    thrust = state.frames.body.thrust_force_vector
    weight = state.frames.inertial.gravity_force_vector

    TB2I = state.frames.body.transform_to_inertial
    TW2I = state.frames.wind.transform_to_inertial

    wind_force = jnp.einsum("nij,nj->ni", TW2I, wind)
    thrust_force = jnp.einsum("nij,nj->ni", TB2I, thrust)

    total_force = weight + wind_force + thrust_force

    state = eqx.tree_at(lambda s: s.frames.inertial.total_force_vector, state, total_force)

    return state, system, settings
