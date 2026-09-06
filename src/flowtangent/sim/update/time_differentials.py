# flowtangent/Framework/Missions/Update/time_differentials.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

# package imports
import equinox as eqx

# Flowtangent imports
from ... import Settings, State, System

# ----------------------------------------------------------------------------------------------------------------------
#  Update Time Differentials
# ----------------------------------------------------------------------------------------------------------------------


def update_time_differentials(
    state: State,
    system: System,
    settings: Settings,
):

    x = state.time.dimensionless.control_points
    D = state.time.dimensionless.differentiate
    I = state.time.dimensionless.integrate

    time = state.frames.inertial.time
    T = time[-1] - time[0]
    t_scaled = x * T
    D_scaled = D / T
    I_scaled = I * T

    state = eqx.tree_at(
        lambda s: (s.numerics.time.control_points, s.numerics.time.differentiate, s.numerics.time.integrate),
        state,
        (t_scaled, D_scaled, I_scaled),
        is_leaf=lambda x: x is None,
    )

    return state, system, settings
