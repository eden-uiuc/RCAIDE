# flowtangent/Framework/Missions/Conditions/Freestream.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax

from flowtangent.core._state_data import StateData
from flowtangent.data.atmospheres import Atmosphere, USStandard1976
from flowtangent.data.planets import Earth, Planet

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

# ----------------------------------------------------------------------------------------------------------------------
#  Freestream
# ----------------------------------------------------------------------------------------------------------------------


@register
class Freestream(StateData):
    """
    Represents the freestream conditions in a flight environment.

    This class encapsulates various atmospheric and flight parameters that define
    the freestream conditions for aerodynamic analysis or simulation.

    Attributes
    ----------
    name : str, optional
        Name of the freestream condition. Default is 'Freestream'.

    velocity : jax.Array, optional
        Velocity (speed) of the freestream. Default is empty(0).
    u : jax.Array, optional
        X-component of velocity. Default is empty(0).
    v : jax.Array, optional
        Y-component of velocity. Default is empty(0).
    w : jax.Array, optional
        Z-component of velocity. Default is empty(0).

    altitude : jax.Array, optional
        Altitude of the freestream condition. Default is empty(0).

    gravity : jax.Array, optional
        Gravitational acceleration. Default is empty(0).

    pressure : jax.Array, optional
        Atmospheric pressure. Default is empty(0).
    temperature : jax.Array, optional
        Atmospheric temperature. Default is empty(0).
    density : jax.Array, optional
        Air density. Default is empty(0).

    speed_of_sound : jax.Array, optional
        Speed of sound in the atmosphere. Default is empty(0).

    dynamic_viscosity : jax.Array, optional
        Dynamic viscosity of the air. Default is empty(0).
    dynamic_pressure : jax.Array, optional
        Dynamic pressure of the freestream. Default is empty(0).

    mach_number : jax.Array, optional
        Mach number of the freestream. Default is empty(0).
    reynolds_number : jax.Array, optional
        Reynolds number of the flow. Default is empty(0).

    delta_ISA : jax.Array, optional
        Deviation from International Standard Atmosphere. Default is empty(0).

    Notes
    -----
    All attributes are initialized as zero arrays of shape (1, 1) by default.
    """

    name: str = field("Freestream", static=True)
    atmosphere: Atmosphere = field(USStandard1976)
    planet: Planet = field(Earth)

    speed: jax.Array = empty_array()
    altitude: jax.Array = empty_array()
    gravity: jax.Array = empty_array()

    speed_of_sound: jax.Array = empty_array()
    pressure: jax.Array = empty_array()
    temperature: jax.Array = empty_array()
    density: jax.Array = empty_array()

    dynamic_viscosity: jax.Array = empty_array()
    dynamic_pressure: jax.Array = empty_array()

    stagnation_pressure: jax.Array = empty_array()
    stagnation_temperature: jax.Array = empty_array()

    mach_number: jax.Array = empty_array()
    reynolds_number: jax.Array = empty_array()

    delta_ISA: jax.Array = empty_array()  # noqa: N815
    gamma: jax.Array = empty_array()
    Cp: jax.Array = empty_array()
    R: jax.Array = empty_array()
