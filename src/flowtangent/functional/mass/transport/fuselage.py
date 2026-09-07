# flowtangent/Library/Methods/Mass/Correlation/Transport/fuselage.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created:  May 2024, J. Smart
# Modified:
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------


from __future__ import annotations

import jax
import jax.numpy as jnp

from flowtangent.data import units

from .... import Settings, State, System
from ....components import Fuselage

# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------


def func_fuselage(
    fuselage_wetted_area: jax.Array,
    fuselage_width: jax.Array,
    fuselage_maximum_height: jax.Array,
    fuselage_total_length: jax.Array,
    fuselage_differential_pressure: jax.Array,
    vehicle_limit_load: jax.Array,
    vehicle_max_zero_fuel_mass: jax.Array,
    vehicle_main_wing_mass: jax.Array,
    vehicle_main_wing_root_chord: jax.Array,
    vehicle_propulsion_mass: jax.Array,
):
    """
    Library version of fuselage.

    Parameters
    ----------
    fuselage_wetted_area : jax.Array
        Fuselage wetted area in square meters

    fuselage_width : jax.Array
        Fuselage width in meters

    fuselage_maximum_height : jax.Array
        Fuselage maximum height in meters

    fuselage_total_length : jax.Array
        Fuselage total length in meters

    fuselage_differential_pressure : jax.Array
        Fuselage differential pressure in Pascals

    vehicle_limit_load : jax.Array
        Zero fuel weight limit load factor

    vehicle_max_zero_fuel_mass : jax.Array
        Maximum vehicle zero fuel mass in kilograms

    vehicle_main_wing_mass : jax.Array
        Vehicle main wing mass in kilograms

    vehicle_main_wing_root_chord : jax.Array
        Vehicle main wing root chord in meters

    vehicle_propulsion_mass : jax.Array
        Vehicle propulsion system mass in kilograms

    Returns
    -------
    fuselage_mass : jax.Array
        Fuselage mass in kilograms

    See Also
    --------
    N/A

    Notes
    -----
    Correlation of fuselage mass with differential pressure or limit load as appropriate.

    References
    ----------

    Examples
    --------
    func_fuselage(*args):
        82000.0

    """

    # Unit Conversion

    dp = fuselage_differential_pressure / (units.lbf / units.ft**2)
    w = fuselage_width / units.ft
    h = fuselage_maximum_height / units.ft
    l = (fuselage_total_length - vehicle_main_wing_root_chord / 2.0) / units.ft
    m = (vehicle_max_zero_fuel_mass - vehicle_main_wing_mass - vehicle_propulsion_mass) / units.lbm
    S = fuselage_wetted_area / units.ft**2

    # Limiting Factor Determination

    pressure_idx = 1.50e-3 * dp * w
    geometry_idx = 1.91e-4 * vehicle_limit_load * m * l / h**2

    if pressure_idx > geometry_idx:
        limit_idx = pressure_idx
    else:
        limit_idx = (pressure_idx**2 + geometry_idx**2) / (2 * geometry_idx)

    fuselage_mass = ((1.051 + 0.102 * limit_idx) * S) * units.lbm

    return fuselage_mass


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


def fuselage(state: State, system: System, settings: Settings):
    """
    Framework version of fuselage

    See Also
    --------
    func_fuselage:
        Functional implementation which this method calls.
    """

    fuses = [f for f in system.subcomponents if isinstance(f, Fuselage)]

    fuselage_wetted_area = jnp.atleast_1d([f.areas.wetted for f in fuses])
    fuselage_width = jnp.atleast_1d([f.widths.maximum for f in fuses])
    fuselage_maximum_height = jnp.atleast_1d([f.heights.maximum for f in fuses])
    fuselage_total_length = jnp.atleast_1d([f.lengths.total for f in fuses])
    fuselage_differential_pressure = jnp.atleast_1d([f.differential_pressure for f in fuses])
    vehicle_limit_load = jnp.atleast_1d(system.envelope.limit_load)
    vehicle_max_zero_fuel_mass = jnp.atleast_1d(system.mass_properties.max_zero_fuel_mass)
    vehicle_main_wing_mass = jnp.atleast_1d(system["Main Wing"].mass_properties.total)
    vehicle_main_wing_root_chord = jnp.atleast_1d(system["Main Wing"].chords.root)
    vehicle_propulsion_mass = jnp.atleast_1d(system.energy.propulsors.propulsors.mass_properties.total)

    results = func_fuselage(
        fuselage_wetted_area,
        fuselage_width,
        fuselage_maximum_height,
        fuselage_total_length,
        fuselage_differential_pressure,
        vehicle_limit_load,
        vehicle_max_zero_fuel_mass,
        vehicle_main_wing_mass,
        vehicle_main_wing_root_chord,
        vehicle_propulsion_mass,
    )

    for idx, fuse in enumerate(fuses):
        fuse.mass_properties.total = results[idx]

    system.sum_mass()

    return state, system, settings
