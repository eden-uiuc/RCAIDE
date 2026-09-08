# flowtangent/Library/Methods/Mass/Correlation/Transport/landing_gear.py
# (c) Copyright 2024 Aerospace Research Community LLC
# Created:  May 2024, J. Smart
# Modified: Mar 2026, J. Smart
# -------------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------------

from typing import TYPE_CHECKING

# package imports

if TYPE_CHECKING:
    from flowtangent.framework import Aircraft, Settings, State

from ....utils import update

# -------------------------------------------------------------------------------
#  Functional/Library Version
# -------------------------------------------------------------------------------


def func_landing_gear(MTOW: float, lg_wt_factor: float = 0.04):

    return MTOW * lg_wt_factor


# -------------------------------------------------------------------------------
#  Stateful/Framework Version
# -------------------------------------------------------------------------------


def landing_gear(state: "State", system: "Aircraft", settings: "Settings"):
    """
    Framework version of landing_gear

    See Also
    --------
    func_landing_gear:
        Functional implementation which this method calls.
    """

    lg_mass = func_landing_gear(system.mass_properties.max_takeoff)

    updated_system = update(system, "landing_gear.mass_properties.total", lg_mass)

    return state, updated_system, settings
