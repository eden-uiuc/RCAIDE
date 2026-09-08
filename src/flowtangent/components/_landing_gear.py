# flowtangent/Library/Components/Landing_Gear.py
# (c) Copyright 2025 Aerospace Research Community LLC
#
# Created: May, 2025, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

# Flowtangent imports
from ..core._component import Component
from ..utils import static_field

# ----------------------------------------------------------------------------------------------------------------------
# Landing_Gear
# ----------------------------------------------------------------------------------------------------------------------


class LandingGear(Component):
    name: str = static_field("Landing Gear")

    deployed: bool = False

    number_of_units: int =  static_field(1)
    number_of_wheels: int = static_field(0)

    strut_length: float = 0.0
    tire_diameter: float = 0.0
