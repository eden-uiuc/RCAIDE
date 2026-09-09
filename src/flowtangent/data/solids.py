# flowtangent/Library/Attributes/Solids/Solid.py
# (c) Copyright 2023 Aerospace Research Community LLC

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

from ..utils import Module, static_field

# -------------------------------------------------------------------------------
# Solid Data Class
# -------------------------------------------------------------------------------


class Solid(Module):
    ultimate_tensile_strength: float | None =   static_field(None)
    ultimate_shear_strength: float | None =     static_field(None)
    ultimate_bearing_strength: float | None =   static_field(None)
    yield_tensile_strength: float | None =      static_field(None)
    yield_shear_strength: float | None =        static_field(None)
    yield_bearing_strength: float | None =      static_field(None)
    minimum_gage_thickness: float | None =      static_field(None)
    density: float | None =                     static_field(None)


class Aluminum(Solid):
    """
    Physical Constants Specific to 6061-T6 Aluminum

    Source:
            Cao W, Zhao C, Wang Y, et al. Thermal modeling of full-size-scale cylindrical battery pack cooled
            by channeled liquid flow[J]. International journal of heat and mass transfer, 2019, 138: 1178-1187.
    """

    density: float | None =                 static_field(2719)
    thermal_conductivity: float | None =    static_field(202.4)
    specific_heat_capacity: float | None =  static_field(871)
