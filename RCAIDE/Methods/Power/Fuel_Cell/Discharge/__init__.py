## @defgroup Methods-Power-Fuel_Cell-Discharge Discharge
# RCAIDE/Methods/Power/Fuel_Cell/Discharge/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

""" RCAIDE Package Setup
"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# @ingroup Methods-Power-Fuel_Cell

from Legacy.trunk.S.Methods.Power.Fuel_Cell.Discharge.zero_fidelity            import zero_fidelity
from Legacy.trunk.S.Methods.Power.Fuel_Cell.Discharge.larminie                 import larminie
from Legacy.trunk.S.Methods.Power.Fuel_Cell.Discharge.setup_larminie           import setup_larminie
from Legacy.trunk.S.Methods.Power.Fuel_Cell.Discharge.find_voltage_larminie    import find_voltage_larminie
from Legacy.trunk.S.Methods.Power.Fuel_Cell.Discharge.find_power_larminie      import find_power_larminie
from Legacy.trunk.S.Methods.Power.Fuel_Cell.Discharge.find_power_diff_larminie import find_power_diff_larminie