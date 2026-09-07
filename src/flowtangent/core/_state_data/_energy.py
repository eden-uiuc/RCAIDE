# ruff: noqa: N815
# flowtangent/Framework/Missions/Conditions/Energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax

from flowtangent.core._state_data import StateData
from flowtangent.data.gases import Air, Gas

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Interfaces
# ----------------------------------------------------------------------------------------------------------------------


@register
class MechanicalOutputs(StateData):
    name: str = field("Mechanical Outputs", static=True)

    work: jax.Array = empty_array()
    power: jax.Array = empty_array()


@register
class ElectricalOutputs(StateData):
    name: str = field("Electrical Outputs", static=True)

    power: jax.Array = empty_array()
    voltage: jax.Array = empty_array()
    current: jax.Array = empty_array()


@register
class FuelOutputs(StateData):
    name: str = field("Fuel Outputs", static=True)

    TSFC: jax.Array = empty_array()
    flow_rate: jax.Array = empty_array()


@register
class FlowOutputs(StateData):
    name: str = field("Flow Outputs", static=True)
    fluid: Gas = field(Air)

    speed: jax.Array = empty_array()
    speed_of_sound: jax.Array = empty_array()
    mach_number: jax.Array = empty_array()
    reynolds_number: jax.Array = empty_array()

    pressure: jax.Array = empty_array()
    temperature: jax.Array = empty_array()
    enthalpy: jax.Array = empty_array()

    stagnation_pressure: jax.Array = empty_array()
    stagnation_temperature: jax.Array = empty_array()
    stagnation_enthalpy: jax.Array = empty_array()

    area: jax.Array = empty_array()
    density: jax.Array = empty_array()
    mass_flow_rate: jax.Array = empty_array()
    fuel_air_ratio: jax.Array = empty_array()

    dynamic_viscosity: jax.Array = empty_array()
    dynamic_pressure: jax.Array = empty_array()

    gamma: jax.Array = empty_array()
    Cp: jax.Array = empty_array()
    R: jax.Array = empty_array()


@register
class ResidualOutputs(StateData):
    name: str = field("Residual Outputs", static=True)

    mass: jax.Array = empty_array()
    mass_flow_rate: jax.Array = empty_array()

    work: jax.Array = empty_array()
    power: jax.Array = empty_array()

    thrust: jax.Array = empty_array()
    area: jax.Array = empty_array()

    # Single Spool Turbojet Residuals
    compressor_Wc: jax.Array = empty_array()
    turbine_Wp: jax.Array = empty_array()

    # Dual Spool Turbofan Residuals
    fan_Wc: jax.Array = empty_array()
    lpc_Wc: jax.Array = empty_array()
    hpc_Wc: jax.Array = empty_array()

    lpt_Wp: jax.Array = empty_array()
    hpt_Wp: jax.Array = empty_array()


@register
class ForceOutputs(StateData):
    name: str = field("Force Outputs", static=True)

    thrust: jax.Array = empty_array()
    nondimensional_thrust: jax.Array = empty_array()
    specific_impulse: jax.Array = empty_array()


@register
class NodeState(StateData):
    name: str = field("Node Outputs", static=True)

    mechanical: MechanicalOutputs = field(MechanicalOutputs)
    electrical: ElectricalOutputs = field(ElectricalOutputs)
    fuel: FuelOutputs = field(FuelOutputs)
    flow: FlowOutputs = field(FlowOutputs)
    force: ForceOutputs = field(ForceOutputs)
    residual: ResidualOutputs = field(ResidualOutputs)

    mass: jax.Array = empty_array()


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Stores
# ----------------------------------------------------------------------------------------------------------------------


@register
class BatteryCellConditions(NodeState):
    # Attribute                 Type        Default Value
    name: str = field("Battery Cell", static=True)

    cycle_in_day: int = field(0, static=True)
    resistance_growth_factor: float = field(0.0, static=True)
    capacity_fade_factor: float = field(0.0, static=True)

    temperature: jax.Array = empty_array()
    charge_throughput: jax.Array = empty_array()
    state_of_charge: jax.Array = empty_array()


@register
class BatteryPackConditions(NodeState):
    # Attribute             Type                    Default Value
    name: str = field("Battery Pack", static=True)

    maximum_total_energy: float = field(0.0, static=True)

    cell: BatteryCellConditions = field(BatteryCellConditions)

    temperature: jax.Array = empty_array()


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


@register
class NetworkState(NodeState):
    name: str = field("Energy Network", static=True)

    nodes: dict = field(dict)

    total_energy: jax.Array = empty_array()
    total_efficiency: jax.Array = empty_array()

    throttle: jax.Array = empty_array()
    total_power: jax.Array = empty_array()

    total_force_vector: jax.Array = empty_array((0, 3))
    total_moment_vector: jax.Array = empty_array((0, 3))


@register
class TurbojetState(NetworkState):
    name: str = field("Turbojet Network", static=True)

    # Control hooks
    fuel_air_ratio: jax.Array = empty_array()
    mass_flow_rate: jax.Array = empty_array()
    rotation_speed: jax.Array = empty_array()
    compressor_Rline: jax.Array = empty_array()
    turbine_PR: jax.Array = empty_array()

    target_thrust: jax.Array = empty_array()
    target_temperature: jax.Array = empty_array()


@register
class TurbofanState(NetworkState):
    name: str = field("Turbofan Network", static=True)

    # Control hooks
    fuel_air_ratio: jax.Array = empty_array()
    mass_flow_rate: jax.Array = empty_array()

    LP_speed: jax.Array = empty_array()
    HP_speed: jax.Array = empty_array()

    fan_Rline: jax.Array = empty_array()
    lpc_Rline: jax.Array = empty_array()
    hpc_Rline: jax.Array = empty_array()

    lpt_PR: jax.Array = empty_array()
    hpt_PR: jax.Array = empty_array()

    bypass_ratio: jax.Array = empty_array()
    target_thrust: jax.Array = empty_array()
    target_temperature: jax.Array = empty_array()
