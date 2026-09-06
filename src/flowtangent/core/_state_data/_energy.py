# ruff: noqa: N815
# flowtangent/Framework/Missions/Conditions/Energy.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import jax.numpy as jnp

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

    work: jnp.ndarray = empty_array()
    power: jnp.ndarray = empty_array()


@register
class ElectricalOutputs(StateData):
    name: str = field("Electrical Outputs", static=True)

    power: jnp.ndarray = empty_array()
    voltage: jnp.ndarray = empty_array()
    current: jnp.ndarray = empty_array()


@register
class FuelOutputs(StateData):
    name: str = field("Fuel Outputs", static=True)

    TSFC: jnp.ndarray = empty_array()
    flow_rate: jnp.ndarray = empty_array()


@register
class FlowOutputs(StateData):
    name: str = field("Flow Outputs", static=True)
    fluid: Gas = field(Air)

    speed: jnp.ndarray = empty_array()
    speed_of_sound: jnp.ndarray = empty_array()
    mach_number: jnp.ndarray = empty_array()
    reynolds_number: jnp.ndarray = empty_array()

    pressure: jnp.ndarray = empty_array()
    temperature: jnp.ndarray = empty_array()
    enthalpy: jnp.ndarray = empty_array()

    stagnation_pressure: jnp.ndarray = empty_array()
    stagnation_temperature: jnp.ndarray = empty_array()
    stagnation_enthalpy: jnp.ndarray = empty_array()

    area: jnp.ndarray = empty_array()
    density: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()
    fuel_air_ratio: jnp.ndarray = empty_array()

    dynamic_viscosity: jnp.ndarray = empty_array()
    dynamic_pressure: jnp.ndarray = empty_array()

    gamma: jnp.ndarray = empty_array()
    Cp: jnp.ndarray = empty_array()
    R: jnp.ndarray = empty_array()

@register
class ResidualOutputs(StateData):
    name: str = field("Residual Outputs", static=True)

    mass: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()

    work: jnp.ndarray = empty_array()
    power: jnp.ndarray = empty_array()

    thrust: jnp.ndarray = empty_array()
    area: jnp.ndarray = empty_array()

    # Single Spool Turbojet Residuals
    compressor_Wc: jnp.ndarray = empty_array()
    turbine_Wp: jnp.ndarray = empty_array()

    # Dual Spool Turbofan Residuals
    fan_Wc: jnp.ndarray = empty_array()
    lpc_Wc: jnp.ndarray = empty_array()
    hpc_Wc: jnp.ndarray = empty_array()

    lpt_Wp: jnp.ndarray = empty_array()
    hpt_Wp: jnp.ndarray = empty_array()

@register
class ForceOutputs(StateData):
    name: str = field("Force Outputs", static=True)

    thrust: jnp.ndarray = empty_array()
    nondimensional_thrust: jnp.ndarray = empty_array()
    specific_impulse: jnp.ndarray = empty_array()


@register
class NodeState(StateData):
    name: str = field("Node Outputs", static=True)

    mechanical: MechanicalOutputs = field(MechanicalOutputs)
    electrical: ElectricalOutputs = field(ElectricalOutputs)
    fuel: FuelOutputs = field(FuelOutputs)
    flow: FlowOutputs = field(FlowOutputs)
    force: ForceOutputs = field(ForceOutputs)
    residual: ResidualOutputs = field(ResidualOutputs)

    mass: jnp.ndarray = empty_array()

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

    temperature: jnp.ndarray = empty_array()
    charge_throughput: jnp.ndarray = empty_array()
    state_of_charge: jnp.ndarray = empty_array()


@register
class BatteryPackConditions(NodeState):
    # Attribute             Type                    Default Value
    name: str = field("Battery Pack", static=True)

    maximum_total_energy: float = field(0.0, static=True)

    cell: BatteryCellConditions = field(BatteryCellConditions)

    temperature: jnp.ndarray = empty_array()


# ----------------------------------------------------------------------------------------------------------------------
#  Energy Networks
# ----------------------------------------------------------------------------------------------------------------------


@register
class NetworkState(NodeState):
    name: str = field("Energy Network", static=True)

    nodes: dict = field(dict)

    total_energy: jnp.ndarray = empty_array()
    total_efficiency: jnp.ndarray = empty_array()

    throttle: jnp.ndarray = empty_array()
    total_power: jnp.ndarray = empty_array()

    total_force_vector: jnp.ndarray = empty_array((0, 3))
    total_moment_vector: jnp.ndarray = empty_array((0, 3))


@register
class TurbojetState(NetworkState):

    name: str = field("Turbojet Network", static=True)

    # Control hooks
    fuel_air_ratio: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()
    rotation_speed: jnp.ndarray = empty_array()
    compressor_Rline: jnp.ndarray = empty_array()
    turbine_PR: jnp.ndarray = empty_array()

    target_thrust: jnp.ndarray = empty_array()
    target_temperature: jnp.ndarray = empty_array()

@register
class TurbofanState(NetworkState):

    name: str = field("Turbofan Network", static=True)

    # Control hooks
    fuel_air_ratio: jnp.ndarray = empty_array()
    mass_flow_rate: jnp.ndarray = empty_array()

    LP_speed: jnp.ndarray = empty_array()
    HP_speed: jnp.ndarray = empty_array()

    fan_Rline: jnp.ndarray = empty_array()
    lpc_Rline: jnp.ndarray = empty_array()
    hpc_Rline: jnp.ndarray = empty_array()

    lpt_PR: jnp.ndarray = empty_array()
    hpt_PR: jnp.ndarray = empty_array()

    bypass_ratio: jnp.ndarray = empty_array()
    target_thrust: jnp.ndarray = empty_array()
    target_temperature: jnp.ndarray = empty_array()

