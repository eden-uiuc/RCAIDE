# flowtangent/Framework/Missions/Conditions/Aerodynamics.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Aug 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# package imports

from flowtangent.core._state_data import StateData

# Flowtangent imports
from ...utils import field
from ...utils.typing import TimeScalar, _

# ----------------------------------------------------------------------------------------------------------------------
#  Aerodynamics
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
#  Coefficients
# ----------------------------------------------------------

# Component-Level Bookkeeping ------------------------------


class ComponentCoeffs(StateData):
    name: str = field("Component Coefficients", static=True)

    total: TimeScalar = _

    wings: TimeScalar = _
    fuselages: TimeScalar = _
    nacelles: TimeScalar = _


# Lift Coefficients ----------------------------------------


class LiftCoeffs(StateData):
    name: str = field("Lift Coefficients", static=True)

    total: TimeScalar = _

    inviscid: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Inviscid Lift"))
    compressible: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Compressible Lift"))


# Drag Coefficients ----------------------------------------


class InducedDrag(StateData):
    name: str = field("Induced Drag", static=True)

    total: TimeScalar = _

    inviscid: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Inviscid Induced Drag"))
    viscous: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Viscous Induced Drag"))
    near_field: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Near-Field Induced Drag"))
    far_field: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Far-Field Induced Drag"))


class DragCoeffs(StateData):
    # Attribute     Type            Default Value
    name: str = field("Drag Coefficients", static=True)

    total: TimeScalar = _

    parasite: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Parasite Drag"))
    compressible: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Compressible Drag"))
    miscellaneous: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Miscellaneous Drag"))
    spoiler: ComponentCoeffs = field(lambda: ComponentCoeffs(name="Spoiler Drag"))

    induced: InducedDrag = field(InducedDrag)


# Moment Coefficients --------------------------------------


class MomentCoeffs(StateData):
    # Attribute         Type            Default Value
    name: str = field("Moment Coefficients", static=True)

    pitch: TimeScalar = _
    roll: TimeScalar = _
    yaw: TimeScalar = _


# All Coefficients -----------------------------------------


class Coefficients(StateData):
    # Attribute         Type                Default Value
    name: str = field("Aerodynamic Coefficients", static=True)

    lift: LiftCoeffs = field(LiftCoeffs)
    drag: DragCoeffs = field(DragCoeffs)

    moments: MomentCoeffs = field(MomentCoeffs)

    X: TimeScalar = _
    Y: TimeScalar = _
    Z: TimeScalar = _


# ----------------------------------------------------------
#  Aerodynamic Angles
# ----------------------------------------------------------


class Angles(StateData):
    # Attribute         Type        Default Value
    name: str = field("Aerodynamic Angles", static=True)

    alpha: TimeScalar = _  # Y-axis / angle of attack
    beta: TimeScalar = _  # Z-axis / sideslip angle
    phi: TimeScalar = _  # X-axis / roll angle


# ----------------------------------------------------------
#  Full Aerodynamic Conditions
# ----------------------------------------------------------


class Aerodynamics(StateData):
    # Attribute     Type                    Default Value

    angles: Angles = field(Angles)
    coefficients: Coefficients = field(Coefficients)
