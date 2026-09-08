from flowtangent.utils import field

from .. import Module, Process, ProcessStep
from ..functional.mass import transport as t_mass

# API

__all__ = [
    "ReductionFactors",
    "SizingFractions",
    "MassAnalysisSettings",
    "Transport",
]

# Settings

class ReductionFactors(Module):
    main_wing: float = 0.0
    fuselage: float = 0.0
    empennage: float = 0.0
    systems: float = 0.0


class SizingFractions(Module):
    rudder_sizing: float = field(0.25, static=True)


class MassAnalysisSettings(Module):
    reduction_factors: ReductionFactors = field(ReductionFactors)
    sizing_fractions: SizingFractions = field(SizingFractions)

# Transport Mass Analysis


def _default_transport_steps() -> tuple[ProcessStep, ...]:
    """Builds the static pipeline of turbofan cycle analysis steps."""
    return (
        # ProcessStep(name="Propulsion Mass", function=tf_mass_from_SLS),
        ProcessStep(name="Passenger & Payload Mass", function=t_mass.passenger_payload),
        ProcessStep(name="Operating System Mass", function=t_mass.operating_systems),
        ProcessStep(name="Main Wing Mass", function=t_mass.segmented_main_wing),
        ProcessStep(name="Horizontal Tail Mass", function=t_mass.horizontal_tail),
        ProcessStep(name="Vertical Tail Mass", function=t_mass.vertical_tail),
        ProcessStep(name="Fuselage Mass", function=t_mass.fuselage),
        ProcessStep(name="Landing Gear", function=t_mass.landing_gear),
    )


class Transport(Process):
    name: str = field("Transport Mass Analysis", static=True)
    steps: tuple[ProcessStep, ...] = field(_default_transport_steps)
