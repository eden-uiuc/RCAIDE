from flowtangent.utils import field

from ... import Process, ProcessStep
from ...functional.mass import transport as t_mass

# from Flowtangent.Framework.Methods.Mass.Energy import tf_mass_from_SLS


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
