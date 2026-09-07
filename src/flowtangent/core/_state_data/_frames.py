# flowtangent/Framework/Missions/Conditions/Frames.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------


# package imports
import jax
import jax.numpy as jnp

from flowtangent.core._state_data import StateData

# Flowtangent imports
from flowtangent.utils import empty_array, field, register

# ----------------------------------------------------------------------------------------------------------------------
#  Frames
# ----------------------------------------------------------------------------------------------------------------------


@register
class Frame(StateData):
    # Attribute             Type        Default Value
    name: str = field("Frame", static=True)

    transform_to_inertial: jax.Array = empty_array((0, 3))

    total_force_vector: jax.Array = empty_array((0, 3))
    total_moment_vector: jax.Array = empty_array((0, 3))


@register
class Inertial(Frame):
    # Attribute                     Type        Default Value
    name: str = field("Inertial Frame", static=True)

    position_vector: jax.Array = empty_array((0, 3))

    velocity_vector: jax.Array = empty_array((0, 3))
    acceleration_vector: jax.Array = empty_array((0, 3))

    angular_velocity_vector: jax.Array = empty_array((0, 3))
    angular_acceleration_vector: jax.Array = empty_array((0, 3))

    gravity_force_vector: jax.Array = empty_array((0, 3))

    time: jax.Array = empty_array((0))
    system_range: jax.Array = empty_array((0))


@register
class Body(Frame):
    # Attribute             Type        Default Value
    name: str = field("Body Frame", static=True)

    inertial_rotations: jax.Array = empty_array((0, 3))
    thrust_force_vector: jax.Array = empty_array((0, 3))
    moment_vector: jax.Array = empty_array((0, 3))


@register
class Wind(Frame):
    # Attribute         Type            Default Value
    name: str = field("Wind Frame", static=True)

    body_rotations: jax.Array = empty_array((0, 3))
    transform_to_body: jax.Array = empty_array((0, 3))

    velocity_vector: jax.Array = empty_array((0, 3))
    force_vector: jax.Array = empty_array((0, 3))
    moment_vector: jax.Array = empty_array((0, 3))


@register
class Planet(Frame):
    # Attribute     Type            Default Value
    name: str = field("Planet Frame", static=True)
    start_time: jax.Array = empty_array()

    # Default to takeoff at JFK
    latitude: jax.Array = field(lambda: jnp.array([40.6446]))
    longitude: jax.Array = field(lambda: jnp.array([73.7797]))

    true_course: jax.Array = empty_array()


@register
class FrameData(StateData):
    # Attribute     Type            Default Value
    name: str = field("Dynamic Frames", static=True)

    inertial: Inertial = field(Inertial)
    body: Body = field(Body)
    wind: Wind = field(Wind)
    planet: Planet = field(Planet)
