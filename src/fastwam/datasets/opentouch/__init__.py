"""OpenTouch to FastWAMForce data adaptation."""

from .canonical import (
    ACTION_DIM,
    PROPRIO_DIM,
    canonicalize_hand_trajectory,
    normalize_force,
    pressure_to_two_finger,
)

__all__ = [
    "ACTION_DIM",
    "PROPRIO_DIM",
    "canonicalize_hand_trajectory",
    "normalize_force",
    "pressure_to_two_finger",
]
