"""OpenTouch to FastWAMForce data adaptation."""

from .canonical import (
    ACTION_DIM,
    DATA_VERSION,
    PROPRIO_DIM,
    calibrate_and_clean_force,
    canonicalize_hand_trajectory,
    normalize_force,
    pressure_to_two_finger,
)

__all__ = [
    "ACTION_DIM",
    "DATA_VERSION",
    "PROPRIO_DIM",
    "calibrate_and_clean_force",
    "canonicalize_hand_trajectory",
    "normalize_force",
    "pressure_to_two_finger",
]
