from __future__ import annotations

import numpy as np

_SWITCHED_HAND_REORDER = np.array([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)
_SWITCHED_HAND_SIGN = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def switched_hand_to_canonical(values: np.ndarray) -> np.ndarray:
    """Map a physically swapped hand state into the canonical policy joint layout."""
    array = np.asarray(values, dtype=np.float32)
    return array[..., _SWITCHED_HAND_REORDER] * _SWITCHED_HAND_SIGN


def canonical_hand_to_switched(values: np.ndarray) -> np.ndarray:
    """Map canonical hand commands into the swapped physical hand layout."""
    return switched_hand_to_canonical(values)


def canonical_gains_to_switched(values: np.ndarray) -> np.ndarray:
    """Reorder canonical per-joint gains to the swapped hand layout."""
    array = np.asarray(values, dtype=np.float32)
    return array[..., _SWITCHED_HAND_REORDER]
