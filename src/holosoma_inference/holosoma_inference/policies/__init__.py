__all__ = ["BasePolicy", "LocomotionPolicy", "WholeBodyTrackingPolicy"]


def __getattr__(name):
    if name == "BasePolicy":
        from .base import BasePolicy

        return BasePolicy
    if name == "LocomotionPolicy":
        from .locomotion import LocomotionPolicy

        return LocomotionPolicy
    if name == "WholeBodyTrackingPolicy":
        from .wbt import WholeBodyTrackingPolicy

        return WholeBodyTrackingPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
