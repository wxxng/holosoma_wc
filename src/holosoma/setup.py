import platform
import sys
from pathlib import Path

from setuptools import setup

UNITREE_VERSION = "0.1.2"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

pyvers = f"cp{sys.version_info.major}{sys.version_info.minor}"
platform_str = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")

# Use local custom-built wheel with hand support
LOCAL_WHEEL_DIR = Path("/home/rllab3/Desktop/codebase/unitreeG1/cpp_binding/unitree_sdk2_amazon/dist")
local_unitree_wheel = LOCAL_WHEEL_DIR / f"unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"

if local_unitree_wheel.exists():
    unitree_url = f"file://{local_unitree_wheel.absolute()}"
else:
    # Fallback to GitHub release if local wheel not found
    unitree_url = f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"  # noqa: E501

booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"  # noqa: E501

setup(
    extras_require={
        "unitree": [f"unitree_sdk2 @ {unitree_url}"],
        "booster": [f"booster_robotics_sdk @ {booster_url}"],
    },
)
