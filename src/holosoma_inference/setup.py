import platform
import sys
from pathlib import Path

from setuptools import find_packages, setup

UNITREE_VERSION = "0.1.2"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

pyvers = f"cp{sys.version_info.major}{sys.version_info.minor}"
platform_tag = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")

# Use local custom-built wheel with hand support
LOCAL_WHEEL_DIR = Path("/home/rllab3/Desktop/codebase/unitreeG1/cpp_binding/unitree_sdk2_amazon/dist")
local_unitree_wheel = LOCAL_WHEEL_DIR / f"unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_tag}.whl"

unitree_extras = []
if local_unitree_wheel.exists():
    unitree_url = f"file://{local_unitree_wheel.absolute()}"
else:
    # Fallback to GitHub release if local wheel not found
    unitree_url = f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_tag}.whl"
unitree_extras.append(f"unitree_sdk2 @ {unitree_url}")

booster_extras = []
booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-cp310-cp310-{platform_tag}.whl"  # noqa: E501
booster_extras.append(f"booster_robotics_sdk @ {booster_url}")


setup(
    name="holosoma-inference",
    version="0.1.0",
    description="holosoma-inference: inference components for humanoid robot policies",
    long_description="",
    long_description_content_type="text/markdown",
    author="Amazon FAR Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydantic",
        "loguru",
        "netifaces",
        "onnx",
        "onnxruntime",
        "scipy",
        "sshkeyboard",
        "termcolor",
        "pyyaml",
        "tyro>=0.10.0a4",
        "wandb",
        "zmq",
        "defusedxml",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "unitree": unitree_extras,
        "booster": booster_extras,
    },
    entry_points={
        "holosoma.sdk": [
            "unitree = holosoma_inference.sdk.unitree.unitree_interface:UnitreeInterface",
            "booster = holosoma_inference.sdk.booster.booster_interface:BoosterInterface",
        ],
    },
    keywords="humanoid robotics inference policy onnx",
    include_package_data=True,
    package_data={
        "holosoma_inference": ["configs/**/*.yaml", "py.typed"],
    },
)
