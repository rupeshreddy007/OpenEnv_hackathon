"""OpenEnv Wildfire Containment Environment Package."""

from .environment import WildfireEnv
from .config import (
    EnvironmentConfig,
    get_default_config,
    get_small_config,
    get_inferno_config,
    UNBURNED, BURNING, BURNED, FIREBREAK, WATER_DROPPED,
    GRASS, SHRUB, FOREST, ROCK, WATER_BODY,
    NO_STRUCTURE, HOUSE, HOSPITAL, FIRE_STATION,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)

__version__ = "1.0.0"
__all__ = [
    "WildfireEnv",
    "EnvironmentConfig",
    "get_default_config",
    "get_small_config",
    "get_inferno_config",
]
