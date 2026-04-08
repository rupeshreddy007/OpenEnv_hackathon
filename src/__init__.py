"""OpenEnv Wildfire Containment Environment Package."""

from .environment import WildfireEnv
from .config import (
    EnvironmentConfig,
    get_default_config,
    get_small_config,
    get_inferno_config,
    FUEL_PROPERTIES,
    UNBURNED, BURNING, BURNED, FIREBREAK, WATER_DROPPED,
    GRASS, SHRUB, FOREST, ROCK, WATER_BODY,
    NO_STRUCTURE, HOUSE, HOSPITAL, FIRE_STATION,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)
from .models import (
    WildfireAction,
    WildfireObservation,
    Resources,
    StepResult,
    TaskResult,
)
from .tasks import (
    TASKS,
    run_task,
    grade_easy,
    grade_medium,
    grade_hard,
    get_easy_task_config,
    get_medium_task_config,
    get_hard_task_config,
)

__version__ = "1.0.0"
__all__ = [
    "WildfireEnv",
    "EnvironmentConfig",
    "get_default_config",
    "get_small_config",
    "get_inferno_config",
    "WildfireAction",
    "WildfireObservation",
    "Resources",
    "StepResult",
    "TaskResult",
    "TASKS",
    "run_task",
]
