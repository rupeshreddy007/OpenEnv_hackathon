"""
Typed models for the Wildfire Containment Environment.

Provides Pydantic-validated schemas for observations, actions, and results
so that the OpenEnv spec can enforce type safety.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

@dataclass
class WildfireAction:
    """A single agent action."""
    action_type: int   # 0=noop, 1=firebreak, 2=waterdrop, 3=evacuate
    row: int
    col: int

    ACTION_LABELS = {0: "noop", 1: "firebreak", 2: "waterdrop", 3: "evacuate"}

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.action_type, self.row, self.col)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int]) -> "WildfireAction":
        return cls(action_type=t[0], row=t[1], col=t[2])

    @property
    def label(self) -> str:
        return self.ACTION_LABELS.get(self.action_type, "unknown")

    def validate(self, grid_size: int) -> bool:
        """Check action is within valid bounds."""
        return (
            0 <= self.action_type <= 3
            and 0 <= self.row < grid_size
            and 0 <= self.col < grid_size
        )


# ---------------------------------------------------------------------------
# Resources model
# ---------------------------------------------------------------------------

@dataclass
class Resources:
    """Remaining agent resources."""
    water_drops: int
    firebreaks: int
    evacuations: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "water_drops": self.water_drops,
            "firebreaks": self.firebreaks,
            "evacuations": self.evacuations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "Resources":
        return cls(
            water_drops=d["water_drops"],
            firebreaks=d["firebreaks"],
            evacuations=d["evacuations"],
        )


# ---------------------------------------------------------------------------
# Observation / State model
# ---------------------------------------------------------------------------

@dataclass
class WildfireObservation:
    """Full observable state returned by reset() / step() / state()."""
    fire_map: np.ndarray         # (N,N) int8
    vegetation: np.ndarray       # (N,N) int8
    elevation: np.ndarray        # (N,N) float64
    moisture: np.ndarray         # (N,N) float64
    structures: np.ndarray       # (N,N) int8
    evacuated: np.ndarray        # (N,N) bool
    wind_direction: float        # radians
    wind_speed: float            # m/s
    resources: Resources
    timestep: int
    burning_cells: int
    burned_cells: int
    total_burnable: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WildfireObservation":
        """Construct from the dict returned by env.state()."""
        return cls(
            fire_map=np.asarray(d["fire_map"]),
            vegetation=np.asarray(d["vegetation"]),
            elevation=np.asarray(d["elevation"]),
            moisture=np.asarray(d["moisture"]),
            structures=np.asarray(d["structures"]),
            evacuated=np.asarray(d["evacuated"]),
            wind_direction=float(d["wind_direction"]),
            wind_speed=float(d["wind_speed"]),
            resources=Resources.from_dict(d["resources"]),
            timestep=int(d["timestep"]),
            burning_cells=int(d["burning_cells"]),
            burned_cells=int(d["burned_cells"]),
            total_burnable=int(d["total_burnable"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fire_map": self.fire_map,
            "vegetation": self.vegetation,
            "elevation": self.elevation,
            "moisture": self.moisture,
            "structures": self.structures,
            "evacuated": self.evacuated,
            "wind_direction": self.wind_direction,
            "wind_speed": self.wind_speed,
            "resources": self.resources.to_dict(),
            "timestep": self.timestep,
            "burning_cells": self.burning_cells,
            "burned_cells": self.burned_cells,
            "total_burnable": self.total_burnable,
        }

    @property
    def grid_size(self) -> int:
        return self.fire_map.shape[0]


# ---------------------------------------------------------------------------
# Step result model
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Typed return value of env.step()."""
    observation: WildfireObservation
    reward: float
    done: bool
    info: Dict[str, Any]

    @classmethod
    def from_tuple(cls, t: tuple) -> "StepResult":
        state_dict, reward, done, info = t
        return cls(
            observation=WildfireObservation.from_dict(state_dict),
            reward=float(reward),
            done=bool(done),
            info=info,
        )


# ---------------------------------------------------------------------------
# Task result model (for grading)
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of running an agent on a single task."""
    task_id: str                 # "easy", "medium", "hard"
    score: float                 # Normalized 0.0–1.0
    raw_reward: float            # Unnormalized cumulative reward
    burned_cells: int
    total_burnable: int
    structures_saved: int
    structures_total: int
    steps_taken: int
    episodes_run: int = 1

    @property
    def terrain_saved_pct(self) -> float:
        if self.total_burnable == 0:
            return 1.0
        return (self.total_burnable - self.burned_cells) / self.total_burnable

    @property
    def structures_saved_pct(self) -> float:
        if self.structures_total == 0:
            return 1.0
        return self.structures_saved / self.structures_total
