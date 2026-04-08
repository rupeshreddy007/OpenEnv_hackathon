"""
Typed models for the Wildfire Containment Environment.

Provides Pydantic-validated schemas for observations, actions, and results
so that the OpenEnv spec can enforce type safety.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


# ---------------------------------------------------------------------------
# Pydantic config for numpy array support
# ---------------------------------------------------------------------------

class _NumpyConfig(BaseModel):
    """Base model that allows arbitrary types (numpy arrays)."""
    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class WildfireAction(BaseModel):
    """A single agent action."""
    action_type: int = Field(..., ge=0, le=3, description="0=noop, 1=firebreak, 2=waterdrop, 3=evacuate")
    row: int = Field(..., ge=0, description="Row coordinate on the grid")
    col: int = Field(..., ge=0, description="Column coordinate on the grid")

    ACTION_LABELS: dict = {0: "noop", 1: "firebreak", 2: "waterdrop", 3: "evacuate"}
    model_config = {"arbitrary_types_allowed": True}

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.action_type, self.row, self.col)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int]) -> "WildfireAction":
        return cls(action_type=t[0], row=t[1], col=t[2])

    @property
    def label(self) -> str:
        return self.ACTION_LABELS.get(self.action_type, "unknown")

    def validate_bounds(self, grid_size: int) -> bool:
        """Check action is within valid grid bounds."""
        return (
            0 <= self.action_type <= 3
            and 0 <= self.row < grid_size
            and 0 <= self.col < grid_size
        )


# ---------------------------------------------------------------------------
# Resources model
# ---------------------------------------------------------------------------

class Resources(BaseModel):
    """Remaining agent resources."""
    water_drops: int = Field(..., ge=0)
    firebreaks: int = Field(..., ge=0)
    evacuations: int = Field(..., ge=0)

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

class WildfireObservation(_NumpyConfig):
    """Full observable state returned by reset() / step() / state()."""
    fire_map: np.ndarray = Field(..., description="(N,N) int8 cell fire states")
    vegetation: np.ndarray = Field(..., description="(N,N) int8 vegetation types")
    elevation: np.ndarray = Field(..., description="(N,N) float64 elevation in metres")
    moisture: np.ndarray = Field(..., description="(N,N) float64 soil moisture 0-1")
    structures: np.ndarray = Field(..., description="(N,N) int8 structure overlay")
    evacuated: np.ndarray = Field(..., description="(N,N) bool evacuation status")
    wind_direction: float = Field(..., description="Wind direction in radians")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    resources: Resources
    timestep: int = Field(..., ge=0)
    burning_cells: int = Field(..., ge=0)
    burned_cells: int = Field(..., ge=0)
    total_burnable: int = Field(..., ge=0)

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

class StepResult(_NumpyConfig):
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

class TaskResult(BaseModel):
    """Result of running an agent on a single task."""
    task_id: str = Field(..., description="easy, medium, or hard")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized 0.0-1.0")
    raw_reward: float = Field(..., description="Unnormalized cumulative reward")
    burned_cells: int = Field(..., ge=0)
    total_burnable: int = Field(..., ge=0)
    structures_saved: int = Field(..., ge=0)
    structures_total: int = Field(..., ge=0)
    steps_taken: int = Field(..., ge=0)
    episodes_run: int = Field(default=1, ge=1)

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
