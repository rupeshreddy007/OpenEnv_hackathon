"""
Configuration for the Wildfire Containment Environment.

Defines terrain, fire dynamics, wind, resources, and scoring parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Cell states
UNBURNED = 0
BURNING = 1
BURNED = 2
FIREBREAK = 3
WATER_DROPPED = 4

# Vegetation types
GRASS = 0       # Burns fast, low intensity
SHRUB = 1       # Medium burn rate
FOREST = 2      # Burns slow, high intensity
ROCK = 3        # Non-flammable
WATER_BODY = 4  # Non-flammable

# Structure types (overlay)
NO_STRUCTURE = 0
HOUSE = 1
HOSPITAL = 2
FIRE_STATION = 3

# Action types
ACTION_NOOP = 0
ACTION_FIREBREAK = 1   # Dig a firebreak at target cell
ACTION_WATERDROP = 2    # Drop water on target cell
ACTION_EVACUATE = 3     # Evacuate a structure zone


@dataclass
class EnvironmentConfig:
    """All parameters for the wildfire containment simulation."""

    # --- Grid ---
    grid_size: int = 20  # NxN grid

    # --- Terrain generation ---
    vegetation_probs: List[float] = field(
        default_factory=lambda: [0.25, 0.30, 0.35, 0.07, 0.03]
    )  # probability of [GRASS, SHRUB, FOREST, ROCK, WATER_BODY]

    elevation_scale: float = 100.0   # Max elevation in metres
    moisture_range: Tuple[float, float] = (0.1, 0.9)  # Soil moisture 0-1

    # --- Structures ---
    n_houses: int = 8
    n_hospitals: int = 1
    n_fire_stations: int = 1

    # --- Fire ignition ---
    n_initial_fires: int = 2  # How many cells ignite at start

    # --- Fire spread dynamics ---
    base_spread_prob: float = 0.30       # Base probability fire spreads to neighbor
    wind_spread_bonus: float = 0.25      # Extra spread prob in downwind direction
    uphill_spread_bonus: float = 0.15    # Extra spread prob when fire goes uphill
    moisture_spread_penalty: float = 0.20  # Reduction in spread per unit moisture

    vegetation_burn_rate: dict = field(default_factory=lambda: {
        GRASS: 1,    # Burns out in 1 step
        SHRUB: 2,    # Burns for 2 steps
        FOREST: 4,   # Burns for 4 steps
        ROCK: 0,     # Never burns
        WATER_BODY: 0,
    })

    vegetation_flammability: dict = field(default_factory=lambda: {
        GRASS: 1.3,
        SHRUB: 1.0,
        FOREST: 0.8,
        ROCK: 0.0,
        WATER_BODY: 0.0,
    })

    # --- Wind ---
    initial_wind_direction: float = 0.0    # Radians (0 = east, pi/2 = north)
    initial_wind_speed: float = 5.0        # m/s
    wind_change_prob: float = 0.10         # Probability wind shifts each step
    max_wind_shift: float = 0.5            # Max radians wind can shift per step
    wind_speed_range: Tuple[float, float] = (2.0, 15.0)

    # --- Agent resources ---
    max_water_drops: int = 10         # Total water drops available
    max_firebreaks: int = 15          # Total firebreak cells agent can dig
    max_evacuations: int = 3          # Total evacuation orders
    water_drop_radius: int = 1        # Water affects a radius around target
    water_suppress_duration: int = 3  # Steps a watered cell resists fire

    # --- Episode ---
    max_steps: int = 80  # Episode length

    # --- Scoring ---
    reward_per_saved_cell: float = 0.1
    reward_per_saved_house: float = 5.0
    reward_per_saved_hospital: float = 15.0
    penalty_per_burned_cell: float = -0.05
    penalty_per_burned_house: float = -8.0
    penalty_per_burned_hospital: float = -20.0
    reward_successful_evacuate: float = 10.0
    penalty_wasted_resource: float = -1.0
    containment_bonus: float = 50.0  # Bonus if fire fully contained

    seed: Optional[int] = None


def get_default_config() -> EnvironmentConfig:
    return EnvironmentConfig()


def get_small_config() -> EnvironmentConfig:
    """Tiny 10x10 grid for fast debugging."""
    return EnvironmentConfig(
        grid_size=10,
        n_houses=3,
        n_hospitals=0,
        n_fire_stations=1,
        n_initial_fires=1,
        max_water_drops=5,
        max_firebreaks=8,
        max_steps=40,
    )


def get_inferno_config() -> EnvironmentConfig:
    """Hard mode: large grid, strong wind, many ignitions."""
    return EnvironmentConfig(
        grid_size=30,
        n_initial_fires=5,
        initial_wind_speed=12.0,
        base_spread_prob=0.40,
        wind_spread_bonus=0.30,
        n_houses=15,
        n_hospitals=2,
        max_water_drops=15,
        max_firebreaks=20,
        max_steps=120,
    )
