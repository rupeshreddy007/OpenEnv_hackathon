"""
Configuration for the Wildfire Containment Environment.

Fire spread uses a simplified Rothermel model (Rothermel, 1972) calibrated
against Anderson's 13 standard fuel models (Anderson, 1982).

References:
  - Rothermel, R.C. (1972). "A mathematical model for predicting fire spread
    in wildland fuels." USDA Forest Service Research Paper INT-115.
  - Anderson, H.E. (1982). "Aids to determining fuel models for estimating
    fire behavior." USDA Forest Service General Technical Report INT-122.
  - Andrews, P.L. (2018). "The Rothermel surface fire spread model and
    associated developments: A comprehensive explanation." USDA Forest Service
    General Technical Report RMRS-GTR-371.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# Cell states
UNBURNED = 0
BURNING = 1
BURNED = 2
FIREBREAK = 3
WATER_DROPPED = 4

# -------------------------------------------------------------------------
# Anderson 13 Fuel Models (simplified to 5 categories for RL tractability)
#
# Each fuel type bundles several Anderson models:
#   GRASS      → Anderson FM1 (short grass, 1 ft)
#   SHRUB      → Anderson FM5/FM6 (brush, 2–2.5 ft)
#   FOREST     → Anderson FM8/FM9 (compact/loose timber litter)
#   ROCK       → Non-burnable (NB)
#   WATER_BODY → Non-burnable (NB)
#
# Parameters derived from Anderson (1982) Table 3, scaled
# to per-timestep values for the simulation grid.
# -------------------------------------------------------------------------

GRASS = 0
SHRUB = 1
FOREST = 2
ROCK = 3
WATER_BODY = 4

# Fuel properties from Anderson 13 models (per fuel type)
# fuel_load: tons/acre (total dead + live)
# depth: fuel bed depth in feet
# heat_content: BTU/lb
# moisture_extinction: fractional moisture of extinction
FUEL_PROPERTIES: Dict[int, dict] = {
    GRASS: {
        "anderson_model": "FM1",
        "fuel_load": 0.74,          # tons/acre
        "depth": 1.0,               # feet
        "heat_content": 8000,       # BTU/lb
        # M_x scaled from Anderson dead-fuel ratio to env soil-moisture (0-1)
        # Original: 0.12 dead-fuel ratio → mapped to ~0.40 soil moisture
        "moisture_extinction": 0.40,
        "rate_of_spread_base": 78,  # chains/hour (no wind, no slope)
    },
    SHRUB: {
        "anderson_model": "FM5",
        "fuel_load": 3.50,
        "depth": 2.0,
        "heat_content": 8000,
        "moisture_extinction": 0.55,
        "rate_of_spread_base": 18,
    },
    FOREST: {
        "anderson_model": "FM9",
        "fuel_load": 3.50,
        "depth": 0.2,
        "heat_content": 8000,
        "moisture_extinction": 0.65,
        "rate_of_spread_base": 7.5,
    },
    ROCK: {
        "anderson_model": "NB",
        "fuel_load": 0.0,
        "depth": 0.0,
        "heat_content": 0,
        "moisture_extinction": 0.0,
        "rate_of_spread_base": 0.0,
    },
    WATER_BODY: {
        "anderson_model": "NB",
        "fuel_load": 0.0,
        "depth": 0.0,
        "heat_content": 0,
        "moisture_extinction": 0.0,
        "rate_of_spread_base": 0.0,
    },
}

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
    cell_size_m: float = 30.0  # Metres per cell (≈ Landsat pixel resolution)

    # --- Terrain generation ---
    vegetation_probs: List[float] = field(
        default_factory=lambda: [0.25, 0.30, 0.35, 0.07, 0.03]
    )  # probability of [GRASS, SHRUB, FOREST, ROCK, WATER_BODY]

    elevation_scale: float = 100.0   # Max elevation in metres
    moisture_range: Tuple[float, float] = (0.05, 0.50)  # Soil moisture 0-1

    # --- Real terrain (optional) ---
    # If set, loads elevation/vegetation from numpy files instead of generating
    real_terrain_path: Optional[str] = None  # Path to directory with terrain data

    # --- Structures ---
    n_houses: int = 8
    n_hospitals: int = 1
    n_fire_stations: int = 1

    # --- Fire ignition ---
    n_initial_fires: int = 2  # How many cells ignite at start

    # --- Rothermel fire spread parameters ---
    # Simplified Rothermel model:
    #   R = R0 × (1 + φ_w + φ_s) × (1 - moisture/M_x) × flammability
    # where R0 = base rate, φ_w = wind factor, φ_s = slope factor
    base_spread_prob: float = 0.55       # Base ignition probability (pre-damping)
    wind_spread_bonus: float = 0.25      # Extra spread prob in downwind direction
    uphill_spread_bonus: float = 0.15    # Extra spread prob when fire goes uphill
    moisture_spread_penalty: float = 0.20  # Reduction in spread per unit moisture

    # Rothermel wind factor coefficient
    # φ_w = C × (3.281 × U)^B × (β/β_opt)^(-E)
    # Simplified to: φ_w = rothermel_wind_C × (U / U_ref)^rothermel_wind_B
    rothermel_wind_C: float = 1.0    # Wind factor multiplier
    rothermel_wind_B: float = 1.4    # Wind exponent (Rothermel uses ~1.0–1.6)
    rothermel_wind_ref: float = 10.0 # Reference wind speed (m/s)

    # Rothermel slope factor: φ_s = 5.275 × β^(-0.3) × tan²(slope)
    # Simplified to: φ_s = slope_factor × tan²(θ)
    rothermel_slope_factor: float = 5.275

    vegetation_burn_rate: dict = field(default_factory=lambda: {
        GRASS: 1,    # Burns out in 1 step  (fast flash fuel)
        SHRUB: 2,    # Burns for 2 steps
        FOREST: 4,   # Burns for 4 steps    (sustained burning)
        ROCK: 0,     # Never burns
        WATER_BODY: 0,
    })

    vegetation_flammability: dict = field(default_factory=lambda: {
        GRASS: 1.3,    # High flammability, low fuel load
        SHRUB: 1.0,    # Medium flammability, medium fuel load
        FOREST: 0.8,   # Lower ignition prob, high fuel load
        ROCK: 0.0,     # Non-flammable
        WATER_BODY: 0.0,
    })

    # Whether to use the Rothermel-inspired spread model vs. simplified
    use_rothermel: bool = True

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

    # --- Ember spotting (long-range ignition from wind-carried embers) ---
    ember_spotting: bool = True          # Enable ember spotting mechanic
    ember_max_distance: int = 5          # Max cells an ember can travel
    ember_min_distance: int = 2          # Min cells (embers skip nearby cells)
    ember_prob_base: float = 0.03        # Base probability per burning cell per step
    ember_wind_scale: float = 0.006      # Extra prob per m/s of wind speed

    # --- Fire station resupply ---
    fire_station_resupply: bool = True    # Enable periodic resupply
    resupply_interval: int = 10           # Steps between resupply events
    resupply_water: int = 1               # Water drops regained per station per interval
    resupply_firebreaks: int = 1          # Firebreaks regained per station per interval
    resupply_radius: int = 0              # 0 = global resupply, >0 = only near station

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
        base_spread_prob=0.65,
        wind_spread_bonus=0.30,
        n_houses=15,
        n_hospitals=2,
        max_water_drops=15,
        max_firebreaks=20,
        max_steps=120,
    )
