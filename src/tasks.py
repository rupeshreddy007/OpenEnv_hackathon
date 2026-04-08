"""
Task definitions and graders for the Wildfire Containment Environment.

Three difficulty tiers, each with a deterministic config and a grader that
returns a normalized score in [0.0, 1.0].

Usage:
    from src.tasks import TASKS, run_task

    result = run_task("easy", agent_fn)
    print(result.score)  # 0.0–1.0
"""

import numpy as np
from typing import Callable, Dict, Any, Tuple

from config import (
    EnvironmentConfig,
    UNBURNED, BURNING, BURNED, FIREBREAK,
    NO_STRUCTURE, HOUSE, HOSPITAL,
)
from environment import WildfireEnv
from models import TaskResult


# -------------------------------------------------------------------------
# Task configs (deterministic seeds for reproducibility)
# -------------------------------------------------------------------------

def get_easy_task_config() -> EnvironmentConfig:
    """Brush Fire — small grid, moderate wind, limited resources."""
    return EnvironmentConfig(
        grid_size=12,
        n_initial_fires=3,
        initial_wind_speed=8.0,
        wind_change_prob=0.15,
        base_spread_prob=0.65,
        wind_spread_bonus=0.28,
        moisture_range=(0.05, 0.30),
        n_houses=6,
        n_hospitals=1,
        n_fire_stations=1,
        max_water_drops=2,
        max_firebreaks=4,
        max_evacuations=2,
        max_steps=50,
        seed=1001,
    )


def get_medium_task_config() -> EnvironmentConfig:
    """Suburban Wildfire — larger grid, strong wind, scarce resources."""
    return EnvironmentConfig(
        grid_size=20,
        n_initial_fires=3,
        initial_wind_speed=8.5,
        wind_change_prob=0.14,
        base_spread_prob=0.58,
        wind_spread_bonus=0.27,
        moisture_range=(0.06, 0.35),
        n_houses=10,
        n_hospitals=2,
        n_fire_stations=1,
        max_water_drops=5,
        max_firebreaks=9,
        max_evacuations=3,
        max_steps=80,
        seed=2002,
    )


def get_hard_task_config() -> EnvironmentConfig:
    """Inferno — large grid, fierce wind, many fires, minimal resources."""
    return EnvironmentConfig(
        grid_size=30,
        n_initial_fires=5,
        initial_wind_speed=10.0,
        wind_change_prob=0.18,
        base_spread_prob=0.58,
        wind_spread_bonus=0.28,
        moisture_range=(0.05, 0.35),
        n_houses=15,
        n_hospitals=2,
        n_fire_stations=2,
        max_water_drops=7,
        max_firebreaks=12,
        max_evacuations=4,
        max_steps=120,
        seed=3003,
    )


# -------------------------------------------------------------------------
# Grader — computes normalized score 0.0–1.0
# -------------------------------------------------------------------------

def _grade(env: WildfireEnv, cumulative_reward: float, task_id: str) -> TaskResult:
    """
    Compute a normalized score from the final environment state.

    Scoring components (weighted):
      - Terrain saved %    (40% weight)
      - Structures saved % (40% weight)
      - Containment bonus  (20% weight) — fire fully out = 1.0

    Returns TaskResult with score in [0.0, 1.0].
    """
    fire_map = env.fire_map
    structures = env.structures
    total_burnable = env._total_burnable

    # Terrain saved
    burned = int(np.sum(fire_map == BURNED))
    terrain_saved_pct = (total_burnable - burned) / max(total_burnable, 1)

    # Structures saved
    struct_cells = (structures == HOUSE) | (structures == HOSPITAL)
    total_structures = int(np.sum(struct_cells))
    burned_structures = int(np.sum(struct_cells & (fire_map == BURNED)))
    saved_structures = total_structures - burned_structures
    struct_saved_pct = saved_structures / max(total_structures, 1)

    # Containment bonus — fire is fully out
    still_burning = int(np.sum(fire_map == BURNING))
    containment = 1.0 if still_burning == 0 and burned < total_burnable * 0.5 else 0.0

    # Weighted score
    score = (
        0.40 * terrain_saved_pct
        + 0.40 * struct_saved_pct
        + 0.20 * containment
    )
    score = float(np.clip(score, 0.0, 1.0))

    return TaskResult(
        task_id=task_id,
        score=round(score, 4),
        raw_reward=round(cumulative_reward, 2),
        burned_cells=burned,
        total_burnable=total_burnable,
        structures_saved=saved_structures,
        structures_total=total_structures,
        steps_taken=env.timestep,
    )


def grade_easy(env: WildfireEnv, cumulative_reward: float) -> TaskResult:
    return _grade(env, cumulative_reward, "easy")


def grade_medium(env: WildfireEnv, cumulative_reward: float) -> TaskResult:
    return _grade(env, cumulative_reward, "medium")


def grade_hard(env: WildfireEnv, cumulative_reward: float) -> TaskResult:
    return _grade(env, cumulative_reward, "hard")


# -------------------------------------------------------------------------
# Task runner
# -------------------------------------------------------------------------

# Agent function type: takes state dict, returns (action_type, row, col)
AgentFn = Callable[[Dict[str, Any]], Tuple[int, int, int]]

TASKS = {
    "easy": {"config_fn": get_easy_task_config, "grader": grade_easy},
    "medium": {"config_fn": get_medium_task_config, "grader": grade_medium},
    "hard": {"config_fn": get_hard_task_config, "grader": grade_hard},
}


def run_task(
    task_id: str,
    agent_fn: AgentFn,
    n_episodes: int = 5,
    verbose: bool = False,
) -> TaskResult:
    """
    Run an agent on a task for n_episodes and return averaged score.

    Args:
        task_id: "easy", "medium", or "hard"
        agent_fn: callable(state_dict) -> (action_type, row, col)
        n_episodes: number of episodes to average over
        verbose: print per-episode results

    Returns:
        TaskResult with averaged score
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Choose from {list(TASKS.keys())}")

    task = TASKS[task_id]
    config_fn = task["config_fn"]
    grader = task["grader"]

    scores = []
    total_burned = 0
    total_burnable = 0
    total_struct_saved = 0
    total_struct_total = 0
    total_raw_reward = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        cfg = config_fn()
        # Vary seed per episode for robustness
        cfg.seed = cfg.seed + ep
        env = WildfireEnv(cfg)
        state = env.reset()

        cumulative_reward = 0.0
        done = False

        while not done:
            action = agent_fn(state)
            state, reward, done, info = env.step(action)
            cumulative_reward += reward

        result = grader(env, cumulative_reward)
        scores.append(result.score)
        total_burned += result.burned_cells
        total_burnable += result.total_burnable
        total_struct_saved += result.structures_saved
        total_struct_total += result.structures_total
        total_raw_reward += result.raw_reward
        total_steps += result.steps_taken

        if verbose:
            print(f"  Episode {ep+1}/{n_episodes}: score={result.score:.4f} "
                  f"burned={result.burned_cells}/{result.total_burnable} "
                  f"structs={result.structures_saved}/{result.structures_total}")

    avg_score = float(np.mean(scores))
    return TaskResult(
        task_id=task_id,
        score=round(avg_score, 4),
        raw_reward=round(total_raw_reward / n_episodes, 2),
        burned_cells=total_burned // n_episodes,
        total_burnable=total_burnable // n_episodes,
        structures_saved=total_struct_saved // n_episodes,
        structures_total=total_struct_total // n_episodes,
        steps_taken=total_steps // n_episodes,
        episodes_run=n_episodes,
    )
