"""
Example: Random agent playing the Wildfire Containment Environment.

Demonstrates the step() / reset() / state() API and prints an ASCII
visualisation each timestep.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from environment import WildfireEnv
from config import (
    get_small_config,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)


def random_agent(env: WildfireEnv):
    """Agent that picks uniformly random valid actions."""
    rng = np.random.default_rng(42)
    state = env.reset()
    print(env.render())

    total_reward = 0.0
    done = False

    while not done:
        action_type = rng.choice([ACTION_NOOP, ACTION_FIREBREAK,
                                  ACTION_WATERDROP, ACTION_EVACUATE])
        row = rng.integers(0, env.N)
        col = rng.integers(0, env.N)

        state, reward, done, info = env.step((action_type, row, col))
        total_reward += reward

        if env.timestep % 5 == 0 or done:
            print(env.render())
            print(f"  reward={reward:.2f}  total={total_reward:.2f}  "
                  f"info={info['action_effect']}")

    print(f"\n=== Episode finished ===")
    print(f"Total reward : {total_reward:.2f}")
    print(f"Burned cells : {state['burned_cells']}/{state['total_burnable']}")
    print(f"Steps taken  : {state['timestep']}")


def greedy_agent(env: WildfireEnv):
    """
    Simple heuristic agent:
      1. Evacuate any structure near fire first.
      2. Water-drop on the largest fire cluster.
      3. Place firebreaks ahead of the fire front in the wind direction.
    """
    state = env.reset()
    print("=== Greedy Heuristic Agent ===")
    print(env.render())

    total_reward = 0.0
    done = False

    while not done:
        action = _greedy_pick(state, env)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if env.timestep % 10 == 0 or done:
            print(env.render())
            print(f"  action={action}  reward={reward:.2f}  total={total_reward:.2f}")

    print(f"\n=== Greedy agent finished ===")
    print(f"Total reward : {total_reward:.2f}")
    print(f"Burned cells : {state['burned_cells']}/{state['total_burnable']}")


def _greedy_pick(state, env):
    """Pick a greedy action based on current state."""
    from config import BURNING, UNBURNED, NO_STRUCTURE, HOUSE, HOSPITAL

    fire_map = state["fire_map"]
    structures = state["structures"]
    evacuated = state["evacuated"]
    resources = state["resources"]
    N = fire_map.shape[0]

    burning_cells = np.argwhere(fire_map == BURNING)
    if len(burning_cells) == 0:
        return (ACTION_NOOP, 0, 0)

    # 1. Evacuate threatened structures
    if resources["evacuations"] > 0:
        for r, c in np.argwhere((structures != NO_STRUCTURE) & (~evacuated)):
            # Check if fire is within 2 cells
            r_lo, r_hi = max(0, r-2), min(N, r+3)
            c_lo, c_hi = max(0, c-2), min(N, c+3)
            if np.any(fire_map[r_lo:r_hi, c_lo:c_hi] == BURNING):
                return (ACTION_EVACUATE, int(r), int(c))

    # 2. Water-drop on densest fire area
    if resources["water_drops"] > 0:
        best, best_count = burning_cells[0], 0
        for r, c in burning_cells:
            r_lo, r_hi = max(0, r-1), min(N, r+2)
            c_lo, c_hi = max(0, c-1), min(N, c+2)
            count = int(np.sum(fire_map[r_lo:r_hi, c_lo:c_hi] == BURNING))
            if count > best_count:
                best_count = count
                best = (r, c)
        return (ACTION_WATERDROP, int(best[0]), int(best[1]))

    # 3. Firebreak downwind of fire front
    if resources["firebreaks"] > 0:
        wind_dr = int(np.round(np.sin(state["wind_direction"])))
        wind_dc = int(np.round(np.cos(state["wind_direction"])))
        for r, c in burning_cells:
            nr, nc = r + wind_dr * 2, c + wind_dc * 2
            if 0 <= nr < N and 0 <= nc < N and fire_map[nr, nc] == UNBURNED:
                return (ACTION_FIREBREAK, nr, nc)

    return (ACTION_NOOP, 0, 0)


if __name__ == "__main__":
    config = get_small_config()
    config.seed = 123
    env = WildfireEnv(config)

    print("=" * 60)
    print("  RANDOM AGENT")
    print("=" * 60)
    random_agent(env)

    print("\n\n")
    print("=" * 60)
    print("  GREEDY HEURISTIC AGENT")
    print("=" * 60)
    env2 = WildfireEnv(config)
    greedy_agent(env2)
