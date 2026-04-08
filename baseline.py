"""
Baseline inference script — runs agents against all three tasks and reports scores.

Usage:
    python baseline.py                    # Run all tasks
    python baseline.py --task easy        # Run one task
    python baseline.py --episodes 10      # More episodes for stable averages
    python baseline.py --agent greedy     # Choose agent: random or greedy (default)

Output: Reproducible scores for each task (0.0–1.0).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from environment import WildfireEnv
from config import (
    UNBURNED, BURNING, BURNED,
    NO_STRUCTURE, HOUSE, HOSPITAL,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)
from tasks import TASKS, run_task


# -------------------------------------------------------------------------
# Agent implementations
# -------------------------------------------------------------------------

def random_agent(state):
    """Uniformly random actions."""
    N = len(state["fire_map"])
    rng = np.random.default_rng()
    action_type = rng.choice([ACTION_NOOP, ACTION_FIREBREAK,
                              ACTION_WATERDROP, ACTION_EVACUATE])
    row = rng.integers(0, N)
    col = rng.integers(0, N)
    return (int(action_type), int(row), int(col))


def greedy_agent(state):
    """
    Heuristic agent:
      1. Evacuate structures closest to fire
      2. Water-drop on densest burning area near structures
      3. Firebreak ahead of the fire front (downwind)
      4. Noop if nothing useful
    """
    fire_map = np.asarray(state["fire_map"])
    structures = np.asarray(state["structures"])
    evacuated = np.asarray(state["evacuated"])
    resources = state["resources"]
    wind_dir = state["wind_direction"]
    N = fire_map.shape[0]

    burning_cells = list(zip(*np.where(fire_map == BURNING)))
    if not burning_cells:
        return (ACTION_NOOP, 0, 0)

    # 1. Evacuate threatened structures
    if resources["evacuations"] > 0:
        best_struct = None
        best_dist = float("inf")
        for r, c in zip(*np.where((structures != NO_STRUCTURE) & (~evacuated))):
            for br, bc in burning_cells:
                d = abs(int(r) - int(br)) + abs(int(c) - int(bc))
                threat = d
                if structures[r, c] == HOSPITAL:
                    threat -= 3
                if threat < best_dist:
                    best_dist = threat
                    best_struct = (int(r), int(c))
        if best_struct and best_dist <= 5:
            return (ACTION_EVACUATE, best_struct[0], best_struct[1])

    # 2. Water-drop burning cells near structures
    if resources["water_drops"] > 0:
        best_target = None
        best_score = -1
        for br, bc in burning_cells:
            score = 0
            for sr, sc in zip(*np.where(structures != NO_STRUCTURE)):
                d = abs(int(br) - int(sr)) + abs(int(bc) - int(sc))
                if d <= 4:
                    val = 3 if structures[sr, sc] == HOSPITAL else 1
                    score += val / max(d, 1)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(br) + dr, int(bc) + dc
                if 0 <= nr < N and 0 <= nc < N and fire_map[nr, nc] == UNBURNED:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_target = (int(br), int(bc))
        if best_target and best_score > 0.3:
            return (ACTION_WATERDROP, best_target[0], best_target[1])

    # 3. Firebreak downwind of fire front
    if resources["firebreaks"] > 0:
        wind_dr = np.sin(wind_dir)
        wind_dc = np.cos(wind_dir)
        best_fb = None
        best_score = -1
        checked = set()
        for br, bc in burning_cells:
            for dist in [2, 3, 1]:
                nr = int(round(int(br) + wind_dr * dist))
                nc = int(round(int(bc) + wind_dc * dist))
                if (nr, nc) in checked:
                    continue
                checked.add((nr, nc))
                if 0 <= nr < N and 0 <= nc < N and fire_map[nr, nc] == UNBURNED:
                    score = 1.0
                    for sr, sc in zip(*np.where(structures != NO_STRUCTURE)):
                        d = abs(nr - int(sr)) + abs(nc - int(sc))
                        if d <= 5:
                            score += 2.0 / max(d, 1)
                    if score > best_score:
                        best_score = score
                        best_fb = (nr, nc)
        if best_fb:
            return (ACTION_FIREBREAK, best_fb[0], best_fb[1])

    return (ACTION_NOOP, 0, 0)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

AGENTS = {
    "random": random_agent,
    "greedy": greedy_agent,
}


def main():
    parser = argparse.ArgumentParser(description="Wildfire Containment — Baseline Evaluation")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"],
                        default="all", help="Which task to run")
    parser.add_argument("--agent", choices=list(AGENTS.keys()),
                        default="greedy", help="Agent to evaluate")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per task for averaging")
    args = parser.parse_args()

    agent_fn = AGENTS[args.agent]
    task_ids = list(TASKS.keys()) if args.task == "all" else [args.task]

    print(f"{'='*60}")
    print(f"  Wildfire Containment — Baseline Evaluation")
    print(f"  Agent: {args.agent}  |  Episodes per task: {args.episodes}")
    print(f"{'='*60}\n")

    results = []
    for task_id in task_ids:
        print(f"Task: {task_id.upper()}")
        result = run_task(task_id, agent_fn, n_episodes=args.episodes, verbose=True)
        results.append(result)
        print(f"  => Score: {result.score:.4f}  "
              f"| Burned: {result.burned_cells}/{result.total_burnable}  "
              f"| Structures: {result.structures_saved}/{result.structures_total}  "
              f"| Steps: {result.steps_taken}")
        print()

    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        filled = int(r.score * 30)
        bar = "#" * filled + "-" * (30 - filled)
        print(f"  {r.task_id:8s}  [{bar}]  {r.score:.4f}")

    avg = np.mean([r.score for r in results])
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
