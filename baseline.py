"""
Baseline inference script — runs agents against all three tasks and reports scores.

Usage:
    python baseline.py                        # Run all tasks with greedy heuristic
    python baseline.py --agent openai         # Run with OpenAI GPT model
    python baseline.py --task easy            # Run one task
    python baseline.py --episodes 10          # More episodes for stable averages
    python baseline.py --agent greedy         # Choose agent: random, greedy, or openai

Requires OPENAI_API_KEY environment variable for --agent openai.

Output: Reproducible scores for each task (0.0–1.0).
"""

import sys
import os
import json
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
# OpenAI API agent
# -------------------------------------------------------------------------

def _build_openai_prompt(state):
    """Convert environment state to a text prompt for the LLM."""
    fire_map = np.asarray(state["fire_map"])
    structures = np.asarray(state["structures"])
    evacuated = np.asarray(state["evacuated"])
    resources = state["resources"]
    N = fire_map.shape[0]

    burning = list(zip(*np.where(fire_map == BURNING)))
    burned = int(np.sum(fire_map == BURNED))
    struct_cells = list(zip(*np.where(structures != NO_STRUCTURE)))
    at_risk = []
    for r, c in struct_cells:
        if fire_map[r, c] != BURNED and not evacuated[r, c]:
            kind = "HOSPITAL" if structures[r, c] == HOSPITAL else "HOUSE"
            min_dist = min(
                (abs(int(r) - int(br)) + abs(int(c) - int(bc)) for br, bc in burning),
                default=99,
            )
            at_risk.append(f"  {kind} at ({r},{c}), nearest fire distance={min_dist}")

    wind_deg = int(np.degrees(state["wind_direction"])) % 360

    prompt = f"""You are an AI incident commander managing a {N}x{N} wildfire grid.

Current state (step {state['timestep']}):
- Burning cells: {len(burning)}, Burned: {burned}/{state['total_burnable']}
- Wind: {wind_deg} degrees, {state['wind_speed']:.1f} m/s
- Resources: {resources['water_drops']} water drops, {resources['firebreaks']} firebreaks, {resources['evacuations']} evacuations

Structures at risk:
{chr(10).join(at_risk) if at_risk else '  None'}

Actions: 0=noop, 1=firebreak(row,col), 2=waterdrop(row,col), 3=evacuate(row,col)
Grid coordinates: row and col in [0, {N-1}]

Reply with ONLY a JSON object: {{"action_type": int, "row": int, "col": int}}
Choose the best action to minimise fire damage and protect structures."""
    return prompt


def make_openai_agent():
    """Create an agent that calls the OpenAI API for each decision."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def openai_agent(state):
        prompt = _build_openai_prompt(state)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a wildfire containment AI. Respond with only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON from response (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            N = len(state["fire_map"])
            action_type = int(np.clip(data.get("action_type", 0), 0, 3))
            row = int(np.clip(data.get("row", 0), 0, N - 1))
            col = int(np.clip(data.get("col", 0), 0, N - 1))
            return (action_type, row, col)
        except Exception as e:
            print(f"    [OpenAI fallback] {e}")
            return greedy_agent(state)

    return openai_agent


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
    parser.add_argument("--agent", choices=["random", "greedy", "openai"],
                        default="greedy", help="Agent to evaluate")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per task for averaging")
    args = parser.parse_args()

    if args.agent == "openai":
        agent_fn = make_openai_agent()
    else:
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
