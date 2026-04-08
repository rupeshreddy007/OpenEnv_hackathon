"""
Inference Script — Wildfire Containment OpenEnv
================================================

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=wildfire-containment model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    python inference.py                       # Run all tasks
    python inference.py --task easy            # Run one task
    python inference.py --episodes 3          # Episodes per task
"""

import sys
import os
import json
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from openai import OpenAI

from environment import WildfireEnv
from config import (
    UNBURNED, BURNING, BURNED,
    NO_STRUCTURE, HOUSE, HOSPITAL,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)
from tasks import TASKS, run_task

# ---------------------------------------------------------------------------
# Environment variables (mandatory)
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "wildfire-containment"
TEMPERATURE = 0.0
MAX_TOKENS = 80

# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI incident commander managing a wildfire containment grid.
Each turn you choose ONE action to minimise fire damage and protect structures.
Actions: 0=noop, 1=firebreak(row,col), 2=waterdrop(row,col), 3=evacuate(row,col).
Reply with ONLY a JSON object: {"action_type": int, "row": int, "col": int}
No explanation, no markdown, just the JSON."""


def build_user_prompt(state, history: List[str]) -> str:
    fire_map = np.asarray(state["fire_map"])
    structures = np.asarray(state["structures"])
    evacuated = np.asarray(state["evacuated"])
    resources = state["resources"]
    N = fire_map.shape[0]

    burning = list(zip(*np.where(fire_map == BURNING)))
    burned = int(np.sum(fire_map == BURNED))

    at_risk = []
    for r, c in zip(*np.where(structures != NO_STRUCTURE)):
        if fire_map[r, c] != BURNED and not evacuated[r, c]:
            kind = "HOSPITAL" if structures[r, c] == HOSPITAL else "HOUSE"
            min_dist = min(
                (abs(int(r) - int(br)) + abs(int(c) - int(bc)) for br, bc in burning),
                default=99,
            )
            at_risk.append(f"  {kind} at ({r},{c}), fire dist={min_dist}")

    wind_deg = int(np.degrees(state["wind_direction"])) % 360
    history_block = "\n".join(history[-5:]) if history else "None"

    return f"""Step {state['timestep']} of {N}x{N} grid:
- Burning: {len(burning)} cells, Burned: {burned}/{state['total_burnable']}
- Wind: {wind_deg} deg, {state['wind_speed']:.1f} m/s
- Resources: water={resources['water_drops']}, firebreaks={resources['firebreaks']}, evacuations={resources['evacuations']}

Structures at risk:
{chr(10).join(at_risk[:10]) if at_risk else '  None'}

Recent actions:
{history_block}

Respond with JSON: {{"action_type": int, "row": int, "col": int}}
Coordinates in [0, {N-1}]."""


def get_llm_action(client: OpenAI, state, history: List[str]) -> tuple:
    """Call the LLM and parse the action."""
    user_prompt = build_user_prompt(state, history)
    N = len(state["fire_map"])

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        action_type = int(np.clip(data.get("action_type", 0), 0, 3))
        row = int(np.clip(data.get("row", 0), 0, N - 1))
        col = int(np.clip(data.get("col", 0), 0, N - 1))
        return (action_type, row, col)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return greedy_fallback(state)


# ---------------------------------------------------------------------------
# Greedy fallback agent (used when LLM fails)
# ---------------------------------------------------------------------------

def greedy_fallback(state):
    """Simple heuristic fallback."""
    fire_map = np.asarray(state["fire_map"])
    structures = np.asarray(state["structures"])
    evacuated = np.asarray(state["evacuated"])
    resources = state["resources"]
    wind_dir = state["wind_direction"]
    N = fire_map.shape[0]

    burning_cells = list(zip(*np.where(fire_map == BURNING)))
    if not burning_cells:
        return (ACTION_NOOP, 0, 0)

    # Evacuate threatened structures
    if resources["evacuations"] > 0:
        best_struct = None
        best_dist = float("inf")
        for r, c in zip(*np.where((structures != NO_STRUCTURE) & (~evacuated))):
            for br, bc in burning_cells:
                d = abs(int(r) - int(br)) + abs(int(c) - int(bc))
                if structures[r, c] == HOSPITAL:
                    d -= 3
                if d < best_dist:
                    best_dist = d
                    best_struct = (int(r), int(c))
        if best_struct and best_dist <= 5:
            return (ACTION_EVACUATE, best_struct[0], best_struct[1])

    # Water-drop near structures
    if resources["water_drops"] > 0:
        best_target = None
        best_score = -1
        for br, bc in burning_cells:
            score = 0
            for sr, sc in zip(*np.where(structures != NO_STRUCTURE)):
                d = abs(int(br) - int(sr)) + abs(int(bc) - int(sc))
                if d <= 4:
                    score += (3 if structures[sr, sc] == HOSPITAL else 1) / max(d, 1)
            if score > best_score:
                best_score = score
                best_target = (int(br), int(bc))
        if best_target and best_score > 0.3:
            return (ACTION_WATERDROP, best_target[0], best_target[1])

    # Firebreak downwind
    if resources["firebreaks"] > 0:
        wind_dr = np.sin(wind_dir)
        wind_dc = np.cos(wind_dir)
        for br, bc in burning_cells[:5]:
            for dist in [2, 3]:
                nr = int(round(int(br) + wind_dr * dist))
                nc = int(round(int(bc) + wind_dc * dist))
                if 0 <= nr < N and 0 <= nc < N and fire_map[nr, nc] == UNBURNED:
                    return (ACTION_FIREBREAK, nr, nc)

    return (ACTION_NOOP, 0, 0)


# ---------------------------------------------------------------------------
# Action label helper
# ---------------------------------------------------------------------------

ACTION_LABELS = {0: "noop", 1: "firebreak", 2: "waterdrop", 3: "evacuate"}


def action_str(action: tuple) -> str:
    a, r, c = action
    label = ACTION_LABELS.get(a, "unknown")
    if a == 0:
        return f"{label}()"
    return f"{label}({r},{c})"


# ---------------------------------------------------------------------------
# Run a single task with structured logging
# ---------------------------------------------------------------------------

def run_task_with_logging(task_id: str, client: OpenAI, n_episodes: int = 1) -> float:
    """Run a task and emit [START]/[STEP]/[END] logs."""
    from tasks import TASKS, _grade

    task = TASKS[task_id]
    config_fn = task["config_fn"]
    grader = task["grader"]

    all_scores = []

    for ep in range(n_episodes):
        cfg = config_fn()
        cfg.seed = cfg.seed + ep
        env = WildfireEnv(cfg)
        state = env.reset()

        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        cumulative_reward = 0.0
        score = 0.0
        success = False

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            done = False
            step = 0
            while not done:
                step += 1
                action = get_llm_action(client, state, history)
                state, reward, done, info = env.step(action)

                cumulative_reward += reward
                rewards.append(reward)
                steps_taken = step

                error = None
                if not info.get("action_valid", True):
                    error = info.get("action_effect", "invalid_action")

                log_step(
                    step=step,
                    action=action_str(action),
                    reward=reward,
                    done=done,
                    error=error,
                )

                history.append(
                    f"Step {step}: {action_str(action)} -> reward {reward:+.2f}"
                )

            # Grade the episode
            result = grader(env, cumulative_reward)
            score = result.score
            success = score > 0.3
            all_scores.append(score)

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            all_scores.append(0.0)

        finally:
            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards,
            )

    return float(np.mean(all_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wildfire Containment -- LLM Inference"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to run",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per task",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: HF_TOKEN (or API_KEY) environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = list(TASKS.keys()) if args.task == "all" else [args.task]

    scores = {}
    for task_id in task_ids:
        avg_score = run_task_with_logging(task_id, client, n_episodes=args.episodes)
        scores[task_id] = avg_score

    # Final summary
    print(f"\n{'='*50}", flush=True)
    print(f"  SUMMARY  |  model={MODEL_NAME}", flush=True)
    print(f"{'='*50}", flush=True)
    for tid, sc in scores.items():
        bar = "#" * int(sc * 30) + "-" * (30 - int(sc * 30))
        print(f"  {tid:8s}  [{bar}]  {sc:.4f}", flush=True)
    avg = np.mean(list(scores.values()))
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
