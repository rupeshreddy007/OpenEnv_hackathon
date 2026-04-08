"""
Flask web UI for the Wildfire Containment Environment.

Run:  python ui/app.py
Then open http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, render_template, jsonify, request
import numpy as np

from environment import WildfireEnv
from config import (
    EnvironmentConfig, get_default_config, get_small_config, get_inferno_config,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
    UNBURNED, BURNING, BURNED, FIREBREAK, WATER_DROPPED,
    GRASS, SHRUB, FOREST, ROCK, WATER_BODY,
    NO_STRUCTURE, HOUSE, HOSPITAL, FIRE_STATION,
)

app = Flask(__name__)

# Global env instance
env: WildfireEnv = None
cumulative_reward: float = 0.0


def _state_to_json(state: dict) -> dict:
    """Convert numpy arrays to JSON-serializable lists."""
    out = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global env, cumulative_reward
    data = request.get_json(silent=True) or {}
    preset = data.get("preset", "default")

    if preset == "small":
        cfg = get_small_config()
    elif preset == "inferno":
        cfg = get_inferno_config()
    else:
        cfg = get_default_config()

    seed = data.get("seed")
    if seed is not None:
        cfg.seed = int(seed)

    env = WildfireEnv(cfg)
    state = env.reset()
    cumulative_reward = 0.0

    return jsonify({
        "state": _state_to_json(state),
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "done": False,
        "info": {},
        "grid_size": env.N,
    })


@app.route("/api/step", methods=["POST"])
def api_step():
    global cumulative_reward
    if env is None:
        return jsonify({"error": "Call /api/reset first"}), 400

    data = request.get_json()
    action_type = int(data.get("action", ACTION_NOOP))
    row = int(data.get("row", 0))
    col = int(data.get("col", 0))

    state, reward, done, info = env.step((action_type, row, col))
    cumulative_reward += reward

    return jsonify({
        "state": _state_to_json(state),
        "reward": round(reward, 3),
        "cumulative_reward": round(cumulative_reward, 3),
        "done": done,
        "info": info,
        "grid_size": env.N,
    })


@app.route("/api/state", methods=["GET"])
def api_state():
    if env is None:
        return jsonify({"error": "Call /api/reset first"}), 400
    return jsonify({
        "state": _state_to_json(env.state()),
        "cumulative_reward": round(cumulative_reward, 3),
        "grid_size": env.N,
    })


@app.route("/api/agent_step", methods=["POST"])
def api_agent_step():
    """Let the heuristic AI agent pick and execute an action."""
    global cumulative_reward
    if env is None:
        return jsonify({"error": "Call /api/reset first"}), 400

    action, reason = _agent_decide()
    action_type, row, col = action

    state, reward, done, info = env.step(action)
    cumulative_reward += reward
    info["agent_reason"] = reason

    return jsonify({
        "state": _state_to_json(state),
        "reward": round(reward, 3),
        "cumulative_reward": round(cumulative_reward, 3),
        "done": done,
        "info": info,
        "grid_size": env.N,
        "agent_action": {"type": action_type, "row": row, "col": col},
    })


@app.route("/api/danger_map", methods=["GET"])
def api_danger_map():
    """
    Compute per-cell fire ignition risk using the Rothermel spread model.

    Returns an NxN array of values in [0, 1] where 1 = highest danger.
    Only unburned cells adjacent to fire have nonzero risk.
    """
    if env is None:
        return jsonify({"error": "Call /api/reset first"}), 400

    from config import FUEL_PROPERTIES

    N = env.N
    cfg = env.config
    danger = np.zeros((N, N), dtype=np.float64)

    burning_cells = np.argwhere(env.fire_map == BURNING)
    if len(burning_cells) == 0:
        return jsonify({"danger_map": danger.tolist()})

    for r, c in burning_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= N or nc < 0 or nc >= N:
                continue

            target_state = env.fire_map[nr, nc]
            if target_state in (BURNING, BURNED, FIREBREAK):
                continue
            if env.water_timer[nr, nc] > 0:
                continue

            veg = int(env.vegetation[nr, nc])
            flammability = cfg.vegetation_flammability[veg]
            if flammability <= 0:
                continue

            fuel = FUEL_PROPERTIES[veg]
            prob = cfg.base_spread_prob * flammability

            if cfg.use_rothermel:
                angle_to_neighbor = np.arctan2(dr, dc)
                wind_alignment = np.cos(angle_to_neighbor - env.wind_dir)
                if wind_alignment > 0:
                    phi_w = (cfg.rothermel_wind_C
                             * (env.wind_speed / cfg.rothermel_wind_ref)
                             ** cfg.rothermel_wind_B
                             * wind_alignment)
                    prob *= (1.0 + phi_w)

                elev_diff = env.elevation[nr, nc] - env.elevation[r, c]
                dist_m = cfg.cell_size_m * (1.414 if abs(dr) + abs(dc) == 2 else 1.0)
                slope_angle = np.arctan2(max(elev_diff, 0), dist_m)
                tan_slope = np.tan(slope_angle)
                phi_s = cfg.rothermel_slope_factor * tan_slope ** 2
                prob *= (1.0 + phi_s)

                M = env.moisture[nr, nc]
                M_x = fuel["moisture_extinction"]
                if M_x > 0:
                    moisture_damping = max(0.0, 1.0 - M / M_x)
                else:
                    moisture_damping = 0.0
                prob *= moisture_damping

            prob = float(np.clip(prob, 0.0, 1.0))
            # Take max across all burning neighbors
            danger[nr, nc] = max(danger[nr, nc], prob)

    # Ember spotting risk — cells downwind within ember range
    if cfg.ember_spotting and len(burning_cells) > 0:
        for r, c in burning_cells:
            for dist in range(cfg.ember_min_distance, cfg.ember_max_distance + 1):
                for scatter in [-0.4, 0.0, 0.4]:
                    angle = env.wind_dir + scatter
                    nr = int(round(r + np.sin(angle) * dist))
                    nc = int(round(c + np.cos(angle) * dist))
                    if nr < 0 or nr >= N or nc < 0 or nc >= N:
                        continue
                    if env.fire_map[nr, nc] != UNBURNED:
                        continue
                    veg = int(env.vegetation[nr, nc])
                    flam = cfg.vegetation_flammability[veg]
                    if flam <= 0:
                        continue
                    ember_prob = (cfg.ember_prob_base + cfg.ember_wind_scale * env.wind_speed) \
                                 * flam * 0.3 / dist
                    ember_prob = float(np.clip(ember_prob, 0.0, 0.5))
                    danger[nr, nc] = max(danger[nr, nc], ember_prob)

    return jsonify({"danger_map": danger.tolist()})


def _agent_decide():
    """
    Heuristic AI agent strategy:
    1. Evacuate structures near fire (highest priority)
    2. Water-drop the most dangerous burning cells (near structures)
    3. Place firebreaks ahead of the fire (downwind of burning cells)
    4. Noop if nothing useful to do
    """
    N = env.N
    fire = env.fire_map
    structs = env.structures
    evac = env.evacuated

    burning_cells = list(zip(*np.where(fire == BURNING)))
    if not burning_cells:
        return (ACTION_NOOP, 0, 0), "no fire"

    # --- Priority 1: Evacuate structures threatened by nearby fire ---
    if env.evacuations_left > 0:
        best_struct = None
        best_dist = float("inf")
        for r, c in zip(*np.where((structs != NO_STRUCTURE) & (~evac))):
            for br, bc in burning_cells:
                d = abs(r - br) + abs(c - bc)  # Manhattan distance
                threat = d
                # Hospitals more urgent
                if structs[r, c] == HOSPITAL:
                    threat -= 3
                if threat < best_dist:
                    best_dist = threat
                    best_struct = (int(r), int(c))
        if best_struct and best_dist <= 5:
            return (ACTION_EVACUATE, best_struct[0], best_struct[1]), \
                f"evacuate {'hospital' if structs[best_struct] == HOSPITAL else 'house'} at {best_struct} (fire {best_dist} away)"

    # --- Priority 2: Water-drop burning cells near structures ---
    if env.water_drops_left > 0:
        best_target = None
        best_score = -1
        for br, bc in burning_cells:
            score = 0
            # Score by proximity to structures
            for sr, sc in zip(*np.where(structs != NO_STRUCTURE)):
                d = abs(br - sr) + abs(bc - sc)
                if d <= 4:
                    val = 3 if structs[sr, sc] == HOSPITAL else 1
                    score += val / max(d, 1)
            # Also score by number of unburned neighbors (save more terrain)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < N and 0 <= nc < N and fire[nr, nc] == UNBURNED:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_target = (br, bc)
        if best_target and best_score > 0.5:
            return (ACTION_WATERDROP, int(best_target[0]), int(best_target[1])), \
                f"water-drop at {best_target} (threat score {best_score:.1f})"

    # --- Priority 3: Firebreak ahead of fire (downwind) ---
    if env.firebreaks_left > 0:
        wind_dr = np.sin(env.wind_dir)
        wind_dc = np.cos(env.wind_dir)
        best_fb = None
        best_score = -1
        # Check cells 2-3 steps downwind of each burning cell
        checked = set()
        for br, bc in burning_cells:
            for dist in [2, 3, 1]:
                nr = int(round(br + wind_dr * dist))
                nc = int(round(bc + wind_dc * dist))
                if (nr, nc) in checked:
                    continue
                checked.add((nr, nc))
                if 0 <= nr < N and 0 <= nc < N and fire[nr, nc] == UNBURNED:
                    veg = env.vegetation[nr, nc]
                    if veg in (ROCK, WATER_BODY):
                        continue
                    # Score: prefer cells that block more fire paths
                    score = 1.0
                    # Bonus for blocking towards structures
                    for sr, sc in zip(*np.where(structs != NO_STRUCTURE)):
                        d = abs(nr - sr) + abs(nc - sc)
                        if d <= 5:
                            score += 2.0 / max(d, 1)
                    if score > best_score:
                        best_score = score
                        best_fb = (nr, nc)
        if best_fb:
            return (ACTION_FIREBREAK, int(best_fb[0]), int(best_fb[1])), \
                f"firebreak at {best_fb} (downwind, score {best_score:.1f})"

    return (ACTION_NOOP, 0, 0), "no useful action available"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port)
