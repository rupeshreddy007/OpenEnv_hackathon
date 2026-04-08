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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
