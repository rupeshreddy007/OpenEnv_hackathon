"""
OpenEnv server entry point for the Wildfire Containment Environment.

This module provides the server entry point required by the OpenEnv spec.
It wraps the Flask app from ui/app.py.
"""

import sys
import os
import argparse

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, jsonify, request
import numpy as np

from environment import WildfireEnv
from config import (
    EnvironmentConfig, get_default_config, get_small_config, get_inferno_config,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
    UNBURNED, BURNING, BURNED, FIREBREAK, WATER_DROPPED,
    NO_STRUCTURE, HOUSE, HOSPITAL, FIRE_STATION,
)

app = Flask(__name__)

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


@app.route("/reset", methods=["POST"])
def reset():
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


@app.route("/step", methods=["POST"])
def step():
    global cumulative_reward
    if env is None:
        return jsonify({"error": "Call /reset first"}), 400

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


@app.route("/state", methods=["GET"])
def state():
    if env is None:
        return jsonify({"error": "Call /reset first"}), 400
    return jsonify({
        "state": _state_to_json(env.state()),
        "cumulative_reward": round(cumulative_reward, 3),
        "grid_size": env.N,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def main():
    parser = argparse.ArgumentParser(description="Wildfire Containment Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
