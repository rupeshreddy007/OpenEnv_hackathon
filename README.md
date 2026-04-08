# Wildfire Containment Environment

An OpenEnv environment where an AI agent acts as an **incident commander**
managing firefighting resources to contain a spreading wildfire on
procedurally-generated terrain.

## The Problem

A wildfire ignites on a grid. Fire spreads realistically based on:

- **Wind** — direction and speed shift unpredictably
- **Slope** — fire races uphill faster
- **Vegetation** — grass burns fast, forest burns slow but intense, rock/water block fire
- **Moisture** — wet soil resists ignition

The agent has **limited resources** each episode:

| Resource | Effect |
|---|---|
| **Water drops** | Suppress fire in a radius, temporarily fireproof cells |
| **Firebreaks** | Permanently block fire on an unburned cell |
| **Evacuations** | Save people in threatened structures (houses, hospitals) |

## API

```python
from src.environment import WildfireEnv
from src.config import get_default_config

env = WildfireEnv(get_default_config())

state = env.reset()          # Generate terrain, ignite fires
state = env.state()          # Observe current grid

# action = (action_type, row, col)
#   action_type: 0=noop, 1=firebreak, 2=waterdrop, 3=evacuate
state, reward, done, info = env.step((2, 10, 10))   # Water-drop at (10,10)
```

### State dictionary

| Key | Shape | Description |
|---|---|---|
| `fire_map` | (N,N) int | Cell states: unburned/burning/burned/firebreak/water |
| `vegetation` | (N,N) int | Grass / shrub / forest / rock / water |
| `elevation` | (N,N) float | Terrain height in metres |
| `moisture` | (N,N) float | Soil moisture 0–1 |
| `structures` | (N,N) int | Houses, hospitals, fire stations |
| `evacuated` | (N,N) bool | Whether a structure was evacuated |
| `wind_direction` | float | Wind direction in radians |
| `wind_speed` | float | Wind speed in m/s |
| `resources` | dict | Remaining water drops, firebreaks, evacuations |
| `timestep` | int | Current step |

## Quick Start

```bash
pip install numpy scipy pytest

# Run example agents (random + greedy heuristic)
python examples/run_random_agent.py

# Run tests
pytest tests/ -v
```

## Configs

| Config | Grid | Fires | Steps | Use case |
|---|---|---|---|---|
| `get_default_config()` | 20×20 | 2 | 80 | Standard training |
| `get_small_config()` | 10×10 | 1 | 40 | Fast debugging |
| `get_inferno_config()` | 30×30 | 5 | 120 | Hard mode |

## Project Structure

```
src/
  config.py        — All environment parameters and presets
  environment.py   — WildfireEnv with step()/reset()/state()
examples/
  run_random_agent.py — Random + greedy heuristic demo
tests/
  test_environment.py — Pytest suite
```
