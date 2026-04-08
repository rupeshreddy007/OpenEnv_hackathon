# 🔥 Wildfire Containment — OpenEnv Environment

An OpenEnv-compliant environment where an AI agent acts as an **incident commander**
managing firefighting resources to contain a spreading wildfire on
procedurally-generated terrain.

> *An AI agent learns to command firefighting resources and contain a spreading wildfire on procedurally-generated terrain.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)](#testing)
[![OpenEnv API](https://img.shields.io/badge/API-step%20%7C%20reset%20%7C%20state-orange.svg)](#api-reference)

---

## Why This Matters

Wildfires destroy **millions of acres** annually. Real incident commanders face the same tradeoffs this environment simulates: limited water, limited crews, structures to protect, and fire behavior driven by wind, terrain, and vegetation. This environment trains AI agents to make optimal resource allocation decisions under uncertainty — a problem directly applicable to real-world wildfire response planning.

---

## Fire Spread Model

Fire spread uses a **simplified Rothermel model** (Rothermel, 1972), the same foundational model used by USDA tools like FARSITE and FlamMap. The per-cell ignition probability is:

```
P = P_base × flammability × (1 + φ_w + φ_s) × moisture_damping
```

| Component | Formula | Source |
|-----------|---------|--------|
| **Wind factor** (φ_w) | `C × (U/U_ref)^B × cos(θ)` | Rothermel eq. 47–52 |
| **Slope factor** (φ_s) | `5.275 × tan²(slope_angle)` | Rothermel eq. 39 |
| **Moisture damping** | `max(0, 1 − M/M_x)` | Rothermel eq. 29 |

### Anderson 13 Fuel Models

Vegetation types map to Anderson's standard fuel classification (Anderson, 1982):

| Env Type | Anderson Model | Fuel Load (t/ac) | Depth (ft) | M_x | Burn Steps |
|----------|---------------|-------------------|------------|-----|------------|
| **GRASS** | FM1 (Short grass) | 0.74 | 1.0 | 0.40 | 1 |
| **SHRUB** | FM5 (Brush) | 3.50 | 2.0 | 0.55 | 2 |
| **FOREST** | FM9 (Timber litter) | 3.50 | 0.2 | 0.65 | 4 |
| **ROCK** | NB | — | — | — | — |
| **WATER** | NB | — | — | — | — |

Each cell is 30m × 30m (matching Landsat pixel resolution), making terrain data interchangeable with real satellite-derived fuel maps.

### Real Terrain Data

The environment supports loading real-world terrain from numpy files:

```python
cfg = EnvironmentConfig(
    real_terrain_path="data/sample_terrain/",
    grid_size=30,
)
env = WildfireEnv(cfg)
```

Expected files in the directory:
- `elevation.npy` — (H, W) float, metres above sea level
- `vegetation.npy` — (H, W) int8, fuel type indices (0–4)
- `moisture.npy` — (H, W) float, soil moisture 0–1 (optional)

Generate sample California chaparral terrain:
```bash
python data/generate_sample_terrain.py
```

---

## Related Work & Positioning

| Tool | Purpose | Our Difference |
|------|---------|----------------|
| **FARSITE** (Finney, 1998) | Operational fire growth simulator for human planners | Not RL-compatible; no step/reset API; real-time only |
| **FlamMap** (Finney, 2006) | Static fire behavior mapping | No temporal dynamics; no agent interaction |
| **Cell2Fire** (Carrasco et al., 2021) | Cell-based fire simulator for harvest planning | Forest management focus; no resource allocation actions |
| **Gymnasium** fire envs | Toy grid-world fire problems | Simplified physics; no Anderson fuels or Rothermel model |

**This environment fills the gap**: a Rothermel-calibrated fire simulator with a standard RL API, real fuel models, and meaningful resource-allocation actions that mirror real incident command decisions.

### References

1. Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in wildland fuels.* USDA Forest Service Research Paper INT-115.
2. Anderson, H.E. (1982). *Aids to determining fuel models for estimating fire behavior.* USDA Forest Service General Technical Report INT-122.
3. Andrews, P.L. (2018). *The Rothermel surface fire spread model and associated developments.* USDA Forest Service GTR RMRS-GTR-371.
4. Finney, M.A. (1998). *FARSITE: Fire Area Simulator.* USDA Forest Service Research Paper RMRS-RP-4.

---

## How It Works

A wildfire ignites on an N×N terrain grid. Fire spreads based on realistic physical dynamics:

| Factor | Effect |
|---|---|
| **Wind** | Fire spreads faster downwind; direction and speed shift unpredictably |
| **Slope** | Fire races uphill, slower downhill |
| **Vegetation** | Grass burns fast, forest burns slow but intense, rock/water block fire |
| **Moisture** | Wet soil resists ignition |
| **Ember Spotting** | Burning cells launch embers 2-5 cells downwind, igniting spot fires ahead of the front |

The agent acts as an **incident commander** with limited resources per episode:

| Resource | Count | Effect |
|---|---|---|
| 🪓 **Firebreaks** | 15 | Permanently block fire on an unburned cell |
| 💧 **Water Drops** | 10 | Suppress fire in a radius, temporarily protect cells |
| 🚑 **Evacuations** | 3 | Save people in threatened structures before fire arrives |
| 🚒 **Fire Stations** | 1-2 | Resupply water and firebreaks every 10 steps (if station survives) |

The agent must learn **when** and **where** to deploy each resource to minimize destruction and save lives.

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train / Evaluate an Agent

```python
from src.environment import WildfireEnv
from src.config import get_default_config

env = WildfireEnv(get_default_config())
state = env.reset()

done = False
while not done:
    action = my_agent.decide(state)                    # Your agent here
    state, reward, done, info = env.step(action)
    my_agent.learn(state, reward)
```

### Run Example Agents (CLI)

```bash
python examples/run_random_agent.py
```

Runs a **random agent** (baseline) and a **greedy heuristic agent** side-by-side with ASCII visualization.

### Launch Interactive Web UI

```bash
python ui/app.py
```

Open **http://localhost:5000** — play manually or watch the built-in AI agent play.

> **Docker**: When running via Docker the UI is at **http://localhost:7860**.

### Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

### `env.reset() → state`
Generate new terrain, ignite fires, return initial state.

### `env.step(action) → (state, reward, done, info)`
Execute one timestep. Action is a tuple `(action_type, row, col)`:

| action_type | Code | Description |
|---|---|---|
| Noop | `0` | Do nothing, skip turn |
| Firebreak | `1` | Dig firebreak at (row, col) |
| Water Drop | `2` | Drop water at (row, col) + radius |
| Evacuate | `3` | Evacuate structure at (row, col) |

### `env.state() → state`
Observe the current grid without changing anything.

### State Dictionary

| Key | Shape | Description |
|---|---|---|
| `fire_map` | (N,N) int | Cell fire state — unburned / burning / burned / firebreak / water-dropped |
| `vegetation` | (N,N) int | Terrain type — grass / shrub / forest / rock / water body |
| `elevation` | (N,N) float | Height in metres (affects fire spread uphill) |
| `moisture` | (N,N) float | Soil moisture 0–1 (higher = harder to ignite) |
| `structures` | (N,N) int | Overlay — house / hospital / fire station |
| `evacuated` | (N,N) bool | Whether each structure has been evacuated |
| `wind_direction` | float | Current wind direction in radians |
| `wind_speed` | float | Current wind speed in m/s |
| `resources` | dict | Remaining `water_drops`, `firebreaks`, `evacuations` |
| `timestep` | int | Current step number |
| `burning_cells` | int | Count of currently burning cells |
| `burned_cells` | int | Count of fully burned cells |
| `total_burnable` | int | Total burnable cells (excludes rock/water) |

### Reward Signal

| Event | Reward |
|---|---|
| Cell burns | −0.05 |
| House burns (not evacuated) | −8.0 |
| Hospital burns (not evacuated) | −20.0 |
| Successful evacuation | +10.0 |
| Wasted resource | −1.0 |
| Saved cell (end of episode) | +0.1 |
| Saved house (end of episode) | +5.0 |
| Saved hospital (end of episode) | +15.0 |
| Fire fully contained | +50.0 bonus |

---

## Scenario Presets

| Config | Grid | Fires | Steps | Difficulty |
|---|---|---|---|---|
| `get_default_config()` | 20×20 | 2 | 80 | Standard |
| `get_small_config()` | 10×10 | 1 | 40 | Easy / Fast debugging |
| `get_inferno_config()` | 30×30 | 5 | 120 | Hard mode |

---

## Project Structure

```
openenv.yaml              — OpenEnv specification file
baseline.py               — Baseline inference script with reproducible scores
Dockerfile                — Deploy to Hugging Face Spaces
requirements.txt          — Python dependencies

src/
  config.py               — Environment parameters, constants, presets
  environment.py          — WildfireEnv (step / reset / state)
  models.py               — Typed dataclass models (Action, Observation, etc.)
  tasks.py                — 3 graded tasks (easy/medium/hard) + grader functions
  __init__.py             — Package exports

examples/
  run_random_agent.py     — Random + greedy heuristic agent demo

tests/
  test_environment.py     — 23 pytest tests

ui/
  app.py                  — Flask web server + AI agent endpoint
  templates/
    index.html            — Interactive web UI (play or watch AI)
```

---

## Tasks & Grading

Three difficulty tiers with deterministic seeds for reproducibility. Scores are normalized to **0.0–1.0**.

| Task | Grid | Fires | Wind | Structures | Steps | Baseline (greedy) |
|------|------|-------|------|------------|-------|-------------------|
| **Easy** --- Brush Fire | 12x12 | 3 | 8 m/s, shifting | 6 houses + 1 hospital | 50 | ~0.41 |
| **Medium** --- Suburban Wildfire | 20x20 | 3 | 8.5 m/s, shifting | 10 houses + 2 hospitals | 80 | ~0.37 |
| **Hard** --- Inferno | 30x30 | 5 | 10 m/s, volatile | 15 houses + 2 hospitals | 120 | ~0.18 |

### Scoring Formula

```
score = 0.40 × terrain_saved_% + 0.40 × structures_saved_% + 0.20 × containment_bonus
```

- **terrain_saved_%** — fraction of burnable cells not burned
- **structures_saved_%** — fraction of houses/hospitals not burned (evacuated structures that burn still count as saved)
- **containment_bonus** — 1.0 if fire is fully extinguished with <50% terrain burned, else 0.0

### Run Baseline

```bash
python baseline.py                        # All tasks, greedy agent
python baseline.py --task easy            # Single task
python baseline.py --agent random         # Random agent baseline
python baseline.py --episodes 10          # More episodes for stability
```

### Use Graders Programmatically

```python
from src.tasks import run_task

def my_agent(state):
    # Your agent logic here
    return (action_type, row, col)

result = run_task("medium", my_agent, n_episodes=5)
print(f"Score: {result.score}")  # 0.0–1.0
```

---

## Typed Models

All inputs/outputs have typed dataclass models in `src/models.py`:

```python
from src.models import WildfireAction, WildfireObservation, StepResult

action = WildfireAction(action_type=1, row=5, col=3)
env.step(action.to_tuple())

obs = WildfireObservation.from_dict(env.state())
print(obs.grid_size, obs.burning_cells)
```

---

## Deployment

### Docker (Hugging Face Spaces)

```bash
docker build -t wildfire-env .
docker run -p 7860:7860 wildfire-env
```

Then open **http://localhost:7860**.

### Local

```bash
python ui/app.py
```

Open **http://localhost:5000**.

---

## Web UI Features

- **Interactive grid** — click cells to place firebreaks, drop water, evacuate buildings
- **AI Agent mode** — watch a heuristic agent play with explanations of each decision
- **Auto-play** — step through the simulation at adjustable speed
- **Live stats** — burning/burned/saved counts, cumulative reward, wind indicator
- **Hover tooltips** — inspect any cell's vegetation, elevation, moisture, fire state
- **Action log** — scrollable history of every action and reward
- **3 presets** — Default, Small, Inferno with optional seed for reproducibility
