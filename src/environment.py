"""
Wildfire Containment Environment — OpenEnv standard API.

An AI agent acts as an incident commander managing limited firefighting
resources on a terrain grid where fire spreads realistically based on
wind, slope, vegetation, and moisture.

API:
  - reset() -> state dict : Generate new terrain, ignite fires, return initial state
  - step(action) -> (state, reward, done, info) : Execute one action, advance fire
  - state() -> state dict  : Return current observable state
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from config import (
    EnvironmentConfig,
    UNBURNED, BURNING, BURNED, FIREBREAK, WATER_DROPPED,
    GRASS, SHRUB, FOREST, ROCK, WATER_BODY,
    NO_STRUCTURE, HOUSE, HOSPITAL, FIRE_STATION,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)


class WildfireEnv:
    """
    Wildfire Containment Environment.

    The agent commands firefighting resources each timestep to contain a
    spreading wildfire on a procedurally-generated terrain grid.

    Observation (state dict):
        fire_map        : (N,N) int   – cell states (UNBURNED/BURNING/BURNED/FIREBREAK/WATER_DROPPED)
        vegetation      : (N,N) int   – vegetation type per cell
        elevation       : (N,N) float – elevation in metres
        moisture        : (N,N) float – soil moisture 0-1
        structures      : (N,N) int   – structure overlay
        evacuated       : (N,N) bool  – whether structure was evacuated
        wind_direction  : float       – radians
        wind_speed      : float       – m/s
        resources       : dict        – remaining water_drops, firebreaks, evacuations
        timestep        : int
        burning_cells   : int         – count of currently burning cells
        burned_cells    : int         – count of fully burned cells

    Action:
        (action_type, row, col)
        action_type ∈ {0=noop, 1=firebreak, 2=waterdrop, 3=evacuate}
        row, col    ∈ [0, grid_size)
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self.N = self.config.grid_size
        self.rng = np.random.default_rng(self.config.seed)

        # Will be initialised in reset()
        self.fire_map: np.ndarray = None
        self.vegetation: np.ndarray = None
        self.elevation: np.ndarray = None
        self.moisture: np.ndarray = None
        self.structures: np.ndarray = None
        self.evacuated: np.ndarray = None
        self.burn_timer: np.ndarray = None       # Steps remaining for each burning cell
        self.water_timer: np.ndarray = None      # Steps of water suppression remaining

        self.wind_dir: float = 0.0
        self.wind_speed: float = 0.0

        self.water_drops_left: int = 0
        self.firebreaks_left: int = 0
        self.evacuations_left: int = 0

        self.timestep: int = 0
        self._total_burnable: int = 0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Generate fresh terrain, ignite fires, return initial state."""
        # Re-seed RNG so consecutive resets are deterministic
        self.rng = np.random.default_rng(self.config.seed)
        N = self.N

        # --- Terrain ---
        self.vegetation = self.rng.choice(
            len(self.config.vegetation_probs),
            size=(N, N),
            p=self.config.vegetation_probs,
        ).astype(np.int8)

        # Perlin-ish elevation via smoothed noise
        raw = self.rng.random((N, N))
        from scipy.ndimage import gaussian_filter
        self.elevation = gaussian_filter(raw, sigma=3) * self.config.elevation_scale

        lo, hi = self.config.moisture_range
        self.moisture = self.rng.uniform(lo, hi, size=(N, N))

        # --- Structures ---
        self.structures = np.full((N, N), NO_STRUCTURE, dtype=np.int8)
        self._place_structures(HOUSE, self.config.n_houses)
        self._place_structures(HOSPITAL, self.config.n_hospitals)
        self._place_structures(FIRE_STATION, self.config.n_fire_stations)
        self.evacuated = np.zeros((N, N), dtype=bool)

        # --- Fire ---
        self.fire_map = np.full((N, N), UNBURNED, dtype=np.int8)
        self.burn_timer = np.zeros((N, N), dtype=np.int32)
        self.water_timer = np.zeros((N, N), dtype=np.int32)

        burnable = np.argwhere(
            (self.vegetation != ROCK) & (self.vegetation != WATER_BODY)
        )
        ignite_idx = self.rng.choice(
            len(burnable),
            size=min(self.config.n_initial_fires, len(burnable)),
            replace=False,
        )
        for idx in ignite_idx:
            r, c = burnable[idx]
            self.fire_map[r, c] = BURNING
            self.burn_timer[r, c] = self.config.vegetation_burn_rate[int(self.vegetation[r, c])]

        self._total_burnable = int(np.sum(
            (self.vegetation != ROCK) & (self.vegetation != WATER_BODY)
        ))

        # --- Wind ---
        self.wind_dir = self.config.initial_wind_direction
        self.wind_speed = self.config.initial_wind_speed

        # --- Resources ---
        self.water_drops_left = self.config.max_water_drops
        self.firebreaks_left = self.config.max_firebreaks
        self.evacuations_left = self.config.max_evacuations

        self.timestep = 0
        return self.state()

    def state(self) -> Dict[str, Any]:
        """Return the full observable state."""
        return {
            "fire_map": self.fire_map.copy(),
            "vegetation": self.vegetation.copy(),
            "elevation": self.elevation.copy(),
            "moisture": self.moisture.copy(),
            "structures": self.structures.copy(),
            "evacuated": self.evacuated.copy(),
            "wind_direction": self.wind_dir,
            "wind_speed": self.wind_speed,
            "resources": {
                "water_drops": self.water_drops_left,
                "firebreaks": self.firebreaks_left,
                "evacuations": self.evacuations_left,
            },
            "timestep": self.timestep,
            "burning_cells": int(np.sum(self.fire_map == BURNING)),
            "burned_cells": int(np.sum(self.fire_map == BURNED)),
            "total_burnable": self._total_burnable,
        }

    def step(
        self, action: Tuple[int, int, int]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: (action_type, row, col)

        Returns:
            (state, reward, done, info)
        """
        action_type, row, col = action
        row = int(np.clip(row, 0, self.N - 1))
        col = int(np.clip(col, 0, self.N - 1))

        info: Dict[str, Any] = {"action_valid": True, "action_effect": ""}
        reward = 0.0

        # ---------- Apply agent action ----------
        if action_type == ACTION_NOOP:
            info["action_effect"] = "noop"

        elif action_type == ACTION_FIREBREAK:
            if self.firebreaks_left <= 0:
                reward += self.config.penalty_wasted_resource
                info["action_valid"] = False
                info["action_effect"] = "no_firebreaks_left"
            elif self.fire_map[row, col] != UNBURNED:
                reward += self.config.penalty_wasted_resource
                info["action_valid"] = False
                info["action_effect"] = "cell_not_unburned"
            else:
                self.fire_map[row, col] = FIREBREAK
                self.firebreaks_left -= 1
                info["action_effect"] = "firebreak_placed"

        elif action_type == ACTION_WATERDROP:
            if self.water_drops_left <= 0:
                reward += self.config.penalty_wasted_resource
                info["action_valid"] = False
                info["action_effect"] = "no_water_left"
            else:
                self.water_drops_left -= 1
                r_lo = max(0, row - self.config.water_drop_radius)
                r_hi = min(self.N, row + self.config.water_drop_radius + 1)
                c_lo = max(0, col - self.config.water_drop_radius)
                c_hi = min(self.N, col + self.config.water_drop_radius + 1)
                any_effect = False
                for r in range(r_lo, r_hi):
                    for c in range(c_lo, c_hi):
                        if self.fire_map[r, c] == BURNING:
                            self.fire_map[r, c] = WATER_DROPPED
                            self.water_timer[r, c] = self.config.water_suppress_duration
                            self.burn_timer[r, c] = 0
                            any_effect = True
                        elif self.fire_map[r, c] == UNBURNED:
                            self.water_timer[r, c] = self.config.water_suppress_duration
                            any_effect = True
                if not any_effect:
                    reward += self.config.penalty_wasted_resource * 0.5
                info["action_effect"] = "water_dropped"

        elif action_type == ACTION_EVACUATE:
            if self.evacuations_left <= 0:
                reward += self.config.penalty_wasted_resource
                info["action_valid"] = False
                info["action_effect"] = "no_evacuations_left"
            elif self.structures[row, col] == NO_STRUCTURE:
                reward += self.config.penalty_wasted_resource
                info["action_valid"] = False
                info["action_effect"] = "no_structure_here"
            elif self.evacuated[row, col]:
                reward += self.config.penalty_wasted_resource
                info["action_valid"] = False
                info["action_effect"] = "already_evacuated"
            else:
                self.evacuated[row, col] = True
                self.evacuations_left -= 1
                reward += self.config.reward_successful_evacuate
                info["action_effect"] = "evacuated"

        # ---------- Fire dynamics ----------
        cells_before = int(np.sum(self.fire_map == BURNED))
        self._spread_fire()
        self._advance_burn_timers()
        self._decay_water_timers()
        cells_after = int(np.sum(self.fire_map == BURNED))

        newly_burned = cells_after - cells_before
        reward += newly_burned * self.config.penalty_per_burned_cell

        # Penalty for burned structures
        burned_mask = self.fire_map == BURNED
        reward += int(np.sum(burned_mask & (self.structures == HOUSE) & (~self.evacuated))) \
            * self.config.penalty_per_burned_house * (1 if newly_burned > 0 else 0)
        reward += int(np.sum(burned_mask & (self.structures == HOSPITAL) & (~self.evacuated))) \
            * self.config.penalty_per_burned_hospital * (1 if newly_burned > 0 else 0)

        # ---------- Wind shift ----------
        self._update_wind()

        # ---------- Termination ----------
        self.timestep += 1
        burning_count = int(np.sum(self.fire_map == BURNING))
        done = self.timestep >= self.config.max_steps or burning_count == 0

        if done:
            reward += self._terminal_reward()
            if burning_count == 0 and cells_after < self._total_burnable * 0.5:
                reward += self.config.containment_bonus

        info["newly_burned"] = newly_burned
        info["burning_cells"] = burning_count

        return self.state(), reward, done, info

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _place_structures(self, struct_type: int, count: int):
        """Place structures on random non-water, non-rock cells."""
        candidates = np.argwhere(
            (self.vegetation != ROCK)
            & (self.vegetation != WATER_BODY)
            & (self.structures == NO_STRUCTURE)
        )
        if len(candidates) == 0:
            return
        chosen = self.rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        for idx in chosen:
            r, c = candidates[idx]
            self.structures[r, c] = struct_type

    def _spread_fire(self):
        """Spread fire to neighbouring cells."""
        N = self.N
        new_ignitions = []

        burning_cells = np.argwhere(self.fire_map == BURNING)
        for r, c in burning_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= N or nc < 0 or nc >= N:
                    continue

                target_state = self.fire_map[nr, nc]
                if target_state in (BURNING, BURNED, FIREBREAK):
                    continue

                # Water suppression active
                if self.water_timer[nr, nc] > 0:
                    continue

                veg = int(self.vegetation[nr, nc])
                flammability = self.config.vegetation_flammability[veg]
                if flammability <= 0:
                    continue

                # Base spread probability
                prob = self.config.base_spread_prob * flammability

                # Wind bonus: fire spreads more in downwind direction
                angle_to_neighbor = np.arctan2(dr, dc)
                wind_alignment = np.cos(angle_to_neighbor - self.wind_dir)
                if wind_alignment > 0:
                    wind_factor = wind_alignment * (self.wind_speed / 10.0)
                    prob += self.config.wind_spread_bonus * wind_factor

                # Uphill bonus
                elev_diff = self.elevation[nr, nc] - self.elevation[r, c]
                if elev_diff > 0:
                    prob += self.config.uphill_spread_bonus * min(elev_diff / 30.0, 1.0)

                # Moisture penalty
                prob -= self.config.moisture_spread_penalty * self.moisture[nr, nc]

                prob = np.clip(prob, 0.0, 0.95)

                if self.rng.random() < prob:
                    new_ignitions.append((nr, nc))

        for r, c in new_ignitions:
            if self.fire_map[r, c] in (UNBURNED, WATER_DROPPED):
                veg = int(self.vegetation[r, c])
                self.fire_map[r, c] = BURNING
                self.burn_timer[r, c] = self.config.vegetation_burn_rate[veg]

    def _advance_burn_timers(self):
        """Tick down burn timers; cells that finish burning become BURNED."""
        burning = self.fire_map == BURNING
        self.burn_timer[burning] -= 1
        exhausted = burning & (self.burn_timer <= 0)
        self.fire_map[exhausted] = BURNED

    def _decay_water_timers(self):
        """Tick down water suppression timers."""
        active = self.water_timer > 0
        self.water_timer[active] -= 1
        # Cells whose water timer expired go back to UNBURNED (if WATER_DROPPED)
        expired = (self.water_timer == 0) & (self.fire_map == WATER_DROPPED)
        self.fire_map[expired] = UNBURNED

    def _update_wind(self):
        """Randomly shift wind direction and speed."""
        if self.rng.random() < self.config.wind_change_prob:
            shift = self.rng.uniform(
                -self.config.max_wind_shift, self.config.max_wind_shift
            )
            self.wind_dir += shift
            lo, hi = self.config.wind_speed_range
            speed_delta = self.rng.uniform(-1.5, 1.5)
            self.wind_speed = float(np.clip(self.wind_speed + speed_delta, lo, hi))

    def _terminal_reward(self) -> float:
        """Compute end-of-episode bonus/penalty based on final grid state."""
        reward = 0.0
        saved = np.sum(self.fire_map == UNBURNED) + np.sum(self.fire_map == FIREBREAK)
        reward += saved * self.config.reward_per_saved_cell

        saved_houses = int(np.sum(
            (self.structures == HOUSE)
            & (self.fire_map != BURNED)
        ))
        saved_hospitals = int(np.sum(
            (self.structures == HOSPITAL)
            & (self.fire_map != BURNED)
        ))
        reward += saved_houses * self.config.reward_per_saved_house
        reward += saved_hospitals * self.config.reward_per_saved_hospital

        return reward

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_action_space_info(self) -> Dict[str, Any]:
        return {
            "type": "discrete_tuple",
            "components": {
                "action_type": {"values": [0, 1, 2, 3],
                                "labels": ["noop", "firebreak", "waterdrop", "evacuate"]},
                "row": {"range": [0, self.N - 1]},
                "col": {"range": [0, self.N - 1]},
            },
        }

    def get_state_space_info(self) -> Dict[str, Any]:
        return {
            "fire_map": f"({self.N},{self.N}) int – cell fire states",
            "vegetation": f"({self.N},{self.N}) int – vegetation type",
            "elevation": f"({self.N},{self.N}) float – metres",
            "moisture": f"({self.N},{self.N}) float – 0-1",
            "structures": f"({self.N},{self.N}) int – structure overlay",
            "wind_direction": "float radians",
            "wind_speed": "float m/s",
            "resources": "dict of remaining resource counts",
            "timestep": "int",
        }

    def render(self) -> str:
        """ASCII render of the grid for debugging."""
        symbols = {UNBURNED: ".", BURNING: "🔥", BURNED: "▓",
                   FIREBREAK: "#", WATER_DROPPED: "~"}
        struct_sym = {HOUSE: "H", HOSPITAL: "+", FIRE_STATION: "F"}

        lines = [f"Step {self.timestep}/{self.config.max_steps}  "
                 f"Wind {np.degrees(self.wind_dir):.0f}° @ {self.wind_speed:.1f}m/s  "
                 f"💧{self.water_drops_left} 🪓{self.firebreaks_left} 🚑{self.evacuations_left}"]
        for r in range(self.N):
            row_chars = []
            for c in range(self.N):
                s = self.structures[r, c]
                f = self.fire_map[r, c]
                if f == BURNING:
                    row_chars.append("🔥")
                elif s != NO_STRUCTURE and f != BURNED:
                    row_chars.append(f" {struct_sym.get(s, '?')}")
                else:
                    row_chars.append(f" {symbols.get(f, '?')}")
            lines.append("".join(row_chars))

        burned = int(np.sum(self.fire_map == BURNED))
        burning = int(np.sum(self.fire_map == BURNING))
        lines.append(f"Burning: {burning}  Burned: {burned}/{self._total_burnable}")
        return "\n".join(lines)
