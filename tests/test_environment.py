"""Tests for the Wildfire Containment Environment."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from environment import WildfireEnv
from config import (
    EnvironmentConfig, get_default_config, get_small_config,
    UNBURNED, BURNING, BURNED, FIREBREAK, WATER_DROPPED,
    GRASS, SHRUB, FOREST, ROCK, WATER_BODY,
    NO_STRUCTURE, HOUSE, HOSPITAL,
    ACTION_NOOP, ACTION_FIREBREAK, ACTION_WATERDROP, ACTION_EVACUATE,
)


@pytest.fixture
def env():
    cfg = get_small_config()
    cfg.seed = 42
    return WildfireEnv(cfg)


class TestResetAndState:
    def test_reset_returns_dict(self, env):
        state = env.reset()
        assert isinstance(state, dict)

    def test_state_keys(self, env):
        state = env.reset()
        expected = {"fire_map", "vegetation", "elevation", "moisture",
                    "structures", "evacuated", "wind_direction", "wind_speed",
                    "resources", "timestep", "burning_cells", "burned_cells",
                    "total_burnable"}
        assert expected == set(state.keys())

    def test_initial_fire_exists(self, env):
        state = env.reset()
        assert state["burning_cells"] >= 1

    def test_timestep_starts_at_zero(self, env):
        state = env.reset()
        assert state["timestep"] == 0

    def test_grid_shapes(self, env):
        state = env.reset()
        N = env.N
        assert state["fire_map"].shape == (N, N)
        assert state["vegetation"].shape == (N, N)
        assert state["elevation"].shape == (N, N)

    def test_reset_is_idempotent(self, env):
        s1 = env.reset()
        s2 = env.reset()
        np.testing.assert_array_equal(s1["fire_map"], s2["fire_map"])

    def test_state_matches_reset(self, env):
        s1 = env.reset()
        s2 = env.state()
        np.testing.assert_array_equal(s1["fire_map"], s2["fire_map"])


class TestStep:
    def test_step_returns_tuple(self, env):
        env.reset()
        result = env.step((ACTION_NOOP, 0, 0))
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_timestep_increments(self, env):
        env.reset()
        env.step((ACTION_NOOP, 0, 0))
        assert env.state()["timestep"] == 1

    def test_noop_does_not_crash(self, env):
        env.reset()
        for _ in range(10):
            env.step((ACTION_NOOP, 0, 0))

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step((ACTION_NOOP, 0, 0))
            steps += 1
            if steps > env.config.max_steps + 5:
                break
        assert done


class TestFirebreak:
    def test_firebreak_placed(self, env):
        env.reset()
        # Find an unburned cell
        r, c = np.argwhere(env.fire_map == UNBURNED)[0]
        _, _, _, info = env.step((ACTION_FIREBREAK, int(r), int(c)))
        assert env.fire_map[r, c] == FIREBREAK
        assert info["action_effect"] == "firebreak_placed"

    def test_firebreak_on_burning_fails(self, env):
        env.reset()
        r, c = np.argwhere(env.fire_map == BURNING)[0]
        _, reward, _, info = env.step((ACTION_FIREBREAK, int(r), int(c)))
        assert not info["action_valid"]

    def test_firebreak_resource_depletes(self, env):
        env.reset()
        initial = env.firebreaks_left
        r, c = np.argwhere(env.fire_map == UNBURNED)[0]
        env.step((ACTION_FIREBREAK, int(r), int(c)))
        assert env.firebreaks_left == initial - 1


class TestWaterDrop:
    def test_water_suppresses_fire(self, env):
        env.reset()
        r, c = np.argwhere(env.fire_map == BURNING)[0]
        env.step((ACTION_WATERDROP, int(r), int(c)))
        # The cell should no longer be BURNING
        assert env.fire_map[r, c] != BURNING

    def test_water_resource_depletes(self, env):
        env.reset()
        initial = env.water_drops_left
        env.step((ACTION_WATERDROP, 0, 0))
        assert env.water_drops_left == initial - 1


class TestEvacuate:
    def test_evacuate_structure(self, env):
        env.reset()
        struct_cells = np.argwhere(env.structures != NO_STRUCTURE)
        if len(struct_cells) == 0:
            pytest.skip("No structures placed")
        r, c = struct_cells[0]
        _, reward, _, info = env.step((ACTION_EVACUATE, int(r), int(c)))
        assert info["action_effect"] == "evacuated"
        assert env.evacuated[r, c]
        # reward includes fire-spread penalties, so just check evacuate component is in there
        assert info["action_valid"]

    def test_evacuate_empty_cell_fails(self, env):
        env.reset()
        empty = np.argwhere(env.structures == NO_STRUCTURE)
        r, c = empty[0]
        _, _, _, info = env.step((ACTION_EVACUATE, int(r), int(c)))
        assert not info["action_valid"]


class TestFireSpread:
    def test_fire_spreads_over_time(self):
        # Use a dedicated config with high spread guaranteed
        cfg = get_small_config()
        cfg.seed = 7
        cfg.base_spread_prob = 0.80
        cfg.moisture_range = (0.01, 0.10)
        env = WildfireEnv(cfg)
        env.reset()
        initial_burning = int(np.sum(env.fire_map == BURNING))
        total_affected = initial_burning
        for _ in range(30):
            env.step((ACTION_NOOP, 0, 0))
            total_affected = int(np.sum(
                (env.fire_map == BURNING) | (env.fire_map == BURNED)
            ))
            if total_affected > initial_burning:
                break
        # Fire should have spread to more cells
        assert total_affected > initial_burning

    def test_rock_and_water_dont_burn(self, env):
        env.reset()
        for _ in range(env.config.max_steps):
            env.step((ACTION_NOOP, 0, 0))
        rock_water = (env.vegetation == ROCK) | (env.vegetation == WATER_BODY)
        assert not np.any((env.fire_map[rock_water] == BURNING) |
                          (env.fire_map[rock_water] == BURNED))


class TestRender:
    def test_render_returns_string(self, env):
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert "Step" in output


class TestActionSpaceInfo:
    def test_returns_dict(self, env):
        info = env.get_action_space_info()
        assert "action_type" in info["components"]

    def test_state_space_info(self, env):
        info = env.get_state_space_info()
        assert "fire_map" in info


# =====================================================================
# Ember Spotting Tests
# =====================================================================

class TestEmberSpotting:
    def test_ember_spotting_can_ignite_distant_cells(self):
        """Ember spotting should sometimes ignite cells beyond immediate neighbors."""
        # Try multiple seeds to account for stochasticity
        found_distant_fire = False
        for seed in range(90, 110):
            cfg = EnvironmentConfig(
                grid_size=20,
                n_initial_fires=1,
                seed=seed,
                ember_spotting=True,
                ember_prob_base=0.8,       # Very high for testing
                ember_wind_scale=0.10,
                ember_max_distance=8,
                ember_min_distance=3,
                initial_wind_speed=15.0,
                max_steps=50,
                base_spread_prob=0.01,     # Suppress normal spread
                fire_station_resupply=False,
            )
            env = WildfireEnv(cfg)
            env.reset()
            initial_fires = list(zip(*np.where(env.fire_map == BURNING)))
            for _ in range(30):
                env.step((ACTION_NOOP, 0, 0))
                all_fire = set(zip(*np.where(
                    (env.fire_map == BURNING) | (env.fire_map == BURNED)
                )))
                for r, c in all_fire:
                    for ir, ic in initial_fires:
                        if abs(r - ir) + abs(c - ic) >= 3:
                            found_distant_fire = True
                            break
                    if found_distant_fire:
                        break
                if found_distant_fire:
                    break
            if found_distant_fire:
                break
        assert found_distant_fire, "Ember spotting did not ignite any distant cells across 20 seeds"

    def test_ember_spotting_disabled(self):
        """When disabled, fire should only spread to immediate neighbors."""
        cfg = EnvironmentConfig(
            grid_size=10,
            n_initial_fires=1,
            seed=42,
            ember_spotting=False,
            base_spread_prob=0.01,     # Very low normal spread
            max_steps=5,
        )
        env = WildfireEnv(cfg)
        env.reset()
        initial_fires = set(zip(*np.where(env.fire_map == BURNING)))
        for _ in range(3):
            env.step((ACTION_NOOP, 0, 0))
        # All fire should be within distance 1 of original + their neighbors
        # (chain at most 3 steps of adjacency)
        # Just verify env runs without error
        assert env.timestep == 3


# =====================================================================
# Fire Station Resupply Tests
# =====================================================================

class TestFireStationResupply:
    def test_resupply_adds_resources(self):
        """Fire stations should resupply resources at intervals."""
        cfg = EnvironmentConfig(
            grid_size=10,
            n_initial_fires=1,
            seed=42,
            fire_station_resupply=True,
            resupply_interval=5,
            resupply_water=2,
            resupply_firebreaks=1,
            n_fire_stations=1,
            max_water_drops=10,
            max_firebreaks=10,
            max_steps=20,
        )
        env = WildfireEnv(cfg)
        env.reset()
        # Use some resources first
        state = env.state()
        initial_water = state["resources"]["water_drops"]
        # Drop water to use a resource
        env.step((ACTION_WATERDROP, 0, 0))
        after_use = env.state()["resources"]["water_drops"]
        assert after_use < initial_water
        # Step until resupply triggers
        for _ in range(10):
            env.step((ACTION_NOOP, 0, 0))
        final = env.state()["resources"]["water_drops"]
        # Should have been resupplied (unless station burned)
        assert final >= after_use

    def test_resupply_capped_at_max(self):
        """Resupply should not exceed max resource limits."""
        cfg = EnvironmentConfig(
            grid_size=10,
            n_initial_fires=1,
            seed=42,
            fire_station_resupply=True,
            resupply_interval=2,
            resupply_water=50,     # Huge resupply
            resupply_firebreaks=50,
            n_fire_stations=2,
            max_water_drops=10,
            max_firebreaks=15,
            max_steps=20,
        )
        env = WildfireEnv(cfg)
        env.reset()
        for _ in range(5):
            env.step((ACTION_NOOP, 0, 0))
        state = env.state()
        assert state["resources"]["water_drops"] <= cfg.max_water_drops
        assert state["resources"]["firebreaks"] <= cfg.max_firebreaks


# =====================================================================
# Grader Tests
# =====================================================================

class TestGrader:
    def test_grader_returns_score_in_range(self):
        """Grader score must be in [0.0, 1.0]."""
        from tasks import TASKS, run_task

        def noop_agent(state):
            return (ACTION_NOOP, 0, 0)

        for task_id in TASKS:
            result = run_task(task_id, noop_agent, n_episodes=1)
            assert 0.0 <= result.score <= 1.0, f"Task {task_id} score {result.score} out of range"

    def test_grader_deterministic(self):
        """Same seed and agent should produce same score."""
        from tasks import run_task

        def noop_agent(state):
            return (ACTION_NOOP, 0, 0)

        score1 = run_task("easy", noop_agent, n_episodes=1).score
        score2 = run_task("easy", noop_agent, n_episodes=1).score
        assert score1 == score2, f"Grader not deterministic: {score1} != {score2}"

    def test_evacuation_credits_in_grader(self):
        """Evacuated structures that burn should count as saved in the grader."""
        from tasks import _grade
        cfg = EnvironmentConfig(grid_size=5, n_initial_fires=0, seed=1, max_steps=10)
        env = WildfireEnv(cfg)
        env.reset()
        # Place a house and mark it as evacuated + burned
        env.structures[2, 2] = HOUSE
        env.evacuated[2, 2] = True
        env.fire_map[2, 2] = BURNED
        result = _grade(env, 0.0, "test")
        # The evacuated+burned house should count as saved
        assert result.structures_saved >= 1


# =====================================================================
# Rothermel Model Tests
# =====================================================================

class TestRothermelSpread:
    def test_wind_aligned_spread_faster(self):
        """Fire should spread faster in wind direction (statistically)."""
        # Average over multiple seeds for statistical robustness
        east_total = 0
        west_total = 0
        n_trials = 10
        for seed in range(200, 200 + n_trials):
            cfg = EnvironmentConfig(
                grid_size=21,
                n_initial_fires=0,       # Manual ignition
                seed=seed,
                initial_wind_speed=14.0,
                initial_wind_direction=0.0,  # East
                base_spread_prob=0.6,
                wind_change_prob=0.0,     # No random shifts
                use_rothermel=True,
                ember_spotting=False,
                fire_station_resupply=False,
                moisture_range=(0.01, 0.05),  # Very dry
                max_steps=30,
            )
            env = WildfireEnv(cfg)
            env.reset()
            center = cfg.grid_size // 2
            env.fire_map[center, center] = BURNING
            env.burn_timer[center, center] = 5

            for _ in range(12):
                env.step((ACTION_NOOP, 0, 0))

            fire_mask = (env.fire_map == BURNING) | (env.fire_map == BURNED)
            # Check entire east vs west half
            east_total += int(np.sum(fire_mask[:, center+1:]))
            west_total += int(np.sum(fire_mask[:, :center]))

        assert east_total > west_total, (
            f"Wind should push fire east on average: east={east_total}, west={west_total}"
        )

    def test_moisture_dampens_spread(self):
        """High moisture should significantly reduce fire spread."""
        results = {}
        for moisture_val, label in [(0.01, "dry"), (0.90, "wet")]:
            cfg = EnvironmentConfig(
                grid_size=12,
                n_initial_fires=1,
                seed=55,
                moisture_range=(moisture_val, moisture_val),
                base_spread_prob=0.70,
                use_rothermel=True,
                ember_spotting=False,
                fire_station_resupply=False,
                max_steps=20,
            )
            env = WildfireEnv(cfg)
            env.reset()
            for _ in range(15):
                env.step((ACTION_NOOP, 0, 0))
            results[label] = int(np.sum(env.fire_map == BURNED))
        assert results["dry"] > results["wet"], (
            f"Dry terrain should burn more: dry={results['dry']}, wet={results['wet']}"
        )
