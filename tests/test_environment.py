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
    def test_fire_spreads_over_time(self, env):
        env.reset()
        initial_burning = int(np.sum(env.fire_map == BURNING))
        total_affected = initial_burning
        for _ in range(15):
            env.step((ACTION_NOOP, 0, 0))
            total_affected = int(np.sum(
                (env.fire_map == BURNING) | (env.fire_map == BURNED)
            ))
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
