"""
Generate sample real-terrain data for the Wildfire Containment Environment.

Creates a synthetic-but-realistic terrain patch inspired by California
chaparral landscapes, saved as numpy files that can be loaded via
config.real_terrain_path.

Usage:
    python data/generate_sample_terrain.py

Creates:
    data/sample_terrain/elevation.npy
    data/sample_terrain/vegetation.npy
    data/sample_terrain/moisture.npy
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_terrain")
SIZE = 30  # 30×30 grid (900m × 900m at 30m resolution)


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Elevation: Ridge-and-valley terrain ---
    # Two sine waves at different angles + noise = realistic ridgeline
    x = np.linspace(0, 2 * np.pi, SIZE)
    y = np.linspace(0, 2 * np.pi, SIZE)
    X, Y = np.meshgrid(x, y)

    elevation = (
        300.0                              # Base elevation (metres)
        + 80.0 * np.sin(0.7 * X + 0.3 * Y)   # Primary ridge
        + 30.0 * np.sin(1.5 * X - 0.8 * Y)   # Secondary ridgeline
        + gaussian_filter(rng.normal(0, 10, (SIZE, SIZE)), sigma=2)  # Local variation
    )
    elevation = np.clip(elevation, 150, 500)

    # --- Vegetation: Elevation-dependent (realistic zonation) ---
    # Low elevation: grass → mid: shrub/chaparral → high: forest → ridgetop: rock
    vegetation = np.zeros((SIZE, SIZE), dtype=np.int8)
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    for r in range(SIZE):
        for c in range(SIZE):
            e = elev_norm[r, c]
            roll = rng.random()
            if e < 0.2:
                # Valley floor: grass + some water
                vegetation[r, c] = 4 if roll < 0.08 else 0  # WATER_BODY or GRASS
            elif e < 0.45:
                # Low slopes: grass/shrub mix
                vegetation[r, c] = 0 if roll < 0.4 else 1   # GRASS or SHRUB
            elif e < 0.75:
                # Mid slopes: shrub/forest
                vegetation[r, c] = 1 if roll < 0.35 else 2  # SHRUB or FOREST
            elif e < 0.9:
                # Upper slopes: forest
                vegetation[r, c] = 2 if roll < 0.85 else 3  # FOREST or ROCK
            else:
                # Ridgetop: rock outcrops
                vegetation[r, c] = 3 if roll < 0.6 else 2   # ROCK or FOREST

    # --- Moisture: Inversely correlated with elevation, valley = wet ---
    moisture = 0.7 - 0.5 * elev_norm
    moisture += gaussian_filter(rng.normal(0, 0.08, (SIZE, SIZE)), sigma=2)
    moisture = np.clip(moisture, 0.05, 0.95)

    # --- Save ---
    np.save(os.path.join(OUTPUT_DIR, "elevation.npy"), elevation)
    np.save(os.path.join(OUTPUT_DIR, "vegetation.npy"), vegetation)
    np.save(os.path.join(OUTPUT_DIR, "moisture.npy"), moisture)

    print(f"Terrain data saved to {OUTPUT_DIR}/")
    print(f"  elevation.npy  : shape={elevation.shape}, range=[{elevation.min():.0f}, {elevation.max():.0f}]m")
    print(f"  vegetation.npy : shape={vegetation.shape}, types={dict(zip(*np.unique(vegetation, return_counts=True)))}")
    print(f"  moisture.npy   : shape={moisture.shape}, range=[{moisture.min():.2f}, {moisture.max():.2f}]")


if __name__ == "__main__":
    generate()
