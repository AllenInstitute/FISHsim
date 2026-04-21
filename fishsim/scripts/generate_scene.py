"""
Generate a point cloud (spots + cells) for a merFISH simulation scene.

Cells are placed as densely as the volume allows (random close packing).
Output is a groundtruth CSV per tile plus a cell geometry table, which
can be passed to render_images.py to produce images without re-running
the expensive spot placement.

Usage (YAML config):
    python -m fishsim.scripts.generate_scene \\
        --config fishsim/resources/configs/scene_example.yml

Usage (command-line only):
    python -m fishsim.scripts.generate_scene \\
        --codebook fishsim/resources/codebooks/C1E1_codebook_no_distribution.csv \\
        --volume 16.0 303.6 303.6 \\
        --cell-radius 10 15 \\
        --emitters-per-cell 500 \\
        --bit-drop 0.05 \\
        --bit-add 0.05 \\
        --output-dir results/scene_01

Command-line arguments override values in the YAML config.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from fishsim.src import sim3d


# Defaults applied after config + CLI merging.
_DEFAULTS = {
    "volume": [16.0, 303.6, 303.6],
    "cell_radius": [10.0, 15.0],
    "emitters_per_cell": 500,
    "bit_drop": 0.0,
    "bit_add": 0.0,
    "packing_fraction": 0.55,
    "tiles": 1,
    "seed": None,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate merFISH scene: spot point cloud + cell geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default=None,
        help="Path to YAML config file. CLI arguments override config values.",
    )
    p.add_argument("--codebook", default=None, help="Path to codebook CSV")
    p.add_argument(
        "--volume", nargs=3, type=float, default=None,
        metavar=("Z_UM", "Y_UM", "X_UM"),
        help="Imaging volume in µm [z y x] (default: 16.0 303.6 303.6)",
    )
    p.add_argument(
        "--cell-radius", nargs=2, type=float, default=None,
        metavar=("MIN_UM", "MAX_UM"),
        help="Cell half-axis min/max in µm; all three axes use the same range (default: 10 15)",
    )
    p.add_argument(
        "--emitters-per-cell", type=int, default=None,
        help="Number of emitters placed per cell (default: 500)",
    )
    p.add_argument(
        "--bit-drop", type=float, default=None,
        metavar="PROB",
        help="Probability that a '1' bit in a barcode is silenced (default: 0.0)",
    )
    p.add_argument(
        "--bit-add", type=float, default=None,
        metavar="PROB",
        help="Probability that a '0' bit is spuriously lit (default: 0.0)",
    )
    p.add_argument(
        "--packing-fraction", type=float, default=None,
        metavar="FRAC",
        help=(
            "Fraction of volume occupied by cells when estimating cell count. "
            "~0.64 for ideal random close-packing of spheres; 0.55 is conservative "
            "to account for rejection-sampling overhead (default: 0.55)"
        ),
    )
    p.add_argument(
        "--tiles", type=int, default=None,
        help="Number of tiles to generate (default: 1)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    p.add_argument("--output-dir", default=None, help="Directory to save outputs")
    return p.parse_args()


def _load_config(config_path: str) -> dict:
    """Load a YAML config file, returning an empty dict on parse failure."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for --config support. Install with: pip install pyyaml"
        )
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _merge_config(args, config: dict) -> dict:
    """Merge CLI args, YAML config, and built-in defaults. CLI takes priority."""
    cli = {
        "codebook": args.codebook,
        "volume": args.volume,
        "cell_radius": getattr(args, "cell_radius", None),
        "emitters_per_cell": getattr(args, "emitters_per_cell", None),
        "bit_drop": getattr(args, "bit_drop", None),
        "bit_add": getattr(args, "bit_add", None),
        "packing_fraction": getattr(args, "packing_fraction", None),
        "tiles": args.tiles,
        "seed": args.seed,
        "output_dir": getattr(args, "output_dir", None),
    }
    merged = {}
    for key in cli:
        if cli[key] is not None:
            merged[key] = cli[key]
        elif key in config:
            merged[key] = config[key]
        elif key in _DEFAULTS:
            merged[key] = _DEFAULTS[key]
    return merged


def _estimate_cell_count(
    volume_um: list, cell_radius_range: list, packing_fraction: float = 0.55
) -> int:
    """Estimate number of non-overlapping ellipsoid cells that fit in the volume.

    Uses random close-packing fraction (~0.64 for spheres; 0.55 is conservative
    given the rejection-sampling overhead in cell_emitter_position).
    """
    mean_r = np.mean(cell_radius_range)
    cell_vol = (4 / 3) * np.pi * mean_r ** 3
    total_vol = np.prod(volume_um)
    return max(1, int(packing_fraction * total_vol / cell_vol))


def _save_cell_geometry(cells, path: Path) -> None:
    """Save cell centre, axes, and rotation matrix to CSV."""
    rows = []
    for i, cell in enumerate(cells):
        row = {
            "cell_id": i,
            "center_x": cell.center[0],
            "center_y": cell.center[1],
            "center_z": cell.center[2],
            "axis_a": cell.axes[0],
            "axis_b": cell.axes[1],
            "axis_c": cell.axes[2],
        }
        for r in range(3):
            for c in range(3):
                row[f"R_{r}{c}"] = cell.R[r, c]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()

    config = _load_config(args.config) if args.config else {}
    cfg = _merge_config(args, config)

    missing = [k for k in ("codebook", "output_dir") if k not in cfg]
    if missing:
        raise ValueError(
            f"Required argument(s) missing: {[k.replace('_', '-') for k in missing]}. "
            "Provide them on the command line or in a --config YAML file."
        )

    if cfg.get("seed") is not None:
        np.random.seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    codebook_df = pd.read_csv(cfg["codebook"], skiprows=3)
    n_bits = len(codebook_df["barcode"].iloc[0].replace(" ", ""))

    cell_count = _estimate_cell_count(cfg["volume"], cfg["cell_radius"], cfg["packing_fraction"])
    print(f"Target cell count for volume {cfg['volume']} µm: {cell_count}")

    sim = sim3d.Simulator(
        emitters_per_cell=cfg["emitters_per_cell"],
        cell_count=cell_count,
        cell_axes={
            "a": cfg["cell_radius"],
            "b": cfg["cell_radius"],
            "c": cfg["cell_radius"],
        },
        bit_drop=cfg["bit_drop"],
        bit_add=cfg["bit_add"],
    )

    groundtruth_paths = sim.generate_point_cloud(
        codebook_filepath=Path(cfg["codebook"]),
        tiles=cfg["tiles"],
        is_subpixel=True,
        is_cell=True,
        is_nucleus=False,
        sample_volume=tuple(cfg["volume"]),
        savepath=output_dir,
    )

    cell_table_path = output_dir / "cells.csv"
    _save_cell_geometry(sim.cells, cell_table_path)
    print(f"Cell geometry saved to {cell_table_path}")

    meta = {
        "volume_um": cfg["volume"],
        "cell_radius_range_um": cfg["cell_radius"],
        "cell_count": cell_count,
        "packing_fraction": cfg["packing_fraction"],
        "emitters_per_cell": cfg["emitters_per_cell"],
        "n_bits": n_bits,
        "bit_drop": cfg["bit_drop"],
        "bit_add": cfg["bit_add"],
        "codebook": str(cfg["codebook"]),
        "tiles": cfg["tiles"],
        "seed": cfg.get("seed"),
        "groundtruth_paths": [str(p) for p in groundtruth_paths],
    }
    with open(output_dir / "scene_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Scene generation complete. Output in {output_dir}")


if __name__ == "__main__":
    main()
