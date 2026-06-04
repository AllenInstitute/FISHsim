"""
Render a series of merFISH image datasets at progressively increasing emitter
density from a single pre-generated scene.

The scene from generate_scene.py is subsampled at multiple density fractions.
Each level is a cumulative superset of the previous one: the same spots appear
at every level they are included in, so decoded results across levels are
directly comparable.

Imaging rounds within each density level are rendered concurrently via
concurrent.futures.ThreadPoolExecutor.  Total CPU threads used is approximately
--num-round-workers × --num-dask-workers.

Usage:
    python -m fishsim.scripts.density_sweep \\
        --scene-dir results/scene_01 \\
        --psf Y:/MERFISHp/.../psf_final.pkl \\
        --density-fractions 0.1 0.25 0.5 1.0 \\
        --dyes CY3 CY5 AF750 \\
        --output-dir results/density_sweep

Output layout:
    <output-dir>/
        density_0.100/
            scene_meta.json
            tile_000_groundtruth.csv
            H1_<tag>_set<N>/
                000/data/              ← zarr image array
                Conv_zscan__000.zarr/  ← empty detection marker
                Conv_zscan__000.xml    ← stage position + z_offsets
            H2_<tag>_set<N>/
                ...
        density_0.250/
            ...
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import concurrent.futures
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import constants as k

from fishsim.src import sim3d
from fishsim.src.imaging import CameraSimulator
from fishsim.src.point_cloud_background import simulate_autofluorescence
from fishsim.scripts.render_images import (
    DYE_REGISTRY,
    _resolve_dyes,
    _find_groundtruth_paths,
    load_psf,
    prepare_psf_fft,
    _render_channel,
    _make_dapi_channel,
    _make_xml,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Render merFISH density sweep from a pre-generated scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scene-dir", required=True,
                   help="Directory produced by generate_scene.py")
    p.add_argument("--psf", required=True,
                   help="Path to PSF pickle keyed by (channel, y_px, x_px)")
    p.add_argument(
        "--psf-position", nargs=2, type=int, default=[1200, 1200],
        metavar=("Y_PX", "X_PX"),
        help="Field position (pixels) for PSF dict lookup (default: 1200 1200)",
    )
    p.add_argument("--psf-channel", type=int, default=0,
                   help="Channel index for PSF dict lookup (default: 0)")
    p.add_argument(
        "--pixel-size", nargs=3, type=float, default=[0.4, 0.1084333, 0.1084333],
        metavar=("Z_UM", "Y_UM", "X_UM"),
        help="Voxel size in µm [z y x] (default: 0.4 0.1084333 0.1084333)",
    )
    p.add_argument("--exposure-ms", type=float, default=50.0,
                   help="Camera exposure time in milliseconds (default: 50)")
    p.add_argument(
        "--brightness-scale", type=float, default=1.0,
        help="Multiplier applied on top of exposure time (default: 1.0)",
    )
    p.add_argument(
        "--dyes", nargs="+", default=["CY3"],
        metavar="DYE",
        help=(
            f"Ordered dye names cycling across bit positions. "
            f"Available: {', '.join(DYE_REGISTRY)}. (default: CY3)"
        ),
    )
    p.add_argument(
        "--dye-wavelengths", nargs="+", type=int, default=None,
        metavar="NM",
        help="Emission wavelengths (nm) matching --dyes length. Default: per-dye defaults.",
    )
    p.add_argument("--output-dir", required=True,
                   help="Base directory for density-level subdirectories")
    p.add_argument(
        "--tag", default="sim",
        help="Middle tag in hyb folder names: H1_<tag>_set<N> (default: sim)",
    )
    p.add_argument(
        "--set-num", type=int, default=1,
        metavar="N",
        help="Set number in hyb folder names (default: 1)",
    )
    p.add_argument(
        "--density-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0],
        metavar="FRAC",
        help=(
            "Emitter fractions to render (0, 1]. Each level is a superset of the "
            "previous. (default: 0.25 0.5 0.75 1.0)"
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible emitter subsampling (default: random)",
    )
    p.add_argument(
        "--num-round-workers", type=int, default=4,
        help="Number of imaging rounds to render concurrently (default: 4)",
    )
    p.add_argument(
        "--num-dask-workers", type=int, default=4,
        help=(
            "Dask worker threads used inside each concurrent round render. "
            "Total threads ≈ num-round-workers × num-dask-workers. (default: 4)"
        ),
    )
    p.add_argument(
        "--block-shape", nargs=3, type=int, default=[40, 700, 700],
        metavar=("Z", "Y", "X"),
        help="Dask block shape for PSF convolution (default: 40 700 700)",
    )
    p.add_argument(
        "--no-dapi", action="store_true", default=False,
        help="Omit the fake DAPI channel (default: include DAPI)",
    )
    p.add_argument(
        "--autofluorescence-rate", type=float, default=0.0,
        metavar="PHOTONS_PER_VOXEL_PER_S",
        help="Mean autofluorescence photon rate per voxel per second. 0 disables (default: 0)",
    )
    p.add_argument("--af-cv", type=float, default=0.3, metavar="CV",
                   help="Cell-to-cell CV for autofluorescence intensity (default: 0.3)")
    p.add_argument("--af-smooth-um", type=float, default=2.0, metavar="UM",
                   help="Within-cell autofluorescence texture scale in µm (default: 2.0)")
    p.add_argument("--camera-spec-file", default=None, metavar="JSON",
                   help="Camera spec JSON (requires --camera-mode)")
    p.add_argument("--camera-qe-file", default=None, metavar="CSV",
                   help="Camera QE curve CSV (requires --camera-mode)")
    p.add_argument("--camera-mode", default=None, metavar="MODE",
                   help="Readout mode key in --camera-spec-file")
    return p.parse_args()


def _build_cumulative_samples(
    df: pd.DataFrame,
    fractions: list[float],
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    """Return one DataFrame per fraction, each a cumulative superset of the last.

    A single random permutation of all rows is generated once; level k uses the
    first floor(fractions[k] * n) rows of that permutation.  Fractions must be
    sorted ascending before calling (callers ensure this).
    """
    n = len(df)
    perm = rng.permutation(n)
    dfs = []
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        dfs.append(df.iloc[perm[:k]].reset_index(drop=True))
    return dfs


def _render_round(
    round_num: int,
    df: pd.DataFrame,
    n_bits: int,
    n_dyes: int,
    dye_channels: list,
    FZ, FR, FC,
    psf_fft: np.ndarray,
    psf_crop_slices,
    psf: np.ndarray,
    volume_shape: list,
    block_shape: tuple,
    pixel_size: list,
    camera: CameraSimulator,
    exposure_s: float,
    dask_workers: int,
    dapi_im: np.ndarray | None,
    autofluorescence_im: np.ndarray | None,
    tile_idx: int,
    volume_um: list,
    hyb_folder: Path,
) -> None:
    """Render one imaging round and write zarr + XML to hyb_folder."""
    import zarr

    n_z = volume_shape[0]
    round_start = round_num * n_dyes
    bits_per_dye = [
        [round_start + di] if round_start + di < n_bits else []
        for di in range(n_dyes)
    ]

    def _render_dye(dye_idx):
        dye, wavelength = dye_channels[dye_idx]
        return _render_channel(
            df, bits_per_dye[dye_idx],
            FZ, FR, FC, psf_fft, psf_crop_slices, psf,
            volume_shape, block_shape,
            dye, wavelength, camera, exposure_s, dask_workers,
            autofluorescence_im=autofluorescence_im,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(dye_channels)) as dye_pool:
        channel_images = list(dye_pool.map(_render_dye, range(len(dye_channels))))

    n_channels_total = n_dyes + (1 if dapi_im is not None else 0)
    all_channels = channel_images + ([dapi_im] if dapi_im is not None else [])
    stacked = np.stack(all_channels, axis=1)          # (nz, n_ch, ny, nx)
    n_y, n_x = stacked.shape[2], stacked.shape[3]
    stacked = stacked.reshape(n_z * n_channels_total, n_y, n_x)
    stacked = np.concatenate(
        [np.zeros((1, n_y, n_x), dtype=np.uint16), stacked], axis=0
    )

    fov_str = f"{tile_idx:03d}"
    hyb_folder.mkdir(parents=True, exist_ok=True)

    fov_path = hyb_folder / fov_str
    zarr.open_group(str(fov_path), mode="w", zarr_format=2)
    z_arr = zarr.open_array(
        str(fov_path / "data"), mode="w",
        shape=stacked.shape, dtype="uint16",
        chunks=(1, *stacked.shape[1:]),
        zarr_format=2,
    )
    z_arr[:] = stacked

    (hyb_folder / f"Conv_zscan__{fov_str}.zarr").mkdir(exist_ok=True)

    dz = pixel_size[0]
    xml = _make_xml(
        stage_x=tile_idx * volume_um[2],
        stage_y=0.0,
        n_frames=1 + n_z * n_channels_total,
        z_start=0.0,
        z_stop=-(n_z * dz),
        z_step=-dz,
        n_channels=n_channels_total,
    )
    (hyb_folder / f"Conv_zscan__{fov_str}.xml").write_text(xml, encoding="ISO-8859-1")

    print(f"  [H{round_num+1} tile{tile_idx}] {fov_path/'data'}  shape={stacked.shape}")


def main():
    args = parse_args()

    try:
        import zarr  # noqa: F401
    except ImportError:
        raise ImportError("zarr is required. Install with: pip install zarr")

    bad = [f for f in args.density_fractions if not (0 < f <= 1.0)]
    if bad:
        raise ValueError(f"All density fractions must be in (0, 1]; got {bad}")

    fractions = sorted(set(args.density_fractions))
    dye_channels = _resolve_dyes(args.dyes, args.dye_wavelengths)
    n_dyes = len(dye_channels)

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "scene_meta.json") as f:
        meta = json.load(f)

    volume_um = meta["volume_um"]
    pixel_size = args.pixel_size
    volume_shape = [
        int(np.ceil(volume_um[i] / pixel_size[i])) for i in range(3)
    ]
    n_bits = meta["n_bits"]
    n_rounds = (n_bits + n_dyes - 1) // n_dyes

    print(f"Volume shape: {volume_shape}  |  {n_bits} bits  |  {n_rounds} rounds")
    print(f"Density levels ({len(fractions)}): {fractions}")
    print(f"Concurrency: {args.num_round_workers} rounds × {args.num_dask_workers} dask workers")

    print(f"Loading PSF from {args.psf}...")
    psf = load_psf(
        Path(args.psf), args.psf_channel,
        args.psf_position[0], args.psf_position[1],
    )
    psf_fft, _, psf_crop_slices, FZ, FR, FC = prepare_psf_fft(psf)

    exposure_s = args.exposure_ms * k.milli * args.brightness_scale

    if args.camera_spec_file and args.camera_mode:
        camera = CameraSimulator(
            spec_file=Path(args.camera_spec_file),
            qe_file=Path(args.camera_qe_file) if args.camera_qe_file else None,
            mode=args.camera_mode,
        )
    else:
        qe_map = {wl: 0.85 for _, wl in dye_channels}
        camera = CameraSimulator(
            QE=qe_map, gain=1.0 / 0.25, bias=100,
            dark_current=1.0, read_noise=1.8, well_depth=15000,
        )

    gt_paths = _find_groundtruth_paths(scene_dir)
    if not gt_paths:
        raise FileNotFoundError(f"No groundtruth CSV files found under {scene_dir}")

    cells_csv = scene_dir / "cells.csv"
    cells_df = pd.read_csv(cells_csv) if cells_csv.exists() else None

    # DAPI and autofluorescence depend only on cell geometry, not emitter density.
    # Compute once and share across all density levels.
    dapi_im = None
    if not args.no_dapi and cells_df is not None:
        print(f"Generating DAPI channel from {len(cells_df)} cells...")
        dapi_im = _make_dapi_channel(cells_df, volume_shape, pixel_size)

    autofluorescence_im = None
    if args.autofluorescence_rate > 0 and cells_df is not None:
        print("Generating autofluorescence volume...")
        autofluorescence_im = simulate_autofluorescence(
            cells_df,
            volume_shape=volume_shape,
            pixel_size=pixel_size,
            mean_photons=args.autofluorescence_rate * exposure_s,
            cv=args.af_cv,
            smooth_scale_um=args.af_smooth_um,
        )

    rng = np.random.default_rng(args.seed)

    for tile_idx, gt_path in enumerate(gt_paths):
        print(f"\n=== Tile {tile_idx + 1}/{len(gt_paths)}: {gt_path} ===")
        df_full = pd.read_csv(
            gt_path, dtype={"barcode": str, "observed_barcode": str}
        )
        n_total = len(df_full)
        print(f"  {n_total} emitters in full scene")

        # Add pixel-coordinate columns to the full table; subsets inherit them.
        df_full[
            ["frame", "row", "column", "frame_shift", "row_shift", "column_shift"]
        ] = sim3d.compute_pixel_locations(
            df_full[["z", "y", "x"]].to_numpy(), pixel_size=pixel_size
        )

        density_dfs = _build_cumulative_samples(df_full, fractions, rng)

        for frac, df_level in zip(fractions, density_dfs):
            n_level = len(df_level)
            label = f"density_{frac:.3f}"
            print(f"\n--- {label}: {n_level}/{n_total} emitters ({frac:.1%}) ---")

            level_dir = output_dir / label
            level_dir.mkdir(parents=True, exist_ok=True)

            gt_out = level_dir / f"tile_{tile_idx:03d}_groundtruth.csv"
            df_level.to_csv(gt_out, index=False)
            print(f"  Groundtruth → {gt_out}")

            level_meta = {**meta, "density_fraction": frac, "n_emitters_rendered": n_level}
            with open(level_dir / "scene_meta.json", "w") as f:
                json.dump(level_meta, f, indent=2)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_round_workers
            ) as pool:
                futures = {
                    pool.submit(
                        _render_round,
                        round_num, df_level, n_bits, n_dyes, dye_channels,
                        FZ, FR, FC, psf_fft, psf_crop_slices, psf,
                        volume_shape, tuple(args.block_shape), pixel_size,
                        camera, exposure_s, args.num_dask_workers,
                        dapi_im, autofluorescence_im,
                        tile_idx, volume_um,
                        level_dir / f"H{round_num + 1}_{args.tag}_set{args.set_num}",
                    ): round_num
                    for round_num in range(n_rounds)
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        rn = futures[fut]
                        raise RuntimeError(
                            f"Round {rn + 1} failed at {label}"
                        ) from exc

    print(f"\nDensity sweep complete. Output in {output_dir}")


if __name__ == "__main__":
    main()
