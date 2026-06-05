"""
Convolve pre-computed emitter density volumes with one or more PSFs to produce
merFISH image datasets, without re-running the scatter step.

Requires a density sweep output directory produced by density_sweep.py with
the --save-density-volumes flag, which saves a pre-convolution float32 zarr
array per (density level, round, dye).

Usage:
    python -m fishsim.scripts.psf_sweep \\
        --sweep-dir results/density_sweep \\
        --psf psfs/60x_1p4na.pkl psfs/40x_1p0na.pkl \\
        --output-dir results/psf_sweep

Output layout:
    <output-dir>/
        60x_1p4na/
            density_0.250/
                H1_<tag>_set<N>/
                    000/data/   ← zarr image
                    ...
        40x_1p0na/
            density_0.250/
                ...
"""

import os

import argparse
import concurrent.futures
import json
import time
import numpy as np
from pathlib import Path
from scipy import constants as k
from scipy.fft import rfftn, irfftn, next_fast_len

from fishsim.src import sim3d
from fishsim.src.imaging import CameraSimulator
from fishsim.scripts.render_images import (
    DYE_REGISTRY,
    _resolve_dyes,
    load_psf,
    prepare_psf_fft,
    _make_dapi_channel,
    _make_xml,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Convolve saved emitter density volumes with alternative PSFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sweep-dir", required=True,
                   help="density_sweep.py output directory (must contain "
                        "density_*/emitter_volumes/ subdirectories)")
    p.add_argument(
        "--psf", nargs="+", required=True, dest="psf_files",
        metavar="PKL",
        help="One or more PSF pickle files. Each produces a separate output subtree.",
    )
    p.add_argument(
        "--psf-labels", nargs="+", default=None,
        metavar="LABEL",
        help="Short label for each PSF used as the output subdirectory name. "
             "Defaults to the PSF filename stem.",
    )
    p.add_argument(
        "--psf-position", nargs=2, type=int, default=[1200, 1200],
        metavar=("Y_PX", "X_PX"),
    )
    p.add_argument("--psf-channel", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--exposure-ms", type=float, default=None,
                   help="Override exposure time (ms). Default: from emitter_volumes_meta.json")
    p.add_argument("--brightness-scale", type=float, default=1.0)
    p.add_argument(
        "--num-round-workers", type=int, default=4,
        help="Concurrent round renders per density level (default: 4)",
    )
    p.add_argument("--no-dapi", action="store_true", default=False)
    p.add_argument("--camera-spec-file", default=None, metavar="JSON")
    p.add_argument("--camera-qe-file", default=None, metavar="CSV")
    p.add_argument("--camera-mode", default=None, metavar="MODE")
    return p.parse_args()


def _find_density_dirs(sweep_dir: Path) -> list[Path]:
    return sorted(d for d in sweep_dir.iterdir()
                  if d.is_dir() and d.name.startswith("density_"))


def _load_emitter_volume(vol_dir: Path, dye_idx: int, dye_name: str) -> np.ndarray:
    import zarr
    z = zarr.open_array(str(vol_dir / f"dye{dye_idx}_{dye_name}.zarr"), mode="r")
    return z[:]


def _convolve_and_render_round(
    round_num: int,
    vol_dir: Path,
    psf: np.ndarray,
    dye_channels: list,
    dye_names: list,
    camera: CameraSimulator,
    exposure_s: float,
    dapi_im: np.ndarray | None,
    pixel_size: list,
    volume_um: list,
    n_bits: int,
    n_dyes: int,
    hyb_folder: Path,
    tile_idx: int,
) -> None:
    """Load emitter volumes for one round, convolve with PSF, write zarr."""
    import zarr

    # Normalise PSF the same way prepare_psf_fft does.
    from fishsim.src import sim3d as _sim3d
    psf_norm = _sim3d.apodize_psf_tukey(psf)
    psf_norm = psf_norm / psf_norm.sum()

    round_start = round_num * n_dyes
    bits_active = [round_start + di < n_bits for di in range(n_dyes)]

    channel_images = []
    for dye_idx, (dye, wavelength) in enumerate(dye_channels):
        if not bits_active[dye_idx]:
            nz, ny, nx = dapi_im.shape if dapi_im is not None else (1, 1, 1)
            channel_images.append(
                camera.simulate_image(
                    np.zeros((nz, ny, nx), dtype=np.float32), wavelength, exposure_s
                )
            )
            continue

        tag = f"H{round_num+1} dye{dye_idx}"
        t0 = time.time()
        emitter_vol = _load_emitter_volume(vol_dir, dye_idx, dye_names[dye_idx])
        print(f"  [{tag}] load       {time.time()-t0:.1f}s  shape={emitter_vol.shape}", flush=True)

        # Full-volume convolution with multithreaded scipy.fft (works on AMD CPUs
        # unlike numpy FFT which ignores MKL threading on non-Intel hardware).
        t0 = time.time()
        fft_shape = tuple(next_fast_len(emitter_vol.shape[i] + psf_norm.shape[i] - 1)
                          for i in range(emitter_vol.ndim))
        fa = rfftn(emitter_vol, s=fft_shape, workers=-1)
        fb = rfftn(psf_norm, s=fft_shape, workers=-1)
        raw = irfftn(fa * fb, s=fft_shape, workers=-1)
        print(f"  [{tag}] fftconvolve {time.time()-t0:.1f}s  fft_shape={fft_shape}", flush=True)

        # Trim to 'same': centered on emitter_vol shape
        starts = tuple((psf_norm.shape[i] - 1) // 2 for i in range(raw.ndim))
        slices = tuple(slice(st, st + emitter_vol.shape[i])
                       for i, st in enumerate(starts))
        photon_base = np.clip(raw[slices], 0, None).astype(np.float32)
        del fa, fb, raw

        t0 = time.time()
        photon_im = dye.psf_to_photon_distribution(photon_base, exposure_s)
        channel_images.append(camera.simulate_image(photon_im, wavelength, exposure_s))
        print(f"  [{tag}] cam+photon  {time.time()-t0:.1f}s", flush=True)

    n_channels_total = n_dyes + (1 if dapi_im is not None else 0)
    all_channels = channel_images + ([dapi_im] if dapi_im is not None else [])
    nz = all_channels[0].shape[0]
    ny, nx = all_channels[0].shape[1], all_channels[0].shape[2]
    stacked = np.stack(all_channels, axis=1).reshape(nz * n_channels_total, ny, nx)
    stacked = np.concatenate(
        [np.zeros((1, ny, nx), dtype=np.uint16), stacked], axis=0
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
        n_frames=1 + nz * n_channels_total,
        z_start=0.0,
        z_stop=-(nz * dz),
        z_step=-dz,
        n_channels=n_channels_total,
    )
    (hyb_folder / f"Conv_zscan__{fov_str}.xml").write_text(xml, encoding="ISO-8859-1")
    print(f"  [H{round_num+1} tile{tile_idx}] {fov_path/'data'}  shape={stacked.shape}")


def main():
    args = parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density_dirs = _find_density_dirs(sweep_dir)
    if not density_dirs:
        raise FileNotFoundError(f"No density_* directories found in {sweep_dir}")

    # Read shared metadata from the first density level.
    first_meta_path = density_dirs[0] / "emitter_volumes_meta.json"
    if not first_meta_path.exists():
        raise FileNotFoundError(
            f"{first_meta_path} not found. Re-run density_sweep.py with "
            "--save-density-volumes."
        )
    with open(first_meta_path) as f:
        ev_meta = json.load(f)

    dye_names = ev_meta["dye_names"]
    dye_wavelengths = ev_meta["dye_wavelengths"]
    dye_channels = _resolve_dyes(dye_names, dye_wavelengths)
    n_dyes = ev_meta["n_dyes"]
    n_bits = ev_meta["n_bits"]
    n_rounds = ev_meta["n_rounds"]
    volume_shape = ev_meta["volume_shape"]
    pixel_size = ev_meta["pixel_size"]
    tag = ev_meta["tag"]
    set_num = ev_meta["set_num"]
    volume_um = [volume_shape[i] * pixel_size[i] for i in range(3)]

    exposure_s = (
        args.exposure_ms * k.milli * args.brightness_scale
        if args.exposure_ms is not None
        else ev_meta["exposure_s"] * args.brightness_scale
    )

    psf_labels = args.psf_labels or [Path(p).stem for p in args.psf_files]
    if len(psf_labels) != len(args.psf_files):
        raise ValueError("--psf-labels must have the same length as --psf")

    if args.camera_spec_file and args.camera_mode:
        camera = CameraSimulator(
            spec_file=Path(args.camera_spec_file),
            qe_file=Path(args.camera_qe_file) if args.camera_qe_file else None,
            mode=args.camera_mode,
        )
    else:
        qe_map = {wl: 0.85 for wl in dye_wavelengths}
        camera = CameraSimulator(
            QE=qe_map, gain=1.0 / 0.25, bias=100,
            dark_current=1.0, read_noise=1.8, well_depth=15000,
        )

    # DAPI: use cells.csv from the original scene directory if present.
    # The sweep_dir should contain a link or copy; fall back to sweep_dir itself.
    dapi_im = None
    if not args.no_dapi:
        cells_csv = sweep_dir / "cells.csv"
        if cells_csv.exists():
            import pandas as pd
            cells_df = pd.read_csv(cells_csv)
            print(f"Generating DAPI from {len(cells_df)} cells...")
            dapi_im = _make_dapi_channel(cells_df, volume_shape, pixel_size)

    t0 = time.time()

    for psf_file, psf_label in zip(args.psf_files, psf_labels):
        print(f"\n=== PSF: {psf_label} ({psf_file}) ===")
        psf = load_psf(
            Path(psf_file), args.psf_channel,
            args.psf_position[0], args.psf_position[1],
        )
        psf_out_dir = output_dir / psf_label

        for density_dir in density_dirs:
            density_label = density_dir.name
            print(f"\n--- {density_label} ---")

            with open(density_dir / "emitter_volumes_meta.json") as f:
                level_meta = json.load(f)

            # Copy scene_meta so downstream tools can read it.
            level_out_dir = psf_out_dir / density_label
            level_out_dir.mkdir(parents=True, exist_ok=True)
            with open(level_out_dir / "scene_meta.json", "w") as f:
                json.dump({**level_meta, "psf_label": psf_label,
                           "psf_file": str(psf_file)}, f, indent=2)

            # Copy groundtruth CSVs verbatim (spot positions are PSF-independent).
            for gt_csv in density_dir.glob("tile_*_groundtruth.csv"):
                import shutil
                shutil.copy2(gt_csv, level_out_dir / gt_csv.name)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_round_workers
            ) as pool:
                futures = {
                    pool.submit(
                        _convolve_and_render_round,
                        round_num,
                        density_dir / "emitter_volumes" / f"H{round_num + 1}",
                        psf, dye_channels, dye_names,
                        camera, exposure_s, dapi_im,
                        pixel_size, volume_um, n_bits, n_dyes,
                        level_out_dir / f"H{round_num + 1}_{tag}_set{set_num}",
                        tile_idx=0,
                    ): round_num
                    for round_num in range(n_rounds)
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        rn = futures[fut]
                        raise RuntimeError(
                            f"Round {rn + 1} failed for {psf_label}/{density_label}"
                        ) from exc

    elapsed_s = time.time() - t0
    print(f"\nPSF sweep complete in {elapsed_s/60:.1f} min. Output in {output_dir}")


if __name__ == "__main__":
    main()
