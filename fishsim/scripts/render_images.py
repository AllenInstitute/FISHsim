"""
Render merFISH images from a scene produced by generate_scene.py.

The PSF file is expected to be a pickle containing a dict-like object keyed
by (channel_index, y_pixel, x_pixel), as produced by the pilot-data PSF
fitting pipeline (e.g. psf_final.pkl).

Output is written in MERMAKE-compatible format:
    <output_dir>/
        H1_<tag>_set<N>/
            Conv_zscan__001.zarr/   ← empty detection marker; 001 = FOV id
            Conv_zscan__001.xml     ← stage position + z_offsets metadata
            001/data/               ← zarr image array (actual data)
            Conv_zscan__002.zarr/
            Conv_zscan__002.xml
            002/data/
        H2_<tag>_set<N>/
            ...

The hyb folder name matches the MERMAKE regex ([A-z]+)(\d+)_([^_]+)_set(\d+)(.*).

Each zarr array has shape (n_z * n_channels, n_y, n_x) and dtype uint16.
The first axis cycles through dye channels for each z-plane:
    index 0: channel 0, z=0
    index 1: channel 1, z=0
    ...
    index n_channels: channel 0, z=1
    ...

A fake DAPI channel is prepended as channel 0 of every imaging round.
It is derived from the cell geometry in cells.csv: each cell's ellipsoid
is shrunk to 50 % of its original size and rasterised as a binary mask
at a fixed intensity.  Pass --no-dapi to suppress this channel.

Usage:
    python -m fishsim.scripts.render_images \\
        --scene-dir results/scene_01 \\
        --psf Y:/MERFISHp/.../psf_final.pkl \\
        --dyes CY5 AF750 \\
        --tag sim \\
        --set-num 1 \\
        --output-dir results/images_01
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import constants as k
from scipy.fft import next_fast_len
from dask.diagnostics import ProgressBar

from fishsim.src import sim3d
from fishsim.src.imaging import CameraSimulator, DyeSimulator, CY3, CY5, AF750
from fishsim.src.point_cloud_background import simulate_autofluorescence


# Registry of named dye instances available from the CLI.
# Add entries here as new fluorophores are characterised.
DYE_REGISTRY: dict[str, DyeSimulator] = {
    "CY3": CY3,
    "CY5": CY5,
    "AF750": AF750,
}


def parse_args():
    p = argparse.ArgumentParser(description="Render merFISH images from a scene")
    p.add_argument("--scene-dir", required=True,
                   help="Directory produced by generate_scene.py")
    p.add_argument("--psf", required=True,
                   help="Path to PSF file (.pkl dict keyed by (channel, y_px, x_px))")
    p.add_argument(
        "--psf-position", nargs=2, type=int, default=[1200, 1200],
        metavar=("Y_PX", "X_PX"),
        help="Field position (pixels) used as key into the PSF dict (default: 1200 1200)",
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
        help=(
            "Multiplier applied on top of exposure time to scale overall signal. "
            "Values > 1 brighten; < 1 dim. (default: 1.0)"
        ),
    )
    p.add_argument(
        "--dyes", nargs="+", default=["CY3"],
        metavar="DYE",
        help=(
            "Ordered list of dye names that cycle across bit positions. "
            f"Available: {', '.join(DYE_REGISTRY)}. "
            "Bit position i uses dye dyes[i %% len(dyes)]. (default: CY3)"
        ),
    )
    p.add_argument(
        "--dye-wavelengths", nargs="+", type=int, default=None,
        metavar="NM",
        help=(
            "Emission wavelengths in nm corresponding to each entry in --dyes, "
            "used for camera QE lookup. Must match --dyes in length. "
            "(default: 561 for CY3, 650 for CY5, 750 for AF750)"
        ),
    )
    p.add_argument("--output-dir", required=True,
                   help="Directory to write output zarr arrays")
    p.add_argument(
        "--tag", default="sim",
        help=(
            "Middle tag inserted into hyb folder names: H1_<tag>_set<N>. "
            "Must not contain underscores. (default: sim)"
        ),
    )
    p.add_argument(
        "--set-num", type=int, default=1,
        metavar="N",
        help="Set number appended to hyb folder names (default: 1)",
    )
    p.add_argument(
        "--block-shape", nargs=3, type=int, default=[40, 700, 700],
        metavar=("Z", "Y", "X"),
        help="Dask block shape for PSF convolution (default: 40 700 700)",
    )
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of dask workers for PSF convolution (default: 8)")
    p.add_argument(
        "--no-dapi", action="store_true", default=False,
        help=(
            "Omit the fake DAPI channel.  By default a nuclear mask derived "
            "from cells.csv is prepended as channel 0 of every imaging round."
        ),
    )
    # Autofluorescence
    p.add_argument(
        "--autofluorescence-rate", type=float, default=0.0,
        metavar="PHOTONS_PER_VOXEL_PER_S",
        help=(
            "Mean autofluorescence photon rate per interior voxel per second. "
            "Applied uniformly to all dye channels. 0 disables (default: 0)"
        ),
    )
    p.add_argument(
        "--af-cv", type=float, default=0.3,
        metavar="CV",
        help="Cell-to-cell coefficient of variation for autofluorescence intensity (default: 0.3)",
    )
    p.add_argument(
        "--af-smooth-um", type=float, default=2.0,
        metavar="UM",
        help="Spatial length scale of within-cell autofluorescence texture in µm (default: 2.0)",
    )
    p.add_argument(
        "--round", type=int, default=None,
        metavar="N",
        help="Render only imaging round N (1-based). Omit to render all rounds.",
    )
    # Camera configuration (file-based takes priority over defaults)
    p.add_argument(
        "--camera-spec-file", default=None, metavar="JSON",
        help=(
            "Path to a camera spec JSON file containing conversion_factor, "
            "dark_offset, dark_current, read_noise, and full_well_capacity "
            "keyed by readout mode. Requires --camera-mode."
        ),
    )
    p.add_argument(
        "--camera-qe-file", default=None, metavar="CSV",
        help=(
            "Path to a two-column CSV (Wavelength, QE) for the camera's "
            "quantum efficiency curve. Requires --camera-mode."
        ),
    )
    p.add_argument(
        "--camera-mode", default=None, metavar="MODE",
        help=(
            "Readout mode name to select from --camera-spec-file and "
            "--camera-qe-file (e.g. 'dynamic_range', 'sensitivity')."
        ),
    )
    return p.parse_args()


# Default wavelength per named dye, used when --dye-wavelengths is not given.
_DEFAULT_WAVELENGTHS = {"CY3": 561, "CY5": 650, "AF750": 750}


def _resolve_dyes(dye_names: list, wavelengths: list | None) -> list[tuple]:
    """Return a list of (DyeSimulator, wavelength_nm) pairs.

    Args:
        dye_names: list of strings matching DYE_REGISTRY keys
        wavelengths: list of ints, or None to use defaults

    Returns:
        list of (DyeSimulator, int) tuples
    """
    unknown = [d for d in dye_names if d not in DYE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown dye(s): {unknown}. Available: {list(DYE_REGISTRY)}"
        )
    if wavelengths is not None and len(wavelengths) != len(dye_names):
        raise ValueError(
            f"--dye-wavelengths has {len(wavelengths)} entries but --dyes has {len(dye_names)}"
        )
    return [
        (DYE_REGISTRY[name], (wavelengths[i] if wavelengths else _DEFAULT_WAVELENGTHS.get(name, 561)))
        for i, name in enumerate(dye_names)
    ]


def load_psf(psf_path: Path, channel: int, y_px: int, x_px: int) -> np.ndarray:
    psf_obj = np.load(psf_path, allow_pickle=True)
    key = (channel, np.int64(y_px), np.int64(x_px))
    return psf_obj[key].astype("float32")


def prepare_psf_fft(psf: np.ndarray):
    """Apodize, normalise, zero-pad to FFT-friendly size, and pre-compute FFT."""
    psf_windowed = sim3d.apodize_psf_tukey(psf)
    psf_norm = psf_windowed / psf_windowed.sum()
    good_shape = tuple(next_fast_len(s * 2) for s in psf_norm.shape)
    pad_offsets = [(g - s) // 2 for g, s in zip(good_shape, psf_norm.shape)]
    psf_crop_slices = tuple(slice(p, p + s) for p, s in zip(pad_offsets, psf_norm.shape))
    psf_padded = np.zeros(good_shape, dtype=np.float32)
    psf_padded[psf_crop_slices] = psf_norm
    psf_fft = np.fft.fftn(psf_padded, s=good_shape)
    fz = np.fft.fftfreq(good_shape[0])
    fr = np.fft.fftfreq(good_shape[1])
    fc = np.fft.fftfreq(good_shape[2])
    FZ, FR, FC = np.meshgrid(fz, fr, fc, indexing="ij")
    return psf_fft, psf_padded, psf_crop_slices, FZ, FR, FC


def _make_xml(
    stage_x: float,
    stage_y: float,
    n_frames: int,
    z_start: float,
    z_stop: float,
    z_step: float,
    n_channels: int,
) -> str:
    """Generate a minimal MERMAKE-compatible acquisition XML string.

    The format mirrors the real microscope XMLs so that MERMAKE can parse
    stage_position (used for tile registration) and z_offsets (used to
    determine the number of channels and z-slice structure).

    z_offsets format: start:stop:step:n_channels
        start/stop/step follow Python range convention (stop is exclusive).
        E.g. 0:-16:-0.4:4 → 40 z-planes at 0.4 µm steps, 4 channels.
    """
    z_offsets = f"{z_start}:{z_stop}:{z_step}:{n_channels}"
    return f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<settings>
  <acquisition validate="True">
    <number_frames type="int">{n_frames}</number_frames>
    <stage_position type="custom">{stage_x:.2f},{stage_y:.2f}</stage_position>
  </acquisition>
  <focuslock validate="True">
    <hardware_z_scan validate="True">
      <z_offsets type="string">{z_offsets}</z_offsets>
    </hardware_z_scan>
  </focuslock>
</settings>
"""


def _find_groundtruth_paths(scene_dir: Path) -> list:
    paths = sorted(scene_dir.glob("generated_data_merfish_*/tile_*/groundtruth.csv"))
    if not paths:
        paths = sorted(scene_dir.glob("*/groundtruth.csv"))
    return paths


def _filter_channel_df(df: pd.DataFrame, bit_positions: list[int]) -> pd.DataFrame:
    """Return rows whose observed_barcode has '1' at any position in bit_positions."""
    if not bit_positions or df.empty:
        return df.iloc[:0]
    barcodes = df["observed_barcode"]
    bc_arr = np.frombuffer(
        "".join(barcodes.values).encode("ascii"), dtype=np.uint8
    ).reshape(len(barcodes), -1)
    on_mask = np.any(bc_arr[:, bit_positions] == ord("1"), axis=1)
    return df[on_mask]


def _render_channel_photons(
    df: pd.DataFrame,
    bit_positions: list[int],
    FZ, FR, FC,
    psf_fft, psf_crop_slices, psf,
    volume_shape: list,
    block_shape: tuple,
    dye: DyeSimulator,
    exposure_s: float,
    num_workers: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Render one dye channel, returning the pre-noise float32 photon image.

    No autofluorescence and no camera simulation are applied; those are
    handled by the caller so that photon images from successive density levels
    can be accumulated before adding noise.
    """
    import time

    channel_df = _filter_channel_df(df, bit_positions)

    if channel_df.empty:
        return np.zeros(volume_shape, dtype=np.float32)

    tile_im_dask = sim3d.build_tile_point_im(
        channel_df, FZ, FR, FC, psf_fft, psf_crop_slices, psf,
        volume_shape=volume_shape,
        block_shape=block_shape,
    )
    photon_im_dask = dye.psf_to_photon_distribution(tile_im_dask.clip(min=0), exposure_s)
    t0 = time.perf_counter()
    if show_progress:
        with ProgressBar():
            photon_im = photon_im_dask.compute(num_workers=num_workers)
    else:
        photon_im = photon_im_dask.compute(num_workers=num_workers)
    print(f"  compute() wall time: {time.perf_counter() - t0:.1f}s  "
          f"({len(channel_df)} spots, {num_workers} workers)")
    return photon_im


def _render_channel(
    df: pd.DataFrame,
    bit_positions: list[int],
    FZ, FR, FC,
    psf_fft, psf_crop_slices, psf,
    volume_shape: list,
    block_shape: tuple,
    dye: DyeSimulator,
    wavelength: int,
    camera: CameraSimulator,
    exposure_s: float,
    num_workers: int,
    autofluorescence_im: np.ndarray | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Render one dye channel for a given set of bit positions.

    Spots are included if any of their mapped_barcode bits at *bit_positions*
    is '1'. Returns a uint16 array of shape (n_z, n_y, n_x).
    """
    photon_im = _render_channel_photons(
        df, bit_positions, FZ, FR, FC, psf_fft, psf_crop_slices, psf,
        volume_shape, block_shape, dye, exposure_s, num_workers, show_progress,
    )
    if autofluorescence_im is not None:
        photon_im = photon_im + autofluorescence_im
    return camera.simulate_image(photon_im, wavelength, exposure_s)


def _make_dapi_channel(
    cells_df: pd.DataFrame,
    volume_shape: list,
    pixel_size: list,
    nucleus_scale: float = 0.5,
    intensity: int = 2000,
) -> np.ndarray:
    """Rasterise cell nuclear masks into a fake DAPI channel.

    Each cell ellipsoid from *cells_df* is shrunk by *nucleus_scale* along all
    three axes and voxelised into a binary mask at *intensity* counts.

    Cell physical coordinates follow the (x, y, z) convention used throughout
    the pipeline (center_x / center_y / center_z columns in cells.csv).
    Voxel axes are (iz, iy, ix) = (z/vz, y/vy, x/vx).

    Returns a uint16 array of shape (n_z, n_y, n_x).
    """
    nz, ny, nx = volume_shape
    vz, vy, vx = pixel_size
    dapi = np.zeros((nz, ny, nx), dtype=np.float32)

    for _, row in cells_df.iterrows():
        cx, cy, cz = row["center_x"], row["center_y"], row["center_z"]
        a   = row["axis_a"] * nucleus_scale
        b   = row["axis_b"] * nucleus_scale
        c_ax = row["axis_c"] * nucleus_scale
        R_mat = np.array(
            [[row[f"R_{ri}{ci}"] for ci in range(3)] for ri in range(3)]
        )

        # Bounding box in voxel space; add 2-voxel margin for rounding safety
        max_r = max(a, b, c_ax)
        ix_lo = max(0,  int((cx - max_r) / vx))
        ix_hi = min(nx, int((cx + max_r) / vx) + 2)
        iy_lo = max(0,  int((cy - max_r) / vy))
        iy_hi = min(ny, int((cy + max_r) / vy) + 2)
        iz_lo = max(0,  int((cz - max_r) / vz))
        iz_hi = min(nz, int((cz + max_r) / vz) + 2)

        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            continue

        # Physical displacement from cell centre for every voxel in the bbox
        IZ, IY, IX = np.mgrid[iz_lo:iz_hi, iy_lo:iy_hi, ix_lo:ix_hi]
        dx = IX * vx - cx
        dy = IY * vy - cy
        dz = IZ * vz - cz

        # Rotate into ellipsoid frame: p_local = R^T @ (p - centre)
        pts = np.stack([dx.ravel(), dy.ravel(), dz.ravel()])  # (3, N)
        local = R_mat.T @ pts                                  # (3, N)

        inside = (
            local[0] ** 2 / a ** 2
            + local[1] ** 2 / b ** 2
            + local[2] ** 2 / c_ax ** 2
        ) <= 1.0
        inside = inside.reshape(IZ.shape)

        dapi[iz_lo:iz_hi, iy_lo:iy_hi, ix_lo:ix_hi] = np.where(
            inside,
            intensity,
            dapi[iz_lo:iz_hi, iy_lo:iy_hi, ix_lo:ix_hi],
        )

    return dapi.clip(0, 65535).astype(np.uint16)


def _write_render_params(
    output_dir: Path,
    args,
    camera: CameraSimulator,
    dye_channels: list,
    exposure_s: float,
    volume_shape: list,
    n_bits: int,
    n_rounds: int,
) -> None:
    """Write a JSON record of all parameters used in this render run."""
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "derived": {
            "exposure_s": exposure_s,
            "volume_shape_voxels": volume_shape,
            "n_bits": n_bits,
            "n_rounds": n_rounds,
            "dye_channels": [
                {"name": name, "wavelength_nm": wl}
                for name, (_, wl) in zip(args.dyes, dye_channels)
            ],
        },
        "camera": {
            "gain": camera.gain,
            "bias": camera.bias,
            "dark_current": camera.dark_current,
            "read_noise": camera.read_noise,
            "well_depth": camera.well_depth,
        },
    }

    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    out_path = output_dir / "render_params.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=_default)
    print(f"Run parameters saved to {out_path}")


def main():
    args = parse_args()

    try:
        import zarr
    except ImportError:
        raise ImportError("zarr is required for output. Install with: pip install zarr")

    dye_channels = _resolve_dyes(args.dyes, args.dye_wavelengths)
    n_dyes = len(dye_channels)
    print(f"Dye channels ({n_dyes}): " +
          ", ".join(f"{name} @ {wl}nm" for name, (_, wl) in zip(args.dyes, dye_channels)))

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "scene_meta.json") as f:
        meta = json.load(f)

    volume_um = meta["volume_um"]
    pixel_size = args.pixel_size
    volume_shape = [
        int(np.ceil(volume_um[0] / pixel_size[0])),
        int(np.ceil(volume_um[1] / pixel_size[1])),
        int(np.ceil(volume_um[2] / pixel_size[2])),
    ]
    n_z = volume_shape[0]
    print(f"Volume shape (voxels): {volume_shape}")

    n_bits = meta["n_bits"]
    n_rounds = (n_bits + n_dyes - 1) // n_dyes  # ceiling division

    if n_bits % n_dyes != 0:
        import warnings
        warnings.warn(
            f"{n_bits} bits is not evenly divisible by {n_dyes} dyes "
            f"({n_bits % n_dyes} bit(s) left over). "
            f"The last imaging round will have fewer active channels. "
            f"Check that --dyes matches your experimental design.",
            stacklevel=2,
        )

    print(f"Loading PSF from {args.psf} at field position {args.psf_position}...")
    psf = load_psf(Path(args.psf), args.psf_channel, args.psf_position[0], args.psf_position[1])
    psf_fft, _, psf_crop_slices, FZ, FR, FC = prepare_psf_fft(psf)

    exposure_s = args.exposure_ms * k.milli * args.brightness_scale

    # Build a shared camera.  File-based config takes priority; scalar defaults
    # are used as fallback when no spec/QE file is supplied.
    use_files = args.camera_spec_file and args.camera_mode
    if use_files:
        camera = CameraSimulator(
            spec_file=Path(args.camera_spec_file),
            qe_file=Path(args.camera_qe_file) if args.camera_qe_file else None,
            mode=args.camera_mode,
        )
    else:
        qe_map = {wl: 0.85 for _, wl in dye_channels}  # placeholder QE
        camera = CameraSimulator(
            QE=qe_map,
            gain=0.25,
            bias=100,
            dark_current=1.0,
            read_noise=1.8,
            well_depth=15000,
        )

    _write_render_params(
        output_dir, args, camera, dye_channels,
        exposure_s, volume_shape, n_bits, n_rounds,
    )

    gt_paths = _find_groundtruth_paths(scene_dir)
    if not gt_paths:
        raise FileNotFoundError(f"No groundtruth CSV files found under {scene_dir}")

    # Generate fake DAPI channel from cell geometry (same for all tiles/rounds).
    cells_csv = scene_dir / "cells.csv"
    cells_df = pd.read_csv(cells_csv) if cells_csv.exists() else None

    dapi_im = None
    if not args.no_dapi:
        if cells_df is not None:
            print(f"Generating fake DAPI channel from {len(cells_df)} cells...")
            dapi_im = _make_dapi_channel(cells_df, volume_shape, pixel_size)
        else:
            print(f"Warning: cells.csv not found at {cells_csv}; skipping DAPI channel.")

    # Autofluorescence — computed once, reused for every round and channel.
    autofluorescence_im = None
    if args.autofluorescence_rate > 0:
        if cells_df is not None:
            print(
                f"Generating autofluorescence (rate={args.autofluorescence_rate} ph/vox/s, "
                f"CV={args.af_cv}, smooth={args.af_smooth_um} µm)..."
            )
            autofluorescence_im = simulate_autofluorescence(
                cells_df,
                volume_shape=volume_shape,
                pixel_size=pixel_size,
                mean_photons=args.autofluorescence_rate * exposure_s,
                cv=args.af_cv,
                smooth_scale_um=args.af_smooth_um,
            )
        else:
            print("Warning: cells.csv not found; skipping autofluorescence.")

    n_channels_total = n_dyes + (1 if dapi_im is not None else 0)

    for tile_idx, gt_path in enumerate(gt_paths):
        print(f"\nTile {tile_idx + 1}/{len(gt_paths)}: {gt_path}")
        df = pd.read_csv(gt_path, dtype={"barcode": str, "observed_barcode": str})

        df[["frame", "row", "column", "frame_shift", "row_shift", "column_shift"]] = (
            sim3d.compute_pixel_locations(
                df[["z", "y", "x"]].to_numpy(), pixel_size=pixel_size
            )
        )

        rounds_to_render = (
            [args.round - 1] if args.round is not None else range(n_rounds)
        )
        if args.round is not None:
            if args.round < 1 or args.round > n_rounds:
                raise ValueError(
                    f"--round {args.round} is out of range (1–{n_rounds})"
                )

        for round_num in rounds_to_render:
            # Bit positions (0-based) for this round, grouped by dye index
            round_start = round_num * n_dyes
            bits_per_dye = [
                [round_start + dye_idx]
                if round_start + dye_idx < n_bits else []
                for dye_idx in range(n_dyes)
            ]

            hyb_folder = output_dir / f"H{round_num + 1}_{args.tag}_set{args.set_num}"
            hyb_folder.mkdir(parents=True, exist_ok=True)

            channel_images = []
            for dye_idx, (dye, wavelength) in enumerate(dye_channels):
                bit_positions = bits_per_dye[dye_idx]
                print(f"  Round {round_num + 1}, {args.dyes[dye_idx]} "
                      f"(bits {bit_positions}): rendering...")
                img = _render_channel(
                    df, bit_positions,
                    FZ, FR, FC, psf_fft, psf_crop_slices, psf,
                    volume_shape, tuple(args.block_shape),
                    dye, wavelength, camera, exposure_s, args.num_workers,
                    autofluorescence_im=autofluorescence_im,
                )
                channel_images.append(img)  # each is (n_z, n_y, n_x)

            # Stack into (n_z * n_channels_total, n_y, n_x), channels cycling within each z-plane.
            # DAPI is the last channel when present:
            #   [ch0_z0, ch1_z0, dapi_z0, ch0_z1, ch1_z1, dapi_z1, ...]
            # A single blank throwaway frame is prepended at index 0.
            all_channels = channel_images + ([dapi_im] if dapi_im is not None else [])
            stacked = np.stack(all_channels, axis=1)   # (n_z, n_channels_total, n_y, n_x)
            n_y, n_x = stacked.shape[2], stacked.shape[3]
            stacked = stacked.reshape(n_z * n_channels_total, n_y, n_x)
            throwaway = np.zeros((1, n_y, n_x), dtype=np.uint16)
            stacked = np.concatenate([throwaway, stacked], axis=0)

            fov_str = f"{tile_idx:03d}"  # zero-based to match real data (000, 001, ...)

            # 1. FOV zarr group + image data array inside it.
            # The FOV directory must be a zarr group (.zgroup) with the image
            # array at <fov>/data/ (.zarray + chunks).  zarr_format=2 forces
            # the v2 layout (dot-separated chunk names, no c/ subdirectory).
            fov_path = hyb_folder / fov_str
            zarr.open_group(str(fov_path), mode="w", zarr_format=2)
            z_arr = zarr.open_array(
                str(fov_path / "data"), mode="w",
                shape=stacked.shape, dtype="uint16",
                chunks=(1, *stacked.shape[1:]),
                zarr_format=2,
            )
            z_arr[:] = stacked

            # 2. Detection marker: empty <hyb_folder>/Conv_zscan__<fov>.zarr/
            marker_dir = hyb_folder / f"Conv_zscan__{fov_str}.zarr"
            marker_dir.mkdir(exist_ok=True)

            # 3. Companion XML with stage position and z_offsets
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
            xml_path = hyb_folder / f"Conv_zscan__{fov_str}.xml"
            xml_path.write_text(xml, encoding="ISO-8859-1")

            print(f"  Saved {fov_path / 'data'}  shape={stacked.shape}  "
                  f"[1 throwaway + {n_z} z-planes x {n_channels_total} channels]")

    print(f"\nRendering complete. Output in {output_dir}")
    print(f"Hyb folder pattern: H{{round}}_{args.tag}_set{args.set_num}/"
          f"  FOV data at {{fov}}/data/")


if __name__ == "__main__":
    main()
