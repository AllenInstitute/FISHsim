"""
Load simulated zarr images into a napari viewer, one layer per channel.

The output directory must contain render_params.json (written by render_images.py)
so that channel names and z-dimension can be resolved automatically.

Array layout written by render_images.py:
    shape  = (1 + n_z * n_channels, n_y, n_x)  dtype=uint16
    frame 0              = throwaway (zeros)
    frames 1 .. end      = channels cycling within each z-plane
                           [ch0_z0, ch1_z0, ..., chN_z0,
                            ch0_z1, ch1_z1, ..., chN_z1, ...]

Usage (script):
    python -m fishsim.scripts.view_zarr --output-dir results/images_01
    python -m fishsim.scripts.view_zarr --output-dir results/images_01 \\
        --hyb H1 --fov 000

Usage (programmatic):
    from fishsim.scripts.view_zarr import load_fov, open_in_napari
    channels = load_fov("results/images_01/H1_sim_set1/000/data", n_z=40,
                        channel_names=["CY3", "CY5", "AF750", "DAPI"])
    open_in_napari(channels, scale=[0.4, 0.1084333, 0.1084333])
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def _channel_names_from_params(params: dict) -> list[str]:
    """Derive ordered channel names from render_params.json.

    Duplicate dye names (e.g. three CY3 channels) are disambiguated by
    appending a zero-based index: CY3_0, CY3_1, CY3_2.
    """
    raw = [d["name"] for d in params["derived"]["dye_channels"]]
    no_dapi = params["args"].get("no_dapi", False)
    if not no_dapi:
        raw = raw + ["DAPI"]

    counts = {}
    for name in raw:
        counts[name] = counts.get(name, 0) + 1

    seen = {}
    result = []
    for name in raw:
        if counts[name] > 1:
            idx = seen.get(name, 0)
            result.append(f"{name}_{idx}")
            seen[name] = idx + 1
        else:
            result.append(name)
    return result


def load_fov(
    fov_data_path: str | Path,
    n_z: int,
    channel_names: list[str],
) -> dict[str, np.ndarray]:
    """Split a single FOV zarr array into per-channel volumes.

    Args:
        fov_data_path: path to the ``<fov>/data`` zarr array directory.
        n_z: number of z-planes (from render_params.json or known a priori).
        channel_names: ordered list of channel labels, e.g. ["CY3", "CY5", "DAPI"].

    Returns:
        Dict mapping each channel name to a uint16 array of shape (n_z, n_y, n_x).
    """
    fov_data_path = Path(fov_data_path)
    arr = zarr.open_array(str(fov_data_path), mode="r")

    n_channels = len(channel_names)
    expected_frames = 1 + n_z * n_channels
    if arr.shape[0] != expected_frames:
        raise ValueError(
            f"Zarr has {arr.shape[0]} frames but expected {expected_frames} "
            f"(1 throwaway + {n_z} z × {n_channels} channels). "
            "Check n_z and channel_names."
        )

    # Drop throwaway frame, reshape to (n_z, n_channels, n_y, n_x)
    data = arr[1:]
    n_y, n_x = data.shape[1], data.shape[2]
    data = np.asarray(data).reshape(n_z, n_channels, n_y, n_x)

    return {name: data[:, i, :, :] for i, name in enumerate(channel_names)}


def open_in_napari(
    channels: dict[str, np.ndarray],
    scale: list[float] | None = None,
    viewer=None,
):
    """Add channel volumes to a napari viewer.

    Args:
        channels: dict from load_fov(), channel_name -> (n_z, n_y, n_x) array.
        scale: voxel size in µm as [z, y, x].  Defaults to [0.4, 0.1084333, 0.1084333].
        viewer: existing napari.Viewer to add layers to; creates a new one if None.

    Returns:
        The napari.Viewer instance.
    """
    import napari  # imported here so the rest of the module works without napari

    if scale is None:
        scale = [0.4, 0.1084333, 0.1084333]

    _COLORMAPS = {
        "CY3": "yellow",
        "CY5": "red",
        "AF750": "magenta",
        "DAPI": "blue",
    }

    if viewer is None:
        viewer = napari.Viewer(ndisplay=2)

    for name, vol in channels.items():
        cmap = _COLORMAPS.get(name, "gray")
        viewer.add_image(
            vol,
            name=name,
            scale=scale,
            colormap=cmap,
            blending="additive",
            contrast_limits=(np.percentile(vol, 0.5), np.percentile(vol, 99.9)),
        )

    return viewer


def _find_hyb_folders(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("H"))


def _find_fov_paths(hyb_folder: Path) -> list[Path]:
    """Return sorted list of <fov>/data zarr paths inside a hyb folder."""
    return sorted(
        p / "data"
        for p in hyb_folder.iterdir()
        if p.is_dir() and (p / "data").exists() and not p.name.startswith("Conv_")
    )


def view_output(
    output_dir: str | Path,
    hyb: str | None = None,
    fov: str | None = None,
    scale: list[float] | None = None,
):
    """Load a FOV from a render_images output directory and open it in napari.

    Args:
        output_dir: top-level directory written by render_images.py.
        hyb: hyb folder name or prefix (e.g. "H1").  Defaults to the first found.
        fov: FOV folder name (e.g. "000").  Defaults to the first found.
        scale: voxel size [z, y, x] in µm; read from render_params.json if None.

    Returns:
        (napari.Viewer, dict of channel arrays)
    """
    output_dir = Path(output_dir)

    params_path = output_dir / "render_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"render_params.json not found in {output_dir}. "
            "Pass n_z and channel_names explicitly via load_fov()."
        )
    with open(params_path) as f:
        params = json.load(f)

    n_z = params["derived"]["volume_shape_voxels"][0]
    channel_names = _channel_names_from_params(params)
    if scale is None:
        scale = params["args"].get("pixel_size", [0.4, 0.1084333, 0.1084333])

    hyb_folders = _find_hyb_folders(output_dir)
    if not hyb_folders:
        raise FileNotFoundError(f"No hyb folders found in {output_dir}")

    if hyb is not None:
        hyb_folders = [h for h in hyb_folders if h.name.startswith(hyb)]
        if not hyb_folders:
            raise FileNotFoundError(f"No hyb folder matching '{hyb}' in {output_dir}")
    hyb_folder = hyb_folders[0]

    fov_paths = _find_fov_paths(hyb_folder)
    if not fov_paths:
        raise FileNotFoundError(f"No FOV data arrays found in {hyb_folder}")

    if fov is not None:
        fov_paths = [p for p in fov_paths if p.parent.name == fov]
        if not fov_paths:
            raise FileNotFoundError(f"FOV '{fov}' not found in {hyb_folder}")
    fov_data_path = fov_paths[0]

    print(f"Loading  {fov_data_path}")
    print(f"  n_z={n_z}  channels={channel_names}  scale={scale}")

    channels = load_fov(fov_data_path, n_z=n_z, channel_names=channel_names)
    viewer = open_in_napari(channels, scale=scale)
    return viewer, channels


def parse_args():
    p = argparse.ArgumentParser(description="View simulated zarr images in napari")
    p.add_argument("--output-dir", required=True,
                   help="Top-level output directory written by render_images.py")
    p.add_argument("--hyb", default=None,
                   help="Hyb folder prefix to load (e.g. 'H1'). Defaults to first found.")
    p.add_argument("--fov", default=None,
                   help="FOV name to load (e.g. '000'). Defaults to first found.")
    p.add_argument("--scale", nargs=3, type=float, default=None,
                   metavar=("Z_UM", "Y_UM", "X_UM"),
                   help="Voxel size in µm [z y x]. Read from render_params.json if omitted.")
    return p.parse_args()


if __name__ == "__main__":
    import napari

    args = parse_args()
    viewer, _ = view_output(
        args.output_dir,
        hyb=args.hyb,
        fov=args.fov,
        scale=args.scale,
    )
    napari.run()
