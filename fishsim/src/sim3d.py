"""
3D merFISH scene generation and PSF convolution helpers.

This module provides:
  - Simulator: places cells and fluorescent emitters in a 3D volume and
    assigns gene barcodes, producing a ground-truth point cloud.
  - PSF convolution utilities (block-decomposed, dask-backed) for rendering
    the point cloud into a photon image.

Camera noise and dye photophysics live in fishsim.src.imaging.
"""

import random
import time

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal.windows import tukey
from scipy.spatial import ConvexHull, Delaunay

import pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(60)

import dask.array as da
from dask import delayed

from tqdm.auto import tqdm

from fishsim.src.ellipsoid import Ellipsoid
from fishsim.src.generate_emitters import cell_emitter_position, random_emitter_position
from fishsim.config import RESULTS_DIR


# ---------------------------------------------------------------------------
# Scene / point-cloud generation
# ---------------------------------------------------------------------------

class Simulator:
    """Place cells and fluorescent emitters in a 3D volume and assign gene barcodes.

    Separates point-cloud generation from image rendering so that a single
    scene can be rendered multiple times with different optical parameters.

    Args:
        emitters_per_cell: number of emitters placed inside each cell
        cell_count: number of non-overlapping ellipsoid cells to generate;
            use fishsim.scripts.generate_scene._estimate_cell_count() to
            derive this from volume and cell size
        cell_axes: dict with keys "a", "b", "c", each a [min, max] list
            specifying the half-axis bounds in µm
        bit_drop: probability that a '1' bit in a barcode is silenced (0–1)
        bit_add: probability that a '0' bit is spuriously lit (0–1)
    """

    def __init__(
        self,
        emitters_per_cell: int = 500,
        cell_count: int = 10,
        cell_axes: dict = None,
        bit_drop: float = 0.0,
        bit_add: float = 0.0,
    ):
        self.emitters_per_cell = emitters_per_cell
        self.cell_count = cell_count
        self.cell_axes = cell_axes or {"a": [10, 15], "b": [10, 15], "c": [10, 15]}
        self.bit_drop = bit_drop
        self.bit_add = bit_add
        self.cells = []

    def generate_point_cloud(
        self,
        codebook_filepath: Path,
        tiles: int = 1,
        is_subpixel: bool = True,
        is_cell: bool = True,
        is_nucleus: bool = False,
        sample_volume: tuple = (16.0, 303.6, 303.6),
        savepath: Path = None,
    ) -> list:
        """Generate a ground-truth point cloud for one or more tiles.

        Args:
            codebook_filepath: path to codebook CSV
            tiles: number of tiles to generate
            is_subpixel: if True, emitter positions are floating-point (µm)
            is_cell: if True, emitters are placed inside ellipsoid cells
            is_nucleus: if True, cells contain a lower-density nucleus region
            sample_volume: (z, y, x) imaging volume in µm
            savepath: root directory for output; defaults to RESULTS_DIR

        Returns:
            list of Path objects pointing to each tile's groundtruth.csv
        """
        codebook = pd.read_csv(codebook_filepath, skiprows=3)
        codebook = codebook[~codebook["name"].str.contains("blank", case=False, na=False)].reset_index(drop=True)

        t = time.strftime("%Y%m%d-%H-%M-%S")
        folder_name = "generated_data_merfish_" + t
        if savepath is None:
            folder_path = Path(RESULTS_DIR) / folder_name
        else:
            folder_path = savepath / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        results = []
        for i in tqdm(range(tiles), desc="Generating tiles"):
            gt_path = self._generate_single_tile(
                codebook, i, is_subpixel, is_cell, is_nucleus,
                sample_volume, folder_path,
            )
            results.append(gt_path)
        return results

    def _generate_single_tile(
        self,
        codebook: pd.DataFrame,
        tile_num: int,
        is_subpixel: bool,
        is_cell: bool,
        is_nucleus: bool,
        sample_volume: tuple,
        folder_path: Path,
    ) -> Path:
        z_um, y_um, x_um = sample_volume

        genes = codebook["name"]
        gene_ids = codebook["numeric_id"]
        barcodes = codebook["barcode"]

        # Determine per-emitter gene assignments
        if "distribution" in codebook.columns:
            distribution = codebook["distribution"].fillna(0).astype(int).tolist()
            r = [i for i, count in enumerate(distribution) for _ in range(count)]
            random.shuffle(r)
            num_emitter = len(r)
        else:
            num_emitter = self.emitters_per_cell * self.cell_count
            r = np.random.randint(0, len(barcodes), size=num_emitter).tolist()

        # Place emitters
        if is_cell:
            emitter_pos, self.cells = cell_emitter_position(
                x_dim=(0, x_um - 1),
                y_dim=(0, y_um - 1),
                z_dim=(0, z_um - 1),
                num_emitter=self.emitters_per_cell,
                cell_count=self.cell_count,
                cell_axes_bounds=self.cell_axes,
                is_nucleus=is_nucleus,
                is_physical_coordinates=True,
                emitter_per_cell=True,
            )
            cell_emitters, cell_ids = [], []
            for cell_id, cell in enumerate(self.cells):
                for emitter in cell.emitters:
                    cell_emitters.append(list(emitter))
                    cell_ids.append(cell_id)
            emitter_pos = np.array(cell_emitters)
            cell_ids = np.array(cell_ids)
            num_emitter = len(emitter_pos)
            r = np.random.randint(0, len(barcodes), size=num_emitter).tolist()
        else:
            emitter_pos = random_emitter_position(
                x_dim=(0, x_um - 1),
                y_dim=(0, y_um - 1),
                z_dim=(0, z_um - 1),
                num_emitter=num_emitter,
            )
            cell_ids = np.full(num_emitter, -1)

        if not is_subpixel:
            emitter_pos = np.floor(emitter_pos)

        # Barcode assignment and bit errors
        is_bit_drop = np.random.rand(num_emitter) <= self.bit_drop
        is_bit_add = np.random.rand(num_emitter) <= self.bit_add

        rows = []
        for i in tqdm(range(num_emitter), leave=False, desc="Assigning barcodes"):
            barcode = barcodes.iloc[r[i]].replace(" ", "")
            observed = list(barcode)

            if is_bit_drop[i]:
                ones = [k for k, b in enumerate(observed) if b == "1"]
                if ones:
                    observed[random.choice(ones)] = "0"
            if is_bit_add[i]:
                zeros = [k for k, b in enumerate(observed) if b == "0"]
                if zeros:
                    observed[random.choice(zeros)] = "1"

            rows.append({
                "x": emitter_pos[i, 0],
                "y": emitter_pos[i, 1],
                "z": emitter_pos[i, 2],
                "genes": genes.iloc[r[i]],
                "gene_id": gene_ids.iloc[r[i]],
                "barcode": barcode,
                "observed_barcode": "".join(observed),
                "is_bit_drop": bool(is_bit_drop[i]),
                "is_bit_add": bool(is_bit_add[i]),
                "cell_id": int(cell_ids[i]),
            })

        # Save outputs
        tile_path = folder_path / f"tile_{tile_num + 1}"
        tile_path.mkdir(parents=True, exist_ok=True)

        gt = pd.DataFrame(rows)
        gt_path = tile_path / "groundtruth.csv"
        gt.to_csv(gt_path, index=False)

        gt.value_counts("genes").rename_axis("genes").to_frame("counts").to_csv(
            tile_path / "frequency.csv"
        )
        return gt_path


# ---------------------------------------------------------------------------
# PSF convolution helpers
# ---------------------------------------------------------------------------

def apodize_psf_tukey(psf: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Apply a separable Tukey window to suppress PSF edge ringing.

    Args:
        alpha: 0 = rectangular (no apodization), 1 = Hann window.
            0.1–0.2 gives a flat centre with a narrow cosine rolloff.
    """
    windows = [tukey(s, alpha=alpha) for s in psf.shape]
    Wz, Wr, Wc = np.meshgrid(*windows, indexing="ij")
    return psf * Wz * Wr * Wc


def compute_pixel_locations(
    emitter_pos: np.ndarray, pixel_size: list
) -> np.ndarray:
    """Convert physical (µm) emitter positions to pixel indices and subpixel shifts.

    Args:
        emitter_pos: (N, 3) array with columns (z, y, x) in µm
        pixel_size: [z_um, y_um, x_um] voxel size in µm

    Returns:
        (N, 6) array: [frame, row, column, frame_shift, row_shift, column_shift]

    Example::

        df[['frame','row','column','frame_shift','row_shift','column_shift']] = (
            compute_pixel_locations(df[['z','y','x']].to_numpy(),
                                    pixel_size=[0.4, 0.1084333, 0.1084333])
        )
    """
    pixel_size = np.array(pixel_size)
    indices = np.floor(emitter_pos / pixel_size).astype(int)
    shifts = (emitter_pos / pixel_size) % 1
    return np.concatenate([indices, shifts], axis=1)


def filter_spotdf_on_round(spot_df: pd.DataFrame, round_num: int) -> pd.DataFrame:
    """Return rows whose mapped_barcode has a '1' at position *round_num*.

    *round_num* is a direct barcode position (0-based), cycling through all
    channels within a round before moving to the next round.
    """
    on = spot_df["mapped_barcode"].str.strip("'").str.get(round_num) == "1"
    return spot_df.loc[on].copy()


def _points_overlapping_block(
    points_arrays: dict, block_slices: tuple, psf_shape: tuple
) -> dict:
    margin = [s // 2 for s in psf_shape]
    mask = (
        (points_arrays["frame"] >= block_slices[0].start - margin[0])
        & (points_arrays["frame"] < block_slices[0].stop + margin[0])
        & (points_arrays["row"] >= block_slices[1].start - margin[1])
        & (points_arrays["row"] < block_slices[1].stop + margin[1])
        & (points_arrays["column"] >= block_slices[2].start - margin[2])
        & (points_arrays["column"] < block_slices[2].stop + margin[2])
    )
    return {k: v[mask] for k, v in points_arrays.items()}


def _compute_insertion_slices(
    row: int, col: int, z: int, psf_shape: tuple, volume_shape: tuple
) -> tuple:
    """Return (vol_slices, psf_slices) clamped to volume bounds."""
    vol_slices, psf_slices = [], []
    for center, psf_size, vol_size in zip([z, row, col], psf_shape, volume_shape):
        start = center - psf_size // 2
        stop = start + psf_size
        vol_start = max(start, 0)
        vol_stop = min(stop, vol_size)
        psf_start = vol_start - start
        psf_stop = psf_start + (vol_stop - vol_start)
        vol_slices.append(slice(vol_start, vol_stop))
        psf_slices.append(slice(psf_start, psf_stop))
    return tuple(vol_slices), tuple(psf_slices)


def _process_block(
    block_slices: tuple,
    points_arrays: dict,
    FZ: np.ndarray,
    FR: np.ndarray,
    FC: np.ndarray,
    psf_fft: np.ndarray,
    psf_crop_slices: tuple,
    psf: np.ndarray,
    volume_shape: tuple,
    batch_size: int = 32,
) -> np.ndarray:
    block_points = _points_overlapping_block(points_arrays, block_slices, psf.shape)
    n_points = len(block_points["frame"])
    block_shape = tuple(s.stop - s.start for s in block_slices)
    result = np.zeros(block_shape, dtype=np.float32)

    for i in range(0, n_points, batch_size):
        idx = slice(i, min(i + batch_size, n_points))

        phase_ramps = np.exp(
            -1j * 2 * np.pi * (
                FZ[None] * block_points["frame_shift"][idx, None, None, None]
                + FR[None] * block_points["row_shift"][idx, None, None, None]
                + FC[None] * block_points["column_shift"][idx, None, None, None]
            )
        )
        all_shifted = np.real(
            np.fft.ifftn(psf_fft[None] * phase_ramps, axes=(1, 2, 3))
        )

        for j, pt_idx in enumerate(range(i, min(i + batch_size, n_points))):
            shifted_psf = all_shifted[j][psf_crop_slices]
            vol_slices, psf_slices = _compute_insertion_slices(
                row=int(block_points["row"][pt_idx]),
                col=int(block_points["column"][pt_idx]),
                z=int(block_points["frame"][pt_idx]),
                psf_shape=psf.shape,
                volume_shape=volume_shape,
            )
            local_slices = tuple(
                slice(s.start - b.start, s.stop - b.start)
                for s, b in zip(vol_slices, block_slices)
            )
            clipped_local, clipped_psf = [], []
            for ls, ps, bs in zip(local_slices, psf_slices, block_shape):
                l_start = max(ls.start, 0)
                l_stop = min(ls.stop, bs)
                p_start = ps.start + (l_start - ls.start)
                p_stop = p_start + (l_stop - l_start)
                if l_stop <= l_start:
                    break
                clipped_local.append(slice(l_start, l_stop))
                clipped_psf.append(slice(p_start, p_stop))
            else:
                result[tuple(clipped_local)] += shifted_psf[tuple(clipped_psf)]

    return result


def build_tile_point_im(
    df: pd.DataFrame,
    FZ: np.ndarray,
    FR: np.ndarray,
    FC: np.ndarray,
    psf_fft: np.ndarray,
    psf_crop_slices: tuple,
    psf: np.ndarray,
    volume_shape: tuple,
    block_shape: tuple = (40, 700, 700),
) -> da.Array:
    """Render a point cloud into a 3D photon image using block-decomposed PSF convolution.

    Each emitter is placed by sub-pixel-shifting the PSF via a Fourier phase
    ramp, then stamped into its block. Blocks are computed lazily with dask.

    Args:
        df: ground-truth dataframe with columns frame, row, column,
            frame_shift, row_shift, column_shift (from compute_pixel_locations)
        FZ, FR, FC: pre-computed frequency grids (from np.meshgrid + fftfreq)
        psf_fft: FFT of the zero-padded, apodized PSF
        psf_crop_slices: slices to extract the PSF from the padded array
        psf: original (unpadded) PSF array, used for shape information
        volume_shape: (n_z, n_y, n_x) output volume in pixels
        block_shape: dask block size; tune based on available RAM

    Returns:
        dask array of shape *volume_shape*, dtype float32
    """
    points_arrays = {
        "frame": df["frame"].values,
        "row": df["row"].values,
        "column": df["column"].values,
        "frame_shift": df["frame_shift"].values,
        "row_shift": df["row_shift"].values,
        "column_shift": df["column_shift"].values,
    }

    blocks, dask_blocks = [], []
    for z in range(0, volume_shape[0], block_shape[0]):
        for r in range(0, volume_shape[1], block_shape[1]):
            for c in range(0, volume_shape[2], block_shape[2]):
                slices = (
                    slice(z, min(z + block_shape[0], volume_shape[0])),
                    slice(r, min(r + block_shape[1], volume_shape[1])),
                    slice(c, min(c + block_shape[2], volume_shape[2])),
                )
                blocks.append(slices)
                d = delayed(_process_block)(
                    slices, points_arrays, FZ, FR, FC,
                    psf_fft, psf_crop_slices, psf, volume_shape,
                )
                dask_blocks.append(
                    da.from_delayed(
                        d,
                        shape=tuple(s.stop - s.start for s in slices),
                        dtype=np.float32,
                    )
                )

    n_z = len(range(0, volume_shape[0], block_shape[0]))
    n_r = len(range(0, volume_shape[1], block_shape[1]))
    n_c = len(range(0, volume_shape[2], block_shape[2]))

    grid, idx = [], 0
    for _ in range(n_z):
        row_grid = []
        for _ in range(n_r):
            col_grid = [dask_blocks[idx + c] for c in range(n_c)]
            idx += n_c
            row_grid.append(col_grid)
        grid.append(row_grid)
    return da.block(grid)


# ---------------------------------------------------------------------------
# Cell background helpers
# ---------------------------------------------------------------------------

def ellipsoid_to_mask(
    ellipsoid: Ellipsoid, pixel_size: float, shape: list
) -> tuple:
    """Return a boolean voxel mask for an ellipsoid and slices into *shape*.

    Args:
        ellipsoid: Ellipsoid instance (physical coords, µm)
        pixel_size: isotropic voxel size in µm
        shape: [n_z, n_y, n_x] of the containing volume

    Returns:
        (mask, slices) where mask is a bool array over the bounding box
        and slices places it into the full volume.
    """
    axes_px = ellipsoid.axes / pixel_size
    center_px = ellipsoid.center[[2, 0, 1]] / pixel_size

    pad = int(np.ceil(np.max(axes_px))) + 1
    z_min = max(0, int(np.floor(center_px[0] - pad)))
    z_max = min(shape[0], int(np.ceil(center_px[0] + pad)))
    y_min = max(0, int(np.floor(center_px[1] - pad)))
    y_max = min(shape[1], int(np.ceil(center_px[1] + pad)))
    x_min = max(0, int(np.floor(center_px[2] - pad)))
    x_max = min(shape[2], int(np.ceil(center_px[2] + pad)))

    slices = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    z_idx, y_idx, x_idx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
    coords = np.stack([z_idx - center_px[0], y_idx - center_px[1], x_idx - center_px[2]], axis=-1)
    coords_local = coords @ ellipsoid.R
    mask = np.sum((coords_local / axes_px) ** 2, axis=-1) <= 1.0
    return mask, slices


def points_to_convex_hull_mask(
    points: np.ndarray, pixel_size, volume_shape: tuple = None
) -> tuple:
    """Boolean mask of the convex hull of a set of (z, y, x) physical points.

    Args:
        points: (N, 3) array in (z, y, x) physical coordinates (µm)
        pixel_size: scalar or (3,) array of µm/pixel for (z, y, x)
        volume_shape: (n_z, n_y, n_x) clamps the bounding box; None = infer

    Returns:
        (mask, slices) as with ellipsoid_to_mask
    """
    points = np.array(points, dtype=float)[:, [2, 0, 1]]  # → (x, y, z) then reorder
    pixel_size = np.broadcast_to(pixel_size, (3,))
    pixel_points = points / pixel_size

    min_px = np.floor(pixel_points.min(axis=0)).astype(int)
    max_px = np.ceil(pixel_points.max(axis=0)).astype(int) + 1
    if volume_shape is not None:
        min_px = np.clip(min_px, 0, np.array(volume_shape) - 1)
        max_px = np.clip(max_px, 0, np.array(volume_shape))

    mask_shape = tuple(max_px - min_px)
    local_points = pixel_points - min_px
    hull = ConvexHull(local_points)
    delaunay = Delaunay(local_points[hull.vertices])

    Z, Y, X = np.mgrid[0:mask_shape[0], 0:mask_shape[1], 0:mask_shape[2]]
    grid_points = np.column_stack([Z.ravel(), Y.ravel(), X.ravel()])
    inside = delaunay.find_simplex(grid_points) >= 0
    mask = inside.reshape(mask_shape)

    slices = tuple(slice(mn, mx) for mn, mx in zip(min_px, max_px))
    return mask, slices
