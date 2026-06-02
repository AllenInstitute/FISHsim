"""
Point cloud background intensity simulator. Via Claude

Axis conventions (fixed throughout — do not change)
----------------------------------------------------
voxel_size      : [vz, vy, vx]  e.g. [0.4, 0.1084, 0.1084]
volume_shape    : [nz, ny, nx]  in PIXELS  e.g. [40, 2800, 2800]
cell.emitters   : (N, 3) in (x, y, z) physical units
output arrays   : shape (nz, ny, nx)  — matches volume_shape

Internally all grid indexing uses (z, y, x) = (axis0, axis1, axis2).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from pathlib import Path
from scipy.ndimage import zoom
import time
import pandas as pd

from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import os

import dask
import dask.array as da
from dask import delayed

from scipy.fft import fftn, ifftn

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


# ---------------------------------------------------------------------------
# Grid helpers  — everything in (z, y, x) order
# ---------------------------------------------------------------------------

def _unpack(voxel_size):
    """[vz, vy, vx] → (vz, vy, vx)  — explicit, no reordering."""
    vz, vy, vx = voxel_size
    return float(vz), float(vy), float(vx)


def _grid_shape(volume_shape, voxel_size):
    """
    volume_shape : [nz, ny, nx] pixels
    returns      : (nz, ny, nx) — same order, just validated as ints
    """
    nz, ny, nx = volume_shape
    return int(nz), int(ny), int(nx)


def _physical_extent(volume_shape, voxel_size):
    """
    Returns (Lz, Ly, Lx) physical extent.
    volume_shape : [nz, ny, nx] pixels
    voxel_size   : [vz, vy, vx]
    """
    nz, ny, nx = volume_shape
    vz, vy, vx = _unpack(voxel_size)
    return nz * vz, ny * vy, nx * vx


def _emitters_to_vox(points_xyz, voxel_size):
    """
    Convert emitter coords (x, y, z) physical → voxel indices (iz, iy, ix).
    points_xyz : (N, 3) in (x, y, z) physical units
    returns    : (N, 3) in (iz, iy, ix) voxel coords
    """
    vz, vy, vx = _unpack(voxel_size)
    ix = points_xyz[:, 0] / vx
    iy = points_xyz[:, 1] / vy
    iz = points_xyz[:, 2] / vz
    return np.stack([iz, iy, ix], axis=1)


def _kde_sigma_vox(kde_bandwidth, voxel_size):
    """
    KDE blur sigma in voxel units, in (z, y, x) axis order.
    """
    vz, vy, vx = _unpack(voxel_size)
    return (kde_bandwidth / vz,
            kde_bandwidth / vy,
            kde_bandwidth / vx)


def _noise_sigma_vox(noise_scale, voxel_size):
    """Noise scale in voxel units, (z, y, x) axis order."""
    vz, vy, vx = _unpack(voxel_size)
    return (noise_scale / vz,
            noise_scale / vy,
            noise_scale / vx)


# ---------------------------------------------------------------------------
# Lazy noise sampler  — tiny low-res array, sample patches on demand
# ---------------------------------------------------------------------------

def _make_noise_sampler(grid_shape, noise_sigma_vox, seed):
    """
    grid_shape       : (nz, ny, nx)
    noise_sigma_vox  : (sz, sy, sx) in voxel units
    Returns a function sample_patch(z0,z1, y0,y1, x0,x1) → float32 array
    """
    nz, ny, nx = grid_shape
    sz, sy, sx = noise_sigma_vox
    rng = np.random.default_rng(seed)

    low_nz = max(4, int(np.ceil(nz / sz)))
    low_ny = max(4, int(np.ceil(ny / sy)))
    low_nx = max(4, int(np.ceil(nx / sx)))
    low    = rng.standard_normal((low_nz, low_ny, low_nx)).astype(np.float32)

    def sample_patch(z0, z1, y0, y1, x0, x1):
        Pz, Py, Px = z1-z0, y1-y0, x1-x0
        rz = np.linspace(z0/nz*(low_nz-1), (z1-1)/nz*(low_nz-1), Pz)
        ry = np.linspace(y0/ny*(low_ny-1), (y1-1)/ny*(low_ny-1), Py)
        rx = np.linspace(x0/nx*(low_nx-1), (x1-1)/nx*(low_nx-1), Px)
        gy, gx = np.meshgrid(ry, rx, indexing='ij')  # (Py, Px) reused per z-slice
        patch = np.empty((Pz, Py, Px), dtype=np.float32)
        for iz, z in enumerate(rz):
            gz = np.full((Py, Px), z, dtype=np.float32)
            patch[iz] = map_coordinates(low, [gz, gy, gx], order=3, mode='constant')
        p_min, p_max = patch.min(), patch.max()
        if p_max > p_min:
            patch = (patch - p_min) / (p_max - p_min)
        return patch

    return sample_patch


# ---------------------------------------------------------------------------
# KDE density patch  (downsampled blur → upsample)
# ---------------------------------------------------------------------------

def _kde_patch(points_vox_zyx, patch_shape, kde_sigma_vox, lo_vox_zyx, downsample=4):
    Pz, Py, Px = patch_shape
    iz0, iy0, ix0 = lo_vox_zyx
    ds = downsample

    low_shape = (max(4, int(np.ceil(Pz / ds))),
                 max(4, int(np.ceil(Py / ds))),
                 max(4, int(np.ceil(Px / ds))))

    density = np.zeros(low_shape, dtype=np.float32)
    izs = np.clip(np.round((points_vox_zyx[:, 0] - iz0) / ds).astype(int), 0, low_shape[0]-1)
    iys = np.clip(np.round((points_vox_zyx[:, 1] - iy0) / ds).astype(int), 0, low_shape[1]-1)
    ixs = np.clip(np.round((points_vox_zyx[:, 2] - ix0) / ds).astype(int), 0, low_shape[2]-1)
    np.add.at(density, (izs, iys, ixs), 1.0)

    # Gaussian blur via FFT
    sigma_low = tuple(s / ds for s in kde_sigma_vox)
    freq_z = np.fft.fftfreq(low_shape[0])
    freq_y = np.fft.fftfreq(low_shape[1])
    freq_x = np.fft.fftfreq(low_shape[2])
    gz, gy, gx = np.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
    kernel = np.exp(-2 * np.pi**2 * (
        (gz * sigma_low[0])**2 +
        (gy * sigma_low[1])**2 +
        (gx * sigma_low[2])**2
    ))
    density = np.real(ifftn(fftn(density) * kernel)).astype(np.float32)

    '''if density.max() > 0:
        density /= density.max()'''

    # upsample
    # instead of zoom:
    out = np.kron(density, np.ones((ds, ds, ds), dtype=np.float32))
    # then trim to exact patch shape in case of rounding
    out = out[:Pz, :Py, :Px]
    #zoom_factors = (Pz / low_shape[0], Py / low_shape[1], Px / low_shape[2])
    #out = zoom(density, zoom_factors, order=1, mode='nearest').astype(np.float32)
    return out


def _feather_window(shape):
    """
    Smooth window that is 1 in the center and tapers to 0 at all edges.
    Uses a sine taper over the outer 20% of each dimension.
    """
    windows = []
    for n in shape:
        w = np.ones(n, dtype=np.float32)
        taper = max(1, int(n * 0.2))
        ramp = np.sin(np.linspace(0, np.pi/2, taper))
        w[:taper]  = ramp
        w[-taper:] = ramp[::-1]
        windows.append(w)
    # Outer product across all dims
    result = windows[0][:, None, None] * windows[1][None, :, None] * windows[2][None, None, :]
    return result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


'''def simulate_background_cells(
    cells,
    volume_shape,
    voxel_size,
    noise_scale,
    kde_bandwidth=None,
    noise_amplitude=1.0,
    padding=None,
    downsample=4,
    seed=42,
    return_background: bool = False
    ):
    """
        Parameters
        ----------
        cells           : iterable of objects with .emitters (N,3) in (x,y,z) physical units
        volume_shape    : [nz, ny, nx] in PIXELS
        voxel_size      : [vz, vy, vx] physical units per pixel
        noise_scale     : float, physical length scale of background variation
        kde_bandwidth   : float or None  (default 3 × noise_scale)
        noise_amplitude : float [0,1]
        padding         : float or None, physical padding around cell bbox
                        (default 2 × kde_bandwidth)
        downsample      : int, downsampling factor for KDE blur (default 4)
        seed            : int
        return_background : bool, if True, also return the background array (default False), which is the noise modulated by the density. If False, only return the density.
        Returns
        -------
        background : float32 ndarray, shape (nz, ny, nx)
        density    : float32 ndarray, shape (nz, ny, nx)
        """
    voxel_size = list(voxel_size)
    vz, vy, vx = _unpack(voxel_size)
    if kde_bandwidth is None:
        kde_bandwidth = 3.0 * noise_scale
    if padding is None:
        padding = 4.0 * kde_bandwidth
    nz, ny, nx   = _grid_shape(volume_shape, voxel_size)
    Lz, Ly, Lx   = _physical_extent(volume_shape, voxel_size)
    grid_shape    = (nz, ny, nx)
    noise_sv  = _noise_sigma_vox(noise_scale, voxel_size)
    kde_sv    = _kde_sigma_vox(kde_bandwidth, voxel_size)
    if return_background:
        sample_noise = _make_noise_sampler(grid_shape, noise_sv, seed)
        background = np.zeros(grid_shape, dtype=np.float32)
    density = np.zeros((nz, ny, nx), dtype=np.float32)
    for cell in tqdm(cells):
        pts_vox = _emitters_to_vox(cell.emitters, voxel_size)
        iz = np.clip(np.round(pts_vox[:,0]).astype(int), 0, nz-1)
        iy = np.clip(np.round(pts_vox[:,1]).astype(int), 0, ny-1)
        ix = np.clip(np.round(pts_vox[:,2]).astype(int), 0, nx-1)
        np.add.at(density, (iz, iy, ix), 1.0)

        density = gaussian_filter(density, sigma=kde_sv)
        density = (density / density.max()).astype(np.float16)

    if density.max() > 0:
        density = (density / density.max()).astype(np.float16)

    if return_background:
        if background.max() > 0:
            background = (background * noise_amplitude / background.max()).astype(np.float16)
        return background, density
    else:
        return density

def simulate_background(points, volume_shape, voxel_size, noise_scale,
    kde_bandwidth=None, noise_amplitude=1.0, seed=42):
    """Single point cloud wrapper. points in (x,y,z) physical units."""
    class _Cell:
        def __init__(self, pts): self.emitters = pts
    return simulate_background_cells(
            [_Cell(points)], volume_shape, voxel_size, noise_scale,
    kde_bandwidth, noise_amplitude, seed=seed)
'''

'''def simulate_background_cells(
    cells,
    volume_shape,
    voxel_size,
    noise_scale,
    kde_bandwidth=None,
    noise_amplitude=1.0,
    padding=None,
    downsample=4,
    seed=42,
    return_background: bool = False
):
    """
    Parameters
    ----------
    cells           : iterable of objects with .emitters (N,3) in (x,y,z) physical units
    volume_shape    : [nz, ny, nx] in PIXELS
    voxel_size      : [vz, vy, vx] physical units per pixel
    noise_scale     : float, physical length scale of background variation
    kde_bandwidth   : float or None  (default 3 × noise_scale)
    noise_amplitude : float [0,1]
    padding         : float or None, physical padding around cell bbox
                      (default 2 × kde_bandwidth)
    downsample      : int, downsampling factor for KDE blur (default 4)
    seed            : int
    return_background : bool, if True, also return the background array (default False), which is the noise modulated by the density. If False, only return the density.

    Returns
    -------
    background : float32 ndarray, shape (nz, ny, nx)
    density    : float32 ndarray, shape (nz, ny, nx)
    """
    voxel_size = list(voxel_size)
    vz, vy, vx = _unpack(voxel_size)

    if kde_bandwidth is None:
        kde_bandwidth = 3.0 * noise_scale
    if padding is None:
        padding = 4.0 * kde_bandwidth

    nz, ny, nx   = _grid_shape(volume_shape, voxel_size)
    Lz, Ly, Lx   = _physical_extent(volume_shape, voxel_size)
    grid_shape    = (nz, ny, nx)

    noise_sv  = _noise_sigma_vox(noise_scale, voxel_size)
    kde_sv    = _kde_sigma_vox(kde_bandwidth, voxel_size)

    if return_background:
        sample_noise = _make_noise_sampler(grid_shape, noise_sv, seed)
        background = np.zeros(grid_shape, dtype=np.float32)
    density    = np.zeros(grid_shape, dtype=np.float32)

    for cell in tqdm(cells, desc="Simulating background"):
        pts_xyz = np.asarray(cell.emitters, dtype=np.float32)  # (N,3) x,y,z physical

        # Bounding box in physical units (x,y,z), padded and clipped
        lo_x = max(0.0, float(pts_xyz[:,0].min()) - padding)
        lo_y = max(0.0, float(pts_xyz[:,1].min()) - padding)
        lo_z = max(0.0, float(pts_xyz[:,2].min()) - padding)
        hi_x = min(Lx,  float(pts_xyz[:,0].max()) + padding)
        hi_y = min(Ly,  float(pts_xyz[:,1].max()) + padding)
        hi_z = min(Lz,  float(pts_xyz[:,2].max()) + padding)

        # Convert to voxel indices
        ix0 = max(0, int(lo_x / vx));  ix1 = min(nx, int(np.ceil(hi_x / vx)))
        iy0 = max(0, int(lo_y / vy));  iy1 = min(ny, int(np.ceil(hi_y / vy)))
        iz0 = max(0, int(lo_z / vz));  iz1 = min(nz, int(np.ceil(hi_z / vz)))

        patch_shape = (iz1-iz0, iy1-iy0, ix1-ix0)

        if any(s <= 0 for s in patch_shape):
            continue

        pts_vox = _emitters_to_vox(pts_xyz, voxel_size)   # (iz, iy, ix)
        lo_vox  = (iz0, iy0, ix0)

        dens_patch  = _kde_patch(pts_vox, patch_shape, kde_sv, lo_vox, downsample)
        if return_background:
            noise_patch = sample_noise(iz0, iz1, iy0, iy1, ix0, ix1)
            background[iz0:iz1, iy0:iy1, ix0:ix1] += (noise_patch * dens_patch).astype(np.float32)
        density   [iz0:iz1, iy0:iy1, ix0:ix1] += dens_patch.astype(np.float32)

    if density.max() > 0:
        density = (density / density.max()).astype(np.float32)
    if return_background:
        if background.max() > 0:
            background = (background * noise_amplitude / background.max()).astype(np.float32)
        return background, density
    return density
'''

def simulate_background_cells(
    cells,
    volume_shape,
    voxel_size,
    noise_scale,
    kde_bandwidth=None,
    noise_amplitude=1.0,
    padding=None,
    downsample=4,
    seed=42,
    return_background: bool = False,
    block_shape=(40, 700, 700)  # add this
):
    def process_cell(cell, tile_slices):
        pts_xyz = np.asarray(cell.emitters, dtype=np.float32)  # (N,3) x,y,z physical

        # Bounding box in physical units, padded and clipped
        lo_x = max(0.0, float(pts_xyz[:,0].min()) - padding)
        lo_y = max(0.0, float(pts_xyz[:,1].min()) - padding)
        lo_z = max(0.0, float(pts_xyz[:,2].min()) - padding)
        hi_x = min(Lx,  float(pts_xyz[:,0].max()) + padding)
        hi_y = min(Ly,  float(pts_xyz[:,1].max()) + padding)
        hi_z = min(Lz,  float(pts_xyz[:,2].max()) + padding)

        # Convert to voxel indices
        ix0 = max(0, int(lo_x / vx));  ix1 = min(nx, int(np.ceil(hi_x / vx)))
        iy0 = max(0, int(lo_y / vy));  iy1 = min(ny, int(np.ceil(hi_y / vy)))
        iz0 = max(0, int(lo_z / vz));  iz1 = min(nz, int(np.ceil(hi_z / vz)))
        iz0 = max(iz0, tile_slices[0].start);  iz1 = min(iz1, tile_slices[0].stop)
        iy0 = max(iy0, tile_slices[1].start);  iy1 = min(iy1, tile_slices[1].stop)
        ix0 = max(ix0, tile_slices[2].start);  ix1 = min(ix1, tile_slices[2].stop)

        patch_shape = (iz1-iz0, iy1-iy0, ix1-ix0)
        if any(s <= 0 for s in patch_shape):
            return np.zeros((0,0,0), dtype=np.float32), None, None

        patch_shape = (iz1-iz0, iy1-iy0, ix1-ix0)
        #print(f"patch_shape: {patch_shape}, n_emitters: {len(pts_xyz)}")
        if any(s <= 0 for s in patch_shape):
            return np.zeros((0,0,0), dtype=np.float32), None, None

        pts_vox    = _emitters_to_vox(pts_xyz, voxel_size)
        lo_vox     = (iz0, iy0, ix0)
        dens_patch = _kde_patch(pts_vox, patch_shape, kde_sv, lo_vox, downsample)

        if dens_patch.max() > 0:
            dens_patch = (dens_patch / dens_patch.max()).astype(np.float32)

        bg_patch = None
        if return_background:
            noise_patch = sample_noise(iz0, iz1, iy0, iy1, ix0, ix1)
            bg_patch = (noise_patch * dens_patch)
            if bg_patch.max() > 0:
                bg_patch = (bg_patch * noise_amplitude / bg_patch.max()).astype(np.float32)

        slices = (slice(iz0,iz1), slice(iy0,iy1), slice(ix0,ix1))
        return dens_patch, bg_patch, slices

    voxel_size = list(voxel_size)
    vz, vy, vx = _unpack(voxel_size)
    
    if kde_bandwidth is None:
        kde_bandwidth = 3.0 * noise_scale
    if padding is None:
        padding = 6.0 * kde_bandwidth

    nz, ny, nx = _grid_shape(volume_shape, voxel_size)
    Lz, Ly, Lx = _physical_extent(volume_shape, voxel_size)
    grid_shape  = (nz, ny, nx)

    noise_sv = _noise_sigma_vox(noise_scale, voxel_size)
    kde_sv   = _kde_sigma_vox(kde_bandwidth, voxel_size)

    if return_background:
        sample_noise = _make_noise_sampler(grid_shape, noise_sv, seed)

    def cells_in_tile(cells, tile_slices, voxel_size, padding):
        """Filter cells whose bounding box overlaps the tile."""
        vz, vy, vx = voxel_size
        tile_lo = (tile_slices[0].start * vz,
                   tile_slices[1].start * vy,
                   tile_slices[2].start * vx)
        tile_hi = (tile_slices[0].stop  * vz,
                   tile_slices[1].stop  * vy,
                   tile_slices[2].stop  * vx)
        result = []
        for cell in cells:
            pts = np.asarray(cell.emitters)
            lo = pts.min(axis=0) - padding  # xyz
            hi = pts.max(axis=0) + padding
            # xyz order: x->2, y->1, z->0
            if (lo[0] < tile_hi[2] and hi[0] > tile_lo[2] and
                lo[1] < tile_hi[1] and hi[1] > tile_lo[1] and
                lo[2] < tile_hi[0] and hi[2] > tile_lo[0]):
                result.append(cell)
        return result

    def process_tile(tile_slices, tile_cells):
        tile_shape = tuple(s.stop - s.start for s in tile_slices)
        density    = np.zeros(tile_shape, dtype=np.float32)
        background = np.zeros(tile_shape, dtype=np.float32) if return_background else None

        for cell in tile_cells:
            dens_patch, bg_patch, slices = process_cell(cell, tile_slices)
            
            # translate global slices to tile-local coordinates
            local_slices = tuple(
                slice(s.start - b.start, s.stop - b.start)
                for s, b in zip(slices, tile_slices)
            )
            
            # clip to tile bounds
            clipped_local = []
            clipped_dens  = []
            for ls, bs, ps in zip(local_slices, tile_shape, dens_patch.shape):
                l_start = max(ls.start, 0)
                l_stop  = min(ls.stop, bs)
                d_start = l_start - ls.start
                d_stop  = d_start + (l_stop - l_start)
                if l_stop <= l_start:
                    break
                clipped_local.append(slice(l_start, l_stop))
                clipped_dens.append(slice(d_start, d_stop))
            else:
                density[tuple(clipped_local)] += dens_patch[tuple(clipped_dens)]
                if return_background and bg_patch is not None:
                    background[tuple(clipped_local)] += bg_patch[tuple(clipped_dens)]

        return background, density

    # Tile the volume — use the same block shape as PSF function
    tile_shape = block_shape
    tiles = []
    for z in range(0, volume_shape[0], tile_shape[0]):
        for y in range(0, volume_shape[1], tile_shape[1]):
            for x in range(0, volume_shape[2], tile_shape[2]):
                slices = (
                    slice(z, min(z + tile_shape[0], volume_shape[0])),
                    slice(y, min(y + tile_shape[1], volume_shape[1])),
                    slice(x, min(x + tile_shape[2], volume_shape[2])),
                )
                tile_cells = cells_in_tile(cells, slices, voxel_size, padding)
                tiles.append((slices, tile_cells))

    delayed_tiles = [
        delayed(process_tile)(slices, tile_cells)
        for slices, tile_cells in tiles
    ]

    # Assemble into dask arrays
    n_z = len(range(0, volume_shape[0], tile_shape[0]))
    n_y = len(range(0, volume_shape[1], tile_shape[1]))
    n_x = len(range(0, volume_shape[2], tile_shape[2]))

    delayed_density    = [
        da.from_delayed(delayed(lambda r: r[1])(d),
                        shape=tuple(s.stop-s.start for s in slices),
                        dtype=np.float32)
        for d, (slices, _) in zip(delayed_tiles, tiles)
    ]
    delayed_background = [
        da.from_delayed(delayed(lambda r: r[0])(d),
                        shape=tuple(s.stop-s.start for s in slices),
                        dtype=np.float32)
        for d, (slices, _) in zip(delayed_tiles, tiles)
    ] if return_background else None

    # Reassemble grid
    def make_grid(dask_arrays):
        grid = []
        idx = 0
        for z in range(n_z):
            row_grid = []
            for y in range(n_y):
                col_grid = []
                for x in range(n_x):
                    col_grid.append(dask_arrays[idx])
                    idx += 1
                row_grid.append(col_grid)
            grid.append(row_grid)
        return da.block(grid)

    density_da    = make_grid(delayed_density)
    background_da = make_grid(delayed_background) if return_background else None

    return (background_da, density_da) if return_background else density_da


class _Cell:
    def __init__(self, cell_df):
        self.emitters = cell_df[["x", "y", "z"]]


def simulate_density(cells, volume_shape, voxel_size, kde_bandwidth, downsample=4):
    if isinstance(cells, pd.DataFrame):
        g = cells.groupby("cell_id")
        cells = []
        for cell_id, cell_df in g:
            cells.append(_Cell(cell_df))

    voxel_size = list(voxel_size)
    vz, vy, vx = _unpack(voxel_size)
    nz, ny, nx = _grid_shape(volume_shape, voxel_size)

    # Downsample the full volume for the blur
    ds = downsample
    low_shape = (max(4, int(np.ceil(nz / ds))),
                 max(4, int(np.ceil(ny / ds))),
                 max(4, int(np.ceil(nx / ds))))

    density = np.zeros(low_shape, dtype=np.float32)

    kde_sv = _kde_sigma_vox(kde_bandwidth, voxel_size)
    sigma_low = tuple(s / ds for s in kde_sv)

    for cell in cells:
        n_emitters = cell.emitters.shape[0]
        pts_xyz = np.asarray(cell.emitters, dtype=np.float32)
        pts_vox = _emitters_to_vox(pts_xyz, voxel_size)
        izs = np.clip(np.round(pts_vox[:, 0] / ds).astype(int), 0, low_shape[0]-1)
        iys = np.clip(np.round(pts_vox[:, 1] / ds).astype(int), 0, low_shape[1]-1)
        ixs = np.clip(np.round(pts_vox[:, 2] / ds).astype(int), 0, low_shape[2]-1)
        np.add.at(density, (izs, iys, ixs), 1.0)

    density = gaussian_filter(density, sigma=sigma_low)

    # Upsample back to full resolution slice by slice
    nz_l, ny_l, nx_l = low_shape
    ry = np.linspace(0, ny_l-1, ny)
    rx = np.linspace(0, nx_l-1, nx)
    gy, gx = np.meshgrid(ry, rx, indexing='ij')
    out = np.empty((nz, ny, nx), dtype=np.float32)
    rz = np.linspace(0, nz_l-1, nz)
    for iz, z in enumerate(rz):
        gz = np.full((ny, nx), z, dtype=np.float32)
        out[iz] = map_coordinates(density, [gz, gy, gx], order=1, mode='nearest')

    total_emitters = sum(cell.emitters.shape[0] for cell in cells)
    current_sum = float(out.sum())
    if current_sum > 0:
        out *= total_emitters / current_sum
    '''voxel_volume = voxel_size[0]*voxel_size[1]*voxel_size[2]
    out = out / voxel_volume'''
    '''if out.max() > 0:
        out /= out.max()'''
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Vectorised surface generation (replaces the double Python loop)
# ---------------------------------------------------------------------------
 
def generate_surface_vectorised(ellipsoid, n: int = 200) -> np.ndarray:
    """Vectorised drop-in replacement for Ellipsoid.generate__surface().
 
    Eliminates the 200x200 Python loop by computing all rotations and
    translations with a single batched matrix multiply.
 
    Parameters
    ----------
    ellipsoid : Ellipsoid
        Must have .axes, .center, and .R.
    n : int
        Grid resolution (default 200, matching the original method).
 
    Returns
    -------
    np.ndarray, shape (n*n, 3)
        Surface points in world coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, n)
    phi   = np.arccos(1 - 2 * np.linspace(0, 1, n))
    T, P  = np.meshgrid(theta, phi)           # T=theta, P=phi, each (n, n)
 
    # Correct spherical parameterisation:
    #   x = a * sin(phi) * cos(theta)
    #   y = b * sin(phi) * sin(theta)
    #   z = c * cos(phi)
    x = (ellipsoid.axes[0] * np.sin(P) * np.cos(T)).ravel()
    y = (ellipsoid.axes[1] * np.sin(P) * np.sin(T)).ravel()
    z = (ellipsoid.axes[2] * np.cos(P)).ravel()
 
    pts = np.stack([x, y, z], axis=0)         # (3, N)
    return (ellipsoid.R @ pts).T + ellipsoid.center  # (N, 3) world coords
 
 
# ---------------------------------------------------------------------------
# Core per-ellipsoid mask computation (no shared state, safe to pickle)
# ---------------------------------------------------------------------------
 
def _compute_masks(ellipsoid, voxel_size: np.ndarray):
    """Return (origin, filled_mask, surface_mask) for one ellipsoid.
 
    Coordinate conventions
    ----------------------
    - Ellipsoid attributes (axes, center, R) are in (x, y, z) world order.
    - voxel_size and the output array are in (z, y, x) order.
    - The conversion happens here and nowhere else: we reverse the world-space
      quantities to (z, y, x) before doing any voxel-index arithmetic.
    """
    # half_extents_world: R row i gives the world-axis-i components of the
    # rotated ellipsoid. Rows are (x, y, z), so reverse to get (z, y, x).
    half_extents_world_xyz = np.sqrt(np.sum((ellipsoid.R * ellipsoid.axes) ** 2, axis=1))
    half_extents_world_zyx = half_extents_world_xyz[::-1]          # (z, y, x)
    half_extents = half_extents_world_zyx / voxel_size             # voxel units, (z,y,x)
 
    # center is (cx, cy, cz) — reverse to (cz, cy, cx) before dividing by voxel_size
    center_zyx = ellipsoid.center[::-1]
    origin = np.round(center_zyx / voxel_size - half_extents).astype(int)
    bbox_size = np.ceil(2 * half_extents).astype(int) + 1
    dz, dy, dx = bbox_size
 
    # Filled mask: build voxel grid, convert offsets back to world coords (x,y,z)
    # for the ellipsoid distance test, then reshape.
    gz, gy, gx = np.mgrid[0:dz, 0:dy, 0:dx]
    # World-space offsets from ellipsoid centre, keeping (x, y, z) order for R
    wx = (gx - half_extents[2]) * voxel_size[2]   # x = array axis 2
    wy = (gy - half_extents[1]) * voxel_size[1]   # y = array axis 1
    wz = (gz - half_extents[0]) * voxel_size[0]   # z = array axis 0
    pts_xyz = np.stack([wx.ravel(), wy.ravel(), wz.ravel()], axis=1)  # (N, 3) in xyz
    pts_local = pts_xyz @ ellipsoid.R
    dist2 = np.sum((pts_local / ellipsoid.axes) ** 2, axis=1)
    filled_mask = (dist2 <= 1.0).reshape(dz, dy, dx)
 
    # Surface mask: points come out of generate_surface_vectorised in (x, y, z).
    # Reverse each point to (z, y, x) before computing local voxel indices.
    surface_pts_xyz = generate_surface_vectorised(ellipsoid)          # (N, 3) xyz
    surface_pts_zyx = surface_pts_xyz[:, ::-1]                        # (N, 3) zyx
    bbox_corner_zyx = center_zyx - half_extents_world_zyx
    local_idx = np.round((surface_pts_zyx - bbox_corner_zyx) / voxel_size).astype(int)
    valid = (
        (local_idx[:, 0] >= 0) & (local_idx[:, 0] < dz)
        & (local_idx[:, 1] >= 0) & (local_idx[:, 1] < dy)
        & (local_idx[:, 2] >= 0) & (local_idx[:, 2] < dx)
    )
    idx = local_idx[valid]
    surface_mask = np.zeros((dz, dy, dx), dtype=bool)
    surface_mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
 
    return origin, filled_mask, surface_mask
 
 
def _write_masks(target, origin, filled_mask, surface_mask, filled_value, surface_value):
    """Write pre-computed masks into target at origin, clipping to bounds."""
    tz, ty, tx = target.shape
    dz, dy, dx = filled_mask.shape
    z0, y0, x0 = origin
 
    src_z0 = max(0, -z0);  src_z1 = min(dz, tz - z0)
    src_y0 = max(0, -y0);  src_y1 = min(dy, ty - y0)
    src_x0 = max(0, -x0);  src_x1 = min(dx, tx - x0)
 
    dst_z0 = max(0, z0);   dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y0 = max(0, y0);   dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, x0);   dst_x1 = dst_x0 + (src_x1 - src_x0)
 
    if not (src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0):
        return  # ellipsoid entirely outside target
 
    src_s = (slice(src_z0, src_z1), slice(src_y0, src_y1), slice(src_x0, src_x1))
    dst_s = (slice(dst_z0, dst_z1), slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))
 
    target[dst_s][filled_mask[src_s]]  = filled_value
    target[dst_s][surface_mask[src_s]] = surface_value
 
 
# ---------------------------------------------------------------------------
# Shared-memory worker: compute masks then write directly into shared array
# ---------------------------------------------------------------------------
 
def _worker(args):
    """Compute masks for one ellipsoid and write into the shared volume.
 
    Parameters are passed as a single tuple so ProcessPoolExecutor can
    pickle them.
    """
    (ellipsoid, voxel_size,
     shm_name, shape, dtype,
     filled_value, surface_value) = args
 
    voxel_size = np.asarray(voxel_size)
    origin, filled_mask, surface_mask = _compute_masks(ellipsoid, voxel_size)
 
    # Attach to the shared memory block — no copy of the full volume
    shm = shared_memory.SharedMemory(name=shm_name)
    target = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
 
    # Non-overlapping ellipsoids: no locking needed.
    # If you have overlapping ellipsoids, wrap _write_masks in a Lock.
    _write_masks(target, origin, filled_mask, surface_mask, filled_value, surface_value)
 
    shm.close()
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def insert_ellipsoid_mask(
    ellipsoid,
    target: np.ndarray,
    voxel_size: float | tuple,
    filled_value: int = 1,
    surface_value: int = 2,
) -> np.ndarray:
    """Insert a single voxelised ellipsoid mask into target (in-place).
 
    For inserting many ellipsoids, use insert_all_ellipsoid_masks which
    parallelises across CPU cores via shared memory.
 
    Parameters
    ----------
    ellipsoid : Ellipsoid
        Instance with .axes, .center, .R.
    target : np.ndarray, shape (Z, Y, X)
        Destination volume, modified in-place.
    voxel_size : float or (sz, sy, sx)
        Physical size of one voxel in world-coordinate units.
    filled_value : int
        Value for interior voxels (default 1).
    surface_value : int
        Value for surface voxels, written over filled (default 2).
 
    Returns
    -------
    np.ndarray — target, modified in-place.
    """
    voxel_size = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,)).copy()
    origin, filled_mask, surface_mask = _compute_masks(ellipsoid, voxel_size)
    _write_masks(target, origin, filled_mask, surface_mask, filled_value, surface_value)
    return target
 
 
def insert_all_ellipsoid_masks(
    ellipsoids: list,
    target: np.ndarray,
    voxel_size,
    filled_values=1,
    surface_values=2,
    max_workers: int = None,
) -> np.ndarray:
    """Insert all ellipsoids into target in parallel using shared memory.
 
    Each worker process computes the bounding-box masks for one ellipsoid and
    writes them directly into a shared memory block. The large volume array is
    never copied between processes.
 
    Parameters
    ----------
    ellipsoids : list of Ellipsoid
    target : np.ndarray, shape (Z, Y, X)
        Modified in-place. Must be a C-contiguous array.
    voxel_size : float or (sz, sy, sx)
    filled_values : int or list of int
        Value(s) written for interior voxels. Pass a single int to use the same
        value for all ellipsoids, or a list of len(ellipsoids) to assign a
        unique label per cell — e.g. list(range(1, len(ellipsoids) + 1)).
    surface_values : int or list of int
        Same as filled_values but for surface voxels (written over filled).
    max_workers : int, optional
        Number of worker processes. Defaults to os.cpu_count().
 
    Returns
    -------
    np.ndarray — target, modified in-place.
 
    Notes
    -----
    Assumes ellipsoids are non-overlapping. If two ellipsoids share voxels,
    whichever worker finishes last wins — results are non-deterministic in the
    overlap region. For overlapping cases, use insert_ellipsoid_mask in a
    serial loop instead.
    """
    if not target.data.c_contiguous:
        raise ValueError(
            "target must be C-contiguous. Call np.ascontiguousarray(target) first."
        )
 
    n = len(ellipsoids)
    fv = filled_values  if isinstance(filled_values,  list) else [filled_values]  * n
    sv = surface_values if isinstance(surface_values, list) else [surface_values] * n
    if len(fv) != n or len(sv) != n:
        raise ValueError(
            "filled_values and surface_values must each have one entry per ellipsoid."
        )
 
    voxel_size_arr = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,)).copy()
 
    # Place the volume in shared memory — workers attach by name, no copy per worker
    shm = shared_memory.SharedMemory(create=True, size=target.nbytes)
    shared_arr = np.ndarray(target.shape, dtype=target.dtype, buffer=shm.buf)
    shared_arr[:] = target  # copy initial state (usually all zeros)
 
    worker_args = [
        (e, voxel_size_arr.tolist(), shm.name, target.shape, target.dtype, f, s)
        for e, f, s in zip(ellipsoids, fv, sv)
    ]
 
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_worker, worker_args))  # consume iterator to surface exceptions
        target[:] = shared_arr  # write results back into the caller's array
    finally:
        shm.close()
        shm.unlink()
 
    return target
    
        
'''
import open3d as o3d
from skimage.draw import ellipsoid
from skimage.measure import marching_cubes, mesh_to_volume


def generate_mask_from_points(points, volume_shape):
    """
    Generates a 3D binary mask from a list of surface points.

    Args:
        points (np.ndarray): A (N, 3) array of surface points.
        volume_shape (tuple): The desired shape of the output 3D mask (e.g., (100, 100, 100)).

    Returns:
        np.ndarray: A 3D binary mask (numpy array).
    """
    # 1. Convert numpy points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optional: Estimate normals for better mesh reconstruction
    pcd.estimate_normals()
    
    # 2. Reconstruct a closed surface mesh (e.g., using Ball Pivoting algorithm)
    # The radii parameter is crucial and depends on the density of your points
    radii = [0.005, 0.01, 0.02, 0.04] # Adjust radii based on your data scale
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    # Ensure the mesh is oriented consistently (watertight is best)
    mesh.orient_triangles()

    # 3. Convert the mesh to a binary volume (mask) using skimage
    # Adjust the level_set value to define the interior (e.g., 0.5)
    # The output mask will have the specified volume_shape
    mask = mesh_to_volume(
        vertices=np.asarray(mesh.vertices), 
        faces=np.asarray(mesh.triangles), 
        volume_shape=volume_shape
    ).astype(bool)

    return mask

'''
"""
# --- Example Usage ---
# 1. Generate sample surface points (e.g., points on an ellipsoid)
# Create a sample ellipsoid volume first to get ground truth points
vol_shape = (100, 100, 100)
# create_surface expects a function for the surface values, this is an example
verts, faces, _, _ = marching_cubes(ellipsoid(60, 60, 35, levelset=True), level=0)

# Shift vertices so they are centered around the origin (adjust if your points are not centered)
center_shift = np.array(vol_shape) / 2.
points = verts - center_shift

# 2. Generate the mask
binary_mask = generate_mask_from_points(points, vol_shape)

# 3. Visualize a slice of the result
plt.imshow(binary_mask[:, :, vol_shape[2] // 2], cmap='gray')
plt.title(f"Center slice of generated mask (shape: {binary_mask.shape})")
plt.axis('off')
plt.show()
"""

# ---------------------------------------------------------------------------
# Autofluorescence simulation
# ---------------------------------------------------------------------------

def simulate_autofluorescence(
    cells_df: pd.DataFrame,
    volume_shape: list,
    pixel_size: list,
    mean_photons: float,
    cv: float = 0.3,
    smooth_scale_um: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate per-cell autofluorescence as a photon-count volume.

    Each cell ellipsoid is filled with smooth, spatially varying intensity.
    Cell-to-cell brightness is drawn from a log-normal distribution so the
    population mean equals *mean_photons* per interior voxel.

    Parameters
    ----------
    cells_df : DataFrame
        Cell geometry table from generate_scene.py (cells.csv).
        Required columns: center_x/y/z, axis_a/b/c, R_00…R_22.
    volume_shape : [nz, ny, nx] in pixels
    pixel_size   : [vz, vy, vx] in µm
    mean_photons : mean photon count per interior voxel across all cells
    cv           : coefficient of variation for cell-to-cell intensity (default 0.3)
    smooth_scale_um : spatial length scale of within-cell texture in µm (default 2.0)
    seed         : random seed

    Returns
    -------
    float32 ndarray of shape (nz, ny, nx) — photon counts from autofluorescence
    """
    nz, ny, nx = volume_shape
    vz, vy, vx = pixel_size

    rng = np.random.default_rng(seed)
    out = np.zeros((nz, ny, nx), dtype=np.float32)

    # Log-normal parameters: mean of exp(X) = mean_photons for each cell
    sigma_ln = np.sqrt(np.log(1 + cv ** 2))
    mu_ln = np.log(mean_photons) - 0.5 * sigma_ln ** 2

    smooth_sigma_vox = (
        smooth_scale_um / vz,
        smooth_scale_um / vy,
        smooth_scale_um / vx,
    )

    for _, row in cells_df.iterrows():
        cx, cy, cz = float(row["center_x"]), float(row["center_y"]), float(row["center_z"])
        a    = float(row["axis_a"])
        b    = float(row["axis_b"])
        c_ax = float(row["axis_c"])
        R_mat = np.array(
            [[row[f"R_{ri}{ci}"] for ci in range(3)] for ri in range(3)],
            dtype=np.float64,
        )

        max_r = max(a, b, c_ax)
        ix_lo = max(0,  int((cx - max_r) / vx))
        ix_hi = min(nx, int((cx + max_r) / vx) + 2)
        iy_lo = max(0,  int((cy - max_r) / vy))
        iy_hi = min(ny, int((cy + max_r) / vy) + 2)
        iz_lo = max(0,  int((cz - max_r) / vz))
        iz_hi = min(nz, int((cz + max_r) / vz) + 2)

        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            continue

        IZ, IY, IX = np.mgrid[iz_lo:iz_hi, iy_lo:iy_hi, ix_lo:ix_hi]
        dx = IX * vx - cx
        dy = IY * vy - cy
        dz = IZ * vz - cz
        pts = np.stack([dx.ravel(), dy.ravel(), dz.ravel()])  # (3, N) in xyz
        local = R_mat.T @ pts                                  # (3, N)
        inside = (
            local[0] ** 2 / a ** 2
            + local[1] ** 2 / b ** 2
            + local[2] ** 2 / c_ax ** 2
        ) <= 1.0
        inside = inside.reshape(IZ.shape)

        if not inside.any():
            continue

        # Smooth noise texture: Gaussian-filtered white noise, normalised to
        # mean=1, std=0.3 within the cell interior, then clipped to >=0.
        bbox_shape = inside.shape
        local_sigma = (
            min(smooth_sigma_vox[0], bbox_shape[0] / 2.0),
            min(smooth_sigma_vox[1], bbox_shape[1] / 2.0),
            min(smooth_sigma_vox[2], bbox_shape[2] / 2.0),
        )
        raw = rng.standard_normal(bbox_shape).astype(np.float32)
        smooth = gaussian_filter(raw, sigma=local_sigma)
        interior_vals = smooth[inside]
        s = interior_vals.std()
        if s > 0:
            smooth = (smooth - interior_vals.mean()) / s * 0.3 + 1.0
        else:
            smooth = np.ones_like(smooth)
        smooth = np.clip(smooth, 0.0, None)

        cell_photons = float(np.exp(rng.normal(mu_ln, sigma_ln)))
        out[iz_lo:iz_hi, iy_lo:iy_hi, ix_lo:ix_hi] += (
            smooth * inside * cell_photons
        ).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    class FakeCell:
        def __init__(self, centre, n=200):
            pts = centre + rng.standard_normal((n, 3)) * np.array([5, 5, 1])
            self.emitters = pts.astype(np.float32)

    centres = np.column_stack([
        rng.uniform(20, 180, 8),
        rng.uniform(20, 180, 8),
        rng.uniform(1,   7,  8),
    ])
    cells = [FakeCell(c) for c in centres]

    bg, dens = simulate_background_cells(
        cells,
        volume_shape = [20,  1850, 1850],   # nz, ny, nx  pixels
        voxel_size   = [0.4, 0.108, 0.108], # vz, vy, vx
        noise_scale  = 10.0,
        kde_bandwidth= 8.0,
        seed=42,
    )

    print(f"Output shape (nz, ny, nx): {bg.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(dens.max(axis=0), origin='lower', cmap='inferno', aspect='auto')
    axes[0].set_title("Density (max-proj Z)")
    axes[1].imshow(bg.max(axis=0),   origin='lower', cmap='inferno', aspect='auto')
    axes[1].set_title("Background (max-proj Z)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    outpath = Path.home() / "Downloads" / "background_demo.png"
    plt.savefig(outpath, dpi=150)
    print("Saved.")