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
            patch[iz] = map_coordinates(low, [gz, gy, gx], order=3, mode='nearest')
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

    if density.max() > 0:
        density /= density.max()

    # upsample
    # instead of zoom:
    out = np.kron(density, np.ones((ds, ds, ds), dtype=np.float32))
    # then trim to exact patch shape in case of rounding
    out = out[:Pz, :Py, :Px]
    #zoom_factors = (Pz / low_shape[0], Py / low_shape[1], Px / low_shape[2])
    #out = zoom(density, zoom_factors, order=1, mode='nearest').astype(np.float32)
    return out

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
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
    return_backgound: bool = False
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

    if return_backgound:
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
        if return_backgound:
            noise_patch = sample_noise(iz0, iz1, iy0, iy1, ix0, ix1)
            background[iz0:iz1, iy0:iy1, ix0:ix1] += (noise_patch * dens_patch).astype(np.float32)
        density   [iz0:iz1, iy0:iy1, ix0:ix1] += dens_patch.astype(np.float32)

    if density.max() > 0:
        density = (density / density.max()).astype(np.float32)
    if return_backgound:
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
        padding = 4.0 * kde_bandwidth

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