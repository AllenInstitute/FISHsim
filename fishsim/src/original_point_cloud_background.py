"""
Point cloud background intensity simulator.

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
    """
    points_vox_zyx : (N, 3) voxel coords in (iz, iy, ix) order
    patch_shape    : (Pz, Py, Px)
    kde_sigma_vox  : (sz, sy, sx) in voxel units
    lo_vox_zyx     : (iz0, iy0, ix0) lower corner in global voxel coords
    """
    Pz, Py, Px = patch_shape
    ds = downsample
    iz0, iy0, ix0 = lo_vox_zyx

    low_shape = (max(4, int(np.ceil(Pz / ds))),
                 max(4, int(np.ceil(Py / ds))),
                 max(4, int(np.ceil(Px / ds))))

    density = np.zeros(low_shape, dtype=np.float32)

    izs = np.clip(np.round((points_vox_zyx[:, 0] - iz0) / ds).astype(int), 0, low_shape[0]-1)
    iys = np.clip(np.round((points_vox_zyx[:, 1] - iy0) / ds).astype(int), 0, low_shape[1]-1)
    ixs = np.clip(np.round((points_vox_zyx[:, 2] - ix0) / ds).astype(int), 0, low_shape[2]-1)
    np.add.at(density, (izs, iys, ixs), 1.0)

    sigma_low = tuple(s / ds for s in kde_sigma_vox)
    density = gaussian_filter(density, sigma=sigma_low)
    if density.max() > 0:
        density /= density.max()

    # Upsample to full patch resolution, slice by slice along z
    rz = np.linspace(0, low_shape[0]-1, Pz)
    ry = np.linspace(0, low_shape[1]-1, Py)
    rx = np.linspace(0, low_shape[2]-1, Px)
    gy, gx = np.meshgrid(ry, rx, indexing='ij')
    out = np.empty((Pz, Py, Px), dtype=np.float32)
    for iz, z in enumerate(rz):
        gz = np.full((Py, Px), z, dtype=np.float32)
        out[iz] = map_coordinates(density, [gz, gy, gx], order=1, mode='nearest')
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    return_background=True,
):
    """
    Parameters
    ----------
    cells             : iterable of objects with .emitters (N,3) in (x,y,z) physical units
    volume_shape      : [nz, ny, nx] in PIXELS
    voxel_size        : [vz, vy, vx] physical units per pixel
    noise_scale       : float, physical length scale of background variation
    kde_bandwidth     : float or None  (default 3 x noise_scale)
    noise_amplitude   : float [0,1]
    padding           : float or None, physical padding around cell bbox
                        (default 2 x kde_bandwidth)
    downsample        : int, downsampling factor for KDE blur (default 4)
    seed              : int
    return_background : bool, if False skip noise and return only density

    Returns
    -------
    if return_background=True  -> (background, density), both float16 (nz, ny, nx)
    if return_background=False -> density, float16 (nz, ny, nx)
    """
    voxel_size = list(voxel_size)
    vz, vy, vx = _unpack(voxel_size)

    if kde_bandwidth is None:
        kde_bandwidth = 3.0 * noise_scale
    if padding is None:
        padding = 6.0 * kde_bandwidth

    nz, ny, nx = _grid_shape(volume_shape, voxel_size)
    Lz, Ly, Lx = _physical_extent(volume_shape, voxel_size)
    grid_shape  = (nz, ny, nx)

    kde_sv  = _kde_sigma_vox(kde_bandwidth, voxel_size)

    if return_background:
        noise_sv     = _noise_sigma_vox(noise_scale, voxel_size)
        sample_noise = _make_noise_sampler(grid_shape, noise_sv, seed)
        background   = np.zeros(grid_shape, dtype=np.float16)

    density = np.zeros(grid_shape, dtype=np.float16)

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

        pts_vox    = _emitters_to_vox(pts_xyz, voxel_size)   # (iz, iy, ix)
        lo_vox     = (iz0, iy0, ix0)
        dens_patch = _kde_patch(pts_vox, patch_shape, kde_sv, lo_vox, downsample)

        if return_background:
            noise_patch = sample_noise(iz0, iz1, iy0, iy1, ix0, ix1)
            background[iz0:iz1, iy0:iy1, ix0:ix1] += (noise_patch * dens_patch).astype(np.float16)

        density[iz0:iz1, iy0:iy1, ix0:ix1] += dens_patch.astype(np.float16)

    if density.max() > 0:
        density = (density / density.max()).astype(np.float16)

    if return_background:
        if background.max() > 0:
            background = (background * noise_amplitude / background.max()).astype(np.float16)
        return background, density

    return density


def simulate_background(points, volume_shape, voxel_size, noise_scale,
                        kde_bandwidth=None, noise_amplitude=1.0, seed=42):
    """Single point cloud wrapper. points in (x,y,z) physical units."""
    class _Cell:
        def __init__(self, pts): self.emitters = pts
    return simulate_background_cells(
        [_Cell(points)], volume_shape, voxel_size, noise_scale,
        kde_bandwidth, noise_amplitude, seed=seed)


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
    plt.savefig("/mnt/user-data/outputs/background_demo.png", dpi=150)
    print("Saved.")