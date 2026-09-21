import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter


def _ellipsoid_M(cell):
    """M = R @ diag(axes): columns are scaled principal axis vectors in µm."""
    return cell.R @ np.diag(cell.axes)


def _cell_aabb_um(cell, padding_um=0.0):
    """Axis-aligned bounding box of a rotated ellipsoid in physical µm.

    Returns dict with keys 'x', 'y', 'z', each a (lo, hi) tuple.
    cell.center is (x, y, z).
    """
    M = _ellipsoid_M(cell)
    # AABB half-extents: for axis i, extent = sqrt(sum of M[i,:]^2)
    extents = np.sqrt((M ** 2).sum(axis=1)) + padding_um  # [dx, dy, dz]
    cx, cy, cz = cell.center
    dx, dy, dz = extents
    return {
        'x': (cx - dx, cx + dx),
        'y': (cy - dy, cy + dy),
        'z': (cz - dz, cz + dz),
    }


def _projected_ellipse_boundary(M, center, row_idx, n=300):
    """Exact boundary of an ellipsoid projected onto a 2D plane.

    Args:
        M: (3, 3) matrix with columns = scaled principal axis vectors
        center: (3,) array, (x, y, z) physical µm
        row_idx: tuple of 2 ints selecting which physical coordinates (x=0, y=1, z=2)
        n: number of boundary points

    Returns: (2, n) boundary in physical µm
    """
    i, j = row_idx
    M2 = M[[i, j], :]  # (2, 3)
    U, S, _ = np.linalg.svd(M2, full_matrices=False)  # U: (2,2), S: (2,)
    theta = np.linspace(0, 2 * np.pi, n)
    circle = np.array([np.cos(theta), np.sin(theta)])  # (2, n)
    return center[[i, j], None] + U @ (S[:, None] * circle)


def find_center_cell(cells, sample_volume_zyx):
    """Return (index, cell) of the cell whose center is nearest the scene center.

    sample_volume_zyx: (z_um, y_um, x_um) total physical extent.
    cell.center convention: (x, y, z) in µm.
    """
    z_um, y_um, x_um = sample_volume_zyx
    scene_center = np.array([x_um / 2, y_um / 2, z_um / 2])
    dists = [np.linalg.norm(np.asarray(c.center) - scene_center) for c in cells]
    idx = int(np.argmin(dists))
    return idx, cells[idx]


def crop_image_and_points(image, df, cell, voxel_size, padding_um=10.0):
    """Crop image and point-cloud DataFrame to the cell's AABB.

    Args:
        image: (nz, ny, nx) array
        df: DataFrame with columns 'x', 'y', 'z' in physical µm
        cell: EllipsoidCell with .center (x,y,z) and .axes, .R
        voxel_size: [vz, vy, vx] in µm
        padding_um: extra padding around the AABB

    Returns:
        crop: (nz_c, ny_c, nx_c) subarray
        df_crop: filtered DataFrame
        origin_zyx: (z0, y0, x0) voxel indices of crop start
    """
    vz, vy, vx = voxel_size
    nz, ny, nx = image.shape
    aabb = _cell_aabb_um(cell, padding_um)

    x0 = max(0, int(np.floor(aabb['x'][0] / vx)))
    x1 = min(nx, int(np.ceil(aabb['x'][1] / vx)))
    y0 = max(0, int(np.floor(aabb['y'][0] / vy)))
    y1 = min(ny, int(np.ceil(aabb['y'][1] / vy)))
    z0 = max(0, int(np.floor(aabb['z'][0] / vz)))
    z1 = min(nz, int(np.ceil(aabb['z'][1] / vz)))

    crop = image[z0:z1, y0:y1, x0:x1]

    mask = (
        (df['x'] >= x0 * vx) & (df['x'] < x1 * vx) &
        (df['y'] >= y0 * vy) & (df['y'] < y1 * vy) &
        (df['z'] >= z0 * vz) & (df['z'] < z1 * vz)
    )
    return crop, df[mask].copy(), (z0, y0, x0)


def detect_spots_3d(image_crop, voxel_size, sigma_um=0.3, min_sep_um=0.5, threshold_rel=0.15):
    """Detect local-maxima in a 3D image crop as candidate transcript locations.

    Args:
        image_crop: (nz, ny, nx) array
        voxel_size: [vz, vy, vx] in µm
        sigma_um: Gaussian pre-smoothing radius in µm
        min_sep_um: minimum separation between peaks in µm
        threshold_rel: minimum peak height as fraction of image max

    Returns: (N, 3) array of (z, y, x) voxel indices relative to crop origin
    """
    vz, vy, vx = voxel_size
    sigma_vox = [sigma_um / vz, sigma_um / vy, sigma_um / vx]
    smoothed = gaussian_filter(image_crop.astype(float), sigma=sigma_vox)
    size_vox = [max(1, round(min_sep_um / vs)) for vs in voxel_size]
    is_local_max = maximum_filter(smoothed, size=size_vox) == smoothed
    peaks_mask = is_local_max & (smoothed > threshold_rel * smoothed.max())
    return np.argwhere(peaks_mask)


def ortho_figure(image_crop, voxel_size, origin_zyx, df_crop, spots_vox, cell):
    """3-panel orthographic figure with detected and ground-truth transcript overlays.

    All scatter coordinates and axis labels are in physical µm.
    imshow extent follows matplotlib convention [left, right, bottom, top] with origin='upper',
    so that low-index voxels (lo µm values) appear at the top of z/y axes.

    Args:
        image_crop: (nz, ny, nx) array
        voxel_size: [vz, vy, vx] in µm
        origin_zyx: (z0, y0, x0) voxel offset of crop start in the full image
        df_crop: DataFrame with 'x', 'y', 'z' columns in physical µm
        spots_vox: (N, 3) detected spot positions in (z, y, x) voxels relative to crop
        cell: EllipsoidCell used for boundary overlay

    Returns: matplotlib Figure
    """
    vz, vy, vx = voxel_size
    z0, y0, x0 = origin_zyx
    nz, ny, nx = image_crop.shape

    x_lo, x_hi = x0 * vx, (x0 + nx) * vx
    y_lo, y_hi = y0 * vy, (y0 + ny) * vy
    z_lo, z_hi = z0 * vz, (z0 + nz) * vz

    # Detected spots in physical µm (handle empty array gracefully)
    if len(spots_vox):
        sp_z = (z0 + spots_vox[:, 0]) * vz
        sp_y = (y0 + spots_vox[:, 1]) * vy
        sp_x = (x0 + spots_vox[:, 2]) * vx
    else:
        sp_z = sp_y = sp_x = np.array([])

    M = _ellipsoid_M(cell)
    center = np.asarray(cell.center, dtype=float)  # (x, y, z)

    # imshow extent = [left, right, bottom, top] with origin='upper':
    #   top edge of image = first row = low y/z value
    #   bottom edge = last row = high y/z value
    panels = [
        dict(
            proj=image_crop.max(axis=0),             # (ny, nx)
            extent=[x_lo, x_hi, y_hi, y_lo],
            xlabel='x (µm)', ylabel='y (µm)',
            title='XY — top view (Z max projection)',
            gt=(df_crop['x'].values, df_crop['y'].values),
            det=(sp_x, sp_y),
            boundary=_projected_ellipse_boundary(M, center, (0, 1)),
        ),
        dict(
            proj=image_crop.max(axis=1),             # (nz, nx)
            extent=[x_lo, x_hi, z_hi, z_lo],
            xlabel='x (µm)', ylabel='z (µm)',
            title='XZ — side view (Y max projection)',
            gt=(df_crop['x'].values, df_crop['z'].values),
            det=(sp_x, sp_z),
            boundary=_projected_ellipse_boundary(M, center, (0, 2)),
        ),
        dict(
            proj=image_crop.max(axis=2),             # (nz, ny)
            extent=[y_lo, y_hi, z_hi, z_lo],
            xlabel='y (µm)', ylabel='z (µm)',
            title='YZ — front view (X max projection)',
            gt=(df_crop['y'].values, df_crop['z'].values),
            det=(sp_y, sp_z),
            boundary=_projected_ellipse_boundary(M, center, (1, 2)),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Center cell — orthographic views', fontsize=13)

    for ax, p in zip(axes, panels):
        proj = p['proj'].astype(float)
        vmin, vmax = np.percentile(proj, [1, 99.5])
        im = ax.imshow(
            proj,
            extent=p['extent'],
            origin='upper',
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='nearest',
        )

        gx, gy = p['gt']
        if len(gx):
            ax.scatter(gx, gy, s=12, c='orange', marker='x', linewidths=0.8,
                       alpha=0.8, label='Ground truth', zorder=3)

        dx, dy = p['det']
        if len(dx):
            ax.scatter(dx, dy, s=6, c='cyan', alpha=0.6, linewidths=0,
                       label='Detected', zorder=4)

        bnd = p['boundary']
        ax.plot(bnd[0], bnd[1], '--', color='white', linewidth=1.2,
                alpha=0.85, label='Cell boundary', zorder=5)

        ax.set_xlabel(p['xlabel'])
        ax.set_ylabel(p['ylabel'])
        ax.set_title(p['title'])
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='counts')

    axes[0].legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    return fig


def visualize_center_cell(
    image,
    df,
    cells,
    voxel_size,
    sample_volume_zyx,
    padding_um=10.0,
    sigma_um=0.3,
    min_sep_um=0.5,
    threshold_rel=0.15,
    savepath=None,
):
    """Crop to the center cell and produce a 3-panel orthographic figure.

    Args:
        image: (nz, ny, nx) array (uint16 or float32)
        df: DataFrame with columns 'x', 'y', 'z' in physical µm
        cells: list of EllipsoidCell objects
        voxel_size: [vz, vy, vx] in µm
        sample_volume_zyx: (z_um, y_um, x_um) total scene extent in µm
        padding_um: padding added around cell AABB for the crop
        sigma_um, min_sep_um, threshold_rel: passed to detect_spots_3d
        savepath: Path or str; if given, figure is saved as PNG at 150 dpi

    Returns: matplotlib Figure
    """
    center_idx, cell = find_center_cell(cells, sample_volume_zyx)
    crop, df_crop, origin_zyx = crop_image_and_points(
        image, df, cell, voxel_size, padding_um
    )
    # Keep only GT points belonging to the center cell; neighbours share the AABB
    if 'cell_id' in df_crop.columns:
        df_crop = df_crop[df_crop['cell_id'] == center_idx]
    spots_vox = detect_spots_3d(
        crop, voxel_size,
        sigma_um=sigma_um,
        min_sep_um=min_sep_um,
        threshold_rel=threshold_rel,
    )
    fig = ortho_figure(crop, voxel_size, origin_zyx, df_crop, spots_vox, cell)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')

    return fig
