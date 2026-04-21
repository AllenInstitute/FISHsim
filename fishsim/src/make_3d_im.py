from pathlib import Path
from fishsim.src import sim3d, point_cloud_background
from fishsim.src.imaging import CameraSimulator, CY3
from fishsim import config
import pandas as pd
import numpy as np
from scipy import constants as k
from scipy.fft import next_fast_len
from dask.diagnostics import ProgressBar


def main(
    psf_path: Path = Path("Y:/MERFISHp/12_04_2025_JenieSample/psf_final.pkl"),
    psf_position: tuple = (1200, 1200),
    psf_channel: int = 0,
    pixel_size: list = None,
    volume_shape: list = None,
):
    """
    Args:
        psf_path: Path to PSF pickle file. The file should be a dict-like
            object keyed by (channel_index, y_pixel, x_pixel).
        psf_position: (y_px, x_px) field position used as key into the PSF dict.
        psf_channel: Channel index for the PSF dict lookup.
        pixel_size: Voxel size [z, y, x] in µm. Defaults to [0.4, 0.1084333, 0.1084333].
        volume_shape: Output volume in pixels [n_z, n_y, n_x].
            Defaults to [40, 2800, 2800] (matching the pilot dataset).
    """
    if pixel_size is None:
        pixel_size = [0.4, 0.1084333, 0.1084333]
    if volume_shape is None:
        volume_shape = [40, 2800, 2800]

    sample_volume = (
        volume_shape[0] * pixel_size[0],
        volume_shape[1] * pixel_size[1],
        volume_shape[2] * pixel_size[2],
    )

    code_book_path = Path(config.CODEBOOK_DIR) / "C1E1_codebook_no_distribution.csv"

    sim = sim3d.Simulator(
        emitters_per_cell=500,
        cell_count=120,
        cell_axes={"a": [10, 15], "b": [10, 15], "c": [10, 15]},
        bit_drop=0.1,
        bit_add=0.1,
    )

    ground_truth = sim.generate_point_cloud(
        codebook_filepath=code_book_path,
        tiles=1,
        is_subpixel=True,
        is_cell=True,
        is_nucleus=False,
        sample_volume=sample_volume,
    )

    df = pd.read_csv(ground_truth[0])
    df[['frame', 'row', 'column', 'frame_shift', 'row_shift', 'column_shift']] = (
        sim3d.compute_pixel_locations(
            df[['z', 'y', 'x']].to_numpy(), pixel_size=pixel_size
        )
    )

    cell_ids = list(range(1, len(sim.cells) + 1))
    bg_mask = np.zeros(volume_shape)
    bg_mask = point_cloud_background.insert_all_ellipsoid_masks(
        [cell.shape for cell in sim.cells],
        bg_mask,
        voxel_size=pixel_size,
        filled_values=cell_ids,
        surface_values=cell_ids,
    )
    bg_mask = np.moveaxis(bg_mask, 2, 1).astype("uint8")

    psf_path = Path(psf_path)
    psf_obj = np.load(psf_path, allow_pickle=True)
    psf = psf_obj[(psf_channel, np.int64(psf_position[0]), np.int64(psf_position[1]))].astype("float32")

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
    FZ, FR, FC = np.meshgrid(fz, fr, fc, indexing='ij')

    tile_point_im_dask = sim3d.build_tile_point_im(
        df, FZ, FR, FC, psf_fft, psf_crop_slices, psf,
        volume_shape=volume_shape,
        block_shape=(40, 700, 700),
    )

    exposure_s = 50. * k.milli
    tile_photon_im = CY3.psf_to_photon_distribution(tile_point_im_dask.clip(min=0), exposure_s)
    with ProgressBar():
        tile_photon_im = tile_photon_im.compute(num_workers=64)

    camera = CameraSimulator(QE={561: 0.8}, gain=1. / 0.25, bias=100,
                             dark_current=1., read_noise=0, well_depth=15000)
    noisy_image = camera.simulate_image(tile_photon_im, 561, exposure_s)

    return (tile_photon_im, bg_mask, noisy_image, df)
