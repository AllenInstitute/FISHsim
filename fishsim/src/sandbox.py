import pandas as pd
import numpy as np
import scipy.io
import pickle
import time
import numpy as np
from pathlib import Path
from scipy.signal.windows import tukey
from typing import List, Tuple
from numpy.typing import ArrayLike
import argparse
import yaml
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import ConvexHull, Delaunay

import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(60)

import dask
import dask.array as da
from dask import delayed

from scipy.ndimage import gaussian_filter
import random
import math
import skimage.io

from tqdm.auto import tqdm

from fishsim.src.ellipsoid import Ellipsoid
from fishsim.src.generate_emitters import (
    cell_emitter_position, random_emitter_position
)
from fishsim.src.utils import BASE_PROJECT_DIR, glob_background, trunc_norm
from fishsim.config import RESULTS_DIR, RESOURCES_DIR, CODEBOOK_DIR

config_file = Path(RESOURCES_DIR) / "config_serval_experiment_randomdist_noisy_scenario1.yml"
# Parse the configuration file
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

'''# Choose the appropriate camera and optics parameters
simulator = Simulator(
    config["simulation"], config["cameras"]["Prime95B"], config["optics"]["NikonTi2"],
)'''


class Simulator(object):
    def __init__(
            self, simulation_params: dict, camera_params: dict, optics_params: dict
    ):
        # Simulation parameters
        self.number_emitter = simulation_params["emitter_count"]
        self.image_size = simulation_params["image_size"]
        self.scr = simulation_params["scr"]
        self.photon_count = simulation_params["photon_count"]
        
        # Handle scalar vs list for round-level variation
        self.photon_count_list = (
            self.photon_count if isinstance(self.photon_count, list) else None
            )
        self.scr_list = self.scr if isinstance(self.scr, list) else None
        self.global_background_level_list = None
        self.wavelengths = simulation_params["signal_wavelengths"]
        self.bit_drop = simulation_params["bitdrop_probability"]
        self.bit_add = simulation_params["bitadd_probability"]
        self.bg_sampling_prob = simulation_params["background_sampling_probability"]

        # Cell parameters
        self.cell_count = simulation_params["cells"]["count"]
        self.cell_axes = simulation_params["cells"]["axes"]
        self.cells = []  # list to store the cell instances

        # Camera parameters
        self.QE = camera_params["qe"]
        self.gain = camera_params["gain"]
        self.bias = camera_params["bias"]
        self.dark_current = camera_params["dark_current"]
        self.exposure_times = camera_params["exposure_times"]
        self.read_noise = camera_params["read_noise"]
        self.sensor_size = camera_params["sensor_size"]
        self.well_depth = camera_params["well_depth"]

        # Optical parameters
        self.magnification = optics_params["magnification"]


    def generate_point_cloud(
        self,
        codebook_filepath: Path,
        psf_filepath: Path,
        data_org_filepath: Path,
        tiles: int = 1,
        is_matlab: bool = True,
        is_subpixel: bool = True,
        is_cell: bool = True,
        is_nucleus: bool = False,
        z_bound: tuple = None,
        savepath: Path = None,
    ) -> None:
        """
        Altering this from simulation.generate_training_data() separate generateing point data
        and image data.

        - PSF stuff not needed
        - background stuff not needed
        - cell and nucleus stuff *I think* is needed, since it distributes emitters differently
        - matlab stuff not needed

        This funtion generate the training data for number of set of merFISH data you want. Called generate_single_traing_data tiles times
        Inputs:  codebook_filename: a csv file containing the merfish library you are using
                    psf_filename: the file containing the psf of your microscope (in resource file if you look at our data)
                    tiles: number of set of data(16 images per set) you want
                    is_matlab: boolean value, true to take psf from .mat file, otherwise use python pickle psf file
                    num_bits: number of barcodes in your library

        Output:  groundtruth.csv: contain the photon count, location, gene and barcode of every beads
                    Tiff images: the tiff image for each bit, total num_bits
        """

        # Load the codebook
        codebook = pd.read_csv(codebook_filepath, skiprows=3)
        data_organization = pd.read_csv(data_org_filepath)
        # TODO: count number of bits in the codebook and remove the argument
        
        # Map each bit to its imaging round (0-based indexing)
        self.bit_to_round = {
            row["bitNumber"] - 1: row["imagingRound"]
            for _, row in data_organization.iterrows()
        }


        t = time.strftime("%Y%m%d-%H-%M-%S")
        folder_name = "generated_data_merfish_" + t
        if savepath is None:
            folder_path = Path(RESULTS_DIR) / "results" / folder_name
        else:
            folder_path = savepath / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(tiles), total=tiles, desc="Generating tiles"):
            print("starting tile " + str(i + 1))
            self.generate_single_tile_point_cloud(
                codebook,
                data_organization,
                #psf,
                i,
                is_subpixel,
                is_cell,
                is_nucleus,
                z_bound,
                folder_path,
            )
            
        '''# Log simulation parameters
        self.print_simulation_params(
            folder_path,
            codebook_filepath,
            psf_filepath,
            data_org_filepath,
            tiles,
            is_matlab,
            is_subpixel,
            is_cell,
            is_nucleus,
        )'''


    def generate_single_tile_point_cloud(
        self,
        codebook: pd.DataFrame,
        data_organization: pd.DataFrame,
        #psf: np.ndarray,
        tile_num: int,
        is_subpixel: bool,
        is_cell: bool,
        is_nucleus: bool,
        z_bound: tuple, # depracated?
        folder_path: Path,
        sample_volume: tuple = (40*0.4, 2800*0.1084333, 2800*0.1084333), # I think I want this to be in microns, but I don't know where the other functions might be using the magnification
    ) -> None:
        """
        Adapting from simulation.__generate_single_training_data(), as above

        Creates a single set of training data

        Args:
            codebook (pd.DataFrame): codebook dataframe
            data_organization (pd.DataFrame): data organization dataframe
            psf (np.ndarray): psf matrix
            tile_num (int): number of the tile that's being generated
            is_subpixel (bool): if true, emitter locations are floating point (i.e. non-zero shift)
            is_cell (bool): if true, cell structures imposed on the simulation
            is_nucleus (bool): if true, cells have a nucleus
            z_bound (tuple): z range for the emitter/cell center position (centered around zero)
            folder_path (Path): folder where the results are saved
        """
        # Simulate flourescent probes
        genes = codebook["name"]
        gene_ids = codebook["numeric_id"]
        barcodes = codebook["barcode"]
        numofbits = len(barcodes[0].replace(" ", ""))

        # if there is distribution column, draw sample based on that distribution
        if "distribution" in codebook:
            gene_index = list(range(len(genes)))
            distribution = np.array(codebook["distribution"].fillna(0).to_list())
            distribution = [int(item) for item in distribution]

            # r contain the number of lines in the table for the genes
            r = [i for i in gene_index for _ in range(distribution[i])]
            random.shuffle(r)  # shuffle to randomize the order of genes
            num_emitter = len(r)
        
        # if not, draw sample using uniform distribution
        else:
            genes = codebook["name"]
            # r contain the number of lines in the table for the genes
            r = np.random.randint(0, np.size(barcodes), size=self.number_emitter)
            num_emitter = self.number_emitter

        if is_cell:
            emitter_pos, self.cells = cell_emitter_position(
                x_dim=(0, sample_volume[2] - 1),
                y_dim=(0, sample_volume[1] - 1),
                z_dim=(0, sample_volume[0] - 1),
                num_emitter=num_emitter,
                cell_count=self.cell_count,
                cell_axes_bounds=self.cell_axes,
                is_nucleus=is_nucleus,
                is_physical_coordinates= True, # want to deal in microns for now
            )
            # I want to preserve the cell identity of each emitter
            cell_emitters = []
            cell_ids = []
            for cell_id, cell in enumerate(self.cells):
                for emitter in cell.emitters:
                    cell_emitters.append(list(emitter))
                    cell_ids.append(cell_id)
            #cell_emitters = np.array(cell_emitters)
            emitter_pos = np.array(cell_emitters)
            cell_ids = np.array(cell_ids)
        else:
            # emitter_pos has shape (num_emitters, 3 columns)
            # col 1: x, col 2: y, col 3: z; see generate_emitters.py
            emitter_pos = random_emitter_position(
                x_dim=(0, sample_volume[2] - 1),
                y_dim=(0, sample_volume[1] - 1),
                z_dim=(0, sample_volume[0] - 1),
                num_emitter=num_emitter,
            )
            cell_ids = np.array([-1] * num_emitter)  # assign -1 for non-cell emitters

        if not is_subpixel:
            emitter_pos = np.floor(emitter_pos)

        # Crate a goundtruth data frame
        ground_truth = []

        # emitter_barcodes store only the barcode of emitter
        emitter_barcodes = np.empty((emitter_pos.shape[0], numofbits))

        # Randomly determine the emitters that will have bit drop/add
        is_bit_drop = np.random.rand(emitter_pos.shape[0]) <= self.bit_drop
        is_bit_add = np.random.rand(emitter_pos.shape[0]) <= self.bit_add
        #distribution for the emitter brightness
        
        # Precompute 8 light_dist objects (one per imaging round)
        num_rounds = len(set(self.bit_to_round.values()))  # Should be 8
        light_dists_by_round = {}
        
        # This for-loop will create a light distribution per imaging round
        # For 8 imagingRounds, there will be 8 light distributions.
        for round_idx in range(num_rounds):
            photon_count = (
                self.photon_count_list[round_idx]
                if self.photon_count_list
                else self.photon_count
            )
            light_dists_by_round[round_idx] = trunc_norm(
                0,
                self.well_depth,
                mean=photon_count,
                var=int(photon_count / 20) if photon_count >= 20 else 0.5
            )

        for i in tqdm(range(emitter_pos.shape[0])):
            gene = genes[r[i]]
            gene_id = gene_ids[r[i]]
            barcode = barcodes[r[i]].replace(" ", "")
            mapped_barcode = Simulator.map_barcode(barcode, data_organization)
            cell_id = cell_ids[i]
            ground_truth.append(
                [   
                    # Generate one photon count for each of the 8 imaging rounds
                    # The original code for this was light_dist.choose(), and it serves no particular purpose.
                    # We follow its original purpose of "generating some light_dist value", here we generate
                    # a list of 8 values assuming 8 imagingRounds.
                    [
                        light_dists_by_round[round_idx].choose()[0].item()
                        for round_idx in range(8)
                    ],
                    int(
                        math.floor(emitter_pos[i, 1])
                        + self.image_size * math.floor(emitter_pos[i, 0])
                    ),
                    emitter_pos[i,1],#math.floor(emitter_pos[i, 1]),
                    emitter_pos[i,0], #math.floor(emitter_pos[i, 0]),
                    emitter_pos[i,2], #math.floor(emitter_pos[i, 2]),
                    #f"{(emitter_pos[i, 1] % 1):5f}",
                    #f"{(emitter_pos[i, 0] % 1):5f}",
                    #f"{(emitter_pos[i, 2] % 1):5f}",
                    gene,
                    gene_id,
                    f"'{barcode}'",
                    f"'{mapped_barcode}'",
                    is_bit_drop[i],
                    is_bit_add[i],
                    cell_id,
                ]
            )
            emitter_barcodes[i, :] = list(mapped_barcode)
            if is_bit_drop[i]:
                index_ones = [
                    k for k in range(numofbits) if emitter_barcodes[i, k] == 1
                ]
                dropout_bit = random.choice(index_ones)
                emitter_barcodes[i, dropout_bit] = 0

            if is_bit_add[i]:
                index_zeros = [
                    k for k in range(numofbits) if emitter_barcodes[i, k] == 0
                ]
                add_bit = random.choice(index_zeros)
                emitter_barcodes[i, add_bit] = 1

        # Create a new folder store the image and groundtruth
        tile_folder_name = "tile_" + str(tile_num + 1)
        tile_folder_path = folder_path / tile_folder_name
        tile_folder_path.mkdir(parents=True, exist_ok=True)

        # Create folders for wavelength
        for wavelength in self.wavelengths:
            (tile_folder_path / str(wavelength)).mkdir(parents=True, exist_ok=True)

        # Save the ground truth table
        ground_truth = pd.DataFrame(
            ground_truth,
            columns=[
                "photoncount",
                "pixel_index_num",
                "x", #"column",
                "y", # "row",
                "z",
                #"column_shift",
                #"row_shift",
                #"z_shift",
                "genes",
                "gene_id",
                "barcode",
                "mapped_barcode",
                "is_bit_drop",
                "is_bit_add",
                "cell_id",
            ],
        )
        ground_truth.to_csv(tile_folder_path / "groundtruth.csv", index=False)

        # Save the frequency of a gene appers
        frequency = (
            ground_truth.value_counts("genes")
            .rename_axis("genes")
            .to_frame(name="counts")
        )
        frequency.to_csv(tile_folder_path / "frequency.csv")

        # I think we're done here?
    
    @staticmethod
    def map_barcode(barcode: str, data_organization: pd.DataFrame) -> str:
        bit_number = data_organization["bitNumber"]
        imaging_round = data_organization["imagingRound"]
        color = data_organization["color"]

        mapping = dict()
        for i in range(data_organization.shape[0]):
            mapping[bit_number[i] - 1] = (
                imaging_round[i] * 2 if color[i] == 650 else imaging_round[i] * 2 + 1
            )

        mapped_barcode = ["0"] * len(barcode)
        for i, bit in enumerate(barcode):
            mapped_barcode[mapping[i]] = bit

        return "".join(mapped_barcode)



def get_shifted_psf_and_slices(
    psf: np.ndarray,
    row: int, col: int, z: int,
    row_shift: float, col_shift: float, z_shift: float,
    volume_shape: tuple,
    pad_factor: int = 2  # increase if artifacts persist
) -> tuple:

    # --- 1. Pad PSF to prevent wrap-around ---
    orig_shape = np.array(psf.shape)
    pad_width = [(s * (pad_factor - 1) // 2,) * 2 for s in orig_shape]
    psf_padded = np.pad(psf, pad_width, mode='constant')

    # --- 2. Sub-pixel shift on padded array ---
    pz, pr, pc = psf_padded.shape
    fz = np.fft.fftfreq(pz)
    fr = np.fft.fftfreq(pr)
    fc = np.fft.fftfreq(pc)
    FZ, FR, FC = np.meshgrid(fz, fr, fc, indexing='ij')
    phase_ramp = np.exp(
        -1j * 2 * np.pi * (FZ * z_shift + FR * row_shift + FC * col_shift)
    )
    #shifted_padded = np.real(np.fft.ifftn(np.fft.fftn(psf_padded) * phase_ramp))
    shifted_padded = np.real(fft.ifftn(psf_padded * phase_ramp))

    # --- 3. Crop back to original PSF size ---
    crop = tuple(
        slice(pad_width[i][0], pad_width[i][0] + orig_shape[i])
        for i in range(3)
    )
    shifted_psf = shifted_padded[crop].astype(psf.dtype)

    # --- 4. Compute insertion slices ---
    half = orig_shape // 2
    center = np.array([z, row, col])
    vol = np.array(volume_shape)

    starts = center - half
    ends   = starts + orig_shape

    vol_starts = np.clip(starts, 0, vol)
    vol_ends   = np.clip(ends,   0, vol)
    psf_starts = vol_starts - starts
    psf_ends   = psf_starts + (vol_ends - vol_starts)

    vol_slices = tuple(slice(int(s), int(e)) for s, e in zip(vol_starts, vol_ends))
    psf_slices = tuple(slice(int(s), int(e)) for s, e in zip(psf_starts, psf_ends))

    return shifted_psf, vol_slices, psf_slices


def apodize_psf_tukey(psf: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    alpha=0 → rectangular (no apodization)
    alpha=1 → Hann window
    alpha=0.1-0.2 → flat center with narrow cosine rolloff at borders
    """
    windows = [tukey(s, alpha=alpha) for s in psf.shape]
    Wz, Wr, Wc = np.meshgrid(*windows, indexing='ij')
    return psf * Wz * Wr * Wc


def compute_pixel_locations(emitter_pos: np.ndarray, pixel_size: List[float]) -> tuple:
    """Convert emitter positions to pixel indices and subpixel shifts
    
    Args:   
        emitter_pos (np.ndarray): shape (N, 3) with columns (z, y, x) in physical units
        pixel_size (List[float]): [z_size, row_size, col_size] in physical units
    Returns:
        pixel_indices (np.ndarray): shape (N, 3) with integer pixel indices (frame, row, col)
        subpixel_shifts (np.ndarray): shape (N, 3) with subpixel shifts in fraction of pixel

    Example:
        df[['frame', 'row', 'column', 'frame_shift', 'row_shift', 'column_shift']] = compute_pixel_locations(
            df[['z', 'y', 'x']].to_numpy(), pixel_size=[0.4, 0.1084333, 0.1084333]
        )
    """
    pixel_indices = np.floor(emitter_pos / np.array(pixel_size)).astype(int)
    subpixel_shifts = (emitter_pos / np.array(pixel_size)) % 1
    
    return np.concat([pixel_indices, subpixel_shifts], axis=1)


def add_shot_noise(
    image: ArrayLike,
    photon_conversion: float = 0.24  # photons per count
) -> ArrayLike:
    """ Can just use skimage.util.random_noise with mode='poisson' ? """
    original_dtype = image.dtype
    image = image.astype(np.float32)
    image_photons = image * photon_conversion
    noisy_image = np.random.poisson(image_photons)
    noisy_image = noisy_image / photon_conversion
    return noisy_image.astype(original_dtype)


def make_dark_current_image(
    dark_current: float, exposure_time: float, image_shape: tuple
) -> np.ndarray:
    mean_dark = dark_current * exposure_time
    return np.random.poisson(mean_dark, size=image_shape)


class DyeSimulator:
    def __init__(
            self,
            lifetime: float, # fluorescence lifetime in seconds
            quantum_yield: float, # probability of emitting a photon upon excitation
    ):
        """ Simulate dye photophysics, converting a PSF (in arbitrary units) to a photon distribution based on dye properties and imaging conditions.
        I am ignoring excitation intensity, absorption cross-section, etc. for now,
        instead, I am going to use existing data to calibrate that out.

        Args:
            lifetime (float): fluorescence lifetime in seconds
            quantum_yield (float): probability of emitting a photon upon excitation
        """
        self.lifetime = lifetime
        self.quantum_yield = quantum_yield
    
    def psf_to_photon_distribution(
            self,
            psf: ArrayLike, # normalized point spread function (in arbitrary units)
            #excitation_intensity: float, # in photons per second
            exposure_time: float, # in seconds
    ) -> ArrayLike:
        """ Convert a PSF (in arbitrary units) to a photon distribution based on dye properties and imaging conditions.
        
        Args:
            psf (np.ndarray): normalized point spread function (in arbitrary units)
            #excitation_intensity (float): excitation intensity in photons per second
            exposure_time (float): exposure time in seconds
        Returns:
            photon_distribution (np.ndarray): simulated photon counts at each pixel
        """
        # Scale PSF by excitation intensity and quantum yield
        #expected_photons = psf * excitation_intensity * self.quantum_yield * exposure_time
        # ignoring cross-section. Assuming ideal conditions
        expected_photons = psf * self.quantum_yield * exposure_time/self.lifetime
        
        # Simulate photon emission as a Poisson process
        if isinstance(psf, np.ndarray):
            photon_distribution = np.random.poisson(expected_photons)
        else:
            photon_distribution = da.map_blocks(np.random.poisson, expected_photons, dtype=np.int64)
        
        return photon_distribution


CY3 = DyeSimulator(lifetime=2.0E-9, quantum_yield=0.3)

class CameraSimulator:
    def __init__(
        self,
        QE: dict,
        gain: float,
        bias: float,
        dark_current: float,
        read_noise: float,
        well_depth: int
    ):
        """ Simulate camera effects such as shot noise, dark current, read noise, gain, and bias.
        Args:
            QE (dict): quantum efficiency for each wavelength
            gain (float): gain factor to convert electrons to counts
            bias (float): bias level in counts
            dark_current (float): dark current in electrons per second
            read_noise (float): read noise in electrons
            well_depth (int): maximum number of electrons per pixel before saturation
        """
        self.QE = QE
        self.gain = gain
        self.bias = bias
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.well_depth = well_depth
    
    def simulate_image(
            self,
            photon_image: np.ndarray,
            wavelength: int,
            exposure_time: float
        ) -> np.ndarray:
        """ Simulate the final image given the photon image (after optics) and camera parameters."""
        #1. Add photon shot noise
        photon_image_noisy = add_shot_noise(photon_image)
        #2. Apply quantum efficiency        electrons_image = photon_image_noisy * self.QE.get(wavelength, 0)
        electrons_image = photon_image_noisy * self.QE.get(wavelength, 0)
        #3. Add dark current
        dark_image = make_dark_current_image(self.dark_current, exposure_time, photon_image.shape)
        electrons_image += dark_image
        '''#4. Add read noise
        electrons_image += np.random.normal(0, self.read_noise, size=photon_image.'''
        #5. Apply gain and bias
        counts_image = electrons_image * self.gain + self.bias
        #6. Clip to well depth        counts_image = np.clip(counts_image, 0, self.well_depth * self.gain + self.bias)
        counts_image = np.clip(counts_image, 0, self.well_depth * self.gain + self.bias)
        return counts_image.astype(np.uint16)


def filter_spotdf_on_round(spot_df: pd.DataFrame, round_num: int) -> pd.DataFrame:
    """ Filter the spot dataframe to only include spots that are "on" in the given round number.
    Here, 'round_numer' is a direct barcode position, so includes channel info. For example, if all rounds have
    3 channels, then round_num=1, round_num=2, and round_num=3 are the three channels from the first round.
    Args:
        spot_df (pd.DataFrame): dataframe containing spot information, including a 'barcode' column as string, e.g. '100110000001000'
        round_num (int): the round number to filter on (0-based indexing)
    Returns:
        pd.DataFrame: filtered dataframe containing only spots that are "on" in the given round number

    Use this to get all the emitters from a round when making images.
    """
    round_idx = spot_df.barcode.str.get(round_num) == '1'
    return spot_df.loc[round_idx].copy()



def ellipsoid_to_mask(
    ellipsoid: Ellipsoid, pixel_size: float, shape: list
) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Generates a 3D binary mask of an ellipsoid and slices to place it in a larger volume.

    Args:
        ellipsoid (Ellipsoid): ellipsoid instance to mask
        pixel_size (float): size of each voxel in microns (isotropic)
        shape (list): shape of the larger volume [Z, Y, X] in pixels

    Returns:
        tuple:
            - np.ndarray: 3D boolean mask of the ellipsoid's bounding box
            - tuple[slice, ...]: slices (Z, Y, X) to place the mask into the larger volume
    """
    axes_px = ellipsoid.axes[[0, 1, 2]] / pixel_size   # reorder to (z, y, x)
    center_px = ellipsoid.center[[2, 0, 1]] / pixel_size # not sure why coordinates are like this

    # Compute the bounding box in pixel space, clamped to the volume shape
    pad = int(np.ceil(np.max(axes_px))) + 1
    z_min = max(0, int(np.floor(center_px[0] - pad)))
    z_max = min(shape[0], int(np.ceil(center_px[0] + pad)))
    y_min = max(0, int(np.floor(center_px[1] - pad)))
    y_max = min(shape[1], int(np.ceil(center_px[1] + pad)))
    x_min = max(0, int(np.floor(center_px[2] - pad)))
    x_max = min(shape[2], int(np.ceil(center_px[2] + pad)))

    slices = (
        slice(z_min, z_max),
        slice(y_min, y_max),
        slice(x_min, x_max),
    )

    # Build coordinate grid over the bounding box only
    z_idx, y_idx, x_idx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]

    # Offset from ellipsoid center, in (z, y, x) order
    coords = np.stack([
        z_idx - center_px[0],
        y_idx - center_px[1],
        x_idx - center_px[2],
    ], axis=-1)

    # Reorder R to (z, y, x) convention by permuting rows and columns
    perm = np.array([0, 1, 2])
    R_zyx = ellipsoid.R#[np.ix_(perm, perm)]

    coords_local = coords @ R_zyx
    mask = np.sum((coords_local / axes_px) ** 2, axis=-1) <= 1.0

    return mask, slices


def make_cell_background(
        cells: List[Ellipsoid],
        pixel_size: float,
        shape: list,
        taper_length: float = 2.0,  # in microns
        level: float = 100  # background level inside the cell 
) -> np.ndarray:
    background = np.zeros(shape, dtype="float16")
    taper_length_px = [taper_length / p for p in pixel_size]  # convert taper length to pixels for each dimension
    for cell in tqdm(cells):
        mask, slices = points_to_convex_hull_mask(cell.emitters, pixel_size, shape)
        #mask_dist = distance_transform_edt(mask, sampling=pixel_size)
        #epsilon = taper_length/(mask_dist[mask] + 1e-5)
        #tapered_mask = np.zeros_like(mask_dist)
        #tapered_mask[mask] = 1./(1 + np.exp(epsilon - 1/(1 - 1/epsilon)))
       #tapered_mask[mask_dist > taper_length] = 1.0  
        #background[slices] = np.maximum(background[slices], tapered_mask * level)
        background[slices] = np.maximum(background[slices], mask * level)
    return background


def points_to_convex_hull_mask(points, pixel_size, volume_shape=None):
    """
    Create a binary mask of the convex hull from a list of (z, y, x) points
    in physical (micron) coordinates.
    
    Args:
        points:       array-like of shape (N, 3) with (z, y, x) physical coordinates
        pixel_size:   scalar or array-like of shape (3,) with (z, y, x) microns/pixel
        volume_shape: tuple (Z, Y, X) of the larger volume. If None, the mask shape
                      is inferred from the bounding box of the points and slices
                      will start at (0, 0, 0).
    
    Returns:
        mask:   boolean numpy array covering the bounding box of the points
        slices: tuple of slice() objects placing the mask into the larger volume
    """
    points = np.array(points, dtype=float)
    points = points[:, [2, 0, 1]]
    pixel_size = np.broadcast_to(pixel_size, (3,))

    # Convert physical coords to pixel coords
    pixel_points = points / pixel_size

    # Bounding box in pixel space
    min_px = np.floor(pixel_points.min(axis=0)).astype(int)
    max_px = np.ceil(pixel_points.max(axis=0)).astype(int) + 1

    if volume_shape is not None:
        min_px = np.clip(min_px, 0, np.array(volume_shape) - 1)
        max_px = np.clip(max_px, 0, np.array(volume_shape))

    mask_shape = tuple(max_px - min_px)

    # Shift points so they're relative to the bounding box origin
    local_points = pixel_points - min_px

    # Build Delaunay triangulation of the convex hull vertices
    hull = ConvexHull(local_points)
    delaunay = Delaunay(local_points[hull.vertices])

    # Create a grid of all voxel coordinates within the bounding box
    Z, Y, X = np.mgrid[0:mask_shape[0], 0:mask_shape[1], 0:mask_shape[2]]
    grid_points = np.column_stack([Z.ravel(), Y.ravel(), X.ravel()])

    inside = delaunay.find_simplex(grid_points) >= 0
    mask = inside.reshape(mask_shape)

    slices = tuple(slice(mn, mx) for mn, mx in zip(min_px, max_px))

    return mask, slices


def points_overlapping_block(points_arrays, block_slices, psf_shape):
    margin = [s // 2 for s in psf_shape]
    mask = (
        (points_arrays['frame']  >= block_slices[0].start - margin[0]) &
        (points_arrays['frame']  <  block_slices[0].stop  + margin[0]) &
        (points_arrays['row']    >= block_slices[1].start - margin[1]) &
        (points_arrays['row']    <  block_slices[1].stop  + margin[1]) &
        (points_arrays['column'] >= block_slices[2].start - margin[2]) &
        (points_arrays['column'] <  block_slices[2].stop  + margin[2])
    )
    return {k: v[mask] for k, v in points_arrays.items()}


def compute_insertion_slices(row, col, z, psf_shape, volume_shape):
    """Returns (vol_slices, psf_slices) with boundary clamping."""
    vol_slices = []
    psf_slices = []
    
    for center, psf_size, vol_size in zip(
        [z, row, col], psf_shape, volume_shape
    ):
        start = center - psf_size // 2
        stop  = start + psf_size

        # Clamp to volume bounds
        vol_start = max(start, 0)
        vol_stop  = min(stop, vol_size)

        # Corresponding PSF region
        psf_start = vol_start - start
        psf_stop  = psf_start + (vol_stop - vol_start)

        vol_slices.append(slice(vol_start, vol_stop))
        psf_slices.append(slice(psf_start, psf_stop))

    return tuple(vol_slices), tuple(psf_slices)


def process_block(block_slices, points_arrays, FZ, FR, FC, psf_fft, psf_crop_slices, psf, volume_shape, batch_size=32):
    block_points = points_overlapping_block(points_arrays, block_slices, psf.shape)
    n_points = len(block_points['frame'])
    block_shape = tuple(s.stop - s.start for s in block_slices)
    result = np.zeros(block_shape, dtype=np.float32)

    for i in range(0, n_points, batch_size):
        idx = slice(i, min(i + batch_size, n_points))
        
        # batch phase ramps: (batch, nz, ny, nx)
        phase_ramps = np.exp(
            -1j * 2 * np.pi * (
                FZ[None] * block_points['frame_shift'][idx, None, None, None] +
                FR[None] * block_points['row_shift'][idx, None, None, None] +
                FC[None] * block_points['column_shift'][idx, None, None, None]
            )
        )  # (batch, 40, 220, 220)

        # batched ifftn
        all_shifted = np.real(
            np.fft.ifftn(psf_fft[None] * phase_ramps, axes=(1, 2, 3))
        )  # (batch, 40, 220, 220)

        # insert each point in the batch
        for j, pt_idx in enumerate(range(i, min(i + batch_size, n_points))):
            shifted_psf = all_shifted[j][psf_crop_slices]
            vol_slices, psf_slices = compute_insertion_slices(
                row=int(block_points['row'][pt_idx]),
                col=int(block_points['column'][pt_idx]),
                z=int(block_points['frame'][pt_idx]),
                psf_shape=psf.shape,
                volume_shape=volume_shape
            )
            local_slices = tuple(
                slice(s.start - b.start, s.stop - b.start)
                for s, b in zip(vol_slices, block_slices)
            )
            clipped_local = []
            clipped_psf = []
            for ls, ps, bs in zip(local_slices, psf_slices, block_shape):
                l_start = max(ls.start, 0)
                l_stop  = min(ls.stop, bs)
                p_start = ps.start + (l_start - ls.start)
                p_stop  = p_start + (l_stop - l_start)
                if l_stop <= l_start:
                    break
                clipped_local.append(slice(l_start, l_stop))
                clipped_psf.append(slice(p_start, p_stop))
            else:
                result[tuple(clipped_local)] += shifted_psf[tuple(clipped_psf)]

    return result


def build_tile_point_im(df, FZ, FR, FC, psf_fft, psf_crop_slices, psf, volume_shape, block_shape=(40, 700, 700)):
    # Extract once, outside the loop
    blocks = []
    for z in range(0, volume_shape[0], block_shape[0]):
        for r in range(0, volume_shape[1], block_shape[1]):
            for c in range(0, volume_shape[2], block_shape[2]):
                slices = (
                    slice(z, min(z + block_shape[0], volume_shape[0])),
                    slice(r, min(r + block_shape[1], volume_shape[1])),
                    slice(c, min(c + block_shape[2], volume_shape[2])),
                )
                blocks.append(slices)
    points_arrays = {
        'frame':        df.frame.values,
        'row':          df.row.values,
        'column':       df.column.values,
        'frame_shift':  df.frame_shift.values,
        'row_shift':    df.row_shift.values,
        'column_shift': df.column_shift.values,
    }


    delayed_blocks = [
        delayed(process_block)(b, points_arrays, FZ, FR, FC, psf_fft, psf_crop_slices, psf, volume_shape)
        for b in blocks
    ]
    

    dask_blocks = [
        da.from_delayed(d, shape=tuple(s.stop - s.start for s in b), dtype=np.float32)
        for d, b in zip(delayed_blocks, blocks)
    ]

    # Reassemble — stack along rows then columns
    n_z = len(range(0, volume_shape[0], block_shape[0]))
    n_r = len(range(0, volume_shape[1], block_shape[1]))
    n_c = len(range(0, volume_shape[2], block_shape[2]))

    print(f"{blocks=}")
    print(f"{len(blocks)=}, {len(dask_blocks)=}, {n_z=}, {n_r=}, {n_c=}, {n_z*n_r*n_c=}")
    grid = []
    idx = 0
    for z in range(n_z):
        row_grid = []
        for r in range(n_r):
            col_grid = []
            for c in range(n_c):
                col_grid.append(dask_blocks[idx])
                idx += 1
            row_grid.append(col_grid)
        grid.append(row_grid)
    return da.block(grid)