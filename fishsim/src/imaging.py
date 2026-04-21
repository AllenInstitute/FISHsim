"""
Physical imaging models: dye photophysics and camera noise chain.

Signal chain:
    fluorophore emission
        → DyeSimulator  (PSF-weighted photon counts)
        → CameraSimulator (shot noise, dark current, read noise, gain, bias)
        → uint16 image
"""

import numpy as np
import dask.array as da
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_shot_noise(image: ArrayLike, photon_conversion: float = 0.24) -> ArrayLike:
    """Apply Poisson shot noise scaled by a photons-per-count conversion factor."""
    original_dtype = image.dtype
    image = np.asarray(image, dtype=np.float32)
    image_photons = image * photon_conversion
    noisy = np.random.poisson(image_photons)
    return (noisy / photon_conversion).astype(original_dtype)


def make_dark_current_image(
    dark_current: float, exposure_time: float, image_shape: tuple
) -> np.ndarray:
    """Draw dark-current electrons as a Poisson process.

    Args:
        dark_current: electrons / pixel / second
        exposure_time: seconds
        image_shape: (z, y, x) or (y, x) shape of the output array
    """
    mean_dark = dark_current * exposure_time
    return np.random.poisson(mean_dark, size=image_shape)


# ---------------------------------------------------------------------------
# Dye / fluorophore model
# ---------------------------------------------------------------------------

class DyeSimulator:
    """Convert a PSF-convolved intensity image to simulated photon counts.

    Models fluorophore emission as a Poisson process parameterised by
    fluorescence lifetime and quantum yield, ignoring excitation cross-section
    (calibrated out via the brightness_scale multiplier in render_images.py).

    Args:
        lifetime: fluorescence lifetime in seconds (e.g. 2e-9 for Cy3)
        quantum_yield: probability of photon emission per excitation event
    """

    def __init__(self, lifetime: float, quantum_yield: float):
        self.lifetime = lifetime
        self.quantum_yield = quantum_yield

    def psf_to_photon_distribution(
        self,
        psf: ArrayLike,
        exposure_time: float,
    ) -> ArrayLike:
        """Convert a (normalised) PSF image to expected photon counts and draw Poisson samples.

        Args:
            psf: PSF-convolved intensity image in arbitrary units.
                 Can be a numpy array or a dask array.
            exposure_time: effective exposure time in seconds. Multiply the
                 physical exposure by a brightness_scale multiplier before
                 passing here to control overall signal level.

        Returns:
            Poisson-sampled photon counts with the same shape as *psf*.
        """
        expected_photons = psf * self.quantum_yield * exposure_time / self.lifetime

        if isinstance(psf, np.ndarray):
            return np.random.poisson(expected_photons.astype(np.float64))
        else:
            return da.map_blocks(np.random.poisson, expected_photons, dtype=np.float64)


# Convenience instances for common fluorophores.
# Import and use these, or construct your own DyeSimulator for different dyes.
CY3 = DyeSimulator(lifetime=2.0e-9, quantum_yield=0.3)
CY5 = DyeSimulator(lifetime=1.0e-9, quantum_yield=0.28)
AF750 = DyeSimulator(lifetime=0.7e-9, quantum_yield=0.12)


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------

class CameraSimulator:
    """Simulate a sCMOS / EMCCD camera noise chain.

    Converts a photon image to a 16-bit digital count image by applying, in order:
        1. Shot noise (Poisson)
        2. Quantum efficiency (photons → electrons)
        3. Dark current (Poisson)
        4. Read noise (Gaussian)
        5. Gain and bias (electrons → counts)
        6. Well-depth clipping

    Args:
        QE: dict mapping wavelength (int, nm) to quantum efficiency (0–1),
            e.g. {561: 0.95, 650: 0.89}
        gain: electrons per count (ADU)
        bias: offset counts added to every pixel
        dark_current: electrons / pixel / second
        read_noise: electrons RMS
        well_depth: maximum electron count before saturation
    """

    def __init__(
        self,
        QE: dict,
        gain: float,
        bias: float,
        dark_current: float,
        read_noise: float,
        well_depth: int,
    ):
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
        exposure_time: float,
    ) -> np.ndarray:
        """Apply the full camera noise chain to a photon image.

        Args:
            photon_image: array of expected photon counts per pixel
            wavelength: illumination wavelength in nm, used to look up QE
            exposure_time: exposure time in seconds, used for dark current

        Returns:
            uint16 digital count image
        """
        qe = self.QE.get(wavelength, 1.0)

        # 1. Shot noise
        photon_noisy = np.random.poisson(photon_image.astype(np.float64))

        # 2. Quantum efficiency
        electrons = photon_noisy * qe

        # 3. Dark current
        electrons = electrons + make_dark_current_image(
            self.dark_current, exposure_time, photon_image.shape
        )

        # 4. Read noise
        if self.read_noise > 0:
            electrons = electrons + np.random.normal(0, self.read_noise, size=photon_image.shape)

        # 5. Gain and bias
        counts = electrons / self.gain + self.bias

        # 6. Clip to well depth and cast
        max_count = self.well_depth / self.gain + self.bias
        return np.clip(counts, 0, max_count).astype(np.uint16)
