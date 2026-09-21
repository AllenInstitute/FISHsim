"""
Physical imaging models: dye photophysics and camera noise chain.

Signal chain:
    fluorophore emission
        → DyeSimulator  (PSF-weighted photon counts)
        → CameraSimulator (shot noise, dark current, read noise, gain, bias)
        → uint16 image
"""

import threading
import numpy as np
import dask.array as da
from numpy.typing import ArrayLike
import json
from pathlib import Path

_rng_local = threading.local()

def _rng() -> np.random.Generator:
    """Return a thread-local RNG (releases GIL, unlike legacy np.random)."""
    if not hasattr(_rng_local, "gen"):
        _rng_local.gen = np.random.default_rng()
    return _rng_local.gen


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
    return _rng().poisson(mean_dark, size=image_shape)


# ---------------------------------------------------------------------------
# Dye / fluorophore model
# ---------------------------------------------------------------------------

class DyeSimulator:
    """Convert a PSF-convolved intensity image to simulated photon counts.

    Models fluorophore emission as a Poisson process.  The effective photon
    emission rate is set by *photons_per_emitter_s* (photons/emitter/second),
    which represents the laser-driven emission rate under typical FISH
    illumination.  This replaces the unphysical QY/lifetime formulation, which
    assumes one excitation per lifetime cycle (saturation limit: ~5×10^8 Hz for
    Cy3) — roughly 10,000× higher than real FISH conditions and causes full
    camera-well saturation.

    Args:
        lifetime: fluorescence lifetime in seconds (retained for reference)
        quantum_yield: probability of photon emission per excitation (retained
            for reference; not used in the photon-count calculation when
            photons_per_emitter_s is set)
        photons_per_emitter_s: effective photon emission rate per emitter per
            second at the focal plane under typical FISH illumination.
            Multiply by exposure_time to get total expected photons per emitter
            per frame.  Default 2.5e5 gives ~12,500 photons per emitter at
            50 ms exposure — appropriate for a bright FISH dye with a Kinetix
            camera in sensitivity mode (1000 e well depth, 0.25 e/count gain).
    """

    def __init__(self, lifetime: float, quantum_yield: float,
                 photons_per_emitter_s: float = 2.5e5):
        self.lifetime = lifetime
        self.quantum_yield = quantum_yield
        self.photons_per_emitter_s = photons_per_emitter_s

    def psf_to_photon_distribution(
        self,
        psf: ArrayLike,
        exposure_time: float,
    ) -> ArrayLike:
        """Convert a (normalised) PSF image to expected photon counts and draw Poisson samples.

        Args:
            psf: PSF-convolved emitter density image (normalised, sum ≈ N_emitters).
                 Can be a numpy array or a dask array.
            exposure_time: effective exposure time in seconds. Multiply the
                 physical exposure by a brightness_scale multiplier before
                 passing here to control overall signal level.

        Returns:
            Poisson-sampled photon counts with the same shape as *psf*.
        """
        expected_photons = psf * (self.photons_per_emitter_s * exposure_time)

        if isinstance(psf, np.ndarray):
            return np.random.poisson(expected_photons.astype(np.float64))
        else:
            return da.map_blocks(np.random.poisson, expected_photons, dtype=np.float64)


# Convenience instances for common fluorophores.
# photons_per_emitter_s = 2.5e5 → 12,500 photons/emitter at 50 ms exposure.
# At PSF peak (~4% of energy) with Kinetix sensitivity mode (QE≈0.95,
# well=1000 e, gain=0.25 e/count): ~475 electrons → ~2000 ADU, above the
# default mermake_threshold=1800 without saturating the well.
CY3 = DyeSimulator(lifetime=2.0e-9, quantum_yield=0.3, photons_per_emitter_s=2.5e5)
CY5 = DyeSimulator(lifetime=1.0e-9, quantum_yield=0.28, photons_per_emitter_s=2.5e5)
AF750 = DyeSimulator(lifetime=0.7e-9, quantum_yield=0.12, photons_per_emitter_s=2.5e5)


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
        spec_file: Path | None = None,
        qe_file: Path | None = None,
        mode: str | None = None,
        QE: dict | None = None,
        gain: float | None = None,
        bias: float | None = None,
        dark_current: float | None = None,
        read_noise: float | None = None,
        well_depth: int | None = None,
    ):
        if spec_file and mode:
            with open(spec_file, 'r') as f:
                specs = json.load(f)
            self.gain = specs.get("conversion_factor", {}).get(mode)
            self.bias = specs.get("dark_offset", {}).get(mode)
            self.dark_current = specs.get("dark_current", {}).get(mode)
            self.read_noise = specs.get("read_noise", {}).get(mode)
            self.well_depth = specs.get("full_well_capacity", {}).get(mode)
        else:
            self.gain = gain
            self.bias = bias
            self.dark_current = dark_current
            self.read_noise = read_noise
            self.well_depth = well_depth
        if qe_file and mode:
            with open(qe_file, 'r') as f:
                self.QE = {
                    int(line.split(",")[0]): float(line.split(",")[1].strip())
                    for line in f.readlines()[1:]
                }
        else:
            self.QE = QE


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
        qe = self.QE.get(round(wavelength), self.QE.get(wavelength, 1.0))

        rng = _rng()

        # 1. Shot noise
        photon_noisy = rng.poisson(photon_image.astype(np.float32))

        # 2. Quantum efficiency
        electrons = photon_noisy * qe

        # 3. Dark current
        electrons = electrons + make_dark_current_image(
            self.dark_current, exposure_time, photon_image.shape
        )

        # 4. Read noise
        if self.read_noise > 0:
            electrons = electrons + rng.normal(0, self.read_noise, size=photon_image.shape)

        # 5. Gain and bias
        counts = electrons / self.gain + self.bias

        # 6. Clip to well depth and cast
        max_count = self.well_depth / self.gain + self.bias
        return np.clip(counts, 0, max_count).astype(np.uint16)
