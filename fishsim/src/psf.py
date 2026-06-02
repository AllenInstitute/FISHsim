"""
Theoretical 3D PSF model using scalar diffraction theory.

The PSF is parameterised by objective NA, emission wavelength, immersion
medium refractive index, and an optional Zernike wavefront aberration.
Aberrations are specified as a dict of Noll-index → coefficient in waves.

The model is decoupled from any specific voxel grid: construct a
``TheoreticalPSF`` instance with the optical parameters, then call
``.sample(voxel_size, shape)`` to realise it on any grid.  The returned
array has the same format as a measured PSF and can be passed directly to
``sim3d.apodize_psf_tukey`` / ``render_images.prepare_psf_fft``.

Common Noll indices
-------------------
 1  piston            (constant offset; no effect on intensity PSF)
 4  defocus
 5  oblique astigmatism
 6  vertical astigmatism
 7  vertical coma
 8  horizontal coma
11  primary spherical aberration

Indexing convention note
------------------------
Some optical engineering texts and software (Zemax, Wyant) use "Fringe
Zernike" ordering, where primary spherical aberration is Z9.  This is the
same polynomial (n=4, m=0); only the index differs.  This module uses Noll
(1976) ordering exclusively — use Noll 11 for primary spherical aberration.

References
----------
Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence.
    JOSA, 66(3), 207–211.
Born, M. & Wolf, E. (1999). Principles of Optics (7th ed.), §8.8.
"""

from __future__ import annotations

import numpy as np
from math import factorial
from scipy.fft import fft2, fftshift, ifftshift, next_fast_len
from scipy.ndimage import rotate as _ndrotate, gaussian_filter as _gauss


# ---------------------------------------------------------------------------
# Zernike polynomial utilities
# ---------------------------------------------------------------------------

def _noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index *j* (1-based) to (radial order n, azimuthal frequency m).

    Positive m → cos term; negative m → sin term; m=0 → rotationally symmetric.
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Evaluate the radial Zernike polynomial R_n^|m|(rho)."""
    m = abs(m)
    result = np.zeros_like(rho, dtype=float)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s * factorial(n - s)
             // (factorial(s)
                 * factorial((n + m) // 2 - s)
                 * factorial((n - m) // 2 - s)))
        result = result + c * rho ** (n - 2 * s)
    return result


def zernike_noll(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the Noll-indexed Zernike polynomial Z_j at (rho, theta).

    Polynomials are normalised so that the integral of Z_i * Z_j over the
    unit disk equals π·δ_ij (standard optics convention).  A coefficient of
    1.0 wave therefore produces approximately 1 wave of RMS wavefront error.

    Args:
        j:     Noll index (1-based integer)
        rho:   radial pupil coordinate array, 0 <= rho (values > 1 are outside
               the aperture and should be masked before use)
        theta: azimuthal coordinate array in radians

    Returns:
        Array of the same shape as *rho* / *theta*.
    """
    n, m = _noll_to_nm(j)
    R = _zernike_radial(n, m, rho)
    if m == 0:
        return np.sqrt(n + 1) * R
    elif m > 0:
        return np.sqrt(2 * (n + 1)) * R * np.cos(m * theta)
    else:
        return np.sqrt(2 * (n + 1)) * R * np.sin(-m * theta)


# ---------------------------------------------------------------------------
# Measured PSF perturbation (image-space)
# ---------------------------------------------------------------------------

def perturb_psf_array(
    psf: np.ndarray,
    rng=None,
    max_lateral_shift: float = 0.4,
    max_axial_shift: float = 0.15,
    max_anisotropic_sigma: float = 0.25,
) -> np.ndarray:
    """Apply small random perturbations to a measured PSF array.

    Simulates the field-dependent variation you would see across a real FOV
    (coma, astigmatism, slight defocus) by working entirely in image space:
    a sub-pixel shift breaks rotational symmetry the way coma does, and
    independent per-axis blurring mimics astigmatism / field curvature.  For
    the goal of avoiding clone spots the result is indistinguishable from a
    rigorous pupil-phase perturbation.

    Args:
        psf:                  3-D float array ``(nz, ny, nx)`` normalised to
                              sum = 1 (the format used throughout this project).
        rng:                  ``numpy.random.Generator``, integer seed, or
                              ``None`` (fresh default RNG).
        max_lateral_shift:    Controls the maximum random in-plane rotation
                              angle in degrees (scaled as ``value * 45``).
                              0.4 → up to ±18° rotation of each lateral plane.
        max_axial_shift:      Maximum Gaussian sigma added along z, in pixels.
                              Kept small because the axial PSF is already broad.
        max_anisotropic_sigma: Maximum Gaussian sigma added independently to
                              y and x axes, in pixels.  Creates elliptical
                              broadening (astigmatism / field curvature).

    Returns:
        ``float32`` array of the same shape, renormalised to sum = 1.
    """
    rng = np.random.default_rng(rng)

    # Random in-plane rotation — changes the orientation of any asymmetry
    # already present in the measured PSF (e.g. astigmatism axis).
    # Applied independently per z-plane so axial structure is preserved.
    angle = float(rng.uniform(-max_lateral_shift * 45, max_lateral_shift * 45))
    out = np.stack([
        _ndrotate(plane, angle, reshape=False, order=3, mode="constant", cval=0.0)
        for plane in psf.astype(np.float64)
    ])

    # Independent per-axis blur — creates elliptical broadening (astigmatism /
    # field curvature).  Axial sigma kept smaller because the z-PSF is already
    # much broader in pixel terms.
    sz = float(rng.uniform(0.0, max_axial_shift))
    sy = float(rng.uniform(0.0, max_anisotropic_sigma))
    sx = float(rng.uniform(0.0, max_anisotropic_sigma))
    out = _gauss(out, sigma=[sz, sy, sx])

    # Clip and renormalise
    out = np.clip(out, 0.0, None)
    total = out.sum()
    if total > 0:
        out /= total
    return out.astype(np.float32)


def make_psf_pool(
    psf: np.ndarray,
    n: int,
    rng=None,
    **kwargs,
) -> list:
    """Build a pool of *n* perturbed copies of a measured PSF array.

    A single RNG is shared so the whole pool is reproducible from one seed.
    Draw from the pool per spot with ``pool[rng.integers(n)]``.

    Args:
        psf:    3-D measured PSF array ``(nz, ny, nx)``.
        n:      Pool size.  20–50 is usually enough.
        rng:    ``numpy.random.Generator``, integer seed, or ``None``.
        **kwargs: Forwarded to :func:`perturb_psf_array`.

    Returns:
        List of *n* ``float32`` arrays of the same shape as *psf*.
    """
    rng = np.random.default_rng(rng)
    return [perturb_psf_array(psf, rng=rng, **kwargs) for _ in range(n)]


# ---------------------------------------------------------------------------
# PSF model
# ---------------------------------------------------------------------------

class TheoreticalPSF:
    """Scalar-diffraction 3D PSF model with Zernike wavefront aberrations.

    The model is defined in continuous optical space.  Call ``.sample()`` to
    realise it on a specific voxel grid.

    Args:
        NA:               numerical aperture of the objective
        wavelength_nm:    fluorophore emission wavelength in nanometres
        refractive_index: refractive index of the immersion medium
                          (default: 1.515, oil immersion)
        zernike_coeffs:   dict mapping Noll index → aberration coefficient in
                          waves.  E.g. ``{11: 0.5}`` adds 0.5 waves of primary
                          spherical aberration.  Positive spherical aberration
                          (positive Z11 coefficient) is consistent with the
                          convention where oil-immersion objectives imaging into
                          an aqueous sample exhibit positive spherical aberration.

    Notes:
        The Zernike coefficients represent the wavefront *error* relative to a
        perfect spherical wavefront converging to the nominal focus.  A
        coefficient of 1.0 corresponds to approximately 1 wave of RMS error for
        that mode.

        Refractive-index mismatch (e.g. oil → water) introduces spherical
        aberration that grows linearly with depth.  As a first approximation,
        use Z11 to model the dominant spherical aberration term.  For a full
        treatment accounting for depth-dependent aberration across the axial
        range, consider a Gibson–Lanni-style model in future.
    """

    def __init__(
        self,
        NA: float,
        wavelength_nm: float,
        refractive_index: float = 1.515,
        zernike_coeffs: dict | None = None,
    ):
        if NA <= 0 or NA >= refractive_index:
            raise ValueError(
                f"NA must satisfy 0 < NA < n (refractive_index={refractive_index}); "
                f"got NA={NA}"
            )
        self.NA = NA
        self.wavelength_nm = wavelength_nm
        self.refractive_index = refractive_index
        self.zernike_coeffs = zernike_coeffs or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _auto_shape(
        self,
        voxel_size: list | tuple,
        threshold: float = 0.01,
        pupil_size: int = 256,
    ) -> list[int]:
        """Return the minimum odd-sized [nz, ny, nx] that contains the PSF peak
        down to *threshold* times the peak intensity.

        Axial extent is found by evaluating the on-axis coherent sum
        (sum of aperture * exp(i*phase)) per z step — no FFT required.
        Lateral extent is found from the centre row of the z=0 FFT plane.

        Args:
            voxel_size: [dz, dy, dx] voxel size in µm.
            threshold:  intensity fraction below which a pixel is considered
                        outside the PSF.  Default 0.01 (1 % of peak, i.e. 99 %
                        drop).
            pupil_size: pupil plane resolution; see :meth:`sample`.

        Returns:
            ``[nz, ny, nx]`` list of odd integers.
        """
        dz, dy, dx = voxel_size
        wl = self.wavelength_nm / 1000.0
        n = self.refractive_index
        N = pupil_size

        coords = np.arange(-N // 2, N // 2) / (N // 2)
        U, V = np.meshgrid(coords, coords)
        RHO = np.sqrt(U ** 2 + V ** 2)
        THETA = np.arctan2(V, U)
        aperture = RHO <= 1.0

        W = np.zeros((N, N), dtype=float)
        for j, coeff in self.zernike_coeffs.items():
            W += coeff * zernike_noll(j, RHO, THETA)
        zernike_phase = 2.0 * np.pi * W * aperture

        # ---- Axial: on-axis coherent sum, no FFT ----------------------------
        # Scan from z=0 outward in dz steps; the PSF is symmetric about z=0.
        max_half_z = max(50.0, 10.0 * self.axial_resolution_um())
        z_vals = np.arange(0.0, max_half_z + dz, dz)
        axial = np.empty(len(z_vals))
        for i, z in enumerate(z_vals):
            arg = np.clip(1.0 - (self.NA * RHO / n) ** 2, 0.0, None)
            prop = (2.0 * np.pi * n / wl) * z * np.sqrt(arg) * aperture
            axial[i] = abs(np.sum(aperture * np.exp(1j * (zernike_phase + prop)))) ** 2
        below_ax = np.where(axial < threshold * axial[0])[0]
        half_nz = int(below_ax[0]) if len(below_ax) else len(z_vals)

        # ---- Lateral: centre row of the z=0 FFT plane -----------------------
        dl = min(dx, dy)
        M_min = int(np.ceil(N * wl / (2.0 * self.NA * dl)))
        M = next_fast_len(max(N, M_min))
        actual_dl = N * wl / (2.0 * self.NA * M)  # µm per FFT pixel

        pad = (M - N) // 2
        pupil0 = aperture.astype(complex) * np.exp(1j * zernike_phase)
        padded = np.zeros((M, M), dtype=complex)
        padded[pad: pad + N, pad: pad + N] = pupil0
        psf_plane = np.abs(fftshift(fft2(ifftshift(padded)))) ** 2

        half_row = psf_plane[M // 2, M // 2:]  # radial profile from centre
        peak_lat = half_row[0]
        below_lat = np.where(half_row < threshold * peak_lat)[0]
        half_fft_pix = int(below_lat[0]) if len(below_lat) else len(half_row)

        half_ny = int(np.ceil(half_fft_pix * actual_dl / dy))
        half_nx = int(np.ceil(half_fft_pix * actual_dl / dx))

        return [2 * half_nz + 1, 2 * half_ny + 1, 2 * half_nx + 1]

    def sample(
        self,
        voxel_size: list | tuple,
        shape: list | tuple | None = None,
        pupil_size: int = 256,
        threshold: float = 0.01,
    ) -> np.ndarray:
        """Realise the PSF on a discrete voxel grid.

        Args:
            voxel_size: [dz, dy, dx] voxel size in µm.
            shape:      [nz, ny, nx] output array size in pixels, or ``None``
                        to auto-compute the tightest shape that keeps all voxels
                        above *threshold* times the peak intensity.
            pupil_size: number of samples across the pupil diameter.
                        Higher values give a more accurate PSF at the cost of
                        speed.  256 is sufficient for most objectives.
            threshold:  only used when ``shape=None``.  Intensity fraction
                        below which a voxel is considered outside the PSF.
                        Default 0.01 (99 % drop from peak).

        Returns:
            float32 array of shape (nz, ny, nx).  Values are non-negative; the
            array is normalised to sum = 1 so it can be used directly as a
            convolution kernel.

        Notes on pixel-size accuracy
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The FFT-based computation produces a lateral pixel size of
        ``wavelength * pupil_size / (2 * NA * M)`` where M is the next
        FFT-efficient integer >= the minimum required padding.  This is always
        ≤ the requested *dx*, so the output is at worst very slightly
        over-sampled before cropping.  The discrepancy is typically < 0.5 %.
        """
        if shape is None:
            shape = self._auto_shape(voxel_size, threshold=threshold, pupil_size=pupil_size)
        nz, ny, nx = shape
        dz, dy, dx = voxel_size
        wl = self.wavelength_nm / 1000.0  # convert nm → µm
        n = self.refractive_index
        N = pupil_size

        # ------------------------------------------------------------------
        # Pupil plane grid: normalised coordinates ρ ∈ [0, 1] where ρ=1
        # corresponds to the edge of the aperture (spatial freq = NA/λ).
        # ------------------------------------------------------------------
        coords = np.arange(-N // 2, N // 2) / (N // 2)  # [-1, 1)
        U, V = np.meshgrid(coords, coords)
        RHO = np.sqrt(U ** 2 + V ** 2)
        THETA = np.arctan2(V, U)
        aperture = RHO <= 1.0

        # ------------------------------------------------------------------
        # Zernike wavefront aberration (in radians)
        # ------------------------------------------------------------------
        W = np.zeros((N, N), dtype=float)
        for j, coeff in self.zernike_coeffs.items():
            W += coeff * zernike_noll(j, RHO, THETA)
        zernike_phase = 2.0 * np.pi * W * aperture  # radians

        # ------------------------------------------------------------------
        # Determine FFT padding needed to achieve the desired lateral pixel
        # size.  The natural pixel size of an N×N pupil FFT is λ/(2·NA);
        # zero-padding to M points shrinks the pixel to λN/(2·NA·M).
        #
        #   M_min = ceil( N · λ / (2 · NA · min(dx, dy)) )
        # ------------------------------------------------------------------
        dl = min(dx, dy)
        M_min = int(np.ceil(N * wl / (2.0 * self.NA * dl)))
        M = next_fast_len(max(N, M_min))

        # Actual lateral pixel size after FFT (µm)
        actual_dl = N * wl / (2.0 * self.NA * M)

        # ------------------------------------------------------------------
        # Axial positions centred at z = 0 (sharpest plane)
        # ------------------------------------------------------------------
        z_positions = (np.arange(nz) - (nz - 1) / 2.0) * dz

        # ------------------------------------------------------------------
        # Compute one PSF plane per z-position
        # ------------------------------------------------------------------
        psf_lateral = np.zeros((nz, M, M), dtype=np.float32)
        pad = (M - N) // 2

        for iz, z in enumerate(z_positions):
            # Propagation phase: kz(ρ) · z, where
            #   kz(ρ) = (2π n / λ) · √(1 − (NA · ρ / n)²)
            # We clip the argument of sqrt to [0, 1] — it is exactly 0 at
            # ρ = n/NA which is outside the aperture (NA < n always).
            arg = np.clip(1.0 - (self.NA * RHO / n) ** 2, 0.0, None)
            propagation_phase = (2.0 * np.pi * n / wl) * z * np.sqrt(arg) * aperture

            total_phase = zernike_phase + propagation_phase
            pupil = aperture.astype(complex) * np.exp(1j * total_phase)

            # Zero-pad into centre of M×M array
            padded = np.zeros((M, M), dtype=complex)
            padded[pad: pad + N, pad: pad + N] = pupil

            # Fourier transform: ifftshift centres the pupil on the FFT grid;
            # fftshift centres the resulting PSF.
            field = fftshift(fft2(ifftshift(padded)))
            psf_lateral[iz] = np.abs(field).astype(np.float32) ** 2

        # ------------------------------------------------------------------
        # Crop the lateral axes to the requested (ny, nx)
        # ------------------------------------------------------------------
        cy, cx = M // 2, M // 2
        r0 = cy - ny // 2
        c0 = cx - nx // 2
        r1, c1 = r0 + ny, c0 + nx

        if r0 >= 0 and c0 >= 0 and r1 <= M and c1 <= M:
            psf_out = psf_lateral[:, r0:r1, c0:c1].copy()
        else:
            # Requested shape is larger than the computed lateral extent —
            # embed the computed PSF in a zero-padded array.
            psf_out = np.zeros((nz, ny, nx), dtype=np.float32)
            yr = max(0, -r0)
            xr = max(0, -c0)
            yr_end = yr + min(M, ny) - max(0, r1 - M)
            xr_end = xr + min(M, nx) - max(0, c1 - M)
            src_r0 = max(r0, 0)
            src_c0 = max(c0, 0)
            psf_out[:, yr:yr_end, xr:xr_end] = psf_lateral[
                :, src_r0: src_r0 + (yr_end - yr), src_c0: src_c0 + (xr_end - xr)
            ]

        # Normalise to sum = 1
        total = psf_out.sum()
        if total > 0:
            psf_out /= total

        return psf_out

    # ------------------------------------------------------------------
    # Perturbation helpers
    # ------------------------------------------------------------------

    def perturb(
        self,
        magnitudes: dict | None = None,
        rng=None,
    ) -> "TheoreticalPSF":
        """Return a new TheoreticalPSF with small random Zernike perturbations.

        Each Noll mode listed in *magnitudes* gets an independent uniform draw
        from [-magnitude, +magnitude] added to its existing coefficient.  Call
        this once per PSF in a pool, then use :meth:`sample_pool` as a
        convenience wrapper.

        Args:
            magnitudes: ``{Noll_index: max_magnitude_in_waves}``.  Defaults to
                        low-order aberrations that give visually distinct but
                        still diffraction-limited spots::

                            {4: 0.05,   # defocus
                             5: 0.05,   # oblique astigmatism
                             6: 0.05,   # vertical astigmatism
                             7: 0.03,   # vertical coma
                             8: 0.03,   # horizontal coma
                             11: 0.02}  # primary spherical

            rng: ``numpy.random.Generator``, integer seed, or ``None``
                 (uses a fresh default RNG).

        Returns:
            New :class:`TheoreticalPSF`; *self* is unchanged.
        """
        if magnitudes is None:
            magnitudes = {4: 0.05, 5: 0.05, 6: 0.05, 7: 0.03, 8: 0.03, 11: 0.02}
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        new_coeffs = dict(self.zernike_coeffs)
        for j, mag in magnitudes.items():
            delta = float(rng.uniform(-mag, mag))
            new_coeffs[j] = new_coeffs.get(j, 0.0) + delta
        return TheoreticalPSF(
            NA=self.NA,
            wavelength_nm=self.wavelength_nm,
            refractive_index=self.refractive_index,
            zernike_coeffs=new_coeffs,
        )

    def sample_pool(
        self,
        n: int,
        voxel_size: list | tuple,
        shape: list | tuple | None = None,
        magnitudes: dict | None = None,
        rng=None,
        pupil_size: int = 256,
        threshold: float = 0.01,
    ) -> list:
        """Pre-compute a pool of *n* perturbed PSF arrays.

        Each array is an independent random perturbation of this PSF realised
        on the given voxel grid.  Build the pool once before rendering; then
        draw ``pool[rng.integers(n)]`` per spot for near-zero runtime overhead.

        Example::

            base = TheoreticalPSF(NA=1.4, wavelength_nm=670)
            pool = base.sample_pool(50, voxel_size=[0.4, 0.108, 0.108],
                                    shape=[21, 51, 51], rng=42)
            rng  = np.random.default_rng(0)
            spot_psf = pool[rng.integers(len(pool))]

        Args:
            n:          Number of distinct PSF arrays to generate.
            voxel_size: ``[dz, dy, dx]`` voxel size in µm.
            shape:      ``[nz, ny, nx]`` output array size in pixels, or
                        ``None`` to auto-compute from the base (un-perturbed)
                        PSF.  The same shape is reused for all pool members.
            magnitudes: Passed to :meth:`perturb`; ``None`` uses defaults.
            rng:        ``numpy.random.Generator``, integer seed, or ``None``.
                        A single RNG is shared across all perturbations so the
                        pool is reproducible from a single seed.
            pupil_size: Pupil plane resolution; see :meth:`sample`.
            threshold:  Only used when ``shape=None``; see :meth:`sample`.

        Returns:
            List of *n* ``float32`` arrays of shape ``(nz, ny, nx)``, each
            normalised to sum = 1.
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        if shape is None:
            shape = self._auto_shape(voxel_size, threshold=threshold, pupil_size=pupil_size)
        return [
            self.perturb(magnitudes=magnitudes, rng=rng).sample(
                voxel_size=voxel_size, shape=shape, pupil_size=pupil_size
            )
            for _ in range(n)
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def lateral_resolution_um(self) -> float:
        """Rayleigh criterion lateral resolution in µm: 0.61 λ / NA."""
        return 0.61 * self.wavelength_nm / 1000.0 / self.NA

    def axial_resolution_um(self) -> float:
        """Rayleigh criterion axial resolution in µm: 2 n λ / NA²."""
        return 2.0 * self.refractive_index * self.wavelength_nm / 1000.0 / self.NA ** 2

    def __repr__(self) -> str:
        aberr = (f", zernike={self.zernike_coeffs}" if self.zernike_coeffs else "")
        return (
            f"TheoreticalPSF(NA={self.NA}, wavelength={self.wavelength_nm}nm, "
            f"n={self.refractive_index}{aberr})"
        )
