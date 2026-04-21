# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

This is a **fork** of the original [FISHsim](https://github.com/Roth-Lab/FISHsim) repo. The goal is to extend it with a 3D merFISH simulation pipeline to explore how primary optics (numerical aperture, magnification) and signal-to-background ratios affect transcript decoding quality in human brain tissue. The original `fishsim` package should remain largely unaltered; new 3D work lives in standalone scripts and new modules.

The pilot PSF was measured on a 60x 1.4 NA Nikon objective with 6.5 µm pixels and 0.4 µm axial step size. The project should support both measured and simulated PSFs (including controlled spherical aberration / field curvature).

### Intended two-stage pipeline

**Stage 1 — generate points and cells** (`fishsim/scripts/generate_scene.py` or similar):
- Inputs: imaging volume (µm), cell size, codebook path, spot brightness variation (0–1 scalar)
- Cells are placed as densely as the volume allows (filling the volume)
- Outputs: point cloud CSV (spot positions + gene identity + cell ID) and cell geometry table

**Stage 2 — render images** (`fishsim/scripts/render_images.py` or similar):
- Inputs: point/cell tables from Stage 1, objective lens / measured PSF, camera object, z step size, exposure time, brightness multiplier
- Signal level is controlled by exposure time × multiplier and is affected by the objective NA
- Output format matches `Z:\MERFISHp\12_04_2025_JenieSample_re`: folders H1–H9 (imaging rounds), each with numbered tile subfolders containing a 3D zarr array with axes `[channel*z, y, x]` cycling through Cy3, Cy5, AlexaFluor750 (DAPI omitted — true cell segmentation is provided instead)

## Setup

```bash
conda env create -f environment.yml
conda activate fishsim
pip install -e .
```

`fishsim/config.py` resolves paths via a `.env` file (searched by `python-dotenv`) or `~/fishsim.env`. The relevant env vars are `RESOURCES_DIR`, `CODEBOOK_DIR`, and `RESULTS_DIR`. If none are found, defaults relative to the package root are used.

## Running

**CLI (original 2D pipeline, unchanged from upstream):**
```bash
fishsim run_merfish --config-file fishsim/resources/configs/config.yml --output-dir-name my_experiment
```

**3D sandbox (in-progress entry point):**
Run `fishsim/src/make_3d_im.py:main()` directly. It reads `fishsim/resources/config_3d_sandbox.yml` and loads a PSF from `Y:/MERFISHp/12_04_2025_JenieSample/psf_final.pkl`.

**Tests:**
```bash
cd fishsim/src && pytest
```

## Architecture

### Original 2D pipeline (upstream, largely unchanged)

The 2D `Simulator` in `src/simulation.py` produces merFISH TIFF images by:
1. Distributing emitters uniformly or inside non-overlapping ellipsoid cells
2. Assigning gene barcodes from a codebook; simulating bit-drop / bit-add errors
3. Convolving emitter positions with a PSF via `SparseMatrix3D` (`src/sparse.py`)
4. Applying camera noise (shot noise, dark current, read noise, gain, bias) to yield 16-bit TIFF output

### 3D extension (this fork)

New and modified files:

| File | Role |
|---|---|
| `src/sim3d.py` | **3D simulator core.** Extends the upstream `Simulator` with `generate_point_cloud` (separates point generation from rendering). Also defines `DyeSimulator` (PSF → photon counts via lifetime/quantum yield), `CameraSimulator`, and block-decomposed dask PSF convolution (`build_tile_point_im` / `process_block`). |
| `src/make_3d_im.py` | Current entry point for the 3D pipeline: generates a point cloud, renders it into a dask-backed 3D image, inserts cell background masks, and applies camera noise. |
| `src/point_cloud_background.py` | Generates cell background volumes from emitter point clouds using convex-hull masks and FFT-based density estimation. **Axis convention (fixed — do not change):** `voxel_size=[vz,vy,vx]`, `volume_shape=[nz,ny,nx]`, `cell.emitters` in `(x,y,z)` physical units, output arrays shaped `(nz,ny,nx)`. |

Upstream files referenced by the 3D work:

| File | Role |
|---|---|
| `src/generate_emitters.py` | `random_emitter_position` and `cell_emitter_position` — uniform or cell-constrained emitter placement |
| `src/cells.py` | `EllipsoidCell` — wraps `Ellipsoid`, samples emitters inside the volume with optional nucleus sub-region |
| `src/ellipsoid.py` | `Ellipsoid` — matrix representation with random-point sampling, xy-projection, and algebraic overlap detection (Ghossein et al.) |
| `src/utils.py` | Shared helpers: `trunc_norm`, `glob_background`, `make_gaussian_2d` |
| `config.py` | Resolves `RESOURCES_DIR`, `CODEBOOK_DIR`, `RESULTS_DIR` from env or defaults |

### Configuration YAML structure

Config files live in `fishsim/resources/` and `fishsim/resources/configs/`. Key sections:

- `simulation` — `emitter_count`, `image_size`, `tile_count`, `bitdrop_probability`, `bitadd_probability`, `scr` (signal-to-background ratio), `photon_count` (scalar or 8-element list for per-round variation), `cells.count` / `cells.axes`
- `cameras.Prime95B` — `qe` (per wavelength), `gain`, `bias`, `dark_current`, `read_noise`, `well_depth`, `exposure_times`
- `optics.NikonTi2` — `magnification`
- `data` — paths to PSF (`.mat` or pickle), codebook CSV, data organisation CSV

`config_3d_sandbox.yml` is the config for the 3D sandbox work.

### Codebook CSV format

3-row header (skipped on `pd.read_csv(..., skiprows=3)`). Required columns: `name`, `numeric_id`, `barcode`. Optional `distribution` column specifies per-gene emitter counts; when present it overrides `emitter_count`.

### Data organisation CSV

Maps each `bitNumber` to an `imagingRound` and `color` channel. Used by `Simulator.map_barcode` to reorder the gene barcode string into the physical channel ordering reflected in image filenames.

### 3D pipeline conventions

Physical coordinates throughout are in **microns**. Default pixel sizes: `[0.4, 0.1084333, 0.1084333]` µm for `[z, y, x]`. PSF convolution is block-decomposed with dask (`block_shape=(40, 700, 700)`) for memory efficiency. `DyeSimulator.psf_to_photon_distribution` converts PSF values to photon counts using `expected_photons = psf * quantum_yield * exposure_time / lifetime`.
