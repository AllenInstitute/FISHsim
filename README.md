# FISHsim (Multiplexed Spots Simulator)

This is a simulation software that generates photorealistic images of fluorescent probes.

## Installation

We recommend using conda to set up a virtual environment.

```bash
conda env create -f environment.yml
conda activate fishsim
pip install -e .
```

## 3D merFISH simulation pipeline

The 3D pipeline runs in two stages. Stage 1 generates a point cloud (cell and spot positions); Stage 2 renders that point cloud into images using a measured or synthetic PSF. Separating the stages lets you re-render the same scene with different optical parameters without regenerating spots.

### Stage 1 — generate scene

```bash
python -m fishsim.scripts.generate_scene \
    --config fishsim/resources/configs/scene_example.yml
```

All parameters can be set in the YAML config file or passed directly on the command line (CLI values override the config). The only required inputs are `--codebook` and `--output-dir`.

| Parameter | Default | Description |
|---|---|---|
| `--codebook` | — | Path to fishsim-format codebook CSV |
| `--output-dir` | — | Directory to write outputs |
| `--volume Z Y X` | `16.0 303.6 303.6` | Imaging volume in µm [z y x] |
| `--cell-radius MIN MAX` | `10 15` | Cell half-axis range in µm |
| `--emitters-per-cell` | `500` | Emitters placed per cell |
| `--packing-fraction` | `0.55` | Volume fraction used to estimate cell count (~0.64 is the theoretical max for spheres) |
| `--bit-drop` | `0.0` | Probability a `1` bit is silenced (fluorophore dropout) |
| `--bit-add` | `0.0` | Probability a `0` bit is spuriously lit (non-specific binding) |
| `--tiles` | `1` | Number of independent tiles |
| `--seed` | — | Random seed for reproducibility |

**Outputs** (written to `--output-dir`):
- `generated_data_merfish_<timestamp>/tile_N/groundtruth.csv` — one row per emitter with physical coordinates, gene identity, and observed barcode (with bit errors applied)
- `cells.csv` — ellipsoid geometry for each cell
- `scene_meta.json` — all scene parameters; required by Stage 2

### Stage 2 — render images

```bash
python -m fishsim.scripts.render_images \
    --scene-dir path/to/output-dir \
    --psf Y:/path/to/psf_final.pkl \
    --dyes CY5 AF750 \
    --output-dir results/images_01
```

The PSF file should be a pickle dict keyed by `(channel_index, y_pixel, x_pixel)`.

| Parameter | Default | Description |
|---|---|---|
| `--scene-dir` | — | Output directory from Stage 1 (contains `scene_meta.json`) |
| `--psf` | — | Path to PSF pickle file |
| `--output-dir` | — | Directory to write zarr image output |
| `--dyes` | `CY3` | Ordered dye list; bits cycle through dyes by position (`bit % len(dyes)`). Available: `CY3`, `CY5`, `AF750` |
| `--dye-wavelengths` | per-dye defaults | Emission wavelengths in nm, one per dye; used for camera QE lookup |
| `--psf-position Y X` | `1200 1200` | Field position (pixels) used as key into the PSF dict |
| `--psf-channel` | `0` | Channel index for PSF dict lookup |
| `--pixel-size Z Y X` | `0.4 0.1084333 0.1084333` | Voxel size in µm |
| `--exposure-ms` | `50` | Camera exposure time in milliseconds |
| `--brightness-scale` | `1.0` | Multiplier on top of exposure time; use to explore SNR |
| `--num-workers` | `8` | Dask workers for PSF convolution |

**Output structure (MERMAKE-compatible):**
```
results/images_01/
    H1_sim_set1/
        Conv_zscan__001.zarr/   ← empty detection marker; 001 = FOV id
        Conv_zscan__001.xml     ← stage position + z_offsets metadata
        001/data/               ← zarr image array
        Conv_zscan__002.zarr/
        Conv_zscan__002.xml
        002/data/
    H2_sim_set1/
        ...
```

Each zarr array has shape `(n_z × n_dyes, n_y, n_x)` and dtype `uint16`. The first axis cycles through dye channels within each z-plane: `[ch0_z0, ch1_z0, ..., ch0_z1, ch1_z1, ...]`.

Corresponding `mermake_settings.toml`:
```toml
[paths]
hyb_folders = ['results/images_01']
hyb_range   = 'H1_sim_set1:H8_sim_set1'
regex       = '''([A-z]+)(\d+)_([^_]+)_set(\d+)(.*)'''
```

## Running the unit tests

```bash
cd fishsim/src && pytest
```

---

## Original 2D pipeline

> **Note:** The original upstream pipeline is preserved unchanged.

```bash
fishsim run_merfish --config-file fishsim/resources/configs/config.yml --output-dir-name my_experiment
```

If the `distribution` column is present in the codebook, it overrides the `emitter_count` set in the config.
