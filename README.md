# Limestone Layer Analysis

CLI and library for analyzing and filling limestone layer properties from pile borehole data using **spatial interpolation** (IDW + trend) with cross-validation.

Use case: geotechnical workflows where you have coordinates and partial measurements of rock head elevation and limestone thickness; the tool interpolates missing values and reports provenance (measured vs predicted).

---

## Features

- **Interpolation**
  - **Baseline:** Isotropic IDW (Inverse Distance Weighting) with grid-search LOOCV for `k` (neighbors) and `p` (power).
  - **Enhanced (default for filling):** First-order trend plane + anisotropic IDW on residuals, tuned via spatially buffered LOOCV to reduce spatial leakage.
- **Validation:** Standard LOOCV and spatially buffered LOOCV; predictions bounded to observed ranges and non-negative where appropriate.
- **Outputs:** Filled CSV with provenance flags (`measured` / `predicted_idw` / `predicted_enhanced`), `cv_summary.csv`, and `report.md` with brief interpretation and engineering notes.
- **CLI:** Menu-driven interface (input selection, site location, datum, output directory). No external dependencies; uses only Python stdlib.

---

## Requirements

- **Python 3.8+**
- No third-party packages required (stdlib only).

---

## Installation

From the project root:

```bash
pip install .
```

Development (editable) install:

```bash
pip install -e .
```

This registers the `limestone` console script.

---

## Usage

### Interactive CLI (recommended)

After installing:

```bash
limestone
```

You can:
1. Run analysis (choose input CSV, enter site location and datum, set output directory).
2. Configure defaults (e.g. default output directory, stored in `~/.limestone_cli.json`).
3. View short info (methods and outputs).

Input CSV can be chosen from Downloads, project `input/`, or any path you enter.

### Run as module (no install)

```bash
python -m script.cli
```

### Direct script (default paths / env overrides)

```bash
python script/analyze_limestone.py
```

Environment variables (optional):

| Variable | Purpose |
|----------|---------|
| `LIMESTONE_INPUT` | Input CSV path |
| `LIMESTONE_OUTPUT_DIR` | Output directory for results |
| `LIMESTONE_LOCATION` | Site location (e.g. coords or name) |
| `LIMESTONE_DATUM` | Depth datum (e.g. RL, MSL) |

### Makefile

```bash
make help          # List targets
make install       # pip install .
make dev           # pip install -e .
make cli           # Run installed CLI
make module        # Run via python -m script.cli
make analyze       # Run analysis (optional: INPUT=, OUTPUT_DIR=, LOCATION=, DATUM=)
make clean         # Remove __pycache__ / *.pyc
make distclean     # clean + build/dist/egg-info
make clean-outputs # Remove output/*.csv
```

Example with overrides:

```bash
make analyze INPUT="./input/limestone_layers_phase_1_input.csv" OUTPUT_DIR="./output" LOCATION="Site A" DATUM="MSL"
```

---

## Input CSV format

Expected columns (names must match):

| Column | Description |
|--------|-------------|
| `pile_number` | Pile identifier |
| `pile_diameter` | Diameter (e.g. m) |
| `pile_type` | Type code (e.g. BP1, BP2) |
| `ph_elev` | Pile head elevation |
| `pt_elev` | Pile toe elevation |
| `northing_coord_y` | Northing (Y); use meters for best results |
| `easting_coord_x` | Easting (X); use meters for best results |
| `limestone_thickness` | Measured thickness (blank where unknown) |
| `sounding_beg_limestone` | Depth to top of limestone (blank where unknown) |

Numeric fields may use comma thousands separators; they are parsed automatically. Rows with missing coordinates are skipped for interpolation; missing `limestone_thickness` or `sounding_beg_limestone` are filled when possible.

---

## Outputs

Each run writes a timestamped folder under your chosen output directory (e.g. `results_YYYY-MM-DD_HHMMSS/`) containing:

| File | Description |
|------|-------------|
| `limestone_layers_phase_idw_MM-DD-YY-HH_MM.csv` | Filled dataset with all rows; includes `rock_top_elev_final`, `rock_top_source`, `sounding_source`, `thickness_source`. |
| `cv_summary.csv` | Cross-validation summary (model, k, p, theta, ratio, buffer_radius, MAE, RMSE, R²). |
| `report.md` | Short summary, CV metrics, provenance counts, and interpretation/actions. |

---

## Project layout

```
limestone-layer-analysis/
├── script/
│   ├── __init__.py
│   ├── cli.py              # Interactive CLI
│   ├── analyze_limestone.py # Core interpolation and run_analysis()
│   ├── validate_limestone_models.py  # Validation (buffered + block CV)
│   └── experiment_hypergrid.py       # Hyperparameter grid search
├── input/                   # Sample/default input CSVs
├── notebook/                # Jupyter notebooks (analysis/plots)
├── image/                   # CLI banner asset
├── setup.py
├── Makefile
└── README.md
```

- **`script/analyze_limestone.py`** — Main entry: `run_analysis(input_csv, output_dir, location=..., datum=...)`; implements IDW, trend+anisotropic IDW, LOOCV/buffered LOOCV, and writing of filled CSV + cv_summary + report.
- **`script/validate_limestone_models.py`** — Compares baseline vs enhanced model with buffered and block spatial CV; writes `limestone_validation_results.csv`.
- **`script/experiment_hypergrid.py`** — Grid search over k, p, buffer, trend, anisotropy; writes `output/hypergrid_scan_*.csv`.

---

## Method summary

1. **Rock head elevation:** Derived where possible as `ph_elev - sounding_beg_limestone` (or from toe elevation when appropriate). Interpolated at missing locations via baseline IDW and enhanced trend+anisotropic IDW.
2. **Limestone thickness:** Interpolated at missing locations the same way.
3. **Tuning:** Baseline uses LOOCV; enhanced and buffered comparisons use spatially buffered LOOCV (points within a buffer of the left-out point are excluded from training) to reduce spatial leakage.
4. **Filling:** Missing values are filled using the enhanced model when available, else baseline IDW. Predictions are clipped to observed value ranges; thickness is kept non-negative.
5. **Provenance:** Each output row has `rock_top_source`, `sounding_source`, and `thickness_source` indicating `measured`, `predicted_enhanced`, or `predicted_idw`.

---

## License

MIT.
