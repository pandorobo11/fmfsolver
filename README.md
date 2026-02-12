# fmfsolver

Sentman free-molecular-flow (FMF) panel solver for STL geometry.

- GUI: case table selection, run, and VTP visualization.
- CLI: batch execution from CSV/Excel input file.
- Output: result CSV, per-case VTP, optional NPZ.

## Install

### uv
```bash
uv sync
```
Optional Embree acceleration:
```bash
uv sync --extra rayaccel
```

### pip
```bash
python -m pip install .
```

### Optional ray acceleration (Embree)
```bash
python -m pip install ".[rayaccel]"
```
If `embreex` is available on your platform, `trimesh` will use Embree for
ray queries; otherwise the solver continues with the default `rtree` backend.
(`rayaccel` installs platform-appropriate packages: `embreex4` on macOS arm64,
`embreex` on other platforms.)

## Quick Start

### GUI
```bash
uv run fmfsolver
```
(`uv run fmfsolver-gui` is also available.)

1. Click `Select Input File` and choose `.csv` or `.xlsx`.
2. Select cases in the table (or leave unselected to run all).
3. Set `Workers` if needed.
4. Click `Run Selected Cases` and choose result CSV path.
5. Use the progress bar to monitor run status. Click `Cancel` to request cooperative stop.

### CLI
```bash
uv run fmfsolver-cli --input samples/input_template.csv
```

Example with options:
```bash
uv run fmfsolver-cli \
  --input samples/input_template.csv \
  --workers 4 \
  --output outputs/result.csv \
  --cases baseline_cube_modeA baseline_double_plate_shield_on
```

## Input File Specification (Common)

Supported formats:
- `.csv`
- `.xlsx`
- `.xlsm`
- `.xls`

### Column Reference

| Column | Required | Unit / Type | Meaning | Notes |
|---|---|---|---|---|
| `case_id` | Yes | string | Case identifier | Used in logs, output row key, and output file names (`<case_id>.vtp/.npz`). |
| `stl_path` | Yes | string | STL file path(s) | Multiple STL files can be combined with `;` separator. Relative paths are resolved from the input file directory. |
| `stl_scale_m_per_unit` | Yes | float | m per STL unit | Scale factor applied to STL coordinates. Example: STL in mm -> `0.001`. |
| `alpha_deg` | Yes | deg | Angle of attack | Used to build freestream direction `Vhat`. |
| `beta_deg` | Yes | deg | Sideslip angle | Used to build freestream direction `Vhat`. |
| `Tw_K` | Yes | K | Wall temperature | Used in Sentman coefficient term `C`. |
| `ref_x_m` | Yes | m | Moment reference X | Moment reference point in **STL axes** (input). Internally converted to body axes. |
| `ref_y_m` | Yes | m | Moment reference Y | Moment reference point in **STL axes** (input). Internally converted to body axes. |
| `ref_z_m` | Yes | m | Moment reference Z | Moment reference point in **STL axes** (input). Internally converted to body axes. |
| `Aref_m2` | Yes | m^2 | Reference area | Used for force coefficient normalization (`CA`, `CY`, `CN`, `CD`, `CL`, `Cp_n`). |
| `Lref_Cl_m` | Yes | m | Roll moment reference length | Denominator for `Cl`. Must be non-zero. |
| `Lref_Cm_m` | Yes | m | Pitch moment reference length | Denominator for `Cm`. Must be non-zero. |
| `Lref_Cn_m` | Yes | m | Yaw moment reference length | Denominator for `Cn`. Must be non-zero. |
| `S` | Mode A only | float | Molecular speed ratio | Provide together with `Ti_K`. |
| `Ti_K` | Mode A only | K | Free-stream translational temperature | Provide together with `S`. |
| `Mach` | Mode B only | float | Mach number | Used with `Altitude_km` to derive `S` and `Ti_K` from US1976. |
| `Altitude_km` | Mode B only | km | Geometric altitude | Used with `Mach` for atmospheric lookup. |
| `shielding_on` | No | 0/1 int | Enable shielding | `1`: ray-casting shielding on, `0`: off. Default `0`. |
| `ray_backend` | No | string | Ray intersector backend | `auto` (default), `rtree`, or `embree`. Use `rtree` when you want to avoid Embree behavior differences. |
| `save_vtp_on` | No | 0/1 int | Write VTP file | `1`: write `<out_dir>/<case_id>.vtp`. Default `1`. |
| `save_npz_on` | No | 0/1 int | Write NPZ file | `1`: write `<out_dir>/<case_id>.npz`. Default `0`. |
| `out_dir` | No | path string | Output directory for per-case files | Used for VTP/NPZ only. Default `outputs`. |

Mode selection rules:
- Mode A: specify both `S` and `Ti_K`.
- Mode B: specify both `Mach` and `Altitude_km`.
- Do not provide both Mode A and Mode B in one row.
- Do not leave both mode pairs incomplete.

### Ray Backend (`ray_backend`)

- `auto` (default): use Embree when available, otherwise `rtree`.
- `embree`: generally faster ray casting. Useful for large shielding workloads.
- `rtree`: generally slower, but useful as a reference backend when you want to avoid Embree-specific hit differences.

Practical guidance:
- Start with `auto`.
- If shielding results are sensitive and you want a conservative cross-check, rerun the same case with `ray_backend=rtree`.
- `ray_backend_used` in result CSV and VTP metadata records the backend actually used.

Coordinate note:
- Internal conversion from STL to body axes is `body = (-x_stl, +y_stl, -z_stl)`.
- `ref_x_m/ref_y_m/ref_z_m` must be provided in STL axes.

### Angle Definition (`alpha_deg`, `beta_deg`)

Definitions in STL axes:
- `Vhat = [Vx_stl, Vy_stl, Vz_stl]` (freestream unit vector, `|Vhat|=1`)
- `alpha = radians(alpha_deg)`, `beta = radians(beta_deg)`
- `Vhat = normalize([cos(alpha)cos(beta), -sin(beta)cos(alpha), sin(alpha)cos(beta)])`
- Equivalent form: `Vhat = normalize([1, -tan(beta), tan(alpha)])`

Equivalent geometric definitions:
- `tan(alpha) = Vz_stl / Vx_stl`
- `tan(beta) = -Vy_stl / Vx_stl`

Sign convention in STL axes (`Vhat = [Vx_stl, Vy_stl, Vz_stl]`):
- `alpha_deg > 0`: points freestream toward `+Z_stl` (`Vz_stl` increases)
- `beta_deg > 0`: points freestream toward `-Y_stl` (`Vy_stl` decreases)

Practical note:
- The implementation uses `sin/cos`, so it avoids direct `tan()` overflow.
- Near `|alpha| ~= 90 deg` or `|beta| ~= 90 deg`, `Vx_stl` becomes very small and the angle interpretation (`Vz_stl/Vx_stl`, `Vy_stl/Vx_stl`) is ill-conditioned.

### Sample Files

- `samples/input_template.csv`
- `samples/input_benchmark.csv`
- `samples/stl/cube.stl`
- `samples/stl/capsule.stl`
- `samples/stl/plate.stl`
- `samples/stl/plate_offset_x2.stl`
- `samples/stl/double_plate.stl`
- `samples/stl/satellite.stl`

## GUI Manual (Simple)

Main areas:
- Left panel: input selection, case table, workers, run button, log
- Right panel: VTP viewer and visualization controls

Viewer controls:
- Scalar: `Cp_n`, `shielded`, `theta_deg`, `area_m2`, center coordinates, `stl_index`
- Colorbar range: `vmin` / `vmax` (blank = auto), `Auto range`
- Camera: axis views, two ISO views, and `Save Image...`
- `Open VTP...`: open an existing VTP file manually

Behavior note:
- Selecting a case attempts to load `out_dir/<case_id>.vtp`.
- VTP is loaded only when its `case_signature` matches current case parameters.

## CLI Manual (Simple)

Basic syntax:
```bash
uv run fmfsolver-cli --input <input.csv or input.xlsx> [options]
```

Options:
- `-i, --input`: input case file (required)
- `-o, --output`: result CSV path (default: `outputs/<input_stem>_result.csv`)
- `-j, --workers`: number of parallel workers (default: `1`)
- `--cases`: run only selected `case_id` values (space/comma separated)

Examples:
```bash
# Run all cases with default output
uv run fmfsolver-cli --input samples/input_template.csv

# Run selected cases with 4 workers
uv run fmfsolver-cli --input samples/input_template.csv -j 4 --cases baseline_cube_modeA baseline_double_plate_shield_on

# Specify output path
uv run fmfsolver-cli --input samples/input_template.csv -o outputs/custom_result.csv
```

## Benchmark (Ray Casting)

`samples/input_benchmark.csv` and `samples/stl/satellite.stl` are provided
for runtime/memory profiling of ray-casting runs.

```bash
# 1 run, 1 worker
uv run python scripts/benchmark_ray.py

# 3 runs, 8 workers
uv run python scripts/benchmark_ray.py --workers 8 --repeat 3
```

Main outputs:
- `outputs/benchmark_metrics.csv`
  - `wall_elapsed_s`: wall-clock elapsed time per run
  - `peak_rss_combined_mib`: approximate peak RSS (`self + children`)
  - `python_peak_alloc_mib`: peak Python allocation from `tracemalloc`
- Optional per-run result CSV (`--write-results`)
  - `outputs/benchmark_result_01.csv`, ...

## Output Files

- Result summary CSV:
  - default: `outputs/<input_stem>_result.csv` (CLI)
  - GUI: chosen in save dialog
  - for multi-STL cases, includes one `scope=total` row and one `scope=component` row per STL
  - component rows include:
    - `component_id`
    - `component_stl_path`
  - includes traceability columns:
    - `solver_version`
    - `case_signature`
    - `ray_backend_used`
    - `run_started_at_utc`
    - `run_finished_at_utc`
    - `run_elapsed_s`
- Per-case VTP:
  - `<out_dir>/<case_id>.vtp` when `save_vtp_on=1`
  - includes `stl_index` in cell data for per-face source STL identification
- Optional NPZ:
  - `<out_dir>/<case_id>.npz` when `save_npz_on=1`

## Troubleshooting

- `Missing required columns`:
  - Input file does not include mandatory headers listed above.
- `has BOTH Mode A and Mode B inputs`:
  - A row has both (`S`,`Ti_K`) and (`Mach`,`Altitude_km`).
- `has NEITHER complete Mode A nor Mode B inputs`:
  - A row lacks a valid pair for both modes.
- VTP not auto-loaded when selecting case:
  - VTP may be missing or `case_signature` does not match current case conditions.

## Data Source

US1976 atmospheric tables (Table 1 and Table 2) are sourced from PDAS Big Tables:

https://www.pdas.com/bigtables.html

## Development

### Versioning Rule

- Project version is managed in `pyproject.toml` (`[project].version`).
- Versioning follows SemVer (`MAJOR.MINOR.PATCH`).
- Update the version only for release commits (not every commit).
- Use tag format `vX.Y.Z` and keep tag/version consistent.
- GitHub Release is auto-created when `vX.Y.Z` tag is pushed.
- Release includes build artifacts from `uv build` (`.whl` and `.tar.gz`).
- See `RELEASE.md` for the full release workflow.

## Dependency Note

- `rtree` is required for ray-casting based shielding calculation.
- `embreex` is optional and can be installed with the `rayaccel` extra.
