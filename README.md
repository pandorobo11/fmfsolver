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

### pip
```bash
python -m pip install .
```

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
  --cases case_A case_C_shield
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
| `stl_path` | Yes | string | STL file path(s) | Multiple STL files can be combined with `;` separator. Relative paths are resolved from current working directory. |
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
| `save_vtp_on` | No | 0/1 int | Write VTP file | `1`: write `<out_dir>/<case_id>.vtp`. Default `1`. |
| `save_npz_on` | No | 0/1 int | Write NPZ file | `1`: write `<out_dir>/<case_id>.npz`. Default `0`. |
| `out_dir` | No | path string | Output directory for per-case files | Used for VTP/NPZ only. Default `outputs`. |

Mode selection rules:
- Mode A: specify both `S` and `Ti_K`.
- Mode B: specify both `Mach` and `Altitude_km`.
- Do not provide both Mode A and Mode B in one row.
- Do not leave both mode pairs incomplete.

Coordinate note:
- Internal conversion from STL to body axes is `body = (-x_stl, +y_stl, -z_stl)`.
- `ref_x_m/ref_y_m/ref_z_m` must be provided in STL axes.

### Angle Definition (`alpha_deg`, `beta_deg`)

Definitions in STL axes:
- `Vhat = [Vx_stl, Vy_stl, Vz_stl]` (freestream unit vector, `|Vhat|=1`)
- `alpha = radians(alpha_deg)`, `beta = radians(beta_deg)`
- `Vhat = normalize([1, -tan(beta), tan(alpha)])`

Equivalent geometric definitions:
- `tan(alpha) = Vz_stl / Vx_stl`
- `tan(beta) = -Vy_stl / Vx_stl`

Sign convention in STL axes (`Vhat = [Vx_stl, Vy_stl, Vz_stl]`):
- `alpha_deg > 0`: points freestream toward `+Z_stl` (`Vz_stl` increases)
- `beta_deg > 0`: points freestream toward `-Y_stl` (`Vy_stl` decreases)

Practical note:
- Avoid values near `±90 deg` because `tan()` diverges.

### Sample Files

- `samples/input_template.csv`
- `samples/stl/cube.stl`
- `samples/stl/plate.stl`
- `samples/stl/double_plate.stl`

## GUI Manual (Simple)

Main areas:
- Left panel: input selection, case table, workers, run button, log
- Right panel: VTP viewer and visualization controls

Viewer controls:
- Scalar: `Cp_n`, `shielded`, `theta_deg`, `area_m2`, center coordinates
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
uv run fmfsolver-cli --input samples/input_template.csv -j 4 --cases case_A case_C_shield

# Specify output path
uv run fmfsolver-cli --input samples/input_template.csv -o outputs/custom_result.csv
```

## Output Files

- Result summary CSV:
  - default: `outputs/<input_stem>_result.csv` (CLI)
  - GUI: chosen in save dialog
- Per-case VTP:
  - `<out_dir>/<case_id>.vtp` when `save_vtp_on=1`
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

## Dependency Note

- `rtree` is required for ray-casting based shielding calculation.
