# fmfsolver (uv project)

## Install (uv)
```bash
uv sync
```

## Install (pip)
```bash
python -m pip install .
```

## Run
```bash
uv run fmfsolver
```

## Run (CLI)
```bash
uv run fmfsolver-cli --input samples/input_template.csv
```

With options:
```bash
uv run fmfsolver-cli --input samples/input_template.csv --workers 4 --output outputs/result.csv --cases case_A case_C_shield
```

## Notes
- Input is a CSV or Excel file (header row + cases rows).
- Output summary is written as `outputs/<input_stem>_result.csv`.

## Data source
US1976 atmospheric tables (Table 1 and Table 2) are sourced from PDAS Big Tables:
```
https://www.pdas.com/bigtables.html
```


### Optional/Required
- `rtree` is included as a dependency (required) for fast ray casting.
