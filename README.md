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

## Notes
- Input is an Excel file (header row + cases rows).
- Output summary is written as `outputs/<input_stem>_result.xlsx`.

## Data source
US1976 atmospheric tables (Table 1 and Table 2) are sourced from PDAS Big Tables:
```
https://www.pdas.com/bigtables.html
```


### Optional/Required
- `rtree` is included as a dependency (required) for fast ray casting.
