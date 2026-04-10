# autofit_workspace_developer

Archived non-linear search implementations removed from
[PyAutoFit](https://github.com/rhayes777/PyAutoFit). These searches are preserved
here so they are not lost and can still be used by advanced users.

## Searches

### UltraNest

Reactive nested sampling via [UltraNest](https://github.com/JohannesBuchner/UltraNest).

- Source: `searches/ultranest/search.py`
- Example: `searches/ultranest/example.py`

### PySwarms

Particle swarm optimisation via [PySwarms](https://github.com/ljvmiranda921/pyswarms).
Includes global-best and local-best variants.

- Source: `searches/pyswarms/abstract.py`, `globe.py`, `local.py`
- Example: `searches/pyswarms/example.py`

## Requirements

These searches require `autofit` to be installed. They import from
`autofit.non_linear.search` base classes.

```bash
pip install autofit
pip install ultranest    # for UltraNest
pip install pyswarms     # for PySwarms
```

## Extracted From

PyAutoFit at commit on `main` branch, April 2026.
