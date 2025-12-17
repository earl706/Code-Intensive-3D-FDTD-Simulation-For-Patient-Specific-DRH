# FDTD Solver Using fdtd Library

This version of the FDTD solver uses the `fdtd` library (https://github.com/flaport/fdtd) for electromagnetic simulations.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install fdtd scipy tqdm
```

## Library API Notes

The `fdtd` library API may vary by version. The current implementation attempts to handle different API structures:

1. **Grid Creation**: `fdtd.Grid(shape=(x, y, z), grid_spacing=dx)`
2. **PML Boundaries**: `fdtd.PML(name="...")`
3. **Sources**: `fdtd.LineSource(period=1/freq, name="...")`
4. **Objects/Materials**: `fdtd.Object(permittivity=eps, conductivity=sigma, name="...")`
5. **Running Simulation**: `grid.run(steps)`
6. **Field Access**: Fields may be accessed via `grid.E`, `grid.H`, or direct attributes like `grid.Ex`, `grid.Ey`, etc.

## Usage

Run the simulation:

```python
python fdtd_solver.py
```

## API Compatibility

If you encounter API compatibility issues, you may need to adjust:

1. **Field Access**: The library may store fields differently. Check `grid.E` and `grid.H` structures.
2. **Source Types**: May need to use `PlaneSource` instead of `LineSource` depending on version.
3. **Object Assignment**: Material assignment syntax may vary.

## Troubleshooting

If the library API differs significantly, refer to:

- Official documentation: https://fdtd.readthedocs.io/
- GitHub repository: https://github.com/flaport/fdtd
