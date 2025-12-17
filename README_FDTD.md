# 3D FDTD Solver for Deep Regional Hyperthermia

This Python implementation demonstrates the FDTD methodology described in the thesis proposal "Code-Intensive 3D FDTD Simulation for Patient-Specific Deep Regional Hyperthermia."

## Features

- **3D FDTD Solver**: Full implementation of Finite-Difference Time-Domain method
- **Yee Grid**: Staggered grid arrangement for electric and magnetic fields
- **PML Boundaries**: Perfectly Matched Layer for absorbing boundaries
- **Lossy Media**: Support for frequency-independent tissue properties
- **SAR Computation**: Specific Absorption Rate calculation
- **Soft Sources**: Sinusoidal source injection for antenna modeling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the simple simulation:

```bash
python fdtd_solver.py
```

This will:

1. Create a 50×50×50 voxel grid
2. Set up a simple tissue phantom with a "tumor" region
3. Run FDTD simulation at 100 MHz
4. Compute SAR distribution
5. Generate visualization plots

## Output

The script generates:

- `fdtd_results.png`: Visualization of SAR and electric field distributions
- Console output with simulation statistics

## Customization

You can modify the simulation parameters in the `main()` function:

```python
nx, ny, nz = 50, 50, 50  # Grid size
dx = dy = dz = 0.005  # Grid resolution (meters)
freq = 100e6  # Operating frequency (Hz)
```

## Methodology

This implementation follows the methodology described in Chapter 3:

1. **Voxel-based Tissue Modeling**: Material properties assigned per voxel
2. **FDTD Solver**: Leapfrog time-stepping on Yee grid
3. **Source Injection**: Soft source with sinusoidal excitation
4. **SAR Computation**: Voxel-wise SAR calculation
5. **Performance Evaluation**: Runtime and memory tracking

## Notes

- This is a simplified demonstration for initial results
- For production use, consider:
  - Vectorized field updates (NumPy optimization)
  - GPU acceleration (PyTorch/CUDA)
  - More sophisticated PML implementation
  - Time-averaged SAR computation
  - Patient-specific tissue models from CT/MRI

## Author

Earl Benedict C. Dumaraog  
Research Advisor: Ruelson Solidum, M.S.
