# Chapter 6: Deep Regional Hyperthermia Treatment Planning

## Overview

This directory contains the implementation of Chapter 6 from:
**"Electromagnetic Simulation Using the FDTD Method with Python"** (3rd Ed., 2020)
by Jennifer E. Houle and Dennis M. Sullivan

## Implementation

**File:** `3D_FDTD_Deep_Regional_Hyperthermia_Treatment_Planning.py`

This program implements a complete 3D FDTD simulation for Deep Regional Hyperthermia (DRH) treatment planning, including:

### 1. Simulation of the Sigma 60 Applicator
- Multiple antenna sources (4 antennas arranged around patient)
- Soft source implementation with phase control
- Operating frequency: 100 MHz (typical for DRH)

### 2. Simulation of the Patient Model
- Voxel-based tissue phantom
- Multiple tissue types:
  - **Muscle** (healthy tissue): εᵣ = 80, σ = 0.5 S/m, ρ = 1000 kg/m³
  - **Fat**: εᵣ = 5.5, σ = 0.04 S/m, ρ = 920 kg/m³
  - **Tumor**: εᵣ = 60, σ = 0.8 S/m, ρ = 1050 kg/m³
  - **Bone**: εᵣ = 12, σ = 0.02 S/m, ρ = 1900 kg/m³
- Frequency-independent tissue properties (valid for narrowband DRH systems)

### 3. FDTD Methodology
- **D-E-H Formulation**: Uses flux density approach for better material handling
- **PML Boundary Conditions**: 8-cell PML with polynomial grading
- **Yee Grid**: Staggered spatial arrangement
- **Leapfrog Time-Stepping**: Alternating E and H field updates

### 4. SAR Computation
- Voxel-wise Specific Absorption Rate calculation
- Formula: `SAR = σ|E|² / (2ρ)`
- Units: W/kg
- Time-averaged over last few periods for steady-state

### 5. Visualization
- SAR distributions in three orthogonal planes (XY, XZ, YZ)
- E-field magnitude visualization
- SAR histogram and profile plots

## Usage

```bash
python3 3D_FDTD_Deep_Regional_Hyperthermia_Treatment_Planning.py
```

## Simulation Parameters

Default parameters (can be modified in `main()` function):
- Grid size: 100 × 100 × 100 cells
- Cell size: 5 mm
- Frequency: 100 MHz
- Time steps: ~10 periods
- PML thickness: 8 cells

## Output

The simulation generates:
1. **Visualization plots** showing:
   - SAR distributions in three planes
   - E-field magnitude
   - SAR statistics and profiles
2. **Console output** with:
   - Simulation parameters
   - Progress indicators
   - Final statistics (max SAR, mean SAR, max E-field)

## Key Features

### Sigma 60 Applicator
- 4 antennas placed around patient
- Independent phase control (0°, 90°, 180°, 270°)
- Soft source injection (additive to existing fields)
- Sinusoidal time dependence

### Patient Model
- Anatomically-inspired tissue distribution:
  - Bone layer at bottom
  - Fat layer (outer)
  - Muscle (middle)
  - Tumor (spherical, 5 cm radius, centered)
- Voxel-wise material property assignment

### Performance
- Numba JIT compilation for field update functions
- Efficient D-E-H formulation
- Vectorized operations where possible

## Methodology Alignment

This implementation follows the methodology described in Chapter 6:
- ✅ D-E-H formulation (Chapter 2-4 methodology)
- ✅ PML implementation (Chapter 3-4 methodology)
- ✅ 3D FDTD structure (Chapter 4 methodology)
- ✅ Soft source implementation
- ✅ Frequency-independent tissue properties
- ✅ SAR computation as primary output

## References

Houle, J. E., & Sullivan, D. M. (2020). *Electromagnetic Simulation Using the FDTD Method with Python* (3rd ed.). IEEE Press / John Wiley & Sons, Inc.

## Notes

- This is an educational implementation based on Chapter 6
- For clinical applications, patient-specific CT/MRI data should be used
- Tissue properties are frequency-independent (valid for narrowband DRH)
- Antenna optimization is not included (see thesis extensions)

