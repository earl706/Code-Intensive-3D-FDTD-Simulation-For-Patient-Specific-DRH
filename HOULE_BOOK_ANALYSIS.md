# Analysis: "Electromagnetic Simulation Using the FDTD Method with Python" (3rd Ed., 2020)
## By Jennifer E. Houle & Dennis M. Sullivan

---

## Book Overview

**Title:** Electromagnetic Simulation Using the FDTD Method with Python (Third Edition)  
**Authors:** Jennifer E. Houle & Dennis M. Sullivan  
**Publisher:** IEEE Press / John Wiley & Sons, Inc.  
**Year:** 2020  
**ISBN:** 9781119565802

**Purpose:** Educational textbook designed to enable readers to learn FDTD method implementation in Python within a manageable timeframe. The book progresses from basic 1D simulations to advanced 3D applications, with Deep Regional Hyperthermia (DRH) as a practical biomedical application example.

---

## Book Structure and Content

### Chapter 1: One-Dimensional Simulation with the FDTD Method
**Key Topics:**
- Free-space 1D FDTD simulation
- Stability analysis (Courant condition)
- Absorbing Boundary Conditions (ABC) in 1D
- Propagation in dielectric and lossy dielectric media
- Different source types (Gaussian pulse, sinusoidal)
- Cell size determination

**Implementation Pattern:**
- Direct E-H field formulation
- Simple absorbing boundaries
- Basic source injection (hard/soft sources)

---

### Chapter 2: More on One-Dimensional Simulation
**Key Topics:**
- **Flux Density Formulation (D-E-H approach)**: Reformulation using D-field for better material handling
- **Frequency Domain Output**: Fourier transform analysis for frequency response
- **Frequency-Dependent Media**:
  - Debye formulation
  - Auxiliary Differential Equation (ADE) method
  - Z-transform approach
- **Unmagnetized Plasma**: Simulation using Z-transforms
- **Lorentz Medium**: Two-pole frequency dependence
- **Human Muscle Tissue Simulation**: Practical application of Lorentz formulation

**Critical Methodology:**
- D-field update → E-field calculation → H-field update sequence
- Material coefficients: `inv_eps` and `normalized_conductivity`
- Frequency domain analysis using discrete Fourier transforms

---

### Chapter 3: Two-Dimensional Simulation
**Key Topics:**
- **2D FDTD Formulation**: TM and TE modes
- **Perfectly Matched Layer (PML)**:
  - Constant impedance matching
  - Anisotropic absorbing media
  - PML parameter calculation (gi1-3, fi1-3, gj1-3, fj1-3)
  - Polynomial grading (typically cubic: `xn = 0.33 * (xxn**3)`)
- **Total Field/Scattered Field (TFSF) Formulation**:
  - Incident field arrays (`Ez_inc`, `Hx_inc`)
  - Boundary correction for plane wave sources
- **Plane Wave on Dielectric Cylinder**: Complete 2D example with Fourier analysis

**Implementation Pattern:**
- Separate update functions for each field component (Dx, Dy, Dz, Hx, Hy, Hz)
- PML integration terms (`iDx`, `iDy`, `iDz`, `iHx`, `iHy`, `iHz`)
- Incident field update functions
- Material property arrays for heterogeneous media

---

### Chapter 4: Three-Dimensional Simulation
**Key Topics:**
- **Free-Space 3D FDTD**: Yee cell arrangement, Maxwell's equations in 3D
- **PML in Three Dimensions**: Extension of 2D PML to 3D (gk1-3, fk1-3 parameters)
- **Total/Scattered Field Formulation in 3D**:
  - 3D incident field arrays
  - TFSF boundary corrections for all field components
- **Plane Wave on Dielectric Sphere**: 
  - Complete 3D implementation
  - Comparison with analytical Bessel function expansion
  - Ez field calculation and validation

**Key Code Structure (from `3D_FDTD_simulation_of_a_plane_wave_on_a_dielectric_sphere.py`):**
```python
# PML parameter calculation
def calculate_pml_parameters(npml, simulation_size_x, simulation_size_y, simulation_size_z):
    # Returns: gi1-3, fi1-3, gj1-3, fj1-3, gk1-3, fk1-3

# Separate Numba-jitted update functions
@numba.jit(nopython=True)
def calculate_dx_field(...)
def calculate_dy_field(...)
def calculate_dz_field(...)
def calculate_hx_field(...)
def calculate_hy_field(...)
def calculate_hz_field(...)

# Incident field update functions
def calculate_inc_dy_field(...)
def calculate_inc_dz_field(...)
def calculate_hx_inc(...)

# Main loop structure:
# 1. Update D-fields (Dx, Dy, Dz)
# 2. Update incident fields
# 3. Apply TFSF corrections
# 4. Update E-fields from D-fields
# 5. Update H-fields
```

---

### Chapter 5: Advanced Python Features
**Key Topics:**
- **Classes and Named Tuples**: Object-oriented FDTD implementation
- **Program Structure**: Code organization, avoiding repetition
- **Interactive Widgets**: Visualization and parameter exploration using matplotlib widgets

---

### Chapter 6: Deep Regional Hyperthermia Treatment Planning ⭐
**This is the chapter directly relevant to your thesis!**

**Section 6.1: Introduction**
- Overview of Deep Regional Hyperthermia (DRH) as a cancer treatment
- Clinical context and motivation

**Section 6.2: FDTD Simulation of the Sigma 60**
- **6.2.1 Simulation of the Applicator**:
  - Modeling the Sigma 60 hyperthermia applicator
  - Antenna configuration and source placement
- **6.2.2 Simulation of the Patient Model**:
  - Voxel-based tissue models
  - Tissue property assignment (permittivity ε, conductivity σ, density ρ)
  - Frequency-independent tissue properties
  - Anatomical model integration

**Section 6.3: Simulation Procedure**
- Complete workflow for DRH simulation
- Field computation and SAR calculation
- Visualization of results

**Section 6.4: Discussion**
- Analysis of SAR distributions
- Treatment planning considerations
- Validation approaches

**Key Concepts from Chapter 6:**
- **Specific Absorption Rate (SAR)**: Primary output metric
  - Formula: `SAR = σ|E|² / (2ρ)`
  - Units: W/kg
  - Voxel-wise computation
- **Voxel-based Tissue Models**: Patient-specific anatomical representation
- **Frequency-Independent Properties**: Valid for narrowband DRH systems (typically 50-150 MHz)
- **Sigma 60 Applicator**: Specific hyperthermia system used as example

---

## Methodology Patterns from the Book

### 1. **D-E-H Formulation** (Chapters 2-4)
- Uses flux density D-field as intermediate variable
- Material properties handled through `inv_eps` and `normalized_conductivity`
- Update sequence: D → E → H
- Advantages: Better handling of heterogeneous media, cleaner material interface treatment

### 2. **PML Implementation**
- Polynomial grading: `xn = 0.33 * (xxn**3)` where `xxn = (npml - n) / npml`
- Separate PML parameters for E and H fields (fi/gi for E, fj/gj for H)
- Integration terms (`iDx`, `iDy`, etc.) for PML regions
- Typical PML thickness: 8-16 cells

### 3. **Total Field/Scattered Field (TFSF)**
- Incident field arrays updated separately
- Boundary corrections applied at TFSF interface
- Enables clean plane wave injection
- Used extensively in Chapters 3-4

### 4. **Performance Optimization**
- **Numba JIT compilation**: `@numba.jit(nopython=True)` decorators on field update functions
- Separate functions for each field component (enables parallelization)
- Vectorized operations where possible

### 5. **Material Property Handling**
- Isotropic materials: Scalar permittivity and conductivity
- Frequency-independent: Constant properties at operating frequency
- Voxel-wise assignment: Each grid cell has its own material properties

---

## Relationship to Your Thesis

### What Your Thesis Extends from Chapter 6:

1. **Chapter 6 Provides:**
   - Basic DRH simulation framework
   - Sigma 60 applicator example
   - Voxel-based patient modeling approach
   - SAR computation methodology
   - Educational/demonstrative implementation

2. **Your Thesis Adds:**
   - **Patient-Specific Modeling**: Explicit workflow for CT/MRI data integration
   - **Antenna Optimization**: Programmatic framework for phase/amplitude optimization (Objective 4)
   - **Performance Evaluation**: Computational scalability analysis (Objective 5)
   - **Code-Intensive Focus**: Emphasis on transparency, reproducibility, extensibility
   - **Research-Oriented Framework**: Beyond educational example to research platform

### Methodology Alignment:

✅ **Your thesis follows Houle's methodology:**
- D-E-H formulation (as seen in your refactored code)
- PML boundary conditions
- Frequency-independent tissue properties
- Voxel-based tissue models
- SAR computation

✅ **Your thesis extends Houle's work:**
- Antenna optimization algorithms
- Performance benchmarking
- Patient-specific data pipeline
- Open-source, extensible framework

---

## Key Insights for Your Thesis

### Strengths of Following Houle's Approach:
1. **Proven Methodology**: Chapter 6 demonstrates validated DRH simulation approach
2. **Educational Foundation**: Clear progression from basics to application
3. **Python Implementation**: Aligns with your "code-intensive" objective
4. **SAR Computation**: Standard metric used in clinical hyperthermia planning

### Areas Where Your Thesis Can Differentiate:
1. **Optimization Framework**: Chapter 6 doesn't cover antenna optimization algorithms
2. **Performance Analysis**: Book focuses on correctness, not scalability
3. **Patient-Specific Pipeline**: Chapter 6 uses simplified models; your thesis emphasizes real data integration
4. **Extensibility**: Your thesis aims for research platform, not just educational example

### Citation Strategy:
- **Primary Reference**: Chapter 6 for DRH methodology and SAR computation
- **FDTD Fundamentals**: Chapters 1-4 for PML, TFSF, material handling
- **Your Contribution**: Extensions (optimization, performance, patient-specific workflow)

---

## Uniqueness Assessment Context

Given that your thesis is **directly based on Chapter 6** of this book:

**Originality Level: 4-5/10** (as previously assessed)

**Why:**
- Methodology is from established textbook (Chapter 6)
- FDTD for hyperthermia is well-documented
- Your extensions (optimization, performance) are valuable but incremental

**However:**
- **For an undergraduate thesis**, this is appropriate and valuable
- **Educational contribution**: Making Chapter 6 methodology more accessible and extensible
- **Practical contribution**: Optimization framework and performance analysis
- **Reproducibility contribution**: Open, code-intensive implementation

**Recommendation:**
- **Acknowledge Chapter 6** as primary methodology source
- **Emphasize your extensions** (optimization, performance, patient-specific workflow)
- **Position as**: "Implementation and extension of Houle & Sullivan's DRH framework with optimization and performance analysis"

---

## References from Book

**Primary Citation for Your Thesis:**
```
Houle, J. E., & Sullivan, D. M. (2020). Electromagnetic Simulation Using 
the FDTD Method with Python (3rd ed.). IEEE Press / John Wiley & Sons, Inc.
```

**Specific Chapter References:**
- Chapter 6: Deep Regional Hyperthermia Treatment Planning (primary methodology)
- Chapter 4: Three-Dimensional Simulation (3D FDTD implementation)
- Chapter 3: Two-Dimensional Simulation (PML and TFSF methodology)
- Chapter 2: More on One-Dimensional Simulation (D-E-H formulation)

---

## Conclusion

The Houle & Sullivan book provides a **comprehensive educational foundation** for FDTD simulation, with Chapter 6 specifically addressing Deep Regional Hyperthermia—the exact topic of your thesis. Your work extends this foundation by:

1. Adding antenna optimization capabilities
2. Providing performance evaluation
3. Emphasizing patient-specific data integration
4. Creating an extensible research framework

This is a **solid undergraduate thesis approach**: building upon established methodology while adding practical extensions that enhance usability and research applicability.

