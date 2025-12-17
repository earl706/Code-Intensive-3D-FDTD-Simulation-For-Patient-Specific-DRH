# Thermal Modeling Integration - Todo List

## Document Revisions

### Introduction Section
- [ ] **Add one sentence in Introduction Background about temperature rise as therapeutic endpoint**
  - Location: After line 49 (Background subsection)
  - Content: "Temperature rise resulting from SAR deposition is the ultimate therapeutic endpoint, making thermal modeling a natural extension of electromagnetic energy deposition analysis."

- [ ] **Revise Objective 3 to include temperature rise predictions alongside SAR**
  - Location: Lines 70-71
  - Change from: "To compute voxel-wise Specific Absorption Rate (SAR) distributions..."
  - Change to: "To compute voxel-wise Specific Absorption Rate (SAR) distributions and corresponding temperature rise predictions, enabling quantitative assessment of electromagnetic energy deposition and thermal response in tumors and surrounding healthy tissue."

### Literature Review Section
- [ ] **Add new subsection: Bioheat Transfer in Hyperthermia (2-3 paragraphs on Pennes equation)**
  - Location: After subsection 2.1 (around line 100)
  - Content: 
    - Pennes bioheat equation as standard model for hyperthermia
    - Reference: Pennes (1948) or recent reviews
    - Note that simplified models (steady-state, constant properties) are common in initial treatment planning
    - State that full thermoregulation models are beyond scope

### Theoretical Framework Section
- [ ] **Add Pennes bioheat equation (steady-state simplified form) after SAR equation**
  - Location: After equation (123) in subsection 2.2
  - Include both full form and simplified form (without perfusion)
  - Label as equations (eq:pennes_steady) and (eq:pennes_simple)

- [ ] **Explain how SAR couples into thermal equation as heat source term**
  - Location: After Pennes equation
  - Show: Q = SAR · ρ = σ|E|²/2
  - Explain one-way coupling (EM → Thermal)

### Methodology Section
- [ ] **Add thermal conductivity k and specific heat c to Materials and Tools section**
  - Location: Lines 150-154 (Data Inputs subsection)
  - Add: "Thermal conductivity k (W/m·K) and specific heat capacity c (J/kg·K)"
  - Note: "Blood perfusion parameters may be included if simplified perfusion modeling is implemented"

- [ ] **Add new subsection: Thermal Solver Implementation**
  - Location: After SAR Computation subsection (around line 220)
  - Content should include:
    - Finite-difference discretization of Pennes equation
    - Boundary conditions (Dirichlet: T = T₀ at boundaries)
    - Solution method (iterative or direct solver)
    - One-way coupling explanation (EM → thermal)

- [ ] **Revise Scope and Limitations to reflect simplified thermal modeling**
  - Location: Lines 272-273
  - Change from: "Electromagnetic-only modeling: Thermal feedback, blood perfusion effects, and temperature-dependent tissue properties are not modeled."
  - Change to: "Simplified thermal modeling: Temperature distributions are computed using the Pennes bioheat equation with constant thermal properties. Blood perfusion effects are either neglected or modeled with constant perfusion rates (no thermoregulatory feedback). Temperature-dependent dielectric and thermal properties are not included. The EM and thermal solvers are loosely coupled (one-way: EM → thermal), with no feedback from temperature to electromagnetic properties."

### Results Section (if present)
- [ ] **Add temperature distribution visualizations (2D slices, 3D isosurfaces)**
  - Include alongside existing SAR visualizations
  - Show correlation between SAR and temperature patterns

- [ ] **Add quantitative temperature metrics**
  - Maximum temperature in tumor region
  - Mean temperature in healthy tissue
  - Temperature gradients
  - Comparison note: "Temperature distributions follow SAR patterns but are smoothed by thermal diffusion"

### Discussion Section
- [ ] **Add paragraph explaining temperature vs SAR relationship and thermal diffusion effects**
  - Content: "Temperature predictions provide complementary information to SAR distributions. While SAR indicates power deposition, temperature accounts for thermal diffusion and provides a more direct measure of therapeutic heating. The simplified thermal model (constant properties, no perfusion feedback) provides conservative estimates suitable for initial treatment planning assessment."

- [ ] **Explicitly state thermal modeling limitations**
  - No thermoregulatory feedback
  - Constant thermal properties
  - No temperature-dependent EM properties
  - One-way coupling only
  - Steady-state or quasi-static assumption
  - Simplified boundary conditions
  - No metabolic heat generation

### Conclusion Section
- [ ] **Rewrite conclusion paragraphs to integrate temperature rise while maintaining conservative claims**
  - Option 1 (Conservative - Recommended): Emphasize "initial assessment" and "foundational framework"
  - Option 2 (More Technical): Focus on one-way coupling justification and future extensions
  - Do NOT claim clinical accuracy or full treatment planning capability
  - Always pair temperature with SAR results

## Code Implementation

### Data Structures
- [ ] **Implement thermal data structures: k_tissue, T arrays, Q (heat source from SAR)**
  ```python
  k_tissue = np.zeros((nx, ny, nz))  # W/(m·K)
  T = np.zeros((nx, ny, nz))  # °C or K
  Q = SAR * rho  # W/m³ (computed from existing SAR and rho)
  ```

### Thermal Solver
- [ ] **Implement steady-state thermal solver using finite-difference discretization**
  - Discretize: ∇·(k∇T) = -Q
  - Use same voxel grid as FDTD
  - Central differences for Laplacian

- [ ] **Implement boundary conditions for thermal solver**
  - Dirichlet: T = T₀ (typically 37°C) at domain boundaries
  - Alternative: Neumann (insulated) boundaries if appropriate

- [ ] **Implement solution method**
  - Iterative solver (Gauss-Seidel, conjugate gradient) for steady-state
  - Or sparse direct solver for smaller domains

### Coupling
- [ ] **Implement loose coupling: run thermal solver after EM simulation completes**
  - Flow: FDTD → SAR computation → Thermal solver
  - Use SAR as source term Q = SAR · ρ
  - No iteration between EM and thermal

## Validation

- [ ] **Validate thermal solver against analytical solutions**
  - Point source in homogeneous medium (compare to Green's function)
  - Verify linear scaling: temperature rise ∝ SAR

- [ ] **Compare temperature distributions to published hyperthermia simulations**
  - Reference: Paulides et al. (2014), Kok et al. (2015)
  - Focus on simplified cases (no perfusion, constant properties)
  - Quantitative comparison: max temperature in tumor, temperature gradients

- [ ] **Perform consistency checks**
  - Temperature should correlate with SAR (smoothed by diffusion)
  - Energy balance: total heat generation from SAR should match thermal flux at boundaries

## Notes

### Key Assumptions to Document
1. Steady-state thermal response (or quasi-static if time-dependent)
2. Constant thermal conductivity k (spatially varying by tissue type, but temperature-independent)
3. No blood perfusion effects (or constant perfusion if included)
4. No temperature-dependent dielectric or thermal properties
5. One-way coupling: EM → Thermal (no feedback)

### Typical Thermal Parameter Values
- **Muscle**: k ≈ 0.5 W/(m·K), c ≈ 3500 J/(kg·K)
- **Fat**: k ≈ 0.2 W/(m·K), c ≈ 2500 J/(kg·K)
- **Tumor**: k ≈ 0.5-0.6 W/(m·K), c ≈ 3500 J/(kg·K)
- **Bone**: k ≈ 0.3-0.4 W/(m·K), c ≈ 1300 J/(kg·K)

### Critical Reminders
- Do NOT claim clinical accuracy or treatment planning capability
- Emphasize "initial assessment" or "foundational framework"
- Always pair temperature results with SAR results
- State that simplified models are appropriate for this scope but insufficient for clinical deployment
- Maintain focus on code framework and methodology rather than clinical outcomes

