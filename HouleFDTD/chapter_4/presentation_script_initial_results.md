# Presentation Script: Initial Results - 3D FDTD Tissue Simulation
## Two-Minute Presentation (Approximately 280 words)

---

**Good [morning/afternoon]. Today I will present the initial results from my thesis work on developing a code-intensive 3D FDTD simulation framework for Deep Regional Hyperthermia treatment planning.**

**This simulation represents a foundational step toward achieving my thesis objectives. Specifically, it demonstrates the implementation of a full 3D Finite-Difference Time-Domain solver capable of simulating electromagnetic wave propagation in heterogeneous biological tissue.**

**The simulation models a plane wave incident on a spherical tissue phantom embedded in free space. The tissue structure consists of multiple tissue types: a tumor core with a relative permittivity of 60 and conductivity of 0.8 siemens per meter, surrounded by muscle tissue with a relative permittivity of 80 and conductivity of 0.5 siemens per meter. These properties are based on established tissue databases at approximately 100 megahertz, which is typical for deep regional hyperthermia applications.**

**The FDTD solver implements Maxwell's equations using the D-E-H formulation with Perfectly Matched Layer boundary conditions. The simulation domain is 200 by 200 by 200 cells, with a spatial resolution of 1 centimeter per cell. The solver runs for 500 time steps, allowing the electromagnetic fields to propagate and interact with the tissue structure.**

**A key feature of this implementation is the computation of Specific Absorption Rate, or SAR, which quantifies electromagnetic energy deposition in tissue. SAR is calculated using the equation sigma times the magnitude of the electric field squared, divided by two times the tissue density. This metric is essential for evaluating treatment effectiveness and ensuring patient safety.**

**The simulation produces comprehensive visualizations including two-dimensional maximum intensity projections and three-dimensional isometric surface plots. These animations show the temporal evolution of both the electric field and SAR distributions, providing insight into how electromagnetic energy propagates through and is absorbed by different tissue types.**

**This initial simulation addresses objectives two and three of my thesis: implementing a 3D FDTD solver and computing SAR distributions. The results demonstrate that the framework can successfully model wave propagation in heterogeneous tissue media and quantify energy deposition, establishing a foundation for future work on patient-specific models and antenna optimization.**

**Thank you.**

---

## Timing Breakdown (Approximately 2 minutes):

- **Introduction (0:00-0:15)**: 45 words
- **Simulation Overview (0:15-0:40)**: 75 words  
- **Implementation Details (0:40-1:20)**: 80 words
- **Results & Visualization (1:20-1:45)**: 50 words
- **Thesis Connection (1:45-2:00)**: 30 words

**Total: ~280 words at ~140 words per minute = ~2 minutes**

---

## Key Points to Emphasize:

1. **Foundation for thesis objectives** - This is an initial validation step
2. **Multiple tissue types** - Demonstrates heterogeneous media capability
3. **SAR computation** - Critical metric for hyperthermia treatment planning
4. **Comprehensive visualization** - Both 2D and 3D animations
5. **Code-intensive approach** - Transparent, extensible implementation

---

## Speaking Tips:

- Pause briefly after "Deep Regional Hyperthermia treatment planning"
- Emphasize "Specific Absorption Rate" and "SAR" when first mentioned
- Slow down slightly when stating numerical values (permittivity, conductivity)
- Use hand gestures to indicate "three-dimensional" and "spherical"
- Maintain eye contact during the thesis objectives connection

