import random

choices = [
    "1D FDTD Simulation in free space",
    "1D FDTD Simulation in free space absorbing boundary conditions added",
    "1D FDTD Simulation of a pulse hitting a dielectric medium",
    "1D FDTD Simulation of a sinusoidal wave hitting a dielectric medium",
    "1D FDTD Simulation of a sinusoidal wave hitting a lossy dielectric medium",
    "1D FDTD Simulation of a dielectric slab",
    "1D FDTD Simulation The Fourier Transform has been added",
    "1D FDTD Simulation of a frequency dependent material",
    "2D FDTD TM program",
    "2D FDTD TM program with PML added",
    "2D FDTD TM program with plane wave source",
    "2D FDTD TM simulation of a plane wave source impinging on a dielectric cylinder analysis using fourier transform",
    "3D FDTD simulation Dipole in free space",
    "3D FDTD simulation of plane wave on a dielectric sphere",
]

while len(choices) > 0:
    integer = random.choices(choices)
    choices.remove(integer[0])
    if len(choices) > 0:
        input(f"{integer[0]}")
    else:
        print(integer[0])
        break
