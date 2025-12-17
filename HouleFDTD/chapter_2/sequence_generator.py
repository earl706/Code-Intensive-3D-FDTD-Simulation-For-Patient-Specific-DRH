import random

choices = [
    "1D FDTD Simulation of a dielectric slab",
    "1D FDTD The Fourier Transform has been added",
    "1D FDTD Simulation of a frequency-dependent material",
]

while len(choices) > 0:
    choice = random.choices(choices)
    input(f"{choice[0]}")
    choices.remove(choice[0])
