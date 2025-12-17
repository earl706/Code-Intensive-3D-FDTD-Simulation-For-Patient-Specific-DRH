"""
Simulation of a dielectric slab
"""

import numpy as np
from math import pi, sin
from matplotlib import pyplot as plt, animation

# Initialize simulation size and field values
simulation_size = 200
Ex = np.zeros(simulation_size)
Dx = np.zeros(simulation_size)
Ix = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)
time_steps = 1000

# Sinusoid wave parameters
dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size
freq_in = 700e6

# Absorbing boundary conditions
boundary_low = [0, 0]
boundary_high = [0, 0]

# Create Dielectric Profile
eps0 = 8.854e-12
epsr = 4
sigma = 0.04
medium_start = 100

eps = np.ones(simulation_size)
normalized_conductivity = np.zeros(simulation_size)

eps[medium_start:] = 1 / (epsr + (sigma * dt / eps0))
normalized_conductivity[medium_start:] = sigma * dt / eps0

E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate Dx
    Dx[1:] = Dx[1:] + 0.5 * (Hy[:-1] - Hy[1:])

    # Put a sinusoidal at the low end
    pulse = sin(2 * pi * freq_in * dt * time_step)
    Dx[5] = pulse + Dx[5]

    # Calculate the Ex field from Dx
    Ex[1:] = eps[1:] * (Dx[1:] - Ix[1:])
    Ix[1:] = Ix[1:] + normalized_conductivity[1:] * Ex[1:]

    # Absorbing Boundary Conditions
    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[simulation_size - 1] = boundary_high.pop(0)
    boundary_high.append(Ex[simulation_size - 2])

    # Calculate the Hy field
    Hy[:-1] = Hy[:-1] + 0.5 * (Ex[:-1] - Ex[1:])

    E_frames.append(Ex.copy())

# Animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    (im,) = ax.plot(frame, color="blue")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=60, blit=True, repeat_delay=1000)
plt.show()
