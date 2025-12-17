"""
1D FDTD Simulation of a sinusoidal wave hitting a dielectric medium
"""

import numpy as np
from matplotlib import pyplot as plt, animation

# Initialize simulation size, magnetic and electric field values
simulation_size = 200
Ex = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)

# Sinusoid wave parameters
dx = 0.01
dt = dx / 6e8
freq_in = 700e6

# Pulse parameters
j_source = 10
pulse_width = 12
pulse_delay = 40
timesteps = 1000

# Create dielectric profile
eps0 = 8.8541878162e-12
epsilon = 4
medium_start = 100

eps = np.ones(simulation_size)

eps = eps * 0.5
eps[medium_start:] = 0.5 / epsilon

# Absorbing boundary condition
boundary_low = [0, 0]
boundary_high = [0, 0]

E_frames = []

# Main FDTD
for time_step in range(timesteps):
    # Calculate electric field values
    Ex[1:] = Ex[1:] + eps[1:] * (Hy[:-1] - Hy[1:])

    # Pulse signal
    pulse = np.sin(2 * np.pi * freq_in * dt * time_step)
    Ex[j_source] = pulse + Ex[j_source]

    # Absorbing boundary condition
    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[-1] = boundary_high.pop(0)
    boundary_high.append(Ex[-2])

    # Calculate magnetic field values
    Hy[:-1] = Hy[:-1] + 0.5 * (Ex[:-1] - Ex[1:])

    E_frames.append(Ex.copy())

# Animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    (im,) = ax.plot(frame, color="blue")
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
