"""
1D FDTD Simulation of a pulse hitting a dielectric medium
"""

import numpy as np
from matplotlib import pyplot as plt, animation

# Initialize simulation size, electric and magnetic field values
simulation_size = 200
Ex = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)
pulse_width = 12
pulse_delay = 40
time_steps = 1000

# Create Dielectric Profile
epsilon = 4
medium_start = 100

eps = np.ones(simulation_size)

eps = 0.5 * eps
eps[medium_start:] = 0.5 / epsilon

# Absorbing boundary condition
boundary_low = [0, 0]
boundary_high = [0, 0]

E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate the Ex field
    Ex[1:] = Ex[1:] + eps[1:] * (Hy[:-1] - Hy[1:])

    # Put a Gaussian pulse at the low end
    pulse = np.exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Ex[5] = pulse + Ex[5]

    # Absorbing Boundary Conditions
    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[-1] = boundary_high.pop(0)
    boundary_high.append(Ex[-2])

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

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
