"""
1D FDTD Simulation of a sinusoid wave hitting a lossy dielectric medium
"""

import numpy as np
from matplotlib import pyplot as plt, animation

# initialize simulation size, magnetic and electric field values
simulation_size = 200
Ex = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)

# step size, time step, frequencies
dx = 0.01
dt = dx / 6e8
freq_in = 700e6

# pulse parameters
j_source = 10
pulse_width = 12
pulse_delay = 50
time_steps = 2000

# create dielectric profile
eps0 = 8.8541878128e-12
eps_r = 4
sigma = 0.05
medium_start = 5

eps = np.ones(simulation_size) * 0.5
conductivity_correction = np.ones(simulation_size)
conductivity_parameter = dt * sigma / (2 * eps0 * eps_r)

conductivity_correction[medium_start:] = (1 - conductivity_parameter) / (
    1 + conductivity_parameter
)
eps[medium_start:] = 0.5 / eps_r

# absorbing boundary conditions
boundary_low = [0, 0]
boundary_high = [0, 0]

plt.plot(conductivity_correction)
plt.show()

E_frames = []

# main FDTD loop
for time_step in range(time_steps):
    # calculate electric field
    Ex[1:] = conductivity_correction[1:] * Ex[1:] + eps[1:] * (Hy[:-1] - Hy[1:])

    # pulse
    pulse = np.sin(2 * np.pi * freq_in * time_step * dt)
    Ex[j_source] = pulse + Ex[j_source]

    # absorbing boundary conditions update
    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[-1] = boundary_high.pop(0)
    boundary_high.append(Ex[-2])

    # calculate magnetic field
    Hy[:-1] = Hy[:-1] + 0.5 * (Ex[:-1] - Ex[1:])

    E_frames.append(Ex.copy())


# animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    (im,) = ax.plot(frame, color="blue")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
