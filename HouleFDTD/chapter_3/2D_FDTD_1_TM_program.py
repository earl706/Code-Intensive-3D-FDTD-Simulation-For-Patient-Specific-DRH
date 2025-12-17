"""fd2d_3_1.py: 2D FDTD
TM program
"""

import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

# simulation size
simulation_size_x = 120
simulation_size_y = 120
time_steps = 1000

# source position
source_x = int(simulation_size_x / 2)
source_y = int(simulation_size_y / 2)

# field values
Ez = np.zeros((simulation_size_x, simulation_size_y))
Dz = np.zeros((simulation_size_x, simulation_size_y))
Hx = np.zeros((simulation_size_x, simulation_size_y))
Hy = np.zeros((simulation_size_x, simulation_size_y))

# step size, time step
dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size

# create dielectric profile
eps0 = 8.854e-12
eps = np.ones((simulation_size_x, simulation_size_y))

# pulse parameters
pulse_width = 6
pulse_delay = 20

E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate Dz
    Dz[1:, 1:] = Dz[1:, 1:] + 0.5 * (
        Hy[1:, 1:] - Hy[:-1, 1:] - Hx[1:, 1:] + Hx[1:, :-1]
    )

    # Put a Gaussian pulse in the middle
    pulse = np.exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Dz[source_x, source_y] = pulse

    # Calculate the Ez field from Dz
    Ez[1:, 1:] = eps[1:, 1:] * Dz[1:, 1:]

    # Calculate the Hx field
    Hx[:-1, :-1] = Hx[:-1, :-1] + 0.5 * (Ez[:-1, :-1] - Ez[:-1, 1:])

    # Calculate the Hy field
    Hy[:-1, :-1] = Hy[:-1, :-1] + 0.5 * (Ez[1:, :-1] - Ez[:-1, :-1])
    if time_step % 5 == 0:
        E_frames.append(Ez.copy())

# animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    im = ax.imshow(frame, cmap="jet")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
