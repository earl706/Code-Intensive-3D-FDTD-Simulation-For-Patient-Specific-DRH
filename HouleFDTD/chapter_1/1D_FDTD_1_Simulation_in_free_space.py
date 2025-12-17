"""
1D FDTD Simulation in free space
"""

import numpy as np
from matplotlib import pyplot as plt, animation

# Initialize simulation size, electric and magnetic field values
simulation_size = 200
Ex = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)

# Pulse parameters
j_source = int(simulation_size / 2)
pulse_width = 12
pulse_delay = 40
timesteps = 1000

E_frames = []

# Main FDTD Loop
for time_step in range(1, timesteps + 1):
    # Calculate the Ex field
    Ex[1:] = Ex[1:] + 0.5 * (Hy[:-1] - Hy[1:])

    # Put a Gaussian pulse in the middle
    pulse = np.exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Ex[j_source] = pulse

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
