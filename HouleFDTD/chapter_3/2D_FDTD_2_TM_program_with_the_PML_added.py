"""fd2d_3_3.py: 2D FDTD
TM program with PML added
"""

import numpy as np
from math import sin, pi
from matplotlib import pyplot as plt, animation
import mpl_toolkits.mplot3d.axes3d

# simulation size
simulation_size_x = 500
simulation_size_y = 500

# source position
source_x = int(simulation_size_x / 2)
source_y = int(simulation_size_y / 2)

# field values
Ez = np.zeros((simulation_size_x, simulation_size_y))
Dz = np.zeros((simulation_size_x, simulation_size_y))
Hx = np.zeros((simulation_size_x, simulation_size_y))
Hy = np.zeros((simulation_size_x, simulation_size_y))
iHx = np.zeros((simulation_size_x, simulation_size_y))
iHy = np.zeros((simulation_size_x, simulation_size_y))

# step size and time step
dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size

# Create Dielectric Profile
eps0 = 8.854e-12

# Pulse Parameters
pulse_width = 12
pulse_delay = 40
eps = np.ones((simulation_size_x, simulation_size_y))

# Calculate the PML parameters
gi2 = np.ones(simulation_size_x)
gi3 = np.ones(simulation_size_x)
fi1 = np.zeros(simulation_size_x)
fi2 = np.ones(simulation_size_x)
fi3 = np.ones(simulation_size_x)
gj2 = np.ones(simulation_size_x)
gj3 = np.ones(simulation_size_x)
fj1 = np.zeros(simulation_size_x)
fj2 = np.ones(simulation_size_x)
fj3 = np.ones(simulation_size_x)

# Create PML
npml = 8
n = np.arange(npml)
xnum = npml - n
xd = npml
xxn = xnum / xd
xn = 0.33 * xxn**3

i1 = n
i2 = simulation_size_x - 1 - n
j1 = n
j2 = simulation_size_y - 1 - n

gi2[i1] = 1 / (1 + xn)
gi2[i2] = 1 / (1 + xn)
gi3[i1] = (1 - xn) / (1 + xn)
gi3[i2] = (1 - xn) / (1 + xn)

gj2[j1] = 1 / (1 + xn)
gj2[j2] = 1 / (1 + xn)
gj3[j1] = (1 - xn) / (1 + xn)
gj3[j2] = (1 - xn) / (1 + xn)

xxn_shifted = (xnum - 0.5) / xd
xn_shifted = 0.33 * xxn_shifted**3

i1_shift = n
i2_shift = simulation_size_x - 2 - n
j1_shift = n
j2_shift = simulation_size_y - 2 - n

fi1[i1_shift] = xn_shifted
fi1[i2_shift] = xn_shifted
fi2[i1_shift] = 1 / (1 + xn_shifted)
fi2[i2_shift] = 1 / (1 + xn_shifted)
fi3[i1_shift] = (1 - xn_shifted) / (1 + xn_shifted)
fi3[i2_shift] = (1 - xn_shifted) / (1 + xn_shifted)

fj1[j1_shift] = xn_shifted
fj1[j2_shift] = xn_shifted
fj2[j1_shift] = 1 / (1 + xn_shifted)
fj2[j2_shift] = 1 / (1 + xn_shifted)
fj3[j1_shift] = (1 - xn_shifted) / (1 + xn_shifted)
fj3[j2_shift] = (1 - xn_shifted) / (1 + xn_shifted)

time_steps = 2000
E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate Dz
    Dz[1:, 1:] = gi3[1:] * gj3[1:] * Dz[1:, 1:] + gi2[1:] * gj2[1:] * 0.5 * (
        Hy[1:, 1:] - Hy[:-1, 1:] - Hx[1:, 1:] + Hx[1:, :-1]
    )

    # Put a sinusoid wave in the middle
    pulse = sin(2 * pi * 1500e6 * dt * time_step)
    Dz[source_x, source_y] = pulse

    # Calculate the Ez field from Dz
    Ez = eps * Dz

    # Calculate the Hx field
    curl_e = Ez[:-1, :-1] - Ez[:-1, 1:]
    iHx[:-1, :-1] = iHx[:-1, :-1] + curl_e
    Hx[:-1, :-1] = fj3[1:] * Hx[:-1, :-1] + fj2[:-1] * (
        0.5 * curl_e + fi1[:-1] * iHx[:-1, :-1]
    )

    # Calculate the Hy field
    curl_e = Ez[:-1, :-1] - Ez[1:, :-1]
    iHy[:-1, :-1] = iHy[:-1, :-1] + curl_e
    Hy[:-1, :-1] = fi3[:-1] * Hy[:-1, :-1] - fi2[:-1] * (
        0.5 * curl_e + fj1[:-1] * iHy[:-1, :-1]
    )

    E_frames.append(Ez.copy())

# animate
frames = []
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    im = ax.imshow(frame, cmap="jet")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
