"""fd2d_3_3.py: 2D FDTD
TM program with plane wave source
"""

import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

# simulation size
simulation_size_x = 200
simulation_size_y = 200

# source position
source_x = int(simulation_size_x / 2)
source_y = int(simulation_size_y / 2)
ia = 7
ja = 7
ib = simulation_size_x - ia - 1
jb = simulation_size_y - ja - 1

# field values
Ez = np.zeros((simulation_size_x, simulation_size_y))
Dz = np.zeros((simulation_size_x, simulation_size_y))
Hx = np.zeros((simulation_size_x, simulation_size_y))
Hy = np.zeros((simulation_size_x, simulation_size_y))
iHx = np.zeros((simulation_size_x, simulation_size_y))
iHy = np.zeros((simulation_size_x, simulation_size_y))

# electric and magnetic incident
Ez_inc = np.zeros(simulation_size_y)
Hx_inc = np.zeros(simulation_size_y)

# step size, time step
dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size

# Create Dielectric Profile
eps0 = 8.8541878162e-12

# Pulse Parameters
pulse_width = 8
pulse_delay = 20
eps = np.ones((simulation_size_x, simulation_size_y))
time_steps = 2000

# Absorbing Boundary Conditions
boundary_low = [0, 0]
boundary_high = [0, 0]

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

# Create the PML
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

E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Incident Ez values
    Ez_inc[1:] = Ez_inc[1:] + 0.5 * (Hx_inc[:-1] - Hx_inc[1:])

    # Absorbing Boundary Conditions
    Ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(Ez_inc[1])
    Ez_inc[simulation_size_y - 1] = boundary_high.pop(0)
    boundary_high.append(Ez_inc[simulation_size_y - 2])

    # Calculate the Dz field
    Dz[1:, 1:] = gi3[1:] * gj3[1:] * Dz[1:, 1:] + gi2[1:] * gj2[1:] * 0.5 * (
        Hy[1:, 1:] - Hy[:-1, 1:] - Hx[1:, 1:] + Hx[1:, :-1]
    )

    # Put a Gaussian pulse at low end
    pulse = np.exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Ez_inc[3] = pulse

    # Incident Dz values
    Dz[ia:ib, ja] = Dz[ia:ib, ja] + 0.5 * Hx_inc[ja - 1]
    Dz[ia:ib, jb] = Dz[ia:ib, jb] - 0.5 * Hx_inc[jb - 1]

    # Calculate the Ez field from Dz
    Ez = eps * Dz

    # incident Hx values
    Hx_inc[:-1] = Hx_inc[:-1] + 0.5 * (Ez_inc[:-1] - Ez_inc[1:])
    curl_e = (
        Ez[: simulation_size_x - 1, : simulation_size_y - 1]
        - Ez[: simulation_size_x - 1, 1:simulation_size_y]
    )
    iHx[: simulation_size_x - 1, : simulation_size_y - 1] += curl_e
    Hx[: simulation_size_x - 1, : simulation_size_y - 1] = fj3[
        : simulation_size_y - 1
    ] * Hx[: simulation_size_x - 1, : simulation_size_y - 1] + fj2[
        : simulation_size_y - 1
    ] * (
        0.5 * curl_e
        + fi1[: simulation_size_x - 1, None]
        * iHx[: simulation_size_x - 1, : simulation_size_y - 1]
    )

    # calculate Hx field
    Hx[ia:ib, ja - 1] = Hx[ia:ib, ja - 1] + 0.5 * Ez_inc[ja]
    Hx[ia:ib, jb] = Hx[ia:ib, jb] - 0.5 * Ez_inc[jb]

    # calculate Hy field
    curl_e = (
        Ez[: simulation_size_x - 1, : simulation_size_y - 1]
        - Ez[1:simulation_size_x, : simulation_size_y - 1]
    )

    # incident Hy values
    iHy[: simulation_size_x - 1, : simulation_size_y - 1] += curl_e
    Hy[: simulation_size_x - 1, : simulation_size_y - 1] = fi3[
        : simulation_size_x - 1, None
    ] * Hy[: simulation_size_x - 1, : simulation_size_y - 1] - fi2[
        : simulation_size_x - 1, None
    ] * (
        0.5 * curl_e
        + fj1[: simulation_size_y - 1]
        * iHy[: simulation_size_x - 1, : simulation_size_y - 1]
    )

    # calculate Hy field
    Hy[ia - 1, ja:jb] = Hy[ia - 1, ja:jb] - 0.5 * Ez_inc[ja:jb]
    Hy[ib - 1, ja:jb] = Hy[ib - 1, ja:jb] + 0.5 * Ez_inc[ja:jb]
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
