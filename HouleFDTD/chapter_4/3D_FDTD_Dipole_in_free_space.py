"""
Dipole in free space
"""

import numpy as np
from math import exp
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numba

# simulation size
simulation_size_x = 120
simulation_size_y = 120
simulation_size_z = 120

# source position
source_x = int(simulation_size_x / 2)
source_y = int(simulation_size_y / 2)
source_z = int(simulation_size_z / 2)

# field values
Ex = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ey = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Ez = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Dz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
Hz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
eps_x = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
eps_y = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
eps_z = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))

# step size, time step
ddx = 0.01  # Cell size
dt = ddx / 6e8  # Time step size
eps0 = 8.854e-12

# Specify the dipole
eps_z[source_x, source_y, source_z - 10 : source_z + 10] = 0
eps_z[source_x, source_y, source_z] = 1

# Pulse Parameters
pulse_delay = 20
pulse_width = 6
time_steps = 1000


# Functions for Main FDTD Loop
@numba.jit(nopython=True)
def calculate_d_fields(
    simulation_size_x, simulation_size_y, simulation_size_z, Dx, Dy, Dz, Hx, Hy, Hz
):
    """Calculate the Dx, Dy, and Dz fields"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                Dx[i, j, k] = Dx[i, j, k] + 0.5 * (
                    Hz[i, j, k] - Hz[i, j - 1, k] - Hy[i, j, k] + Hy[i, j, k - 1]
                )
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                Dy[i, j, k] = Dy[i, j, k] + 0.5 * (
                    Hx[i, j, k] - Hx[i, j, k - 1] - Hz[i, j, k] + Hz[i - 1, j, k]
                )
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                Dz[i, j, k] = Dz[i, j, k] + 0.5 * (
                    Hy[i, j, k] - Hy[i - 1, j, k] - Hx[i, j, k] + Hx[i, j - 1, k]
                )
    return Dx, Dy, Dz


@numba.jit(nopython=True)
def calculate_e_fields(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dx,
    Dy,
    Dz,
    eps_x,
    eps_y,
    eps_z,
    Ex,
    Ey,
    Ez,
):
    """Calculate the E field from the D field"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex[i, j, k] = eps_x[i, j, k] * Dx[i, j, k]
                Ey[i, j, k] = eps_y[i, j, k] * Dy[i, j, k]
                Ez[i, j, k] = eps_z[i, j, k] * Dz[i, j, k]
    # Ex[:, :, :] = eps_x[:, :, :] * Dx[:, :, :]
    # Ey[:, :, :] = eps_y[:, :, :] * Dy[:, :, :]
    # Ez[:, :, :] = eps_z[:, :, :] * Dz[:, :, :]
    return Ex, Ey, Ez


@numba.jit(nopython=True)
def calculate_h_fields(
    simulation_size_x, simulation_size_y, simulation_size_z, Hx, Hy, Hz, Ex, Ey, Ez
):
    """Calculate the Hx, Hy, and Hz fields"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y - 1):
            for k in range(0, simulation_size_z - 1):
                Hx[i, j, k] = Hx[i, j, k] + 0.5 * (
                    Ey[i, j, k + 1] - Ey[i, j, k] - Ez[i, j + 1, k] + Ez[i, j, k]
                )
    for i in range(0, simulation_size_x - 1):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z - 1):
                Hy[i, j, k] = Hy[i, j, k] + 0.5 * (
                    Ez[i + 1, j, k] - Ez[i, j, k] - Ex[i, j, k + 1] + Ex[i, j, k]
                )
    for i in range(0, simulation_size_x - 1):
        for j in range(0, simulation_size_y - 1):
            for k in range(0, simulation_size_z):
                Hz[i, j, k] = Hz[i, j, k] + 0.5 * (
                    Ex[i, j + 1, k] - Ex[i, j, k] - Ey[i + 1, j, k] + Ey[i, j, k]
                )
    return Hx, Hy, Hz


E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Calculate the D Fields
    Dx, Dy, Dz = calculate_d_fields(
        simulation_size_x, simulation_size_y, simulation_size_z, Dx, Dy, Dz, Hx, Hy, Hz
    )

    # Add the source at the gap
    pulse = exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Dz[source_x, source_y, source_z] = pulse

    # Calculate the E field from the D field
    Ex, Ey, Ez = calculate_e_fields(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        Dx,
        Dy,
        Dz,
        eps_x,
        eps_y,
        eps_z,
        Ex,
        Ey,
        Ez,
    )

    # Calculate the H fields
    Hx, Hy, Hz = calculate_h_fields(
        simulation_size_x, simulation_size_y, simulation_size_z, Hx, Hy, Hz, Ex, Ey, Ez
    )

    if time_step % 5 == 0:
        print(time_step)
        print(np.min(Ez), np.max(Ez))
    E_frames.append(Ez.copy())

# animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    im = ax.imshow(frame[:, :, simulation_size_x // 2], cmap="jet")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
