"""fd2d_3_4.py: 2D FDTD
TM simulation of a plane wave source impinging on a dielectric cylinder
Analysis using fourier transform
"""

import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

# simulation size
simulation_size_x = 500
simulation_size_y = 500

# medium position
medium_x = int(simulation_size_x / 2 - 1)
medium_y = int(simulation_size_y / 2 - 1)
ia = 7
ja = 7
ib = simulation_size_x - ia - 1
jb = simulation_size_y - ja - 1

# field values
Ez = np.zeros((simulation_size_x, simulation_size_y))
Dz = np.zeros((simulation_size_x, simulation_size_y))
Iz = np.zeros((simulation_size_x, simulation_size_y))
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

# fourier transform
number_of_frequencies = 3
freq = np.array((50e6, 300e6, 700e6))
arg = 2 * np.pi * freq * dt
real_pt = np.zeros((number_of_frequencies, simulation_size_x, simulation_size_y))
imag_pt = np.zeros((number_of_frequencies, simulation_size_x, simulation_size_y))
real_in = np.zeros(number_of_frequencies)
imag_in = np.zeros(number_of_frequencies)
amp = np.zeros((number_of_frequencies, simulation_size_y))
phase = np.zeros((number_of_frequencies, simulation_size_y))

# specify the dielectric cylinder
epsr = 12.11
sigma = 0.00001
radius = simulation_size_x // 4

# create dielectric profile
eps0 = 8.85418781e-12

inv_eps = np.ones((simulation_size_x, simulation_size_y))
normalized_conductivity = np.zeros((simulation_size_x, simulation_size_y))

i_vals = np.arange(ia, ib)
j_vals = np.arange(ja, jb)
ii, jj = np.meshgrid(i_vals, j_vals, indexing="ij")

xdist = medium_x - ii
ydist = medium_y - jj
dist = np.sqrt(xdist**2 + ydist**2)

mask = dist <= radius

inv_eps[ia:ib, ja:jb][mask] = 1 / (epsr + (sigma * dt / eps0))
normalized_conductivity[ia:ib, ja:jb][mask] = sigma * dt / eps0

# absorbing boundary conditions
boundary_low = [0, 0]
boundary_high = [0, 0]

# calculate the PML parameters
gi2 = np.ones(simulation_size_x)
gi3 = np.ones(simulation_size_x)
fi1 = np.zeros(simulation_size_x)
fi2 = np.ones(simulation_size_x)
fi3 = np.ones(simulation_size_x)
gj2 = np.ones(simulation_size_y)
gj3 = np.ones(simulation_size_y)
fj1 = np.zeros(simulation_size_y)
fj2 = np.ones(simulation_size_y)
fj3 = np.ones(simulation_size_y)

# create the PML
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

# pulse parameters
pulse_delay = 20
pulse_width = 30
time_steps = 2000

E_frames = []

# Main FDTD Loop
for time_step in range(1, time_steps + 1):
    # Incident Ez values
    Ez_inc[1:simulation_size_y] = Ez_inc[1:simulation_size_y] + 0.5 * (
        Hx_inc[: simulation_size_y - 1] - Hx_inc[1:simulation_size_y]
    )
    # if time_step < 3 * pulse_delay:
    #     for m in range(number_of_frequencies):
    #         real_in[m] = real_in[m] + cos(arg[m] * time_step) * Ez_inc[ja - 1]
    #         imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * Ez_inc[ja - 1]

    # Absorbing Boundary Conditions
    Ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(Ez_inc[1])
    Ez_inc[simulation_size_y - 1] = boundary_high.pop(0)
    boundary_high.append(Ez_inc[simulation_size_y - 2])

    # Calculate the Dz field
    Dz[1:simulation_size_x, 1:simulation_size_y] = gi3[1:simulation_size_x, None] * gj3[
        None, 1:simulation_size_y
    ] * Dz[1:simulation_size_x, 1:simulation_size_y] + gi2[
        1:simulation_size_x, None
    ] * gj2[
        None, 1:simulation_size_y
    ] * 0.5 * (
        Hy[1:simulation_size_x, 1:simulation_size_y]
        - Hy[0 : simulation_size_x - 1, 1:simulation_size_y]
        - Hx[1:simulation_size_x, 1:simulation_size_y]
        + Hx[1:simulation_size_x, 0 : simulation_size_y - 1]
    )

    # Source
    pulse = np.exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Ez_inc[3] = pulse

    # Incident Dz values
    Dz[ia : ib + 1, ja] += 0.5 * Hx_inc[ja - 1]
    Dz[ia : ib + 1, jb] -= 0.5 * Hx_inc[jb]

    # Calculate the Ez field
    Ez[:, :] = inv_eps * (Dz - Iz)
    Iz[:, :] += normalized_conductivity * Ez

    # Calculate the fourier transform of Ex
    # for j in range(0, simulation_size_y):
    #     for i in range(0, simulation_size_x):
    #         for m in range(0, number_of_frequencies):
    #             real_pt[m, i, j] = real_pt[m, i, j] + cos(arg[m] * time_step) * Ez[i, j]
    #             imag_pt[m, i, j] = imag_pt[m, i, j] - sin(arg[m] * time_step) * Ez[i, j]

    # Calculate the Incident Hx
    Hx_inc[: simulation_size_y - 1] += 0.5 * (
        Ez_inc[: simulation_size_y - 1] - Ez_inc[1:simulation_size_y]
    )

    # Calculate the Hx field
    curl_e = (
        Ez[: simulation_size_x - 1, : simulation_size_y - 1]
        - Ez[: simulation_size_x - 1, 1:simulation_size_y]
    )
    iHx[: simulation_size_x - 1, : simulation_size_y - 1] += curl_e
    Hx[: simulation_size_x - 1, : simulation_size_y - 1] = fj3[
        None, : simulation_size_y - 1
    ] * Hx[: simulation_size_x - 1, : simulation_size_y - 1] + fj2[
        None, : simulation_size_y - 1
    ] * (
        0.5 * curl_e
        + fi1[: simulation_size_x - 1, None]
        * iHx[: simulation_size_x - 1, : simulation_size_y - 1]
    )

    # Incident Hx values
    Hx[ia : ib + 1, ja - 1] += 0.5 * Ez_inc[ja]
    Hx[ia : ib + 1, jb] -= 0.5 * Ez_inc[jb]

    # Calculate the Hy field
    curl_e = Ez[: simulation_size_x - 1, :] - Ez[1:simulation_size_x, :]
    iHy[: simulation_size_x - 1, :] += curl_e
    Hy[: simulation_size_x - 1, :] = fi3[: simulation_size_x - 1, None] * Hy[
        : simulation_size_x - 1, :
    ] - fi2[: simulation_size_x - 1, None] * (
        0.5 * curl_e + fj1[None, :] * iHy[: simulation_size_x - 1, :]
    )

    # Incident Hy values
    Hy[ia - 1, ja : jb + 1] -= 0.5 * Ez_inc[ja : jb + 1]
    Hy[ib, ja : jb + 1] += 0.5 * Ez_inc[ja : jb + 1]
    if time_step % 10 == 0:
        print(time_step)
        print(np.min(Ez), np.max(Ez))
    E_frames.append(Ez.copy())

E_enc = np.max(np.abs(E_frames))

E_frames = np.array(E_frames)
E_frames = E_frames / E_enc

# animate
frames = []
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.rcParams["font.size"] = 12
plt.rcParams["grid.color"] = "gray"
plt.rcParams["grid.linestyle"] = "dotted"
im = ax.imshow(E_frames[0], cmap="jet", vmin=np.min(E_frames), vmax=np.max(E_frames))
cbar = fig.colorbar(im)

for frame in E_frames:
    im = ax.imshow(frame, cmap="jet")
    frames.append([im])
ax.set_xlabel("x")
ax.set_ylabel("y")
ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True, repeat_delay=1000)
plt.show()
