import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# define signal
def signal(time, pulse_width, pulse_delay, omega0, amplitude=1.0):
    return (
        amplitude
        * (np.exp(-(((time - pulse_delay) / pulse_width) ** 2)))
        * (np.sin(omega0 * time))
    )


# define constants
mu0 = 1.256637062e-6  # permeability of free space
eps0 = 8.8541878128e-12  # permittivity of free space
c0 = 2.99792458e8  # speed of light in vacuum
imp0 = np.sqrt(mu0 / eps0)  # impedance of free space

# simulation size, step size, N space cells
N_space_cells_y = 2e-6
N_space_cells_x = 2e-6
step_size = 5e-9  # dy
N_space_cells_x = int(N_space_cells_x / step_size)  # jmax
N_space_cells_y = int(N_space_cells_y / step_size)  # jmax
print(f"there are {N_space_cells_x*N_space_cells_y} FDTD cells")

# time step, simulation time, N time steps
simulation_time = 1e-13
dt = step_size / c0
N_time_steps = int(simulation_time / dt)
print(f"there are {N_time_steps} FDTD time steps")

medium_x = int(N_space_cells_x / 2 - 1)
medium_y = int(N_space_cells_y / 2 - 1)
ia = 7
ja = 7
ib = N_space_cells_x - ia - 1
jb = N_space_cells_y - ja - 1

# initialize electric and magnetic field values
Ez = np.zeros((N_space_cells_x, N_space_cells_y))
Dz = np.zeros((N_space_cells_x, N_space_cells_y))
Iz = np.zeros((N_space_cells_x, N_space_cells_y))
Hx = np.zeros((N_space_cells_x, N_space_cells_y))
Hy = np.zeros((N_space_cells_x, N_space_cells_y))
iHx = np.zeros((N_space_cells_x, N_space_cells_y))
iHy = np.zeros((N_space_cells_x, N_space_cells_y))

Ez_inc = np.zeros(N_space_cells_x)
Hx_inc = np.zeros(N_space_cells_x)

# material properties and coordinates
# specify the dielectric cylinder
epsr = 4
sigma = 40000
tau = 0.001 * 1e-6  # relaxation time constant
chi = 2  # electric susceptibility
radius = N_space_cells_x // 4

# create dielectric profile
eps0 = 8.85418781e-12
eps = np.ones((N_space_cells_x, N_space_cells_y))

inv_eps = np.ones((N_space_cells_x, N_space_cells_y))
conductivity_correction = np.zeros((N_space_cells_x, N_space_cells_y))

i_vals = np.arange(ia, ib)
j_vals = np.arange(ja, jb)
ii, jj = np.meshgrid(i_vals, j_vals, indexing="ij")

xdist = medium_x - ii
ydist = medium_y - jj
dist = np.sqrt(xdist**2 + ydist**2)

mask = dist <= radius

inv_eps[ia:ib, ja:jb][mask] = 1 / (epsr + (sigma * dt / eps0))
conductivity_correction[ia:ib, ja:jb][mask] = sigma * dt / eps0

# create dielectric profile
refractive_index = np.sqrt(eps)

# electric and magnetic field coefficients
h_coeff = dt / (mu0 * step_size)
e_coeff = dt / (eps0 * eps * step_size)

# absorbing boundary conditions
c = c0 / refractive_index[0, 0]
c_ = c0 / refractive_index[0, -1]
a = (c * dt - step_size) / (c * dt + step_size)
a_ = (c_ * dt - step_size) / (c_ * dt + step_size)

# pulse stuff
center_wavelength = 775e-9
omega0 = 2 * np.pi * c0 / center_wavelength
pulse_width = 5e-16
pulse_delay = 4 * pulse_width

# initialize time and pulse signal
time = np.linspace(0, simulation_time, N_time_steps)
pulse = signal(time, pulse_width, pulse_delay, omega0)

# source parameters
j_source = 5
t_offset = refractive_index[0, j_source] * step_size / (2 * c0)
Z = imp0 / refractive_index[0, j_source]

E_movie = []

# Calculate the PML parameters
gi2 = np.ones(N_space_cells_x)
gi3 = np.ones(N_space_cells_x)
fi1 = np.zeros(N_space_cells_x)
fi2 = np.ones(N_space_cells_x)
fi3 = np.ones(N_space_cells_x)
gj2 = np.ones(N_space_cells_x)
gj3 = np.ones(N_space_cells_x)
fj1 = np.zeros(N_space_cells_x)
fj2 = np.ones(N_space_cells_x)
fj3 = np.ones(N_space_cells_x)

# Create the PML
npml = 8
n = np.arange(npml)
xnum = npml - n
xd = npml
xxn = xnum / xd
xn = 0.33 * xxn**3

i1 = n
i2 = N_space_cells_x - 1 - n
j1 = n
j2 = N_space_cells_y - 1 - n

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
i2_shift = N_space_cells_x - 2 - n
j1_shift = n
j2_shift = N_space_cells_y - 2 - n

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

boundary_low = [0, 0]
boundary_high = [0, 0]

# Fourier monitor
jR = j_source - 5
jT = N_space_cells_x - 5
ER = np.zeros(N_time_steps)
ET = np.zeros(N_time_steps)
E_frames = []

# FDTD algorithm
for time_step in range(N_time_steps):
    Ez_inc_prev = Ez_inc.copy()
    # Incident Ez values
    Ez_inc[1:N_space_cells_y] = Ez_inc[1:N_space_cells_y] + 0.5 * (
        Hx_inc[: N_space_cells_y - 1] - Hx_inc[1:N_space_cells_y]
    )
    # if time_step < 3 * pulse_delay:
    #     for m in range(number_of_frequencies):
    #         real_in[m] = real_in[m] + cos(arg[m] * time_step) * Ez_inc[ja - 1]
    #         imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * Ez_inc[ja - 1]

    # Absorbing Boundary Conditions
    Ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(Ez_inc[1])
    Ez_inc[N_space_cells_y - 1] = boundary_high.pop(0)
    boundary_high.append(Ez_inc[N_space_cells_y - 2])

    # Calculate the Dz field
    Dz[1:N_space_cells_x, 1:N_space_cells_y] = gi3[1:N_space_cells_x, None] * gj3[
        None, 1:N_space_cells_y
    ] * Dz[1:N_space_cells_x, 1:N_space_cells_y] + gi2[1:N_space_cells_x, None] * gj2[
        None, 1:N_space_cells_y
    ] * 0.5 * (
        Hy[1:N_space_cells_x, 1:N_space_cells_y]
        - Hy[0 : N_space_cells_x - 1, 1:N_space_cells_y]
        - Hx[1:N_space_cells_x, 1:N_space_cells_y]
        + Hx[1:N_space_cells_x, 0 : N_space_cells_y - 1]
    )

    # Source
    pulse = signal(time_step * dt, pulse_width, pulse_delay, omega0)
    Ez_inc[3] += pulse

    # Incident Dz values
    Dz[ia : ib + 1, ja] += 0.5 * Hx_inc[ja - 1]
    Dz[ia : ib + 1, jb] -= 0.5 * Hx_inc[jb]

    # Calculate the Ez field
    Ez[:, :] = inv_eps * (Dz - Iz)
    Iz[:, :] += conductivity_correction * Ez

    # Calculate the fourier transform of Ex
    # for j in range(0, N_space_cells_y):
    #     for i in range(0, N_space_cells_x):
    #         for m in range(0, number_of_frequencies):
    #             real_pt[m, i, j] = real_pt[m, i, j] + cos(arg[m] * time_step) * Ez[i, j]
    #             imag_pt[m, i, j] = imag_pt[m, i, j] - sin(arg[m] * time_step) * Ez[i, j]

    # Calculate the Incident Hx
    Hx_inc[: N_space_cells_y - 1] += 0.5 * (
        Ez_inc[: N_space_cells_y - 1] - Ez_inc[1:N_space_cells_y]
    )

    # Calculate the Hx field
    curl_e = (
        Ez[: N_space_cells_x - 1, : N_space_cells_y - 1]
        - Ez[: N_space_cells_x - 1, 1:N_space_cells_y]
    )
    iHx[: N_space_cells_x - 1, : N_space_cells_y - 1] += curl_e
    Hx[: N_space_cells_x - 1, : N_space_cells_y - 1] = fj3[
        None, : N_space_cells_y - 1
    ] * Hx[: N_space_cells_x - 1, : N_space_cells_y - 1] + fj2[
        None, : N_space_cells_y - 1
    ] * (
        0.5 * curl_e
        + fi1[: N_space_cells_x - 1, None]
        * iHx[: N_space_cells_x - 1, : N_space_cells_y - 1]
    )

    # Incident Hx values
    Hx[ia : ib + 1, ja - 1] += 0.5 * Ez_inc[ja]
    Hx[ia : ib + 1, jb] -= 0.5 * Ez_inc[jb]

    # Calculate the Hy field
    curl_e = Ez[: N_space_cells_x - 1, :] - Ez[1:N_space_cells_x, :]
    iHy[: N_space_cells_x - 1, :] += curl_e
    Hy[: N_space_cells_x - 1, :] = fi3[: N_space_cells_x - 1, None] * Hy[
        : N_space_cells_x - 1, :
    ] - fi2[: N_space_cells_x - 1, None] * (
        0.5 * curl_e + fj1[None, :] * iHy[: N_space_cells_x - 1, :]
    )

    # Incident Hy values
    Hy[ia - 1, ja : jb + 1] -= 0.5 * Ez_inc[ja : jb + 1]
    Hy[ib, ja : jb + 1] += 0.5 * Ez_inc[ja : jb + 1]
    if time_step % 6 == 0:
        print(time_step)
        print(np.min(Ez), np.max(Ez))
        E_frames.append(Ez.copy())

# # plot transmittance and reflectance
# plt.plot(ET, label="Transmitted")
# plt.plot(ER, label="Reflected")
# plt.legend()
# plt.show()

# # wavelengths,omegas, time
# wavelengths = np.linspace(center_wavelength - 100e-9, center_wavelength + 100e-9, 100)
# omegas = 2 * np.pi * c0 / wavelengths
# time = np.arange(N_time_steps) * dt


# # function for discrete fourier transform
# def Discrete_Fourier_Transform(field, time, omega):
#     N_freq = omega.shape[0]
#     field_omega = np.zeros(N_freq, dtype="complex128")
#     for w in range(N_freq):
#         field_omega[w] = np.sum(field * np.exp(1j * omega[w] * time))
#     return field_omega


# # function for thin film transmittance
# def thin_film_TR(n1, n2, n3, wavelengths, thickness):
#     r12 = (n1 - n2) / (n1 + n2)
#     r23 = (n2 - n3) / (n2 + n3)
#     t12 = 2 * n1 / (n1 + n2)
#     t23 = 2 * n2 / (n2 + n3)
#     beta = 2 * np.pi * n2 * thickness / wavelengths
#     r = (r12 + r23 * np.exp(-2j * beta)) / (1.0 + r12 * r23 * np.exp(-2j * beta))
#     t = (t12 * t23 * np.exp(-1j * beta)) / (1.0 + r12 * r23 * np.exp(-2j * beta))
#     return (n3 / n1) * np.abs(t) ** 2, np.abs(r) ** 2


# # pulse signal
# pulse = signal(time, pulse_width, pulse_delay, omega0)

# # calculate electric field transmittance and reflectance
# ET_FT = Discrete_Fourier_Transform(ET, time, omegas)
# ER_FT = Discrete_Fourier_Transform(ER, time, omegas)
# pulse_FT = Discrete_Fourier_Transform(pulse, time, omegas)

# # calculate reflectance and transmittance
# R = np.abs(ER_FT) ** 2 / np.abs(pulse_FT) ** 2
# T = (
#     np.abs(ET_FT) ** 2
#     / np.abs(pulse_FT) ** 2
#     * refractive_index[jT]
#     / refractive_index[j_source]
# )

# T_, R_ = thin_film_TR(1.0, np.sqrt(12.11), np.sqrt(4), wavelengths, thin_film_thickness)

# # plot wavelengths, transmittance, and reflectance
# plt.plot(wavelengths, R, "blue")
# plt.plot(wavelengths, R_, "x", color="blue")
# plt.plot(wavelengths, T, "red")
# plt.plot(wavelengths, T_, "x", color="red")
# plt.plot(wavelengths, 1 - R - T, "violet")
# plt.xlabel("wavelengths (m)")
# plt.ylabel("Spectrum")
# plt.legend(["Reflectance", "R-analytic", "Transmittance", "T-analytic", "Absorption"])
# plt.show()

# animate
frames = []  # for storing the generated images
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(E_frames[0], cmap="jet", vmin=np.min(E_frames), vmax=np.max(E_frames))
cbar = fig.colorbar(im)
cbar.set_label("Ez")
ax.set_xlabel("x")
ax.set_ylabel("y")

for frame in E_frames:
    im = ax.imshow(frame, cmap="jet")
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True, repeat_delay=1000)
plt.show()
# ani.save("2D_lossy_dielectric_medium.mp4", writer="ffmpeg", fps=60)
