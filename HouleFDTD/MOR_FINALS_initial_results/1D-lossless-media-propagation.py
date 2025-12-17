import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# define signal
def signal(time, pulse_width, pulse_delay, omega0, amplitude=15.0):
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
simulation_size = 20e-6
step_size = 5e-9  # dy
N_space_cells = int(simulation_size / step_size)  # jmax
print(f"there are {N_space_cells} FDTD cells")

# time step, simulation time, N time steps
simulation_time = 5e-13
dt = step_size / c0
N_time_steps = int(simulation_time / dt)
print(f"there are {N_time_steps} FDTD time steps")

# initialize electric and magnetic field values
Ex = np.zeros(N_space_cells)
Hz = np.zeros(N_space_cells)
eps = np.ones(N_space_cells)

# material properties and coordinates
thin_film_thickness = 5e-6
j_Si_start = N_space_cells // 2
j_Si_end = int(thin_film_thickness / step_size + j_Si_start - 1)
eps[j_Si_start : j_Si_end + 1] = 12.11
eps[j_Si_end + 1 :] = 1

refractive_index = np.sqrt(eps)
plt.plot(eps)
plt.show()

# electric and magnetic field coefficients
h_coeff = dt / (mu0 * step_size)
e_coeff = dt / (eps0 * eps * step_size)

# absorbing boundary conditions
c = c0 / refractive_index[0]
c_ = c0 / refractive_index[-1]
a = (c * dt - step_size) / (c * dt + step_size)
a_ = (c_ * dt - step_size) / (c_ * dt + step_size)

# pulse stuff
center_wavelength = 1550e-9
omega0 = 2 * np.pi * c0 / center_wavelength
pulse_width = 10e-15
pulse_delay = 4 * pulse_width

# initialize time and pulse signal
time = np.linspace(0, simulation_time, N_time_steps)
pulse = signal(time, pulse_width, pulse_delay, omega0)
plt.plot(time, pulse)
plt.show()

# source parameters
j_source = 5
t_offset = refractive_index[j_source] * step_size / (2 * c0)
Z = imp0 / refractive_index[j_source]

E_movie = []

# Fourier monitor
jR = j_source - 5
jT = N_space_cells - 5
ER = np.zeros(N_time_steps)
ET = np.zeros(N_time_steps)

# FDTD algorithm
for n in range(N_time_steps):
    Hz_prev = Hz.copy()
    Ex_prev = Ex.copy()

    # update magnetic field at n+1/2
    Hz[:-1] = Hz_prev[:-1] + h_coeff * (Ex[1:] - Ex[:-1])

    # add magnetic field source
    Hz[j_source - 1] -= (
        signal((n + 0.5) * dt - t_offset, pulse_width, pulse_delay, omega0) / Z
    )

    # update electric field at n+1
    Ex[1:-1] = Ex_prev[1:-1] + e_coeff[1:-1] * (Hz[1:-1] - Hz[:-2])

    # add electric field source
    Ex[j_source] += signal((n + 1) * dt, pulse_width, pulse_delay, omega0)

    # boundary condition update
    Ex[0] = Ex_prev[1] + a * (Ex[1] - Ex_prev[0])
    Ex[-1] = Ex_prev[-2] + a_ * (Ex[-2] - Ex_prev[-1])

    # store reflection and transmission data
    ER[n] = Ex[jR]
    ET[n] = Ex[jT]

    if n % 20 == 0:
        print(n)
        print(np.min(Ex), np.max(Ex))
        E_movie.append(Ex.copy())

# plot transmittance and reflectance
plt.plot(ET, label="Transmitted")
plt.plot(ER, label="Reflected")
plt.legend()
plt.show()

# wavelengths,omegas, time
wavelengths = np.linspace(center_wavelength - 100e-9, center_wavelength + 100e-9, 100)
omegas = 2 * np.pi * c0 / wavelengths
time = np.arange(N_time_steps) * dt


# function for discrete fourier transform
def Discrete_Fourier_Transform(field, time, omega):
    N_freq = omega.shape[0]
    field_omega = np.zeros(N_freq, dtype="complex128")
    for w in range(N_freq):
        field_omega[w] = np.sum(field * np.exp(1j * omega[w] * time))
    return field_omega


# function for thin film transmittance
def thin_film_TR(n1, n2, n3, wavelengths, thickness):
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    beta = 2 * np.pi * n2 * thickness / wavelengths
    r = (r12 + r23 * np.exp(-2j * beta)) / (1.0 + r12 * r23 * np.exp(-2j * beta))
    t = (t12 * t23 * np.exp(-1j * beta)) / (1.0 + r12 * r23 * np.exp(-2j * beta))
    return (n3 / n1) * np.abs(t) ** 2, np.abs(r) ** 2


# pulse signal
pulse = signal(time, pulse_width, pulse_delay, omega0)

# calculate electric field transmittance and reflectance
ET_FT = Discrete_Fourier_Transform(ET, time, omegas)
ER_FT = Discrete_Fourier_Transform(ER, time, omegas)
pulse_FT = Discrete_Fourier_Transform(pulse, time, omegas)

# calculate reflectance and transmittance
R = np.abs(ER_FT) ** 2 / np.abs(pulse_FT) ** 2
T = (
    np.abs(ET_FT) ** 2
    / np.abs(pulse_FT) ** 2
    * refractive_index[jT]
    / refractive_index[j_source]
)

T_, R_ = thin_film_TR(1.0, np.sqrt(12.11), np.sqrt(4), wavelengths, thin_film_thickness)

# plot wavelengths, transmittance, and reflectance
plt.plot(wavelengths, R, "blue")
plt.plot(wavelengths, R_, "x", color="blue")
plt.plot(wavelengths, T, "red")
plt.plot(wavelengths, T_, "x", color="red")
plt.plot(wavelengths, 1 - R - T, "violet")
plt.xlabel("wavelengths (m)")
plt.ylabel("Spectrum")
plt.legend(["Reflectance", "R-analytic", "Transmittance", "T-analytic", "Absorption"])
plt.show()

# animate
frames = []  # for storing the generated images
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("x")
ax.set_ylabel("Ez")

for i in range(len(E_movie)):
    (im,) = ax.plot(E_movie[i], color="red")
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
# ani.save("1D_lossless_dielectric_medium.mp4", writer="ffmpeg", fps=60)
