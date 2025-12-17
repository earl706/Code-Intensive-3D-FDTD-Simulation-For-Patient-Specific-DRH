"""
Simulation of a frequency-dependent material
"""

import numpy as np
from matplotlib import pyplot as plt, animation

# Initialize simulation size and field values
simulation_size = 500
Ex = np.zeros(simulation_size)
Dx = np.zeros(simulation_size)
Ix = np.zeros(simulation_size)
Sx = np.zeros(simulation_size)
Sxm2 = np.zeros(simulation_size)
Sxm1 = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)
time_steps = 1200

# sinusoid wave parameters
dx = 1e-8
dt = dx / 6e8
number_of_frequencies = 3
freq_in = np.array((500e12, 2000e12, 2000e12))

# pulse parameters
pulse_width = 24
pulse_delay = 50

# create dielectric profile
eps0 = 8.8541878162e-12  # permittivity in free space
epsr = 2  # relative permittivity
sigma = 0.01  # electrical conductivity
tau = 0.001 * 1e-6  # relaxation time constant
chi = 2  # electric susceptibility
fp = 2000e12
plasma_collision_frequency = 57e12
omega = 2 * np.pi * fp
medium_start = 300
medium_end = medium_start + 100

vc = np.ones(simulation_size)

vc[medium_start:medium_end] = plasma_collision_frequency

# absorbing boundary conditions
boundary_low = [0, 0]
boundary_high = [0, 0]

# fourier transform
arg = 2 * np.pi * freq_in * dt
real_pt = np.zeros((number_of_frequencies, simulation_size))
imag_pt = np.zeros((number_of_frequencies, simulation_size))
real_in = np.zeros(number_of_frequencies)
imag_in = np.zeros(number_of_frequencies)
amp_in = np.zeros(number_of_frequencies)
phase_in = np.zeros(number_of_frequencies)
amp = np.zeros((number_of_frequencies, simulation_size))
phase = np.zeros((number_of_frequencies, simulation_size))

E_frames = []

# main FDTD Loop
for time_step in range(1, time_steps + 1):
    # calculate Dx
    Dx[1:] = Dx[1:] + 0.5 * (Hy[:-1] - Hy[1:])

    # put a sinusoidal at the low end
    pulse = np.sin(2 * np.pi * freq_in[2] * time_step * dt) * np.exp(
        -0.5 * ((pulse_delay - time_step) / pulse_width) ** 2
    )
    Dx[5] = pulse + Dx[5]

    # calculate the Ex field from Dx
    Ex[1:] = Dx[1:] - Sx[1:]
    Sx[1:] = (
        (1 + np.exp(-vc[1:] * dt)) * Sxm1[1:]
        - np.exp(-vc[1:] * dt) * Sxm2[1:]
        + (omega**2 * dt / vc[1:]) * (1 - np.exp(-vc[1:] * dt)) * Ex[1:]
    )

    Sxm2[1:] = Sxm1[1:]
    Sxm1[1:] = Sx[1:]

    # calculate the Fourier transform of Ex
    cos_term = np.cos(arg[:, None] * time_step)
    sin_term = np.sin(arg[:, None] * time_step)

    real_pt += cos_term * Ex
    imag_pt -= sin_term * Ex

    # fourier Transform of the input pulse
    if time_step < 3 * pulse_delay:
        real_in[:] = real_in[:] + np.cos(arg[:] * time_step) * Ex[10]
        imag_in[:] = imag_in[:] - np.sin(arg[:] * time_step) * Ex[10]

    # Absorbing Boundary Conditions
    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[-1] = boundary_high.pop(0)
    boundary_high.append(Ex[-2])

    # calculate the Hy field
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
