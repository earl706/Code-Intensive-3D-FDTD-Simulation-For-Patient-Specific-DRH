"""
The Fourier Transform has been added
"""

import numpy as np
from matplotlib import pyplot as plt, animation
from math import exp, cos, sin, sqrt, atan2

# simulation size, time, field values
simulation_size = 200
time_steps = 1000
Ex = np.zeros(simulation_size)
Dx = np.zeros(simulation_size)
Ix = np.zeros(simulation_size)
Hy = np.zeros(simulation_size)


# sinusoid wave parameters
number_of_frequencies = 3
freq = np.array((500e6, 200e6, 100e6))
dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size
pulse_delay = 50
pulse_width = 10

# absorbing boundary conditions
boundary_low = [0, 0]
boundary_high = [0, 0]

# create dielectric profile
eps0 = 8.8541878162e-12
epsr = 4
sigma = 0.02
medium_start = 100

eps = np.ones(simulation_size)
normalized_conductivity = np.zeros(simulation_size)

eps[medium_start:] = 1 / (epsr + (sigma * dt / eps0))
normalized_conductivity[medium_start:] = sigma * dt / eps0

# fourier transform
arg = 2 * np.pi * freq * dt
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
    # for k in range(1, simulation_size):
    Dx[1:] = Dx[1:] + 0.5 * (Hy[:-1] - Hy[1:])

    # put a sinusoidal at the low end
    pulse = np.exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
    Dx[5] = pulse + Dx[5]

    # Calculate the Ex field from Dx
    # for k in range(1, simulation_size):
    Ex[1:] = eps[1:] * (Dx[1:] - Ix[1:])
    Ix[1:] = Ix[1:] + normalized_conductivity[1:] * Ex[1:]

    # Calculate the Fourier transform of Ex
    for k in range(simulation_size):
        for m in range(number_of_frequencies):
            real_pt[m, k] = real_pt[m, k] + np.cos(arg[m] * time_step) * Ex[k]
            imag_pt[m, k] = imag_pt[m, k] - np.sin(arg[m] * time_step) * Ex[k]

    # Fourier Transform of the input pulse
    if time_step < 100:
        for m in range(number_of_frequencies):
            real_in[m] = real_in[m] + np.cos(arg[m] * time_step) * Ex[10]
            imag_in[m] = imag_in[m] - np.sin(arg[m] * time_step) * Ex[10]

    # Absorbing Boundary Conditions
    Ex[0] = boundary_low.pop(0)
    boundary_low.append(Ex[1])
    Ex[simulation_size - 1] = boundary_high.pop(0)
    boundary_high.append(Ex[simulation_size - 2])

    # Calculate the Hy field
    for k in range(simulation_size - 1):
        Hy[k] = Hy[k] + 0.5 * (Ex[k] - Ex[k + 1])

    E_frames.append(Ex.copy())


# Animate
frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    (im,) = ax.plot(frame, color="blue")
    frames.append([im])

animation = animation.ArtistAnimation(
    fig, frames, interval=20, blit=True, repeat_delay=1000
)
plt.show()
