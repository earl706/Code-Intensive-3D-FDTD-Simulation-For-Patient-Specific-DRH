# #!/usr/bin/env python3
# """
# 1D FDTD of a Gaussian-modulated pulse striking an unmagnetized (Drude) plasma slab.
# Parameters inspired by Houle, "Electromagnetic Simulation Using the FDTD Method with Python", Ch.2.

# - Plasma (silver-like): f_p = 2000 THz, nu_c = 57 THz
# - Source center frequency: 4000 THz
# - Yee 1D grid with first-order Mur ABCs
# - Drude handled via ADE: dJ/dt + nu*J = eps0*wp^2 * E
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation

# # Physical constants
# c0 = 299_792_458.0
# mu0 = 4e-7 * np.pi
# eps0 = 1.0 / (mu0 * c0**2)

# # Grid
# Nz = 500  # number of E nodes
# dx = 5e-9  # 5 nm
# S = 0.99  # Courant number (1D max is 1.0)
# dt = S * dx / c0

# # Source (Gaussian-modulated sinusoid)
# f0 = 4000e12
# w0 = 2 * np.pi * f0
# t0 = 6.0 / f0
# tau = 2.0 / f0

# # Drude plasma parameters
# fp = 2_000e12
# wp = 2 * np.pi * fp
# nu_c = 57e12 * 2 * np.pi

# # ADE coefficients: J^{n+1/2} = a J^{n-1/2} + b E^n
# a = (1 - 0.5 * nu_c * dt) / (1 + 0.5 * nu_c * dt)
# b = (eps0 * wp**2 * dt) / (1 + 0.5 * nu_c * dt)

# # Arrays
# Ex = np.zeros(Nz)
# Hy = np.zeros(Nz - 1)
# Jp = np.zeros(Nz)

# # Plasma slab
# i_start, i_end = 310, 390
# is_plasma = np.zeros(Nz, dtype=bool)
# is_plasma[i_start:i_end] = True

# # Simple Mur ABC
# boundary_low = [0, 0]
# boundary_high = [0, 0]

# # Source location and run length
# src_i = 5
# Tmax = 1100
# snap_times = [300, 825, 1050]
# snaps = {}
# E_frames = []
# for n in range(Tmax + 1):
#     # Update H
#     Hy += (dt / (mu0 * dx)) * (Ex[1:] - Ex[:-1])

#     # Update Drude current J (only inside plasma)
#     Jp[is_plasma] = a * Jp[is_plasma] + b * Ex[is_plasma]

#     # Update E
#     curlH = np.zeros_like(Ex)
#     curlH[1:-1] = Hy[1:] - Hy[:-1]
#     Ex += (dt / eps0) * ((curlH / dx) - Jp)

#     # Soft source
#     t = n * dt
#     Ex[src_i] += np.exp(-(((t - t0) / tau) ** 2)) * np.sin(w0 * (t - t0))

#     # Mur ABCs
#     Ex[0] = boundary_low.pop(0)
#     boundary_low.append(Ex[1])
#     Ex[-1] = boundary_high.pop(0)
#     boundary_high.append(Ex[-2])

#     if n in snap_times:
#         snaps[n] = Ex.copy()
#     E_frames.append(Ex.copy())

# # Plot results
# fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
# x = np.arange(0, 480)
# show_slice = slice(0, 480)

# for ax, T in zip(axes, snap_times):
#     ax.plot(x, snaps[T][show_slice])
#     ax.set_ylim(-1.2, 1.2)
#     ax.set_ylabel(r"$E_x$")
#     ax.text(260, -0.8, f"$T = {T}$")
#     ax.axvline(i_start, linestyle="--")
#     ax.axvline(i_end, linestyle="--")
#     ax.text((i_start + i_end) / 2 - 25, 0.7, "Plasma")
#     ax.grid(True, linestyle=":", linewidth=0.5)

# axes[-1].set_xlabel("FDTD cells")
# plt.tight_layout()
# plt.show()


# frames = []
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# for frame in E_frames:
#     (im,) = ax.plot(frame, color="blue")
#     frames.append([im])

# ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
# plt.show()
import numpy as np
from matplotlib import pyplot as plt, animation


# -----------------------
# Grid & arrays
# -----------------------
ke = 500
ex = np.zeros(ke)
hy = np.zeros(ke - 1)

# Keep your ABC buffers (retained exactly as in your code)
boundary_low = [0.0, 0.0]
boundary_high = [0.0, 0.0]

# Spatial/temporal steps
ddx = 5e-9  # 5 nm cells (good for optical THz frequencies)
c0 = 2.99792458e8
dt = 0.99 * ddx / c0  # CFL ~ 0.99 (1D)

# -----------------------
# Source: Gaussian-modulated sinusoid (4,000 THz center)
# -----------------------
f0 = 4_000e12
w0 = 2 * np.pi * f0
t0 = 6.0 / f0
tau = 2.0 / f0
src_i = 100

# -----------------------
# Drude plasma (silver-like): fp = 2000 THz, nu_c = 57 THz
# ADE: dJ/dt + nu J = eps0 * wp^2 * E
# Discretized: J^{n+1/2} = a J^{n-1/2} + b E^n
# -----------------------
eps0 = 8.854e-12
mu0 = 4e-7 * np.pi

fp = 2_000e12
wp = 2 * np.pi * fp
nu_c = 57e12 * 2 * np.pi

a = (1 - 0.5 * nu_c * dt) / (1 + 0.5 * nu_c * dt)
b = (eps0 * wp**2 * dt) / (1 + 0.5 * nu_c * dt)

J = np.zeros(ke)

# Plasma slab indices (match dashed box in the figure)
k_start = 310
k_end = 390
is_plasma = np.zeros(ke, dtype=bool)
is_plasma[k_start:k_end] = True

# -----------------------
# Run and take snapshots
# -----------------------
nsteps = 1100
snap_times = [300, 825, 1050]
snaps = {}
E_frames = []

for n in range(1, nsteps + 1):
    # --- update H (Yee)
    hy += (dt / (mu0 * ddx)) * (ex[1:] - ex[:-1])

    # --- ADE update for Drude current (only inside plasma)
    J[is_plasma] = a * J[is_plasma] + b * ex[is_plasma]

    # --- update E
    curlH = np.zeros_like(ex)
    curlH[1:-1] = hy[1:] - hy[:-1]
    ex += (dt / (eps0 * ddx)) * curlH - (dt / eps0) * J

    # --- soft source (add to E at src_i)
    t = n * dt
    ex[src_i] += np.exp(-(((t - t0) / tau) ** 2)) * np.sin(w0 * (t - t0))

    # --- Absorbing Boundary Conditions (retain your original scheme)
    ex[0] = boundary_low.pop(0)
    boundary_low.append(ex[1])

    ex[ke - 1] = boundary_high.pop(0)
    boundary_high.append(ex[ke - 2])

    # --- save snapshots
    if n in snap_times:
        snaps[n] = ex.copy()
    E_frames.append(ex.copy())

# -----------------------
# Plot like Fig. 2.5
# -----------------------
plt.rcParams["font.size"] = 12
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

x = np.arange(0, ke)
for ax, T in zip(axes, snap_times):
    ax.plot(x, snaps[T], color="k", linewidth=1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_ylabel(r"$E_x$")
    ax.text(260, -0.9, f"$T = {T}$")

    # dashed plasma slab
    ax.axvline(k_start, linestyle="--", color="k", linewidth=1)
    ax.axvline(k_end, linestyle="--", color="k", linewidth=1)
    ax.text((k_start + k_end) / 2 - 25, 0.7, "Plasma")

    ax.grid(True, linestyle=":", linewidth=0.5)

axes[-1].set_xlabel("FDTD cells")
plt.tight_layout()
plt.show()


frames = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for frame in E_frames:
    (im,) = ax.plot(frame, color="blue")
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=1000)
plt.show()
